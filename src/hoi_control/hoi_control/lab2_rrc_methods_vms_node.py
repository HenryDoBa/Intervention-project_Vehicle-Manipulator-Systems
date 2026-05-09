#!/usr/bin/env python3
import math
import datetime
import os

import rclpy
from rclpy.node import Node           
from rclpy.parameter import Parameter 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from tf2_ros import Buffer, TransformListener, LookupException, \
    ConnectivityException, ExtrapolationException
from hoi_control.swiftpro_robotics_rrc import (
    SwiftProManipulator4DOF,      # 4-DOF arm state + kinematics queries
    swiftpro_fk,                  # Arm-local FK: [q1, q2, q3] -> position relative to link1
    swiftpro_fk_vms_5dof,         # 5-DOF VMS FK: base (x,y,psi) + arm (q1,q2,q3)
    swiftpro_jacobian_vms_5dof,   # 5-DOF VMS Jacobian: 3x5 matrix (base + arm DOF)
    swiftpro_jacobian_vms_6dof,   # 6-DOF VMS Jacobian: 4x6 matrix (pos + yaw, includes q4)
    # swiftpro_fk_vms_5dof_ned,     # FK variant for world_ned frame (not used in this node)
    # swiftpro_jacobian_vms_5dof_ned, # Jacobian variant for world_ned frame (not used in this node)
    # swiftpro_fk_vms_5dof_ned,     # 5-DOF VMS FK (world_ned → world_enu via TF) [USED]
    # swiftpro_jacobian_vms_5dof_ned,   # 5-DOF VMS Jacobian (world_ned → world_enu via TF) [USED]
    DLS,                          # Damped Least-Squares pseudo-inverse
    scale_velocities,             # uniform velocity scaling to stay within max_vel
    Q1_MIN, Q1_MAX,               # joint1 base yaw URDF limits  (+-pi/2)
    Q2_MIN, Q2_MAX,               # joint2 shoulder URDF limits  (-pi/2, +0.05)
    Q3_MIN, Q3_MAX,               # joint3 elbow    URDF limits  (-pi/2, +0.05)
    Q4_MIN, Q4_MAX,               # joint4 EE yaw   URDF limits  (+-pi/2)
    # Task-priority infrastructure
    VMSRobotState,
    VMSPositionTask,
    VMSYawQ4Task,
    VMSJointCenteringTask,
    VMSJointLimitsTask,
    vms_task_priority_step,
)

# ---------------------------------------------------------------------------
# Topics and TF frames
# ---------------------------------------------------------------------------
# The joint velocity controller listens on this topic; it expects a
# Float64MultiArray with one value per active joint: [dq1, dq2, dq3, dq4]
JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'

# The simulator publishes all joint positions and velocities on this topic
JOINT_STATE_TOPIC = '/turtlebot/joint_states'

# Mobile base velocity controller (TurtleBot differential drive)
BASE_CMD_TOPIC    = '/turtlebot/cmd_vel'  # Twist message: linear.x, angular.z 

# marker topic 
MARKER_TOPIC      = '/hoi/rrc_methods_vms_markers'

# TF frame in which all world-frame positions are expressed (ENU = East-North-Up)
WORLD_FRAME       = 'world_enu'

# TF frame of the physical end-effector (published by the static_transform_publisher in the launch file)
EE_FRAME          = 'end_effector'

# TF frame of the arm's first joint / base (link1); used to convert between
# world_enu and arm-local ENU by subtracting J1's world position
J1_FRAME          = 'turtlebot/swiftpro/manipulator_base_link'

# Control loop period: 60 Hz matches the simulator physics timestep
DT = 1.0 / 60.0   # seconds per control tick


# Marker IDs — each ID corresponds to a persistent RViz marker object.
# Using fixed IDs means successive publishes UPDATE the marker in place
# rather than creating a new one each tick.
ID_TARGET    = 0   # RED    sphere  — desired target position
ID_TF_EE     = 1   # GREEN  sphere  — true EE from TF (colour interpolates to yellow as error grows)
ID_FK_EE     = 2   # CYAN   sphere  — FK estimate of EE (shows FK accuracy vs TF)
ID_J1_BASE   = 3   # YELLOW sphere  — J1 arm base (origin of arm-local ENU)
ID_ERR_LINE  = 4   # WHITE  line    — vector from current EE to target (visual error)
ID_TEXT      = 5   # WHITE  text    — method name, error magnitude, and goal index
ID_PATH_LINE = 6   # ORANGE line    — straight-line path from start to goal (full intended path)
ID_PATH_WP   = 7   # ORANGE sphere  — current interpolated waypoint being tracked


# Automatic goal cycling — DISTANT goals in world_enu (robot-relative, not arm-relative)
# ---------------------------------------------------------------------------
# These positions are set DISTANT from the robot base in world_enu frame.
# The VMS FK/Jacobian will account for the robot pose transformation.
# These are absolute world positions, not offsets from the arm J1.
# Each goal is (position_world_enu, target_yaw).
# target_yaw = float('nan') -> position-only  (3×5 Jacobian, q4 uncontrolled)
# target_yaw = finite value  -> position+yaw  (4×6 Jacobian, q4 controlled)
# Orient goals are split into 3 consecutive entries (position → yaw → Q4_RESET)
# so position and yaw control never interfere with each other, and q4 is clean
# for the next goal. 'Q4_RESET' drives q4 back to 0 using VMSQ4ZeroTask.
# One entry per real goal. For orient goals (finite yaw), the code internally
# runs three time-based sub-tasks sequentially: position → yaw(q4) → q4_reset.
# Each sub-task runs for goal_cycle_period seconds. Position-only goals run
# a single position task for one period.
GOAL_SEQUENCE = [
    (np.array([ 1.20,  0.00,  0.30]), float('nan')),  # position-only
    (np.array([ 0.00,  1.20,  0.35]),  0.5),           # orient: q4 yaw = +0.5 rad
    (np.array([ 1.00,  0.60,  0.28]), float('nan')),  # position-only
    (np.array([ 0.00, -1.20,  0.30]), -0.5),           # orient: q4 yaw = -0.5 rad
    (np.array([ 0.80,  0.80,  0.36]),  0.0),           # orient: q4 yaw =  0.0 rad
    (np.array([ 1.10, -0.40,  0.28]),  0.8),           # orient: q4 yaw = +0.8 rad
]

# How long (seconds) the arm tries to reach each goal before cycling to the next
DEFAULT_GOAL_CYCLE_PERIOD = 60.0

# Safety margin applied inside the URDF hard limits when clamping velocities.
# Stops the joint slightly before the hard limit to give the controller time
# to decelerate
LIMIT_MARGIN = 0.07   # radians — activation threshold α
LIMIT_HYSTERESIS_RATIO = 1.5  # δ = LIMIT_MARGIN * ratio; must be > 1 to avoid chatter


class Lab2RRCMethodsVMSNode(Node):

    def __init__(self):
        super().__init__('lab2_rrc_methods_vms')

        # ── Declare runtime-tunable parameters ────────────────────────────

        # Goal position in world_enu frame (East-North-Up, metres).
        # These are absolute world coordinates, not relative to the arm or base.
        self.declare_parameter('target_x',  0.50)
        self.declare_parameter('target_y',   0.00)
        self.declare_parameter('target_z',   0.30)

        # Desired end-effector yaw angle (radians) for the suction cup joint (q4).
        # NaN   → position-only mode: q4 is not controlled.
        # float → q4 is driven to this angle after the position sub-task completes.
        self.declare_parameter('target_yaw', float('nan'))

        # Proportional gain K for the position task: larger = faster convergence,
        # but too large causes overshoot. Scaled by the identity matrix.
        self.declare_parameter('gain_k',    0.5)

        # Proportional gain for the q4 yaw sub-task. Kept lower than gain_k
        # because yaw correction is a secondary objective.
        self.declare_parameter('yaw_gain_k', 0.3)

        # Maximum arm joint speed after uniform scaling (rad/s).
        # All joint velocities are clipped so the largest does not exceed this.
        self.declare_parameter('max_vel',   0.3)

        # DLS damping factor λ. Larger → smoother motion near singularities
        # but larger steady-state position error. Smaller → more accurate but
        # can produce large velocities at singular configurations.
        self.declare_parameter('damping',   0.05)

        # Inverse kinematics method selector:
        #   0 = Jacobian transpose  (fast, no matrix inversion, may not converge)
        #   1 = Pseudo-inverse      (exact at non-singular configs, unstable near singularities)
        #   2 = DLS (Damped Least Squares) — recommended, stable near singularities
        self.declare_parameter('method',    1)

        # Master enable switch. False → publish zero velocities, arm holds still.
        self.declare_parameter('enabled',   True)

        # Goal cycling: auto-advance through GOAL_SEQUENCE every goal_cycle_period seconds.
        self.declare_parameter('goal_auto_cycle',   True)
        # Time (s) each sub-task runs before the timer fires and advances to the next.
        self.declare_parameter('goal_cycle_period', DEFAULT_GOAL_CYCLE_PERIOD)

        # Maximum base linear speed (m/s). Lower values prevent orbit by reducing
        # the step size of each DLS update relative to the workspace curvature.
        self.declare_parameter('base_linear_max',  0.1)

        # Maximum base angular speed (rad/s). Keep proportional to base_linear_max
        # to avoid generating circular trajectories (orbit) near goals.
        self.declare_parameter('base_angular_max', 0.2)

        # Position error below which the base stops moving (m).
        # Prevents micro-oscillations when the EE is very close to the target.
        self.declare_parameter('error_stop',       0.03)

        # Yaw error below which the yaw sub-task is considered complete (rad).
        self.declare_parameter('yaw_stop',         0.08)

        # SwiftProManipulator4DOF internally holds [q1, q2, q3, q4] and provides
        # FK / Jacobian queries.  Initialised to zeros; updated by _js_cb.
        self.arm = SwiftProManipulator4DOF()

        # Task-priority infrastructure — state container + task objects.
        # Joint limit tasks have highest priority (indices 2-5 = q1-q4 in dzeta).
        # Position task is next; yaw task is lowest priority (null-space of position).
        self._vms_state = VMSRobotState()
        self._joint_limit_tasks = [
            VMSJointLimitsTask('q1_limits', 2, Q1_MIN, Q1_MAX, margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q2_limits', 3, Q2_MIN, Q2_MAX, margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q3_limits', 4, Q3_MIN, Q3_MAX, margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q4_limits', 5, Q4_MIN, Q4_MAX, margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
        ]
        self._pos_task  = VMSPositionTask('ee_position', np.zeros((3, 1)))
        self._yaw_task  = VMSYawQ4Task('ee_yaw_q4', 0.0)   # q4-only, zero position coupling

        # Arm reset task: drives all arm joints [q1,q2,q3,q4] to 0 after each yaw phase.
        self._q4_zero_task = VMSJointCenteringTask('arm_reset', [0.0, 0.05, 0.05, 0.0])
        self._q4_zero_task.setGain(np.eye(4) * 0.8)

        # Guard flag: skip the control loop until the first /joint_states message
        # arrives so we never compute with stale zero-initialised joint angles
        self._got_js = False

        # Cached robot base pose in world_enu frame (extracted from TF: base_footprint)
        # Used by 5-DOF VMS FK/Jacobian
        self._base_x = 0.0       # robot base x position (East, m)
        self._base_y = 0.0       # robot base y position (North, m)
        self._base_psi = 0.0     # robot base yaw angle (rad)
        self._omega_lpf = 0.0    # low-pass filter state for base angular velocity

        # Initial base yaw — latched on first TF reading.
        # swiftpro_fk() is calibrated for the scenario's initial robot yaw.
        # delta_yaw = base_psi - initial_base_psi corrects for any base rotation.
        self._initial_base_psi = None
        
        # Cached arm link1 position in world_enu (used for FK offset)
        self._link1_world_enu = np.array([0.0, 0.0, 0.0])

        # Most recent TF-based EE position (world_enu); retained between ticks
        # so the debug log can show it even when a TF lookup temporarily fails
        self._last_tf_ee = None

        # Incrementing counter used to throttle the verbose terminal log:
        # we only print every 30 ticks (~0.5 s) to avoid terminal flooding
        self._loop_count = 0

        # Index into GOAL_SEQUENCE pointing to the currently active goal.
        # Starts at 0 (first goal) and wraps around after the last goal.
        self._goal_idx = 0

        # A dedicated ROS timer fires _advance_goal() every DEFAULT_GOAL_CYCLE_PERIOD
        # seconds, independently of the control-loop rate.  Using a separate timer
        # (rather than counting control-loop ticks) means the goal period is accurate
        # even if the control loop occasionally misses a tick.
        self._goal_cycle_timer = self.create_timer(
            DEFAULT_GOAL_CYCLE_PERIOD, self._advance_goal)

        # Sub-task index within the current goal (time-based, timer-driven).
        # Position-only goals have 1 sub-task (index 0 only).
        # Orient goals have 3 sub-tasks: 0=position, 1=yaw(q4), 2=q4_reset.
        self._sub_phase_idx = 0

        # Straight-line path tracking for the position sub-task.
        # _path_start  : EE position (world_enu) when the position sub-task began.
        # _path_start_t: ROS time when the position sub-task began.
        # The desired position is interpolated linearly from _path_start to the
        # goal position over goal_cycle_period seconds, so the EE follows a
        # straight line in task space rather than curving directly toward the goal.
        self._path_start   = None   # (3,) ndarray or None
        self._path_start_t = None   # rclpy.time.Time or None

        # Accumulated log entries — written to file on Ctrl+C.
        self._log_entries = []

        # Buffer stores a sliding window of recent transforms from /tf and /tf_static
        self._tf_buffer = Buffer()

        self._tf_listener = TransformListener(self._tf_buffer, self)

        
        self.sub_js = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)

        # Publisher for joint velocity commands; the controller reads this at ~60 Hz
        # TODO What does the above statement mean?
        self.pub_cmd = self.create_publisher(Float64MultiArray, JOINT_CMD_TOPIC, 10)

        # Publisher for mobile base velocity commands (Twist: linear.x, angular.z)
        self.pub_base_cmd = self.create_publisher(Twist, BASE_CMD_TOPIC, 10)

        # Publisher for RViz debug markers (spheres, lines, text)
        self.pub_markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        # Main control timer: fires _control_loop at exactly 60 Hz (every DT seconds)
        self.timer = self.create_timer(DT, self._control_loop)

        # Startup log 
        self.get_logger().info('=' * 62)
        self.get_logger().info('Lab2 RRC Methods VMS node started (4-DOF on Turtlebot)')
        self.get_logger().info('Tune: ros2 param set /lab2_rrc_methods_vms <param> <value>')
        self.get_logger().info('Arm params:  target_x/y/z, target_yaw, gain_k, max_vel, '
                               'damping, method, enabled, goal_auto_cycle, goal_cycle_period')
        self.get_logger().info('Base params: base_vel_enabled, base_linear_max, base_angular_max, '
                               'error_threshold')
        self.get_logger().info('Publishing: arm commands → /turtlebot/.../joint_velocity_controller/command')
        self.get_logger().info('Publishing: base commands → /turtlebot/cmd_vel  (namespaced!)')
        self.get_logger().info('RViz:  Fixed Frame=world_enu  '
                               'MarkerArray from ' + MARKER_TOPIC)
        self.get_logger().info(
            f'Goals: {len(GOAL_SEQUENCE)} DISTANT targets in world_enu, cycling every '
            f'{DEFAULT_GOAL_CYCLE_PERIOD:.0f} s  (goal_auto_cycle=True)')
        self.get_logger().info('FK/Jacobian: Arm-local (proven), errors in world_enu frame')
        self.get_logger().info('Method: DLS (default=2), World EE = J1_world + FK_arm_local')
        self.get_logger().info('Base moves when error > error_threshold using base_linear_max')
        self.get_logger().info('=' * 62)

    # Callbacks

    def _js_cb(self, msg: JointState):
        """Update arm state from the latest joint_states message."""
        # Parse the JointState message and extract q1-q4 into self.arm.q.
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))

        # Signal that at least one valid joint-state has arrived; the control
        # loop blocks until this is True
        self._got_js = True

    def _tf_pos(self, child_frame):
        """Look up position of child_frame in world_enu. Returns (3,) or None."""
        try:
            t = self._tf_buffer.lookup_transform(
                WORLD_FRAME, child_frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
    
    def _tf_pose(self, frame_id):
        """Look up pose (position + yaw) of frame_id in world_enu. Returns [x, y, psi] or None."""
        try:
            t = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame_id, rclpy.time.Time())
            tr = t.transform.translation
            quat = t.transform.rotation
            qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
            yaw = np.arctan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz))
            return np.array([tr.x, tr.y, yaw])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # Goal cycling 

    def _advance_goal(self):
        """
        Timer callback: rotate to the next goal in GOAL_SEQUENCE every
        `goal_cycle_period` seconds (when goal_auto_cycle is enabled).

        The new target is written back into the ROS parameters so that
        external observers (ros2 param get / dynamic_reconfigure) see the
        current goal, and so the control loop reads them the same way it
        reads manually-set parameters.
        """
        # If cycling is disabled at runtime, fire but do nothing
        if not self.get_parameter('goal_auto_cycle').value:
            return

        # Re-read the period in case it was changed at runtime via ros2 param set.
        # timer_period_ns is the period this timer was created with (nanoseconds).
        period = float(self.get_parameter('goal_cycle_period').value)
        # TODO What is the purpose of this line of code? Why do we need to re-read the period?
        
        
        if abs(self._goal_cycle_timer.timer_period_ns / 1e9 - period) > 0.5:
            # Period was changed by more than 0.5 s — recreate the timer.
            # ROS 2 timers cannot be reconfigured after creation, so we destroy
            # the old one and create a fresh timer with the new period.
            self._goal_cycle_timer.destroy()
            self._goal_cycle_timer = self.create_timer(period, self._advance_goal)

        # Current goal determines how many sub-tasks exist.
        goal_pos, goal_yaw = GOAL_SEQUENCE[self._goal_idx]
        # num_phases = 3 if math.isfinite(goal_yaw) else 1   # 3 = position + yaw + arm_reset
        num_phases = 2 if math.isfinite(goal_yaw) else 1     # 2 = position + yaw (no reset)

        # Advance sub-task first; only move to the next goal when all sub-tasks done.
        self._sub_phase_idx += 1
        # Reset path-tracking state so the new sub-task starts a fresh straight line.
        self._path_start   = self._last_tf_ee.copy() if self._last_tf_ee is not None else None
        self._path_start_t = self.get_clock().now()
        if self._sub_phase_idx < num_phases:
            sub_names = ['position', 'yaw/q4', 'q4_reset']
            self.get_logger().info(
                f'[GOAL CYCLE] -> goal {self._goal_idx}/{len(GOAL_SEQUENCE)-1} '
                f'sub-task {self._sub_phase_idx}: {sub_names[self._sub_phase_idx]}')
            return

        # All sub-tasks done — advance to next goal and reset sub-task index.
        self._sub_phase_idx = 0
        self._goal_idx = (self._goal_idx + 1) % len(GOAL_SEQUENCE)

        goal_pos, goal_yaw = GOAL_SEQUENCE[self._goal_idx]

        # Write the new target back into the ROS parameter server.
        self.set_parameters([
            Parameter('target_x',   Parameter.Type.DOUBLE, float(goal_pos[0])),
            Parameter('target_y',   Parameter.Type.DOUBLE, float(goal_pos[1])),
            Parameter('target_z',   Parameter.Type.DOUBLE, float(goal_pos[2])),
            Parameter('target_yaw', Parameter.Type.DOUBLE, float(goal_yaw)),
        ])

        yaw_str = f'{goal_yaw:.2f} rad ({math.degrees(goal_yaw):.1f} deg)' \
                  if math.isfinite(goal_yaw) else 'position-only'
        self.get_logger().info(
            f'[GOAL CYCLE] -> goal {self._goal_idx}/{len(GOAL_SEQUENCE)-1}: '
            f'[{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}] world_enu  '
            f'yaw={yaw_str}')

    def _control_loop(self):
        # Block until the first /joint_states message has been received.
        # Without this guard the FK and Jacobian would run on q = [0,0,0,0],
        # which is a singular configuration for some arm poses.
        if not self._got_js:
            return

        # Increment tick counter; used below to throttle the terminal log
        self._loop_count += 1

        # Read runtime parameters 
        # All parameters are re-read every tick so that ros2 param set changes
        # take effect within one control cycle (~16 ms at 60 Hz).

        tx     = self.get_parameter('target_x').value    # target East  (m, world_enu, DISTANT)
        ty     = self.get_parameter('target_y').value    # target North (m, world_enu, DISTANT)
        tz     = self.get_parameter('target_z').value    # target Up    (m, world_enu, DISTANT)
        t_yaw  = self.get_parameter('target_yaw').value  # desired EE yaw (rad); NaN = position-only
        K_gain = self.get_parameter('gain_k').value      # scalar proportional gain applied to error
        mv          = self.get_parameter('max_vel').value          # joint velocity ceiling (rad/s)
        dam         = self.get_parameter('damping').value          # DLS damping factor lambda
        mth         = int(self.get_parameter('method').value)      # integer selector: 0/1/2
        ena         = self.get_parameter('enabled').value          # bool: if False, publish zeros
        base_lin_max = float(self.get_parameter('base_linear_max').value)   # m/s
        base_ang_max = float(self.get_parameter('base_angular_max').value)  # rad/s

        # Pack the three scalars into a NumPy vector for later arithmetic
        target_world_enu = np.array([tx, ty, tz])

        # Convert the integer method selector to a human-readable label for logging
        # mth % 3 ensures any integer input maps into the valid range 0-2
        method_name = ['transpose', 'pseudoinverse', 'DLS'][mth % 3]

        # Orientation mode is active when target_yaw is a real number (not NaN).
        # In position-only mode the 4th column of the Jacobian is [0,0,0], so q4
        # receives no velocity command and the arm behaves as a 3-DOF position task.
        orient_mode = math.isfinite(t_yaw)

        # TF lookups: robot base pose, link1 position, and end-effector position
        ee_world_enu = self._tf_pos(EE_FRAME)
        link1_world_enu = self._tf_pos(J1_FRAME)  # arm base in world (needed for FK)
        base_pose = self._tf_pose('turtlebot/base_footprint')
        
        if base_pose is not None:
            self._base_x, self._base_y, self._base_psi = base_pose

        # Latch initial base yaw on first valid TF reading.
        # swiftpro_fk() is calibrated for the scenario's initial robot yaw;
        # delta_yaw corrects for any base rotation since then.
        if self._initial_base_psi is None and base_pose is not None:
            self._initial_base_psi = self._base_psi
            self.get_logger().info(
                f'Initial base yaw latched: {self._initial_base_psi:.4f} rad')

        if link1_world_enu is not None:
            self._link1_world_enu = link1_world_enu

        if ee_world_enu is not None:
            self._last_tf_ee = ee_world_enu

        # ── 5-DOF VMS FK ──────────────────────────────────────────────────────
        # swiftpro_fk_vms_5dof(j1_world, arm_q, base_yaw):
        #   substitutes q1_mod = q1 − (ψ − π/2) so swiftpro_fk is correct at
        #   any base yaw, with no rotation matrix needed.
        q_arm = np.array(self.arm.q)   # [q1, q2, q3, q4] — full 4-DOF
        fk_world_enu = swiftpro_fk_vms_5dof(
            self._link1_world_enu, q_arm, self._base_psi, self._tf_buffer)
        # fk_world_enu = swiftpro_fk_vms_5dof_ned(q_arm, self._tf_buffer)
        
        if fk_world_enu is None:
            self.get_logger().warn('FK computation failed, skipping control tick',
                                   throttle_duration_sec=1.0)
            return

        # Path-tracking defaults — overwritten inside the position sub-task branch.
        path_alpha   = float('nan')
        path_desired = target_world_enu.copy()

        # Position error: TF-based if available, otherwise FK-based
        if ee_world_enu is not None:
            pos_err = (target_world_enu - ee_world_enu).reshape(3, 1)
            err_norm = float(np.linalg.norm(pos_err))
            fk_vs_tf_mm = float(np.linalg.norm(fk_world_enu - ee_world_enu)) * 1000
            position_source = 'TF'
            ee_for_jac = ee_world_enu   # use ground-truth EE for Jacobian ω column

        else:
            pos_err = (target_world_enu - fk_world_enu).reshape(3, 1)
            err_norm = float(np.linalg.norm(pos_err))
            fk_vs_tf_mm = float('nan')
            position_source = 'FK_fallback'
            ee_for_jac = fk_world_enu
            self.get_logger().warn('TF unavailable — using 5-DOF FK for error',
                                   throttle_duration_sec=1.0)

        # Task-Priority Control 
        # Update the shared robot state container used by all task objects.
        self._vms_state.update(
            ee_for_jac,
            [self._base_x, self._base_y],
            self._base_psi,
            q_arm,
            self._tf_buffer)

        # Select sub-task based on time-based index (timer-driven, no thresholds).
        # Sub-task 0: position task (all goals).
        # Sub-task 1: q4 yaw task  (orient goals only, timer fires after period).
        # Sub-task 2: q4 reset     (orient goals only, timer fires after period).
        yaw_k = float(self.get_parameter('yaw_gain_k').value)
        sub  = self._sub_phase_idx

        # if sub == 2:
        #     # Arm reset: drive all joints to 0 (disabled for this test).
        #     tasks = self._joint_limit_tasks + [self._q4_zero_task]
        if sub == 1 and orient_mode:
            # Q4 yaw alignment: only q4 moves, zero position coupling.
            self._yaw_task.setDesired(np.array([[t_yaw]]))
            self._yaw_task.setGain(np.array([[yaw_k]]))
            self._pos_task.setFF(np.zeros((3, 1)))   # clear feedforward from position phase
            tasks = self._joint_limit_tasks + [self._yaw_task]
        else:
            # Position task — straight-line path from start to goal.
            # Latch the start on the very first tick of this sub-task.
            if self._path_start is None:
                self._path_start   = (ee_world_enu.copy() if ee_world_enu is not None
                                       else fk_world_enu.copy())
                self._path_start_t = self.get_clock().now()

            elapsed = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
            period  = float(self.get_parameter('goal_cycle_period').value)
            path_alpha = float(np.clip(elapsed / period, 0.0, 1.0))

            # Interpolated waypoint along the straight line: start → goal.
            path_desired = self._path_start + path_alpha * (target_world_enu - self._path_start)

            # Feedforward velocity = σ̇_E (velocity of the moving desired point).
            # From the RRC control law:  ẋ_E = σ̇_E + K·σ̃_E
            # σ̇_E = (goal - start) / T  — constant along the straight-line path.
            # Without this term the controller always lags behind the moving waypoint.
            # Zero the feedforward once the path is complete (alpha=1) so the
            # feedback term alone handles the final convergence to the fixed goal.
            if path_alpha < 1.0:
                ff_vel = (target_world_enu - self._path_start) / period   # (3,) m/s
            else:
                ff_vel = np.zeros(3)

            self._pos_task.setDesired(path_desired.reshape(3, 1))
            self._pos_task.setGain(np.diag([K_gain] * 3))
            self._pos_task.setFF(ff_vel.reshape(3, 1))
            tasks = self._joint_limit_tasks + [self._pos_task]

        # Run one step of the recursive Task-Priority algorithm (always DLS).
        if ena:
            dzeta = vms_task_priority_step(tasks, self._vms_state, damping=dam).flatten()
        else:
            vms_task_priority_step(tasks, self._vms_state, damping=dam)
            dzeta = np.zeros(6)

        # Extract logged quantities.
        # if sub == 2:   # arm reset logging (disabled)
        #     self._q4_zero_task.update(self._vms_state)
        #     yaw_err  = float('nan')
        #     pos_err  = np.zeros((3, 1))
        #     J_cond   = float(np.linalg.cond(self._q4_zero_task.getJacobian()))
        if sub == 1 and orient_mode:
            self._yaw_task.update(self._vms_state)
            yaw_err = float(self._yaw_task.getError()[0, 0])
            pos_err = (target_world_enu - (ee_world_enu if ee_world_enu is not None
                                           else fk_world_enu)).reshape(3, 1)
            J_cond  = float(np.linalg.cond(self._yaw_task.getJacobian()))
        else:
            self._pos_task.update(self._vms_state)
            pos_err = self._pos_task.getError()
            yaw_err = float('nan')
            J_cond  = float(np.linalg.cond(self._pos_task.getJacobian()))
        err_norm = float(np.linalg.norm(pos_err))

        # Active joint-limit status for log
        active_limits = [t.name for t in self._joint_limit_tasks if t.isActive()]

        # Extract velocity components from the 6-DOF dzeta vector.
        v_base_linear  = dzeta[0]
        v_base_angular = dzeta[1]
        dq_arm         = dzeta[2:5]   # [dq1, dq2, dq3]
        dq4            = dzeta[5]

        # Scale base velocities independently (different units and limits).
        v_base_linear  = float(scale_velocities(np.array([v_base_linear]),  base_lin_max)[0])
        v_base_angular = float(scale_velocities(np.array([v_base_angular]), base_ang_max)[0])

        # J_cond-based base velocity scaling — primary orbit fix.
        #
        # Root cause of orbit: J_cond spikes to 20-26 when q3 hits its limit in
        # goal 2's workspace. The DLS amplifies velocities unreliably near
        # singularities, saturating both vx and omega simultaneously → circular
        # motion. The error-proportional cap alone doesn't trigger because omega
        # is already below the cap at that range.
        #
        # Fix: scale base velocities DOWN proportional to J_cond when ill-conditioned.
        # Arm joints (dq1-dq4) are NOT scaled so they keep reconfiguring the arm,
        # naturally reducing J_cond and exiting the singular region.
        #   J_cond=10 → scale=1.0 (no change)
        #   J_cond=20 → scale=0.5 (half speed base)
        #   J_cond=25 → scale=0.4 (orbit region: forces slow creep vs spinning)
        # JCOND_THRESHOLD = 10.0
        # if J_cond > JCOND_THRESHOLD:
        #     jcond_scale = JCOND_THRESHOLD / J_cond
        #     v_base_linear  *= jcond_scale
        #     v_base_angular *= jcond_scale

        # Error-proportional omega cap — secondary guard for close-range spinning.
        # orbit_dist = 0.8   # metres
        # omega_cap = base_ang_max * min(1.0, err_norm / orbit_dist)
        # v_base_angular = float(np.clip(v_base_angular, -omega_cap, omega_cap))

        # Scale arm joints uniformly so direction is preserved.
        all_dq = scale_velocities(np.array([dq_arm[0], dq_arm[1], dq_arm[2], dq4]), mv)
        dq_arm = all_dq[:3]
        dq4    = all_dq[3]

        # Goal deadzone: stop all motion when close enough.
        # Without this, the controller keeps making tiny corrections that overshoot,
        # causing the q4/q3 joint limit tasks to chatter on and off at the boundary.
        pos_stop = float(self.get_parameter('error_stop').value)
        yaw_stop = float(self.get_parameter('yaw_stop').value)
        yaw_converged = not orient_mode or abs(yaw_err) < yaw_stop
        # if err_norm < pos_stop and yaw_converged:
        #     v_base_linear  = 0.0
        #     v_base_angular = 0.0
        #     self._omega_lpf = 0.0   # reset filter so it doesn't lag on the next goal
        #     dq_arm = np.zeros(3)
        #     dq4    = 0.0

        # Publish base velocity.
        base_cmd = Twist()
        base_cmd.linear.x  = v_base_linear
        base_cmd.angular.z = v_base_angular
        self.pub_base_cmd.publish(base_cmd)

        dq    = np.array([dq_arm[0], dq_arm[1], dq_arm[2], dq4])
        dq_arr = dq.flatten()

        if ena:
            cmd = Float64MultiArray()
            cmd.data = dq_arr.tolist()
            self.pub_cmd.publish(cmd)
        else:
            cmd = Float64MultiArray()
            cmd.data = [0.0, 0.0, 0.0, 0.0]
            self.pub_cmd.publish(cmd)

        # Verbose debug log (every 30 loops ≈ 0.5 s)
        if self._loop_count % 30 == 0:
            q      = self.arm.q
            # if sub == 2:   # arm_reset disabled
            #     q = self.arm.q
            #     mode_str = f'arm_reset  (q=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}] → [0,0,0,0])'
            if sub == 1 and orient_mode:
                mode_str = f'yaw/q4-only  (target={t_yaw:.2f} rad, q4={self.arm.q[3]:.3f})'
            elif orient_mode:
                mode_str = f'orient/position  (target_yaw={t_yaw:.2f} rad)'
            else:
                mode_str = 'position-only'


            if orient_mode and math.isfinite(yaw_err):
                yaw_ok = abs(yaw_err) < 0.05
                pos_ok = err_norm < 0.01
                if pos_ok and yaw_ok:
                    orient_status = 'ACHIEVED (pos + yaw)'
                elif yaw_ok:
                    orient_status = 'YAW OK — pos converging'
                elif pos_ok:
                    orient_status = 'POS OK — yaw converging'
                else:
                    orient_status = 'converging...'
                orient_lines = (
                    f'  yaw target  : {t_yaw:.3f} rad  ({math.degrees(t_yaw):.1f} deg)\n'
                    f'  EE yaw      : {self._base_psi + q[0] + q[3]:.3f} rad  '
                    f'(psi={self._base_psi:.3f} + q1={q[0]:.3f} + q4={q[3]:.3f})\n'
                    f'  yaw_err     : {math.degrees(yaw_err):.2f} deg  ({yaw_err:.4f} rad)\n'
                    f'  orient status: {orient_status}\n'
                )
            else:
                orient_lines = ''

            log_str = (
                f'\n{"─"*70}\n'
                f'  solver      : task-priority | method: {method_name}  |  enabled: {ena}  |  mode: {mode_str}\n'
                f'  goal_idx    : {self._goal_idx}/{len(GOAL_SEQUENCE)-1}\n'
                f'  BASE POSE   : x={self._base_x:.3f}  y={self._base_y:.3f}  '
                f'psi={self._base_psi:.3f}\n'
                f'  ARM q[1-4]  : [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] rad\n'
                f'  target(wenu): [{tx:.3f}, {ty:.3f}, {tz:.3f}]\n'
                f'  TF EE(wenu) : {np.round(ee_world_enu,3) if ee_world_enu is not None else "N/A"}\n'
                f'  FK EE(wenu) : {np.round(fk_world_enu,3)}\n'
                f'  FK_vs_TF    : {fk_vs_tf_mm:.3f} mm\n'
                f'  pos_source  : {position_source}\n'
                f'  err_norm    : {err_norm:.4f} m\n'
                f'  err_vec     : {np.round(pos_err.flatten(),4)}\n'
                f'  path_alpha  : {path_alpha:.3f}  '
                f'(waypoint {np.round(path_desired,3) if sub == 0 or not orient_mode else np.round(target_world_enu,3)})\n'
                + orient_lines +
                f'  [vx, omega] : [{v_base_linear:.4f}, {v_base_angular:.4f}]\n'
                f'  dq[1-4]     : {np.round(dq_arr,4)}\n'
                f'  J_cond(pos) : {J_cond:.1f}\n'
                f'  active_lims : {active_limits if active_limits else "none"}\n'
                f'  params      : K={K_gain}  yaw_k={yaw_k:.2f}  mv={mv}  dam={dam}\n'
                f'{"─"*70}'
            )
            self.get_logger().info(log_str)
            self._log_entries.append(log_str)

        # Markers
        self._publish_markers(
            target_world_enu, ee_world_enu, fk_world_enu,
            np.array([self._base_x, self._base_y, 0.0]), err_norm, method_name, yaw_err,
            path_start=self._path_start, path_desired=path_desired)

    # Marker publisher 

    def _publish_markers(self, target, tf_ee, fk_ee, j1, err_norm, method,
                         yaw_err=float('nan'), path_start=None, path_desired=None):
        """
        Publish RViz markers:
          RED    sphere  — desired target position
          GREEN  sphere  — TF EE position (yellow→green as error shrinks)
          CYAN   sphere  — FK estimate of EE
          YELLOW sphere  — J1 base
          WHITE  line    — error vector from EE to target
          WHITE  text    — method name, error, goal index
          ORANGE line    — straight-line path from start to goal (path_start → target)
          ORANGE sphere  — current interpolated waypoint being tracked
        """
        # Stamp all markers with the current node time so RViz knows they are fresh
        now = self.get_clock().now().to_msg()

        # MarkerArray allows publishing all markers in a single message
        ma = MarkerArray()

        def mk(mid, mtype):
            """Create a base Marker with shared fields pre-filled."""
            m = Marker()
            m.header.frame_id    = WORLD_FRAME   # all markers live in world_enu
            m.header.stamp       = now            # timestamp for RViz staleness check
            m.ns                 = 'rrc_methods_vms'  # namespace groups markers in RViz
            m.id                 = mid            # unique ID within the namespace
            m.type               = mtype          # SPHERE, LINE_STRIP, TEXT_VIEW_FACING, etc.
            m.action             = Marker.ADD     # ADD creates or updates; DELETE removes
            m.pose.orientation.w = 1.0            # identity quaternion (no rotation)
            return m

        def sphere(mid, pos, r, g, b, size=0.025):
            """Create a coloured sphere marker at the given world_enu position."""
            m = mk(mid, Marker.SPHERE)
            m.pose.position.x = float(pos[0])  
            m.pose.position.y = float(pos[1])   
            m.pose.position.z = float(pos[2])   
            m.scale.x = m.scale.y = m.scale.z = size   # uniform sphere diameter
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            return m

        # RED sphere — desired target position (DISTANT goal in world_enu, static shape, bright red)
        ma.markers.append(sphere(ID_TARGET, target, 1.0, 0.0, 0.0, 0.030))

        # GREEN/YELLOW sphere — true EE from TF; colour encodes how close we are.
        # closeness=1 (err=0) -> pure green.  closeness=0 (err>=10cm) -> yellow.
        # This gives an at-a-glance indication of convergence without reading numbers.
        if tf_ee is not None:
            # Normalise error to [0,1] over the range 0-10 cm
            closeness = float(np.clip(1.0 - err_norm / 0.1, 0.0, 1.0))
            m = mk(ID_TF_EE, Marker.SPHERE)
            m.pose.position.x = float(tf_ee[0])
            m.pose.position.y = float(tf_ee[1])
            m.pose.position.z = float(tf_ee[2])
            m.scale.x = m.scale.y = m.scale.z = 0.022   # slightly smaller than target
            m.color.r = 1.0 - closeness   # 1=yellow (far), 0=green (close)
            m.color.g = 1.0               # green channel always full
            m.color.b = 0.0
            m.color.a = 1.0
            ma.markers.append(m)

        # CYAN sphere — VMS FK estimate of EE.  If this drifts far from the green TF sphere,
        # the VMS FK model needs recalibration (FK_vs_TF metric in the log will be large).
        ma.markers.append(sphere(ID_FK_EE, fk_ee, 0.0, 0.8, 1.0, 0.015))

        # YELLOW sphere — J1 base position (origin of the arm-local ENU frame).
        # Helps verify that the arm_local→world transform is correct.
        if j1 is not None:
            ma.markers.append(sphere(ID_J1_BASE, j1, 1.0, 1.0, 0.0, 0.018))

        # WHITE line — error vector drawn from the current TF EE to the target.
        # The length and direction of this line visually encode the 3-D error.
        if tf_ee is not None:
            m2 = mk(ID_ERR_LINE, Marker.LINE_STRIP)
            m2.scale.x = 0.004              # line width in metres
            m2.color.r = m2.color.g = m2.color.b = 1.0   # white
            m2.color.a = 0.9

            # LINE_STRIP requires a list of Point messages for its vertices.
            # Two points define a single line segment (current EE → target).
            p1 = Point()
            p1.x = float(tf_ee[0]); p1.y = float(tf_ee[1]); p1.z = float(tf_ee[2])
            p2 = Point()
            p2.x = float(target[0]); p2.y = float(target[1]); p2.z = float(target[2])
            m2.points = [p1, p2]
            ma.markers.append(m2)

        # WHITE text — overlaid above the target sphere showing method, error, and goal.
        # TEXT_VIEW_FACING always rotates to face the camera, so it is readable from any angle.
        mt = mk(ID_TEXT, Marker.TEXT_VIEW_FACING)
        mt.pose.position.x = float(target[0])
        mt.pose.position.y = float(target[1])
        mt.pose.position.z = float(target[2]) + 0.06   # offset above the target sphere
        mt.scale.z = 0.025    # text height in metres (scale.z controls font size for text markers)
        mt.color.r = mt.color.g = mt.color.b = mt.color.a = 1.0   # fully opaque white
        t_yaw_cur = self.get_parameter('target_yaw').value
        if math.isfinite(t_yaw_cur) and math.isfinite(yaw_err):
            mt.text = (f'[{method}] err={err_norm:.3f}m  goal{self._goal_idx}\n'
                       f'yaw_err={math.degrees(yaw_err):.1f}deg')
        else:
            mt.text = f'[{method}] err={err_norm:.3f}m  goal{self._goal_idx}'
        ma.markers.append(mt)

        # ORANGE line — full intended straight-line path (path_start → target).
        # Visible as soon as the position sub-task begins; stays fixed until the
        # goal changes, giving a clear reference for how straight the motion is.
        if path_start is not None:
            mp = mk(ID_PATH_LINE, Marker.LINE_STRIP)
            mp.scale.x = 0.006
            mp.color.r = 1.0; mp.color.g = 0.5; mp.color.b = 0.0; mp.color.a = 0.8
            ps = Point(); ps.x = float(path_start[0]); ps.y = float(path_start[1]); ps.z = float(path_start[2])
            pe = Point(); pe.x = float(target[0]);     pe.y = float(target[1]);     pe.z = float(target[2])
            mp.points = [ps, pe]
            ma.markers.append(mp)

        # ORANGE sphere — current interpolated waypoint the controller is tracking.
        # Slides from path_start to target over the goal_cycle_period, showing
        # exactly which point the DLS is driving toward at each tick.
        if path_desired is not None:
            ma.markers.append(sphere(ID_PATH_WP, path_desired, 1.0, 0.5, 0.0, 0.018))

        # Publish all markers in a single message to minimise latency jitter
        self.pub_markers.publish(ma)


    def save_log(self):
        """Write all accumulated log entries to a timestamped text file."""
        if not self._log_entries:
            self.get_logger().info('No log entries to save.')
            return
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.expanduser('~/ROS2_Crash_Course/ros2_ws/src/hoi_control/logs')
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f'vms_log_{timestamp}.txt')
        with open(filepath, 'w') as f:
            f.write(f'VMS node log — saved {datetime.datetime.now()}\n')
            f.write(f'Total entries: {len(self._log_entries)}\n')
            f.write('\n'.join(self._log_entries))
        self.get_logger().info(f'Log saved → {filepath}  ({len(self._log_entries)} entries)')


def main(args=None):
    rclpy.init(args=args)
    node = Lab2RRCMethodsVMSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_log()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
