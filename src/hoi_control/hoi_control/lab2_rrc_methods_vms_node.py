#!/usr/bin/env python3
"""
Resolved-Rate Control VMS node — 4-DOF uArm Swift Pro on Turtlebot (joint1-4).
All control math accounts for mobile base: FK and Jacobian in arm-local, then world_enu.

Vehicle Manipulator System (VMS): The arm is mounted on a mobile Turtlebot.
Goals are set in world_enu frame (distant from the robot), not relative to the arm.

ARM CONTROL (4-DOF):
  ros2 param set /lab2_rrc_methods_vms target_x   0.50      # world goal (m)
  ros2 param set /lab2_rrc_methods_vms target_y   0.00
  ros2 param set /lab2_rrc_methods_vms target_z   0.40
  ros2 param set /lab2_rrc_methods_vms target_yaw nan       # position-only
  ros2 param set /lab2_rrc_methods_vms target_yaw 0.5       # position+yaw
  ros2 param set /lab2_rrc_methods_vms gain_k     1.0       # arm control gain
  ros2 param set /lab2_rrc_methods_vms max_vel    0.5       # arm max velocity (rad/s)
  ros2 param set /lab2_rrc_methods_vms damping    0.05      # DLS damping (lower=accuracy)
  ros2 param set /lab2_rrc_methods_vms method     2         # 0=transpose, 1=pinv, 2=DLS
  ros2 param set /lab2_rrc_methods_vms enabled    true/false
  ros2 param set /lab2_rrc_methods_vms goal_auto_cycle   true/false
  ros2 param set /lab2_rrc_methods_vms goal_cycle_period 30.0

BASE CONTROL (Mobile base):
  ros2 param set /lab2_rrc_methods_vms base_vel_enabled   true/false    # move base when error > threshold
  ros2 param set /lab2_rrc_methods_vms error_threshold    0.15         # (m) when to move base
  ros2 param set /lab2_rrc_methods_vms base_linear_max    0.5          # (m/s) max base speed
  ros2 param set /lab2_rrc_methods_vms base_angular_max   1.0          # (rad/s) max turning speed

All targets in world_enu: x=East, y=North, z=Up.
These are WORLD goals (not relative to arm), accounting for robot pose.
"""


import math


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
    # swiftpro_fk_vms_5dof_ned,     # FK variant for world_ned frame (not used in this node)
    # swiftpro_jacobian_vms_5dof_ned, # Jacobian variant for world_ned frame (not used in this node)
    swiftpro_fk_vms_5dof_ned,     # 5-DOF VMS FK (world_ned → world_enu via TF) [USED]
    swiftpro_jacobian_vms_5dof_ned,   # 5-DOF VMS Jacobian (world_ned → world_enu via TF) [USED]
    DLS,                          # Damped Least-Squares pseudo-inverse
    Q1_MIN, Q1_MAX,               # joint1 base yaw URDF limits  (+-pi/2)
    Q2_MIN, Q2_MAX,               # joint2 shoulder URDF limits  (-pi/2, +0.05)
    Q3_MIN, Q3_MAX,               # joint3 elbow    URDF limits  (-pi/2, +0.05)
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
# Must publish to the namespaced topic where the simulator's diff_drive_controller is listening
BASE_CMD_TOPIC    = '/turtlebot/cmd_vel'  # Twist message: linear.x, angular.z (namespaced!)

# Separate marker topic from lab2_rrc_debug so both nodes can run simultaneously
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

# ---------------------------------------------------------------------------
# Marker IDs — each ID corresponds to a persistent RViz marker object.
# Using fixed IDs means successive publishes UPDATE the marker in place
# rather than creating a new one each tick.
ID_TARGET   = 0   # RED    sphere  — desired target position
ID_TF_EE    = 1   # GREEN  sphere  — true EE from TF (colour interpolates to yellow as error grows)
ID_FK_EE    = 2   # CYAN   sphere  — FK estimate of EE (shows FK accuracy vs TF)
ID_J1_BASE  = 3   # YELLOW sphere  — J1 arm base (origin of arm-local ENU)
ID_ERR_LINE = 4   # WHITE  line    — vector from current EE to target (visual error)
ID_TEXT     = 5   # WHITE  text    — method name, error magnitude, and goal index

# ---------------------------------------------------------------------------
# Automatic goal cycling — DISTANT goals in world_enu (robot-relative, not arm-relative)
# ---------------------------------------------------------------------------
# These positions are set DISTANT from the robot base in world_enu frame.
# The VMS FK/Jacobian will account for the robot pose transformation.
# These are absolute world positions, not offsets from the arm J1.
GOAL_SEQUENCE = [
    np.array([ 1.20,  0.00,  0.30]),   # East — base drives straight forward
    np.array([ 0.00,  1.20,  0.35]),   # North — robot rotates ~90° then advances
    np.array([ 1.00,  0.60,  0.28]),   # East-North diagonal
    np.array([ 0.00, -1.20,  0.30]),   # South — robot rotates ~-90° then advances
    np.array([ 0.80,  0.80,  0.36]),   # NE diagonal, higher z
    np.array([ 1.10, -0.40,  0.28]),   # East-South, lower z
]

# How long (seconds) the arm tries to reach each goal before cycling to the next
DEFAULT_GOAL_CYCLE_PERIOD = 30.0

# Safety margin applied inside the URDF hard limits when clamping velocities.
# Stops the joint slightly before the hard limit to give the controller time
# to decelerate
LIMIT_MARGIN = 0.07   # radians


class Lab2RRCMethodsVMSNode(Node):

    def __init__(self):
        super().__init__('lab2_rrc_methods_vms')

        # ── Declare runtime-tunable parameters ────────────────────────────
        self.declare_parameter('target_x',  0.50)   # East component of target (m, WORLD_enu)
        self.declare_parameter('target_y',   0.00)  # North component of target (m, WORLD_enu)
        self.declare_parameter('target_z',   0.30)  # Up component of target (m, WORLD_enu)

        # Optional EE yaw target (radians).
        # NaN  -> position-only mode: uses 3x4 Jacobian, q4 stays uncontrolled.
        # finite -> orientation mode: uses 4x4 Jacobian, controls position + yaw.
        self.declare_parameter('target_yaw', float('nan'))

        self.declare_parameter('gain_k',    0.5)   # proportional gain K applied to error
        self.declare_parameter('max_vel',   0.3)   # maximum joint speed after scaling (rad/s)
        self.declare_parameter('damping',   0.05)  # DLS lambda: higher = smoother near singularities
        self.declare_parameter('method',    2)     # active RRC method: 0=transpose, 1=pinv, 2=DLS
        self.declare_parameter('enabled',   True)  # False -> send zeros, arm holds still

        # Goal cycling: whether to auto-advance goals and at what rate
        self.declare_parameter('goal_auto_cycle',   True)                    # enable/disable cycling
        self.declare_parameter('goal_cycle_period', DEFAULT_GOAL_CYCLE_PERIOD)  # seconds per goal

        # Mobile base velocity control parameters
        self.declare_parameter('base_vel_enabled', True)  # enable/disable base movement
        self.declare_parameter('base_linear_max',  0.5)   # max linear velocity (m/s)
        self.declare_parameter('base_angular_max', 1.0)   # max angular velocity (rad/s)
        self.declare_parameter('error_threshold',  0.15)  # distance (m): base moves if error > this
        self.declare_parameter('error_stop',       0.05)  # distance (m): base STOPS if error < this

        # SwiftProManipulator4DOF internally holds [q1, q2, q3, q4] and provides
        # FK / Jacobian queries.  Initialised to zeros; updated by _js_cb.
        self.arm = SwiftProManipulator4DOF()

        # Guard flag: skip the control loop until the first /joint_states message
        # arrives so we never compute with stale zero-initialised joint angles
        self._got_js = False

        # Cached robot base pose in world_enu frame (extracted from TF: base_footprint)
        # Used by 5-DOF VMS FK/Jacobian
        self._base_x = 0.0       # robot base x position (East, m)
        self._base_y = 0.0       # robot base y position (North, m)
        self._base_psi = 0.0     # robot base yaw angle (rad)

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

        # Advance goal index, wrapping back to 0 after the last goal
        self._goal_idx = (self._goal_idx + 1) % len(GOAL_SEQUENCE)

        # Retrieve the next goal position from the sequence
        goal = GOAL_SEQUENCE[self._goal_idx]

        # Write the new target back into the ROS parameter server
        self.set_parameters([
            Parameter('target_x', Parameter.Type.DOUBLE, float(goal[0])),  
            Parameter('target_y', Parameter.Type.DOUBLE, float(goal[1])),  
            Parameter('target_z', Parameter.Type.DOUBLE, float(goal[2])),  
        ])

        # Inform which goal is now active
        self.get_logger().info(
            f'[GOAL CYCLE] -> goal {self._goal_idx}/{len(GOAL_SEQUENCE)-1}: '
            f'[{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}] world_enu (DISTANT target)')

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
        mv     = self.get_parameter('max_vel').value     # joint velocity ceiling (rad/s)
        dam    = self.get_parameter('damping').value     # DLS damping factor lambda
        mth    = int(self.get_parameter('method').value) # integer selector: 0/1/2
        
        ena    = self.get_parameter('enabled').value     # bool: if False, publish zeros

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
        q_arm = np.array(self.arm.q[0:3])   # [q1, q2, q3]
        fk_world_enu = swiftpro_fk_vms_5dof(
            self._link1_world_enu, q_arm, self._base_psi, self._tf_buffer)
        # fk_world_enu = swiftpro_fk_vms_5dof_ned(q_arm, self._tf_buffer)
        
        if fk_world_enu is None:
            self.get_logger().warn('FK computation failed, skipping control tick',
                                   throttle_duration_sec=1.0)
            return

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

        #  3×5 VMS Jacobian (computed in world_ned, transformed to world_enu via TF) ──
        #  3×5 VMS Jacobian 
        # swiftpro_jacobian_vms_5dof(ee_world, base_world, base_yaw, arm_q)
        # Same q1_mod substitution as FK — no delta_yaw needed.
        base_world = np.array([self._base_x, self._base_y, 0.0])
        # err = pos_err   # (3, 1)
        J = swiftpro_jacobian_vms_5dof(
            ee_for_jac, base_world, self._base_psi, q_arm, self._tf_buffer)  # (3, 5)
        err = pos_err   # (3, 1)
        # J = swiftpro_jacobian_vms_5dof_ned(q_arm, self._tf_buffer)  # (3, 5)
        if J is None:
            self.get_logger().warn('Jacobian computation failed, skipping control tick',
                                   throttle_duration_sec=1.0)
            return
        K = np.diag([K_gain] * 3)   # (3, 3)

        J_cond = float(np.linalg.cond(J))

        # ── Resolved-Rate Control: ζ = [vx, ω, dq1, dq2, dq3] ───────────────
        dzeta = np.zeros(5)
        if ena:
            if mth % 3 == 0:
                dzeta = 4.0 * (J.T @ K @ err).flatten()
            elif mth % 3 == 1:
                dzeta = (np.linalg.pinv(J) @ K @ err).flatten()
                if np.linalg.norm(dzeta) > 2.0:
                    dzeta = dzeta / np.linalg.norm(dzeta) * 2.0
            else:
                dzeta = (DLS(J, dam) @ K @ err).flatten()

        v_base_linear  = dzeta[0]   # vx: body-frame forward velocity
        v_base_angular = dzeta[1]   # ω:  body-frame yaw rate
        dq_arm         = dzeta[2:5] # [dq1, dq2, dq3]: arm joint velocities

        # Scale arm velocities to respect max_vel
        if np.linalg.norm(dq_arm) > 1e-6:
            scale_factor = mv / np.max(np.abs(dq_arm))
            dq_arm = dq_arm * min(scale_factor, 1.0)

        # Publish base velocity
        base_cmd = Twist()
        base_cmd.linear.x  = v_base_linear
        base_cmd.angular.z = v_base_angular
        self.pub_base_cmd.publish(base_cmd)

        # Arm joint command: controller expects [dq1, dq2, dq3, dq4].
        # q4 is pure EE yaw — set to 0 (position control only).
        # IMPORTANT: append q4=0, do NOT prepend — joint order is 1,2,3,4.
        dq = np.hstack([dq_arm, [0.0]])
        
        # Publish arm joint velocities
        if ena:
            # Joint limit safety clamping (optional)
            dq_arr = dq.flatten()
            
            # Uncomment the following to clamp velocities to joint limits:
            # q = self.arm.q[0:3]
            # m = LIMIT_MARGIN
            # if q[0] <= Q1_MIN + m and dq_arr[0] < 0: dq_arr[0] = 0.0
            # if q[0] >= Q1_MAX - m and dq_arr[0] > 0: dq_arr[0] = 0.0
            # if q[1] <= Q2_MIN + m and dq_arr[1] < 0: dq_arr[1] = 0.0
            # if q[1] >= Q2_MAX - m and dq_arr[1] > 0: dq_arr[1] = 0.0
            # if q[2] <= Q3_MIN + m and dq_arr[2] < 0: dq_arr[2] = 0.0
            # if q[2] >= Q3_MAX - m and dq_arr[2] > 0: dq_arr[2] = 0.0
            
            # Pack joint velocities into the message and publish
            cmd = Float64MultiArray()
            cmd.data = dq_arr.tolist()
            self.pub_cmd.publish(cmd)

        else:
            # Disabled mode: send zero velocity to all joints
            cmd = Float64MultiArray()
            cmd.data = [0.0, 0.0, 0.0, 0.0]
            self.pub_cmd.publish(cmd)

        # Verbose debug log (every 30 loops ≈ 0.5 s)
        if self._loop_count % 30 == 0:
            q = self.arm.q[0:3]
            self.get_logger().info(
                f'\n{"─"*70}\n'
                f'  method      : {method_name}  |  enabled: {ena}  |  mode: 5-DOF VMS\n'
                f'  goal_idx    : {self._goal_idx}/{len(GOAL_SEQUENCE)-1}\n'
                f'  BASE POSE   : x={self._base_x:.3f}  y={self._base_y:.3f}  '
                f'psi={self._base_psi:.3f}\n'
                f'  ARM q[1-3]  : [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}] rad\n'
                f'  target(wenu): [{tx:.3f}, {ty:.3f}, {tz:.3f}]\n'
                f'  TF EE(wenu) : {np.round(ee_world_enu,3) if ee_world_enu is not None else "N/A"}\n'
                f'  FK EE(wenu) : {np.round(fk_world_enu,3)}\n'
                f'  FK_vs_TF    : {fk_vs_tf_mm:.3f} mm\n'
                f'  pos_source  : {position_source}\n'
                f'  err_norm    : {err_norm:.4f} m\n'
                f'  err_vec     : {np.round(pos_err.flatten(),4)}\n'
                f'  [vx, omega] : [{v_base_linear:.4f}, {v_base_angular:.4f}]\n'
                f'  dq[1-3]     : {np.round(dq_arm,4)}\n'
                f'  J_cond      : {J_cond:.1f}\n'
                f'  params      : K={K_gain}  mv={mv}  dam={dam}\n'
                f'{"─"*70}'
            )

        # Markers 
        self._publish_markers(
            target_world_enu, ee_world_enu, fk_world_enu,
            np.array([self._base_x, self._base_y, 0.0]), err_norm, method_name)

    # Marker publisher 

    def _publish_markers(self, target, tf_ee, fk_ee, j1, err_norm, method):
        """
        Publish 6 RViz markers (same set as lab2_rrc_debug_node.py):
          RED    sphere  — target position (DISTANT/WORLD goal)
          GREEN  sphere  — TF EE position (yellow->green as error shrinks)
          CYAN   sphere  — FK estimate of EE (VMS FK)
          YELLOW sphere  — J1 base
          WHITE  line    — error vector from EE to target (visual error)
          WHITE  text    — method name and error magnitude
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
        mt.text = f'[{method}] err={err_norm:.3f}m  goal{self._goal_idx}'
        ma.markers.append(mt)

        # Publish all markers in a single message to minimise latency jitter
        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = Lab2RRCMethodsVMSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
