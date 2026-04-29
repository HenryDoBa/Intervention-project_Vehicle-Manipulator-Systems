#!/usr/bin/env python3
"""
Resolved-Rate Control debug node — 4-DOF uArm Swift Pro (joint1-4).
All control math remains in arm-local ENU (relative to J1/link1).

Runtime parameter adjustment:
  ros2 param set /lab2_rrc_methods_debug target_x   0.05
  ros2 param set /lab2_rrc_methods_debug target_y   0.15
  ros2 param set /lab2_rrc_methods_debug target_z   0.40
  ros2 param set /lab2_rrc_methods_debug target_yaw nan       # position-only
  ros2 param set /lab2_rrc_methods_debug target_yaw 0.5       # position+yaw
  ros2 param set /lab2_rrc_methods_debug gain_k     1.0
  ros2 param set /lab2_rrc_methods_debug max_vel    0.5
  ros2 param set /lab2_rrc_methods_debug damping    0.1
  ros2 param set /lab2_rrc_methods_debug method     0         # 0=transpose,1=pinv,2=DLS
  ros2 param set /lab2_rrc_methods_debug enabled    true/false
  ros2 param set /lab2_rrc_methods_debug goal_auto_cycle   true/false
  ros2 param set /lab2_rrc_methods_debug goal_cycle_period 30.0

All targets in world_enu: x=East, y=North, z=Up.
"""


import math


import rclpy
from rclpy.node import Node           
from rclpy.parameter import Parameter 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from tf2_ros import Buffer, TransformListener, LookupException, \
    ConnectivityException, ExtrapolationException
from hoi_control.swiftpro_robotics_rrc import (
    SwiftProManipulator4DOF,      # 4-DOF arm state + kinematics queries
    swiftpro_fk,
    swiftpro_fk_with_tf_transform,                  # geometric FK: q -> [x, y, z] in arm-local ENU
    swiftpro_jacobian,            # 3x3 position Jacobian (3-DOF, used for FK comparison)
    swiftpro_jacobian_pos4,       # 3x4 position Jacobian (4-DOF, q4 col = [0,0,0])
    swiftpro_jacobian_full4,      # 4x4 position + EE-yaw Jacobian (orientation mode)
    DLS,                          # Damped Least-Squares pseudo-inverse
    scale_velocities,             # uniform velocity scaling to stay within max_vel
    Q1_MIN, Q1_MAX,               # joint1 base yaw URDF limits  (+-pi/2)
    Q2_MIN, Q2_MAX,               # joint2 shoulder URDF limits  (-pi/2, +0.05)
    Q3_MIN, Q3_MAX,               # joint3 elbow    URDF limits  (-pi/2, +0.05)
    Q4_MIN, Q4_MAX,               # joint4 EE yaw   URDF limits  (+-pi/2)
)

# ---------------------------------------------------------------------------
# Topics and TF frames
# ---------------------------------------------------------------------------
# The joint velocity controller listens on this topic; it expects a
# Float64MultiArray with one value per active joint: [dq1, dq2, dq3, dq4]
JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'

# The simulator publishes all joint positions and velocities on this topic
JOINT_STATE_TOPIC = '/turtlebot/joint_states'

# Separate marker topic from lab2_rrc_debug so both nodes can run simultaneously
MARKER_TOPIC      = '/hoi/rrc_methods_markers'

# TF frame in which all world-frame positions are expressed (ENU = East-North-Up)
WORLD_FRAME       = 'world_enu'

# TF frame of the physical end-effector (published by the static_transform_publisher in the launch file)
EE_FRAME          = 'end_effector'

# TF frame of the arm's first joint / base (link1); used to convert between
# world_enu and arm-local ENU by subtracting J1's world position
J1_FRAME          = 'turtlebot/swiftpro/link1'

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
# Automatic goal cycling — workspace targets in world_enu
# ---------------------------------------------------------------------------
# These positions were chosen to lie comfortably inside the arm's reachable
# workspace.  J1 (link1) sits at approximately [0, -0.051, 0.198] world_enu
# when the Turtlebot base is at the origin.
# The node cycles through these goals in order every `goal_cycle_period`
# seconds (default 30 s), demonstrating each RRC method on varied targets.
GOAL_SEQUENCE = [
    np.array([-0.25,  0.00,  0.30]),   
    np.array([ 0.00,  -0.25,  0.35]),   
    np.array([-0.15,  -0.15,  0.28]),   
    np.array([0.10,  0.00,  0.40]),   
    np.array([ 0.00,  -0.25,  0.30]),   
    np.array([-0.20,  -0.10,  0.36]),   
]

# How long (seconds) the arm tries to reach each goal before cycling to the next
DEFAULT_GOAL_CYCLE_PERIOD = 30.0

# Safety margin applied inside the URDF hard limits when clamping velocities.
# Stops the joint slightly before the hard limit to give the controller time
# to decelerate
LIMIT_MARGIN = 0.07   # radians


class Lab2RRCMethodsDebugNode(Node):

    def __init__(self):
        super().__init__('lab2_rrc_methods_debug')

        # ── Declare runtime-tunable parameters ────────────────────────────
        self.declare_parameter('target_x',  -0.25)   # East component of target (m)
        self.declare_parameter('target_y',   0.00)   # North component of target (m)
        self.declare_parameter('target_z',   0.30)   # Up component of target (m)

        # Optional EE yaw target (radians).
        # NaN  -> position-only mode: uses 3x4 Jacobian, q4 stays uncontrolled.
        # finite -> orientation mode: uses 4x4 Jacobian, controls position + yaw.
        self.declare_parameter('target_yaw', float('nan'))

        self.declare_parameter('gain_k',    0.5)   # proportional gain K applied to error
        self.declare_parameter('max_vel',   0.3)   # maximum joint speed after scaling (rad/s)
        self.declare_parameter('damping',   0.1)   # DLS lambda: higher = smoother near singularities
        self.declare_parameter('method',    1)     # active RRC method: 0=transpose, 1=pinv, 2=DLS
        self.declare_parameter('enabled',   True)  # False -> send zeros, arm holds still

        # Goal cycling: whether to auto-advance goals and at what rate
        self.declare_parameter('goal_auto_cycle',   True)                    # enable/disable cycling
        self.declare_parameter('goal_cycle_period', DEFAULT_GOAL_CYCLE_PERIOD)  # seconds per goal

        # SwiftProManipulator4DOF internally holds [q1, q2, q3, q4] and provides
        # FK / Jacobian queries.  Initialised to zeros; updated by _js_cb.
        self.arm = SwiftProManipulator4DOF()

        # Guard flag: skip the control loop until the first /joint_states message
        # arrives so we never compute with stale zero-initialised joint angles
        self._got_js = False

        # Cached world-frame position of J1 (link1) from TF.
        # Subtracting this from any world_enu position converts it to arm-local ENU,
        # which is the frame our FK and Jacobian operate in.
        self._j1_world_enu = None

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

        # Publisher for RViz debug markers (spheres, lines, text)
        self.pub_markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        # Main control timer: fires _control_loop at exactly 60 Hz (every DT seconds)
        self.timer = self.create_timer(DT, self._control_loop)

        # Startup log 
        self.get_logger().info('=' * 62)
        self.get_logger().info('Lab2 RRC Methods DEBUG node started (4-DOF)')
        self.get_logger().info('Tune: ros2 param set /lab2_rrc_methods_debug <param> <value>')
        self.get_logger().info('Params: target_x/y/z, target_yaw, gain_k, max_vel, '
                               'damping, method, enabled, goal_auto_cycle, goal_cycle_period')
        self.get_logger().info('RViz:  Fixed Frame=world_enu  '
                               'MarkerArray from ' + MARKER_TOPIC)
        self.get_logger().info(
            f'Goals: {len(GOAL_SEQUENCE)} targets, cycling every '
            f'{DEFAULT_GOAL_CYCLE_PERIOD:.0f} s  (goal_auto_cycle=True)')
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
            # Request the most recent available transform
            # Returns a TransformStamped; we only need the translation part.
            t = self._tf_buffer.lookup_transform(
                WORLD_FRAME, child_frame, rclpy.time.Time())

            # Extract the x, y, z translation fields from the transform
            tr = t.transform.translation
            
            return np.array([tr.x, tr.y, tr.z])

        except (LookupException, ConnectivityException, ExtrapolationException):
            # TF not yet available (sim still starting) or the frame is unknown.
            # Return None so the caller can decide whether to fall back to FK.
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
            f'[{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}] world_enu')

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

        tx     = self.get_parameter('target_x').value    # target East  (m, world_enu)
        ty     = self.get_parameter('target_y').value    # target North (m, world_enu)
        tz     = self.get_parameter('target_z').value    # target Up    (m, world_enu)
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

        # TF lookups
        # Query the live TF tree for the current world-frame positions of the EE and of J1 (the arm base).
        ee_world_enu = self._tf_pos(EE_FRAME)   # true EE position in world_enu
        j1_world_enu = self._tf_pos(J1_FRAME)   # J1 base position in world_enu

        # Cache the J1 position so it is available on ticks where TF lookup fails
        if j1_world_enu is not None:
            self._j1_world_enu = j1_world_enu

        # Cache the EE position for the debug log
        if ee_world_enu is not None:
            self._last_tf_ee = ee_world_enu

        # ── FK estimate (Jacobian only — position truth comes from TF) ────
        # Compute the FK-predicted EE position in arm-local ENU.
        # This is used only to:
        #   1. Evaluate the Jacobian at the current configuration (J depends on q)
        #   2. Display the FK estimate in RViz for comparison with TF
        # The actual position ERROR uses the TF ground-truth, not this FK value.
        # fk_arm_enu = swiftpro_fk(self.arm.q)   # arm-local ENU, relative to J1
        fk_arm_enu = swiftpro_fk_with_tf_transform(self.arm.q, tf_buffer=self._tf_buffer)
        # Convert FK estimate to world_enu for display (add J1 world offset)
        fk_world_enu = (self._j1_world_enu + fk_arm_enu
                        if self._j1_world_enu is not None
                        else fk_arm_enu)   # fall back to arm-local if J1 unknown

        # Position error: TF-based
        if ee_world_enu is not None and self._j1_world_enu is not None:
            # Convert both EE and target from world_enu to arm-local ENU by
            # subtracting the J1 (arm base) position in world_enu.
            # This places both in the same frame as the FK and Jacobian.
            ee_arm_enu     = ee_world_enu     - self._j1_world_enu  # true EE, arm-local ENU
            target_arm_enu = target_world_enu - self._j1_world_enu  # desired EE, arm-local ENU

            # Position error vector: (desired - current), reshaped to column vector (3x1)
            pos_err = (target_arm_enu - ee_arm_enu).reshape(3, 1)

            # Scalar error magnitude, used for logging and marker colour
            err_norm = float(np.linalg.norm(pos_err))

            # FK accuracy metric: how far is the FK prediction from TF ground truth?
            fk_vs_tf_mm = float(np.linalg.norm(fk_arm_enu - ee_arm_enu)) * 1000

            # Tag for the log: TF data was available
            position_source = 'TF'

        else:
            # TF not yet available — fall back to FK-based error as a last resort.
            # This keeps the arm moving even while the TF tree is still warming up.
            target_arm_enu = (target_world_enu - self._j1_world_enu
                              if self._j1_world_enu is not None
                              else target_world_enu)   # use arm-local if J1 known

            # Error based on FK prediction (less accurate than TF)
            pos_err = (target_arm_enu - fk_arm_enu).reshape(3, 1)

            err_norm    = float(np.linalg.norm(pos_err))
            fk_vs_tf_mm = float('nan')        # can't compute FK vs TF without TF
            position_source = 'FK_fallback'

            # Warn at most once per second so the log stays readable
            self.get_logger().warn('TF unavailable — using FK for error',
                                   throttle_duration_sec=1.0)

        # Build Jacobian and error vector 
        if orient_mode:
            # Orientation + position mode (4x4 Jacobian)
            # Retrieve the current combined EE yaw: yaw = q1 + q4
            current_yaw = self.arm.getEEYaw()

            # Raw yaw error: how far is the current EE yaw from the desired yaw?
            raw_yaw_err = t_yaw - current_yaw

            # Wrap the yaw error to [-pi, pi] to prevent the controller from
            # choosing the long way around (e.g. rotating 350 deg instead of -10 deg)
            yaw_err = (raw_yaw_err + np.pi) % (2 * np.pi) - np.pi

            # Stack position error (3x1) and yaw error (1x1) into a 4x1 error vector
            err = np.vstack([pos_err, [[yaw_err]]])    # shape (4, 1)

            # 4x4 Jacobian: rows = [dx, dy, dz, d(yaw)], cols = [q1, q2, q3, q4]
            J = swiftpro_jacobian_full4(self.arm.q)    # shape (4, 4)

            # Diagonal gain matrix: same scalar gain on all four error dimensions
            K = np.diag([K_gain] * 4)                 # shape (4, 4)

        else:
            # Position-only mode (3x4 Jacobian)
            # 3x4 Jacobian: rows = [dx, dy, dz], cols = [q1, q2, q3, q4]
            # The q4 column is [0, 0, 0] (joint4 is a pure orientation DOF with
            # no moment arm to the EE when L_EE_TOOL = 0), so q4 naturally receives
            # dq4 = 0 — it is left uncontrolled until orientation mode is enabled.
            err = pos_err                              # shape (3, 1)
            J   = swiftpro_jacobian_pos4(self.arm.q, tf_buffer=self._tf_buffer)  # shape (3, 4)
            K   = np.diag([K_gain] * 3)               # shape (3, 3)

        # Condition number of the Jacobian: measures proximity to a singularity.
        # cond >> 100 means the Jacobian is nearly rank-deficient — DLS should be
        # preferred over pseudoinverse in that region.
        J_cond = float(np.linalg.cond(J))

        # Three Resolved-Rate Control methodologies 
        if ena:
            if mth % 3 == 0:
                #  Method 0: Jacobian Transpose 
                # Law:  dq = alpha * J^T * K * e
                dq = 4.0 * (J.T @ K @ err)

            elif mth % 3 == 1:
                # Method 1: Moore-Penrose Pseudo-Inverse
                # Law:  dq = J^+ * K * e   where J^+ = J^T (J J^T)^-1
                # Near singularities J*J^T becomes ill-conditioned and the solution
                # explodes — the hard clamp at 2.0 rad/s prevents motor damage.
                dq = np.linalg.pinv(J) @ K @ err

                # Hard clamp: if the raw command exceeds 2.0 rad/s in any joint,
                # rescale the whole vector (preserving direction) to 2.0 rad/s
                if np.linalg.norm(dq) > 2.0:
                    dq = dq / np.linalg.norm(dq) * 2.0

            else:
                # Method 2: Damped Least-Squares (DLS)
                # Law:  dq = J^T (J J^T + lambda^2 * I)^-1 * K * e
                # Tradeoff: larger lambda -> smoother motion, larger final error.
                #           smaller lambda -> more accurate, but closer to pinv behaviour.
                dq = DLS(J, dam) @ K @ err

            # Uniform velocity scaling 
            # If any joint velocity exceeds max_vel, scale the entire vector down
            # proportionally so the fastest joint runs at exactly max_vel.
            # This preserves the direction of motion (arm still moves toward goal).
            dq = scale_velocities(dq, mv)

            # Joint limit safety clamping
            # zero out any joint velocity that would push a joint past
            # its URDF hard limit minus the LIMIT_MARGIN safety buffer.
            # Without this, the RRC controller will keep commanding positive dq
            # into a saturated joint, wasting the commanded velocity on a joint
            # that physically cannot move.

            # After uncommenting, the block reads joint angles from self.arm.q
            # and zeroes out dq_arr[i] whenever joint i is within LIMIT_MARGIN
            # of either its lower or upper URDF limit AND the velocity would
            # push it further past that limit.  Adjust LIMIT_MARGIN 
            # to change the buffer size; the default is 0.07 rad (~4 degrees).
            #
            # WHEN TO LEAVE IT COMMENTED:
            #   During experiments where we deliberately want to observe what
            #   happens when the controller hits a joint limit unconstrained
            #   (e.g. to study steady-state error, limit-cycling behaviour, or
            #   to verify that the goals in GOAL_SEQUENCE are inside the
            #   reachable workspace before re-enabling the clamp).
            dq_arr = dq.flatten()   # collapse to 1-D for indexed assignment
            q      = self.arm.q     # current joint angles [q1, q2, q3, q4]
            m = LIMIT_MARGIN        # shorthand: LIMIT_MARGIN radians inside each hard limit

            # joint1 (base yaw, +-pi/2)
            # if q[0] <= Q1_MIN + m and dq_arr[0] < 0: dq_arr[0] = 0.0
            # if q[0] >= Q1_MAX - m and dq_arr[0] > 0: dq_arr[0] = 0.0

            # joint2 (shoulder, -pi/2 to +0.05)
            # if q[1] <= Q2_MIN + m and dq_arr[1] < 0: dq_arr[1] = 0.0
            # if q[1] >= Q2_MAX - m and dq_arr[1] > 0: dq_arr[1] = 0.0

            # joint3 (elbow, -pi/2 to +0.05)
            # if q[2] <= Q3_MIN + m and dq_arr[2] < 0: dq_arr[2] = 0.0
            # if q[2] >= Q3_MAX - m and dq_arr[2] > 0: dq_arr[2] = 0.0

            # joint4 (EE yaw, +-pi/2)
            # if q[3] <= Q4_MIN + m and dq_arr[3] < 0: dq_arr[3] = 0.0
            # if q[3] >= Q4_MAX - m and dq_arr[3] > 0: dq_arr[3] = 0.0

            # Pack joint velocities into the message and publish.
            # cmd.data must have exactly one value per controlled joint: [dq1, dq2, dq3, dq4]
            cmd = Float64MultiArray()
            cmd.data = dq_arr.tolist()   # convert NumPy array to plain Python list
            self.pub_cmd.publish(cmd)

        else:
            # Disabled mode: send zero velocity to all joints so the arm holds still.
            # We still publish (rather than stopping the publisher) so the controller
            # receives a continuous stream and does not time out into an error state.
            dq_arr = np.zeros(4)           # four zeros, one per joint
            cmd = Float64MultiArray()
            cmd.data = [0.0, 0.0, 0.0, 0.0]
            self.pub_cmd.publish(cmd)

        # Verbose debug log (every 30 loops ≈ 0.5 s) 
        # Throttled to avoid flooding the terminal; 30 ticks * (1/60 s) = 0.5 s
        if self._loop_count % 30 == 0:
            q = self.arm.q   # snapshot for the log (avoid re-reading mid-format)

            # Build a human-readable mode string for the log header
            mode_str = (f'orient(target_yaw={t_yaw:.2f}rad)'
                        if orient_mode else 'position-only')

            self.get_logger().info(
                f'\n{"─"*60}\n'
                f'  method      : {method_name}  |  enabled: {ena}  |  mode: {mode_str}\n'
                f'  goal_idx    : {self._goal_idx}/{len(GOAL_SEQUENCE)-1}'
                f'  (auto_cycle={self.get_parameter("goal_auto_cycle").value})\n'
                f'  q           : [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] rad\n'
                # Target in world_enu (what was commanded)
                f'  target(wenu): [{tx:.3f}, {ty:.3f}, {tz:.3f}]\n'
                # TF ground-truth EE position (what the sim reports)
                f'  TF EE(wenu) : {np.round(ee_world_enu,3) if ee_world_enu is not None else "N/A"}\n'
                # FK-predicted EE position (what our model thinks)
                f'  FK EE(wenu) : {np.round(fk_world_enu,3)}\n'
                # Discrepancy between FK and TF; should stay below ~20 mm for good control
                f'  FK_vs_TF    : {fk_vs_tf_mm:.3f} mm  (<20 mm = good FK accuracy)\n'
                # Which source was used for the error vector this tick
                f'  pos_source  : {position_source}\n'
                # Euclidean distance from EE to target (the quantity being minimised)
                    f'  err_norm    : {err_norm:.4f} m\n'
                    # Individual x/y/z components of the position error
                f'  err_vec     : {np.round(pos_err.flatten(),4)}\n'
                # Joint velocity command sent this tick
                f'  dq[1-4]     : {np.round(dq_arr,4)}\n'
                f'  J_cond      : {J_cond:.1f}  (>100 = near singularity)\n'
                # Key tuning parameters echoed for reference
                f'  params      : K={K_gain}  mv={mv}  dam={dam}\n'
                f'  J1_world_enu      : {self._j1_world_enu[0]:.3f}, {self._j1_world_enu[1]:.3f}, {self._j1_world_enu[2]:.3f}\n'
                f' fk_arm_enu      : {fk_arm_enu[0]:.3f}, {fk_arm_enu[1]:.3f}, {fk_arm_enu[2]:.3f}\n'
                f'{"─"*60}'
            )

        # Markers 
        # Publish all six RViz debug markers every control tick so the
        # visualisation stays in sync with the control state
        self._publish_markers(
            target_world_enu, ee_world_enu, fk_world_enu,
            self._j1_world_enu, err_norm, method_name)

    # Marker publisher 

    def _publish_markers(self, target, tf_ee, fk_ee, j1, err_norm, method):
        """
        Publish 6 RViz markers (same set as lab2_rrc_debug_node.py):
          RED    sphere  — target position
          GREEN  sphere  — TF EE position (yellow->green as error shrinks)
          CYAN   sphere  — FK estimate of EE
          YELLOW sphere  — J1 base
          WHITE  line    — error vector from EE to target
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
            m.ns                 = 'rrc_methods_debug'  # namespace groups markers in RViz
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

        # RED sphere — desired target position (static shape, bright red)
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

        # CYAN sphere — FK estimate of EE.  If this drifts far from the green TF sphere,
        # the FK model needs recalibration (FK_vs_TF metric in the log will be large).
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
    node = Lab2RRCMethodsDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
