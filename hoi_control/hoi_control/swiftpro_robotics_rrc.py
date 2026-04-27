import math
import numpy as np



from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import Point, Vector3Stamped, TransformStamped
import rclpy
TF_AVAILABLE = True
from tf_transformations import quaternion_matrix
# ---------------------------------------------------------------------------
# uArm Swift Pro – geometric link parameters (metres)
# ---------------------------------------------------------------------------
# These constants were derived by tracing the URDF joint chain geometrically
# (no DH convention — invalid for closed-chain parallelogram linkages).
#
# The kinematic chain traced is:
#   link1 → joint2 → link2 → passive_joint1 → link3
#         → passive_joint7 → link8A → joint4 → link8B → end_effector
#
# L2 and L3 come directly from URDF joint origins:
#   passive_joint1: xyz="0 0 -0.142"  → L2 = 0.142 m (upper arm)
#   passive_joint7: xyz="0.1587 0 0"  → L3 = 0.1588 m (forearm, rounded)
#
# The base constants in swiftpro_fk (0.0127 for reach, -0.0382 for height)
# are NOT pure URDF values — they were empirically recalibrated by comparing
# FK output against TF ground truth at known joint configurations after
# switching the EE frame from link9 to end_effector (link8B + 0.0722m up).
# See swiftpro_fk docstring for the full derivation and calibration history. 
A1 = 0.0      # effective horizontal J1→J2 offset (negligible, from URDF joint2 x=0.0133)
D1 = 0.0333   # vertical height: J1 axis → J2 shoulder pivot (33.3 mm, from URDF joint2 z=-0.1056)
L2 = 0.142    # upper arm length  J2 → J3 pivot (from URDF passive_joint1 z=-0.142)
L3 = 0.1588   # forearm length    J3 → link8A  (from URDF passive_joint7 x=0.1587, rounded)
L4 = 0.0445   # wrist link; always horizontal due to parallelogram constraint

L_J4      = 0.0565   # link8A origin → joint4 pivot / link8B origin (m)
L_EE_TOOL = 0.0      # tool-tip extension beyond link8B origin along link8B x-axis (m)

# Joint angle limits (rad) — copied directly from the URDF <limit> tags.
# A small inset margin (LIMIT_MARGIN = 0.07 rad) is applied in the control node
# to stop joints slightly before the hard limit, giving the controller time
# to decelerate.
Q1_MIN = -1.571   # joint1 base yaw   lower (−π/2)
Q1_MAX =  1.571   # joint1 base yaw   upper (+π/2)
Q2_MIN = -1.571   # joint2 shoulder   lower (−π/2)
Q2_MAX =  0.05    # joint2 shoulder   upper
Q3_MIN = -1.571   # joint3 elbow      lower (−π/2)
Q3_MAX =  0.05    # joint3 elbow      upper
Q4_MIN = -1.571   # joint4 EE yaw     lower (−π/2)
Q4_MAX =  1.571   # joint4 EE yaw     upper (+π/2)

# Joint index mapping in /joint_states
JOINT_NAMES = [
    'turtlebot/swiftpro/joint1',   # index 0 – base yaw   (q1)
    'turtlebot/swiftpro/joint2',   # index 1 – shoulder   (q2)
    'turtlebot/swiftpro/joint3',   # index 2 – elbow      (q3)
]

# 4-DOF name list — same order as the q vector [q1, q2, q3, q4]
JOINT_NAMES_4DOF = JOINT_NAMES + ['turtlebot/swiftpro/joint4']


# Forward Kinematics with TF Buffer Transform
def swiftpro_fk_with_tf_transform(q, tf_buffer=None, source_frame='world_ned', target_frame='world_enu'):
    """
    Forward kinematics using TF buffer to look up coordinate frame transformations.
    
    This function computes the FK in NED frame, then uses the provided TF buffer
    to look up the transformation between source_frame and target_frame, and applies
    that transformation to convert the result.
    
    Arguments
    ---------
    q : array-like, shape (3,) or (4,)
        Joint angles [q1, q2, q3] or [q1, q2, q3, q4]
    
    tf_buffer : tf2_ros.Buffer, optional
        TF buffer containing the transform between source and target frames.
        If None, falls back to direct computation without TF.
    
    source_frame : str
        Source coordinate frame (default: 'world_ned')
    
    target_frame : str
        Target coordinate frame (default: 'world_enu')
    
    Returns
    -------
    p : np.ndarray, shape (3,)
        [x, y, z] displacement from J1 in target_frame-aligned axes.
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
    
    # Compute position in NED frame
    # This is the intermediate representation before frame transformation
    # q1_eff = q1 + np.pi / 2
    q1_eff = q1 - np.pi
    r = 0.0127 - L2 * np.sin(q2) + L3 * np.cos(q3)
    z_enu = -0.0382 + L2 * np.cos(q2) + L3 * np.sin(q3)
    r_ee = r + L_J4
    z_ee = z_enu + 0.0722
    
    # In NED frame
    x_ned = r_ee * np.cos(q1_eff)     # North component in NED
    y_ned = r_ee * np.sin(q1_eff)     # East component in NED
    z_ned = -z_ee                      # Down component in NED
    
    pos_ned = np.array([x_ned, y_ned, z_ned])
    

    if tf_buffer is not None and TF_AVAILABLE:
        try:
            
            transform = tf_buffer.lookup_transform(
                target_frame, 
                source_frame, 
                rclpy.time.Time()
            )
            
            # Extract rotation matrix from the quaternion in the transform

            quat = transform.transform.rotation
            q_array = [quat.x, quat.y, quat.z, quat.w]
            transform_matrix = quaternion_matrix(q_array)
            rotation_matrix = transform_matrix[:3, :3]
            
            # Apply the rotation to convert from NED to ENU
            pos_transformed = rotation_matrix @ pos_ned
            
            return pos_transformed
            
        except Exception as e:
            # If lookup fails, fall back to manual rotation
            print(f"TF lookup failed: {e}. Using manual rotation.")
            # Apply manual 180° rotation around x-axis
            rotation_matrix = np.array([
                [1.0,  0.0,   0.0],
                [0.0, -1.0,   0.0],
                [0.0,  0.0,  -1.0],
            ])
            pos_transformed = rotation_matrix @ pos_ned
            return pos_transformed
    else:
        # Fallback: apply manual rotation matrix (180° around x-axis)
        # NED to ENU: [1, 0, 0; 0, -1, 0; 0, 0, -1]
        rotation_matrix = np.array([
            [1.0,  0.0,   0.0],
            [0.0, -1.0,   0.0],
            [0.0,  0.0,  -1.0],
        ])
        pos_transformed = rotation_matrix @ pos_ned
        return pos_transformed

# ---------------------------------------------------------------------------
# Geometric Forward Kinematics
# ---------------------------------------------------------------------------

def swiftpro_fk(q):
    """
    Arguments
    ---------
    q : array-like, shape (3,) or (4,)
        [q1, q2, q3] or [q1, q2, q3, q4].
        Only q1, q2, q3 affect EE position; q4 adds EE yaw (orientation only,
        assuming L_EE_TOOL = 0).

    Returns
    -------
    p : np.ndarray, shape (3,)
        [x, y, z] displacement from J1 in world_enu-aligned axes (metres).
        Add J1's world_enu position from TF to get absolute world coordinates.
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])

    # q1 phase correction 
    # The formula x=-r·cos(q1), y=r·sin(q1) assumes q1=0 means arm points West
    #(-x in world_enu).  In this simulator, q1=0 means arm points South (-y).
    # Subtracting π/2 bridges that 90° gap. 
    q1_eff = q1 - np.pi / 2

    # Horizontal reach from J1 yaw axis to link8A 
    # r = base_offset - L2·sin(q2) + L3·cos(q3)
    r    = 0.0127 - L2 * np.sin(q2) + L3 * np.cos(q3)

    # Height above J1 in ENU
    # z_enu = base_height + L2·cos(q2) + L3·sin(q3)
    z_enu = -0.0382 + L2 * np.cos(q2) + L3 * np.sin(q3)

    # Extend to end_effector 
    # L_J4 = 0.0565 m: horizontal distance from link8A to link8B (joint4 pivot).
    # This offset is always along the arm's reach direction because the
    # parallelogram keeps link8A horizontal regardless of q2, q3.
    r_ee = r + L_J4

    # 0.0722 m: vertical offset from link8B to end_effector, defined by the
    # static_transform_publisher in the launch file (xyz="0 0 0.0722").
    # This is always vertical (world_enu z) because the parallelogram keeps
    # link8B level at all configurations.
    z_ee = z_enu + 0.0722

    # Apply q1 rotation to get world_enu x and y 
    # The arm sweeps a horizontal circle of radius r_ee as q1 changes.
    # The negative sign on x comes from the original NED derivation, confirmed
    # empirically against TF data.
    x_enu = -r_ee * np.cos(q1_eff)
    y_enu =  r_ee * np.sin(q1_eff)

    return np.array([x_enu, y_enu, z_ee])



# 3-DOF Geometric Jacobian (position only, 3×3)
def swiftpro_jacobian(q):
    """
    Arguments
    ---------
    q : array-like, shape (3,) or (4,) - only first 3 elements used.

    Returns
    -------
    J : np.ndarray, shape (3, 3)
        Rows → [dx, dy, dz] in world_enu-aligned axes.
        Columns → [dq1, dq2, dq3].
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])

    # Same π/2 shift as FK — Jacobian must be consistent with FK
    q1_eff = q1 - np.pi / 2
    s1, c1 = np.sin(q1_eff), np.cos(q1_eff)

    s2, c2 = np.sin(q2), np.cos(q2)
    s3, c3 = np.sin(q3), np.cos(q3)

    r      = 0.0127 - L2 * s2 + L3 * c3   # reach to link8A (note: uses original r constant)
    r_ee   = r + L_J4                       # total reach including joint4 offset

    dr_dq2 = -L2 * c2   # ∂r/∂q2
    dr_dq3 = -L3 * s3   # ∂r/∂q3
    dz_dq2 = -L2 * s2   # ∂z/∂q2
    dz_dq3 =  L3 * c3   # ∂z/∂q3

    J = np.array([
        # ── q1 column ──────    ── q2 column ─────────    ── q3 column ──────
        [  r_ee * s1,            -dr_dq2 * c1,             -dr_dq3 * c1  ],  # ∂x/∂qi
        [  r_ee * c1,             dr_dq2 * s1,              dr_dq3 * s1  ],  # ∂y/∂qi
        [  0.0,                   dz_dq2,                    dz_dq3      ],  # ∂z/∂qi
    ])
    return J



# 4-DOF Jacobians  (include joint4 EE yaw)
def swiftpro_jacobian_pos4(q):
    """
    Arguments
    ---------
    q : array-like, shape (4,) – [q1, q2, q3, q4]

    Returns
    -------
    J : np.ndarray, shape (3, 4)
    """
    q1, q4 = float(q[0]), float(q[3])
    J3 = swiftpro_jacobian(q)   # (3, 3) — first three columns

    # 4th column: EE position sensitivity to q4
    # With L_EE_TOOL=0 this is [0,0,0]; non-zero only if a tool is mounted.
    angle_j4 = q1 + q4   # direction of link8B in the horizontal plane
    col4 = np.array([
        L_EE_TOOL * np.sin(angle_j4),
        L_EE_TOOL * np.cos(angle_j4),
        0.0
    ])

    return np.hstack([J3, col4.reshape(3, 1)])   # (3, 4)


def swiftpro_jacobian_full4(q):
    """
    4-DOF full Jacobian mapping [dq1, dq2, dq3, dq4] → [dx, dy, dz, d(yaw)].
    Shape: (4, 4).  Use this for combined position + EE yaw control.

    The top 3 rows are swiftpro_jacobian_pos4 (position).
    The bottom row maps joint velocities to EE yaw rate.

    Yaw row derivation:
    ───────────────────
    EE yaw in world_enu = q1 + q4.
    Both q1 (base yaw) and q4 (EE yaw joint) rotate the EE about the same
    vertical axis.  The parallelogram constraint keeps pitch and roll fixed.

        dyaw/dq1 = 1,  dyaw/dq2 = 0,  dyaw/dq3 = 0,  dyaw/dq4 = 1

    When using this Jacobian, the error vector must be 4X1: [ex, ey, ez, e_yaw].
    Always wrap e_yaw to [−π, π] before use to avoid the controller spinning
    the long way around.

    Arguments
    ---------
    q : array-like, shape (4,)

    Returns
    -------
    J : np.ndarray, shape (4, 4)
    """
    J_pos4  = swiftpro_jacobian_pos4(q)              # (3, 4)
    yaw_row = np.array([[1.0, 0.0, 0.0, 1.0]])       # dyaw/d[q1,q2,q3,q4]
    return np.vstack([J_pos4, yaw_row])               # (4, 4)



# Damped Least-Squares pseudo-inverse
def DLS(A, damping):
    """
    Damped Least-Squares pseudo-inverse.
        A_dls = A^T (A A^T + lambda^2*I)^(-1)

    Preferred over the plain Moore-Penrose pseudo-inverse near singular
    configurations because it trades a small amount of accuracy for numerical
    stability.  Near singularities, A·A^T becomes ill-conditioned and its
    plain inverse explodes — adding lambda^2·I ensures the matrix is always
    invertible.

    Trade-off:
      larger lambda → smoother motion, larger steady-state position error.
      smaller lambda → more accurate, approaches plain pseudo-inverse behaviour.
      lambda = 0     → exactly equivalent to plain pseudo-inverse.
    """
    lam2 = damping ** 2
    m    = A.shape[0]
    return A.T @ np.linalg.inv(A @ A.T + lam2 * np.eye(m))


def weighted_DLS(A, damping, W):
    """
    Weighted Damped Least-Squares pseudo-inverse.
        A_wdls = W^(-1) Aᵀ (A W^(-1) Aᵀ + lambda^2*I)^(-1)

    W is a positive-definite weight matrix that penalises certain joints more
    heavily.  For example, a diagonal W with joint velocity limits on the
    diagonal prevents fast joints from dominating the solution.
    """
    lam2  = damping ** 2
    m     = A.shape[0]
    W_inv = np.linalg.inv(W)
    return W_inv @ A.T @ np.linalg.inv(A @ W_inv @ A.T + lam2 * np.eye(m))


def scale_velocities(zeta, max_vel):
    """
    Scale the joint velocity vector uniformly so that no element exceeds
    max_vel in absolute value.  Preserves the direction of motion — all
    joints slow down by the same factor so the arm still moves toward the
    target, just slower.

    If the maximum absolute value in zeta is already ≤ max_vel, no scaling
    is applied and zeta is returned unchanged.
    """
    s = np.max(np.abs(zeta)) / max_vel
    if s > 1.0:
        zeta = zeta / s
    return zeta



#  SwiftProManipulator4DOF  –  4-DOF kinematic state manager
class SwiftProManipulator4DOF:
    """
    Kinematic state manager for the uArm Swift Pro with 4 active joints.

    Wraps the four active joint angles [q1, q2, q3, q4] and exposes FK and
    Jacobian queries needed by the RRC control loop.  Updated each control
    tick from the /joint_states topic via update_from_joint_states().

    Joint roles:
      q1 — base yaw:     rotates the whole arm around the vertical axis
      q2 — shoulder:     tilts the upper arm up/down
      q3 — elbow:        tilts the forearm (but due to parallelogram, this is
                         independent of q2 in world_enu — see FK docstring)
      q4 — EE yaw:       rotates the end-effector about the vertical axis

    Usage inside a ROS node:
        arm = SwiftProManipulator4DOF()
        arm.update_from_joint_states(msg.name, msg.position)
        p   = arm.getEEPosition()       # (3,) EE position relative to J1
        J   = arm.getEEJacobianPos()   # (3, 4) position Jacobian
        J4  = arm.getEEJacobianFull()  # (4, 4) position + yaw Jacobian
        yaw = arm.getEEYaw()           # scalar EE yaw = q1 + q4
    """

    DOF = 4   # q1, q2, q3, q4 (all active)

    def __init__(self, q0=None):
        """
        q0 : initial joint angles [q1, q2, q3, q4] (rad).
             Defaults to all zeros.
        """
        self.q = np.zeros(4) if q0 is None else np.array(q0, dtype=float)

    #  State update                                                        
    def update_from_joint_states(self, names, positions):
        """
        Parse a sensor_msgs/JointState message and extract q1-q4.

        Joints are matched by name (not by index) so the ordering in the
        JointState message does not matter.  If joint4 is not present in the
        message (e.g. the simulator does not publish it), q4 retains its
        previous value.

        Arguments
        ---------
        names     : list[str]   - JointState.name
        positions : list[float] - JointState.position  (same length as names)
        """
        pos_map = dict(zip(names, positions))
        for i, jn in enumerate(JOINT_NAMES_4DOF):
            if jn in pos_map:
                self.q[i] = pos_map[jn]

    def integrate(self, dq, dt):
        """
        Forward-Euler integration: q <- q + dq · dt.

        Used when the control node maintains its own internal joint angle
        estimate (e.g. when /joint_states is unavailable or delayed).
        In normal operation, update_from_joint_states() is preferred because
        it uses the simulator's ground-truth joint positions.
        """
        self.q = self.q + np.array(dq, dtype=float).flatten()[:4] * dt


    #  Kinematics queries                                                  
    def getDOF(self):
        return self.DOF

    def getJointAngles(self):
        """Returns a copy of [q1, q2, q3, q4] in radians."""
        return self.q.copy()

    def getJointPos(self, idx):
        """Return scalar angle of joint idx (0-based) in radians."""
        return float(self.q[idx])

    def getEEPosition(self):
        """
        3-D EE position in world_enu-aligned axes, relative to J1 (metres).
        Only q1, q2, q3 affect position; q4 is pure orientation (yaw only).
        Add J1's world_enu position from TF to get absolute world coordinates.
        """
        return swiftpro_fk(self.q)

    def getEEYaw(self):
        """
        Scalar EE yaw angle in world_enu (radians).
        yaw = q1 + q4: both joints rotate the EE about the vertical axis.
        The parallelogram constraint keeps pitch and roll fixed at all times.
        """
        return float(self.q[0]) + float(self.q[3])

    def getEEJacobianPos(self):
        """
        3X4 position Jacobian mapping dq -> linear EE velocity in
        world_enu-aligned axes.  4th column = [0,0,0] when L_EE_TOOL=0
        (q4 is a pure orientation DOF with no positional effect).
        """
        return swiftpro_jacobian_pos4(self.q)

    def getEEJacobianFull(self):
        """
        4X4 Jacobian mapping dq -> [dx, dy, dz, d(yaw)] in world_enu-aligned
        axes.  Use for combined position + EE yaw control tasks.
        """
        return swiftpro_jacobian_full4(self.q)