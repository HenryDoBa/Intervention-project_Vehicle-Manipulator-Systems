"""
swiftpro_robotics.py
====================
Core kinematics and task-priority library for the uArm Swift Pro mounted on
the Kobuki Turtlebot 2.

KEY DESIGN DECISION – No Denavit-Hartenberg:
  The uArm Swift Pro uses a parallelogram (closed-chain) linkage, which makes
  the standard DH convention invalid for the full structure.  Instead, all
  kinematics are derived geometrically by reasoning about the arm geometry
  directly.

  The parallelogram constraint enforces:
      q4 = -(q2 + q3)          (passive joint – handled by swiftpro_controller)

  This means the wrist link is always horizontal, reducing the independent
  position DOF to 3:  q = [q1, q2, q3]

Coordinate convention – NED (North-East-Down):
  The Stonefish simulator uses NED.  In NED, Z points *downward*, so the
  arm extending upward produces *negative* Z coordinates.  All FK / Jacobian
  expressions below are consistent with NED.

  The spec notes that all revolute joints rotate in the *opposite* direction
  to the drawing.  Joint angles read from /joint_states are already in NED
  convention, so NO additional sign flip is needed inside this library.

Authors : HOI Team  (Haadi / Huy / Phu)
"""

import numpy as np

# ---------------------------------------------------------------------------
# uArm Swift Pro – geometric link parameters (metres)
# ---------------------------------------------------------------------------
# Calibrated against live TF data (ros2 run tf2_ros tf2_echo world_enu link9)
# at q = [π/2, -0.997, 0.05], which gave EE at [0, 0.159, 0.480] world_enu
# and link1 (J1 axis) at [0, -0.051, 0.198] world_enu.
# EE relative to J1 = [0, 0.210, 0.282] world_enu (arm-local ENU).
#
# KEY DISCOVERIES vs. the DH table in Schütz & Hepworth (2024):
#
#   D1: The DH table lists d1=106.1mm — this is a CUMULATIVE DH offset that
#       includes the full mounting column, NOT the kinematic J1→J2 distance.
#       From the drawing, the green measurement between J1 level and J2/J3
#       level is 33.3mm.  Back-calculation from TF confirms D1=33.3mm.
#
#   A1: The DH table lists a1=13.2mm — this is the horizontal offset from the
#       Turtlebot deck centre to J1's mounting position, NOT a kinematic arm
#       parameter.  J2 is directly above J1 so A1=0 in the FK.
#       The 13.2mm appears in the drawing as the small blue arrow at the base.
#
#   L2, L3, L4: confirmed correct from the DH table.

A1 = 0.0      # effective horizontal J1→J2 offset (negligible; see note above)
D1 = 0.0333   # vertical height J1 axis → J2 shoulder pivot (33.3mm from drawing)
L2 = 0.142    # upper arm length  J2 → J3 pivot
L3 = 0.1588   # forearm length    J3 → J4 bracket
L4 = 0.0445   # wrist link length J4 → end-effector (always horizontal, parallelogram)
               # Due to the parallelogram constraint, this link is always
               # horizontal (parallel to the ground plane).

# Joint index mapping in the /joint_states message
# (verify against your URDF; adjust if needed)
JOINT_NAMES = [
    'turtlebot/swiftpro/joint1',   # index 0 – base yaw   (q1)
    'turtlebot/swiftpro/joint2',   # index 1 – shoulder   (q2)
    'turtlebot/swiftpro/joint3',   # index 2 – elbow      (q3)
]

# ---------------------------------------------------------------------------
# Geometric Forward Kinematics
# ---------------------------------------------------------------------------

def swiftpro_fk(q):
    """
    Geometric forward kinematics for the uArm Swift Pro.

    OUTPUT FRAME: arm-local ENU, relative to J1 (link1) position.
      x_enu = East   (arm reach direction when q1=0)
      y_enu = North  (arm reach direction when q1=π/2)
      z_enu = Up     (positive = above J1)

    Why arm-local ENU (not NED)?
    ─────────────────────────────
    The TF tree (world_enu → link1) confirms that:
      - At q1=π/2, the arm reaches in the +y_enu (North) direction.
      - The formula  [r·cos(q1), r·sin(q1)]  correctly produces [0, r]
        at q1=π/2, matching TF measurements.
      - Height is POSITIVE upward (z_enu > 0 when arm is raised).

    Derivation (in the vertical plane, before applying q1 rotation):
    ─────────────────────────────────────────────────────────────────
    Horizontal reach from J1 vertical axis to EE:
        r = L2·cos(q2) + L3·cos(q2+q3) + L4

    The NED joint reversal (spec: joints rotate opposite to drawing) means:
      q2_sim < 0 → arm raised   (drawing's positive q2 direction reversed)
      q2_sim > 0 → arm lowered

    Height above J1 (z_enu, positive = up):
        z = D1 + L2·sin(−q2) + L3·sin(−q2−q3)
          = D1 − L2·sin(q2) − L3·sin(q2+q3)

    Note: at q2_sim = -0.997 (arm raised ~57°):
        z = 0.033 − 0.142·sin(−0.997) − 0.1588·sin(−0.947)
          = 0.033 + 0.119 + 0.129 = 0.281 m   (TF ground truth: 0.282 m ✓)

    3-D position:
        x_enu = r · cos(q1)
        y_enu = r · sin(q1)
        z_enu = D1 − L2·sin(q2) − L3·sin(q2+q3)

    Arguments
    ---------
    q : array-like, shape (3,)  – [q1, q2, q3] simulator joint angles (rad)

    Returns
    -------
    p : np.ndarray, shape (3,)  – [x, y, z] in arm-local ENU (metres)
        Origin is at link1 (J1 yaw axis).  Add link1_world_enu for world frame.
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])

    # ── URDF-derived FK (corrected April 2026) ────────────────────────────
    # Chain: joint2_pivot → upper_arm(0.142m, -z of link2) → link3
    #        link3 → forearm(0.1587m, +x of link3) → link8A
    #        link8A → wrist(0.0274, 0, -0.027) → link9
    #
    # URDF uses NED convention (z-down). At q2=0: arm points straight UP (-z).
    # At q2=-π/2: arm is horizontal (+x).  So q2=0 is MINIMUM reach, not max.
    #
    # joint2 pivot offset from link1: (0.0133, 0, -0.1056) [NED]
    # Parallelogram keeps link3 orientation fixed; q3 sets forearm absolute tilt.
    #
    # Horizontal reach in NED from link1:
    #   r = 0.0133 + (-0.142·sin(q2)) + 0.1587·cos(q3) + 0.0274
    #     = 0.0407 - 0.142·sin(q2) + 0.1587·cos(q3)
    #
    # Height above link1 in ENU (= -z_NED):
    #   z = 0.1056 + 0.142·cos(q2) + 0.1587·sin(q3) + 0.027
    #     = 0.1326 + 0.142·cos(q2) + 0.1587·sin(q3)
    #
    # VERIFIED against TF:
    #   q=(π/2, 0.050, 0.050): r_pred=192mm, r_tf=192mm ✓  z_pred=283mm, z_tf=283mm ✓
    #   q=(π/2, -1.571, 0.050): r_pred=341mm, r_tf≈337mm ✓
    #
    # x_enu = -r·cos(q1)  [NEGATIVE sign confirmed by arm_dir=q1 test]
    # ─────────────────────────────────────────────────────────────────────

    # r     = 0.0407 - L2*np.sin(q2) + L3*np.cos(q3)

    r     = 0.0698 - L2*np.sin(q2) + L3*np.cos(q3)

    # z_enu = 0.1326 + L2*np.cos(q2) + L3*np.sin(q3)

    z_enu = 0.0334 + L2*np.cos(q2) + L3*np.sin(q3)
    
    x_enu = -r * np.cos(q1)
    y_enu =  r * np.sin(q1)

    return np.array([x_enu, y_enu, z_enu])


def swiftpro_fk_full(q):
    """
    Returns the full 4×4 homogeneous transformation of the end-effector
    in arm-local ENU (relative to J1 axis).
    Orientation: pure yaw = q1 (EE always level, parallelogram constraint).
    """
    q1 = float(q[0])
    p  = swiftpro_fk(q)   # arm-local ENU

    c1, s1 = np.cos(q1), np.sin(q1)
    T = np.array([
        [ c1, -s1,  0,  p[0]],
        [ s1,  c1,  0,  p[1]],
        [  0,   0,  1,  p[2]],
        [  0,   0,  0,  1   ]
    ])
    return T


# ---------------------------------------------------------------------------
# Geometric Jacobian (position part, 3×3)
# ---------------------------------------------------------------------------

def swiftpro_jacobian(q):
    """
    Geometric (analytic) Jacobian for the uArm Swift Pro arm.

    Maps joint velocities [dq1, dq2, dq3] → EE linear velocity in arm-local ENU.
    OUTPUT FRAME: same as swiftpro_fk — arm-local ENU, relative to J1.

    Derivation (partial derivatives of swiftpro_fk):
    ─────────────────────────────────────────────────
        r   = L2·c2 + L3·c23 + L4
        x   = r·cos(q1),   y = r·sin(q1),   z = D1 − L2·s2 − L3·s23

    ∂x/∂q1 = −r·sin(q1)          ∂y/∂q1 = r·cos(q1)     ∂z/∂q1 = 0

    dr/dq2 = −L2·s2 − L3·s23
    ∂x/∂q2 = dr/dq2 · cos(q1)    ∂y/∂q2 = dr/dq2 · sin(q1)
    ∂z/∂q2 = −L2·c2 − L3·c23     (derivative of −L2·sin(q2) − L3·sin(q2+q3))

    ∂x/∂q3 = −L3·s23 · cos(q1)   ∂y/∂q3 = −L3·s23 · sin(q1)
    ∂z/∂q3 = −L3·c23

    Note: the z-row has the SAME formula as the original NED Jacobian because
    D1 is a constant and disappears in the derivative.  Only the FK value
    for z changed, not its partial derivatives.

    Arguments
    ---------
    q : array-like, shape (3,)

    Returns
    -------
    J : np.ndarray, shape (3,3)
        Rows → [dx_enu, dy_enu, dz_enu];  Columns → [q1, q2, q3].
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])

    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s3, c3 = np.sin(q3), np.cos(q3)

    # r = 0.0407 - L2*sin(q2) + L3*cos(q3)
    # z = 0.1326 + L2*cos(q2) + L3*sin(q3)
    # x = -r*cos(q1),  y = r*sin(q1)
    #
    # ∂r/∂q2 = -L2*cos(q2)        [NEGATIVE when arm raised → arm contracts ✓]
    # ∂r/∂q3 = -L3*sin(q3)        [small at q3≈0 ✓]
    # ∂z/∂q2 = -L2*sin(q2)
    # ∂z/∂q3 =  L3*cos(q3)
    #
    # ∂x/∂q1 = r*sin(q1)          [from d(-r*cos)/dq1]
    # ∂y/∂q1 = r*cos(q1)          [from d(r*sin)/dq1]
    # ∂x/∂q2 = (L2*cos(q2))*cos(q1)
    # ∂y/∂q2 = -(L2*cos(q2))*sin(q1)   ← sign from x=-r*cos + chain rule
    # ∂x/∂q3 = (L3*sin(q3))*cos(q1)
    # ∂y/∂q3 = -(L3*sin(q3))*sin(q1)

    # r = 0.0407 - L2*s2 + L3*c3
    r     = 0.0698 - L2*np.sin(q2) + L3*np.cos(q3)
    
    dr_dq2 = -L2*c2
    dr_dq3 = -L3*s3
    dz_dq2 = -L2*s2
    dz_dq3 =  L3*c3

    J = np.array([
        # ── q1 ──────    ── q2 ───────────────    ── q3 ────────────
        [  r*s1,           -dr_dq2*c1,              -dr_dq3*c1  ],   # dx_enu
        [  r*c1,            dr_dq2*s1,               dr_dq3*s1  ],   # dy_enu
        [  0.0,             dz_dq2,                  dz_dq3     ],   # dz_enu
    ])
    return J


# ---------------------------------------------------------------------------
# Geometric Inverse Kinematics (analytical, for reference / initialisation)
# ---------------------------------------------------------------------------

def swiftpro_ik(p_enu):
    """
    Closed-form geometric IK for the uArm Swift Pro.

    Arguments
    ---------
    p_enu : array-like, shape (3,)
        Desired EE position [x, y, z] in arm-local ENU (relative to J1).

    Returns
    -------
    q : np.ndarray, shape (3,)  or  None if unreachable
    """
    x, y, z = float(p_enu[0]), float(p_enu[1]), float(p_enu[2])

    # Base rotation
    q1 = np.arctan2(y, x)

    # Horizontal reach and height in the vertical arm plane
    horiz = np.sqrt(x**2 + y**2)    # horizontal reach from J1 axis
    vert  = z - D1                   # height above J2 (ENU: positive = up)

    # Straight-line distance J2 → EE (wrist included in horiz via L4 subtraction)
    s = horiz - L4
    l_sq = s**2 + vert**2
    l    = np.sqrt(l_sq)

    # Cosine rule
    cos_q3_eff = (L2**2 + L3**2 - l_sq) / (2.0 * L2 * L3)
    if abs(cos_q3_eff) > 1.0:
        return None

    # Effective angle (arm plane convention: positive = raised)
    q3_eff = np.arccos(np.clip(cos_q3_eff, -1.0, 1.0))

    alpha   = np.arctan2(vert, s)
    beta    = np.arctan2(L3 * np.sin(q3_eff), L2 + L3 * np.cos(q3_eff))
    q2_eff  = alpha - beta

    # Convert back to simulator angles (NED reversal: q_sim = -q_eff)
    q2 = -q2_eff
    q3 = -q3_eff

    return np.array([q1, q2, q3])


# ---------------------------------------------------------------------------
# Utility – robot characteristic points for 3-D visualisation
# ---------------------------------------------------------------------------

def swiftpro_points(q):
    """
    Returns the 3-D positions of: base, J2 pivot, J3 pivot, and end-effector.
    Useful for publishing RViz markers or checking workspace.

    Returns
    -------
    pts : np.ndarray, shape (4,3)   – one row per point, NED frame.
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
    c1, s1 = np.cos(q1), np.sin(q1)

    # Base origin
    p0 = np.array([0.0, 0.0, 0.0])

    # J2 pivot  (top of vertical column)
    p1 = np.array([A1*c1, A1*s1, -D1])

    # J3 pivot
    r2 = A1 + L2*np.cos(q2)
    p2 = np.array([r2*c1, r2*s1, -(D1 + L2*np.sin(q2))])

    # End-effector
    p3 = swiftpro_fk(q)

    return np.array([p0, p1, p2, p3])


# ---------------------------------------------------------------------------
# Frame conversion helpers  (ENU ↔ NED)
# ---------------------------------------------------------------------------
# The Stonefish simulator and the arm kinematics use NED (North-East-Down).
# RViz, odometry, and all ROS nav topics use ENU (East-North-Up).
#
# Conversion (same rotation applied both ways, just swapping axes):
#   x_enu =  y_ned      x_ned =  y_enu
#   y_enu =  x_ned      y_ned =  x_enu
#   z_enu = -z_ned      z_ned = -z_enu

def ned_to_enu(p):
    """Convert a 3-D point/vector from NED to ENU."""
    p = np.array(p, dtype=float).flatten()
    return np.array([ p[1],  p[0], -p[2]])

def enu_to_ned(p):
    """Convert a 3-D point/vector from ENU to NED."""
    p = np.array(p, dtype=float).flatten()
    return np.array([ p[1],  p[0], -p[2]])

# ---------------------------------------------------------------------------
# DLS  (unchanged from lab2_robotics.py)
# ---------------------------------------------------------------------------

def DLS(A, damping):
    """
    Damped Least-Squares pseudo-inverse.
    A_dls = Aᵀ (A Aᵀ + λ²I)⁻¹
    """
    lam2 = damping ** 2
    m    = A.shape[0]
    return A.T @ np.linalg.inv(A @ A.T + lam2 * np.eye(m))


def weighted_DLS(A, damping, W):
    """Weighted Damped Least-Squares:  W⁻¹ Aᵀ (A W⁻¹ Aᵀ + λ²I)⁻¹"""
    lam2  = damping ** 2
    m     = A.shape[0]
    W_inv = np.linalg.inv(W)
    return W_inv @ A.T @ np.linalg.inv(A @ W_inv @ A.T + lam2 * np.eye(m))


# ===========================================================================
#  Manipulator class  –  wraps geometric kinematics + joint state
# ===========================================================================

class SwiftProManipulator:
    """
    Manages the kinematic state of the uArm Swift Pro arm.

    This class replaces the generic Manipulator from lab4_robotics.py.
    Instead of DH parameters, it stores the three active joint angles and
    uses swiftpro_fk / swiftpro_jacobian for all computations.

    Usage (inside a ROS node):
    --------------------------
        arm = SwiftProManipulator()
        arm.update_from_joint_states(joint_names, positions)
        J   = arm.getEEJacobian()    # 3×3
        p   = arm.getEEPosition()    # (3,)
        arm.integrate(dq, dt)        # update joint angles
    """

    DOF = 3   # q1, q2, q3  (q4 is passive, not counted)

    def __init__(self, q0=None):
        """
        q0 : initial joint angles [q1, q2, q3] (rad).  Defaults to zeros.
        """
        self.q = np.zeros(3) if q0 is None else np.array(q0, dtype=float)

    # ------------------------------------------------------------------ #
    #  State update                                                        #
    # ------------------------------------------------------------------ #

    def update_from_joint_states(self, names, positions):
        """
        Parse a sensor_msgs/JointState message and extract the three
        active joint angles.  Call this in your /joint_states callback.

        Arguments
        ---------
        names     : list[str]   – JointState.name
        positions : list[float] – JointState.position  (same order)
        """
        pos_map = dict(zip(names, positions))
        for i, jn in enumerate(JOINT_NAMES):
            if jn in pos_map:
                self.q[i] = pos_map[jn]

    def integrate(self, dq, dt):
        """Euler-integrate joint velocities to update joint angles."""
        self.q = self.q + np.array(dq, dtype=float).flatten()[:3] * dt

    # ------------------------------------------------------------------ #
    #  Kinematics queries                                                  #
    # ------------------------------------------------------------------ #

    def getDOF(self):
        return self.DOF

    def getJointAngles(self):
        """Returns a copy of [q1, q2, q3]."""
        return self.q.copy()

    def getJointPos(self, idx):
        """Return the scalar position of joint idx (0-based)."""
        return float(self.q[idx])

    def getEEPosition(self):
        """3-D end-effector position (NED, metres)."""
        return swiftpro_fk(self.q)

    def getEETransform(self):
        """4×4 homogeneous EE transform."""
        return swiftpro_fk_full(self.q)

    def getEEJacobian(self):
        """3×3 position Jacobian (rows: x,y,z; cols: q1,q2,q3)."""
        return swiftpro_jacobian(self.q)

    def getLinkPosition(self, link_idx):
        """
        3-D position of intermediate link frame.
        link_idx: 0=base, 1=J2 pivot, 2=J3 pivot, 3=EE
        """
        pts = swiftpro_points(self.q)
        return pts[link_idx]

    def getLinkJacobian(self, link_idx):
        """
        Jacobian for an intermediate link (approximate: uses sub-chain FK).
        Needed for orientation or partial-chain tasks.
        link_idx: 1=J2, 2=J3, 3=EE
        """
        q1, q2, q3 = self.q
        c1, s1 = np.cos(q1), np.sin(q1)

        if link_idx <= 1:
            # Only q1 matters for J2 pivot position
            r = A1
            J = np.array([
                [-r*s1, 0, 0],
                [ r*c1, 0, 0],
                [ 0,    0, 0]
            ])
        elif link_idx == 2:
            # J3 pivot: depends on q1 and q2
            r    = A1 + L2*np.cos(q2)
            dr2  = -L2*np.sin(q2)
            J = np.array([
                [-r*s1,      dr2*c1,  0],
                [ r*c1,      dr2*s1,  0],
                [ 0,        -L2*np.cos(q2), 0]   # NED
            ])
        else:
            # Full EE Jacobian
            J = swiftpro_jacobian(self.q)

        return J

    def getCharacteristicPoints(self):
        """All 4 characteristic points (base, J2, J3, EE) as (4,3) array."""
        return swiftpro_points(self.q)


# ===========================================================================
#  MobileManipulator class  –  base (2 DOF) + arm (3 DOF) = 5 DOF total
# ===========================================================================

class MobileManipulator:
    """
    Combined kinematic model of the Kobuki Turtlebot 2 + uArm Swift Pro.

    State vector: [x, y, theta, q1, q2, q3]
      x, y, theta  – base pose in world frame (ENU, from odometry)
      q1, q2, q3   – arm joint angles (NED, from joint_states)

    Control inputs: [v, omega, dq1, dq2, dq3]
      v, omega  – base linear and angular velocity (cmd_vel)
      dq1..dq3  – arm joint velocities (joint_velocity_controller)

    DOF = 5 (2 base + 3 arm), making this a redundant system for a 3-D
    position task – ideal for Task-Priority redundancy resolution.
    """

    DOF = 5

    def __init__(self):
        # Base pose (ENU world frame)
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0   # heading (rad)
        # Arm joints
        self.arm   = SwiftProManipulator()

    # ------------------------------------------------------------------ #
    #  State updates                                                       #
    # ------------------------------------------------------------------ #

    def update_base_from_odom(self, x, y, theta):
        self.x, self.y, self.theta = float(x), float(y), float(theta)

    def update_arm_from_joint_states(self, names, positions):
        self.arm.update_from_joint_states(names, positions)

    def integrate(self, xi, dt):
        """
        xi : [v, omega, dq1, dq2, dq3]  –  velocity command vector (5×1 or (5,))
        """
        xi = np.array(xi).flatten()
        v, omega       = xi[0], xi[1]
        dq             = xi[2:5]

        # Unicycle model for the differential-drive base
        self.x     += v * np.cos(self.theta) * dt
        self.y     += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # Arm joint integration
        self.arm.q += dq * dt

    # ------------------------------------------------------------------ #
    #  Combined Jacobian  (3×5)                                           #
    # ------------------------------------------------------------------ #

    def getEEJacobian(self):
        """
        Full 3×5 Jacobian mapping [v, omega, dq1, dq2, dq3] → [dpx, dpy, dpz].

        Base contribution:
          The EE position in the world frame is:
              p_ee_world = base_position + R(theta) · p_ee_arm
          where R(theta) is the 2-D rotation of the base and p_ee_arm is the
          EE position in the arm (robot) frame.

          ∂p_ee/∂v     = [cos θ, sin θ, 0]ᵀ
          ∂p_ee/∂omega = [–(py_arm·cos θ + px_arm·sin θ),  ...] (see below)

        Arm contribution:
          Rotated version of the arm-frame Jacobian (q1 is already in NED so
          it naturally aligns with the base yaw).

        NOTE: This is an approximate Jacobian that treats the base and arm
        frames as perfectly aligned (no mounting offset modelling).  For a
        more accurate version the arm base transform should be included.
        """
        theta = self.theta
        cT, sT = np.cos(theta), np.sin(theta)

        # Arm EE position in arm (robot) frame
        p_arm = self.arm.getEEPosition()   # [px_a, py_a, pz_a]
        px_a, py_a = p_arm[0], p_arm[1]

        # -- Base columns ------------------------------------------------
        # Column 0: linear velocity v
        Jv = np.array([cT, sT, 0.0])

        # Column 1: angular velocity omega (rotates the entire arm around base Z)
        # d/dt [cT·px_a – sT·py_a] = –sT·px_a – cT·py_a  (chain rule)
        J_omega = np.array([
            -sT * px_a - cT * py_a,   # d(px_world)/d(omega)
             cT * px_a - sT * py_a,   # d(py_world)/d(omega)
             0.0
        ])

        # -- Arm columns (rotate arm Jacobian into world frame) -----------
        J_arm = self.arm.getEEJacobian()   # 3×3 in arm frame
        # Rotation matrix (2-D, applied to x-y rows)
        R = np.array([
            [ cT, -sT,  0],
            [ sT,  cT,  0],
            [  0,   0,  1]
        ])
        J_arm_world = R @ J_arm   # 3×3 in world frame

        # Assemble full 3×5 Jacobian
        J = np.hstack([
            Jv.reshape(3,1),
            J_omega.reshape(3,1),
            J_arm_world
        ])
        return J

    def getEEPosition(self):
        """EE position in the world frame (ENU, metres)."""
        cT, sT = np.cos(self.theta), np.sin(self.theta)
        p_arm  = self.arm.getEEPosition()
        # Rotate arm-frame position into world frame and add base position
        px_w = self.x + cT*p_arm[0] - sT*p_arm[1]
        py_w = self.y + sT*p_arm[0] + cT*p_arm[1]
        pz_w = p_arm[2]
        return np.array([px_w, py_w, pz_w])

    def getDOF(self):
        return self.DOF

    def getJointPos(self, idx):
        """
        Joint ordering for task classes: [0,1]=base (not used as joints),
        [2]=q1, [3]=q2, [4]=q3.
        """
        if idx < 2:
            return 0.0
        return self.arm.getJointPos(idx - 2)


# ===========================================================================
#  Task classes  –  REUSED from lab4_robotics.py (logic unchanged)
#  Only the update() methods are adapted for SwiftProManipulator /
#  MobileManipulator instead of the generic Manipulator.
# ===========================================================================

class Task:
    """Abstract base class – identical to lab4_robotics.Task."""
    def __init__(self, name, desired):
        self.name    = name
        self.sigma_d = desired
        self.ff_vel  = np.zeros_like(desired)
        self.K       = np.eye(desired.shape[0])

    def update(self, robot):  pass
    def setDesired(self, v):  self.sigma_d = v
    def getDesired(self):     return self.sigma_d
    def getJacobian(self):    return self.J
    def getError(self):       return self.err
    def setFF(self, v):       self.ff_vel = v
    def getFF(self):          return self.ff_vel
    def setGain(self, K):     self.K = K
    def getGain(self):        return self.K
    def isActive(self):       return True


class Position3D(Task):
    """
    3-D end-effector position task.
    Replaces Position2D – the uArm operates in 3-D.
    """
    def __init__(self, name, desired):
        super().__init__(name, desired)   # desired: (3,1)
        self.J   = np.zeros((3, 3))
        self.err = np.zeros((3, 1))

    def update(self, robot):
        p        = robot.getEEPosition().reshape(3, 1)
        self.err = self.sigma_d - p
        self.J   = robot.getEEJacobian()   # 3×3 or 3×5 depending on robot type


class Position2D_XY(Task):
    """
    2-D end-effector position task (X-Y plane only).
    Useful for planar navigation / pick-and-place.
    """
    def __init__(self, name, desired):
        super().__init__(name, desired)   # desired: (2,1)
        self.J   = np.zeros((2, 3))
        self.err = np.zeros((2, 1))

    def update(self, robot):
        p        = robot.getEEPosition()
        self.err = self.sigma_d - p[:2].reshape(2, 1)
        J_full   = robot.getEEJacobian()
        self.J   = J_full[:2, :]   # top 2 rows


class HeightTask(Task):
    """
    1-D height (Z) control task – useful for pick / place approach.
    """
    def __init__(self, name, desired_z):
        super().__init__(name, np.array([[desired_z]]))
        self.J   = np.zeros((1, 3))
        self.err = np.zeros((1, 1))

    def update(self, robot):
        pz       = robot.getEEPosition()[2]
        self.err = self.sigma_d - np.array([[pz]])
        J_full   = robot.getEEJacobian()
        self.J   = J_full[2:3, :]   # bottom row


class YawTask(Task):
    """
    End-effector yaw (orientation around vertical axis) task.
    Direct equivalent of Orientation2D from lab4_robotics.py.

    Why yaw only?
    -------------
    The uArm Swift Pro's parallelogram linkage enforces:
        pitch = 0,  roll = 0   (EE always stays level)
    The only freely controllable orientation angle is the yaw, which equals
    the base joint angle q1.  Therefore:

        σ   = q1                         (scalar, radians)
        dσ  = dq1
        J_yaw = [1,  0,  0]  (1×3)      ∂q1/∂[q1,q2,q3]

    This is exactly analogous to Orientation2D, which used the Z-rotation
    column of the geometric Jacobian.  Here we derive it directly from the
    geometry: q1 IS the yaw, so the Jacobian row is trivially [1, 0, 0].

    Usage
    -----
        task = YawTask('EE yaw', desired_yaw_rad)
        task.setGain(np.array([[2.0]]))
    """
    def __init__(self, name, desired_yaw):
        # desired_yaw: scalar (float) or (1,1) array
        super().__init__(name, np.array([[float(desired_yaw)]]))
        self.J   = np.zeros((1, 3))
        self.err = np.zeros((1, 1))

    def update(self, robot):
        # Current yaw = q1 (the base rotation joint)
        current_yaw = robot.getJointPos(0)   # q1

        raw_err = float(self.sigma_d[0, 0]) - current_yaw
        # Wrap error to [−π, π] to avoid spinning the long way around
        raw_err = (raw_err + np.pi) % (2 * np.pi) - np.pi

        self.err = np.array([[raw_err]])

        # Jacobian: d(yaw)/d[q1, q2, q3] = [1, 0, 0]
        # This is derived purely geometrically: yaw = q1, so ∂yaw/∂q1 = 1
        # and yaw is independent of q2 and q3.
        n = robot.getDOF()
        self.J = np.zeros((1, n))
        self.J[0, 0] = 1.0


class Configuration3D(Task):
    """
    Combined end-effector position AND yaw task.
    Direct equivalent of Configuration2D from lab4_robotics.py.

    σ = [px, py, pz, yaw]ᵀ  (4×1)

    Jacobian (4×3):
        J = [ J_position  ]   (3×3 geometric position Jacobian)
            [ J_yaw       ]   (1×3)  = [1, 0, 0]

    This is exactly the 3-D analogue of Configuration2D which stacked
    the position Jacobian (2×N) with the orientation row (1×N).

    Why only yaw?
    -------------
    Pitch and roll are permanently zero (parallelogram constraint), so
    they cannot be controlled and should not appear as task variables.

    Usage
    -----
        desired = np.array([[0.20], [0.05], [-0.15], [0.3]])  # [x,y,z,yaw]
        task = Configuration3D('EE config', desired)
        task.setGain(np.diag([2., 2., 2., 1.5]))
    """
    def __init__(self, name, desired):
        # desired: (4,1) – [px, py, pz, yaw]
        assert desired.shape == (4, 1), \
            "Configuration3D desired must be shape (4,1): [px,py,pz,yaw]"
        super().__init__(name, desired)
        self.J   = np.zeros((4, 3))
        self.err = np.zeros((4, 1))

    def update(self, robot):
        # Current position
        p = robot.getEEPosition().reshape(3, 1)

        # Current yaw = q1
        current_yaw = robot.getJointPos(0)

        # Position error
        pos_err = self.sigma_d[:3] - p          # (3,1)

        # Yaw error (wrapped)
        raw_yaw_err = float(self.sigma_d[3, 0]) - current_yaw
        raw_yaw_err = (raw_yaw_err + np.pi) % (2 * np.pi) - np.pi
        yaw_err = np.array([[raw_yaw_err]])      # (1,1)

        self.err = np.vstack([pos_err, yaw_err]) # (4,1)

        # Jacobian: stack position Jacobian and yaw row
        J_pos = robot.getEEJacobian()            # (3,3) or (3,5)
        n = robot.getDOF()
        J_yaw = np.zeros((1, n))
        J_yaw[0, 0] = 1.0                        # d(yaw)/d(q1) = 1

        self.J = np.vstack([J_pos, J_yaw])       # (4,3) or (4,5)


class JointPosition(Task):
    """Joint position task – identical to lab4_robotics.JointPosition."""
    def __init__(self, name, desired, joint_idx):
        super().__init__(name, desired)
        self.joint_idx = joint_idx
        self.J   = None
        self.err = None

    def update(self, robot):
        current  = robot.getJointPos(self.joint_idx)
        self.err = self.sigma_d - np.array([[current]])
        self.J   = np.zeros((1, robot.getDOF()))
        self.J[0, self.joint_idx] = 1.0


class JointLimits(Task):
    """
    Joint limits inequality task – identical to lab4_robotics.JointLimits.
    Works with SwiftProManipulator or MobileManipulator.
    """
    def __init__(self, name, joint_idx, q_min, q_max, margin=0.05):
        super().__init__(name, np.zeros((1, 1)))
        self.joint_idx   = joint_idx
        self.q_min       = q_min
        self.q_max       = q_max
        self.margin      = margin
        self.active      = False
        self.active_state = 0
        self.current_q   = 0.0
        self.J   = np.zeros((1, 1))
        self.err = np.zeros((1, 1))

    def update(self, robot):
        self.current_q = float(robot.getJointPos(self.joint_idx))
        dof = robot.getDOF()

        if not self.active:
            if self.current_q >= (self.q_max - self.margin):
                self.active_state = -1
                self.active       = True
            elif self.current_q <= (self.q_min + self.margin):
                self.active_state =  1
                self.active       = True
        else:
            delta = self.margin * 1.5
            if self.q_min + delta < self.current_q < self.q_max - delta:
                self.active       = False
                self.active_state = 0

        if self.active:
            self.err           = np.array([[self.active_state]])
            self.J             = np.zeros((1, dof))
            self.J[0, self.joint_idx] = 1.0
        else:
            self.err = np.array([[0.0]])
            self.J   = np.zeros((1, dof))

    def isActive(self):       return self.active
    def getCurrentPosition(self): return self.current_q


class Obstacle3D(Task):
    """
    3-D spherical obstacle avoidance task (inequality).
    Adapted from lab4_robotics.Obstacle2D to work in 3-D.
    """
    def __init__(self, name, obstacle_pos, r_activate, r_deactivate):
        super().__init__(name, np.zeros((1, 1)))
        self.obs_pos      = np.array(obstacle_pos, dtype=float).reshape(3,)
        self.r_a          = r_activate
        self.r_d          = r_deactivate
        self.active       = False
        self.distance     = np.inf
        self.J   = np.zeros((1, 1))
        self.err = np.zeros((1, 1))

    def update(self, robot):
        ee_pos        = robot.getEEPosition()
        disp          = ee_pos - self.obs_pos
        self.distance = np.linalg.norm(disp)

        if not self.active:
            if self.distance <= self.r_a:
                self.active = True
        else:
            if self.distance >= self.r_d:
                self.active = False

        if self.active:
            self.err = np.array([[self.r_a - self.distance]])
            if self.distance > 1e-6:
                direction = disp / self.distance
                J_ee      = robot.getEEJacobian()    # 3×N
                self.J    = direction.reshape(1, 3) @ J_ee
            else:
                self.J = np.zeros((1, robot.getDOF()))
        else:
            self.err = np.array([[0.0]])
            self.J   = np.zeros((1, robot.getDOF()))

    def isActive(self):   return self.active
    def getDistance(self): return self.distance


# ===========================================================================
#  Recursive Task-Priority solver  –  identical algorithm to all lab files
# ===========================================================================

def task_priority_step(tasks, robot, damping=0.1):
    """
    Run one step of the recursive Task-Priority algorithm.

    This is the SAME algorithm used across lab3, lab4, lab5 – extracted here
    as a reusable function so every ROS node can call it without copy-pasting.

    Arguments
    ---------
    tasks   : list[Task]   – ordered highest priority → lowest priority
    robot   : SwiftProManipulator | MobileManipulator
    damping : float        – DLS damping factor λ

    Returns
    -------
    zeta : np.ndarray, shape (DOF, 1) – joint velocity command
    """
    n    = robot.getDOF()
    P    = np.eye(n)
    zeta = np.zeros((n, 1))

    for task in tasks:
        task.update(robot)

        # Skip inactive inequality tasks
        if not task.isActive():
            continue

        err_i = task.getError()

        # Skip already-satisfied inequality tasks
        if not isinstance(task, (JointLimits,)) and np.all(err_i <= 0):
            pass   # equality tasks always proceed

        Ji     = task.getJacobian()
        Ki     = task.getGain()
        ff_i   = task.getFF()
        xi_dot = Ki @ err_i + ff_i

        Ji_bar     = Ji @ P
        Ji_bar_dls = DLS(Ji_bar, damping)

        zeta = zeta + Ji_bar_dls @ (xi_dot - Ji @ zeta)

        Ji_bar_pinv = np.linalg.pinv(Ji_bar)
        P = P - Ji_bar_pinv @ Ji_bar

    return zeta


def scale_velocities(zeta, max_vel):
    """
    Scale the joint velocity vector so that the maximum absolute velocity
    does not exceed max_vel.  Same as in all lab files.
    """
    s = np.max(np.abs(zeta)) / max_vel
    if s > 1.0:
        zeta = zeta / s
    return zeta
