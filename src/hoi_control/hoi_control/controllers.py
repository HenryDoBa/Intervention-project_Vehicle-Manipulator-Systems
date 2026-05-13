"""
controllers.py
ControllersMixin — VMS and arm-only control steps used by the FSM handlers.

Expected attributes on `self`:
    self._last_tf_ee        : np.ndarray | None
    self._vms_state         : VMSRobotState
    self._pos_task          : VMSPositionTask
    self._joint_limit_tasks : list[VMSJointLimitsTask]
    self._path_start        : np.ndarray | None
    self._path_start_t      : rclpy.time.Time | None
    self._path_desired      : np.ndarray | None
    self._path_control_pt   : np.ndarray | None
    self._path_control_pt2  : np.ndarray | None

Requires (from main node class):
    self._send_base(vx, omega)
    self._send_arm(dq)
    self.get_clock()                (Node)

Provides:
    self._vms_step(...)             cubic-Bézier VMS path
    self._task_priority_weighted(W) weighted DLS in the position-task layer
    self._arm_step(...)             arm-only linear path with feedforward
    self._arm_bezier_step(...)      arm-only quadratic-Bézier path
    self._arm_step_direct(...)      arm-only direct target (no path tracking)
    self._vms_nav_step(...)         straight-line VMS path
"""

import numpy as np

from hoi_control.swiftpro_robotics_rrc import (
    DLS, weighted_DLS, vms_task_priority_step,
)

from bezier import bezier, bezier_vel, bezier_cubic, bezier_cubic_vel
from config import (
    VMS_K, VMS_DAMPING, VMS_PATH_PERIOD, PLACE_APPROACH_PATH_PERIOD,
    VMS_CURVE_HEIGHT, VMS_CURVE_LATERAL, VMS_CURVE_P2_PULLBACK,
    ARM_MAX_VEL, BASE_MAX_LINEAR, BASE_MAX_ANGULAR,
)


class ControllersMixin:

    # ── VMS step — cubic Bézier path with feedforward ────────────────────────
    def _vms_step(self, target_world: np.ndarray, weight_matrix=None):
        """
        Drive EE toward target_world using full VMS (base + arm) along a
        cubic Bézier path with feedforward velocity.

        Cubic Bézier has two control points (P1, P2) which together force the
        robot to rotate through a wide arc before arriving at the target:

          P0  — path start (current EE)
          P1  — near P0, pushed sideways by VMS_CURVE_LATERAL + lifted
                by VMS_CURVE_HEIGHT  →  robot initially swings outward
          P2  — near P3 (goal), also pushed sideways by VMS_CURVE_LATERAL
                →  robot comes in from the side, having rotated through the arc
          P3  — approach target

        VMS_CURVE_LATERAL (m): perpendicular offset; increase for wider arc /
            more robot rotation.  Positive = left of path, negative = right.
        VMS_CURVE_HEIGHT (m): upward bulge at apex to keep arm clear.
        Set both to 0.0 for a straight-line path.
        """
        ee = self._last_tf_ee

        # ── Latch path start and control points on first call ─────────────────
        if self._path_start is None:
            self._path_start   = (ee.copy() if ee is not None
                                  else target_world.copy())
            self._path_start_t = self.get_clock().now()

            P0, P3 = self._path_start, target_world

            # Perpendicular unit vector in the XY plane
            path_xy  = P3[:2] - P0[:2]
            path_len = float(np.linalg.norm(path_xy))
            if path_len > 0.01:
                perp_xy = np.array([-path_xy[1], path_xy[0]]) / path_len
            else:
                perp_xy = np.zeros(2)

            lat  = VMS_CURVE_LATERAL
            lift = VMS_CURVE_HEIGHT

            # P1: anchored near P0, pulled sideways — robot swings out at start
            self._path_control_pt  = P0 + np.array([
                lat * perp_xy[0],
                lat * perp_xy[1],
                lift,
            ])

            # P2: placed behind P3 along the straight P0→P3 direction with no
            # lateral offset — robot arrives smoothly in-line, no sideways curve at end.
            # VMS_CURVE_P2_PULLBACK controls how far back P2 sits; increase it for a
            # longer straight run-in, decrease for a tighter final approach.
            if path_len > 0.01:
                fwd_xy = path_xy / path_len
            else:
                fwd_xy = np.zeros(2)
            self._path_control_pt2 = P3 - np.array([
                fwd_xy[0] * VMS_CURVE_P2_PULLBACK,
                fwd_xy[1] * VMS_CURVE_P2_PULLBACK,
                0.0,
            ])

        P0 = self._path_start
        P1 = self._path_control_pt
        P2 = self._path_control_pt2
        P3 = target_world

        # ── Bézier parameter α ∈ [0, 1] ──────────────────────────────────────
        elapsed = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
        alpha   = float(np.clip(elapsed / VMS_PATH_PERIOD, 0.0, 1.0))

        path_desired       = bezier_cubic(alpha, P0, P1, P2, P3)
        self._path_desired = path_desired

        # ── Feedforward: cubic Bézier tangent / T ────────────────────────────
        if alpha < 1.0:
            ff_vel = bezier_cubic_vel(alpha, P0, P1, P2, P3) / VMS_PATH_PERIOD
        else:
            ff_vel = np.zeros(3)

        # ── Update task ───────────────────────────────────────────────────────
        self._pos_task.setDesired(path_desired.reshape(3, 1))
        self._pos_task.setGain(np.eye(3) * VMS_K)
        self._pos_task.setFF(ff_vel.reshape(3, 1))

        if weight_matrix is not None:
            # Weighted task priority: joint limit tasks use standard DLS (high
            # priority), position task uses weighted DLS (prefers arm over base).
            zeta = self._task_priority_weighted(weight_matrix)
        else:
            tasks = self._joint_limit_tasks + [self._pos_task]
            zeta  = vms_task_priority_step(
                tasks, self._vms_state, damping=VMS_DAMPING, method=2)
        zeta  = zeta.flatten()

        vx    = float(np.clip(zeta[0], -BASE_MAX_LINEAR,  BASE_MAX_LINEAR))
        omega = float(np.clip(zeta[1], -BASE_MAX_ANGULAR, BASE_MAX_ANGULAR))
        dq    = np.clip(zeta[2:6], -ARM_MAX_VEL, ARM_MAX_VEL)

        self._send_base(vx, omega)
        self._send_arm(dq)

    # ── Weighted task-priority loop ───────────────────────────────────────────
    def _task_priority_weighted(self, W: np.ndarray) -> np.ndarray:
        """
        Task priority step where:
          - Joint limit tasks  → standard DLS   (highest priority)
          - Position task      → weighted DLS(W) (lower priority, in null-space)

        W is a 6×6 positive-definite diagonal weight matrix.
        Higher W_ii → DOF i is more expensive → optimizer avoids it.
        """
        n    = self._vms_state.getDOF()
        P    = np.eye(n)
        zeta = np.zeros((n, 1))

        # High-priority joint limit tasks — standard DLS
        for task in self._joint_limit_tasks:
            task.update(self._vms_state)
            if not task.isActive():
                continue
            Ji     = task.getJacobian()
            xi_dot = task.getGain() @ task.getError() + task.getFF()
            Ji_bar = Ji @ P
            Ji_bar_inv = DLS(Ji_bar, VMS_DAMPING)
            zeta   = zeta + Ji_bar_inv @ (xi_dot - Ji @ zeta)
            P      = P - np.linalg.pinv(Ji_bar) @ Ji_bar

        # Position task — weighted DLS (prefers arm joints over base)
        self._pos_task.update(self._vms_state)
        Ji     = self._pos_task.getJacobian()
        xi_dot = self._pos_task.getGain() @ self._pos_task.getError() + self._pos_task.getFF()
        Ji_bar = Ji @ P
        Ji_bar_inv = weighted_DLS(Ji_bar, VMS_DAMPING, W)
        zeta   = zeta + Ji_bar_inv @ (xi_dot - Ji @ zeta)

        return zeta

    # ── Arm-only step — sliding target + feedforward + visualization ─────────
    def _arm_step(self, target_world: np.ndarray):
        """
        Drive EE toward target_world with the base frozen.

        Uses the same sliding-target + feedforward pattern as _vms_nav_step:
          path_desired = start + alpha*(goal-start)   alpha ∈ [0,1]
          ff_vel       = (goal-start) / T             zeroed when alpha=1

        This gives the orange path line in RViz and prevents the controller
        from trying to jump to the goal in one step.  Base outputs are zeroed
        after task-priority; joint limit tasks still protect all arm joints.
        """
        ee = self._last_tf_ee
        if ee is None:
            self._send_base(0.0, 0.0)
            self._send_arm(np.zeros(4))
            return

        # Latch path start on first call
        if self._path_start is None:
            self._path_start   = ee.copy()
            self._path_start_t = self.get_clock().now()

        # Sliding waypoint with feedforward
        elapsed      = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
        alpha        = float(np.clip(elapsed / VMS_PATH_PERIOD, 0.0, 1.0))
        path_desired = self._path_start + alpha * (target_world - self._path_start)
        self._path_desired = path_desired   # stored for orange sphere in RViz

        ff_vel = ((target_world - self._path_start) / VMS_PATH_PERIOD
                  if alpha < 1.0 else np.zeros(3))

        self._pos_task.setDesired(path_desired.reshape(3, 1))
        self._pos_task.setGain(np.eye(3) * VMS_K)
        self._pos_task.setFF(ff_vel.reshape(3, 1))

        tasks = self._joint_limit_tasks + [self._pos_task]
        zeta  = vms_task_priority_step(
            tasks, self._vms_state, damping=VMS_DAMPING, method=2).flatten()

        # Zero base — only arm joint velocities sent
        dq = np.clip(zeta[2:6], -ARM_MAX_VEL, ARM_MAX_VEL)
        self._send_base(0.0, 0.0)
        self._send_arm(dq)

    # ── Arm-only cubic Bézier step — curved trajectory, base frozen ──────────
    def _arm_bezier_step(self, target_world: np.ndarray):
        """
        Follow a cubic Bézier curved path to target_world using arm joints only.
        Same control point geometry as _vms_step (VMS_CURVE_LATERAL for outward
        swing, VMS_CURVE_HEIGHT for lift) but base velocity is zeroed — only
        arm FK/Jacobian drives the motion.

        Used for PLACE_APPROACH so the arm arcs outward from the pick position
        to the drop position without colliding with the robot body.
        Visualization: orange cubic curve + sliding orange sphere (automatic).
        """
        ee = self._last_tf_ee
        if ee is None:
            self._send_base(0.0, 0.0)
            self._send_arm(np.zeros(4))
            return

        # Latch path start and control points on first call
        if self._path_start is None:
            self._path_start   = ee.copy()
            self._path_start_t = self.get_clock().now()

            P0, P3 = self._path_start, target_world
            path_xy  = P3[:2] - P0[:2]
            path_len = float(np.linalg.norm(path_xy))
            if path_len > 0.01:
                perp_xy = np.array([-path_xy[1], path_xy[0]]) / path_len
                fwd_xy  = path_xy / path_len
            else:
                perp_xy = np.zeros(2)
                fwd_xy  = np.zeros(2)

            lat  = -VMS_CURVE_LATERAL   # opposite side from the VMS approach arc
            lift = VMS_CURVE_HEIGHT

            # Single quadratic control point at path midpoint, pulled sideways + lifted
            midpoint = (P0 + P3) / 2.0
            self._path_control_pt  = midpoint + np.array([
                lat * perp_xy[0], lat * perp_xy[1], lift])
            self._path_control_pt2 = None   # quadratic — no second control point

        P0 = self._path_start
        P1 = self._path_control_pt
        P2 = target_world

        elapsed      = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
        alpha        = float(np.clip(elapsed / PLACE_APPROACH_PATH_PERIOD, 0.0, 1.0))
        path_desired = bezier(alpha, P0, P1, P2)
        self._path_desired = path_desired

        ff_vel = (bezier_vel(alpha, P0, P1, P2) / PLACE_APPROACH_PATH_PERIOD
                  if alpha < 1.0 else np.zeros(3))

        self._pos_task.setDesired(path_desired.reshape(3, 1))
        self._pos_task.setGain(np.eye(3) * VMS_K)
        self._pos_task.setFF(ff_vel.reshape(3, 1))

        tasks = self._joint_limit_tasks + [self._pos_task]
        zeta  = vms_task_priority_step(
            tasks, self._vms_state, damping=VMS_DAMPING, method=2).flatten()

        # Zero base — arm joints only
        dq = np.clip(zeta[2:6], -ARM_MAX_VEL, ARM_MAX_VEL)
        self._send_base(0.0, 0.0)
        self._send_arm(dq)

    # ── Arm-only direct step — no path tracking, arm moves freely ────────────
    def _arm_step_direct(self, target_world: np.ndarray):
        """
        Drive EE toward target_world with arm joints only — no sliding waypoint,
        no feedforward.  The task directly targets the goal each tick so the arm
        moves as freely as joint limits allow.  Base is frozen.
        """
        self._pos_task.setDesired(target_world.reshape(3, 1))
        self._pos_task.setGain(np.eye(3) * VMS_K)
        self._pos_task.setFF(np.zeros((3, 1)))

        tasks = self._joint_limit_tasks + [self._pos_task]
        zeta  = vms_task_priority_step(
            tasks, self._vms_state, damping=VMS_DAMPING, method=2).flatten()

        dq = np.clip(zeta[2:6], -ARM_MAX_VEL, ARM_MAX_VEL)
        self._send_base(0.0, 0.0)
        self._send_arm(dq)

    # ── VMS nav step — straight-line path (optional weighted DLS) ────────────
    def _vms_nav_step(self, target_world: np.ndarray, weight_matrix=None):
        """
        Drive EE toward target_world using full VMS with a straight-line
        interpolated path and feedforward velocity.

        weight_matrix : optional 6×6 diagonal np.ndarray.
            If provided, the position task uses weighted_DLS(W) so that
            expensive DOFs (e.g. base) are avoided in favour of cheap ones
            (e.g. arm joints).  None → standard DLS for all tasks.
        """
        ee = self._last_tf_ee

        if self._path_start is None:
            self._path_start   = (ee.copy() if ee is not None
                                  else target_world.copy())
            self._path_start_t = self.get_clock().now()

        elapsed      = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
        alpha        = float(np.clip(elapsed / VMS_PATH_PERIOD, 0.0, 1.0))
        path_desired = self._path_start + alpha * (target_world - self._path_start)
        self._path_desired = path_desired

        ff_vel = ((target_world - self._path_start) / VMS_PATH_PERIOD
                  if alpha < 1.0 else np.zeros(3))

        self._pos_task.setDesired(path_desired.reshape(3, 1))
        self._pos_task.setGain(np.eye(3) * VMS_K)
        self._pos_task.setFF(ff_vel.reshape(3, 1))

        if weight_matrix is not None:
            zeta = self._task_priority_weighted(weight_matrix).flatten()
        else:
            tasks = self._joint_limit_tasks + [self._pos_task]
            zeta  = vms_task_priority_step(
                tasks, self._vms_state, damping=VMS_DAMPING, method=2).flatten()

        vx    = float(np.clip(zeta[0], -BASE_MAX_LINEAR,  BASE_MAX_LINEAR))
        omega = float(np.clip(zeta[1], -BASE_MAX_ANGULAR, BASE_MAX_ANGULAR))
        dq    = np.clip(zeta[2:6], -ARM_MAX_VEL, ARM_MAX_VEL)

        self._send_base(vx, omega)
        self._send_arm(dq)
