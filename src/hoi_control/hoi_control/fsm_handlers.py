"""
fsm_handlers.py
FSMHandlersMixin — all `_run_<state>` methods plus the two target-computation
helpers (`_compute_align_target`, `_compute_floor_target`).

Expected attributes on `self` (set up by PickPlaceVMSNode.__init__):
    self._state, self._state_entry_time
    self._box_top_world, self._box_locked, self._approach_target
    self._place_floor_target
    self._marker_rvec, self._marker_tvec, self._marker_detect_time
    self._align_target_world, self._align_search_psi, self._align_search_idx,
        self._align_search_hold
    self._search_initial_psi, self._search_idx, self._search_hold_start,
        self._search_advancing, self._search_advance_start
    self._base_x, self._base_y, self._base_psi
    self._path_start

Requires (from other mixins / main node):
    self._send_base, self._send_arm, self._call_suction
    self._transition, self._elapsed, self._at_position
    self._vms_nav_step, self._arm_step, self._task_priority_weighted, etc.
    self._camera_to_world  (TFHelperMixin)
    self.get_logger()      (Node)
"""

import math
import time

import cv2
import numpy as np

from config import (
    EE_TOUCH_FORWARD_OFFSET, EE_TOUCH_Z_OFFSET, APPROACH_HEIGHT_ABOVE,
    PICK_ASCEND_Z_EXTRA, NAV_EE_Z,
    GOAL_BASE_X, GOAL_BASE_Y, FLOOR_PLACE_FORWARD,
    PLACE_APPROACH_Z_ABOVE, PLACE_TOUCH_Z_OFFSET, BOX_HEIGHT,
    EE_REACH_TOL, SUCTION_SETTLE_S,
    W_BASE_COST, W_ARM_COST,
    SEARCH_SEQUENCE, SEARCH_SEQUENCE_ALIGN, SEARCH_OMEGA, SEARCH_HOLD_S,
    SEARCH_FWD_DIST, SEARCH_FWD_VEL,
    ALIGN_TARGET_DIST, ALIGN_DIST_NAV_TOL,
    ALIGN_DIST_K_HEAD, ALIGN_DIST_K_FWD,
    ALIGN_DIST_MAX_VX, ALIGN_DIST_MAX_OMEGA, ALIGN_HEAD_TOL,
    ALIGN_ANGLE_TOL, ALIGN_K_ANGLE, ALIGN_K_CENTER, ALIGN_MAX_OMEGA,
)
from fsm_states import State
from math_utils import angle_wrap


class FSMHandlersMixin:

    # ── FSM: SEARCH ───────────────────────────────────────────────────────────
    def _run_search(self):
        # Check for marker on every tick — even during forward advance
        if self._marker_rvec is not None and self._marker_tvec is not None:
            target = self._compute_align_target()
            if target is not None:
                self._align_target_world = target
                self.get_logger().info(
                    f'Marker detected → stand-off target {np.round(target, 3)} → ALIGN_DIST')
                self._send_base(0.0, 0.0)
                self._transition(State.ALIGN_DIST)
                return

        # ── Forward advance phase (after a full sweep) ────────────────────────
        if self._search_advancing:
            if self._search_advance_start is None:
                self._search_advance_start = time.time()
                self.get_logger().info(
                    f'Search: advancing forward {SEARCH_FWD_DIST:.2f} m')

            elapsed      = time.time() - self._search_advance_start
            advance_time = SEARCH_FWD_DIST / SEARCH_FWD_VEL

            if elapsed < advance_time:
                self._send_base(SEARCH_FWD_VEL, 0.0)
            else:
                # Advance complete — start a fresh sweep from current heading
                self._search_advancing     = False
                self._search_advance_start = None
                self._search_initial_psi   = self._base_psi
                self._search_idx           = 0
                self.get_logger().info('Search: advance done, starting next sweep')
            return

        # ── Rotation sweep phase ──────────────────────────────────────────────
        if self._search_initial_psi is None:
            self._search_initial_psi = self._base_psi
            self.get_logger().info(
                f'Search sweep started, yaw = {math.degrees(self._base_psi):.1f}°')

        target_psi = (self._search_initial_psi
                      + SEARCH_SEQUENCE[self._search_idx % len(SEARCH_SEQUENCE)])
        psi_err    = angle_wrap(target_psi - self._base_psi)

        if abs(psi_err) < 0.05:
            # At target heading — hold and check
            if self._search_hold_start is None:
                self._search_hold_start = time.time()
            self._send_base(0.0, 0.0)

            if time.time() - self._search_hold_start >= SEARCH_HOLD_S:
                self._search_idx       += 1
                self._search_hold_start = None
                if self._search_idx >= len(SEARCH_SEQUENCE):
                    # Full sweep done, nothing found — move forward then sweep again
                    self._search_idx      = 0
                    self._search_advancing = True
        else:
            # Rotate toward target heading
            omega = float(np.clip(SEARCH_OMEGA * np.sign(psi_err),
                                  -SEARCH_OMEGA, SEARCH_OMEGA))
            self._search_hold_start = None
            self._send_base(0.0, omega)

    # ── FSM: ALIGN_DIST ──────────────────────────────────────────────────────
    def _run_align_dist(self):
        """
        Drive the robot base to the stand-off target computed in SEARCH.
        No angular correction — just navigate to the XY point using a
        heading-first differential-drive P-controller.
        Transitions to ALIGN_ANGLE once within ALIGN_DIST_NAV_TOL.
        """
        if self._align_target_world is None:
            self._transition(State.ALIGN_ANGLE)
            return

        tx, ty = self._align_target_world[0], self._align_target_world[1]
        dx     = tx - self._base_x
        dy     = ty - self._base_y
        dist   = math.hypot(dx, dy)

        if dist < ALIGN_DIST_NAV_TOL:
            self.get_logger().info(
                f'At stand-off point (dist={dist:.3f}m) → ALIGN_ANGLE')
            self._send_base(0.0, 0.0)
            self._transition(State.ALIGN_ANGLE)
            return

        heading_to_target = math.atan2(dy, dx)
        heading_err       = angle_wrap(heading_to_target - self._base_psi)

        omega = float(np.clip(ALIGN_DIST_K_HEAD * heading_err,
                              -ALIGN_DIST_MAX_OMEGA, ALIGN_DIST_MAX_OMEGA))
        vx    = 0.0
        if abs(heading_err) < ALIGN_HEAD_TOL:
            vx = float(np.clip(ALIGN_DIST_K_FWD * dist,
                               0.0, ALIGN_DIST_MAX_VX))

        self._send_base(vx, omega)
        self._send_arm(np.zeros(4))

    # ── FSM: ALIGN_ANGLE ─────────────────────────────────────────────────────
    def _run_align_angle(self):
        """
        Rotate in place to align the camera face-on to the marker.

        On state entry (first tick): clears stale rvec/tvec so only fresh
        camera detections drive the controller — avoids reacting to data from
        before the robot reached the stand-off point.

        While marker not visible: sweeps ±SEARCH_SWEEP_ANGLE to find it.
        While marker visible:     P-controller on center_angle + face_angle.
        Done when both within ALIGN_ANGLE_TOL.
        """
        # ── State entry: flush stale PnP data ────────────────────────────────
        if self._align_search_psi is None:
            self._marker_rvec        = None
            self._marker_tvec        = None
            self._marker_detect_time = None
            self._align_search_psi   = self._base_psi   # latch entry heading
            self._align_search_idx   = 0
            self.get_logger().info(
                f'ALIGN_ANGLE entered, flushed stale PnP, '
                f'psi={math.degrees(self._base_psi):.1f}°')

        if self._marker_rvec is not None:
            # ── Marker visible — align ────────────────────────────────────────
            tv = self._marker_tvec.flatten()
            R, _         = cv2.Rodrigues(self._marker_rvec)
            mz           = R @ np.array([0.0, 0.0, 1.0])
            center_angle = math.atan2(float(tv[0]), float(tv[2]))
            face_angle   = math.atan2(float(mz[0]), -float(mz[2]))

            self.get_logger().info(
                f'ALIGN_ANGLE: face={math.degrees(face_angle):+.1f}° '
                f'ctr={math.degrees(center_angle):+.1f}° '
                f'tol={math.degrees(ALIGN_ANGLE_TOL):.1f}°',
                throttle_duration_sec=0.5)

            if (abs(face_angle) < ALIGN_ANGLE_TOL and
                    abs(center_angle) < ALIGN_ANGLE_TOL):
                self.get_logger().info(
                    f'Face-on (face={math.degrees(face_angle):+.1f}° '
                    f'ctr={math.degrees(center_angle):+.1f}°) → APPROACH_BOX_VMS')
                self._send_base(0.0, 0.0)
                self._transition(State.APPROACH_BOX_VMS)
                return

            omega = float(np.clip(
                -ALIGN_K_CENTER * center_angle - ALIGN_K_ANGLE * face_angle,
                -ALIGN_MAX_OMEGA, ALIGN_MAX_OMEGA))
            self._send_base(0.0, omega)
            self._send_arm(np.zeros(4))
            # Reset sweep state so a loss-and-reacquire starts a fresh sweep
            self._align_search_idx  = 0
            self._align_search_hold = None
        else:
            # ── Marker not visible — sweep ±SEARCH_SWEEP_ANGLE to find it ────
            target_psi = (self._align_search_psi
                          + SEARCH_SEQUENCE_ALIGN[self._align_search_idx % len(SEARCH_SEQUENCE_ALIGN)])
            psi_err    = angle_wrap(target_psi - self._base_psi)

            if abs(psi_err) < 0.05:
                if self._align_search_hold is None:
                    self._align_search_hold = time.time()
                self._send_base(0.0, 0.0)
                if time.time() - self._align_search_hold >= SEARCH_HOLD_S:
                    self._align_search_idx  += 1
                    self._align_search_hold  = None
            else:
                omega = float(np.clip(SEARCH_OMEGA * np.sign(psi_err),
                                      -SEARCH_OMEGA, SEARCH_OMEGA))
                self._align_search_hold = None
                self._send_base(0.0, omega)

            self._send_arm(np.zeros(4))

    # ── FSM: APPROACH_BOX_VMS ────────────────────────────────────────────────
    def _run_approach_box_vms(self):
        """
        Single-target weighted VMS: drive EE directly above the box at
        approach height.  Base DOFs are very expensive so the arm handles
        almost all the motion; the base only nudges if the arm truly cannot
        reach alone.  Tune W_BASE_COST to control how much the base moves.
        """
        if self._approach_target is None:
            bx, by, bz = self._box_top_world
            fwd = np.array([math.cos(self._base_psi),
                            math.sin(self._base_psi), 0.0])
            fwd_offset = EE_TOUCH_FORWARD_OFFSET * fwd
            self._approach_target = np.array([bx, by, bz + APPROACH_HEIGHT_ABOVE]) + fwd_offset
            self._box_locked = True
            self.get_logger().info(
                f'Approach target (above box, psi={math.degrees(self._base_psi):.1f}°): '
                f'{np.round(self._approach_target, 3)}')

        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(self._approach_target, weight_matrix=W)

        if self._at_position(self._approach_target, EE_REACH_TOL):
            self.get_logger().info('Approach reached → PICK_DESCEND')
            self._transition(State.PICK_DESCEND)

    # ── FSM: PICK_DESCEND ────────────────────────────────────────────────────
    def _run_pick_descend(self):
        bx, by, bz  = self._box_top_world
        fwd = np.array([math.cos(self._base_psi),
                            math.sin(self._base_psi), 0.0])
        fwd_offset = EE_TOUCH_FORWARD_OFFSET * fwd
        pick_target = np.array([bx, by, bz + EE_TOUCH_Z_OFFSET]) + fwd_offset

        # # Arm-only descent (base frozen)
        # self._arm_step(pick_target)

        # Weighted VMS descent — same logic as APPROACH_BOX_VMS, high base cost
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(pick_target, weight_matrix=W)

        if self._at_position(pick_target, EE_REACH_TOL):
            self.get_logger().info('Pick position reached → SUCTION_ON')
            self._transition(State.SUCTION_ON)

    # ── FSM: SUCTION_ON ──────────────────────────────────────────────────────
    def _run_suction_on(self):
        if self._state_entry_time is None:
            self._call_suction(True)
            self.get_logger().info('Suction ON — waiting for settle')
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        if self._elapsed() >= SUCTION_SETTLE_S:
            self.get_logger().info('Suction settled → PICK_ASCEND')
            self._transition(State.PICK_ASCEND)
            # self.get_logger().info('Suction settled → DONE (iterative stop)')
            # self._transition(State.DONE)

    # ── FSM: PICK_ASCEND ─────────────────────────────────────────────────────
    def _run_pick_ascend(self):
        """Lift EE to approach position + PICK_ASCEND_Z_EXTRA — weighted VMS."""
        ascend_target = self._approach_target + np.array([0., 0., PICK_ASCEND_Z_EXTRA])
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(ascend_target, weight_matrix=W)
        if self._at_position(ascend_target, EE_REACH_TOL):
            self.get_logger().info('Ascended → NAVIGATE_TO_GOAL')
            self._transition(State.NAVIGATE_TO_GOAL)

    # ── FSM: NAVIGATE_TO_GOAL ────────────────────────────────────────────────
    def _run_navigate_to_goal(self):
        """
        VMS: drive robot to goal position carrying the box.
        Z is kept the same as the end of PICK_ASCEND (approach_z + PICK_ASCEND_Z_EXTRA)
        so the box stays at a safe height throughout navigation.
        """
        # Clear any stale floor target so PLACE_VMS_APPROACH latches a fresh one.
        if self._path_start is None:
            self._place_floor_target = None

        nav_z = (self._approach_target[2] + PICK_ASCEND_Z_EXTRA
                 if self._approach_target is not None else NAV_EE_Z)
        # EE target is FLOOR_PLACE_FORWARD ahead of GOAL_BASE_Y so the arm's
        # natural forward reach forces the base to actually reach GOAL_BASE_Y.
        # Without this offset the arm extends to cover the gap and the base
        # parks ~0.3m short, making PLACE_VMS_APPROACH target behind the EE.
        vms_goal = np.array([GOAL_BASE_X,
                             GOAL_BASE_Y + FLOOR_PLACE_FORWARD,
                             nav_z])
        self._vms_nav_step(vms_goal)

        if self._at_position(vms_goal, EE_REACH_TOL):
            self.get_logger().info(
                f'Goal reached (EE at goal) → PLACE_VMS_APPROACH')
            self._transition(State.PLACE_VMS_APPROACH)

    # ── FSM: PLACE_VMS_APPROACH ──────────────────────────────────────────────
    def _run_place_vms_approach(self):
        """
        VMS: drive EE to approach height above the floor drop point.
        _place_floor_target is latched once (fixed world_enu position) so the
        base navigating does not shift the target.
        Orange path line visible via _vms_nav_step.
        """
        if self._place_floor_target is None:
            self._place_floor_target = self._compute_floor_target()
            self.get_logger().info(
                f'PLACE_VMS_APPROACH target latched: '
                f'{np.round(self._place_floor_target, 3)}')

        approach = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(approach, weight_matrix=W)

        if self._at_position(approach, EE_REACH_TOL):
            self.get_logger().info(
                f'Above floor drop point (approach={np.round(approach, 3)}) → PLACE_DESCEND')
            self._transition(State.PLACE_DESCEND)

    # ── FSM: PLACE_DESCEND ───────────────────────────────────────────────────
    def _run_place_descend(self):
        """
        Arm-only: descend to PLACE_DESCEND target (BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET).
        Target Z encodes how high the EE is when releasing the box — increase
        PLACE_TOUCH_Z_OFFSET if box is pressed into ground.
        """
        target = self._place_floor_target   # Z = BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET
        if target is None:
            return
        # self._arm_step(target)
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(target, weight_matrix=W)
        if self._at_position(target, EE_REACH_TOL):
            self.get_logger().info(
                f'Box at floor level (target={np.round(target, 3)}) → SUCTION_OFF')
            self._transition(State.SUCTION_OFF)

    # ── FSM: SUCTION_OFF ─────────────────────────────────────────────────────
    def _run_suction_off(self):
        """Deactivate suction to release the box, then lift away."""
        if self._state_entry_time is None:
            self._call_suction(False)
            self.get_logger().info('Suction OFF — waiting for release')
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        if self._elapsed() >= SUCTION_SETTLE_S:
            self.get_logger().info('Box released → PLACE_ASCEND')
            self._transition(State.PLACE_ASCEND)

    # ── FSM: PLACE_ASCEND ────────────────────────────────────────────────────
    def _run_place_ascend(self):
        """
        Arm-only: lift EE back to approach height above floor drop point.
        Mirror of PICK_ASCEND. Orange path line visible via _vms_nav_step.
        """
        if self._place_floor_target is None:
            self._transition(State.DONE)
            return
        approach = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(approach, weight_matrix=W)
        if self._at_position(approach, EE_REACH_TOL):
            self.get_logger().info('Lifted from floor → DONE')
            self._transition(State.DONE)

    # ── FSM: DONE ────────────────────────────────────────────────────────────
    def _run_done(self):
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))

    # ── Target-computation helpers ────────────────────────────────────────────
    def _compute_align_target(self):
        """
        Compute world-frame XY stand-off point directly in front of the marker face.
        Point = marker_world_pos + ALIGN_TARGET_DIST * marker_Z_in_world
        (marker Z points toward the camera, so we follow it back from the marker).
        Returns np.array([x, y]) or None on TF failure.
        """
        if self._marker_rvec is None or self._marker_tvec is None:
            return None
        R, _    = cv2.Rodrigues(self._marker_rvec)
        mz_cam  = R @ np.array([0.0, 0.0, 1.0])   # marker Z in camera frame
        # Stand-off point in camera frame: move along marker Z (toward camera)
        tv      = self._marker_tvec.flatten()
        standoff_cam = tv + ALIGN_TARGET_DIST * mz_cam
        world   = self._camera_to_world(standoff_cam, 0.0)
        if world is None:
            return None
        return world[:2]   # only XY — base navigates on the ground plane

    def _compute_floor_target(self):
        """
        Fixed world-frame EE target for placing the box on the floor.

        XY is defined as (GOAL_BASE_X, GOAL_BASE_Y + FLOOR_PLACE_FORWARD) —
        the same world point the NAVIGATE_TO_GOAL EE target uses — so
        PLACE_VMS_APPROACH never moves the robot backward regardless of
        the heading the robot arrives with.
        """
        return np.array([GOAL_BASE_X,
                         GOAL_BASE_Y + FLOOR_PLACE_FORWARD,
                         BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET])
