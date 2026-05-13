"""
logging_utils.py
LoggingMixin — every-1-second terminal diagnostic log.

Expected attributes on `self`:
    self._log_tick, self._log_entries
    All FSM / target-tracking attributes used to summarise current state.

Requires (from other mixins):
    self._joint_limit_tasks  (controllers infrastructure)
"""

import math
import numpy as np

from config import (
    EE_TOUCH_Z_OFFSET, PICK_ASCEND_Z_EXTRA, NAV_EE_Z,
    GOAL_BASE_X, GOAL_BASE_Y, FLOOR_PLACE_FORWARD,
    PLACE_APPROACH_Z_ABOVE, PLACE_TOUCH_Z_OFFSET, BOX_HEIGHT,
    VMS_PATH_PERIOD,
)
from fsm_states import State


class LoggingMixin:

    # ── Terminal log — printed every 20 ticks (≈1 s at 20 Hz) ───────────────
    def _terminal_log(self):
        if self._log_tick % 20 != 0:
            return

        q   = self._arm_q
        ee  = self._last_tf_ee
        pd  = self._path_desired

        # Current target from publish_markers logic (reuse the same logic)
        tgt = None
        if self._state == State.ALIGN_DIST and self._align_target_world is not None:
            tgt = np.array([self._align_target_world[0],
                            self._align_target_world[1], 0.0])
        elif self._state == State.ALIGN_ANGLE and self._box_top_world is not None:
            tgt = self._box_top_world
        elif self._state == State.APPROACH_BOX_VMS:
            tgt = self._approach_target
        elif self._state in (State.PICK_DESCEND, State.SUCTION_ON) \
                and self._box_top_world is not None:
            bx, by, bz = self._box_top_world
            tgt = np.array([bx, by, bz + EE_TOUCH_Z_OFFSET])
        elif self._state == State.PICK_ASCEND and self._approach_target is not None:
            tgt = self._approach_target + np.array([0., 0., PICK_ASCEND_Z_EXTRA])
        # (State.PLACE_APPROACH removed — was unreachable old state)
        elif self._state == State.NAVIGATE_TO_GOAL:
            nav_z = (self._approach_target[2] + PICK_ASCEND_Z_EXTRA
                     if self._approach_target is not None else NAV_EE_Z)
            tgt = np.array([GOAL_BASE_X, GOAL_BASE_Y + FLOOR_PLACE_FORWARD, nav_z])
        elif self._state in (State.PLACE_VMS_APPROACH, State.PLACE_ASCEND) \
                and self._place_floor_target is not None:
            tgt = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        elif self._state in (State.PLACE_DESCEND, State.SUCTION_OFF) \
                and self._place_floor_target is not None:
            tgt = self._place_floor_target  # Z = BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET

        err  = float(np.linalg.norm(tgt - ee)) if (tgt is not None and ee is not None) else float('nan')
        ee_s = f'[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]' if ee is not None else 'N/A'
        tgt_z_note = ''
        if self._state in (State.PLACE_VMS_APPROACH, State.PLACE_ASCEND) \
                and self._place_floor_target is not None:
            tgt_z_note = f'  (floor+{PLACE_APPROACH_Z_ABOVE:.3f}m approach)'
        elif self._state == State.PLACE_DESCEND and self._place_floor_target is not None:
            tgt_z_note = f'  (BOX_HEIGHT={BOX_HEIGHT:.3f} + PLACE_TOUCH_Z_OFFSET={PLACE_TOUCH_Z_OFFSET:.3f})'
        tgt_s = (f'[{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]{tgt_z_note}'
                 if tgt is not None else 'N/A')
        wp_s  = f'[{pd[0]:.3f}, {pd[1]:.3f}, {pd[2]:.3f}]' if pd is not None else 'N/A'
        box_s = (f'[{self._box_top_world[0]:.3f}, {self._box_top_world[1]:.3f}, '
                 f'{self._box_top_world[2]:.3f}]') if self._box_top_world is not None else 'N/A'

        lim_active = [t.name for t in self._joint_limit_tasks if t.isActive()]

        if self._path_start_t is not None:
            elapsed_s = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
            alpha = float(np.clip(elapsed_s / VMS_PATH_PERIOD, 0.0, 1.0))
        else:
            alpha = float('nan')

        log = (
            f'\n{"─"*68}\n'
            f'  FSM state   : {self._state.name}\n'
            f'  BASE        : x={self._base_x:.3f}  y={self._base_y:.3f}  '
            f'psi={math.degrees(self._base_psi):.1f} deg\n'
            f'  ARM q[1-4]  : [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] rad\n'
            f'  EE (world)  : {ee_s}\n'
            f'  Target      : {tgt_s}\n'
            f'  Waypoint    : {wp_s}  (alpha={alpha:.3f})\n'
            f'  err_to_tgt  : {err:.4f} m\n'
            f'  box_top     : {box_s}  locked={self._box_locked}\n'
            f'  active_lims : {lim_active if lim_active else "none"}\n'
            f'{"─"*68}'
        )
        self.get_logger().info(log)
