"""
fsm_states.py
Finite state machine states for the pick-and-place node.
"""

from enum import Enum, auto


class State(Enum):
    SEARCH             = auto()
    ALIGN_DIST         = auto()   # drive base to stand-off point in front of marker
    ALIGN_ANGLE        = auto()   # rotate in place until face-on to marker
    APPROACH_BOX_VMS   = auto()   # VMS: EE above box (pick approach)
    PICK_DESCEND       = auto()   # arm-only: descend to box top
    SUCTION_ON         = auto()   # activate suction
    PICK_ASCEND        = auto()   # arm-only: lift EE back up
    NAVIGATE_TO_GOAL   = auto()   # VMS: drive to destination carrying box
    PLACE_VMS_APPROACH = auto()   # VMS: EE above floor drop point at destination
    PLACE_DESCEND      = auto()   # arm-only: lower box to floor
    SUCTION_OFF        = auto()   # deactivate suction (release box)
    PLACE_ASCEND       = auto()   # arm-only: lift EE away from floor
    DONE               = auto()
    # ── old states — kept for reference, not active in current FSM ───────────
    # PLACE_APPROACH    = auto()   # old: approach robot back (unreachable)
    # PLACE_DROP        = auto()   # old: lower onto robot back
    # SUCTION_OFF_1     = auto()
    # RETREAT_FROM_DROP = auto()
    # UNLOAD_APPROACH   = auto()
    # UNLOAD_GRAB       = auto()
    # SUCTION_ON_2      = auto()
    # UNLOAD_ASCEND     = auto()
    # FLOOR_APPROACH    = auto()
    # FLOOR_DROP        = auto()
    # SUCTION_OFF_FINAL = auto()
