"""
config.py
All tunable constants and configuration for the pick-and-place node:
topics, frames, geometry, gains, tolerances, search/align behaviour,
navigation goal, RViz marker IDs.

Camera intrinsics live in `camera_intrinsics.py` since they are derived
from image size and HFOV rather than hand-tuned.
"""

import math
import cv2

# ─── Topics / Frames ─────────────────────────────────────────────────────────
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
BASE_CMD_TOPIC    = '/turtlebot/cmd_vel'
CAMERA_TOPIC      = '/turtlebot/camera/color/image_color'
SUCTION_SRV       = '/turtlebot/swiftpro/vacuum_gripper/set_pump'
MARKER_TOPIC      = '/hoi/pick_place_markers'

WORLD_FRAME   = 'world_enu'
EE_FRAME      = 'end_effector'
J1_FRAME      = 'turtlebot/swiftpro/manipulator_base_link'
BASE_FRAME    = 'turtlebot/base_footprint'
CAMERA_FRAME  = 'camera_color_optical_frame'

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# ─── RViz marker IDs ──────────────────────────────────────────────────────────
ID_TARGET   = 0   # RED    sphere — current VMS/arm target position
ID_TF_EE    = 1   # GREEN/YELLOW sphere — true EE from TF (yellow=far, green=close)
ID_FK_EE    = 2   # CYAN   sphere — VMS FK estimate of EE
ID_J1_BASE  = 3   # YELLOW sphere — J1 arm base position
ID_ERR_LINE = 4   # WHITE  line   — error vector EE → target
ID_TEXT     = 5   # WHITE  text   — FSM state + error magnitude
ID_BOX_TOP    = 6   # MAGENTA sphere — detected box top position (world_enu)
ID_ALIGN_TGT  = 7   # CYAN   sphere — ALIGN_DIST stand-off target (world XY)
ID_PATH_LINE  = 8   # ORANGE line   — straight-line path start → VMS goal
ID_PATH_WP    = 9   # ORANGE sphere — current interpolated waypoint being tracked

# ─── Tunable geometry offsets ─────────────────────────────────────────────────
# ArUco detection
# The texture files are named aruco_original*.png → DICT_ARUCO_ORIGINAL is the
# most likely match for the Stonefish aruco_box texture.
# If detection still fails, try uncommenting a different dictionary below:
#   cv2.aruco.DICT_4X4_50
#   cv2.aruco.DICT_4X4_100
#   cv2.aruco.DICT_4X4_250
#   cv2.aruco.DICT_5X5_50
#   cv2.aruco.DICT_5X5_100
#   cv2.aruco.DICT_5X5_250
#   cv2.aruco.DICT_6X6_250
#   cv2.aruco.DICT_7X7_250
ARUCO_DICT_ID     = cv2.aruco.DICT_ARUCO_ORIGINAL
ARUCO_MARKER_ID   = 1       # which marker ID to track (tune once detection works)
ARUCO_MARKER_SIZE = 0.050   # physical marker side length (m)

# Box geometry (from aruco_box.obj: 70×70×150 mm)
BOX_HEIGHT   = 0.150    # m  total box height
BOX_HALF_W   = 0.035    # m  half-width (x and y)

# When PnP detects the marker centre, how far above it is the box top?
# Set to 0 if marker is on the TOP face.
# Set to ~BOX_HEIGHT/2 if marker is on the FRONT face (most likely).
MARKER_TO_BOX_TOP_Z = 0.075   # m  (tune: 0 = top face, 0.075 = front face mid-height)

# Approach clearance above the box top before descending to pick
APPROACH_HEIGHT_ABOVE = 0.12   # m above box top

# Extra Z added on top of the approach target during PICK_ASCEND so the arm
# lifts slightly higher before the next state takes over.
PICK_ASCEND_Z_EXTRA = 0.00     # m (tune: 0 = same as approach, positive = higher)

# How close the EE needs to be to the box top for suction contact
# (positive = EE slightly above, negative = pressed in)
EE_TOUCH_Z_OFFSET = -0.003      # m above box top surface
# Forward offset applied along the robot's heading direction (not world-X).
# Positive = shift pick point further from the robot (away from base),
# Negative = shift pick point closer to the robot.
# Components are automatically decomposed: Δx = offset·cos(ψ), Δy = offset·sin(ψ).
EE_TOUCH_FORWARD_OFFSET = 0.0   # m along robot heading (tune: range -0.05 – +0.05)
# Suction settle time (seconds) after toggling the pump
SUCTION_SETTLE_S = 1.2

# Navigation goal (world_enu coordinates the BASE should reach)
GOAL_BASE_X  = 0.0     # m  (East)
GOAL_BASE_Y  = 1.8     # m  (North, i.e. 1.8 m forward from start)
GOAL_TOL_XY  = 0.15    # m  base position tolerance to declare "at goal"
NAV_EE_Z     = 0.30    # m  EE height during VMS navigation

# Floor placement after navigation: place this far in front of the arm base
FLOOR_PLACE_FORWARD = 0.0    # m in front of arm base (tune for arm reach)

# ── Place-phase Z offsets (tune these two for floor placement) ────────────────
# PLACE_APPROACH_Z_ABOVE : clearance above the drop point for PLACE_VMS_APPROACH
#   and PLACE_ASCEND.  Reduce to shorten PLACE_DESCEND travel; increase if arm
#   needs more clearance on the way down.
# PLACE_TOUCH_Z_OFFSET   : added to BOX_HEIGHT to get the PLACE_DESCEND target Z.
#   Positive → EE releases box slightly above floor (box drops gently, no pressing).
#   Zero     → EE exactly at box top when box rests on floor.
#   Negative → EE presses box into ground (avoid).
#   Tune: start +0.03, reduce toward 0 once placement looks stable.
PLACE_APPROACH_Z_ABOVE = 0.10   # m  (typical range 0.06 – 0.15)
PLACE_TOUCH_Z_OFFSET   = -0.005   # m  (range -0.01 – +0.05)

# VMS controller gains / limits
VMS_K            = 0.5    # proportional gain for VMSPositionTask
VMS_DAMPING      = 0.08   # DLS damping λ
BASE_MAX_LINEAR  = 0.15   # m/s base linear speed cap
BASE_MAX_ANGULAR = 0.45   # rad/s base yaw-rate cap

# Bézier path tracking for VMS states.
# The desired waypoint slides along a quadratic Bézier curve from the current
# EE position (P0) to the goal (P2) over VMS_PATH_PERIOD seconds.
# The control point P1 sits at the midpoint of the straight path, pushed by:
#   VMS_CURVE_HEIGHT   — upward lift (m) at the apex of the arc
#   VMS_CURVE_LATERAL  — sideways offset (m) perpendicular to path direction;
#                        positive = left of path (CCW), negative = right (CW)
# Set both to 0.0 for the original straight-line behaviour.
VMS_PATH_PERIOD            = 30.0   # s  path period for VMS states (approach, navigation)
PLACE_APPROACH_PATH_PERIOD = 20.0   # s  path period for PLACE_APPROACH Bézier arc
#                                        longer = slower sliding waypoint + weaker feedforward
VMS_CURVE_HEIGHT     = 0.0   # m  upward lift applied to P1 (start control point)
VMS_CURVE_LATERAL    = -0.40  # m  sideways pull on P1 — controls tightness of start curve
#                              positive = left of path (CCW), negative = right (CW)
VMS_CURVE_P2_PULLBACK = 0.15  # m  how far behind the goal P2 is placed along the
#                              straight-line direction (no lateral offset on P2,
#                              so the arrival is always smooth and in-line)

ARM_MAX_VEL = 0.50   # rad/s per joint (used for arm velocity capping in vms_step)

# Weighted DLS cost factors for the swing phase of PICK_DESCEND.
# Higher W_BASE_COST → optimizer strongly prefers arm joints over base movement.
# Ratio of 10:1 means the base is 10× more "expensive" than arm joints.
W_BASE_COST = 10.0   # cost on vx and ω (indices 0, 1 in quasi-velocity space)
W_ARM_COST  = 1.0    # cost on dq1–dq4 (indices 2–5)

# Goal-reached tolerance for arm states (EE position error in metres)
EE_REACH_TOL = 0.005   # m
# EE_REACH_TOL = 0.005   # m
# XY alignment threshold for PICK_DESCEND swing phase:
# once the EE is within this horizontal distance of the box centre, descend.
XY_ALIGN_TOL = 0.030   # m

# Joint limit avoidance — matches lab2_rrc_methods_vms_node settings
LIMIT_MARGIN           = 0.10   # rad — activation threshold α
LIMIT_HYSTERESIS_RATIO = 1.5    # δ = margin × ratio; must be > 1 to avoid chatter

# Approach standoff — EE approach target is this far behind the box so the
# base parks at a safe distance and only the arm moves for the final pick.
# Roughly equal to the arm's max forward reach (~0.20 m).
APPROACH_STANDOFF  = 0.20   # m behind box (in robot heading direction)

# ── Search sweep ──────────────────────────────────────────────────────────────
SEARCH_SWEEP_ANGLE = math.radians(45)  # rad each side (tune: 30°–90°)
SEARCH_SWEEP_ANGLE_ALIGN = math.radians(90)  # rad each side (tune: 30°–90°)
SEARCH_SEQUENCE    = [0.0, SEARCH_SWEEP_ANGLE, 0.0, -SEARCH_SWEEP_ANGLE, 0.0]
SEARCH_SEQUENCE_ALIGN    = [0.0, SEARCH_SWEEP_ANGLE_ALIGN, 0.0, -SEARCH_SWEEP_ANGLE_ALIGN, 0.0]
SEARCH_OMEGA       = 0.35    # rad/s angular speed during search rotation
SEARCH_HOLD_S      = 1.0     # seconds to hold each orientation before checking
SEARCH_FWD_DIST    = 0.30    # m to advance forward after each complete sweep
SEARCH_FWD_VEL     = 0.10    # m/s during forward advance

# ── ALIGN_DIST: drive base to stand-off point computed from marker pose ───────
# Robot drives to a world-frame XY target (marker_pos + ALIGN_TARGET_DIST *
# marker_normal). No angular correction during this phase.
ALIGN_TARGET_DIST    = 0.80   # m  stand-off from marker face (tune)
ALIGN_DIST_NAV_TOL   = 0.08   # m  XY arrival tolerance
ALIGN_DIST_K_HEAD    = 1.20   # rad/s per rad  heading-to-target P-gain
ALIGN_DIST_K_FWD     = 0.40   # m/s  per m     forward speed P-gain
ALIGN_DIST_MAX_VX    = 0.15   # m/s cap
ALIGN_DIST_MAX_OMEGA = 0.40   # rad/s cap
ALIGN_HEAD_TOL       = math.radians(20)  # must face target before driving fwd

# ── ALIGN_ANGLE: rotate in place until face-on ────────────────────────────────
# From the stand-off point, sweep to re-find marker then align.
ALIGN_ANGLE_TOL    = math.radians(4.0)  # rad  face-on + centering tolerance
ALIGN_K_ANGLE      = 0.30              # rad/s per rad — face-on error (rvec)
ALIGN_K_CENTER     = 0.80              # rad/s per rad — bearing-to-centre error
ALIGN_MAX_OMEGA    = 0.35              # rad/s cap

# ── Diagnostic log path ───────────────────────────────────────────────────────
# Where the terminal-log entries are dumped on Ctrl+C in main().
LOG_PATH = '/home/huy/Desktop/Intervention/src/hoi_control/logs/place_approach_log.txt' # change 
