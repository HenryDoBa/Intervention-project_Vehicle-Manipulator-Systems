#!/usr/bin/env python3
"""
lab2_pick_place_vms_node.py
Autonomous ArUco box pick-and-place using VMS (vehicle-manipulator system)
and arm-only resolved-rate FK control.

FSM sequence
------------
SEARCH               → rotate base, scan for ArUco marker
APPROACH_BOX_VMS     → VMS: drive EE to approach height above box
PICK_DESCEND         → arm-only: lower EE to box top (suction contact)
SUCTION_ON           → activate suction cup, wait SUCTION_SETTLE_S
PICK_ASCEND          → arm-only: lift EE back to approach height
PLACE_APPROACH       → arm-only: EE to approach height above drop_point (robot back)
PLACE_DROP           → arm-only: lower EE so box bottom touches drop surface
SUCTION_OFF_1        → deactivate suction, wait SUCTION_SETTLE_S
RETREAT_FROM_DROP    → arm-only: lift EE away from drop_point
NAVIGATE_TO_GOAL     → VMS: drive robot to goal position
UNLOAD_APPROACH      → arm-only: EE to approach height above drop_point (grab box)
UNLOAD_GRAB          → arm-only: lower to grab box from robot back
SUCTION_ON_2         → activate suction, wait SUCTION_SETTLE_S
UNLOAD_ASCEND        → arm-only: lift box from robot back
FLOOR_APPROACH       → arm-only: move EE to approach height above floor target
FLOOR_DROP           → arm-only: lower EE so box touches floor
SUCTION_OFF_FINAL    → deactivate suction, wait SUCTION_SETTLE_S
DONE                 → stop

All geometry offsets are tunable constants — see config.py.
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener

from hoi_control.swiftpro_robotics_rrc import (
    Q1_MIN, Q1_MAX,
    Q2_MIN, Q2_MAX,
    Q3_MIN, Q3_MAX,
    Q4_MIN, Q4_MAX,
    JOINT_NAMES_4DOF,
    VMSRobotState,
    VMSPositionTask,
    VMSJointLimitsTask,
)

from config import (
    JOINT_STATE_TOPIC, JOINT_CMD_TOPIC, BASE_CMD_TOPIC,
    CAMERA_TOPIC, SUCTION_SRV, MARKER_TOPIC,
    CONTROL_HZ, DT,
    ARUCO_DICT_ID, ARUCO_MARKER_SIZE,
    VMS_K, LIMIT_MARGIN, LIMIT_HYSTERESIS_RATIO,
    LOG_PATH,
)
from fsm_states import State

from aruco_detector import ArucoDetectorMixin
from controllers import ControllersMixin
from fsm_handlers import FSMHandlersMixin
from logging_utils import LoggingMixin
from rviz_markers import RVizMarkersMixin
from tf_helpers import TFHelperMixin


class PickPlaceVMSNode(
    FSMHandlersMixin,
    ControllersMixin,
    RVizMarkersMixin,
    LoggingMixin,
    ArucoDetectorMixin,
    TFHelperMixin,
    Node,
):
    """
    Top-level ROS 2 node. All behaviour comes from the mixins above;
    this class is responsible only for:
      - declaring the shared mutable state (in __init__)
      - wiring ROS pubs/subs/timers
      - the trivial helpers (_send_base, _send_arm, _call_suction,
        _transition, _elapsed, _at_position, _js_cb, _control_loop)
    """

    def __init__(self):
        super().__init__('pick_place_vms_node')

        # ── state machine ────────────────────────────────────────────────────
        self._state            = State.SEARCH
        self._state_entry_time = None   # wall-clock time of last transition

        # ── detected box target (world_enu) ──────────────────────────────────
        self._box_top_world = None   # [x, y, z] of detected box top in world_enu
        self._box_locked    = False  # True once we commit to a pick position

        # ── diagnostic log (saved to file on Ctrl+C) ──────────────────────────
        self._log_entries   = []
        self._log_tick      = 0     # incremented each control tick

        # ── cached robot state ────────────────────────────────────────────────
        self._arm_q            = np.zeros(4)
        self._base_x           = 0.0
        self._base_y           = 0.0
        self._base_psi         = 0.0
        self._last_tf_ee       = None
        self._link1_world      = np.zeros(3)

        # ── search / align sub-state ─────────────────────────────────────────
        self._search_initial_psi   = None
        self._search_idx           = 0
        self._search_hold_start    = None
        self._search_advancing     = False   # True while driving forward between sweeps
        self._search_advance_start = None    # wall time when forward advance began
        self._marker_cx_px         = None    # x pixel of target marker centre (overlay only)
        self._marker_rvec          = None    # rvec from latest solvePnP
        self._marker_tvec          = None    # tvec from latest solvePnP
        self._marker_detect_time   = None    # wall time of last solvePnP success
        self._align_target_world   = None    # world XY stand-off point (ALIGN_DIST target)
        self._align_search_psi     = None    # heading reference for ALIGN_ANGLE sweep
        self._align_search_idx     = 0       # current step in sweep sequence
        self._align_search_hold    = None    # hold-start time at each sweep waypoint

        # ── approach cached targets ───────────────────────────────────────────
        self._approach_target     = None  # standoff position above box
        self._place_floor_target  = None  # latched in PLACE_VMS_APPROACH (fixed world pos)

        # ── VMS path-tracking state ──────────────────────────────────────────
        # Reset on every state transition; latched on the first VMS tick.
        self._path_start       = None   # EE position at path start (np.array(3))
        self._path_start_t     = None   # rclpy.time.Time when path began
        self._path_desired     = None   # current Bézier waypoint (for visualisation)
        self._path_control_pt  = None   # cubic Bézier P1 (for visualisation)
        self._path_control_pt2 = None   # cubic Bézier P2 (for visualisation)

        # ── VMS infrastructure ────────────────────────────────────────────────
        self._vms_state = VMSRobotState()

        self._pos_task = VMSPositionTask('pos', np.zeros(3))
        self._pos_task.setGain(np.eye(3) * VMS_K)

        self._joint_limit_tasks = [
            VMSJointLimitsTask('q1_lim', 2, Q1_MIN, Q1_MAX,
                               margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q2_lim', 3, Q2_MIN, Q2_MAX,
                               margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q3_lim', 4, Q3_MIN, Q3_MAX,
                               margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q4_lim', 5, Q4_MIN, Q4_MAX,
                               margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
        ]

        # ── ArUco detector ────────────────────────────────────────────────────
        aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        aruco_params = cv2.aruco.DetectorParameters()
        #self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self._aruco_dict = aruco_dict
        self._aruco_params = aruco_params

        # Marker corner object points (marker centred at origin, in marker plane)
        h = ARUCO_MARKER_SIZE / 2.0
        self._marker_obj_pts = np.array([
            [-h,  h, 0.0],
            [ h,  h, 0.0],
            [ h, -h, 0.0],
            [-h, -h, 0.0],
        ], dtype=np.float64)

        # ── camera visualisation ──────────────────────────────────────────────
        self._bridge        = CvBridge()
        self._vis_frame     = None   # latest annotated frame (updated in camera cb)
        cv2.namedWindow('ArUco Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ArUco Detection', 960, 540)

        # ── ROS I/O ───────────────────────────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._sub_js  = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self._sub_img = self.create_subscription(
            Image, CAMERA_TOPIC, self._camera_cb, 5)

        self._pub_arm     = self.create_publisher(
            Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self._pub_base    = self.create_publisher(Twist, BASE_CMD_TOPIC, 10)
        self._pub_markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        self._suction_cli = self.create_client(SetBool, SUCTION_SRV)

        # ── timers ────────────────────────────────────────────────────────────
        self._ctrl_timer = self.create_timer(DT, self._control_loop)
        self._vis_timer  = self.create_timer(1.0 / 15.0, self._vis_tick)

        self.get_logger().info('PickPlaceVMSNode started — State: SEARCH')

    # ── Joint-state callback ──────────────────────────────────────────────────
    def _js_cb(self, msg):
        pos_map = dict(zip(msg.name, msg.position))
        for i, jn in enumerate(JOINT_NAMES_4DOF):
            if jn in pos_map:
                self._arm_q[i] = pos_map[jn]

    # ── Main control loop (CONTROL_HZ) ────────────────────────────────────────
    def _control_loop(self):
        self._log_tick += 1

        if not self._tf_ready():
            return

        # Refresh TF-based robot state
        self._update_vms_state()

        dispatch = {
            State.SEARCH:             self._run_search,
            State.ALIGN_DIST:         self._run_align_dist,
            State.ALIGN_ANGLE:        self._run_align_angle,
            State.APPROACH_BOX_VMS:   self._run_approach_box_vms,
            State.PICK_DESCEND:       self._run_pick_descend,
            State.SUCTION_ON:         self._run_suction_on,
            State.PICK_ASCEND:        self._run_pick_ascend,
            State.NAVIGATE_TO_GOAL:   self._run_navigate_to_goal,
            State.PLACE_VMS_APPROACH: self._run_place_vms_approach,
            State.PLACE_DESCEND:      self._run_place_descend,
            State.SUCTION_OFF:        self._run_suction_off,
            State.PLACE_ASCEND:       self._run_place_ascend,
            State.DONE:               self._run_done,
        }
        dispatch[self._state]()
        self._publish_markers()
        self._terminal_log()

    # ── Small helpers used by every mixin ─────────────────────────────────────
    def _at_position(self, target_world: np.ndarray, tol: float) -> bool:
        ee = self._last_tf_ee
        if ee is None:
            return False
        return float(np.linalg.norm(target_world - ee)) < tol

    def _send_base(self, vx: float, omega: float):
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.angular.z = float(omega)
        self._pub_base.publish(msg)

    def _send_arm(self, dq: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in dq[:4]]
        self._pub_arm.publish(msg)

    def _call_suction(self, enable: bool):
        if not self._suction_cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Suction service not available', throttle_duration_sec=2.0)
            return
        req = SetBool.Request()
        req.data = enable
        self._suction_cli.call_async(req)

    def _transition(self, new_state: State):
        self.get_logger().info(f'{self._state.name} → {new_state.name}')
        self._state            = new_state
        self._state_entry_time = None   # set on first tick of new state
        # Reset Bézier path so the next VMS state latches a fresh start
        self._path_start       = None
        self._path_start_t     = None
        self._path_desired     = None
        self._path_control_pt  = None
        self._path_control_pt2 = None
        # NOTE: _place_floor_target is intentionally NOT reset here.
        # It is latched in PLACE_VMS_APPROACH and must persist through
        # PLACE_DESCEND → SUCTION_OFF → PLACE_ASCEND.

    def _elapsed(self) -> float:
        """Seconds since state entry (sets entry time on first call)."""
        if self._state_entry_time is None:
            self._state_entry_time = time.time()
        return time.time() - self._state_entry_time


# ── Entry point ──────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceVMSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save diagnostic log FIRST before any cleanup that might fail
        try:
            with open(LOG_PATH, 'w') as f:
                f.write('\n'.join(node._log_entries))
            print(f'\nLog saved → {LOG_PATH}  ({len(node._log_entries)} entries)')
        except Exception as e:
            print(f'Failed to save log: {e}')
        try:
            node._send_base(0.0, 0.0)
            node._send_arm(np.zeros(4))
        except Exception:
            pass
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()