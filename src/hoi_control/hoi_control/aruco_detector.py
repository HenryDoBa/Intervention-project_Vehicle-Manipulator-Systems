"""
aruco_detector.py
ArucoDetectorMixin — ROS camera callback, ArUco detection, PnP pose estimation
and the OpenCV HUD overlay window.

Expected attributes on `self`:
    self._bridge               : cv_bridge.CvBridge
    self._aruco_dict           : cv2.aruco.Dictionary
    self._aruco_params         : cv2.aruco.DetectorParameters
    self._marker_obj_pts       : np.ndarray (4, 3)
    self._marker_cx_px         : int | None
    self._marker_rvec          : np.ndarray | None
    self._marker_tvec          : np.ndarray | None
    self._marker_detect_time   : float | None
    self._box_top_world        : np.ndarray | None
    self._box_locked           : bool
    self._state                : State
    self._align_target_world   : np.ndarray | None
    self._base_x, self._base_y, self._base_psi : float
    self._last_tf_ee           : np.ndarray | None
    self._vis_frame            : np.ndarray | None

Requires (from other mixins):
    self._camera_to_world(...)   (TFHelperMixin)
"""

import math
import time

import cv2
import numpy as np

from config import (
    ARUCO_MARKER_ID, ARUCO_MARKER_SIZE, MARKER_TO_BOX_TOP_Z,
    ALIGN_ANGLE_TOL,
)
from camera_intrinsics import CAMERA_MATRIX, DIST_COEFFS
from fsm_states import State


class ArucoDetectorMixin:

    # ── Camera callback — ArUco detection + visualisation ─────────────────────
    def _camera_cb(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._aruco_params)

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── Draw ALL detected markers so you can see which ID is on the box ──
        target_found = False
        detected_ids = []
        if ids is not None:
            # Draw outlines + ID labels for every detected marker
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            detected_ids = ids.flatten().tolist()

            for i, mid in enumerate(detected_ids):
                mc     = corners[i]
                cx_px  = int(mc[0, :, 0].mean())
                cy_px  = int(mc[0, :, 1].mean())

                # Highlight our target ID with a different color label
                if mid == ARUCO_MARKER_ID:
                    label_color = (0, 255, 0)      # green  — this is the one we want
                    target_found = True
                else:
                    label_color = (0, 165, 255)    # orange — detected but not our target

                cv2.putText(annotated, f'ID {mid}',
                            (cx_px - 20, cy_px - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

                # Full PnP + axes only for our target marker
                if mid == ARUCO_MARKER_ID:
                    ok, rvec, tvec = cv2.solvePnP(
                        self._marker_obj_pts,
                        mc.reshape(4, 2),
                        CAMERA_MATRIX,
                        DIST_COEFFS,
                    )
                    if ok:
                        cv2.drawFrameAxes(annotated, CAMERA_MATRIX, DIST_COEFFS,
                                          rvec, tvec, ARUCO_MARKER_SIZE * 0.7)
                        dist_m = float(np.linalg.norm(tvec))
                        cv2.putText(annotated,
                                    f'd = {dist_m:.3f} m',
                                    (cx_px - 45, cy_px + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                        # Track pixel centre (overlay) and raw PnP result
                        self._marker_cx_px      = cx_px
                        self._marker_rvec       = rvec
                        self._marker_tvec       = tvec
                        self._marker_detect_time = time.time()

                        if not self._box_locked:
                            box_top = self._camera_to_world(tvec, MARKER_TO_BOX_TOP_Z)
                            if box_top is not None:
                                self._box_top_world = box_top

        # ── Alignment guide: vertical centre line + pixel error (ALIGN state) ──
        cx_img = w // 2
        cv2.line(annotated, (cx_img, 55), (cx_img, h - 5), (100, 100, 255), 1)
        if self._state in (State.ALIGN_DIST, State.ALIGN_ANGLE) \
                and self._marker_rvec is not None:
            tv_ov       = self._marker_tvec.flatten()
            R_ov, _     = cv2.Rodrigues(self._marker_rvec)
            mz_ov       = R_ov @ np.array([0.0, 0.0, 1.0])
            face_ov     = math.atan2(float(mz_ov[0]), -float(mz_ov[2]))
            centre_ov   = math.atan2(float(tv_ov[0]), float(tv_ov[2]))
            d_ov        = float(tv_ov[2])
            if self._state == State.ALIGN_DIST:
                color = (0, 165, 255)  # orange — navigating
                label = (f'ALIGN_DIST  dist_to_goal='
                         f'{math.hypot(self._align_target_world[0] - self._base_x, self._align_target_world[1] - self._base_y):.2f}m'
                         if self._align_target_world is not None else 'ALIGN_DIST')
            else:
                all_ok = (abs(face_ov) < ALIGN_ANGLE_TOL and
                          abs(centre_ov) < ALIGN_ANGLE_TOL)
                color  = (0, 255, 0) if all_ok else (0, 165, 255)
                label  = (f'face={math.degrees(face_ov):+.1f}deg  '
                          f'ctr={math.degrees(centre_ov):+.1f}deg  '
                          f'dist={d_ov:.3f}m')
            cv2.putText(annotated, label,
                        (cx_img - 420, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── Detection status banner (top of frame) ────────────────────────────
        banner_y = 36
        if ids is None or len(detected_ids) == 0:
            # Nothing at all detected
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(annotated, 'NO MARKERS DETECTED',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        elif target_found:
            # Our marker is visible
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 140, 0), -1)
            cv2.putText(annotated,
                        f'TARGET ID {ARUCO_MARKER_ID} DETECTED  |  IDs in view: {detected_ids}',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            # Markers detected but not our target ID
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 100, 200), -1)
            cv2.putText(annotated,
                        f'Looking for ID {ARUCO_MARKER_ID}  |  Found IDs: {detected_ids}',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # ── HUD overlays (bottom-left) ────────────────────────────────────────
        y = h - 110
        cv2.putText(annotated,
                    f'FSM: {self._state.name}',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        y += 28
        if self._box_top_world is not None:
            bx, by, bz = self._box_top_world
            locked_tag = ' [LOCKED]' if self._box_locked else ' [live]'
            cv2.putText(annotated,
                        f'Box top ENU: ({bx:.2f}, {by:.2f}, {bz:.2f}){locked_tag}',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 0), 2)
        y += 26
        if self._last_tf_ee is not None:
            ex, ey, ez = self._last_tf_ee
            cv2.putText(annotated,
                        f'EE ENU:     ({ex:.2f}, {ey:.2f}, {ez:.2f})',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 2)
        y += 26
        cv2.putText(annotated,
                    f'Base: ({self._base_x:.2f}, {self._base_y:.2f})  '
                    f'psi={math.degrees(self._base_psi):.1f} deg',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        self._vis_frame = annotated

    # ── Visualisation timer ───────────────────────────────────────────────────
    def _vis_tick(self):
        if self._vis_frame is not None:
            cv2.imshow('ArUco Detection', self._vis_frame)
        cv2.waitKey(1)