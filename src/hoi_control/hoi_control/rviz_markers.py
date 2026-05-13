"""
rviz_markers.py
RVizMarkersMixin — builds and publishes the RViz MarkerArray each control tick.

Expected attributes on `self`:
    All the FSM and path-tracking attributes used to determine the current
    target (see fsm_handlers.py docstring), plus:
    self._pub_markers          : Publisher
    self._link1_world          : np.ndarray (3,)
    self._arm_q                : np.ndarray (4,)

Requires (from other mixins):
    self._tf_buffer            (TFHelperMixin's dependency, used here for FK)
"""

import math

import cv2
import numpy as np

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from hoi_control.swiftpro_robotics_rrc import swiftpro_fk_vms_5dof

from bezier import bezier, bezier_cubic
from config import (
    WORLD_FRAME,
    ID_TARGET, ID_TF_EE, ID_FK_EE, ID_J1_BASE, ID_ERR_LINE, ID_TEXT,
    ID_BOX_TOP, ID_ALIGN_TGT, ID_PATH_LINE, ID_PATH_WP,
    EE_TOUCH_Z_OFFSET, PICK_ASCEND_Z_EXTRA, NAV_EE_Z,
    GOAL_BASE_X, GOAL_BASE_Y, FLOOR_PLACE_FORWARD,
    PLACE_APPROACH_Z_ABOVE,
    ALIGN_DIST_NAV_TOL,
)
from fsm_states import State


class RVizMarkersMixin:

    def _publish_markers(self):
        """
        Publish RViz MarkerArray each control tick:
          RED     sphere — current VMS/arm target
          GREEN/YELLOW sphere — TF EE (yellow=far, green=close)
          CYAN    sphere — VMS FK estimate of EE
          YELLOW  sphere — J1 arm base
          WHITE   line   — error vector EE → target
          WHITE   text   — FSM state + error magnitude
          MAGENTA sphere — detected box top (world_enu)
        """
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        def mk(mid, mtype):
            m = Marker()
            m.header.frame_id    = WORLD_FRAME
            m.header.stamp       = now
            m.ns                 = 'pick_place_vms'
            m.id                 = mid
            m.type               = mtype
            m.action             = Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        def sphere(mid, pos, r, g, b, size=0.025):
            m = mk(mid, Marker.SPHERE)
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.scale.x = m.scale.y = m.scale.z = size
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            return m

        # ── Determine current target from FSM state ───────────────────────────
        target = None

        if self._state == State.SEARCH:
            target = None                                   # no target yet

        elif self._state == State.ALIGN_DIST and self._align_target_world is not None:
            target = np.array([self._align_target_world[0],
                               self._align_target_world[1], 0.0])

        elif self._state == State.ALIGN_ANGLE:
            target = self._box_top_world   # show box location while aligning

        elif self._state == State.APPROACH_BOX_VMS:
            target = self._approach_target   # above-box weighted VMS target

        elif self._state == State.PICK_ASCEND and self._approach_target is not None:
            target = self._approach_target + np.array([0., 0., PICK_ASCEND_Z_EXTRA])

        elif self._state in (State.PICK_DESCEND, State.SUCTION_ON) \
                and self._box_top_world is not None:
            bx, by, bz = self._box_top_world
            target = np.array([bx, by, bz + EE_TOUCH_Z_OFFSET])

        elif self._state == State.NAVIGATE_TO_GOAL:
            nav_z = (self._approach_target[2] + PICK_ASCEND_Z_EXTRA
                     if self._approach_target is not None else NAV_EE_Z)
            target = np.array([GOAL_BASE_X, GOAL_BASE_Y + FLOOR_PLACE_FORWARD, nav_z])

        elif self._state in (State.PLACE_VMS_APPROACH, State.PLACE_ASCEND) \
                and self._place_floor_target is not None:
            target = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])

        elif self._state in (State.PLACE_DESCEND, State.SUCTION_OFF) \
                and self._place_floor_target is not None:
            target = self._place_floor_target

        # (old robot-back states removed from enum — see commented block in State class)

        # RED sphere — current target
        if target is not None:
            ma.markers.append(sphere(ID_TARGET, target, 1.0, 0.0, 0.0, 0.030))

        # GREEN/YELLOW sphere — TF EE (colour encodes error proximity)
        ee = self._last_tf_ee
        if ee is not None:
            err_norm = float(np.linalg.norm(target - ee)) if target is not None else 0.0
            closeness = float(np.clip(1.0 - err_norm / 0.10, 0.0, 1.0))
            m = mk(ID_TF_EE, Marker.SPHERE)
            m.pose.position.x = float(ee[0])
            m.pose.position.y = float(ee[1])
            m.pose.position.z = float(ee[2])
            m.scale.x = m.scale.y = m.scale.z = 0.022
            m.color.r = 1.0 - closeness
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            ma.markers.append(m)

        # CYAN sphere — VMS FK estimate
        fk_ee = swiftpro_fk_vms_5dof(
            self._link1_world, self._arm_q, self._base_psi, self._tf_buffer)
        if fk_ee is not None:
            ma.markers.append(sphere(ID_FK_EE, fk_ee, 0.0, 0.8, 1.0, 0.015))

        # YELLOW sphere — J1 arm base
        ma.markers.append(sphere(ID_J1_BASE,
                                  np.array([self._base_x, self._base_y, 0.0]),
                                  1.0, 1.0, 0.0, 0.018))

        # WHITE line — error vector EE → target
        if ee is not None and target is not None:
            ml = mk(ID_ERR_LINE, Marker.LINE_STRIP)
            ml.scale.x = 0.004
            ml.color.r = ml.color.g = ml.color.b = 1.0
            ml.color.a = 0.9
            p1 = Point(); p1.x = float(ee[0]);     p1.y = float(ee[1]);     p1.z = float(ee[2])
            p2 = Point(); p2.x = float(target[0]); p2.y = float(target[1]); p2.z = float(target[2])
            ml.points = [p1, p2]
            ma.markers.append(ml)

        # WHITE text — FSM state + relevant error metric
        txt_pos = target if target is not None else np.array([self._base_x, self._base_y, 0.5])
        mt = mk(ID_TEXT, Marker.TEXT_VIEW_FACING)
        mt.pose.position.x = float(txt_pos[0])
        mt.pose.position.y = float(txt_pos[1])
        mt.pose.position.z = float(txt_pos[2]) + 0.07
        mt.scale.z = 0.025
        mt.color.r = mt.color.g = mt.color.b = mt.color.a = 1.0
        if self._state == State.ALIGN_DIST and self._align_target_world is not None:
            d_nav = math.hypot(self._align_target_world[0] - self._base_x,
                               self._align_target_world[1] - self._base_y)
            detail = f'  nav_dist={d_nav:.3f}m / tol={ALIGN_DIST_NAV_TOL:.2f}m'
        elif self._state == State.ALIGN_ANGLE and self._marker_rvec is not None:
            tv_t     = self._marker_tvec.flatten()
            R_t, _   = cv2.Rodrigues(self._marker_rvec)
            mz_t     = R_t @ np.array([0.0, 0.0, 1.0])
            face_t   = math.atan2(float(mz_t[0]), -float(mz_t[2]))
            centre_t = math.atan2(float(tv_t[0]), float(tv_t[2]))
            detail   = (f'  face={math.degrees(face_t):+.1f}deg'
                        f'  ctr={math.degrees(centre_t):+.1f}deg')
        elif ee is not None and target is not None:
            detail = f'  err={np.linalg.norm(target - ee):.3f}m'
        else:
            detail = ''
        mt.text = f'[{self._state.name}]{detail}'
        ma.markers.append(mt)

        # MAGENTA sphere — detected box top
        if self._box_top_world is not None:
            ma.markers.append(sphere(ID_BOX_TOP, self._box_top_world,
                                      1.0, 0.0, 1.0, 0.028))

        # CYAN sphere — ALIGN_DIST stand-off target (world XY, shown at ground level)
        if self._align_target_world is not None:
            at = np.array([self._align_target_world[0],
                           self._align_target_world[1], 0.05])
            ma.markers.append(sphere(ID_ALIGN_TGT, at, 0.0, 1.0, 1.0, 0.035))

        # ORANGE line — path visualisation (VMS states only)
        if self._path_start is not None and target is not None:
            ml = mk(ID_PATH_LINE, Marker.LINE_STRIP)
            ml.scale.x = 0.006
            ml.color.r = 1.0; ml.color.g = 0.5; ml.color.b = 0.0; ml.color.a = 0.8

            if self._path_control_pt is not None and self._path_control_pt2 is not None:
                # Cubic Bézier (_vms_step)
                P0, P1, P2, P3 = (self._path_start, self._path_control_pt,
                                   self._path_control_pt2, target)
                for i in range(31):
                    t  = i / 30.0
                    pt = bezier_cubic(t, P0, P1, P2, P3)
                    p  = Point(); p.x = float(pt[0]); p.y = float(pt[1]); p.z = float(pt[2])
                    ml.points.append(p)
            elif self._path_control_pt is not None:
                # Quadratic Bézier (_arm_bezier_step)
                P0, P1, P2 = self._path_start, self._path_control_pt, target
                for i in range(21):
                    t  = i / 20.0
                    pt = bezier(t, P0, P1, P2)
                    p  = Point(); p.x = float(pt[0]); p.y = float(pt[1]); p.z = float(pt[2])
                    ml.points.append(p)
            else:
                # Straight-line (_vms_nav_step / _arm_step) — two endpoints only
                ps = Point()
                ps.x = float(self._path_start[0])
                ps.y = float(self._path_start[1])
                ps.z = float(self._path_start[2])
                pe = Point()
                pe.x = float(target[0])
                pe.y = float(target[1])
                pe.z = float(target[2])
                ml.points = [ps, pe]

            ma.markers.append(ml)

        # ORANGE sphere — current interpolated waypoint sliding along the path
        if self._path_desired is not None:
            ma.markers.append(sphere(ID_PATH_WP, self._path_desired,
                                      1.0, 0.5, 0.0, 0.018))

        self._pub_markers.publish(ma)