"""
tf_helpers.py
TFHelperMixin — TF-based state lookups and VMS state refresh.

Expected attributes on `self`:
    self._tf_buffer        : tf2_ros.Buffer
    self._vms_state        : VMSRobotState
    self._last_tf_ee       : np.ndarray | None
    self._link1_world      : np.ndarray
    self._base_x, self._base_y, self._base_psi : float
    self._arm_q            : np.ndarray (4,)

Provides:
    self._tf_ready()
    self._tf_pos(frame)
    self._tf_pose(frame)
    self._camera_to_world(tvec, extra_z)
    self._update_vms_state()
"""

import math
import numpy as np
import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PointStamped
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from config import (
    WORLD_FRAME, EE_FRAME, J1_FRAME, BASE_FRAME, CAMERA_FRAME,
)


class TFHelperMixin:

    def _tf_ready(self) -> bool:
        """Returns True once all required TF frames are available."""
        required = [
            (WORLD_FRAME, BASE_FRAME),
            (WORLD_FRAME, EE_FRAME),
            (WORLD_FRAME, J1_FRAME),
            (WORLD_FRAME, CAMERA_FRAME),
        ]
        for parent, child in required:
            try:
                if not self._tf_buffer.can_transform(parent, child, rclpy.time.Time()):
                    self.get_logger().info(
                        f'Waiting for TF: {child} → {parent}',
                        throttle_duration_sec=2.0)
                    return False
            except Exception:
                self.get_logger().info(
                    f'Waiting for TF: {child} → {parent}',
                    throttle_duration_sec=2.0)
                return False
        return True

    def _tf_pos(self, frame: str):
        """Look up frame origin in world_enu. Returns np.array(3) or None."""
        try:
            t  = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _tf_pose(self, frame: str):
        """Look up (x, y, yaw) of frame in world_enu. Returns tuple or None."""
        try:
            t   = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, rclpy.time.Time())
            tr  = t.transform.translation
            rot = t.transform.rotation
            siny = 2.0 * (rot.w * rot.z + rot.x * rot.y)
            cosy = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
            yaw  = math.atan2(siny, cosy)
            return float(tr.x), float(tr.y), float(yaw)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _camera_to_world(self, tvec: np.ndarray, extra_z: float = 0.0):
        """
        Transform a position from the camera optical frame to world_enu via TF.
        extra_z is added to the z component BEFORE the transform
        (useful to shift from marker centre to box top in the camera frame).
        Note: 'extra_z' is added in the camera frame's z direction (optical axis).
        After transform, the resulting world_enu point is returned.
        """
        try:
            pt = PointStamped()
            pt.header.frame_id = CAMERA_FRAME
            pt.header.stamp    = rclpy.time.Time().to_msg()   # time=0 → use latest available
            tv = tvec.flatten()
            pt.point.x = float(tv[0])
            pt.point.y = float(tv[1])
            pt.point.z = float(tv[2])
            # Transform marker centre to world_enu
            pt_world = self._tf_buffer.transform(
                pt, WORLD_FRAME,
                timeout=Duration(seconds=0.05))
            pos = np.array([pt_world.point.x,
                            pt_world.point.y,
                            pt_world.point.z])
            # Shift vertically in world_enu by extra_z
            # (MARKER_TO_BOX_TOP_Z compensates for marker-not-on-top)
            pos[2] += extra_z
            return pos
        except Exception as e:
            self.get_logger().warn(
                f'TF cam→world failed: {e}', throttle_duration_sec=1.0)
            return None

    def _update_vms_state(self):
        """Refresh all TF-based state fields and VMSRobotState."""
        ee = self._tf_pos(EE_FRAME)
        if ee is not None:
            self._last_tf_ee = ee

        j1 = self._tf_pos(J1_FRAME)
        if j1 is not None:
            self._link1_world = j1

        pose = self._tf_pose(BASE_FRAME)
        if pose is not None:
            self._base_x, self._base_y, self._base_psi = pose

        ee_for_jac = self._last_tf_ee if self._last_tf_ee is not None else np.zeros(3)
        self._vms_state.update(
            ee_for_jac,
            [self._base_x, self._base_y],
            self._base_psi,
            self._arm_q,
            self._tf_buffer,
        )
