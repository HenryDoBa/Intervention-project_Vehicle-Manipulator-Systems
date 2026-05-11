#!/usr/bin/env python3
"""
lab2_rrc_node.py
================
Resolved-Rate Motion Control for the uArm Swift Pro.

All control math runs in arm-local ENU (x=East, y=North, z=Up relative to J1).
swiftpro_fk() and swiftpro_jacobian() now both output arm-local ENU directly.
No NED/ENU conversions needed in the control loop.

Visualization:
  GREEN sphere = true EE position from TF (world_enu)
  RED   sphere = target (world_enu)
  WHITE line   = error vector
  WHITE text   = method + error norm + FK_vs_TF diff
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

from tf2_ros import Buffer, TransformListener, LookupException, \
    ConnectivityException, ExtrapolationException

from hoi_control.swiftpro_robotics import (
    SwiftProManipulator, swiftpro_fk, swiftpro_jacobian, DLS, scale_velocities
)

# ---------------------------------------------------------------------------
# Control parameters
# ---------------------------------------------------------------------------
DT      = 1.0 / 60.0
MAX_VEL = 1.0
DAMPING = 0.05
K       = np.diag([1.0, 1.0, 1.0])

# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------
JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
MARKER_TOPIC      = '/hoi/markers'

# ---------------------------------------------------------------------------
# Target in world_enu (x=East, y=North, z=Up, metres).
# The node subtracts link1 (J1) position to get arm-local ENU for control.
# Red sphere appears here in RViz.
# ---------------------------------------------------------------------------
TARGET_WORLD_ENU = np.array([0.05, 0.15, 0.3])

# ---------------------------------------------------------------------------
# TF frames
# ---------------------------------------------------------------------------
WORLD_FRAME = 'world_enu'
EE_FRAME    = 'turtlebot/swiftpro/link9'
J1_FRAME    = 'turtlebot/swiftpro/link1'

METHODS         = ['transpose', 'pseudoinverse', 'DLS']
METHOD_DURATION = 10.0

# Marker IDs
ID_TARGET  = 0
ID_CURRENT = 1
ID_ERROR   = 2
ID_TEXT    = 3


class Lab2RRCNode(Node):

    def __init__(self):
        super().__init__('lab2_rrc_node')
        self.arm = SwiftProManipulator()

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.sub_js = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.pub_cmd     = self.create_publisher(Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self.pub_markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        self.timer = self.create_timer(DT, self._control_loop)

        self._got_js       = False
        self._t            = 0.0
        self._method_idx   = 0
        self._j1_world_enu = None   # J1 (link1) position in world_enu from TF

        self.get_logger().info(
            f'Lab2 RRC node started.\n'
            f'  Target world_enu : {TARGET_WORLD_ENU}\n'
            f'  RViz Fixed Frame : {WORLD_FRAME}\n'
            f'  Add by topic     : {MARKER_TOPIC}  (MarkerArray)\n'
            f'  FK output        : arm-local ENU (no NED conversion needed)')

    def _js_cb(self, msg: JointState):
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))
        self._got_js = True

    def _tf_pos(self, child_frame):
        """Look up position of child_frame in world_enu. Returns (3,) or None."""
        try:
            t = self._tf_buffer.lookup_transform(
                WORLD_FRAME, child_frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _control_loop(self):
        if not self._got_js:
            return
        from hoi_control.swiftpro_robotics import D1
        self.get_logger().info(f'RUNTIME D1={D1}', throttle_duration_sec=5.0)
        # ── Method cycling ────────────────────────────────────────────────
        self._t += DT
        if self._t >= METHOD_DURATION:
            self._t = 0.0
            self._method_idx = (self._method_idx + 1) % len(METHODS)
            self.get_logger().info(
                f'Switching to method: {METHODS[self._method_idx]}')
        method = METHODS[self._method_idx]

        # ── TF lookups ────────────────────────────────────────────────────
        ee_world_enu = self._tf_pos(EE_FRAME)    # for visualization
        j1_world_enu = self._tf_pos(J1_FRAME)    # arm base in world_enu
        if j1_world_enu is not None:
            self._j1_world_enu = j1_world_enu
            self.get_logger().info(f'J1_world_enu: {j1_world_enu}', throttle_duration_sec=2.0)


        # ── FK in arm-local ENU (relative to J1) — NO conversions needed ──
        ee_fk_arm_enu = swiftpro_fk(self.arm.q)   # arm-local ENU directly
        ee_rel_tf = (ee_world_enu - self._j1_world_enu) if (ee_world_enu is not None and self._j1_world_enu is not None) else None
        self.get_logger().info(
            f'DEBUG | q=[{self.arm.q[0]:.3f},{self.arm.q[1]:.3f},{self.arm.q[2]:.3f}] | '
            f'FK_arm={np.round(ee_fk_arm_enu,3)} | '
            f'TF_rel_J1={np.round(ee_rel_tf,3) if ee_rel_tf is not None else "N/A"}',
            throttle_duration_sec=2.0)
        # ── Target in arm-local ENU ───────────────────────────────────────
        # Subtract J1 world position to get arm-local ENU offset
        if self._j1_world_enu is not None:
            target_arm_enu = TARGET_WORLD_ENU - self._j1_world_enu
        else:
            # J1 not yet available — skip this cycle
            self.get_logger().warn('Waiting for TF...', throttle_duration_sec=1.0)
            return

        # ── Error (both in arm-local ENU — same frame, direct subtraction) ─
        err      = (target_arm_enu - ee_fk_arm_enu).reshape(3, 1)
        J        = swiftpro_jacobian(self.arm.q)   # arm-local ENU Jacobian
        err_norm = float(np.linalg.norm(err))

        # ── Resolved-Rate Control ─────────────────────────────────────────
        if method == 'transpose':
            dq = 4.0 * (J.T @ K @ err)
        elif method == 'pseudoinverse':
            dq = np.linalg.pinv(J) @ K @ err
            if np.linalg.norm(dq) > 1.5:
                dq = dq / np.linalg.norm(dq) * 1.5
        else:   # DLS
            dq = DLS(J, DAMPING) @ K @ err

        dq = scale_velocities(dq, MAX_VEL)

        # ── Publish joint velocity command (4 values: joint4 passive = 0) ─
        # 2. Add joint limit clamping before publishing.
        # Replace the cmd publish block with this:
        Q2_MIN, Q2_MAX = -1.40, 0.0
        Q3_MIN, Q3_MAX = -1.20, 0.60

        dq_arr = dq.flatten()
        q = self.arm.q
        # Zero out velocity if joint is at limit and moving further past it
        if q[1] <= Q2_MIN and dq_arr[1] < 0: dq_arr[1] = 0.0
        if q[1] >= Q2_MAX and dq_arr[1] > 0: dq_arr[1] = 0.0
        if q[2] <= Q3_MIN and dq_arr[2] < 0: dq_arr[2] = 0.0
        if q[2] >= Q3_MAX and dq_arr[2] > 0: dq_arr[2] = 0.0

        cmd = Float64MultiArray()
        cmd.data = dq.flatten().tolist() + [0.0]
        self.pub_cmd.publish(cmd)

        # ── Markers in world_enu ──────────────────────────────────────────
        # Green sphere: true EE from TF; fall back to FK + J1 offset
        if ee_world_enu is not None:
            green_pos = ee_world_enu
        elif self._j1_world_enu is not None:
            green_pos = self._j1_world_enu + ee_fk_arm_enu
        else:
            green_pos = ee_fk_arm_enu
        self._publish_markers(green_pos, TARGET_WORLD_ENU, method, err_norm)

        # ── FK vs TF diagnostic ───────────────────────────────────────────
        if ee_world_enu is not None and self._j1_world_enu is not None:
            fk_world = self._j1_world_enu + ee_fk_arm_enu
            fk_diff  = float(np.linalg.norm(ee_world_enu - fk_world))
            diff_str = f'  FK_vs_TF={fk_diff*1000:.1f}mm'
        else:
            diff_str = ''

        dqf = dq.flatten()
        self.get_logger().info(
            f'[{method}] ctrl_err={err_norm:.4f}m | '
            f'dq=[{float(dqf[0]):.3f},{float(dqf[1]):.3f},{float(dqf[2]):.3f}]'
            f'{diff_str}',
            throttle_duration_sec=0.5)

        # self.arm.integrate(dq.flatten(), DT)

    def _publish_markers(self, ee_enu, target_enu, method, err_norm):
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        def mk(mid, mtype):
            m = Marker()
            m.header.frame_id    = WORLD_FRAME
            m.header.stamp       = now
            m.ns                 = 'hoi_rrc'
            m.id                 = mid
            m.type               = mtype
            m.action             = Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        # RED sphere — target
        m0 = mk(ID_TARGET, Marker.SPHERE)
        m0.pose.position.x = float(target_enu[0])
        m0.pose.position.y = float(target_enu[1])
        m0.pose.position.z = float(target_enu[2])
        m0.scale.x = m0.scale.y = m0.scale.z = 0.025
        m0.color.r = 1.0; m0.color.g = 0.0; m0.color.b = 0.0; m0.color.a = 1.0
        ma.markers.append(m0)

        # GREEN→YELLOW sphere — current EE
        closeness = float(np.clip(1.0 - err_norm / 0.1, 0.0, 1.0))
        m1 = mk(ID_CURRENT, Marker.SPHERE)
        m1.pose.position.x = float(ee_enu[0])
        m1.pose.position.y = float(ee_enu[1])
        m1.pose.position.z = float(ee_enu[2])
        m1.scale.x = m1.scale.y = m1.scale.z = 0.018
        m1.color.r = 1.0 - closeness
        m1.color.g = 1.0; m1.color.b = 0.0; m1.color.a = 1.0
        ma.markers.append(m1)

        # WHITE line — error vector
        m2 = mk(ID_ERROR, Marker.LINE_STRIP)
        m2.scale.x = 0.003
        m2.color.r = m2.color.g = m2.color.b = 1.0; m2.color.a = 0.8
        p1 = Point(); p1.x=float(ee_enu[0]);     p1.y=float(ee_enu[1]);     p1.z=float(ee_enu[2])
        p2 = Point(); p2.x=float(target_enu[0]); p2.y=float(target_enu[1]); p2.z=float(target_enu[2])
        m2.points = [p1, p2]
        ma.markers.append(m2)

        # WHITE text
        m3 = mk(ID_TEXT, Marker.TEXT_VIEW_FACING)
        m3.pose.position.x = float(target_enu[0])
        m3.pose.position.y = float(target_enu[1])
        m3.pose.position.z = float(target_enu[2]) + 0.05
        m3.scale.z = 0.02
        m3.color.r = m3.color.g = m3.color.b = m3.color.a = 1.0
        m3.text = f'[{method}]  err={err_norm:.4f}m'
        ma.markers.append(m3)

        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = Lab2RRCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
