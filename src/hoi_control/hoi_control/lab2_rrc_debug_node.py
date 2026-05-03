#!/usr/bin/env python3
"""
lab2_rrc_DEBUG_node.py
======================
Debug version of RRC with:
  - TF-based position error  (main fix: error = TF_EE - target, not FK_EE - target)
  - FK used ONLY for Jacobian direction
  - All parameters settable at runtime via ros2 param set
  - 4 RViz spheres: red=target, green=TF_EE, cyan=FK_estimate, yellow=J1_base
  - Comprehensive terminal logging

Runtime parameter adjustment:
  ros2 param set /lab2_rrc_debug target_x 0.05
  ros2 param set /lab2_rrc_debug target_y 0.15
  ros2 param set /lab2_rrc_debug target_z 0.40
  ros2 param set /lab2_rrc_debug gain_k 1.0
  ros2 param set /lab2_rrc_debug max_vel 0.5
  ros2 param set /lab2_rrc_debug damping 0.1
  ros2 param set /lab2_rrc_debug method 0    (0=transpose, 1=pseudoinverse, 2=DLS)
  ros2 param set /lab2_rrc_debug enabled true/false

All targets in world_enu: x=East, y=North, z=Up
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

from tf2_ros import Buffer, TransformListener, LookupException, \
    ConnectivityException, ExtrapolationException

from hoi_control.swiftpro_robotics import (
    L2, L3, L4,
    SwiftProManipulator, swiftpro_fk, swiftpro_jacobian, DLS, scale_velocities
)

# Topics
JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
MARKER_TOPIC      = '/hoi/debug_markers'
WORLD_FRAME       = 'world_enu'
EE_FRAME          = 'turtlebot/swiftpro/ee_link'
J1_FRAME          = 'turtlebot/swiftpro/link1'
DT                =  1.0 / 60.0

# Marker IDs
ID_TARGET   = 0   # RED   — desired target
ID_TF_EE    = 1   # GREEN — true EE from TF
ID_FK_EE    = 2   # CYAN  — FK estimate of EE
ID_J1_BASE  = 3   # YELLOW — J1 arm base
ID_ERR_LINE = 4   # WHITE line from TF_EE to target
ID_TEXT     = 5   # info text

class Lab2RRCDebugNode(Node):

    def __init__(self):
        super().__init__('lab2_rrc_debug')

        # ── Declare all runtime-tunable parameters ─────────────────────────
        # Target position in world_enu — START VERY CLOSE TO REST POSITION
        # Rest EE is at approx [0, 0.159, 0.480] world_enu
        self.declare_parameter('target_x',  -0.25)   # East
        self.declare_parameter('target_y',  0.0)  # North — same as rest
        self.declare_parameter('target_z',  0.30)   # Up    — 80mm lower than rest

        self.declare_parameter('gain_k',    0.5)    # low gain for safety
        self.declare_parameter('max_vel',   0.3)    # conservative velocity limit
        self.declare_parameter('damping',   0.1)    # DLS damping
        self.declare_parameter('method',    1)      # 0=transpose,1=pinv,2=DLS
        self.declare_parameter('enabled',   False)   # master enable/disable

        # Manual mode: directly command joint velocities for calibration
        self.declare_parameter('manual_mode', False)  # if True, use manual dq
        self.declare_parameter('manual_dq1',  0.0)
        self.declare_parameter('manual_dq2',  0.0)
        self.declare_parameter('manual_dq3',  0.0)

        # ── State ─────────────────────────────────────────────────────────
        self.arm = SwiftProManipulator()
        self._got_js       = False
        self._j1_world_enu = None
        self._loop_count   = 0
        self._last_tf_ee   = None

        # ── TF ────────────────────────────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── ROS I/O ───────────────────────────────────────────────────────
        self.sub_js = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.pub_cmd     = self.create_publisher(Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self.pub_markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self.timer       = self.create_timer(DT, self._control_loop)

        self.get_logger().info('=' * 60)
        self.get_logger().info('Lab2 RRC DEBUG node started')
        self.get_logger().info('Tune with: ros2 param set /lab2_rrc_debug <param> <value>')
        self.get_logger().info('Parameters: target_x/y/z, gain_k, max_vel, damping, method, enabled')
        self.get_logger().info('RViz: Fixed Frame=world_enu, Add MarkerArray from /hoi/debug_markers')
        self.get_logger().info('=' * 60)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))
        self._got_js = True

    def _tf_pos(self, child_frame):
        try:
            t = self._tf_buffer.lookup_transform(
                WORLD_FRAME, child_frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        
    def _tf_rotation(self, child_frame):
        """Get the 3x3 rotation matrix of child_frame in world_enu."""
        try:
            t = self._tf_buffer.lookup_transform(
                WORLD_FRAME, child_frame, rclpy.time.Time())
            q = t.transform.rotation
            # Convert quaternion to rotation matrix
            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
                [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
                [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)]
            ])
            return R
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # ── Main loop ─────────────────────────────────────────────────────────

    def _control_loop(self):
        if not self._got_js:
            return

        self._loop_count += 1

        # ── Read runtime parameters ───────────────────────────────────────
        tx  = self.get_parameter('target_x').value
        ty  = self.get_parameter('target_y').value
        tz  = self.get_parameter('target_z').value
        K   = np.diag([self.get_parameter('gain_k').value] * 3)
        mv  = self.get_parameter('max_vel').value
        dam = self.get_parameter('damping').value
        mth = int(self.get_parameter('method').value)
        ena = self.get_parameter('enabled').value

        target_world_enu = np.array([tx, ty, tz])
        method_name      = ['transpose', 'pseudoinverse', 'DLS'][mth % 3]

        # ── TF lookups ────────────────────────────────────────────────────
        ee_world_enu = self._tf_pos(EE_FRAME)
        j1_world_enu = self._tf_pos(J1_FRAME)

        if j1_world_enu is not None:
            self._j1_world_enu = j1_world_enu
        if ee_world_enu is not None:
            self._last_tf_ee = ee_world_enu

        # ── FK estimate (for Jacobian AND comparison only) ─────────────────
        fk_arm_enu    = swiftpro_fk(self.arm.q)
        fk_world_enu  = (self._j1_world_enu + fk_arm_enu
                         if self._j1_world_enu is not None
                         else fk_arm_enu)

        # ── Position error: TF-based (THE KEY FIX) ───────────────────────
        # We use the TRUE EE position from TF, NOT the FK estimate.
        # This means even if FK is wrong, the error vector points correctly.
        # Get live rotation of link1 (arm base frame orientation in world_enu)
        R_link1 = self._tf_rotation(J1_FRAME)

        if ee_world_enu is not None and self._j1_world_enu is not None and R_link1 is not None:
            ee_disp     = ee_world_enu     - self._j1_world_enu
            target_disp = target_world_enu - self._j1_world_enu
            
            # R_link1.T rotates world_enu displacements into arm-local FK frame
            ee_arm_enu     = R_link1.T @ ee_disp
            target_arm_enu = R_link1.T @ target_disp
            
            pos_err  = (target_arm_enu - ee_arm_enu).reshape(3, 1)
            err_norm = float(np.linalg.norm(pos_err))
            fk_vs_tf_mm = float(np.linalg.norm(fk_arm_enu - ee_arm_enu)) * 1000
            position_source = 'TF'
        # if ee_world_enu is not None and self._j1_world_enu is not None:
        #     ee_arm_enu     = ee_world_enu - self._j1_world_enu   # TF, arm-local ENU
        #     target_arm_enu = target_world_enu - self._j1_world_enu
        #     err            = (target_arm_enu - ee_arm_enu).reshape(3, 1)
        #     err_norm       = float(np.linalg.norm(err))
        #     fk_vs_tf       = float(np.linalg.norm(fk_arm_enu - ee_arm_enu)) * 1000
        #     position_source = 'TF'
        else:
            # Fallback to FK if TF unavailable
            target_arm_enu = (target_world_enu - self._j1_world_enu
                              if self._j1_world_enu is not None
                              else target_world_enu)
            err            = (target_arm_enu - fk_arm_enu).reshape(3, 1)
            err_norm       = float(np.linalg.norm(err))
            fk_vs_tf       = float('nan')
            position_source = 'FK_fallback'
            self.get_logger().warn('TF unavailable, using FK for error',
                                   throttle_duration_sec=1.0)

        # ── Jacobian (from FK formula — used only for direction mapping) ──
        J = swiftpro_jacobian(self.arm.q)
        J_cond = float(np.linalg.cond(J))

        # ── Arm direction diagnostic ─────────────────────────────────────
        if ee_world_enu is not None and self._j1_world_enu is not None:
            ee_rel = ee_world_enu - self._j1_world_enu
            arm_dir = float(np.arctan2(ee_rel[1], -ee_rel[0]))
            r_tf    = float(np.sqrt(ee_rel[0]**2 + ee_rel[1]**2))
            z_tf    = float(ee_rel[2])   # ADD THIS
            if self._loop_count % 30 == 0:
                self.get_logger().info(
                    f'  arm_dir(TF)  : {arm_dir:.3f} rad  q1={self.arm.q[0]:.3f} rad\n'
                    f'  r_tf={r_tf*1000:.1f}mm  z_tf={z_tf*1000:.1f}mm  '
                    f'r_fk: {float(np.sqrt(fk_arm_enu[0]**2 + fk_arm_enu[1]**2))*1000:.1f}mm')


        # ── Resolved-Rate Control ─────────────────────────────────────────
        manual_mode = self.get_parameter('manual_mode').value
        if manual_mode:
            dq_arr = np.array([
                self.get_parameter('manual_dq1').value,
                self.get_parameter('manual_dq2').value,
                self.get_parameter('manual_dq3').value,
            ])
            cmd = Float64MultiArray()
            cmd.data = dq_arr.tolist() + [0.0]
            self.pub_cmd.publish(cmd)
            if self._loop_count % 30 == 0:
                self.get_logger().info(f'  MANUAL MODE: dq={dq_arr}')
            self._publish_markers(target_world_enu, ee_world_enu, fk_world_enu, self._j1_world_enu, err_norm, 'MANUAL')
            return

        if ena:
            if mth % 3 == 0:    # transpose
                dq = 4.0 * (J.T @ K @ err)
            elif mth % 3 == 1:  # pseudoinverse
                dq = np.linalg.pinv(J) @ K @ err
                if np.linalg.norm(dq) > 2.0:
                    dq = dq / np.linalg.norm(dq) * 2.0
            else:               # DLS
                dq = DLS(J, dam) @ K @ err

            dq = scale_velocities(dq, mv)

            # Joint limit clamping (safety)
            dq_arr = dq.flatten()
            q      = self.arm.q
            if q[1] <= -1.50 and dq_arr[1] < 0: dq_arr[1] = 0.0
            if q[1] >=  0.045 and dq_arr[1] > 0: dq_arr[1] = 0.0
            if q[2] <= -1.50 and dq_arr[2] < 0: dq_arr[2] = 0.0
            if q[2] >=  0.045 and dq_arr[2] > 0: dq_arr[2] = 0.0

            cmd = Float64MultiArray()
            cmd.data = dq_arr.tolist() + [0.0]
            self.pub_cmd.publish(cmd)
        else:
            # Disabled: send zero velocity
            dq_arr = np.zeros(3)
            cmd = Float64MultiArray()
            cmd.data = [0.0, 0.0, 0.0, 0.0]
            self.pub_cmd.publish(cmd)

        # ── Verbose debug log (every 30 loops ≈ 0.5 sec) ─────────────────
        if self._loop_count % 30 == 0:
            q = self.arm.q
            self.get_logger().info(
                f'\n{"─"*55}\n'
                f'  method      : {method_name}  |  enabled: {ena}\n'
                f'  q           : [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}] rad\n'
                f'  target(wenu): [{tx:.3f}, {ty:.3f}, {tz:.3f}]\n'
                f'  TF EE(wenu) : {np.round(ee_world_enu,3) if ee_world_enu is not None else "N/A"}\n'
                f'  FK EE(wenu) : {np.round(fk_world_enu,3)}\n'
                f'  FK_vs_TF    : {fk_vs_tf:.1f} mm  (< 20mm is good)\n'
                f'  pos_source  : {position_source}\n'
                f'  err_norm    : {err_norm:.4f} m\n'
                f'  err_vec     : {np.round(err.flatten(),4)}\n'
                f'  dq          : {np.round(dq_arr,4)}\n'
                f'  J_cond      : {J_cond:.1f}  (> 100 = near singularity)\n'
                f'  params      : K={self.get_parameter("gain_k").value}  '
                f'mv={mv}  dam={dam}\n'
                f'{"─"*55}'
            )

        # ── Markers ───────────────────────────────────────────────────────
        self._publish_markers(
            target_world_enu,
            ee_world_enu,
            fk_world_enu,
            self._j1_world_enu,
            err_norm,
            method_name
        )

    # ── Markers ───────────────────────────────────────────────────────────

    def _publish_markers(self, target, tf_ee, fk_ee, j1, err_norm, method):
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        def mk(mid, mtype):
            m = Marker()
            m.header.frame_id    = WORLD_FRAME
            m.header.stamp       = now
            m.ns                 = 'debug_rrc'
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

        # RED — target
        ma.markers.append(sphere(ID_TARGET, target, 1.0, 0.0, 0.0, 0.030))

        # GREEN — TF EE (ground truth)
        if tf_ee is not None:
            closeness = float(np.clip(1.0 - err_norm / 0.1, 0.0, 1.0))
            m = mk(ID_TF_EE, Marker.SPHERE)
            m.pose.position.x = float(tf_ee[0])
            m.pose.position.y = float(tf_ee[1])
            m.pose.position.z = float(tf_ee[2])
            m.scale.x = m.scale.y = m.scale.z = 0.022
            m.color.r = 1.0 - closeness  # yellow=far, green=close
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            ma.markers.append(m)

        # CYAN — FK EE estimate (for comparison)
        ma.markers.append(sphere(ID_FK_EE, fk_ee, 0.0, 0.8, 1.0, 0.015))

        # YELLOW — J1 base position
        if j1 is not None:
            ma.markers.append(sphere(ID_J1_BASE, j1, 1.0, 1.0, 0.0, 0.018))

        # WHITE line — error vector from TF_EE to target
        if tf_ee is not None:
            m2 = mk(ID_ERR_LINE, Marker.LINE_STRIP)
            m2.scale.x = 0.004
            m2.color.r = m2.color.g = m2.color.b = 1.0; m2.color.a = 0.9
            p1 = Point(); p1.x = float(tf_ee[0]); p1.y = float(tf_ee[1]); p1.z = float(tf_ee[2])
            p2 = Point(); p2.x = float(target[0]); p2.y = float(target[1]); p2.z = float(target[2])
            m2.points = [p1, p2]
            ma.markers.append(m2)

        # Text above target
        mt = mk(ID_TEXT, Marker.TEXT_VIEW_FACING)
        mt.pose.position.x = float(target[0])
        mt.pose.position.y = float(target[1])
        mt.pose.position.z = float(target[2]) + 0.06
        mt.scale.z = 0.025
        mt.color.r = mt.color.g = mt.color.b = mt.color.a = 1.0
        mt.text = f'[{method}] err={err_norm:.3f}m'
        ma.markers.append(mt)

        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = Lab2RRCDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# NOTE: To explore how q2 and q3 affect the arm, use manual mode:
#   ros2 param set /lab2_rrc_debug enabled false
#   ros2 param set /lab2_rrc_debug manual_dq2 0.2   # apply dq2 velocity
#   ros2 param set /lab2_rrc_debug manual_dq3 0.0
# Watch FK_vs_TF and TF EE position in the terminal.
# This will let us calibrate the FK formula empirically.
