#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
# import Pose3D
import numpy as np
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
# from tutorial_interfaces.srv import Pose

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

class Localisation_Node(Node):
    def __init__(self):
        super().__init__('differential_drive_ekf')

        # Parameters
        self.declare_parameter('odom_frame', 'world_enu') # Per Lab Requirement
        # self.declare_parameter('odom_frame', 'odom') #<- for rosbag file
        self.declare_parameter('base_frame', 'turtlebot/base_footprint')
        # self.declare_parameter('base_frame', 'base_footprint')#<- for rosbag file

        self.declare_parameter('wheel_left_joint_name', 'turtlebot/wheel_left_joint')
        self.declare_parameter('wheel_right_joint_name', 'turtlebot/wheel_right_joint')


        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.left_wheel_name = self.get_parameter('wheel_left_joint_name').value
        self.right_wheel_name = self.get_parameter('wheel_right_joint_name').value

        # Robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        # self.wheel_radius *= 1.1 # introduce bias
        # self.wheel_base_distance *= 1.1
        self.wheel_vel_noise = np.array([0.1**2, 0.1**2]) # Q matrix tuning
        # self.imu_yaw_noise = 0.01 # R matrix tuning (IMU variance)

        # State Variables: [x, y, theta]
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.P = np.diag([0.0, 0.0, 0.0]) # Initial uncertainty
        
        self.first_received = False
        self.last_time = self.get_clock().now()
        
        # Ground truth odometry for comparison
        self.ground_truth_odom = None

        # Publishers, Subscribers, and TF
        self.js_sub = self.create_subscription(JointState, '/turtlebot/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/turtlebot/sensors/imu_data', self.imu_callback, 10)
        self.gt_odom_sub = self.create_subscription(Odometry, '/turtlebot/odom_ground_truth', self.ground_truth_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_br = TransformBroadcaster(self)
        # self.get_logger().info(f"Odom Node Started. Base: {self.base_frame}, Odom: {self.odom_frame}")


    def euler_from_quaternion(self, q):
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(t3, t4)

    def quaternion_from_euler(self, yaw):
        return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]

    def ground_truth_callback(self, msg):
        """Store ground truth odometry for comparison"""
        self.ground_truth_odom = msg

    #  EKF Prediction (Motion Model) 
    def predict(self, dt, wheel_velocities):
        v = (wheel_velocities.sum() * self.wheel_radius) / 2.0
        w = (wheel_velocities[1] - wheel_velocities[0]) * self.wheel_radius / self.wheel_base_distance
        
        # State Prediction
        self.x += np.cos(self.th) * v * dt
        self.y += np.sin(self.th) * v * dt
        self.th += w * dt
        # self.th = math.atan2(math.sin(self.th), math.cos(self.th)) # Normalize

        # Covariance Prediction: P = FPF' + WQW'
        F = np.array([[1, 0, -v * np.sin(self.th) * dt],
                      [0, 1,  v * np.cos(self.th) * dt],
                      [0, 0,  1]])
        
        W = np.array([[np.cos(self.th) * dt / 2.0, np.cos(self.th) * dt / 2.0],
                      [np.sin(self.th) * dt / 2.0, np.sin(self.th) * dt / 2.0],
                      [-dt / self.wheel_base_distance, dt / self.wheel_base_distance]])
        
        Q = np.diag(self.wheel_vel_noise)
        self.P = F @ self.P @ F.T + W @ Q @ W.T

    #  EKF Correction (IMU Update) 
    def imu_callback(self, msg):
        """Task 3: Update filter based on IMU orientation """
        # self.get_logger().info(f"IMU update")
        if not self.first_received: return

        # Measurement: Extract Yaw from IMU
        z = -self.euler_from_quaternion(msg.orientation) # IMU in NED frame so need conversion to ENU
        
        R = msg.orientation_covariance[8]
        # Innovation (Residual): y = z - Hx
        # H is [0, 0, 1] because we only measure theta
        H = np.array([[0, 0, 1]])
        y = z - self.th
        y = math.atan2(math.sin(y), math.cos(y)) # Normalize innovation

        # Kalman Gain: K = PH' / (HPH' + R)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update State and Covariance
        delta_x = K * y
        self.x += delta_x[0, 0]
        self.y += delta_x[1, 0]
        self.th += delta_x[2, 0]
        self.th = math.atan2(math.sin(self.th), math.cos(self.th))
        
        self.P = (np.eye(3) - K @ H) @ self.P

    def joint_state_callback(self, msg):
        try:
            l_idx = msg.name.index(self.left_wheel_name)
            r_idx = msg.name.index(self.right_wheel_name)
            wheel_vels = np.array([msg.velocity[l_idx], msg.velocity[r_idx]])
            stdev_wheel_vel = np.sqrt(self.wheel_vel_noise)
            # wheel_vels += np.random.normal(np.array([0.0, 0.0]), stdev_wheel_vel) # simulate random wheel noise
        except ValueError: return

        now = self.get_clock().now()
        if not self.first_received:
            self.last_time, self.first_received = now, True
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0: return

        # Predict based on encoders
        self.predict(dt, wheel_vels)

        # Publish results (TF and Odom)
        self.broadcast_and_publish(now)

    def broadcast_and_publish(self, time):
        q = self.quaternion_from_euler(self.th)
        
        # TF Broadcast for RViz Lidar projection 
        t = TransformStamped()
        t.header.stamp = time.to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x, t.transform.translation.y = self.x, self.y
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = 0.0, 0.0, q[2], q[3]
        self.tf_br.sendTransform(t)

        # Odometry Message
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = time.to_msg(), self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = q[2], q[3]
        
        # Mapping 3x3 P to 6x6 Odom Covariance 
        cov = np.zeros((6, 6))
        cov[0:2, 0:2] = self.P[0:2, 0:2]
        cov[0:2, 5] = self.P[0:2, 2]
        cov[5, 0:2] = self.P[2, 0:2]
        cov[5, 5] = self.P[2, 2]
        odom.pose.covariance = cov.flatten().tolist()
        self.odom_pub.publish(odom)
        
        # Log comparison: EKF predicted vs Ground truth
        if self.ground_truth_odom is not None:
            gt_x = self.ground_truth_odom.pose.pose.position.x
            gt_y = self.ground_truth_odom.pose.pose.position.y
            gt_z = self.ground_truth_odom.pose.pose.orientation.z
            gt_w = self.ground_truth_odom.pose.pose.orientation.w
            gt_yaw = -math.atan2(2.0 * (gt_w * gt_z), 1.0 - 2.0 * (gt_z * gt_z))
            
            # Calculate errors
            error_x = abs(self.x - gt_x)
            error_y = abs(self.y - gt_y)
            error_yaw = abs(self.th - gt_yaw)
            error_yaw = min(error_yaw, 2*math.pi - error_yaw)  # Shortest angle difference
            
            # self.get_logger().info(
            #     f"\n--- EKF vs Ground Truth ---\n"
            #     f"EKF:  x={self.x:.4f}, y={self.y:.4f}, yaw={self.th:.4f}\n"
            #     f"GT:   x={gt_x:.4f}, y={gt_y:.4f}, yaw={gt_yaw:.4f}\n"
            #     f"Error: Δx={error_x:.4f}, Δy={error_y:.4f}, Δyaw={error_yaw:.4f}\n"
            # )


def main():
    rclpy.init()
    node = Localisation_Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()