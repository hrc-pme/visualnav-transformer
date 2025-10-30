from typing import Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool, Float32MultiArray
from scipy.spatial.transform import Rotation as R

# Load Config
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1 / robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1  # seconds


class Waypoint2Goal(Node):
    def __init__(self):
        super().__init__("waypoint2goal")

        qos = QoSProfile(depth=10)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, qos)  # 新增速度指令發佈器
        self.create_subscription(Float32MultiArray, "waypoint", self.callback_drive, qos)
        self.create_subscription(Bool, "/reach_goal", self.callback_reached_goal, qos)  # 修正 topic 名稱

        self.waypoint = None
        self.reached_goal = False

        self.create_timer(1.0 / RATE, self.control_loop)

        # 初始化時清空並輸出單行狀態
        print("\r[Waypoint2Goal] Initialized. Waiting for waypoint..." + " "*50, flush=True)

    def convert_waypoint_pose(self, waypoint: np.ndarray) -> PoseStamped:
        assert len(waypoint) in [2, 4], "waypoint must be 2D or 4D"
        if len(waypoint) == 2:
            dx, dy = waypoint
            hx, hy = 1.0, 0.0  # 預設朝向X軸
        else:
            dx, dy, hx, hy = waypoint

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "base_link"
        goal_pose.pose.position.x = float(dx)
        goal_pose.pose.position.y = float(dy)
        goal_pose.pose.position.z = 0.0

        # 使用 scipy 計算四元數 (x, y, z, w 格式)
        rotation = R.from_euler('z', np.arctan2(hy, hx))
        quaternion = rotation.as_quat()  # 返回 [x, y, z, w] 格式
        goal_pose.pose.orientation.x = quaternion[0]
        goal_pose.pose.orientation.y = quaternion[1]
        goal_pose.pose.orientation.z = quaternion[2]
        goal_pose.pose.orientation.w = quaternion[3]

        return goal_pose

    def callback_drive(self, waypoint_msg: Float32MultiArray):
        self.waypoint = np.array(waypoint_msg.data)
        # 不需要每次都輸出 waypoint 接收訊息，改為在處理時輸出

    def callback_reached_goal(self, reached_goal_msg: Bool):
        self.reached_goal = reached_goal_msg.data
        # 移除 log 輸出，改為在 control_loop 中統一輸出

    def control_loop(self):
        if self.reached_goal:
            # 目標達成，發布停止訊號並清除航點
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.vel_pub.publish(twist)
            self.waypoint = None  # 清除航點以避免繼續處理
            print(f"\r[Waypoint2Goal] 🎯 GOAL REACHED! Robot stopped." + " "*80, end='', flush=True)
            return

        if self.waypoint is not None:
            # 發佈目標位姿
            goal_pose = self.convert_waypoint_pose(self.waypoint)
            self.goal_pub.publish(goal_pose)

            # 計算速度指令（簡單比例控制器）
            x, y = self.waypoint[0], self.waypoint[1]
            distance = np.linalg.norm([x, y])
            target_angle = np.arctan2(y, x)

            # 線速度控制增益
            k_v = 0.5
            linear_vel = min(k_v * distance, MAX_V)

            # 角速度控制增益
            k_w = 1.0
            angular_vel = np.clip(k_w * target_angle, -MAX_W, MAX_W)

            twist = Twist()
            twist.linear.x = float(linear_vel)
            twist.angular.z = float(angular_vel)

            self.vel_pub.publish(twist)
            # 使用 \r 清除同行並輸出新的狀態訊息，加上空格填充以清除舊內容
            print(f"\r[Waypoint2Goal] Vel: lin={linear_vel:.2f}m/s, ang={angular_vel:.2f}rad/s | Waypoint: [{x:.2f}, {y:.2f}] | Dist: {distance:.2f}m" + " "*20, end='', flush=True)

        else:
            # 靜默等待 waypoint，顯示等待狀態
            print(f"\r[Waypoint2Goal] Waiting for waypoint..." + " "*80, end='', flush=True)
            pass


def main(args=None):
    rclpy.init(args=args)
    pd_controller_node = Waypoint2Goal()
    rclpy.spin(pd_controller_node)
    pd_controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
