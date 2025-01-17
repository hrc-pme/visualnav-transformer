from typing import Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool, Float32MultiArray

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


class PDController(Node):
    def __init__(self):
        super().__init__("pd_controller")

        # Parameters
        self.declare_parameter("reverse_mode", False)
        self.reverse_mode = self.get_parameter("reverse_mode").value

        # Publisher and Subscribers
        qos = QoSProfile(depth=10)
        self.vel_publisher = self.create_publisher(Twist, VEL_TOPIC, qos)
        self.create_subscription(Float32MultiArray, "waypoint", self.callback_drive, qos)
        self.create_subscription(Bool, "reached_goal", self.callback_reached_goal, qos)

        # Variables
        self.vel_msg = Twist()
        self.waypoint = None
        self.reached_goal = False

        # Timer
        self.create_timer(1.0 / RATE, self.control_loop)

        self.get_logger().info("PD Controller initialized and waiting for waypoints.")

    def clip_angle(self, theta) -> float:
        """Clip angle to [-pi, pi]"""
        theta %= 2 * np.pi
        if -np.pi < theta < np.pi:
            return theta
        return theta - 2 * np.pi

    def pd_controller(self, waypoint: np.ndarray) -> Tuple[float, float]:
        """PD controller for the robot"""
        assert len(waypoint) in [2, 4], "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint

        if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
            v = 0
            w = self.clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            v = 0
            w = np.sign(dy) * np.pi / (2 * DT)
        else:
            v = dx / DT
            w = np.arctan(dy / dx) / DT

        v = np.clip(v, 0, MAX_V)
        w = np.clip(w, -MAX_W, MAX_W)
        return v, w

    def callback_drive(self, waypoint_msg: Float32MultiArray):
        """Callback function for the waypoint subscriber"""
        self.get_logger().info("Waypoint received.")
        self.waypoint = np.array(waypoint_msg.data)

    def callback_reached_goal(self, reached_goal_msg: Bool):
        """Callback function for the reached goal subscriber"""
        self.reached_goal = reached_goal_msg.data
        if self.reached_goal:
            self.get_logger().info("Goal reached! Stopping robot.")

    def control_loop(self):
        """Main control loop"""
        if self.reached_goal:
            self.vel_msg = Twist()
            self.vel_publisher.publish(self.vel_msg)
            return

        if self.waypoint is not None:
            v, w = self.pd_controller(self.waypoint)
            if self.reverse_mode:
                v *= -1
            self.vel_msg.linear.x = v
            self.vel_msg.angular.z = w
            self.get_logger().info(f"Publishing velocity: linear={v}, angular={w}")
        else:
            self.get_logger().warn("No valid waypoint received.")

        self.vel_publisher.publish(self.vel_msg)


def main(args=None):
    rclpy.init(args=args)
    pd_controller_node = PDController()
    rclpy.spin(pd_controller_node)
    pd_controller_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
