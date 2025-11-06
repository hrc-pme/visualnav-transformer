"""
ROS2 Topic Bridge
Handles all ROS2 topics and communicates with GPU container for inference.
Merges navigation logic and PD control into a single bridge.
"""
import argparse
import os
import sys
import time
import socket
from typing import Optional, List

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, Int32
from scipy.spatial.transform import Rotation as R
import cv2

# Add shared utilities
sys.path.append('/workspace/deploy/shared')
from socket_utils import send_data, recv_data

# Topic names
sys.path.append('/workspace/deploy/src')
from topic_names import (
    IMAGE_TOPIC, SAMPLED_ACTIONS_TOPIC, WAYPOINT_TOPIC,
    CURRENT_NODE_TOPIC, CANDIDATE_WAYPOINTS_TOPIC, CHOSEN_WAYPOINT_TOPIC,
    START_NODE_TOPIC, END_NODE_TOPIC
)

# Load robot configuration
ROBOT_CONFIG_PATH = "../config/robot.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1 / RATE

# Socket configuration
SOCKET_PATH = "/tmp/ipc_socket/gpu.sock"
SOCKET_TIMEOUT = 10.0


class TopicBridge(Node):
    """ROS2 Topic Bridge - handles all ROS communication and control logic"""
    
    def __init__(self, args: argparse.Namespace):
        super().__init__("topic_bridge")
        
        self.args = args
        self.bridge = CvBridge()
        
        # Navigation state
        self.context_queue: List[PILImage.Image] = []
        self.context_size = 5  # Will be updated from GPU server
        self.start_node = args.start_node if args.start_node is not None else 0
        self.goal_node = args.goal_node if args.goal_node is not None else -1
        self.closest_node = self.start_node
        self.reached_goal = False
        self.waypoint: Optional[np.ndarray] = None
        
        # Socket connection
        self.socket: Optional[socket.socket] = None
        self.connect_to_gpu_server()
        
        # Publishers
        qos = QoSProfile(depth=10)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, qos)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, qos)
        self.candidate_waypoints_pub = self.create_publisher(Float32MultiArray, CANDIDATE_WAYPOINTS_TOPIC, qos)
        self.chosen_waypoint_pub = self.create_publisher(Float32MultiArray, CHOSEN_WAYPOINT_TOPIC, qos)
        self.image_pub = self.create_publisher(Image, "camera/image/visualnav", qos_profile_sensor_data)
        self.reach_goal_pub = self.create_publisher(Bool, "/reach_goal", qos)
        self.current_node_pub = self.create_publisher(Int32, CURRENT_NODE_TOPIC, qos)
        self.start_node_pub = self.create_publisher(Int32, START_NODE_TOPIC, qos)
        self.end_node_pub = self.create_publisher(Int32, END_NODE_TOPIC, qos)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, qos)
        
        # Subscribers
        self.create_subscription(
            CompressedImage,
            IMAGE_TOPIC + "/compressed",
            self.callback_camera,
            qos_profile_sensor_data
        )
        
        # Control loop timer
        self.create_timer(1.0 / RATE, self.control_loop)
        
        print("[ROS2] Topic bridge initialized")
        print(f"[ROS2] Navigation: Start node {self.start_node} → Goal node {self.goal_node}")
    
    def connect_to_gpu_server(self):
        """Connect to GPU inference server"""
        max_retries = 10
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.socket.settimeout(SOCKET_TIMEOUT)
                self.socket.connect(SOCKET_PATH)
                print(f"[ROS2] Connected to GPU server at {SOCKET_PATH}")
                
                # Send ping to verify connection and get topomap info
                ping_request = {'type': 'ping'}
                if send_data(self.socket, ping_request):
                    response = recv_data(self.socket)
                    if response and response.get('type') == 'pong':
                        print("[ROS2] GPU server connection verified")
                        
                        # Get topomap info to set goal_node if not specified
                        num_nodes = response.get('num_nodes', 0)
                        context_size = response.get('context_size', 5)
                        
                        if num_nodes > 0:
                            print(f"[ROS2] Topomap has {num_nodes} nodes")
                            # Update goal_node if it was -1 (not specified)
                            if self.goal_node == -1:
                                self.goal_node = num_nodes - 1
                                print(f"[ROS2] Auto-set goal_node to {self.goal_node} (last node)")
                        
                        # Update context size from server
                        if context_size != self.context_size:
                            self.context_size = context_size
                            print(f"[ROS2] Context size set to {self.context_size}")
                        
                        return
                
            except Exception as e:
                print(f"[ROS2] Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if self.socket:
                    self.socket.close()
                    self.socket = None
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Failed to connect to GPU server after {max_retries} attempts")
    
    def callback_camera(self, msg: CompressedImage):
        """Process incoming compressed camera images"""
        try:
            # Convert compressed ROS image to PIL
            obs_img = self.compressed_msg_to_pil(msg).rotate(270, expand=True)
            
            # Publish visualnav image
            cv_img = np.array(obs_img)
            image_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(image_msg)
            
            # Update context queue
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)
        
        except Exception as e:
            print(f"[ROS2] Failed to process camera image: {e}")
    
    def compressed_msg_to_pil(self, msg: CompressedImage) -> PILImage.Image:
        """Convert ROS CompressedImage message to PIL Image"""
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        return PILImage.fromarray(cv_img)
    
    def msg_to_pil(self, msg: Image) -> PILImage.Image:
        """Convert ROS Image message to PIL Image"""
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        return PILImage.fromarray(img)
    
    def control_loop(self):
        """Main control loop - runs at RATE Hz"""
        # If goal reached, publish stop command
        if self.reached_goal:
            self.publish_zero_velocity()
            self.reach_goal_pub.publish(Bool(data=True))
            print("\r[ROS2] 🎯 Goal reached! Robot stopped." + " " * 50, end='', flush=True)
            return
        
        # Check if we have enough context
        if len(self.context_queue) <= self.context_size:
            print(f"\r[ROS2] Gathering context... ({len(self.context_queue)}/{self.context_size + 1})" 
                  + " " * 30, end='', flush=True)
            return
        
        # Request inference from GPU server
        inference_result = self.request_inference()
        
        if inference_result is None:
            print("[ROS2] Failed to get inference result")
            return
        
        # Extract results
        self.closest_node = inference_result['closest_node']
        chosen_waypoint = np.array(inference_result['chosen_waypoint'])
        model_type = inference_result.get('model_type', 'unknown')
        
        # Publish candidate waypoints
        if model_type == 'nomad' and inference_result.get('sampled_actions') is not None:
            sampled_actions = inference_result['sampled_actions']
            # Publish sampled actions
            self.sampled_actions_pub.publish(
                Float32MultiArray(data=np.concatenate(([0], sampled_actions.flatten())).tolist())
            )
            # Publish candidate waypoints (all samples at waypoint index)
            candidate_wps = sampled_actions[:, self.args.waypoint, :]
            self.candidate_waypoints_pub.publish(Float32MultiArray(data=candidate_wps.flatten().tolist()))
        
        elif inference_result.get('waypoints') is not None:
            waypoints = inference_result['waypoints']
            # Publish candidate waypoints (all candidate nodes at waypoint index)
            candidate_wps = waypoints[:, self.args.waypoint, :]
            self.candidate_waypoints_pub.publish(Float32MultiArray(data=candidate_wps.flatten().tolist()))
        
        # Publish chosen waypoint
        self.chosen_waypoint_pub.publish(Float32MultiArray(data=chosen_waypoint.tolist()))
        
        # Normalize waypoint (scale by velocity limits)
        # Note: Normalization is done on GPU side if model requires it
        # Here we scale by velocity limits
        self.waypoint = chosen_waypoint.copy()
        self.waypoint[0:2] *= MAX_V / RATE  # XY scaling
        self.waypoint[2] *= MAX_W / RATE * 0.3  # Z (yaw) scaling with ratio
        
        # Publish waypoint
        waypoint_msg = Float32MultiArray(data=self.waypoint.tolist())
        self.waypoint_pub.publish(waypoint_msg)
        
        # Publish current node
        self.current_node_pub.publish(Int32(data=int(self.closest_node)))
        
        # Publish start/end nodes
        self.start_node_pub.publish(Int32(data=int(self.start_node)))
        self.end_node_pub.publish(Int32(data=int(self.goal_node)))
        
        # Check if goal reached
        goal_reached = bool(self.closest_node == self.goal_node)
        self.reach_goal_pub.publish(Bool(data=goal_reached))
        
        if goal_reached:
            print("[ROS2] Goal node reached! Stopping robot...")
            self.reached_goal = True
            self.publish_zero_velocity()
            return
        
        # Publish velocity command (PD control)
        self.publish_velocity_command()
        
        # Status logging
        waypoint_str = f"[{self.waypoint[0]:.2f} {self.waypoint[1]:.2f} {self.waypoint[2]:.2f}]"
        print(f"\r[ROS2] Node: {self.closest_node}/{self.goal_node} | Waypoint: {waypoint_str}" 
              + " " * 30, end='', flush=True)
    
    def request_inference(self) -> Optional[dict]:
        """Request inference from GPU server"""
        try:
            # Calculate reference node range
            start = max(self.closest_node - self.args.radius, self.start_node)
            end = min(self.closest_node + self.args.radius, self.goal_node)
            
            # Create request
            request = {
                'type': 'inference',
                'context_queue': self.context_queue,
                'start_idx': start,
                'end_idx': end,
                'close_threshold': self.args.close_threshold,
                'num_samples': self.args.num_samples,
                'waypoint_idx': self.args.waypoint
            }
            
            # Send request
            if not send_data(self.socket, request):
                print("[ROS2] Failed to send inference request")
                return None
            
            # Receive response
            response = recv_data(self.socket)
            
            if response is None:
                print("[ROS2] Failed to receive inference response")
                return None
            
            if not response.get('success', False):
                print(f"[ROS2] Inference failed: {response.get('error', 'Unknown error')}")
                return None
            
            return response
        
        except Exception as e:
            print(f"[ROS2] Error during inference request: {e}")
            return None
    
    def publish_velocity_command(self):
        """Publish velocity command using simple PD control"""
        if self.waypoint is None:
            return
        
        # Extract position from waypoint
        x, y = self.waypoint[0], self.waypoint[1]
        distance = np.linalg.norm([x, y])
        target_angle = np.arctan2(y, x)
        
        # Simple proportional control
        k_v = 0.5
        k_w = 1.0
        
        linear_vel = min(k_v * distance, MAX_V)
        angular_vel = np.clip(k_w * target_angle, -MAX_W, MAX_W)
        
        # Publish velocity command
        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)
        self.vel_pub.publish(twist)
        
        # Also publish goal pose for visualization
        goal_pose = self.convert_waypoint_to_pose(self.waypoint)
        self.goal_pub.publish(goal_pose)
    
    def convert_waypoint_to_pose(self, waypoint: np.ndarray) -> PoseStamped:
        """Convert waypoint to PoseStamped message"""
        assert len(waypoint) in [2, 4], "waypoint must be 2D or 4D"
        
        if len(waypoint) == 2:
            dx, dy = waypoint
            hx, hy = 1.0, 0.0
        else:
            dx, dy, hx, hy = waypoint
        
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "base_link"
        goal_pose.pose.position.x = float(dx)
        goal_pose.pose.position.y = float(dy)
        goal_pose.pose.position.z = 0.0
        
        # Calculate quaternion from heading
        rotation = R.from_euler('z', np.arctan2(hy, hx))
        quaternion = rotation.as_quat()
        goal_pose.pose.orientation.x = quaternion[0]
        goal_pose.pose.orientation.y = quaternion[1]
        goal_pose.pose.orientation.z = quaternion[2]
        goal_pose.pose.orientation.w = quaternion[3]
        
        return goal_pose
    
    def publish_zero_velocity(self):
        """Publish zero velocity to stop robot"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.socket:
            self.socket.close()
        print("[ROS2] Topic bridge cleaned up")


def main():
    parser = argparse.ArgumentParser(description="ROS2 Topic Bridge")
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="Waypoint index to use (default: 2)"
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=None,
        type=int,
        help="Goal node index (default: last node)"
    )
    parser.add_argument(
        "--start-node",
        "-s",
        default=None,
        type=int,
        help="Start node index (default: 0)"
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="Node proximity threshold (default: 3)"
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="Local node search radius (default: 4)"
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="Number of samples for NOMAD (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create and run bridge node
        bridge = TopicBridge(args)
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        print("\n[ROS2] Shutting down...")
    except Exception as e:
        print(f"[ROS2] Error: {e}")
    finally:
        if 'bridge' in locals():
            bridge.cleanup()
            bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
