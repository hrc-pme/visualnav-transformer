#!/usr/bin/env python3
"""
Unified Navigation Module
Combines visual navigation model inference and waypoint-to-goal conversion
Supports GNM, ViNT, and NoMaD models
"""

import argparse
import os
import sys
import time
import subprocess
from typing import Optional, List, Tuple

import numpy as np
import rclpy
import torch
import yaml
from cv_bridge import CvBridge
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from geometry_msgs.msg import PoseStamped, Twist
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data, QoSReliabilityPolicy, QoSDurabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32

from topic_names import (
    IMAGE_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    WAYPOINT_TOPIC,
    CURRENT_NODE_TOPIC,
    CANDIDATE_WAYPOINTS_TOPIC,
    CHOSEN_WAYPOINT_TOPIC,
    START_NODE_TOPIC,
    END_NODE_TOPIC
)
from utils import load_model, msg_to_pil, to_numpy, transform_images
from vint_train.training.train_utils import get_action

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Paths
TOPOMAP_IMAGES_DIR = "../topomaps"
TOPOMAP_NAME = "6e-dr"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

# Navigation defaults
MODEL = "vint"  # Default model: gnm/vint/nomad
NODE_RANGE = [-1, -1]  # [start_node, goal_node] - if [-1, -1] auto-detect full range
Z_RATIO = 0.3  # scale z (yaw) speed
XY_RATIO = 1.0

# Model download URLs
MODEL_URLS = {
    "gnm": "https://drive.google.com/file/d/1bzCPd_OsXjS2aGPTQladbI8ImxLZwrQh/view?usp=drive_link",
    "vint": "https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view?usp=drive_link",
    "nomad": "https://drive.google.com/file/d/1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb/view?usp=drive_link"
}

# Load robot configuration
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
    
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
VEL_TOPIC = robot_config["vel_navi_topic"]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_google_drive_id(url: str) -> Optional[str]:
    """Extract file ID from Google Drive URL"""
    if "drive.google.com" in url and "/file/d/" in url:
        return url.split("/file/d/")[1].split("/")[0]
    return None


def download_model_from_google_drive(file_id: str, destination: str) -> bool:
    """Download model file from Google Drive using gdown"""
    try:
        import gdown
    except ImportError:
        print("gdown not found. Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        except Exception as e:
            print(f"Failed to install gdown: {e}")
            return False
    
    try:
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading model to {destination}...")
        gdown.download(download_url, destination, quiet=False)
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


def ensure_model_exists(model_name: str, model_path: str) -> bool:
    """Check if model exists, download if missing"""
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
        return True
    
    print(f"Model {model_name} not found at {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if model_name in MODEL_URLS:
        file_id = extract_google_drive_id(MODEL_URLS[model_name])
        if file_id and download_model_from_google_drive(file_id, model_path):
            print(f"Successfully downloaded {model_name} model")
            return True
    
    print(f"Failed to download {model_name} model")
    return False


# ============================================================================
# UNIFIED NAVIGATION NODE
# ============================================================================

class UnifiedNavigationNode(Node):
    """
    Unified navigation node that combines:
    1. Visual navigation model inference
    2. Waypoint-to-goal pose conversion
    3. Velocity command publishing
    """
    
    def __init__(self, model, model_params, topomap, args):
        super().__init__("unified_navigation_node")
        
        # Model and navigation parameters
        self.model = model
        self.model_params = model_params
        self.topomap = topomap
        self.num_nodes = len(topomap)
        self.args = args
        
        # Navigation state
        self.context_queue = []
        self.context_size = model_params["context_size"]
        self.closest_node = args.start_node
        self.start_node = args.start_node
        self.goal_node = args.goal_node
        self.reached_goal = False
        self.current_waypoint = None
        
        # Setup noise scheduler for NoMaD
        if model_params["model_type"] == "nomad":
            num_diffusion_iters = model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )
        else:
            self.noise_scheduler = None
        
        # ROS setup
        self.bridge = CvBridge()
        qos = QoSProfile(depth=10)
        
        # Camera QoS - Use BEST_EFFORT to match sensor data publishers
        camera_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Subscribers
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self._callback_obs,
            camera_qos  # Use camera-compatible QoS
        )
        
        self.get_logger().info("📷 Subscribed to /camera/camera/color/image_raw with BEST_EFFORT QoS")
        
        # Publishers
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, qos)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, qos)
        self.candidate_waypoints_pub = self.create_publisher(Float32MultiArray, CANDIDATE_WAYPOINTS_TOPIC, qos)
        self.chosen_waypoint_pub = self.create_publisher(Float32MultiArray, CHOSEN_WAYPOINT_TOPIC, qos)
        self.image_pub = self.create_publisher(Image, "camera/image/visualnav", qos_profile_sensor_data)
        self.reach_goal_pub = self.create_publisher(Bool, "/reach_goal", qos)
        self.current_node_pub = self.create_publisher(Int32, CURRENT_NODE_TOPIC, qos)
        self.start_node_pub = self.create_publisher(Int32, START_NODE_TOPIC, qos)
        self.end_node_pub = self.create_publisher(Int32, END_NODE_TOPIC, qos)
        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, qos)
        
        # Control gains for waypoint following
        self.k_v = 0.5  # Linear velocity gain
        self.k_w = 1.0  # Angular velocity gain
        
        # Publish initial start and end nodes
        self._publish_node_info()
        
        print(f"🤖 Unified Navigation Node initialized")
        print(f"   Start: {self.start_node} → Goal: {self.goal_node} (Total: {self.num_nodes} nodes)")
        print(f"📷 Waiting for camera images on topic: /camera/camera/color/image_raw")
        print(f"   Need {self.context_size + 1} images to start navigation...")
    
    def _callback_obs(self, msg: Image):
        """Process incoming camera observations"""
        try:
            obs_img = msg_to_pil(msg).rotate(270, expand=True)
            
            # Publish visualization
            try:
                cv_img = np.array(obs_img)
                image_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
                image_msg.header.stamp = self.get_clock().now().to_msg()
                self.image_pub.publish(image_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish visualnav image: {e}")
            
            # Update context queue
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
                # Log progress during initial collection - use print for immediate feedback
                print(f"\r[Camera] 📸 Collected image {len(self.context_queue)}/{self.context_size + 1}" + " "*50, end='', flush=True)
                if len(self.context_queue) == self.context_size + 1:
                    print()  # New line when ready
                    print("✅ Context collection complete! Starting navigation...")
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)
        except Exception as e:
            self.get_logger().error(f"Error in camera callback: {e}")
    
    def _publish_node_info(self):
        """Publish start, end, and current node information"""
        self.start_node_pub.publish(Int32(data=int(self.start_node)))
        self.end_node_pub.publish(Int32(data=int(self.goal_node)))
        self.current_node_pub.publish(Int32(data=int(self.closest_node)))
    
    def _publish_zero_velocity(self):
        """Publish zero velocity to stop the robot"""
        stop_cmd = Twist()
        self.vel_pub.publish(stop_cmd)
    
    def _convert_waypoint_to_pose(self, waypoint: np.ndarray) -> PoseStamped:
        """Convert waypoint to PoseStamped message"""
        assert len(waypoint) in [2, 4], "waypoint must be 2D or 4D"
        
        if len(waypoint) == 2:
            dx, dy = waypoint
            hx, hy = 1.0, 0.0  # Default heading along X-axis
        else:
            dx, dy, hx, hy = waypoint
        
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "base_link"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(dx)
        goal_pose.pose.position.y = float(dy)
        goal_pose.pose.position.z = 0.0
        
        # Compute quaternion from heading
        rotation = R.from_euler('z', np.arctan2(hy, hx))
        quaternion = rotation.as_quat()  # Returns [x, y, z, w]
        goal_pose.pose.orientation.x = quaternion[0]
        goal_pose.pose.orientation.y = quaternion[1]
        goal_pose.pose.orientation.z = quaternion[2]
        goal_pose.pose.orientation.w = quaternion[3]
        
        return goal_pose
    
    def _compute_velocity_command(self, waypoint: np.ndarray) -> Twist:
        """Compute velocity command from waypoint using proportional control"""
        x, y = waypoint[0], waypoint[1]
        distance = np.linalg.norm([x, y])
        target_angle = np.arctan2(y, x)
        
        # Proportional control
        linear_vel = min(self.k_v * distance, MAX_V)
        angular_vel = np.clip(self.k_w * target_angle, -MAX_W, MAX_W)
        
        twist = Twist()
        twist.linear.x = float(linear_vel)
        twist.angular.z = float(angular_vel)
        
        return twist
    
    def _infer_waypoint_nomad(self) -> Tuple[np.ndarray, int]:
        """Infer waypoint using NoMaD model"""
        start = max(self.closest_node - self.args.radius, self.start_node)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        
        # Prepare observations
        obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(device)
        mask = torch.zeros(1).long().to(device)
        
        # Prepare goal images
        goal_image = [
            transform_images(img, self.model_params["image_size"], center_crop=False).to(device)
            for img in self.topomap[start:end+1]
        ]
        goal_image = torch.cat(goal_image, dim=0)
        
        # Vision encoding and distance prediction
        obsgoal_cond = self.model(
            "vision_encoder",
            obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
            goal_img=goal_image,
            input_goal_mask=mask.repeat(len(goal_image))
        )
        dists = to_numpy(self.model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten())
        min_idx = np.argmin(dists)
        closest_node = min_idx + start
        
        # Select subgoal
        sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
        
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(self.args.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
        
        # Diffusion sampling
        naction = torch.randn((self.args.num_samples, self.model_params["len_traj_pred"], 2), device=device)
        self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
        
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.noise_scheduler.step(noise_pred, k, naction).prev_sample
        
        naction = to_numpy(get_action(naction))
        
        # Publish sampled actions
        self.sampled_actions_pub.publish(
            Float32MultiArray(data=np.concatenate(([0], naction.flatten())).tolist())
        )
        
        # Publish candidate waypoints
        candidate_wps = naction[:, self.args.waypoint, :]
        self.candidate_waypoints_pub.publish(Float32MultiArray(data=candidate_wps.flatten().tolist()))
        
        # Choose waypoint
        chosen_waypoint = naction[0][self.args.waypoint]
        
        return chosen_waypoint, closest_node
    
    def _infer_waypoint_gnm_vint(self) -> Tuple[np.ndarray, int]:
        """Infer waypoint using GNM or ViNT model"""
        start = max(self.closest_node - self.args.radius, self.start_node)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        
        # Prepare batch observations and goals
        batch_obs_imgs = [
            transform_images(self.context_queue, self.model_params["image_size"])
            for _ in range(end - start + 1)
        ]
        batch_goal_data = [
            transform_images(self.topomap[i], self.model_params["image_size"])
            for i in range(start, end + 1)
        ]
        batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
        batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)
        
        # Model inference
        distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
        distances = to_numpy(distances)
        waypoints = to_numpy(waypoints)
        
        min_dist_idx = np.argmin(distances)
        
        # Publish candidate waypoints
        candidate_wps = waypoints[:, self.args.waypoint, :]
        self.candidate_waypoints_pub.publish(Float32MultiArray(data=candidate_wps.flatten().tolist()))
        
        # Choose waypoint based on distance threshold
        if distances[min_dist_idx] > self.args.close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][self.args.waypoint]
            closest_node = start + min_dist_idx
        else:
            cur_wp = waypoints[min_dist_idx][self.args.waypoint]
            next_idx = min(min_dist_idx + 1, len(waypoints) - 1)
            next_wp = waypoints[next_idx][self.args.waypoint]
            
            delta = np.abs(np.array(cur_wp) - np.array(next_wp))
            if np.any(delta < 0.25):
                chosen_waypoint = next_wp
                closest_node = min(start + min_dist_idx + 1, self.goal_node)
            else:
                chosen_waypoint = cur_wp
                closest_node = start + min_dist_idx
        
        return chosen_waypoint, closest_node
    
    def navigation_step(self):
        """Execute one navigation step"""
        # If goal reached, stop and return
        if self.reached_goal:
            self._publish_zero_velocity()
            self.reach_goal_pub.publish(Bool(data=True))
            print("\r[Navigation] 🎯 GOAL REACHED! Robot stopped." + " "*80, end='', flush=True)
            return
        
        # Wait for sufficient context
        if len(self.context_queue) <= self.model_params["context_size"]:
            # Don't print here - callback handles progress display
            return
        
        # Infer waypoint based on model type
        if self.model_params["model_type"] == "nomad":
            chosen_waypoint, self.closest_node = self._infer_waypoint_nomad()
        else:
            chosen_waypoint, self.closest_node = self._infer_waypoint_gnm_vint()
        
        # Normalize and scale waypoint
        if self.model_params["normalize"]:
            chosen_waypoint = chosen_waypoint.copy()
            chosen_waypoint[0:2] *= MAX_V / RATE * XY_RATIO
            if len(chosen_waypoint) > 2:
                chosen_waypoint[2] *= MAX_W / RATE * Z_RATIO
        
        # Store and publish waypoint
        self.current_waypoint = chosen_waypoint
        self.waypoint_pub.publish(Float32MultiArray(data=chosen_waypoint.tolist()))
        self.chosen_waypoint_pub.publish(Float32MultiArray(data=chosen_waypoint.tolist()))
        
        # Publish goal pose
        goal_pose = self._convert_waypoint_to_pose(chosen_waypoint)
        self.goal_pose_pub.publish(goal_pose)
        
        # Compute and publish velocity command
        vel_cmd = self._compute_velocity_command(chosen_waypoint)
        self.vel_pub.publish(vel_cmd)
        
        # Publish node information
        self._publish_node_info()
        
        # Check if goal reached
        if self.closest_node == self.goal_node:
            self.reached_goal = True
            self._publish_zero_velocity()
        
        self.reach_goal_pub.publish(Bool(data=self.reached_goal))
        
        # Status output
        waypoint_str = f"[{chosen_waypoint[0]:.2f} {chosen_waypoint[1]:.2f}"
        if len(chosen_waypoint) > 2:
            waypoint_str += f" {chosen_waypoint[2]:.2f}"
        waypoint_str += "]"
        
        distance = np.linalg.norm(chosen_waypoint[:2])
        print(
            f"\r[Navigation] Node: {self.closest_node}/{self.goal_node} | "
            f"Waypoint: {waypoint_str} | "
            f"Vel: lin={vel_cmd.linear.x:.2f}m/s ang={vel_cmd.angular.z:.2f}rad/s | "
            f"Dist: {distance:.2f}m" + " "*20,
            end='', flush=True
        )


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(args: argparse.Namespace):
    """Main navigation loop"""
    print(f"🤖 Selected model: {args.model}")
    
    # Load model configuration
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)
    
    print(f"📋 Available models: {list(model_paths.keys())}")
    
    if args.model not in model_paths:
        raise ValueError(
            f"Model '{args.model}' not found in {MODEL_CONFIG_PATH}. "
            f"Available models: {list(model_paths.keys())}"
        )
    
    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)
    
    # Check and download model if necessary
    ckpth_path = model_paths[args.model]["ckpt_path"]
    if not ensure_model_exists(args.model, ckpth_path):
        raise FileNotFoundError(
            f"Failed to download or locate model weights for {args.model} at {ckpth_path}"
        )
    
    print(f"Loading model from {ckpth_path}")
    model = load_model(ckpth_path, model_params, device).to(device).eval()
    
    # Load topomap
    topomap_filenames = sorted(
        os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, TOPOMAP_NAME)),
        key=lambda x: int(x.split(".")[0]),
    )
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{TOPOMAP_NAME}"
    topomap = [PILImage.open(os.path.join(topomap_dir, fname)) for fname in topomap_filenames]
    num_nodes = len(topomap)
    
    # Determine start and goal nodes
    if args.start_node is not None or args.goal_node is not None:
        start_node = args.start_node if args.start_node is not None else NODE_RANGE[0]
        goal_node = args.goal_node if args.goal_node is not None else NODE_RANGE[1]
        goal_node = goal_node if goal_node != -1 else num_nodes - 1
    else:
        if NODE_RANGE[0] == -1 and NODE_RANGE[1] == -1:
            start_node = 0
            goal_node = num_nodes - 1
        else:
            start_node = NODE_RANGE[0]
            goal_node = NODE_RANGE[1] if NODE_RANGE[1] != -1 else num_nodes - 1
    
    assert 0 <= start_node < num_nodes, f"Invalid start index. Must be between 0 and {num_nodes-1}"
    assert 0 <= goal_node < num_nodes, f"Invalid goal index. Must be between 0 and {num_nodes-1}"
    assert start_node <= goal_node, "Start node must be <= goal node"
    
    args.start_node = start_node
    args.goal_node = goal_node
    
    print(f"🗺️ Navigation setup: Start node {start_node} → Goal node {goal_node} (Total: {num_nodes} nodes)")
    
    # Initialize ROS
    rclpy.init()
    nav_node = UnifiedNavigationNode(model, model_params, topomap, args)
    
    # Main navigation loop
    rate = nav_node.create_rate(RATE)
    
    try:
        while rclpy.ok():
            rclpy.spin_once(nav_node, timeout_sec=0)
            nav_node.navigation_step()
            rate.sleep()
    except KeyboardInterrupt:
        print("\n\n[Navigation] Interrupted by user")
    finally:
        nav_node._publish_zero_velocity()
        nav_node.destroy_node()
        rclpy.shutdown()
        print("\n[Navigation] Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified visual navigation with GNM/ViNT/NoMaD models"
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL,
        type=str,
        choices=["gnm", "vint", "nomad"],
        help="Model name to use for navigation (default: vint)",
    )
    parser.add_argument(
        "--waypoint", "-w",
        default=2,
        type=int,
        help="Index of waypoint used for navigation (default: 2)",
    )
    parser.add_argument(
        "--goal-node", "-g",
        default=None,
        type=int,
        help="Goal node index in the topomap (-1 for last node)",
    )
    parser.add_argument(
        "--start-node", "-s",
        default=None,
        type=int,
        help="Start node index in the topomap",
    )
    parser.add_argument(
        "--close-threshold", "-t",
        default=3,
        type=int,
        help="Temporal distance threshold for localizing to next node (default: 3)",
    )
    parser.add_argument(
        "--radius", "-r",
        default=4,
        type=int,
        help="Temporal radius of local nodes for localization (default: 4)",
    )
    parser.add_argument(
        "--num-samples", "-n",
        default=8,
        type=int,
        help="Number of action samples for NoMaD model (default: 8)",
    )
    
    args = parser.parse_args()
    print(f"Using device: {device}")
    main(args)
