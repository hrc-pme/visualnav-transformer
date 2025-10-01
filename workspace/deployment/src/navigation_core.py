#!/usr/bin/env python3
"""
Navigation Core Module
Provides ROS2-based navigation functionality for visual navigation models.
Used by GUI applications (Tkinter, PyQt5, Web) to interact with the robot.
"""

import os
import time
from typing import Optional, List
import numpy as np
import torch
import yaml
from PIL import Image as PILImage

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, Int32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

# Local imports
from topic_names import IMAGE_TOPIC, COMPRESSED_IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC, CMD_VEL_TOPIC, CURRENT_NODE_TOPIC, CANDIDATE_WAYPOINTS_TOPIC, CHOSEN_WAYPOINT_TOPIC, START_NODE_TOPIC, END_NODE_TOPIC
from utils import load_model, msg_to_pil, to_numpy, transform_images
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.training.train_utils import get_action


# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

# Load robot config
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]


class TrajectoryVisualizer:
    """Helper class for trajectory visualization"""
    def __init__(self):
        self.sampled_actions = None
        self.selected_action = None
        
    def update(self, sampled_actions: Optional[np.ndarray] = None, 
               selected_action: Optional[np.ndarray] = None):
        """Update visualization data"""
        if sampled_actions is not None:
            self.sampled_actions = sampled_actions
        if selected_action is not None:
            self.selected_action = selected_action
    
    def get_visualization_data(self):
        """Get current visualization data"""
        return {
            'sampled_actions': self.sampled_actions,
            'selected_action': self.selected_action
        }


class NavigationCore(Node):
    """
    Core navigation functionality for visual navigation models.
    Handles ROS2 communication, model inference, and navigation logic.
    """
    
    def __init__(self, init_rclpy: bool = True):
        """
        Initialize NavigationCore.
        
        Args:
            init_rclpy: If True, call rclpy.init() automatically. 
                       Set to False if rclpy is already initialized.
        """
        # Initialize rclpy if needed
        if init_rclpy and not rclpy.ok():
            rclpy.init()
        
        super().__init__('navigation_core')
        
        # State variables
        self.latest_image = None
        self.latest_twist = None
        self.context_queue = []
        self.current_node = -1  # Will be updated from /vn/node topic
        self.goal_node = -1
        self.start_node = -1  # Will be updated from /vn/start_node topic
        self.end_node = -1    # Will be updated from /vn/end_node topic
        self.reached_goal = False
        self.waypoint = None
        self.candidate_waypoints = None  # Candidate waypoints from navigate.ros2.py
        self.chosen_waypoint = None  # Chosen waypoint from navigate.ros2.py
        
        # Model parameters
        self.model = None
        self.model_params = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_scheduler = None
        self.topomap = []
        
        # Visualization
        self.visualizer = TrajectoryVisualizer()
        
        # ROS2 bridge
        self.bridge = CvBridge()
        
        # ROS2 publishers and subscribers (will be initialized later)
        self.waypoint_pub = None
        self.sampled_actions_pub = None
        self.reach_goal_pub = None
        
        self.get_logger().info(f"Navigation Core initialized on {self.device}")
    
    def initialize(self, topomap_dir: str, model_name: str = "vint", 
                   goal_node: int = -1, close_threshold: int = 3,
                   radius: int = 4, waypoint_idx: int = 2,
                   num_samples: int = 8) -> bool:
        """
        Initialize the navigation system with model and topomap.
        
        Args:
            topomap_dir: Directory containing topomap images
            model_name: Name of the model to use
            goal_node: Goal node index (-1 for last node)
            close_threshold: Distance threshold for node localization
            radius: Radius for local node search
            waypoint_idx: Index of waypoint to use for navigation
            num_samples: Number of action samples for diffusion models
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load model configuration
            with open(MODEL_CONFIG_PATH, "r") as f:
                model_paths = yaml.safe_load(f)
            
            if model_name not in model_paths:
                self.get_logger().error(f"Model {model_name} not found in config")
                return False
            
            model_config_path = model_paths[model_name]["config_path"]
            with open(model_config_path, "r") as f:
                self.model_params = yaml.safe_load(f)
            
            # Load model weights
            ckpt_path = model_paths[model_name]["ckpt_path"]
            if not os.path.exists(ckpt_path):
                self.get_logger().error(f"Model weights not found at {ckpt_path}")
                return False
            
            self.get_logger().info(f"Loading model from {ckpt_path}")
            self.model = load_model(ckpt_path, self.model_params, self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Initialize noise scheduler for diffusion models
            if self.model_params.get("model_type") == "nomad":
                self.noise_scheduler = DDPMScheduler(
                    num_train_timesteps=self.model_params["num_diffusion_iters"],
                    beta_schedule="squaredcos_cap_v2",
                    clip_sample=True,
                    prediction_type="epsilon"
                )
            
            # Load topomap
            # Try multiple possible paths
            possible_paths = [
                os.path.join(topomap_dir, "images"),  # path/to/topomap/images
                topomap_dir,                           # path/to/topomap (direct)
            ]
            
            topomap_images_dir = None
            for path in possible_paths:
                if os.path.exists(path):
                    # Check if there are image files
                    image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    if image_files:
                        topomap_images_dir = path
                        break
            
            if topomap_images_dir is None:
                self.get_logger().error(f"No topomap images found in {topomap_dir} or subdirectories")
                return False
            
            # Load images
            topomap_filenames = sorted(
                [f for f in os.listdir(topomap_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))],
                key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else 0
            )
            
            for filename in topomap_filenames:
                image_path = os.path.join(topomap_images_dir, filename)
                self.topomap.append(PILImage.open(image_path))
            
            self.get_logger().info(f"Loaded {len(self.topomap)} topomap images from {topomap_images_dir}")
            
            # Set goal node
            if goal_node == -1 and len(self.topomap) > 0:
                self.goal_node = len(self.topomap) - 1
            else:
                self.goal_node = goal_node
            
            # Navigation parameters
            self.close_threshold = close_threshold
            self.radius = radius
            self.waypoint_idx = waypoint_idx
            self.num_samples = num_samples
            self.context_size = self.model_params["context_size"]
            
            # Setup ROS2 publishers and subscribers
            self._setup_ros2()
            
            self.get_logger().info("Navigation Core initialized successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_ros2(self):
        """Setup ROS2 publishers and subscribers"""
        qos = QoSProfile(depth=10)
        
        # Publishers
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, qos)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, qos)
        self.reach_goal_pub = self.create_publisher(Bool, "/reach_goal", qos)
        
        # Subscribers - 使用壓縮影像以減少網路頻寬
        self.create_subscription(
            CompressedImage,  # 改用壓縮影像格式
            COMPRESSED_IMAGE_TOPIC, 
            self._compressed_image_callback,  # 使用新的壓縮影像回調函數
            qos_profile_sensor_data
        )
        
        # Subscribe to waypoint from navigate.ros2.py
        self.create_subscription(
            Float32MultiArray,
            WAYPOINT_TOPIC,
            self._waypoint_callback,
            qos
        )
        
        # Subscribe to cmd_vel from waypoint_to_goal_pose.py
        self.create_subscription(
            Twist,
            CMD_VEL_TOPIC,
            self._cmd_vel_callback,
            qos
        )
        
        # Subscribe to current_node from navigate.ros2.py
        self.create_subscription(
            Int32,
            CURRENT_NODE_TOPIC,
            self._current_node_callback,
            qos
        )
        
        # Subscribe to candidate_waypoints from navigate.ros2.py
        self.create_subscription(
            Float32MultiArray,
            CANDIDATE_WAYPOINTS_TOPIC,
            self._candidate_waypoints_callback,
            qos
        )
        
        # Subscribe to chosen_waypoint from navigate.ros2.py
        self.create_subscription(
            Float32MultiArray,
            CHOSEN_WAYPOINT_TOPIC,
            self._chosen_waypoint_callback,
            qos
        )
        
        # Subscribe to start_node from navigate.ros2.py
        self.create_subscription(
            Int32,
            START_NODE_TOPIC,
            self._start_node_callback,
            qos
        )
        
        # Subscribe to end_node from navigate.ros2.py
        self.create_subscription(
            Int32,
            END_NODE_TOPIC,
            self._end_node_callback,
            qos
        )
        
        self.get_logger().info(f"Subscribed to {COMPRESSED_IMAGE_TOPIC} (compressed)")
        self.get_logger().info(f"Subscribed to {WAYPOINT_TOPIC}")
        self.get_logger().info(f"Subscribed to {CMD_VEL_TOPIC}")
        self.get_logger().info(f"Subscribed to {CURRENT_NODE_TOPIC}")
        self.get_logger().info(f"Subscribed to {CANDIDATE_WAYPOINTS_TOPIC}")
        self.get_logger().info(f"Subscribed to {CHOSEN_WAYPOINT_TOPIC}")
        self.get_logger().info(f"Subscribed to {START_NODE_TOPIC}")
        self.get_logger().info(f"Subscribed to {END_NODE_TOPIC}")
    
    def _compressed_image_callback(self, msg: CompressedImage):
        """Callback for compressed camera images"""
        try:
            # 解壓縮影像 - 使用 cv2 直接從壓縮數據解碼
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error("Failed to decompress image")
                return
            
            # 轉換為 RGB (OpenCV 使用 BGR)
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # 轉換為 PIL Image
            obs_img = PILImage.fromarray(cv_image_rgb)
            
            # Rotate image (adjust as needed for your camera orientation)
            obs_img = obs_img.rotate(90, expand=True)
            
            # Update latest image (轉回 BGR 供顯示使用)
            self.latest_image = cv2.cvtColor(np.array(obs_img), cv2.COLOR_RGB2BGR)
            
            # Log first image reception
            if len(self.context_queue) == 0:
                self.get_logger().info("First compressed image received and decompressed!")
            
            # Update context queue
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)
                
        except Exception as e:
            self.get_logger().error(f"Compressed image callback error: {e}")
    
    def _image_callback(self, msg: Image):
        """Callback for camera images (原始未壓縮格式 - 已棄用，保留以備不時之需)"""
        try:
            # Convert ROS image to PIL
            obs_img = msg_to_pil(msg)
            
            # Rotate image (adjust as needed for your camera orientation)
            obs_img = obs_img.rotate(90, expand=True)
            
            # Update latest image
            self.latest_image = cv2.cvtColor(np.array(obs_img), cv2.COLOR_RGB2BGR)
            
            # Log first image reception
            if len(self.context_queue) == 0:
                self.get_logger().info("First image received!")
            
            # Update context queue
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)
                
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")
    
    def _waypoint_callback(self, msg: Float32MultiArray):
        """Callback for waypoint from navigate.ros2.py"""
        try:
            if len(msg.data) >= 2:
                self.waypoint = np.array(msg.data[:2])
        except Exception as e:
            self.get_logger().error(f"Waypoint callback error: {e}")
    
    def _cmd_vel_callback(self, msg: Twist):
        """Callback for cmd_vel from waypoint_to_goal_pose.py"""
        try:
            self.latest_twist = msg
        except Exception as e:
            self.get_logger().error(f"Cmd_vel callback error: {e}")
    
    def _current_node_callback(self, msg: Int32):
        """Callback for current_node from navigate.ros2.py"""
        try:
            self.current_node = msg.data
            # self.get_logger().info(f"Received current_node: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Current_node callback error: {e}")
    
    def _candidate_waypoints_callback(self, msg: Float32MultiArray):
        """Callback for candidate_waypoints from navigate.ros2.py"""
        try:
            # Reshape to (num_candidates, 2)
            data = np.array(msg.data)
            if len(data) > 0:
                self.candidate_waypoints = data.reshape(-1, 2)
        except Exception as e:
            self.get_logger().error(f"Candidate waypoints callback error: {e}")
    
    def _chosen_waypoint_callback(self, msg: Float32MultiArray):
        """Callback for chosen_waypoint from navigate.ros2.py"""
        try:
            if len(msg.data) >= 2:
                self.chosen_waypoint = np.array(msg.data[:2])
        except Exception as e:
            self.get_logger().error(f"Chosen waypoint callback error: {e}")
    
    def _start_node_callback(self, msg: Int32):
        """Callback for start_node from navigate.ros2.py"""
        try:
            self.start_node = msg.data
            # self.get_logger().info(f"Received start_node: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Start_node callback error: {e}")
    
    def _end_node_callback(self, msg: Int32):
        """Callback for end_node from navigate.ros2.py"""
        try:
            self.end_node = msg.data
            # self.get_logger().info(f"Received end_node: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"End_node callback error: {e}")
    
    def step(self):
        """
        Perform one navigation step.
        Processes current observations and publishes navigation commands.
        """
        if self.model is None or len(self.context_queue) <= self.context_size:
            return
        
        try:
            chosen_waypoint = np.zeros(2)
            
            if self.model_params["model_type"] == "nomad":
                chosen_waypoint = self._step_nomad()
            else:
                chosen_waypoint = self._step_gnm_vint()
            
            # Normalize and publish waypoint
            if self.model_params.get("normalize", False):
                chosen_waypoint *= MAX_V / RATE
            
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint.tolist()
            self.waypoint_pub.publish(waypoint_msg)
            self.waypoint = chosen_waypoint
            
            # Update goal status - 使用 end_node 來判斷是否到達終點
            reached = bool(self.current_node == self.end_node) if self.end_node >= 0 else False
            self.reach_goal_pub.publish(Bool(data=reached))
            self.reached_goal = reached
            
        except Exception as e:
            self.get_logger().error(f"Navigation step error: {e}")
    
    def _step_nomad(self) -> np.ndarray:
        """Navigation step for NoMaD model"""
        # Transform observations
        obs_images = transform_images(
            self.context_queue, 
            self.model_params["image_size"], 
            center_crop=False
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)
        
        mask = torch.zeros(1).long().to(self.device)
        
        # Get goal images in local radius
        start = max(self.current_node - self.radius, 0)
        end = min(self.current_node + self.radius + 1, self.goal_node)
        
        goal_images = [
            transform_images(g_img, self.model_params["image_size"], center_crop=False).to(self.device)
            for g_img in self.topomap[start:end + 1]
        ]
        goal_images = torch.concat(goal_images, dim=0)
        
        # Encode observations and goals
        with torch.no_grad():
            obsgoal_cond = self.model(
                "vision_encoder",
                obs_img=obs_images.repeat(len(goal_images), 1, 1, 1),
                goal_img=goal_images,
                input_goal_mask=mask.repeat(len(goal_images))
            )
            
            # Predict distances
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            
            # Find closest node
            min_idx = np.argmin(dists)
            # NOTE: current_node is now received from navigate.ros2.py via /vn/node topic
            # self.current_node = min_idx + start
            
            # Select subgoal
            sg_idx = min(
                min_idx + int(dists[min_idx] < self.close_threshold),
                len(obsgoal_cond) - 1
            )
            obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
            
            # Generate action samples
            obs_cond = obs_cond.repeat(self.num_samples, 1)
            
            # Run diffusion sampling
            with torch.no_grad():
                naction = torch.randn(
                    (self.num_samples, self.model_params["len_traj_pred"], 2),
                    device=self.device
                )
                self.noise_scheduler.set_timesteps(self.model_params["num_diffusion_iters"])
                
                for k in self.noise_scheduler.timesteps:
                    naction = self.noise_scheduler.scale_model_input(naction, k)
                    naction_pred = self.model(
                        "noise_pred_net",
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    naction = self.noise_scheduler.step(naction_pred, k, naction).prev_sample
            
            naction = to_numpy(get_action(naction))
            
            # Update visualizer
            self.visualizer.update(
                sampled_actions=naction,
                selected_action=naction[0]
            )
            
            # Select waypoint
            chosen_waypoint = naction[0][self.waypoint_idx]
        
        return chosen_waypoint
    
    def _step_gnm_vint(self) -> np.ndarray:
        """Navigation step for GNM/ViNT models"""
        # Transform observations
        obs_images = transform_images(
            self.context_queue,
            self.model_params["image_size"]
        ).to(self.device)
        
        # Get goal images in local radius
        start = max(self.current_node - self.radius, 0)
        end = min(self.current_node + self.radius + 1, self.goal_node)
        
        batch_obs = []
        batch_goal = []
        
        for goal_img in self.topomap[start:end + 1]:
            transf_obs = transform_images(
                self.context_queue,
                self.model_params["image_size"]
            )
            transf_goal = transform_images(
                goal_img,
                self.model_params["image_size"]
            )
            batch_obs.append(transf_obs)
            batch_goal.append(transf_goal)
        
        batch_obs = torch.cat(batch_obs, dim=0).to(self.device)
        batch_goal = torch.cat(batch_goal, dim=0).to(self.device)
        
        # Predict distances and waypoints
        with torch.no_grad():
            distances, waypoints = self.model(batch_obs, batch_goal)
            distances = to_numpy(distances)
            waypoints = to_numpy(waypoints)
        
        # Find closest node
        min_dist_idx = np.argmin(distances)
        
        # Choose subgoal and waypoint
        # NOTE: current_node is now received from navigate.ros2.py via /vn/node topic
        if distances[min_dist_idx] > self.close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][self.waypoint_idx]
            # self.current_node = start + min_dist_idx
        else:
            next_idx = min(min_dist_idx + 1, len(waypoints) - 1)
            chosen_waypoint = waypoints[next_idx][self.waypoint_idx]
            # self.current_node = min(start + min_dist_idx + 1, self.goal_node)
        
        # Update visualizer
        self.visualizer.update(selected_action=waypoints[min_dist_idx])
        
        return chosen_waypoint


def main():
    """Test function for NavigationCore"""
    rclpy.init()
    
    nav_core = NavigationCore()
    success = nav_core.initialize(
        topomap_dir="../topomaps/6e-elevator",
        model_name="vint"
    )
    
    if success:
        print("Navigation Core initialized successfully")
        
        # Spin and process navigation
        try:
            while rclpy.ok():
                rclpy.spin_once(nav_core, timeout_sec=0.1)
                nav_core.step()
                time.sleep(1.0 / RATE)
        except KeyboardInterrupt:
            print("\nShutting down...")
    
    nav_core.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
