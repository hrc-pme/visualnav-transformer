import argparse
import os
import sys
import time
import subprocess
import urllib.request
import shutil
import signal
import atexit

import numpy as np
import rclpy
import torch
import yaml
from cv_bridge import CvBridge
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from geometry_msgs.msg import Twist, PoseStamped
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray, Int32
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger
from topic_names import IMAGE_TOPIC, SAMPLED_ACTIONS_TOPIC, WAYPOINT_TOPIC, CURRENT_NODE_TOPIC, CANDIDATE_WAYPOINTS_TOPIC, CHOSEN_WAYPOINT_TOPIC, START_NODE_TOPIC, END_NODE_TOPIC
from utils import load_model, msg_to_pil, to_numpy, transform_images, compressed_msg_to_pil
from vint_train.training.train_utils import get_action
from scipy.spatial.transform import Rotation as R
import cv2

# ========== Check and install ROS dependencies ==========
try:
    from check_ros_dependencies import check_and_install_ros_packages, set_cyclonedds
    if check_and_install_ros_packages():
        set_cyclonedds()
except ImportError:
    print("⚠️  check_ros_dependencies.py not found, skipping dependency check")
except Exception as e:
    print(f"⚠️  Dependency check failed: {e}")
# ========================================================

# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps"
MODEL_WEIGHTS_PATH = "../model"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

# Load robot configuration
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

# Extract parameters from robot config
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1 / robot_config["frame_rate"]
WAYPOINT_CONTROL_RATE = 9  # Hz for waypoint control loop
EPS = 1e-8

# Navigation parameters from robot config
TOPOMAP_NAME = robot_config.get("topomap_name", "se1")
MODEL = robot_config.get("model", "vint")
NODE_RANGE = robot_config.get("node_range", [-1, -1])
REACH_TOLERANCE = robot_config.get("reach_tolerance", 5)  # Nodes before goal to consider reached
USE_MODEL_YAW = robot_config.get("use_model_yaw", True)  # Yaw control mode for GNM/ViNT

# Waypoint scaling parameters (scale model output waypoints)
WAYPOINT_XY_SCALE = robot_config.get("waypoint_xy_scale", 1.0)  # Scale waypoint distances
WAYPOINT_YAW_SCALE = robot_config.get("waypoint_yaw_scale", 1.0)  # Scale waypoint yaw angles

# PD Controller gains (control aggressiveness of waypoint following)
LINEAR_VEL_GAIN = robot_config.get("linear_vel_gain", 0.5)  # k_v: linear velocity gain
ANGULAR_VEL_GAIN = robot_config.get("angular_vel_gain", 1.0)  # k_w: angular velocity gain

# Camera calibration settings (for Stretch robot head)
# Camera calibration is MANDATORY - navigation will abort if it fails
CAMERA_CALIB_CONFIG = robot_config.get("camera_calibration", {})
CAMERA_CALIB_PAN = CAMERA_CALIB_CONFIG.get("pan", 0.0)
CAMERA_CALIB_TILT = CAMERA_CALIB_CONFIG.get("tilt", 0.0)
CAMERA_CALIB_TIMEOUT = 10.0  # Fixed timeout: 10 seconds (sufficient for head movement + communication)

# GLOBALS
context_queue = []
context_size = None
node = None
shutdown_requested = False

# Model download URLs
MODEL_URLS = {
    "gnm": "https://drive.google.com/file/d/1bzCPd_OsXjS2aGPTQladbI8ImxLZwrQh/view?usp=drive_link",
    "vint": "https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view?usp=drive_link", 
    "nomad": "https://drive.google.com/file/d/1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb/view?usp=drive_link"
}

def extract_google_drive_id(url):
    """從 Google Drive URL 提取檔案 ID"""
    if "drive.google.com" in url:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
    return None

def download_model_from_google_drive(file_id, destination):
    """從 Google Drive 下載模型檔案"""
    # 使用 gdown 下載 Google Drive 檔案
    try:
        import gdown
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading model to {destination}...")
        gdown.download(download_url, destination, quiet=False)
        return True
    except ImportError:
        print("gdown not found. Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            download_url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading model to {destination}...")
            gdown.download(download_url, destination, quiet=False)
            return True
        except Exception as e:
            print(f"Failed to install gdown or download file: {e}")
            return False
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False

def ensure_model_exists(model_name, model_path):
    """檢查模型檔案是否存在，如果不存在則下載"""
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
        return True
    
    print(f"Model {model_name} not found at {model_path}")
    
    # 確保模型目錄存在
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    if model_name in MODEL_URLS:
        file_id = extract_google_drive_id(MODEL_URLS[model_name])
        if file_id:
            print(f"Downloading {model_name} model...")
            if download_model_from_google_drive(file_id, model_path):
                print(f"Successfully downloaded {model_name} model")
                return True
            else:
                print(f"Failed to download {model_name} model")
                return False
        else:
            print(f"Invalid Google Drive URL for {model_name}")
            return False
    else:
        print(f"No download URL configured for model: {model_name}")
        return False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg):
    obs_img = compressed_msg_to_pil(msg).rotate(270, expand=True)

    if 'node' in globals():
        try:
            cv_img = np.array(obs_img)
            image_msg = node.bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
            image_msg.header.stamp = node.get_clock().now().to_msg()
            node.image_pub.publish(image_msg)
        except Exception as e:
            print(f"Failed to publish visualnav image: {e}")

    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


class NavigationNode(Node):
    def __init__(self):
        super().__init__("navigation_node")
        
        # ========== 系統診斷與清理 ==========
        self.log_system_status()
        self.cleanup_resources_before_start()
        # ====================================
        
        # Use explicit QoS profile for image subscription
        from rclpy.qos import ReliabilityPolicy, HistoryPolicy
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.create_subscription(CompressedImage, "/camera/camera/color/image_raw/compressed", callback_obs, image_qos)
        qos = QoSProfile(depth=10)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, qos)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, qos)
        self.candidate_waypoints_pub = self.create_publisher(Float32MultiArray, CANDIDATE_WAYPOINTS_TOPIC, qos)
        self.chosen_waypoint_pub = self.create_publisher(Float32MultiArray, CHOSEN_WAYPOINT_TOPIC, qos)
        self.image_pub = self.create_publisher(Image, "camera/image/visualnav", qos_profile_sensor_data)
        self.reach_goal_pub = self.create_publisher(Bool, "/reach_goal", qos)
        self.current_node_pub = self.create_publisher(Int32, CURRENT_NODE_TOPIC, qos)  # 添加 current node 發布器
        self.start_node_pub = self.create_publisher(Int32, START_NODE_TOPIC, qos)  # 添加 start node 發布器
        self.end_node_pub = self.create_publisher(Int32, END_NODE_TOPIC, qos)  # 添加 end node 發布器
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, qos)  # 添加速度控制發布器
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", qos)  # 添加 goal pose 發布器
        self.bridge = CvBridge()
        
        # Create action client for head control
        self.head_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory'
        )
        
        # Create service client for mode switching
        self.switch_to_navigation_mode_client = self.create_client(
            Trigger,
            '/switch_to_navigation_mode'
        )
        
        # Waypoint to goal pose conversion variables
        self.current_waypoint = None
        self.reached_goal = False
        
        # Diagnostics
        self.waypoint_count = 0
        self.last_diagnostic_time = time.time()
        
        # Create waypoint control timer
        self.create_timer(1.0 / WAYPOINT_CONTROL_RATE, self.waypoint_control_loop)
    
    def log_system_status(self):
        """診斷系統狀態"""
        try:
            import psutil
        except ImportError:
            self.get_logger().warn("psutil not installed, skipping RAM diagnostics")
            psutil = None
        
        self.get_logger().info("="*60)
        self.get_logger().info("🔍 System Diagnostics - Navigation Start")
        self.get_logger().info("="*60)
        
        # 1. GPU Memory
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**2
            gpu_mem_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
            self.get_logger().info(f"📊 GPU Memory:")
            self.get_logger().info(f"   - Allocated: {gpu_mem_allocated:.2f} MB")
            self.get_logger().info(f"   - Reserved:  {gpu_mem_reserved:.2f} MB")
            self.get_logger().info(f"   - Free:      {gpu_mem_free:.2f} MB")
            
            if gpu_mem_allocated > 100:
                self.get_logger().warn(f"⚠️  GPU memory already occupied: {gpu_mem_allocated:.2f} MB")
                self.get_logger().warn("   This may indicate previous run didn't clean up properly")
        else:
            self.get_logger().info("📊 GPU: Not available (using CPU)")
        
        # 2. CPU Memory
        if psutil:
            ram = psutil.virtual_memory()
            self.get_logger().info(f"💾 RAM Usage: {ram.percent:.1f}% ({ram.used / 1024**3:.2f} GB / {ram.total / 1024**3:.2f} GB)")
            if ram.percent > 80:
                self.get_logger().warn(f"⚠️  High RAM usage: {ram.percent:.1f}%")
        
        # 3. ROS2 Nodes
        try:
            node_names = self.get_node_names()
            self.get_logger().info(f"🤖 Active ROS2 Nodes: {len(node_names)}")
            if len(node_names) > 20:
                self.get_logger().warn(f"⚠️  Many nodes active ({len(node_names)}), may have residual nodes")
        except Exception as e:
            self.get_logger().warn(f"Cannot get node list: {e}")
        
        # 4. Context Queue
        self.get_logger().info(f"📸 Context Queue: {len(context_queue)} / {context_size}")
        if len(context_queue) > 0:
            self.get_logger().warn("⚠️  Context queue not empty! Old data present.")
        
        self.get_logger().info("="*60)
    
    def cleanup_resources_before_start(self):
        """在開始前清理資源"""
        self.get_logger().info("🧹 Cleaning up resources before navigation...")
        
        # 1. Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.get_logger().info("   ✅ GPU cache cleared")
        
        # 2. Force garbage collection
        import gc
        gc.collect()
        self.get_logger().info("   ✅ Garbage collection completed")
        
        # 3. Clear context queue
        global context_queue
        context_queue.clear()
        self.get_logger().info("   ✅ Context queue cleared")
        
        # Small delay to ensure cleanup
        time.sleep(0.5)
        
        self.get_logger().info("🧹 Resource cleanup completed")
        
    def publish_zero_velocity(self):
        """發布零速度指令以停止機器人"""
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.linear.y = 0.0
        stop_cmd.linear.z = 0.0
        stop_cmd.angular.x = 0.0
        stop_cmd.angular.y = 0.0
        stop_cmd.angular.z = 0.0
        self.vel_pub.publish(stop_cmd)
        self.get_logger().info("Published zero velocity command to stop robot")
    
    def switch_to_navigation_mode(self, timeout_sec=5.0):
        """
        切換機器人到 navigation mode
        
        Args:
            timeout_sec: Service 呼叫的超時時間
            
        Returns:
            bool: 成功返回 True，失敗返回 False
        """
        self.get_logger().info("[INFO] Switching robot to navigation mode...")
        
        # Wait for service to be available
        if not self.switch_to_navigation_mode_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"[ERROR] Service /switch_to_navigation_mode not available after {timeout_sec}s")
            return False
        
        # Call the service
        request = Trigger.Request()
        future = self.switch_to_navigation_mode_client.call_async(request)
        
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            if future.result() is not None:
                response = future.result()
                if response.success:
                    self.get_logger().info(f"✅ Successfully switched to navigation mode: {response.message}")
                    return True
                else:
                    self.get_logger().error(f"❌ Failed to switch to navigation mode: {response.message}")
                    return False
            else:
                self.get_logger().error("[ERROR] Service call failed - no response received")
                return False
        except Exception as e:
            self.get_logger().error(f"[ERROR] Exception during service call: {e}")
            return False
    
    def calibrate_camera_position(self, pan=0.0, tilt=0.0, timeout_sec=5.0):
        """
        校正相機位置到指定角度
        
        Args:
            pan: 水平角度 (joint_head_pan) in radians
            tilt: 垂直角度 (joint_head_tilt) in radians  
            timeout_sec: Action 執行的超時時間
            
        Common presets:
            - ahead: pan=0.0, tilt=0.0
            - up: pan=0.0, tilt=0.52
            - down: pan=0.0, tilt=-0.79
            - left: pan=1.57, tilt=0.0
            - right: pan=-1.57, tilt=0.0
        """
        self.get_logger().info(f"Calibrating camera to pan={pan:.2f}, tilt={tilt:.2f}")
        
        # Wait for action server - keep trying until available (no timeout)
        self.get_logger().info("Waiting for head control action server...")
        self.get_logger().info("⚠️  If robot is powered off, this will wait indefinitely until you power it on")
        
        while not self.head_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Head control action server not available, retrying...")
            self.get_logger().warn("   Please ensure robot is powered ON and driver is running")
            time.sleep(2.0)
        
        self.get_logger().info("✅ Action server connected!")
        
        # Small delay to ensure connection is stable
        time.sleep(0.5)
        
        # Create goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint_head_pan', 'joint_head_tilt']
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [float(pan), float(tilt)]
        point.time_from_start.sec = 2  # 2 seconds to reach position
        point.time_from_start.nanosec = 0
        
        goal_msg.trajectory.points = [point]
        
        # Send goal and wait for result
        self.get_logger().info("Sending camera calibration goal...")
        try:
            future = self.head_action_client.send_goal_async(goal_msg)
            self.get_logger().info("Goal sent, waiting for acceptance...")
            
            # Wait for goal to be accepted
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            
            if not future.done():
                self.get_logger().error("Failed to send goal - future not completed!")
                return False
            
            goal_handle = future.result()
            if goal_handle is None:
                self.get_logger().error("Goal handle is None - failed to send goal!")
                return False
                
            if not goal_handle.accepted:
                self.get_logger().error("Camera calibration goal rejected!")
                return False
            
            self.get_logger().info("Camera calibration goal accepted, waiting for result...")
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
            
        except Exception as e:
            self.get_logger().error(f"Exception during camera calibration: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check if we got a result (not timed out)
        if not result_future.done():
            self.get_logger().error(f"Camera calibration timed out after {timeout_sec} seconds!")
            return False
        
        result = result_future.result()
        
        # Check the status code
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("✅ Camera calibration completed successfully!")
            return True
        elif result.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(f"❌ Camera calibration ABORTED by server! (status={result.status})")
            return False
        elif result.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error(f"❌ Camera calibration CANCELED! (status={result.status})")
            return False
        else:
            self.get_logger().error(f"❌ Camera calibration failed with status: {result.status}")
            return False
    
    def convert_waypoint_pose(self, waypoint: np.ndarray) -> PoseStamped:
        """將 waypoint 轉換為 PoseStamped 消息"""
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
    
    def waypoint_control_loop(self):
        """Waypoint to velocity control loop"""
        global shutdown_requested
        
        # Check if shutdown was requested
        if shutdown_requested:
            self.publish_zero_velocity()
            self.current_waypoint = None
            return
        
        # Periodic diagnostics (every 30 seconds)
        self.waypoint_count += 1
        if time.time() - self.last_diagnostic_time > 30.0:
            self.periodic_diagnostic()
            self.last_diagnostic_time = time.time()
        
        if self.reached_goal:
            # 目標達成，發布停止訊號並清除航點
            self.publish_zero_velocity()
            self.current_waypoint = None  # 清除航點以避免繼續處理
            print(f"\r[Waypoint2Goal] 🎯 GOAL REACHED! Robot stopped." + " "*80, end='', flush=True)
            return

        if self.current_waypoint is not None:
            # 發佈目標位姿
            goal_pose = self.convert_waypoint_pose(self.current_waypoint)
            self.goal_pub.publish(goal_pose)

            # 計算速度指令（簡單比例控制器）
            x, y = self.current_waypoint[0], self.current_waypoint[1]
            distance = np.linalg.norm([x, y])
            
            # 根據配置選擇 yaw 計算方式
            if len(self.current_waypoint) > 2 and USE_MODEL_YAW:
                # 使用模型預測的軌跡切線方向 (GNM/ViNT with use_model_yaw=true)
                target_angle = self.current_waypoint[2]
            else:
                # 使用指向 waypoint 的方向 (NoMaD 或 use_model_yaw=false)
                target_angle = np.arctan2(y, x)

            # PD 控制器：使用配置的增益
            linear_vel = min(LINEAR_VEL_GAIN * distance, MAX_V)
            angular_vel = np.clip(ANGULAR_VEL_GAIN * target_angle, -MAX_W, MAX_W)

            twist = Twist()
            twist.linear.x = float(linear_vel)
            twist.angular.z = float(angular_vel)

            self.vel_pub.publish(twist)
            # 使用 \r 清除同行並輸出新的狀態訊息，加上空格填充以清除舊內容
            # print(f"\r[Waypoint2Goal] Vel: lin={linear_vel:.2f}m/s, ang={angular_vel:.2f}rad/s | Waypoint: [{x:.2f}, {y:.2f}] | Dist: {distance:.2f}m" + " "*20, end='', flush=True)

        else:
            # 靜默等待 waypoint
            pass
    
    def periodic_diagnostic(self):
        """定期診斷系統狀態"""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            if gpu_mem > 500:  # More than 500 MB
                self.get_logger().warn(f"⚠️  High GPU usage: {gpu_mem:.2f} MB")
        
        self.get_logger().info(f"🔄 Waypoint count: {self.waypoint_count} | Context queue: {len(context_queue)}")

def emergency_stop_handler(sig, frame):
    """Emergency stop handler for Ctrl+C"""
    global node, shutdown_requested
    shutdown_requested = True
    print("\n\n⚠️  Emergency stop triggered (Ctrl+C detected)!")
    print("   Stopping robot and cleaning up resources...")
    
    if node is not None:
        try:
            # Stop robot immediately
            node.publish_zero_velocity()
            print("   ✅ Robot stopped (zero velocity published)")
            
            # Give time for message to be sent
            time.sleep(0.3)
            node.publish_zero_velocity()  # Send twice to ensure delivery
            time.sleep(0.2)
            
            # Mark as reached goal to stop waypoint loop
            node.reached_goal = True
            
        except Exception as e:
            print(f"   ⚠️  Error during emergency stop: {e}")
    
    print("   Exiting...\n")
    sys.exit(0)

def cleanup_resources():
    """Cleanup function called on exit"""
    global node
    if node is not None:
        try:
            node.publish_zero_velocity()
            node.destroy_node()
        except:
            pass
    
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except:
        pass

def main(args: argparse.Namespace):
    global context_size, node

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, emergency_stop_handler)
    signal.signal(signal.SIGTERM, emergency_stop_handler)
    
    # Register cleanup function
    atexit.register(cleanup_resources)
    
    # ========== Pre-execution cleanup ==========
    print("🧹 Pre-execution cleanup...")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("   ✅ GPU cache cleared")
    
    # Force garbage collection
    import gc
    gc.collect()
    print("   ✅ Garbage collection completed")
    
    # Clear context queue
    global context_queue
    context_queue.clear()
    print("   ✅ Context queue cleared")
    
    print("🧹 Pre-execution cleanup completed\n")
    # ===========================================

    print(f"🤖 Selected model: {args.model}")
    print(f"🎯 Yaw control mode: {'Model prediction' if USE_MODEL_YAW else 'Point-to-point (arctan2)'}")
    print(f"🎯 Reach tolerance: {REACH_TOLERANCE} nodes before goal")
    
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    print(f"📋 Available models: {list(model_paths.keys())}")
    
    if args.model not in model_paths:
        raise ValueError(f"Model '{args.model}' not found in {MODEL_CONFIG_PATH}. Available models: {list(model_paths.keys())}")

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    ckpth_path = model_paths[args.model]["ckpt_path"]
    
    # 檢查並下載模型（如果需要）
    if not ensure_model_exists(args.model, ckpth_path):
        raise FileNotFoundError(f"Failed to download or locate model weights for {args.model} at {ckpth_path}")
    
    if not os.path.exists(ckpth_path):
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    print(f"Loading model from {ckpth_path}")

    model = load_model(ckpth_path, model_params, device).to(device).eval()

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    topomap_filenames = sorted(
        os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, TOPOMAP_NAME)),
        key=lambda x: int(x.split(".")[0]),
    )
    topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{TOPOMAP_NAME}"
    topomap = [PILImage.open(os.path.join(topomap_dir, fname)) for fname in topomap_filenames]
    num_nodes = len(topomap)
    
    # Determine start and goal nodes from NODE_RANGE or command line arguments
    if args.start_node is not None or args.goal_node is not None:
        # Use command line arguments if provided, otherwise use NODE_RANGE defaults
        start_node = args.start_node if args.start_node is not None else NODE_RANGE[0]
        goal_node = args.goal_node if args.goal_node is not None else NODE_RANGE[1]
        goal_node = goal_node if goal_node != -1 else num_nodes - 1
    else:
        # Use NODE_RANGE global settings
        if NODE_RANGE[0] == -1 and NODE_RANGE[1] == -1:
            # Auto-detect full range
            start_node = 0
            goal_node = num_nodes - 1
        else:
            start_node = NODE_RANGE[0]
            goal_node = NODE_RANGE[1] if NODE_RANGE[1] != -1 else num_nodes - 1
    
    assert 0 <= start_node < num_nodes, f"Invalid start index. Must be between 0 and {num_nodes-1}"
    assert 0 <= goal_node < num_nodes, f"Invalid goal index. Must be between 0 and {num_nodes-1}"
    assert start_node <= goal_node, "Start node must be <= goal node"
    
    print(f"🗺️ Navigation setup: Start node {start_node} → Goal node {goal_node} (Total: {num_nodes} nodes)")

    try:
        rclpy.init()
    except:
        print("⚠️  ROS2 already initialized or initialization failed")
    
    node = NavigationNode()
    
    # 發布 start 和 end node 資訊
    start_node_msg = Int32()
    start_node_msg.data = int(start_node)
    node.start_node_pub.publish(start_node_msg)
    
    end_node_msg = Int32()
    end_node_msg.data = int(goal_node)
    node.end_node_pub.publish(end_node_msg)
    
    # ========== 切換到 Navigation Mode (MANDATORY) ==========
    print("🔄 Switching robot to navigation mode...")
    mode_switch_success = node.switch_to_navigation_mode(timeout_sec=5.0)
    
    if not mode_switch_success:
        print("❌ Failed to switch to navigation mode! Navigation aborted.")
        print("   Possible reasons:")
        print("   - Service /switch_to_navigation_mode not available")
        print("   - Robot driver (hellorobot_stretch3-driver) not running")
        print("   - Network communication issue between containers")
        rclpy.shutdown()
        return
    # ========================================================
    
    # ========== 相機校正 (MANDATORY) ==========
    # Camera MUST be calibrated before navigation starts
    print(f"📷 Calibrating camera to navigation position (pan={CAMERA_CALIB_PAN:.2f}, tilt={CAMERA_CALIB_TILT:.2f})...")
    calibration_success = node.calibrate_camera_position(
        pan=CAMERA_CALIB_PAN,
        tilt=CAMERA_CALIB_TILT,
        timeout_sec=CAMERA_CALIB_TIMEOUT
    )
    
    if calibration_success:
        print("✅ Camera calibration completed successfully!")
        print("   Camera position verified by joint trajectory controller")
    else:
        print("❌ Camera calibration FAILED! Navigation aborted.")
        print("   Possible reasons:")
        print("   - Action server (/stretch_controller/follow_joint_trajectory) not running")
        print("   - Camera joint is mechanically blocked or stuck")
        print("   - Network communication issue between containers")
        print("   - Joint controller timeout (movement took > 10 seconds)")
        rclpy.shutdown()
        return
    # ==========================================

    
    closest_node = start_node  # 從指定的起始節點開始
    reached_goal = False
    start, end = -1, -1
    
    # 記錄收到影像的時間（用於計算延遲）
    last_image_time = time.time()

    try:
        while rclpy.ok() and not shutdown_requested:
            loop_start_time = time.time()
            chosen_waypoint = np.zeros(4)
            processing_delay = 0.0  # 初始化處理延遲

            # 如果已經到達目標，發布停止信號並退出
            if reached_goal:
                node.publish_zero_velocity()
                node.reach_goal_pub.publish(Bool(data=True))
                print("\n[Navigation] 🎯 Goal reached! Robot stopped. Exiting...")
                break

            if len(context_queue) > model_params["context_size"]:
                # 記錄開始推理的時間
                inference_start_time = time.time()
                
                start = max(closest_node - args.radius, start_node)  # 不能小於起始節點
                end = min(closest_node + args.radius + 1, goal_node)

                if model_params["model_type"] == "nomad":
                    obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
                    obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1).to(device)
                    mask = torch.zeros(1).long().to(device)

                    goal_image = [transform_images(img, model_params["image_size"], center_crop=False).to(device)
                                for img in topomap[start:end+1]]
                    goal_image = torch.cat(goal_image, dim=0)

                    obsgoal_cond = model("vision_encoder", obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                                        goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                    dists = to_numpy(model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten())
                    min_idx = np.argmin(dists)
                    closest_node = min_idx + start

                    sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                    obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                    naction = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    for k in noise_scheduler.timesteps:
                        noise_pred = model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)
                        naction = noise_scheduler.step(noise_pred, k, naction).prev_sample

                    naction = to_numpy(get_action(naction))
                    node.sampled_actions_pub.publish(Float32MultiArray(data=np.concatenate(([0], naction.flatten())).tolist()))
                    
                    # Publish candidate waypoints for NOMAD (all samples at the chosen waypoint index)
                    candidate_wps = naction[:, args.waypoint, :]  # shape: (num_samples, 2)
                    node.candidate_waypoints_pub.publish(Float32MultiArray(data=candidate_wps.flatten().tolist()))
                    
                    chosen_waypoint = naction[0][args.waypoint]
                    
                    # Publish chosen waypoint
                    node.chosen_waypoint_pub.publish(Float32MultiArray(data=chosen_waypoint.tolist()))

                else:
                    batch_obs_imgs = [transform_images(context_queue, model_params["image_size"]) for _ in range(end - start + 1)]
                    batch_goal_data = [transform_images(topomap[i], model_params["image_size"]) for i in range(start, end + 1)]
                    batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                    batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                    distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                    distances = to_numpy(distances)
                    waypoints = to_numpy(waypoints)

                    min_dist_idx = np.argmin(distances)
                    
                    # Publish candidate waypoints for GNM/ViNT (all candidate nodes at the chosen waypoint index)
                    candidate_wps = waypoints[:, args.waypoint, :]  # shape: (num_candidates, 2)
                    node.candidate_waypoints_pub.publish(Float32MultiArray(data=candidate_wps.flatten().tolist()))
                    
                    if distances[min_dist_idx] > args.close_threshold:
                        chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                        closest_node = start + min_dist_idx
                    else:
                        cur_wp = waypoints[min_dist_idx][args.waypoint]
                        next_idx = min(min_dist_idx + 1, len(waypoints) - 1)
                        next_wp = waypoints[next_idx][args.waypoint]

                        delta = np.abs(np.array(cur_wp) - np.array(next_wp))
                        if np.any(delta < 0.25):
                            chosen_waypoint = next_wp
                            closest_node = min(start + min_dist_idx + 1, goal_node)
                        else:
                            chosen_waypoint = cur_wp
                            closest_node = start + min_dist_idx
                    
                    # Publish chosen waypoint
                    node.chosen_waypoint_pub.publish(Float32MultiArray(data=chosen_waypoint.tolist()))
                
                # 計算處理延遲
                inference_end_time = time.time()
                processing_delay = inference_end_time - inference_start_time

                # Normalize and scale waypoint distances (model output → actual distances)
            if model_params["normalize"]:
                # Scale XY waypoint distances
                chosen_waypoint[0:2] *= MAX_V / RATE * WAYPOINT_XY_SCALE
                # Scale yaw angle (only if waypoint has 3+ dimensions)
                if len(chosen_waypoint) > 2:
                    chosen_waypoint[2] *= MAX_W / RATE * WAYPOINT_YAW_SCALE

            waypoint_msg = Float32MultiArray(data=chosen_waypoint.tolist())
            node.waypoint_pub.publish(waypoint_msg)
            
            # 更新 current_waypoint 供 waypoint control loop 使用
            node.current_waypoint = chosen_waypoint
            
            # 發布 current node
            current_node_msg = Int32()
            current_node_msg.data = int(closest_node)
            node.current_node_pub.publish(current_node_msg)
            
            # 檢查是否到達目標 (考慮容忍度)
            # 如果當前節點距離目標節點在容忍範圍內，則視為已到達
            goal_reached = bool(closest_node >= goal_node - REACH_TOLERANCE)
            node.reach_goal_pub.publish(Bool(data=goal_reached))
            
            # 更新 reached_goal 狀態供 waypoint control loop 使用
            node.reached_goal = goal_reached

            # 持續發布 start 和 end node 資訊（確保GUI能收到）
            start_node_msg = Int32()
            start_node_msg.data = int(start_node)
            node.start_node_pub.publish(start_node_msg)
            
            end_node_msg = Int32()
            end_node_msg.data = int(goal_node)
            node.end_node_pub.publish(end_node_msg)
            
            # 改善的輸出訊息格式
            if len(context_queue) > model_params["context_size"]:
                # Format waypoint based on its dimensions
                if len(chosen_waypoint) == 2:
                    waypoint_str = f"[{chosen_waypoint[0]:6.2f}, {chosen_waypoint[1]:6.2f}]"
                else:
                    waypoint_str = f"[{chosen_waypoint[0]:6.2f}, {chosen_waypoint[1]:6.2f}, {chosen_waypoint[2]:6.2f}]"
                print(f"[{args.model}] Node: {closest_node:03d}/{goal_node:03d} | Waypoint: {waypoint_str} | Delay: {processing_delay:.3f}s")

            if goal_reached:
                print(f"\n[{args.model}] 🎯 GOAL REACHED! Navigation complete.")
                # 設置 reached_goal 為 True，下一次循環會退出
                reached_goal = True

            time.sleep(max(0, (1.0 / RATE) - (time.time() - loop_start_time)))
            rclpy.spin_once(node, timeout_sec=0)
    
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt in main loop")
    except Exception as e:
        print(f"\n❌ Error in navigation loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure robot is stopped
        print("\n[Cleanup] Stopping robot and cleaning up...")
        if node is not None:
            node.publish_zero_velocity()
            time.sleep(0.3)
            node.publish_zero_velocity()  # Send twice
            time.sleep(0.2)
            
            try:
                node.destroy_node()
                print("[Cleanup] ✅ Node destroyed")
            except:
                pass
        
        try:
            if rclpy.ok():
                rclpy.shutdown()
                print("[Cleanup] ✅ ROS2 shutdown complete")
        except:
            pass
        
        print("[Cleanup] ✅ All resources cleaned up\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default=MODEL,
        type=str,
        choices=["gnm", "vint", "nomad"],
        help="model name to use for navigation (gnm/vint/nomad) (check config/models.yaml for available models) (default: vint)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=None,
        type=int,
        help=f"""goal node index in the topomap (if -1, then the goal node is
        the last node in the topomap) (default: use NODE_RANGE[1]={NODE_RANGE[1]})""",
    )
    parser.add_argument(
        "--start-node",
        "-s",
        default=None,
        type=int,
        help=f"""start node index in the topomap (index of the image to start navigation from) (default: use NODE_RANGE[0]={NODE_RANGE[0]})""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
