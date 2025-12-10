import argparse
import os
import sys
import shutil
import time
import pickle
import yaml
import signal
import atexit

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

# ROS 2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger
from utils import msg_to_pil, compressed_msg_to_pil
import cv2

from scipy.spatial.transform import Rotation as R

IMAGE_TOPIC = "/camera/camera/color/image_raw"
USE_COMPRESSED = True  # Set to True to use compressed image topic
ODOM_TOPIC = "/odom"
TOPOMAP_IMAGES_DIR = "../../topomaps"
TOPOMAP_NAME = "gui2"
RECORD_PKL = False
AUTO_SHUTDOWN = False  # Whether to automatically shutdown when no image is received
TIMEOUT_DURATION = None  # Timeout duration in seconds (None means 2 * dt)

# Load camera calibration config from robot.yaml
ROBOT_CONFIG_PATH = "../../config/robot.yaml"
try:
    with open(ROBOT_CONFIG_PATH, "r") as f:
        robot_config = yaml.safe_load(f)
    CAMERA_CALIB_CONFIG = robot_config.get("camera_calibration", {})
    CAMERA_CALIB_PAN = CAMERA_CALIB_CONFIG.get("pan", 0.0)
    CAMERA_CALIB_TILT = CAMERA_CALIB_CONFIG.get("tilt", 0.0)
    CAMERA_CALIB_TIMEOUT = 10.0
except Exception as e:
    print(f"Warning: Could not load camera calibration config: {e}")
    CAMERA_CALIB_PAN = 0.0
    CAMERA_CALIB_TILT = 0.0
    CAMERA_CALIB_TIMEOUT = 10.0

# Global node reference for cleanup
topomap_node = None
shutdown_requested = False


class TopomapNode(Node):
    def __init__(self, image_topic, odom_topic, output_dir, dt, auto_shutdown=True, timeout_duration=None, use_compressed=False):
        super().__init__("create_topomap")
        self.image_topic = image_topic
        self.odom_topic = odom_topic
        self.output_dir = output_dir
        self.dt = dt
        self.auto_shutdown = auto_shutdown
        self.timeout_duration = timeout_duration if timeout_duration is not None else 2 * dt
        self.use_compressed = use_compressed
        self.obs_img = None
        self.last_odom = None
        self.traj_data = []

        # Create action client for head control
        self.head_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory'
        )
        
        # Create service clients for mode switching
        self.switch_to_navigation_mode_client = self.create_client(
            Trigger,
            '/switch_to_navigation_mode'
        )
        self.switch_to_gamepad_mode_client = self.create_client(
            Trigger,
            '/switch_to_gamepad_mode'
        )

        # Subscribe to compressed or raw image based on flag
        if self.use_compressed:
            self.sub_image = self.create_subscription(
                CompressedImage, self.image_topic + "/compressed", self.callback_compressed_image, 10
            )
            self.get_logger().info(f"Subscribed to {self.image_topic}/compressed (compressed)")
        else:
            self.sub_image = self.create_subscription(
                Image, self.image_topic, self.callback_image, 10
            )
            self.get_logger().info(f"Subscribed to {self.image_topic} (raw)")
            
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.callback_odom, 10
        )

        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.callback_odom, 10
        )

        self.get_logger().info(f"Subscribed to {self.odom_topic}")
        self.remove_files_in_dir(self.output_dir)
        self.get_logger().info(f"Saving data to {self.output_dir}")

        # Don't start timer immediately - wait for start_recording() call
        self.timer = None
        self.image_counter = 0
        self.start_time = float("inf")

    def remove_files_in_dir(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            for f in os.listdir(dir_path):
                file_path = os.path.join(dir_path, f)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    self.get_logger().error(f"Failed to delete {file_path}. Reason: {e}")

    def callback_image(self, msg: Image):
        self.obs_img = msg_to_pil(msg).rotate(270, expand=True)
    
    def callback_compressed_image(self, msg: CompressedImage):
        self.obs_img = compressed_msg_to_pil(msg).rotate(270, expand=True)

    def callback_odom(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        quat = [ori.x, ori.y, ori.z, ori.w]
        # Convert quaternion to euler angles using scipy
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=False)
        yaw = euler[2]  # yaw is the rotation around z-axis
        self.last_odom = {
            "x": pos.x,
            "y": pos.y,
            "z": pos.z,
            "yaw": yaw
        }

    def start_recording(self):
        """Start the timer to begin recording topomap"""
        if self.timer is None:
            self.timer = self.create_timer(self.dt, self.timer_callback)
            self.start_time = time.time()
            self.get_logger().info("✅ Recording timer started")
        else:
            self.get_logger().warn("⚠️  Recording timer already started")
    
    def timer_callback(self):
        global shutdown_requested
        
        # Stop timer if shutdown requested
        if shutdown_requested:
            return
        
        if self.obs_img is not None and self.last_odom is not None:
            img_path = os.path.join(self.output_dir, f"{self.image_counter}.png")
            self.obs_img.save(img_path)
            self.traj_data.append(self.last_odom)
            self.get_logger().info(f"Saved image {self.image_counter} to {img_path}")
            self.image_counter += 1
            self.start_time = time.time()
            self.obs_img = None

        elif self.auto_shutdown and time.time() - self.start_time > self.timeout_duration:
            self.get_logger().warn("No image received. Shutting down...")
            self.save_traj_data()
            rclpy.shutdown()
        elif not self.auto_shutdown and time.time() - self.start_time > self.timeout_duration:
            self.get_logger().warn(f"No image received for {self.timeout_duration:.1f} seconds, but continuing due to auto_shutdown=False")

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

    def switch_to_gamepad_mode(self, timeout_sec=5.0):
        """
        切換機器人到 gamepad mode (用於手動控制)
        
        Args:
            timeout_sec: Service 呼叫的超時時間
            
        Returns:
            bool: 成功返回 True，失敗返回 False
        """
        self.get_logger().info("[INFO] Switching robot to gamepad mode (gamepad control)...")
        
        # Wait for service to be available
        if not self.switch_to_gamepad_mode_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"[ERROR] Service /switch_to_gamepad_mode not available after {timeout_sec}s")
            return False
        
        # Call the service
        request = Trigger.Request()
        future = self.switch_to_gamepad_mode_client.call_async(request)
        
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            if future.result() is not None:
                response = future.result()
                if response.success:
                    self.get_logger().info(f"✅ Successfully switched to gamepad mode: {response.message}")
                    return True
                else:
                    self.get_logger().error(f"❌ Failed to switch to gamepad mode: {response.message}")
                    return False
            else:
                self.get_logger().error("[ERROR] Service call failed - no response received")
                return False
        except Exception as e:
            self.get_logger().error(f"[ERROR] Exception during service call: {e}")
            return False

    def calibrate_camera_position(self, pan=0.0, tilt=0.0, timeout_sec=10.0):
        """Calibrate camera position before recording topomap"""
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
        point.time_from_start.sec = 2
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

    def save_traj_data(self):
        if RECORD_PKL:
            output_pkl = os.path.join(self.output_dir, "traj_data.pkl")
            with open(output_pkl, "wb") as f:
                pickle.dump(self.traj_data, f)
            self.get_logger().info(f"Saved odometry data to {output_pkl}")
        else:
            self.get_logger().info("RECORD_PKL is False, skipping odometry data save")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global topomap_node, shutdown_requested
    shutdown_requested = True
    
    print("\n\n⚠️  Stopping topomap recording (Ctrl+C detected)...")
    
    if topomap_node is not None:
        try:
            print("   Saving trajectory data...")
            topomap_node.save_traj_data()
            print("   ✅ Data saved")
        except Exception as e:
            print(f"   ⚠️  Error saving data: {e}")
    
    print("   Cleaning up...\n")
    sys.exit(0)

def cleanup_resources():
    """Cleanup function called on exit"""
    global topomap_node
    
    if topomap_node is not None:
        try:
            topomap_node.save_traj_data()
            topomap_node.destroy_node()
        except:
            pass
    
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except:
        pass

def main():
    global topomap_node
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_resources)
    
    parser = argparse.ArgumentParser(
        description=f"Generate topomap images and odometry data from {IMAGE_TOPIC}"
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=0.1,
        type=float,
        help="Sampling period (default: 0.1 seconds)",
    )
    parser.add_argument(
        "--no-auto-shutdown",
        action="store_true",
        help="Disable automatic shutdown when no image is received (default: False)",
    )
    parser.add_argument(
        "--timeout",
        default=TIMEOUT_DURATION,
        type=float,
        help="Timeout duration in seconds before warning/shutdown (default: 2 * dt)",
    )
    args = parser.parse_args()

    try:
        rclpy.init()
    except:
        print("⚠️  ROS2 already initialized or initialization failed")
    
    topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, TOPOMAP_NAME)
    # Use global AUTO_SHUTDOWN as default, but allow command line override
    auto_shutdown = AUTO_SHUTDOWN and not args.no_auto_shutdown
    topomap_node = TopomapNode(
        IMAGE_TOPIC, 
        ODOM_TOPIC, 
        topomap_name_dir, 
        args.dt,
        auto_shutdown=auto_shutdown,
        timeout_duration=args.timeout,
        use_compressed=USE_COMPRESSED
    )
    
    # ========== Switch to Navigation Mode (MANDATORY) ==========
    print("🔄 Switching robot to navigation mode...")
    mode_switch_success = topomap_node.switch_to_navigation_mode(timeout_sec=5.0)
    
    if not mode_switch_success:
        print("❌ Failed to switch to navigation mode! Topomap recording aborted.")
        print("   Possible reasons:")
        print("   - Service /switch_to_navigation_mode not available")
        print("   - Robot driver (hellorobot_stretch3-driver) not running")
        print("   - Network communication issue between containers")
        topomap_node.destroy_node()
        rclpy.shutdown()
        return
    # ============================================================
    
    # ========== Camera Calibration (MANDATORY) ==========
    print(f"📷 Calibrating camera to topomap position (pan={CAMERA_CALIB_PAN:.2f}, tilt={CAMERA_CALIB_TILT:.2f})...")
    calibration_success = topomap_node.calibrate_camera_position(
        pan=CAMERA_CALIB_PAN,
        tilt=CAMERA_CALIB_TILT,
        timeout_sec=CAMERA_CALIB_TIMEOUT
    )
    
    if calibration_success:
        print("✅ Camera calibration completed successfully!")
        print("   Camera position verified - ready to record topomap")
    else:
        print("❌ Camera calibration FAILED! Topomap recording aborted.")
        print("   Possible reasons:")
        print("   - Action server (/stretch_controller/follow_joint_trajectory) not running")
        print("   - Camera joint is mechanically blocked or stuck")
        print("   - Network communication issue between containers")
        topomap_node.destroy_node()
        rclpy.shutdown()
        return
    # ====================================================
    
    # ========== Switch to gamepad Mode for Manual Control ==========
    print("🔄 Switching robot to gamepad mode for manual recording...")
    mode_switch_success = topomap_node.switch_to_gamepad_mode(timeout_sec=5.0)
    
    if not mode_switch_success:
        print("⚠️  Failed to switch to gamepad mode, but continuing anyway...")
        print("   You may need to manually switch modes using:")
        print("   ros2 service call /switch_to_gamepad_mode std_srvs/srv/Trigger {}")
    else:
        print("✅ Robot is now in position mode - ready for manual control")
    # =================================================================
    
    # ========== Wait before starting recording ==========
    print("⏳ Waiting 3 seconds before starting topomap recording...")
    time.sleep(3.0)
    # ====================================================
    
    # ========== Start recording timer NOW ==========
    print("🎬 Starting topomap recording now!")
    topomap_node.start_recording()  # This actually starts the timer
    # ===============================================
    
    try:
        print("📸 Recording in progress! Press Ctrl+C to stop.\n")
        rclpy.spin(topomap_node)
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt detected")
        topomap_node.save_traj_data()
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[Cleanup] Saving data and cleaning up...")
        try:
            topomap_node.save_traj_data()
            print("[Cleanup] ✅ Data saved")
        except:
            pass
        
        try:
            topomap_node.destroy_node()
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
    main()