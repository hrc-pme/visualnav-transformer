#!/usr/bin/env python3
"""
ROS2 Dataset Recording Script

This script records ROS2 bag files for training the Visual Navigation Transformer.
Configure settings in data.yaml before running.
"""

import sys
import os

# ========== Check and install ROS dependencies ==========
try:
    # Add deploy/src to path to import check_ros_dependencies
    deploy_src_path = os.path.join(os.path.dirname(__file__), '../../deploy/src')
    if os.path.exists(deploy_src_path):
        sys.path.insert(0, deploy_src_path)
    
    from check_ros_dependencies import check_and_install_ros_packages, set_cyclonedds
    if check_and_install_ros_packages():
        set_cyclonedds()
except ImportError:
    print("⚠️  check_ros_dependencies.py not found, skipping dependency check")
except Exception as e:
    print(f"⚠️  Dependency check failed: {e}")
# ========================================================

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import subprocess
import time
from datetime import datetime
import yaml
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from action_msgs.msg import GoalStatus
from std_srvs.srv import Trigger

# ============================================================================
# Load Configuration from YAML
# ============================================================================

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "data.yaml")

# Load configuration
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_PATH}")
    print("Please create data.yaml with required settings.")
    exit(1)

# Extract configuration variables
DATASET_NAME = config['dataset']['name']
IMAGE_TOPIC = config['topics']['image']
POSE_TOPIC = config['topics']['pose']
ADDITIONAL_TOPICS = config['topics'].get('additional', [])

COMPRESSION_MODE = config['recording']['compression_mode']
COMPRESSION_FORMAT = config['recording']['compression_format']
STORAGE_TYPE = config['recording']['storage_type']

# ROS2 Bag output directory
BAG_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "../datasets",
    DATASET_NAME,
    "rosbags"
)

# Camera calibration settings (try to load from robot.yaml in deploy folder)
try:
    # Try to find robot.yaml in the deploy workspace
    deploy_config_path = os.path.join(SCRIPT_DIR, "../../deploy/config/robot.yaml")
    if os.path.exists(deploy_config_path):
        with open(deploy_config_path, 'r') as f:
            robot_config = yaml.safe_load(f)
        CAMERA_CALIB_CONFIG = robot_config.get("camera_calibration", {})
        CAMERA_CALIB_PAN = CAMERA_CALIB_CONFIG.get("pan", 0.0)
        CAMERA_CALIB_TILT = CAMERA_CALIB_CONFIG.get("tilt", 0.0)
    else:
        # Use defaults if robot.yaml not found
        CAMERA_CALIB_PAN = 0.0
        CAMERA_CALIB_TILT = 0.0
except Exception:
    CAMERA_CALIB_PAN = 0.0
    CAMERA_CALIB_TILT = 0.0

CAMERA_CALIB_TIMEOUT = 10.0

# Global references for cleanup
recorder_node = None
recording_process = None
shutdown_requested = False

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


class DatasetRecorder(Node):
    """Node for recording ROS2 bag files for dataset collection."""
    
    def __init__(self):
        super().__init__('dataset_recorder')
        
        self.recording_process = None
        
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
            '/switch_to_position_mode'
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(BAG_OUTPUT_DIR, exist_ok=True)
        
        # Generate bag file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bag_name = f"{DATASET_NAME}_{timestamp}"
        self.bag_path = os.path.join(BAG_OUTPUT_DIR, self.bag_name)
        
        self.get_logger().info(f"Dataset Recorder initialized")
        self.get_logger().info(f"Bag output path: {self.bag_path}")
    
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
        切換機器人到 position mode (用於手動控制)
        
        Args:
            timeout_sec: Service 呼叫的超時時間
            
        Returns:
            bool: 成功返回 True，失敗返回 False
        """
        self.get_logger().info("[INFO] Switching robot to position mode (gamepad control)...")
        
        # Wait for service to be available
        if not self.switch_to_gamepad_mode_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"[ERROR] Service /switch_to_position_mode not available after {timeout_sec}s")
            return False
        
        # Call the service
        request = Trigger.Request()
        future = self.switch_to_gamepad_mode_client.call_async(request)
        
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
            if future.result() is not None:
                response = future.result()
                if response.success:
                    self.get_logger().info(f"✅ Successfully switched to position mode: {response.message}")
                    return True
                else:
                    self.get_logger().error(f"❌ Failed to switch to position mode: {response.message}")
                    return False
            else:
                self.get_logger().error("[ERROR] Service call failed - no response received")
                return False
        except Exception as e:
            self.get_logger().error(f"[ERROR] Exception during service call: {e}")
            return False
    
    def calibrate_camera_position(self, pan=0.0, tilt=0.0, timeout_sec=10.0):
        """Calibrate camera position before recording dataset"""
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
        
    def start_recording(self):
        """Start recording ROS2 bag."""
        global recording_process
        
        # Build topic list
        topics = [IMAGE_TOPIC, POSE_TOPIC] + ADDITIONAL_TOPICS
        topics_str = " ".join(topics)
        
        # Build ros2 bag record command
        cmd = [
            "ros2", "bag", "record",
            "-o", self.bag_path,
            "--storage", STORAGE_TYPE,
        ]
        
        if COMPRESSION_MODE != "none":
            cmd.extend(["--compression-mode", COMPRESSION_MODE])
            cmd.extend(["--compression-format", COMPRESSION_FORMAT])
        
        cmd.extend(topics)
        
        self.get_logger().info(f"Starting recording with topics: {topics}")
        self.get_logger().info(f"Command: {' '.join(cmd)}")
        self.get_logger().info("Press Ctrl+C to stop recording")
        
        try:
            # Use Popen instead of run for better control
            self.recording_process = subprocess.Popen(cmd)
            recording_process = self.recording_process
            
            # Wait for process to complete
            self.recording_process.wait()
            
        except KeyboardInterrupt:
            self.get_logger().info("\n⚠️  Recording interrupted by user")
            self.stop_recording()
        except Exception as e:
            self.get_logger().error(f"Recording failed: {e}")
            self.stop_recording()
    
    def stop_recording(self):
        """Stop the recording process gracefully."""
        if self.recording_process is not None:
            try:
                self.get_logger().info("Stopping recording process...")
                # Send SIGINT to allow ros2 bag to close properly
                self.recording_process.send_signal(signal.SIGINT)
                
                # Wait for process to terminate (with timeout)
                try:
                    self.recording_process.wait(timeout=5.0)
                    self.get_logger().info("✅ Recording stopped gracefully")
                except subprocess.TimeoutExpired:
                    self.get_logger().warn("Recording process didn't stop, forcing termination...")
                    self.recording_process.kill()
                    self.recording_process.wait()
                    self.get_logger().info("✅ Recording process terminated")
                
            except Exception as e:
                self.get_logger().error(f"Error stopping recording: {e}")
            finally:
                self.recording_process = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global recorder_node, recording_process, shutdown_requested
    shutdown_requested = True
    
    print("\n\n⚠️  Stopping dataset recording (Ctrl+C detected)...")
    
    # Stop recording process first
    if recording_process is not None:
        try:
            print("   Stopping bag recording...")
            recording_process.send_signal(signal.SIGINT)
            recording_process.wait(timeout=5.0)
            print("   ✅ Recording stopped")
        except subprocess.TimeoutExpired:
            print("   ⚠️  Forcing termination...")
            recording_process.kill()
            recording_process.wait()
        except Exception as e:
            print(f"   ⚠️  Error stopping recording: {e}")
    
    if recorder_node is not None and hasattr(recorder_node, 'stop_recording'):
        try:
            recorder_node.stop_recording()
        except:
            pass
    
    print("   Cleaning up...\n")
    sys.exit(0)

def cleanup_resources():
    """Cleanup function called on exit"""
    global recorder_node, recording_process
    
    # Stop recording
    if recording_process is not None:
        try:
            recording_process.send_signal(signal.SIGINT)
            recording_process.wait(timeout=3.0)
        except:
            try:
                recording_process.kill()
            except:
                pass
    
    if recorder_node is not None:
        try:
            if hasattr(recorder_node, 'stop_recording'):
                recorder_node.stop_recording()
            recorder_node.destroy_node()
        except:
            pass
    
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except:
        pass

def main(args=None):
    """Main function to run the dataset recorder."""
    global recorder_node, recording_process
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_resources)
    
    try:
        rclpy.init(args=args)
    except:
        print("⚠️  ROS2 already initialized or initialization failed")
    
    recorder_node = DatasetRecorder()
    
    print("\n" + "="*70)
    print("ROS2 Dataset Recorder for Visual Navigation Transformer")
    print("="*70)
    print(f"\nConfiguration (from data.yaml):")
    print(f"  Image Topic:    {IMAGE_TOPIC}")
    print(f"  Pose Topic:     {POSE_TOPIC}")
    print(f"  Additional:     {ADDITIONAL_TOPICS if ADDITIONAL_TOPICS else 'None'}")
    print(f"  Output Dir:     {BAG_OUTPUT_DIR}")
    print(f"  Dataset Name:   {DATASET_NAME}")
    print(f"  Compression:    {COMPRESSION_MODE} ({COMPRESSION_FORMAT})")
    print(f"  Storage:        {STORAGE_TYPE}")
    print("="*70 + "\n")
    
    # ========== Switch to Navigation Mode (MANDATORY) ==========
    print("🔄 Switching robot to navigation mode...")
    mode_switch_success = recorder_node.switch_to_navigation_mode(timeout_sec=5.0)
    
    if not mode_switch_success:
        print("❌ Failed to switch to navigation mode! Dataset recording aborted.")
        print("   Possible reasons:")
        print("   - Service /switch_to_navigation_mode not available")
        print("   - Robot driver (hellorobot_stretch3-driver) not running")
        print("   - Network communication issue between containers")
        recorder_node.destroy_node()
        rclpy.shutdown()
        return
    # ============================================================
    
    # ========== Camera Calibration (MANDATORY) ==========
    print(f"📷 Calibrating camera to dataset recording position (pan={CAMERA_CALIB_PAN:.2f}, tilt={CAMERA_CALIB_TILT:.2f})...")
    calibration_success = recorder_node.calibrate_camera_position(
        pan=CAMERA_CALIB_PAN,
        tilt=CAMERA_CALIB_TILT,
        timeout_sec=CAMERA_CALIB_TIMEOUT
    )
    
    if calibration_success:
        print("✅ Camera calibration completed successfully!")
        print("   Camera position verified - ready to record dataset")
    else:
        print("❌ Camera calibration FAILED! Dataset recording aborted.")
        print("   Possible reasons:")
        print("   - Action server (/stretch_controller/follow_joint_trajectory) not running")
        print("   - Camera joint is mechanically blocked or stuck")
        print("   - Network communication issue between containers")
        recorder_node.destroy_node()
        rclpy.shutdown()
        return
    # ====================================================
    
    # ========== Switch to Position Mode for Manual Control ==========
    print("🔄 Switching robot to position mode for manual recording...")
    mode_switch_success = recorder_node.switch_to_gamepad_mode(timeout_sec=5.0)
    
    if not mode_switch_success:
        print("⚠️  Failed to switch to position mode, but continuing anyway...")
        print("   You may need to manually switch modes using:")
        print("   ros2 service call /switch_to_position_mode std_srvs/srv/Trigger {}")
    else:
        print("✅ Robot is now in position mode - ready for manual control")
    # =================================================================
    
    try:
        print("\n🎬 Starting dataset recording! Press Ctrl+C to stop.\n")
        recording_process = recorder_node.recording_process
        recorder_node.start_recording()
    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt detected")
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[Cleanup] Stopping recording and cleaning up...")
        
        # Stop recording process
        try:
            if recorder_node is not None:
                recorder_node.stop_recording()
                print("[Cleanup] ✅ Recording stopped")
        except:
            pass
        
        # Destroy node
        try:
            if recorder_node is not None:
                recorder_node.destroy_node()
                print("[Cleanup] ✅ Node destroyed")
        except:
            pass
        
        # Shutdown ROS2
        try:
            if rclpy.ok():
                rclpy.shutdown()
                print("[Cleanup] ✅ ROS2 shutdown complete")
        except:
            pass
        
        print("[Cleanup] ✅ All resources cleaned up\n")


if __name__ == "__main__":
    main()
