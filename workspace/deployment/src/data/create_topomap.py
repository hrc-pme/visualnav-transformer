import argparse
import os
import sys
import shutil
import time
import pickle

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from utils import msg_to_pil, compressed_msg_to_pil
import cv2

from scipy.spatial.transform import Rotation as R

IMAGE_TOPIC = "/camera/camera/color/image_raw"
USE_COMPRESSED = True  # Set to True to use compressed image topic
ODOM_TOPIC = "/odom"
TOPOMAP_IMAGES_DIR = "../../topomaps"
TOPOMAP_NAME = "6e-dr"
RECORD_PKL = False
AUTO_SHUTDOWN = False  # Whether to automatically shutdown when no image is received
TIMEOUT_DURATION = None  # Timeout duration in seconds (None means 2 * dt)


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

        self.timer = self.create_timer(self.dt, self.timer_callback)
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

    def timer_callback(self):
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

    def save_traj_data(self):
        if RECORD_PKL:
            output_pkl = os.path.join(self.output_dir, "traj_data.pkl")
            with open(output_pkl, "wb") as f:
                pickle.dump(self.traj_data, f)
            self.get_logger().info(f"Saved odometry data to {output_pkl}")
        else:
            self.get_logger().info("RECORD_PKL is False, skipping odometry data save")


def main():
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

    rclpy.init()
    topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, TOPOMAP_NAME)
    # Use global AUTO_SHUTDOWN as default, but allow command line override
    auto_shutdown = AUTO_SHUTDOWN and not args.no_auto_shutdown
    node = TopomapNode(
        IMAGE_TOPIC, 
        ODOM_TOPIC, 
        topomap_name_dir, 
        args.dt,
        auto_shutdown=auto_shutdown,
        timeout_duration=args.timeout,
        use_compressed=USE_COMPRESSED
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt. Shutting down...")
        node.save_traj_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()