import argparse
import os
import shutil
import time

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from utils import msg_to_pil

# IMAGE_TOPIC = "/navigation_camera/image_raw"
IMAGE_TOPIC = "/camera/camera/color/image_raw"
TOPOMAP_IMAGES_DIR = "../topomap"


class TopomapNode(Node):
    def __init__(self, image_topic, output_dir, dt):
        super().__init__("create_topomap")
        self.image_topic = image_topic
        self.output_dir = output_dir
        self.dt = dt
        self.obs_img = None
        self.subscriber = self.create_subscription(
            Image, self.image_topic, self.callback_obs, 10
        )
        self.get_logger().info(f"Subscribed to {self.image_topic}")
        self.remove_files_in_dir(self.output_dir)
        self.get_logger().info(f"Saving images to {self.output_dir}")
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

    def callback_obs(self, msg: Image):
        self.obs_img = msg_to_pil(msg).rotate(90, expand=True)

    def timer_callback(self):
        if self.obs_img is not None:
            img_path = os.path.join(self.output_dir, f"{self.image_counter}.png")
            self.obs_img.save(img_path)
            self.get_logger().info(f"Saved image {self.image_counter} to {img_path}")
            self.image_counter += 1
            self.start_time = time.time()
            self.obs_img = None
        elif time.time() - self.start_time > 2 * self.dt:
            self.get_logger().warn(f"Topic {self.image_topic} not publishing. Shutting down...")
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description=f"Code to generate topomaps from the {IMAGE_TOPIC} topic"
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="Path to save topological map images in ../topomaps/images directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=0.1,
        type=float,
        help=f"Time between images sampled from the {IMAGE_TOPIC} topic (default: 1.0 seconds)",
    )
    args = parser.parse_args()

    rclpy.init()
    topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    node = TopomapNode(IMAGE_TOPIC, topomap_name_dir, args.dt)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt. Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
