#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import subprocess
import signal


class BagRecorder(Node):
    def __init__(self):
        super().__init__("gui_bag_recorder")

        deploy_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(deploy_dir, "config", "record.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.deploy_dir = deploy_dir
        self.recording_name = config.get("name", "default_recording")

        self.topics = [
            '/camera/camera/color/camera_info',
            '/camera/camera/color/image_raw',
            '/current_node_image',
            '/relative_pose_stamped',
            '/robot_trajectory',
            '/tf',
            '/tf_static',
            '/vn/candidate_waypoints',
            '/vn/chosen_waypoint',
            '/vn/end_node',
            '/vn/node',
            '/vn/start_node',
            '/waypoint'
        ]

        self.record_process = None

    def create_bag_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"{self.recording_name}_{timestamp}"
        bag_dir = os.path.join(self.deploy_dir, "bags", folder)

        self.get_logger().info(f"Recording to: {bag_dir}")
        return bag_dir


    def start_recording(self):
        if self.record_process:
            return

        bag_dir = self.create_bag_dir()

        cmd = [
            "ros2", "bag", "record",
            "-o", bag_dir,
            "--storage", "mcap",
            *self.topics
        ]

        # 非阻塞啟動 rosbag2
        self.record_process = subprocess.Popen(cmd)
        self.get_logger().info(f"Started ros2 bag record to {bag_dir}")

    def stop_recording(self):
        if not self.record_process:
            return

        self.record_process.send_signal(signal.SIGINT)
        self.record_process.wait()
        self.record_process = None

        self.get_logger().info("Stopped recording.")


class RecorderGUI:
    def __init__(self, node):
        self.node = node

        self.root = tk.Tk()
        self.root.title("ROS2 Bag Recorder")
        self.root.geometry("320x200")

        self.btn_record = tk.Button(self.root, text="Record", font=("Arial", 16),
                                    command=self.on_record)
        self.btn_record.pack(pady=15)

        self.btn_stop = tk.Button(self.root, text="Stop", font=("Arial", 16),
                                  command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.pack(pady=15)

        self.btn_exit = tk.Button(self.root, text="Exit", font=("Arial", 14),
                                  command=self.on_exit)
        self.btn_exit.pack(pady=15)

        threading.Thread(target=self.spin_ros, daemon=True).start()

    def spin_ros(self):
        rclpy.spin(self.node)

    def on_record(self):
        self.node.start_recording()
        self.btn_record.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_exit.config(state=tk.DISABLED)

    def on_stop(self):
        self.node.stop_recording()
        self.btn_record.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_exit.config(state=tk.NORMAL)

    def on_exit(self):
        if self.node.record_process:
            messagebox.showwarning("Recording", "Stop recording before exiting.")
            return
        self.root.destroy()
        rclpy.shutdown()

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = BagRecorder()
    gui = RecorderGUI(node)
    gui.run()


if __name__ == "__main__":
    main()
