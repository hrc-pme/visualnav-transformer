#!/usr/bin/env python3

import sys
import os
import time
from threading import Thread, Lock
import queue
import numpy as np
import cv2
from PIL import Image as PILImage, ImageTk
import tkinter as tk
from tkinter import ttk, Frame, Label, Canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from topic_names import (
    IMAGE_TOPIC, WAYPOINT_TOPIC, SAMPLED_ACTIONS_TOPIC, 
    VIZ_NAV_IMAGE_TOPIC
)

class NavigationGUI:
    def __init__(self, root, topomap_dir="../topomaps/6e-elevator"):
        self.root = root
        self.root.title("Visual Navigation Monitor")
        self.root.geometry("1400x900")
        
        # Thread-safe data structures
        self.data_lock = Lock()
        self.current_image = None
        self.current_waypoint = [0, 0, 0, 0]
        self.sampled_actions = []
        self.current_node = 0
        self.goal_node = 167
        self.reach_goal = False
        self.velocity_cmd = [0, 0, 0]  # [linear_x, linear_y, angular_z]
        
        # Load topomap
        self.load_topomap(topomap_dir)
        
        # Initialize ROS2
        self.init_ros()
        
        # Create GUI elements
        self.create_widgets()
        
        # Start ROS2 thread
        self.ros_thread = Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()
        
        # Start GUI update loop
        self.update_gui()
    
    def load_topomap(self, topomap_dir):
        """載入拓撲地圖圖像"""
        try:
            if os.path.exists(topomap_dir):
                filenames = sorted(
                    [f for f in os.listdir(topomap_dir) if f.endswith(('.png', '.jpg', '.jpeg'))],
                    key=lambda x: int(x.split(".")[0])
                )
                self.topomap = []
                self.topomap_thumbnails = []
                
                for fname in filenames:
                    img_path = os.path.join(topomap_dir, fname)
                    img = PILImage.open(img_path)
                    self.topomap.append(img)
                    
                    # Create thumbnail for GUI
                    thumbnail = img.copy()
                    thumbnail.thumbnail((80, 60), PILImage.Resampling.LANCZOS)
                    self.topomap_thumbnails.append(ImageTk.PhotoImage(thumbnail))
                
                self.goal_node = len(self.topomap) - 1
                print(f"Loaded {len(self.topomap)} topomap images")
            else:
                print(f"Topomap directory not found: {topomap_dir}")
                self.topomap = []
                self.topomap_thumbnails = []
        except Exception as e:
            print(f"Error loading topomap: {e}")
            self.topomap = []
            self.topomap_thumbnails = []
    
    def init_ros(self):
        """初始化 ROS2 節點和訂閱者"""
        rclpy.init()
        self.node = Node('navigation_gui')
        self.bridge = CvBridge()
        
        # Subscribers
        qos = QoSProfile(depth=10)
        self.node.create_subscription(
            Image, "/camera/camera/color/image_raw", 
            self.image_callback, qos_profile_sensor_data
        )
        self.node.create_subscription(
            Float32MultiArray, WAYPOINT_TOPIC,
            self.waypoint_callback, qos
        )
        self.node.create_subscription(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC,
            self.sampled_actions_callback, qos
        )
        self.node.create_subscription(
            Bool, "/reach_goal",
            self.reach_goal_callback, qos
        )
        self.node.create_subscription(
            Twist, "/stretch/cmd_vel",
            self.velocity_callback, qos
        )
    
    def image_callback(self, msg):
        """攝影機圖像回調"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            with self.data_lock:
                self.current_image = cv_image
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def waypoint_callback(self, msg):
        """路徑點回調"""
        with self.data_lock:
            self.current_waypoint = msg.data
    
    def sampled_actions_callback(self, msg):
        """採樣動作回調"""
        with self.data_lock:
            if len(msg.data) > 1:
                # First element is node index, rest are flattened actions
                self.current_node = int(msg.data[0])
                actions = np.array(msg.data[1:]).reshape(-1, 5, 2)  # [num_samples, 5_waypoints, 2_coords]
                self.sampled_actions = actions
    
    def reach_goal_callback(self, msg):
        """到達目標回調"""
        with self.data_lock:
            self.reach_goal = msg.data
    
    def velocity_callback(self, msg):
        """速度指令回調"""
        with self.data_lock:
            self.velocity_cmd = [msg.linear.x, msg.linear.y, msg.angular.z]
    
    def create_widgets(self):
        """創建 GUI 組件"""
        # Main layout
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top row - Status and controls
        self.create_status_panel(main_frame)
        
        # Middle row - Images and visualization
        image_frame = Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left - Current camera view
        self.create_camera_panel(image_frame)
        
        # Center - Trajectory visualization
        self.create_trajectory_panel(image_frame)
        
        # Right - Topomap panel
        self.create_topomap_panel(image_frame)
        
        # Bottom row - Detailed info
        self.create_info_panel(main_frame)
    
    def create_status_panel(self, parent):
        """創建狀態面板"""
        status_frame = Frame(parent, relief=tk.RAISED, borderwidth=1)
        status_frame.pack(fill=tk.X, pady=2)
        
        # Navigation status
        nav_frame = Frame(status_frame)
        nav_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        Label(nav_frame, text="Navigation Status", font=("Arial", 12, "bold")).pack()
        self.status_label = Label(nav_frame, text="Node: 0/167", font=("Arial", 10))
        self.status_label.pack()
        self.goal_status_label = Label(nav_frame, text="Goal: Not Reached", font=("Arial", 10))
        self.goal_status_label.pack()
        
        # Velocity info
        vel_frame = Frame(status_frame)
        vel_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        Label(vel_frame, text="Velocity Commands", font=("Arial", 12, "bold")).pack()
        self.vel_linear_label = Label(vel_frame, text="Linear: 0.00 m/s", font=("Arial", 10))
        self.vel_linear_label.pack()
        self.vel_angular_label = Label(vel_frame, text="Angular: 0.00 rad/s", font=("Arial", 10))
        self.vel_angular_label.pack()
        
        # Current waypoint
        wp_frame = Frame(status_frame)
        wp_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        Label(wp_frame, text="Current Waypoint", font=("Arial", 12, "bold")).pack()
        self.waypoint_label = Label(wp_frame, text="[0.00, 0.00, 0.00]", font=("Arial", 10))
        self.waypoint_label.pack()
    
    def create_camera_panel(self, parent):
        """創建攝影機視圖面板"""
        camera_frame = Frame(parent, relief=tk.RAISED, borderwidth=1)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        Label(camera_frame, text="Current Camera View", font=("Arial", 12, "bold")).pack()
        
        self.camera_canvas = Canvas(camera_frame, width=400, height=300, bg='black')
        self.camera_canvas.pack(pady=5)
    
    def create_trajectory_panel(self, parent):
        """創建軌跡可視化面板"""
        traj_frame = Frame(parent, relief=tk.RAISED, borderwidth=1)
        traj_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        Label(traj_frame, text="Trajectory Visualization", font=("Arial", 12, "bold")).pack()
        
        # Matplotlib figure for trajectory
        self.traj_fig = Figure(figsize=(4, 3), dpi=100)
        self.traj_ax = self.traj_fig.add_subplot(111)
        self.traj_canvas = FigureCanvasTkAgg(self.traj_fig, traj_frame)
        self.traj_canvas.get_tk_widget().pack(pady=5)
        
        self.init_trajectory_plot()
    
    def create_topomap_panel(self, parent):
        """創建拓撲地圖面板"""
        topo_frame = Frame(parent, relief=tk.RAISED, borderwidth=1)
        topo_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=2)
        
        Label(topo_frame, text="Topomap", font=("Arial", 12, "bold")).pack()
        
        # Scrollable frame for topomap thumbnails
        self.topo_canvas = Canvas(topo_frame, width=200, height=400)
        scrollbar = ttk.Scrollbar(topo_frame, orient="vertical", command=self.topo_canvas.yview)
        self.scrollable_frame = Frame(self.topo_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.topo_canvas.configure(scrollregion=self.topo_canvas.bbox("all"))
        )
        
        self.topo_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.topo_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.topo_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.create_topomap_thumbnails()
    
    def create_topomap_thumbnails(self):
        """創建拓撲地圖縮略圖"""
        self.thumbnail_labels = []
        
        for i, thumbnail in enumerate(self.topomap_thumbnails):
            frame = Frame(self.scrollable_frame, relief=tk.RIDGE, borderwidth=1)
            frame.pack(fill=tk.X, padx=2, pady=1)
            
            label = Label(frame, image=thumbnail)
            label.pack()
            
            text_label = Label(frame, text=f"Node {i}", font=("Arial", 8))
            text_label.pack()
            
            self.thumbnail_labels.append((frame, label, text_label))
    
    def create_info_panel(self, parent):
        """創建詳細資訊面板"""
        info_frame = Frame(parent, relief=tk.RAISED, borderwidth=1)
        info_frame.pack(fill=tk.X, pady=2)
        
        Label(info_frame, text="Detailed Information", font=("Arial", 12, "bold")).pack()
        
        details_frame = Frame(info_frame)
        details_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(details_frame, height=4, font=("Courier", 9))
        self.info_text.pack(fill=tk.X)
    
    def init_trajectory_plot(self):
        """初始化軌跡圖"""
        self.traj_ax.set_xlim(-2, 2)
        self.traj_ax.set_ylim(-1, 4)
        self.traj_ax.set_xlabel('X (m)')
        self.traj_ax.set_ylabel('Y (m)')
        self.traj_ax.grid(True, alpha=0.3)
        self.traj_ax.set_aspect('equal')
        
        # Robot position (origin)
        self.traj_ax.plot(0, 0, 'ro', markersize=8, label='Robot')
        self.traj_ax.legend()
        
        self.traj_canvas.draw()
    
    def update_trajectory_plot(self, actions, chosen_waypoint):
        """更新軌跡圖"""
        self.traj_ax.clear()
        self.init_trajectory_plot()
        
        if len(actions) > 0:
            # Plot all sampled trajectories in light gray
            for action in actions:
                x_coords = [0] + action[:, 0].tolist()
                y_coords = [0] + action[:, 1].tolist()
                self.traj_ax.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=1)
                self.traj_ax.scatter(x_coords[1:], y_coords[1:], c='lightgray', s=20, alpha=0.5)
        
        # Plot chosen waypoint in red
        if len(chosen_waypoint) >= 3:
            self.traj_ax.plot([0, chosen_waypoint[0]], [0, chosen_waypoint[1]], 'r-', linewidth=3, label='Chosen Path')
            self.traj_ax.plot(chosen_waypoint[0], chosen_waypoint[1], 'ro', markersize=10, label='Target Waypoint')
        
        self.traj_ax.legend()
        self.traj_canvas.draw()
    
    def update_camera_view(self, image):
        """更新攝影機視圖"""
        if image is not None:
            # Resize image to fit canvas
            h, w = image.shape[:2]
            canvas_w, canvas_h = 400, 300
            
            # Calculate scaling factor
            scale = min(canvas_w/w, canvas_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            
            # Resize and convert to PhotoImage
            image_resized = cv2.resize(image, (new_w, new_h))
            image_pil = PILImage.fromarray(image_resized)
            photo = ImageTk.PhotoImage(image_pil)
            
            # Update canvas
            self.camera_canvas.delete("all")
            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2
            self.camera_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            
            # Keep reference to prevent garbage collection
            self.camera_canvas.image = photo
    
    def update_topomap_display(self, current_node, goal_node):
        """更新拓撲地圖顯示"""
        for i, (frame, label, text_label) in enumerate(self.thumbnail_labels):
            if i == current_node:
                frame.configure(bg='red', relief=tk.RAISED, borderwidth=3)
                text_label.configure(text=f"Node {i} (Current)", bg='red', fg='white')
            elif i == goal_node:
                frame.configure(bg='green', relief=tk.RAISED, borderwidth=3)
                text_label.configure(text=f"Node {i} (Goal)", bg='green', fg='white')
            else:
                frame.configure(bg='white', relief=tk.RIDGE, borderwidth=1)
                text_label.configure(text=f"Node {i}", bg='white', fg='black')
    
    def update_gui(self):
        """更新 GUI 顯示"""
        with self.data_lock:
            # Update status labels
            self.status_label.config(text=f"Node: {self.current_node}/{self.goal_node}")
            
            if self.reach_goal:
                self.goal_status_label.config(text="Goal: REACHED", fg='green')
            else:
                self.goal_status_label.config(text="Goal: Not Reached", fg='red')
            
            # Update velocity info
            self.vel_linear_label.config(text=f"Linear: {self.velocity_cmd[0]:.2f} m/s")
            self.vel_angular_label.config(text=f"Angular: {self.velocity_cmd[2]:.2f} rad/s")
            
            # Update waypoint info
            if len(self.current_waypoint) >= 3:
                self.waypoint_label.config(
                    text=f"[{self.current_waypoint[0]:.2f}, {self.current_waypoint[1]:.2f}, {self.current_waypoint[2]:.2f}]"
                )
            
            # Update camera view
            self.update_camera_view(self.current_image)
            
            # Update trajectory plot
            if len(self.sampled_actions) > 0:
                self.update_trajectory_plot(self.sampled_actions, self.current_waypoint)
            
            # Update topomap display
            if self.thumbnail_labels:
                self.update_topomap_display(self.current_node, self.goal_node)
            
            # Update detailed info
            info_text = f"""
Sampled Actions: {len(self.sampled_actions)} trajectories
Current Waypoint: [{self.current_waypoint[0]:.3f}, {self.current_waypoint[1]:.3f}, {self.current_waypoint[2]:.3f}]
Velocity Command: Linear={self.velocity_cmd[0]:.3f} m/s, Angular={self.velocity_cmd[2]:.3f} rad/s
Navigation Progress: {self.current_node}/{self.goal_node} ({(self.current_node/max(self.goal_node,1)*100):.1f}%)
"""
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info_text.strip())
        
        # Schedule next update
        self.root.after(100, self.update_gui)  # Update every 100ms
    
    def ros_spin(self):
        """ROS2 spinning thread"""
        while rclpy.ok():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            except Exception as e:
                print(f"ROS2 spin error: {e}")
                break
    
    def cleanup(self):
        """清理資源"""
        if hasattr(self, 'node'):
            self.node.destroy_node()
        rclpy.shutdown()

def main():
    # 檢查命令行參數
    topomap_dir = "../topomaps/6e-elevator"
    if len(sys.argv) > 1:
        topomap_dir = sys.argv[1]
    
    # 創建並運行 GUI
    root = tk.Tk()
    app = NavigationGUI(root, topomap_dir)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down GUI...")
        app.cleanup()

if __name__ == "__main__":
    main()
