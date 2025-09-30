#!/usr/bin/env python3
"""
Visual Navigation Tkinter GUI - Lightweight Desktop Version

Advantages:
- Tkinter is built-in (no installation needed)
- Very lightweight and stable
- No Qt plugin conflicts
- Works in Docker with X11 forwarding

Requirements:
- Python standard library (tkinter)
- opencv-python (already installed)
- PIL/Pillow (for image display)

Usage:
python navigate.tk.py [topomap_dir]

Note: Requires GUI environment (X server)
"""

import sys
import os
import time
import cv2
import numpy as np
from threading import Thread, Lock
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import psutil
import yaml

# ROS2
import rclpy

from navigation_core import NavigationCore

# Load waypoint visualization config
WAYPOINT_VIZ_CONFIG_PATH = "../config/waypoint_visualization.yaml"


class NavigationTkGUI:
    def __init__(self, master, topomap_dir="../topomaps/6e-elevator"):
        self.master = master
        self.master.title('Visual Navigation - Tkinter GUI')
        self.master.geometry('1400x900')
        
        print("[INFO] Initializing Navigation Tkinter GUI...")
        
        # Load waypoint visualization config
        self.viz_config = self.load_viz_config()
        
        # Initialize ROS2 first
        if not rclpy.ok():
            rclpy.init()
        
        # Initialize ROS2 core (don't init rclpy again)
        self.nav_core = NavigationCore(init_rclpy=False)
        self.topomap_dir = topomap_dir
        
        # Thread safety
        self.data_lock = Lock()
        self.current_frame = None
        
        # ROS2 status (accessed by threads)
        self.ros_initialized = False
        self.ros_error = None
        
        # Topomap display (single current image)
        self.topomap_images = []  # Store all topomap image paths
        self.current_displayed_node = -1
        
        # State variables (must be set before setup_ui)
        self.frame_count = 0
        self.running = True
        
        # Setup UI
        self.setup_ui()
        
        # Load topomap image paths after GUI is ready
        self.master.after(100, lambda: self.load_topomap(topomap_dir))
        
        # Initialize ROS2 in background
        self.ros_thread = Thread(target=self.initialize_ros, daemon=True)
        self.ros_thread.start()
        
        # Setup update loop
        self.update_gui()
    
    def load_viz_config(self):
        """Load waypoint visualization configuration"""
        try:
            if os.path.exists(WAYPOINT_VIZ_CONFIG_PATH):
                with open(WAYPOINT_VIZ_CONFIG_PATH, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"[INFO] Loaded waypoint visualization config from {WAYPOINT_VIZ_CONFIG_PATH}")
                return config
            else:
                print(f"[WARN] Config file not found: {WAYPOINT_VIZ_CONFIG_PATH}, using defaults")
                return self.get_default_viz_config()
        except Exception as e:
            print(f"[ERROR] Failed to load viz config: {e}, using defaults")
            return self.get_default_viz_config()
    
    def get_default_viz_config(self):
        """Get default visualization configuration"""
        return {
            'visualization': {
                'robot_position': {'x_fraction': 0.5, 'y_fraction': 0.95},
                'scaling': {'base_horizontal': 2.5, 'base_vertical': 1.5},
                'perspective': {'horizontal_amplification': 0.3, 'vertical_compression': 0.5},
                'curve_smoothness': 15,
                'appearance': {
                    'candidate_color': [0, 0, 255],
                    'candidate_thickness': 2,
                    'candidate_point_radius': 3,
                    'chosen_color': [0, 255, 0],
                    'chosen_thickness': 5,
                    'chosen_point_radius': 7,
                    'chosen_outline_radius': 9,
                    'robot_color': [255, 255, 255],
                    'robot_radius': 8,
                    'robot_outline_radius': 10,
                    'robot_outline_color': [0, 0, 0],
                    'arrow_length': 25,
                    'arrow_color': [255, 255, 255],
                    'arrow_thickness': 2
                }
            }
        }
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        container = tk.Frame(self.master)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Top section with two columns
        main_frame = tk.Frame(container)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left column - Camera
        left_frame = tk.Frame(main_frame, width=700)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Camera feed
        camera_group = tk.LabelFrame(left_frame, text="Camera Feed", font=('Arial', 12, 'bold'))
        camera_group.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = tk.Label(camera_group, bg='black')
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right column - Status and Controls
        right_frame = tk.Frame(main_frame, width=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Navigation Status
        status_group = tk.LabelFrame(right_frame, text="Navigation Status", font=('Arial', 12, 'bold'))
        status_group.pack(fill=tk.BOTH, pady=(0, 5))
        
        status_inner = tk.Frame(status_group)
        status_inner.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # Status labels
        self.status_labels = {}
        status_items = [
            ('Current Node:', 'current_node'),
            ('Goal Node:', 'goal_node'),
            ('Waypoint:', 'waypoint'),
            ('Linear Vel:', 'linear_vel'),
            ('Angular Vel:', 'angular_vel'),
            ('Reach Goal:', 'reach_goal'),
        ]
        
        for i, (label, key) in enumerate(status_items):
            tk.Label(status_inner, text=label, font=('Arial', 10, 'bold'), anchor='w').grid(
                row=i, column=0, sticky='w', pady=2)
            value_label = tk.Label(status_inner, text='N/A', font=('Arial', 10), anchor='w')
            value_label.grid(row=i, column=1, sticky='w', pady=2, padx=(10, 0))
            self.status_labels[key] = value_label
        
        # Progress bar
        tk.Label(status_inner, text='Progress:', font=('Arial', 10, 'bold'), anchor='w').grid(
            row=len(status_items), column=0, sticky='w', pady=2)
        self.progress_bar = ttk.Progressbar(status_inner, mode='determinate', length=200)
        self.progress_bar.grid(row=len(status_items), column=1, sticky='w', pady=2, padx=(10, 0))
        
        # Current Topomap Node (single image)
        topomap_group = tk.LabelFrame(right_frame, text="Current Topomap Node", font=('Arial', 12, 'bold'))
        topomap_group.pack(fill=tk.BOTH, expand=True, pady=(5, 5))
        
        # Single image label for current topomap node
        self.topomap_label = tk.Label(topomap_group, bg='white', text='Waiting for navigation...', 
                                     font=('Arial', 10), fg='gray')
        self.topomap_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Node ID label
        self.topomap_id_label = tk.Label(topomap_group, text='Node: N/A', 
                                        font=('Arial', 11, 'bold'), bg='white', fg='#4CAF50')
        self.topomap_id_label.pack(pady=5)
        
        # Controls
        controls_group = tk.LabelFrame(right_frame, text="Controls", font=('Arial', 12, 'bold'))
        controls_group.pack(fill=tk.X)
        
        controls_inner = tk.Frame(controls_group)
        controls_inner.pack(pady=10)
        
        # ROS2 status
        self.ros_status = tk.Label(controls_inner, text='⚪ ROS2: Connecting...', 
                                   font=('Arial', 10), anchor='w')
        self.ros_status.pack(pady=5)
        
        # Quit button
        quit_btn = tk.Button(controls_inner, text='Quit', command=self.on_closing, 
                            font=('Arial', 12), bg='#f44336', fg='white', padx=20, pady=5)
        quit_btn.pack(pady=5)
        
        # Bottom status bar for system resources
        status_bar = tk.Frame(container, bg='#e0e0e0', relief=tk.SUNKEN, bd=1)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # System resource labels (horizontal layout)
        self.resource_labels = {}
        
        tk.Label(status_bar, text='CPU:', font=('Arial', 9), bg='#e0e0e0').pack(side=tk.LEFT, padx=(10, 2))
        self.resource_labels['cpu'] = tk.Label(status_bar, text='N/A', font=('Arial', 9, 'bold'), bg='#e0e0e0', fg='#1976D2')
        self.resource_labels['cpu'].pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(status_bar, text='Mem:', font=('Arial', 9), bg='#e0e0e0').pack(side=tk.LEFT, padx=(0, 2))
        self.resource_labels['memory'] = tk.Label(status_bar, text='N/A', font=('Arial', 9, 'bold'), bg='#e0e0e0', fg='#388E3C')
        self.resource_labels['memory'].pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(status_bar, text='GPU:', font=('Arial', 9), bg='#e0e0e0').pack(side=tk.LEFT, padx=(0, 2))
        self.resource_labels['gpu'] = tk.Label(status_bar, text='N/A', font=('Arial', 9, 'bold'), bg='#e0e0e0', fg='#F57C00')
        self.resource_labels['gpu'].pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(status_bar, text='GPU Mem:', font=('Arial', 9), bg='#e0e0e0').pack(side=tk.LEFT, padx=(0, 2))
        self.resource_labels['gpu_memory'] = tk.Label(status_bar, text='N/A', font=('Arial', 9, 'bold'), bg='#e0e0e0', fg='#D32F2F')
        self.resource_labels['gpu_memory'].pack(side=tk.LEFT, padx=(0, 10))
    
    def load_topomap(self, topomap_dir):
        """Load topomap image paths (without displaying all images)"""
        # Try multiple possible paths
        possible_paths = [
            os.path.join(topomap_dir, "images"),
            topomap_dir,
        ]
        
        topomap_images_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if image_files:
                    topomap_images_dir = path
                    break
        
        if topomap_images_dir is None:
            print("[WARN] No topomap images found")
            return
        
        # Load ALL image paths (sorted by numeric value)
        image_files = sorted([f for f in os.listdir(topomap_images_dir) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))],
                            key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else 0)
        
        if not image_files:
            print("[WARN] No images found in topomap")
            return
        
        # Store image paths
        self.topomap_images = [os.path.join(topomap_images_dir, f) for f in image_files]
        
        print(f"[INFO] Loaded {len(self.topomap_images)} topomap image paths")
    
    def update_topomap_display(self, node_id):
        """Update topomap to show specific node image"""
        if node_id < 0 or node_id >= len(self.topomap_images):
            return
        
        if node_id == self.current_displayed_node:
            return  # Already displaying this node
        
        try:
            # Load and resize image
            img_path = self.topomap_images[node_id]
            img = Image.open(img_path)
            
            # Resize to fit (reasonable size for display)
            display_height = 400
            w, h = img.size
            display_width = int(w * display_height / h)
            img_resized = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img_resized)
            
            # Update label
            self.topomap_label.config(image=photo, text='')
            self.topomap_label.image = photo  # Keep reference
            
            # Update node ID label
            self.topomap_id_label.config(text=f"Node: #{node_id}")
            
            self.current_displayed_node = node_id
            
        except Exception as e:
            print(f"[ERROR] Failed to display topomap node {node_id}: {e}")
    
    def draw_waypoints_on_image(self, img):
        """
        Draw candidate waypoints (red curves) and chosen waypoint (green curve) on the image.
        Waypoints are in robot frame (x forward, y left), need to convert to image coordinates.
        
        Camera perspective considerations:
        - Camera is mounted on top of robot, looking forward and down
        - Robot center (wheel base) is at bottom center of image
        - Perspective projection: distant points move up slower, horizontal spread increases
        
        Args:
            img: BGR image from camera
        
        Returns:
            Annotated image with waypoints drawn
        """
        annotated = img.copy()
        h, w = annotated.shape[:2]
        
        # Get config parameters
        viz_cfg = self.viz_config['visualization']
        robot_pos_cfg = viz_cfg['robot_position']
        scaling_cfg = viz_cfg['scaling']
        perspective_cfg = viz_cfg['perspective']
        appearance_cfg = viz_cfg['appearance']
        
        # Robot position based on config
        robot_x = int(w * robot_pos_cfg['x_fraction'])
        robot_y = int(h * robot_pos_cfg['y_fraction'])
        
        # Scaling factors
        base_scale_x = w / scaling_cfg['base_horizontal']
        base_scale_y = h / scaling_cfg['base_vertical']
        
        # Perspective parameters
        h_amp = perspective_cfg['horizontal_amplification']
        v_comp = perspective_cfg['vertical_compression']
        
        def waypoint_to_pixel(wp):
            """
            Convert waypoint (x, y) in robot frame to pixel coordinates.
            Applies perspective transformation to simulate camera view from above.
            
            Robot frame: x forward (away from robot), y left
            Camera perspective: points further away (larger x) appear higher in image
            """
            x_robot = wp[0]  # forward distance (meters)
            y_robot = wp[1]  # left distance (meters)
            
            # Perspective factor: points further away appear to move up less
            perspective_factor = 1.0 / (1.0 + x_robot * v_comp)
            
            # Horizontal position with perspective amplification
            horizontal_scale = base_scale_x * (1.0 + x_robot * h_amp)
            px = int(robot_x + y_robot * horizontal_scale)
            
            # Vertical position with perspective compression
            vertical_displacement = x_robot * base_scale_y * perspective_factor
            py = int(robot_y - vertical_displacement)
            
            # Clamp to image boundaries
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            
            return (px, py)
        
        def draw_trajectory_curve(img, waypoints, color, thickness):
            """
            Draw a smooth curve through waypoints using cubic Bezier or spline.
            This simulates the predicted vehicle trajectory.
            """
            if len(waypoints) < 1:
                return
            
            # Convert all waypoints to pixel coordinates
            points = [waypoint_to_pixel(wp) for wp in waypoints]
            
            # Add robot position as starting point
            points = [(robot_x, robot_y)] + points
            
            if len(points) >= 2:
                # Generate smooth curve through points using interpolation
                smooth_points = []
                for i in range(len(points) - 1):
                    p1 = np.array(points[i], dtype=float)
                    p2 = np.array(points[i + 1], dtype=float)
                    
                    # Create intermediate points for smooth curve
                    num_interp = viz_cfg['curve_smoothness']
                    for t in np.linspace(0, 1, num_interp):
                        # Cubic interpolation for smoother curve
                        t2 = t * t
                        t3 = t2 * t
                        # Hermite spline coefficients
                        h1 = 2*t3 - 3*t2 + 1
                        h2 = -2*t3 + 3*t2
                        
                        point = h1 * p1 + h2 * p2
                        smooth_points.append(point)
                
                if len(smooth_points) > 1:
                    smooth_array = np.array(smooth_points, dtype=np.int32)
                    cv2.polylines(img, [smooth_array], False, color, thickness, cv2.LINE_AA)
        
        # Draw candidate waypoints (red curves)
        if self.nav_core.candidate_waypoints is not None and len(self.nav_core.candidate_waypoints) > 0:
            cand_color = tuple(appearance_cfg['candidate_color'])
            cand_thickness = appearance_cfg['candidate_thickness']
            cand_radius = appearance_cfg['candidate_point_radius']
            
            for wp in self.nav_core.candidate_waypoints:
                if len(wp) >= 2:
                    # Draw trajectory curve for each candidate
                    draw_trajectory_curve(annotated, [wp], cand_color, cand_thickness)
                    
                    # Draw small circle at endpoint
                    pt = waypoint_to_pixel(wp)
                    cv2.circle(annotated, pt, cand_radius, cand_color, -1)
        
        # Draw chosen waypoint (green thick curve)
        if self.nav_core.chosen_waypoint is not None and len(self.nav_core.chosen_waypoint) >= 2:
            wp = self.nav_core.chosen_waypoint
            chosen_color = tuple(appearance_cfg['chosen_color'])
            chosen_thickness = appearance_cfg['chosen_thickness']
            chosen_radius = appearance_cfg['chosen_point_radius']
            chosen_outline = appearance_cfg['chosen_outline_radius']
            
            # Draw thick trajectory curve for chosen waypoint
            draw_trajectory_curve(annotated, [wp], chosen_color, chosen_thickness)
            
            # Draw larger circle at endpoint
            pt = waypoint_to_pixel(wp)
            cv2.circle(annotated, pt, chosen_radius, chosen_color, -1)
            cv2.circle(annotated, pt, chosen_outline, (255, 255, 255), 2)
        
        # Draw robot position indicator at bottom center
        robot_color = tuple(appearance_cfg['robot_color'])
        robot_radius = appearance_cfg['robot_radius']
        robot_outline_radius = appearance_cfg['robot_outline_radius']
        robot_outline_color = tuple(appearance_cfg['robot_outline_color'])
        
        cv2.circle(annotated, (robot_x, robot_y), robot_radius, robot_color, -1)
        cv2.circle(annotated, (robot_x, robot_y), robot_outline_radius, robot_outline_color, 2)
        
        # Draw robot direction indicator (small arrow pointing up/forward)
        arrow_length = appearance_cfg['arrow_length']
        arrow_color = tuple(appearance_cfg['arrow_color'])
        arrow_thickness = appearance_cfg['arrow_thickness']
        arrow_tip = (robot_x, robot_y - arrow_length)
        cv2.arrowedLine(annotated, (robot_x, robot_y), arrow_tip, arrow_color, arrow_thickness, cv2.LINE_AA, tipLength=0.3)
        
        return annotated
    
    def initialize_ros(self):
        """Initialize ROS2 in background thread"""
        try:
            print("[INFO] Initializing ROS2 navigation...")
            success = self.nav_core.initialize(self.topomap_dir)
            
            if success:
                # Set flag instead of updating GUI directly
                self.ros_initialized = True
                print("[INFO] ROS2 navigation initialized successfully")
                
                # Start ROS2 spinning in background
                self.ros_spin_thread = Thread(target=self._ros_spin_loop, daemon=True)
                self.ros_spin_thread.start()
            else:
                self.ros_error = "Initialization failed"
                print("[ERROR] ROS2 navigation initialization failed")
        except Exception as e:
            self.ros_error = str(e)
            print(f"[ERROR] ROS2 initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _ros_spin_loop(self):
        """ROS2 spinning loop in background thread"""
        while self.running and rclpy.ok():
            try:
                rclpy.spin_once(self.nav_core, timeout_sec=0.01)
                self.nav_core.step()  # Perform navigation step
                time.sleep(0.05)  # 20 Hz
            except Exception as e:
                print(f"[ERROR] ROS2 spin error: {e}")
    
    def update_gui(self):
        """Main GUI update loop"""
        if not self.running:
            return
        
        try:
            # Update ROS2 status from background thread flags
            if self.ros_initialized:
                self.ros_status.config(text='🟢 ROS2: Connected', fg='green')
                self.ros_initialized = None  # Clear flag
            elif self.ros_error is not None:
                self.ros_status.config(text=f'🔴 ROS2: {self.ros_error}', fg='red')
                self.ros_error = None  # Clear error
            
            # Check a: Update camera feed (only if image topic is publishing)
            if self.nav_core.latest_image is not None:
                with self.data_lock:
                    self.current_frame = self.nav_core.latest_image.copy()
                
                # Rotate 180 degrees (90 + 90)
                rotated = cv2.rotate(self.current_frame, cv2.ROTATE_180)
                
                # Draw waypoints on the image
                annotated = self.draw_waypoints_on_image(rotated)
                
                h, w = annotated.shape[:2]
                display_height = 800  # Further increased for better visibility
                display_width = int(w * display_height / h)
                resized = cv2.resize(annotated, (display_width, display_height))
                
                # Convert to PhotoImage
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                photo = ImageTk.PhotoImage(img)
                
                self.camera_label.config(image=photo)
                self.camera_label.image = photo
                
                self.frame_count += 1
            
            # Check c: Update navigation status
            # Get current_node from ROS2 topic (subscribed from navigate.ros2.py)
            current_node = getattr(self.nav_core, 'current_node', -1)
            goal_node = getattr(self.nav_core, 'goal_node', -1)
            
            if current_node >= 0:
                self.status_labels['current_node'].config(text=str(current_node))
                # Update topomap to show current node image
                self.update_topomap_display(current_node)
            else:
                self.status_labels['current_node'].config(text='N/A')
            
            if goal_node > 0:
                self.status_labels['goal_node'].config(text=str(goal_node))
            else:
                self.status_labels['goal_node'].config(text='N/A')
            
            # Waypoint (from navigate.ros2.py via /waypoint topic)
            if hasattr(self.nav_core, 'waypoint') and self.nav_core.waypoint is not None:
                wp = self.nav_core.waypoint
                waypoint_str = f"({wp[0]:.2f}, {wp[1]:.2f})"
                self.status_labels['waypoint'].config(text=waypoint_str)
            else:
                self.status_labels['waypoint'].config(text='N/A')
            
            # Progress
            if goal_node > 0 and current_node >= 0:
                progress = (current_node / goal_node) * 100
                self.progress_bar['value'] = progress
            
            # Reach goal
            reach_goal = getattr(self.nav_core, 'reached_goal', False)
            self.status_labels['reach_goal'].config(
                text='✓ Yes' if reach_goal else '✗ No',
                fg='green' if reach_goal else 'red'
            )
            
            # Check b: Update velocity (only if waypoint_to_goal_pose.py is running)
            # We check if latest_twist exists and is recent
            if hasattr(self.nav_core, 'latest_twist') and self.nav_core.latest_twist is not None:
                twist = self.nav_core.latest_twist
                self.status_labels['linear_vel'].config(text=f"{twist.linear.x:.2f} m/s")
                self.status_labels['angular_vel'].config(text=f"{twist.angular.z:.2f} rad/s")
            else:
                self.status_labels['linear_vel'].config(text='N/A')
                self.status_labels['angular_vel'].config(text='N/A')
            
            # Update system resources (less frequently)
            if self.frame_count % 4 == 0:  # Every 4 frames
                self.update_system_resources()
            
        except Exception as e:
            pass  # Silently handle errors
        
        # Schedule next update
        self.master.after(500, self.update_gui)  # 2 Hz
    
    def update_system_resources(self):
        """Update system resource monitoring"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.resource_labels['cpu'].config(text=f"{cpu_percent:.1f}%")
            
            # Memory usage
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            mem_used_gb = mem.used / (1024**3)
            mem_total_gb = mem.total / (1024**3)
            self.resource_labels['memory'].config(text=f"{mem_percent:.1f}% ({mem_used_gb:.1f}/{mem_total_gb:.1f} GB)")
            
            # GPU usage (try to get, if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    self.resource_labels['gpu'].config(text=f"{gpu.load*100:.1f}%")
                    self.resource_labels['gpu_memory'].config(
                        text=f"{gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB ({gpu.memoryUtil*100:.1f}%)")
                else:
                    self.resource_labels['gpu'].config(text='No GPU')
                    self.resource_labels['gpu_memory'].config(text='N/A')
            except:
                # GPU monitoring not available
                self.resource_labels['gpu'].config(text='N/A')
                self.resource_labels['gpu_memory'].config(text='N/A')
                
        except Exception as e:
            pass  # Silently handle errors
    
    def on_closing(self):
        """Clean up on window close"""
        print("[INFO] Shutting down...")
        self.running = False
        
        # Give threads time to finish
        time.sleep(0.5)
        
        # Cleanup ROS2
        if hasattr(self, 'nav_core'):
            try:
                self.nav_core.destroy_node()
            except:
                pass
        
        # Shutdown rclpy
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
        
        self.master.quit()
        self.master.destroy()


def main():
    print("=" * 60)
    print("Navigation Tkinter GUI - Lightweight Desktop Version")
    print("=" * 60)
    
    # Get topomap directory from command line
    topomap_dir = sys.argv[1] if len(sys.argv) > 1 else "../topomaps/6e-elevator"
    
    # Create Tkinter app
    root = tk.Tk()
    app = NavigationTkGUI(root, topomap_dir)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received")
        app.on_closing()


if __name__ == '__main__':
    main()
