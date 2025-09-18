#!/usr/bin/env python3

"""
Installation Requirements:
pip install flask flask-socketio
pip install opencv-python pillow numpy matplotlib
pip install rclpy sensor-msgs std-msgs geometry-msgs cv-bridge

For ROS2 installation, ensure you have:
- ros-humble-desktop (or your ROS2 distribution)
- ros-humble-cv-bridge
- ros-humble-sensor-msgs

Usage:
python navigate.gui.py [topomap_dir]
Then open your browser to http://localhost:5000
"""

import sys
import os
import time
import json
import base64
from threading import Thread, Lock
import queue
import numpy as np
import cv2
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

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

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'navigation_gui_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class NavigationWebGUI:
    def __init__(self, topomap_dir="../topomaps/6e-elevator"):
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
        
        # Start ROS2 thread
        self.ros_thread = Thread(target=self.ros_spin, daemon=True)
        self.ros_thread.start()
        
        # Start data broadcast thread
        self.broadcast_thread = Thread(target=self.broadcast_data, daemon=True)
        self.broadcast_thread.start()
    
    def load_topomap(self, topomap_dir):
        """載入拓撲地圖圖像"""
        try:
            if os.path.exists(topomap_dir):
                filenames = sorted(
                    [f for f in os.listdir(topomap_dir) if f.endswith(('.png', '.jpg', '.jpeg'))],
                    key=lambda x: int(x.split(".")[0])
                )
                self.topomap = []
                self.topomap_base64 = []
                
                for fname in filenames:
                    img_path = os.path.join(topomap_dir, fname)
                    img = PILImage.open(img_path)
                    self.topomap.append(img)
                    
                    # Convert to base64 for web display
                    thumbnail = img.copy()
                    thumbnail.thumbnail((120, 90), PILImage.Resampling.LANCZOS)
                    buffer = BytesIO()
                    thumbnail.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    self.topomap_base64.append(img_base64)
                
                self.goal_node = len(self.topomap) - 1
                print(f"Loaded {len(self.topomap)} topomap images")
            else:
                print(f"Topomap directory not found: {topomap_dir}")
                self.topomap = []
                self.topomap_base64 = []
        except Exception as e:
            print(f"Error loading topomap: {e}")
            self.topomap = []
            self.topomap_base64 = []
    
    def init_ros(self):
        """初始化 ROS2 節點和訂閱者"""
        rclpy.init()
        self.node = Node('navigation_web_gui')
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
    
    def generate_trajectory_plot(self):
        """生成軌跡圖並返回 base64 編碼"""
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 4)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Robot position (origin)
        ax.plot(0, 0, 'ro', markersize=8, label='Robot')
        
        with self.data_lock:
            if len(self.sampled_actions) > 0:
                # Plot all sampled trajectories in light gray
                for action in self.sampled_actions:
                    x_coords = [0] + action[:, 0].tolist()
                    y_coords = [0] + action[:, 1].tolist()
                    ax.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=1)
                    ax.scatter(x_coords[1:], y_coords[1:], c='lightgray', s=20, alpha=0.5)
            
            # Plot chosen waypoint in red
            if len(self.current_waypoint) >= 3:
                ax.plot([0, self.current_waypoint[0]], [0, self.current_waypoint[1]], 'r-', linewidth=3, label='Chosen Path')
                ax.plot(self.current_waypoint[0], self.current_waypoint[1], 'ro', markersize=10, label='Target Waypoint')
        
        ax.legend()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return plot_base64
    
    def get_current_data(self):
        """獲取當前所有數據"""
        with self.data_lock:
            # Convert current image to base64
            camera_base64 = None
            if self.current_image is not None:
                # Resize image for web display
                h, w = self.current_image.shape[:2]
                max_size = 400
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    resized_img = cv2.resize(self.current_image, (new_w, new_h))
                else:
                    resized_img = self.current_image
                
                # Convert to base64
                img_pil = PILImage.fromarray(resized_img)
                buffer = BytesIO()
                img_pil.save(buffer, format='PNG')
                camera_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Generate trajectory plot
            trajectory_base64 = self.generate_trajectory_plot()
            
            return {
                'current_node': self.current_node,
                'goal_node': self.goal_node,
                'reach_goal': self.reach_goal,
                'waypoint': self.current_waypoint[:3] if len(self.current_waypoint) >= 3 else [0, 0, 0],
                'velocity': self.velocity_cmd,
                'sampled_actions_count': len(self.sampled_actions),
                'camera_image': camera_base64,
                'trajectory_plot': trajectory_base64,
                'topomap_images': self.topomap_base64,
                'navigation_progress': (self.current_node / max(self.goal_node, 1)) * 100
            }
    
    def broadcast_data(self):
        """定期廣播數據到所有連接的客戶端"""
        while True:
            try:
                data = self.get_current_data()
                socketio.emit('data_update', data)
                time.sleep(0.1)  # 10Hz update rate
            except Exception as e:
                print(f"Broadcast error: {e}")
                time.sleep(1)
    
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

# Global instance
nav_gui = None

@app.route('/')
def index():
    """主頁面"""
    return render_template('navigation.html')

@socketio.on('connect')
def handle_connect():
    """客戶端連接處理"""
    print('Client connected')
    if nav_gui:
        emit('data_update', nav_gui.get_current_data())

@socketio.on('disconnect')
def handle_disconnect():
    """客戶端斷開連接處理"""
    print('Client disconnected')

@socketio.on('request_data')
def handle_request_data():
    """處理數據請求"""
    if nav_gui:
        emit('data_update', nav_gui.get_current_data())

def create_html_template():
    """創建 HTML 模板"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Navigation Monitor</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .status-panel {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-around;
        }
        .status-item {
            text-align: center;
        }
        .status-item h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        .status-value {
            font-size: 18px;
            font-weight: bold;
        }
        .main-content {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .camera-panel {
            flex: 1;
        }
        .trajectory-panel {
            flex: 1;
        }
        .topomap-panel {
            width: 250px;
            max-height: 600px;
            overflow-y: auto;
        }
        .panel h2 {
            margin: 0 0 15px 0;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        .image-container {
            text-align: center;
            background: #000;
            border-radius: 4px;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 4px;
        }
        .topomap-item {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 4px;
            border: 2px solid transparent;
        }
        .topomap-item.current {
            background-color: #ffebee;
            border-color: #f44336;
        }
        .topomap-item.goal {
            background-color: #e8f5e8;
            border-color: #4caf50;
        }
        .topomap-item img {
            width: 60px;
            height: 45px;
            margin-right: 10px;
            border-radius: 4px;
        }
        .info-panel {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .info-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }
        .info-label {
            font-weight: bold;
            color: #555;
            margin-bottom: 5px;
        }
        .info-value {
            font-family: monospace;
            font-size: 14px;
            color: #333;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }
        .progress-fill {
            height: 100%;
            background-color: #4caf50;
            transition: width 0.3s ease;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-indicator.green { background-color: #4caf50; }
        .status-indicator.red { background-color: #f44336; }
        .status-indicator.yellow { background-color: #ff9800; }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center; color: #333; margin-bottom: 30px;">Visual Navigation Monitor</h1>
        
        <div class="status-panel">
            <div class="status-item">
                <h3>Navigation Status</h3>
                <div class="status-value">
                    <span class="status-indicator" id="nav-indicator"></span>
                    Node: <span id="current-node">0</span>/<span id="goal-node">0</span>
                </div>
                <div style="margin-top: 10px;">
                    <span id="goal-status">Goal: Not Reached</span>
                </div>
            </div>
            <div class="status-item">
                <h3>Velocity Commands</h3>
                <div class="status-value">
                    Linear: <span id="linear-vel">0.00</span> m/s
                </div>
                <div class="status-value">
                    Angular: <span id="angular-vel">0.00</span> rad/s
                </div>
            </div>
            <div class="status-item">
                <h3>Current Waypoint</h3>
                <div class="status-value" id="waypoint-display">[0.00, 0.00, 0.00]</div>
            </div>
            <div class="status-item">
                <h3>Progress</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                </div>
                <div style="margin-top: 5px; font-size: 14px;">
                    <span id="progress-text">0.0%</span>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="panel camera-panel">
                <h2>Current Camera View</h2>
                <div class="image-container">
                    <img id="camera-image" src="" alt="No camera feed" style="display: none;">
                    <div id="no-camera" style="color: #666;">No camera feed available</div>
                </div>
            </div>
            
            <div class="panel trajectory-panel">
                <h2>Trajectory Visualization</h2>
                <div class="image-container">
                    <img id="trajectory-plot" src="" alt="No trajectory data" style="display: none;">
                    <div id="no-trajectory" style="color: #666;">No trajectory data</div>
                </div>
            </div>
            
            <div class="panel topomap-panel">
                <h2>Topomap</h2>
                <div id="topomap-container">
                    <div style="color: #666; text-align: center;">Loading topomap...</div>
                </div>
            </div>
        </div>
        
        <div class="info-panel">
            <h2>Detailed Information</h2>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Sampled Actions</div>
                    <div class="info-value" id="sampled-actions">0 trajectories</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Precise Waypoint</div>
                    <div class="info-value" id="precise-waypoint">[0.000, 0.000, 0.000]</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Velocity Command</div>
                    <div class="info-value" id="precise-velocity">Linear=0.000 m/s, Angular=0.000 rad/s</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Navigation Progress</div>
                    <div class="info-value" id="detailed-progress">0/0 (0.0%)</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to server');
        });
        
        socket.on('data_update', function(data) {
            updateDisplay(data);
        });
        
        function updateDisplay(data) {
            // Update navigation status
            document.getElementById('current-node').textContent = data.current_node;
            document.getElementById('goal-node').textContent = data.goal_node;
            
            const navIndicator = document.getElementById('nav-indicator');
            const goalStatus = document.getElementById('goal-status');
            if (data.reach_goal) {
                navIndicator.className = 'status-indicator green';
                goalStatus.textContent = 'Goal: REACHED';
                goalStatus.style.color = 'green';
            } else {
                navIndicator.className = 'status-indicator red';
                goalStatus.textContent = 'Goal: Not Reached';
                goalStatus.style.color = 'red';
            }
            
            // Update velocity
            document.getElementById('linear-vel').textContent = data.velocity[0].toFixed(2);
            document.getElementById('angular-vel').textContent = data.velocity[2].toFixed(2);
            
            // Update waypoint
            const waypoint = data.waypoint;
            document.getElementById('waypoint-display').textContent = 
                `[${waypoint[0].toFixed(2)}, ${waypoint[1].toFixed(2)}, ${waypoint[2].toFixed(2)}]`;
            
            // Update progress
            const progress = data.navigation_progress;
            document.getElementById('progress-fill').style.width = progress + '%';
            document.getElementById('progress-text').textContent = progress.toFixed(1) + '%';
            
            // Update camera image
            const cameraImg = document.getElementById('camera-image');
            const noCamera = document.getElementById('no-camera');
            if (data.camera_image) {
                cameraImg.src = 'data:image/png;base64,' + data.camera_image;
                cameraImg.style.display = 'block';
                noCamera.style.display = 'none';
            } else {
                cameraImg.style.display = 'none';
                noCamera.style.display = 'block';
            }
            
            // Update trajectory plot
            const trajectoryImg = document.getElementById('trajectory-plot');
            const noTrajectory = document.getElementById('no-trajectory');
            if (data.trajectory_plot) {
                trajectoryImg.src = 'data:image/png;base64,' + data.trajectory_plot;
                trajectoryImg.style.display = 'block';
                noTrajectory.style.display = 'none';
            } else {
                trajectoryImg.style.display = 'none';
                noTrajectory.style.display = 'block';
            }
            
            // Update topomap
            updateTopomap(data.topomap_images, data.current_node, data.goal_node);
            
            // Update detailed info
            document.getElementById('sampled-actions').textContent = data.sampled_actions_count + ' trajectories';
            document.getElementById('precise-waypoint').textContent = 
                `[${waypoint[0].toFixed(3)}, ${waypoint[1].toFixed(3)}, ${waypoint[2].toFixed(3)}]`;
            document.getElementById('precise-velocity').textContent = 
                `Linear=${data.velocity[0].toFixed(3)} m/s, Angular=${data.velocity[2].toFixed(3)} rad/s`;
            document.getElementById('detailed-progress').textContent = 
                `${data.current_node}/${data.goal_node} (${progress.toFixed(1)}%)`;
        }
        
        function updateTopomap(images, currentNode, goalNode) {
            const container = document.getElementById('topomap-container');
            
            if (!images || images.length === 0) {
                container.innerHTML = '<div style="color: #666; text-align: center;">No topomap available</div>';
                return;
            }
            
            let html = '';
            for (let i = 0; i < images.length; i++) {
                let className = 'topomap-item';
                let label = `Node ${i}`;
                
                if (i === currentNode) {
                    className += ' current';
                    label += ' (Current)';
                } else if (i === goalNode) {
                    className += ' goal';
                    label += ' (Goal)';
                }
                
                html += `
                    <div class="${className}">
                        <img src="data:image/png;base64,${images[i]}" alt="Node ${i}">
                        <span>${label}</span>
                    </div>
                `;
            }
            
            container.innerHTML = html;
        }
        
        // Request initial data
        socket.emit('request_data');
        
        // Periodic data request as backup
        setInterval(function() {
            socket.emit('request_data');
        }, 1000);
    </script>
</body>
</html>
'''
    
    with open(os.path.join(template_dir, 'navigation.html'), 'w') as f:
        f.write(html_content)

def main():
    global nav_gui
    
    # 檢查命令行參數
    topomap_dir = "../topomaps/6e-elevator"
    if len(sys.argv) > 1:
        topomap_dir = sys.argv[1]
    
    # 創建 HTML 模板
    create_html_template()
    
    # 創建導航 GUI 實例
    nav_gui = NavigationWebGUI(topomap_dir)
    
    # 啟動 Flask-SocketIO 服務器
    print("Starting navigation web GUI...")
    print("Open your browser to: http://localhost:5000")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down web GUI...")
        if nav_gui:
            nav_gui.cleanup()

if __name__ == "__main__":
    main()
