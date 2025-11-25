#!/usr/bin/env python3
"""
Record deployment topics for analysis and debugging.
Records various sensor and navigation topics to a bag file with timestamp.
Press Ctrl+C to stop recording.
"""

import rclpy
from rclpy.node import Node
import yaml
import os
from datetime import datetime
import subprocess
import signal
import sys


class RecordDeployNode(Node):
    def __init__(self):
        super().__init__('record_deploy_node')
        
        # Load configuration (config is in ../config/record.yaml)
        deploy_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(
            deploy_dir,
            'config',
            'record.yaml'
        )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get recording name from config
        self.recording_name = config.get('name', 'default_recording')
        
        # Create timestamp for unique folder name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_folder_name = f"{self.recording_name}_{timestamp}"
        
        # Set up bags directory (bags is in ../bags/)
        self.bags_dir = os.path.join(
            deploy_dir,
            'bags',
            bag_folder_name
        )
        
        # Create bags directory if it doesn't exist
        os.makedirs(self.bags_dir, exist_ok=True)
        
        self.get_logger().info(f'Recording will be saved to: {self.bags_dir}')
        
        # Define topics to record
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
        
        # Start recording
        self.start_recording()
    
    def start_recording(self):
        """Start ros2 bag record process"""
        # Build the ros2 bag record command
        cmd = ['ros2', 'bag', 'record', '-o', self.bags_dir] + self.topics
        
        self.get_logger().info('Starting recording...')
        self.get_logger().info(f'Recording topics: {", ".join(self.topics)}')
        self.get_logger().info('Press Ctrl+C to stop recording')
        
        # Start the recording process
        try:
            self.record_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for the process to finish (will be terminated by Ctrl+C)
            self.record_process.wait()
            
        except KeyboardInterrupt:
            self.get_logger().info('Stopping recording...')
            self.stop_recording()
        except Exception as e:
            self.get_logger().error(f'Error during recording: {str(e)}')
            self.stop_recording()
    
    def stop_recording(self):
        """Stop the recording process"""
        if hasattr(self, 'record_process'):
            self.record_process.terminate()
            try:
                self.record_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.record_process.kill()
        
        self.get_logger().info(f'Recording saved to: {self.bags_dir}')


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nStopping recording...')
    sys.exit(0)


def main(args=None):
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    rclpy.init(args=args)
    
    try:
        node = RecordDeployNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
