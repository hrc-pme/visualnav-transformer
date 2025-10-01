#!/usr/bin/env python3
"""
Test script to verify compressed image subscription
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class CompressedImageTest(Node):
    def __init__(self):
        super().__init__('compressed_image_test')
        
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )
        
        self.count = 0
        self.get_logger().info("Waiting for compressed images...")
    
    def image_callback(self, msg):
        try:
            # 解壓縮影像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                self.get_logger().error("Failed to decompress image")
                return
            
            self.count += 1
            
            # 顯示影像資訊
            height, width, channels = cv_image.shape
            compressed_size = len(msg.data) / 1024  # KB
            
            self.get_logger().info(
                f"Frame #{self.count}: {width}x{height}x{channels}, "
                f"Compressed size: {compressed_size:.2f} KB, "
                f"Format: {msg.format}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main():
    rclpy.init()
    node = CompressedImageTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
