#!/usr/bin/env python3
"""
Check and install required ROS2 dependencies for camera calibration
"""
import subprocess
import sys
import os

def check_and_install_ros_packages():
    """Check if required ROS2 packages are installed, install if missing"""
    required_packages = [
        'ros-humble-control-msgs',
        'ros-humble-trajectory-msgs', 
        'ros-humble-action-msgs',
        'ros-humble-rmw-cyclonedds-cpp'
    ]
    
    missing_packages = []
    
    print("🔍 Checking ROS2 dependencies for camera calibration...")
    
    for package in required_packages:
        # Check if package is installed
        result = subprocess.run(
            ['dpkg', '-l', package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            missing_packages.append(package)
            print(f"  ❌ Missing: {package}")
        else:
            print(f"  ✅ Found: {package}")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing package(s)...")
        try:
            # Update apt cache
            subprocess.run(['apt-get', 'update'], check=True, capture_output=True)
            
            # Install missing packages
            install_cmd = ['apt-get', 'install', '-y'] + missing_packages
            subprocess.run(install_cmd, check=True, capture_output=True)
            
            print("✅ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            print("   Please run manually:")
            print(f"   sudo apt-get update && sudo apt-get install -y {' '.join(missing_packages)}")
            return False
    else:
        print("✅ All required packages are already installed!")
        return True

def set_cyclonedds():
    """Set CycloneDDS as RMW implementation"""
    if 'RMW_IMPLEMENTATION' not in os.environ:
        os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
        print("🔧 Set RMW_IMPLEMENTATION=rmw_cyclonedds_cpp")
    else:
        print(f"🔧 Using RMW_IMPLEMENTATION={os.environ['RMW_IMPLEMENTATION']}")

if __name__ == "__main__":
    check_and_install_ros_packages()
    set_cyclonedds()
