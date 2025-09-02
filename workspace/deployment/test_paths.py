#!/usr/bin/env python3

"""
測試腳本：檢查 create_topomap.py 和 navigate.ros2.py 的路徑設定是否一致
"""

import os
import sys

def test_path_consistency():
    # 模擬兩個檔案的路徑設定
    TOPOMAP_IMAGES_DIR_CREATE = "../topomaps"
    TOPOMAP_IMAGES_DIR_NAVIGATE = "../topomaps"
    
    # 測試範例資料夾名稱
    test_folder = "6e6"
    
    # create_topomap.py 的路徑
    create_path = os.path.join(TOPOMAP_IMAGES_DIR_CREATE, test_folder)
    
    # navigate.ros2.py 的路徑  
    navigate_path = os.path.join(TOPOMAP_IMAGES_DIR_NAVIGATE, test_folder)
    
    print("Path consistency test:")
    print(f"create_topomap.py will save to: {create_path}")
    print(f"navigate.ros2.py will read from: {navigate_path}")
    print(f"Paths are consistent: {create_path == navigate_path}")
    
    # 檢查絕對路徑
    abs_create_path = os.path.abspath(create_path)
    abs_navigate_path = os.path.abspath(navigate_path)
    
    print(f"\nAbsolute paths:")
    print(f"create_topomap.py: {abs_create_path}")
    print(f"navigate.ros2.py: {abs_navigate_path}")
    
    return create_path == navigate_path

if __name__ == "__main__":
    is_consistent = test_path_consistency()
    if is_consistent:
        print("\n✅ 路徑設定一致！")
        print("\n使用範例：")
        print("1. 錄製地圖：python create_topomap.py --dir 6e6")
        print("2. 導航：python navigate.ros2.py --dir 6e6") 
        print("\n兩個檔案都會使用 ../topomaps/6e6/ 目錄")
    else:
        print("\n❌ 路徑設定不一致！")
        sys.exit(1)
