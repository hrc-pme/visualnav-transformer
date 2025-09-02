#!/usr/bin/env python3

"""
測試腳本：檢查修改後的 TOPOMAP_NAME 設定
"""

import os
import sys

def test_topomap_name_consistency():
    # 從檔案中讀取 TOPOMAP_NAME 設定
    create_topomap_file = "/home/pomelo925/Desktop/helloRobot_stretch3/visualnav-transformer/workspace/deployment/src/create_topomap.py"
    navigate_file = "/home/pomelo925/Desktop/helloRobot_stretch3/visualnav-transformer/workspace/deployment/src/navigate.ros2.py"
    
    # 提取 TOPOMAP_NAME 從兩個檔案
    create_topomap_name = None
    navigate_topomap_name = None
    
    with open(create_topomap_file, 'r') as f:
        for line in f:
            if line.strip().startswith('TOPOMAP_NAME = '):
                create_topomap_name = line.split('=')[1].strip().strip('"\'')
                break
    
    with open(navigate_file, 'r') as f:
        for line in f:
            if line.strip().startswith('TOPOMAP_NAME = '):
                navigate_topomap_name = line.split('=')[1].strip().strip('"\'')
                break
    
    print("TOPOMAP_NAME consistency test:")
    print(f"create_topomap.py TOPOMAP_NAME: {create_topomap_name}")
    print(f"navigate.ros2.py TOPOMAP_NAME: {navigate_topomap_name}")
    
    if create_topomap_name and navigate_topomap_name:
        is_consistent = create_topomap_name == navigate_topomap_name
        print(f"TOPOMAP_NAME values are consistent: {is_consistent}")
        
        if is_consistent:
            topomap_name = create_topomap_name.strip('"\'')
            topomap_path = f"../topomaps/{topomap_name}"
            print(f"\n✅ 設定一致！")
            print(f"地圖名稱: {topomap_name}")
            print(f"地圖路徑: {topomap_path}")
            print(f"\n使用方式:")
            print("1. 錄製地圖：python3 create_topomap.py")
            print("2. 導航：python3 navigate.ros2.py")
            print(f"\n兩個檔案都會使用 {topomap_path}/ 目錄")
            
            return True
        else:
            print("\n❌ TOPOMAP_NAME 設定不一致！")
            return False
    else:
        print("\n❌ 無法找到 TOPOMAP_NAME 設定！")
        return False

if __name__ == "__main__":
    is_consistent = test_topomap_name_consistency()
    if not is_consistent:
        sys.exit(1)
