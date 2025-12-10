#!/usr/bin/env python3
"""
Test GUI - 靜態網格地圖視覺化
顯示一個 Tesla 風格的透視網格地圖
"""

import cv2
import numpy as np


def create_static_minimap():
    """創建靜態的網格地圖"""
    
    mini_w, mini_h = 400, 400  # 較大視窗以便觀察
    mini = np.zeros((mini_h, mini_w, 3), dtype=np.uint8)
    mini[:, :, :] = (30, 30, 30)  # 深灰色背景

    # 地板區域（下方 70%）
    floor_y0 = int(mini_h * 0.30)
    floor_y1 = mini_h

    # 投影參數
    depth_px = floor_y1 - floor_y0
    lateral_px = mini_w * 4  # 增加橫向投影範圍
    shrink_factor = 0.75
    gamma = 1.25

    # 世界範圍（擴大到能填滿整個視窗，包含上方 30% 天空區域）
    X_MIN = -0.5  # 允許負值以延伸到底部和頂部
    X_MAX = 1.0   # 大幅增加深度範圍
    Y_MAX = 1.8   # 大幅增加橫向範圍，確保完全覆蓋左右邊界

    # 世界座標 → 螢幕座標轉換
    def world_to_screen(X, Y):
        t = (X - X_MIN) / (X_MAX - X_MIN)  # 不限制範圍，允許超出
        t2 = t ** gamma

        v = int(floor_y1 - depth_px * t2)
        width_here = (1.0 - shrink_factor * t2) * lateral_px
        u = int(mini_w / 2 + (Y / Y_MAX) * width_here)

        alpha = int((1 - t2) ** 2 * 255) if 0 <= t <= 1 else 0
        return u, v, alpha

    # ==========================================
    # (1) 繪製網格線（填滿整個視窗）
    # ==========================================
    # 大幅增加網格密度以確保完全覆蓋
    grid_xs = np.linspace(X_MIN, X_MAX, 25)  # 增加縱向密度
    grid_ys = np.linspace(-Y_MAX, Y_MAX, 50)  # 增加橫向密度

    # 縱向網格線（確保延伸到整個視窗高度）
    for Xg in grid_xs:
        points = []
        for Yg in np.linspace(-Y_MAX, Y_MAX, 150):  # 增加採樣點
            u, v, a = world_to_screen(Xg, Yg)
            points.append((u, v))
        
        # 繪製所有線段，根據 y 座標計算透明度（使用漸變）
        for i in range(len(points) - 1):
            u1, v1 = points[i]
            u2, v2 = points[i+1]
            
            # 漸變區域設定：從 0% (頂部) 到 30% (底部起始點) 高度
            fade_start = 0  # 頂部，完全透明
            fade_end = mini_h * 0.30  # 30% 高度，完全不透明
            
            # 計算兩個端點的透明度
            def calc_alpha(y):
                if y >= fade_end:
                    return 1.0  # 下方區域，完全不透明
                elif y <= fade_start:
                    return 0.0  # 頂部，完全透明
                else:
                    # 漸變區域：使用平滑的漸變曲線
                    t = (y - fade_start) / (fade_end - fade_start)
                    return t ** 0.5  # 使用平方根讓漸變更平滑
            
            alpha1 = calc_alpha(v1)
            alpha2 = calc_alpha(v2)
            avg_alpha = (alpha1 + alpha2) / 2
            
            # 只繪製有透明度的線段
            if avg_alpha > 0:
                color = tuple(int(70 * avg_alpha) for _ in range(3))
                cv2.line(mini, (u1, v1), (u2, v2), color, 1, cv2.LINE_AA)

    # 橫向網格線（確保延伸到整個視窗寬度和高度）
    for Yg in grid_ys:
        points = []
        for Xg in np.linspace(X_MIN, X_MAX, 150):  # 增加採樣點
            u, v, a = world_to_screen(Xg, Yg)
            points.append((u, v))
        
        # 繪製所有線段，根據 y 座標計算透明度（使用漸變）
        for i in range(len(points) - 1):
            u1, v1 = points[i]
            u2, v2 = points[i+1]
            
            # 漸變區域設定：從 0% (頂部) 到 30% (底部起始點) 高度
            fade_start = 0  # 頂部，完全透明
            fade_end = mini_h * 0.30  # 30% 高度，完全不透明
            
            # 計算兩個端點的透明度
            def calc_alpha(y):
                if y >= fade_end:
                    return 1.0  # 下方區域，完全不透明
                elif y <= fade_start:
                    return 0.0  # 頂部，完全透明
                else:
                    # 漸變區域：使用平滑的漸變曲線
                    t = (y - fade_start) / (fade_end - fade_start)
                    return t ** 0.5  # 使用平方根讓漸變更平滑
            
            alpha1 = calc_alpha(v1)
            alpha2 = calc_alpha(v2)
            avg_alpha = (alpha1 + alpha2) / 2
            
            # 只繪製有透明度的線段
            if avg_alpha > 0:
                color = tuple(int(70 * avg_alpha) for _ in range(3))
                cv2.line(mini, (u1, v1), (u2, v2), color, 1, cv2.LINE_AA)

    # ==========================================
    # (2) 中央縱軸（y=0）
    # ==========================================
    points = []
    for Xg in np.linspace(X_MIN, X_MAX, 100):
        u, v, a = world_to_screen(Xg, 0.0)
        points.append((u, v))
    
    for i in range(len(points) - 1):
        u1, v1 = points[i]
        u2, v2 = points[i+1]
        
        # 漸變區域設定
        fade_start = 0
        fade_end = mini_h * 0.30
        
        def calc_alpha(y):
            if y >= fade_end:
                return 1.0
            elif y <= fade_start:
                return 0.0
            else:
                t = (y - fade_start) / (fade_end - fade_start)
                return t ** 0.5
        
        alpha1 = calc_alpha(v1)
        alpha2 = calc_alpha(v2)
        avg_alpha = (alpha1 + alpha2) / 2
        
        if avg_alpha > 0:
            color = tuple(int(180 * avg_alpha) for _ in range(3))
            cv2.line(mini, (u1, v1), (u2, v2), color, 4, cv2.LINE_AA)

    # ==========================================
    # (3) 底部橫軸（x=0）
    # ==========================================
    u1, v1, _ = world_to_screen(0.0, -Y_MAX)
    u2, v2, _ = world_to_screen(0.0,  Y_MAX)
    
    # 漸變區域設定
    fade_start = 0
    fade_end = mini_h * 0.30
    
    def calc_alpha(y):
        if y >= fade_end:
            return 1.0
        elif y <= fade_start:
            return 0.0
        else:
            t = (y - fade_start) / (fade_end - fade_start)
            return t ** 0.5
    
    alpha1 = calc_alpha(v1)
    alpha2 = calc_alpha(v2)
    avg_alpha = (alpha1 + alpha2) / 2
    
    if avg_alpha > 0:
        color = tuple(int(180 * avg_alpha) for _ in range(3))
        cv2.line(mini, (u1, v1), (u2, v2), color, 4, cv2.LINE_AA)

    # ==========================================
    # (4) 原點標記
    # ==========================================
    u0, v0, _ = world_to_screen(0.0, 0.0)
    cv2.circle(mini, (u0, v0), 8, (0, 255, 0), -1)
    cv2.circle(mini, (u0, v0), 10, (255, 255, 255), 2, cv2.LINE_AA)

    # ==========================================
    # (5) 添加標題
    # ==========================================
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Tesla-Style Minimap Test"
    cv2.putText(mini, title, (20, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    info = "Press ESC or 'q' to exit"
    cv2.putText(mini, info, (20, mini_h - 20), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    return mini


def main():
    """主程式"""
    print("=" * 60)
    print("Tesla-Style Minimap Test GUI")
    print("=" * 60)
    print("Creating static minimap...")
    
    # 創建靜態地圖
    minimap = create_static_minimap()
    
    print("✅ Minimap created successfully!")
    print("Displaying window... (Press ESC or 'q' to exit)")
    
    # 顯示視窗
    window_name = "Test GUI - Static Minimap"
    cv2.imshow(window_name, minimap)
    
    # 等待按鍵
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
    
    # 清理
    cv2.destroyAllWindows()
    print("✅ Window closed. Test completed.")


if __name__ == "__main__":
    main()
