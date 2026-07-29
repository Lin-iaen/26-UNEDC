import cv2
import numpy as np

# 创建一张黑色背景的图片
img = np.zeros((480, 800, 3), dtype=np.uint8)

# 在图片上写字
cv2.putText(img, "Cage + OpenCV Works!", (50, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

# 创建全屏窗口
cv2.namedWindow("HDMI Test", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("HDMI Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("正在推送画面到 HDMI，5秒后自动退出...")

# 显示图片
cv2.imshow("HDMI Test", img)

# 等待 5000 毫秒 (5秒)
cv2.waitKey(5000)

# 清理退出
cv2.destroyAllWindows()