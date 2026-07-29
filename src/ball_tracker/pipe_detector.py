"""管道检测（诊断用）—— Sobel 边缘投影法，不参与主流程。

此模块保留作为一次性标定诊断工具。
主流程使用校准数据中的固定 ROI（pipe_roi_y1 / pipe_roi_y2）。
"""
import cv2
import numpy as np


def detect(frame: np.ndarray) -> dict | None:
    return None  # 已废弃 — 改用固定 ROI
