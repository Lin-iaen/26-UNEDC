from abc import ABC, abstractmethod

import numpy as np


class BaseTracker(ABC):

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> tuple[dict, np.ndarray]:
        """Run detection / tracking on one frame.

        Args:
            frame: Raw camera frame in RGB format, shape (H, W, 3), dtype uint8.

        Returns:
            tuple[dict, np.ndarray]:
                - dict: Detection results, e.g.
                    {"center_x": float, "center_y": float, "angle": float}.
                - np.ndarray: Debug visualization with annotations drawn, in
                    BGR format (for cv2.imencode / cv2.imwrite), shape (H, W, 3).
        """
