"""
utils.py

Utility functions for Fire Detection Service.

Responsibilities:
- FPS calculation
- Drawing detections
- Drawing FPS
"""

import time
import cv2
import config


# ==========================================================
# Colors (BGR)
# ==========================================================

COLORS = {
    0: (128, 128, 128),   # Smoke - Gray
    1: (0, 0, 255),       # Fire - Red
}


# ==========================================================
# FPS Counter
# ==========================================================

class FPSCounter:
    """Simple moving FPS calculator."""

    def __init__(self):

        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def update(self):

        self.frame_count += 1

        elapsed = time.time() - self.start_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        return self.fps


# ==========================================================
# Drawing Functions
# ==========================================================

def draw_detections(frame, detections):
    """
    Draw detections on frame.

    Args:
        frame
        detections : List[Detection]

    Returns:
        Annotated frame
    """

    for det in detections:

        x1, y1, x2, y2 = det.bbox

        color = COLORS.get(det.class_id, (255, 255, 255))

        # Bounding Box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            config.BOX_THICKNESS
        )

        label = f"{det.label} {det.confidence:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            1
        )

        # Filled rectangle behind text
        cv2.rectangle(
            frame,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 6, y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return frame


def draw_fps(frame, fps):
    """
    Draw FPS on frame.
    """

    if not config.SHOW_FPS:
        return frame

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return frame
