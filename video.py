"""
video.py

Video stream handling for the Fire Detection Service.
Responsible for:
- Opening the HTTP stream
- Managing the VideoCapture object
- Releasing resources
"""

import cv2
import config
import time

class VideoStream:
    """Handles camera connection and frame acquisition."""

    def __init__(self):
        self.url = config.STREAM_URL
        self.cap = None
        self.is_connected = False

    def open(self):
        """Open the video stream."""

        print(f"[INFO] Connecting to: {self.url}")

        self.cap = cv2.VideoCapture(self.url)

        if self.cap is None or not self.cap.isOpened():
            print("[ERROR] Failed to connect to video stream.")
            self.is_connected = False
            return False

        # Reduce latency by minimizing the capture buffer
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.BUFFER_SIZE)

        self.is_connected = True
        print("[INFO] Video stream connected successfully.")

        return True

    def release(self):
        """Release the video stream."""

        if self.cap is not None:
            self.cap.release()

        self.is_connected = False
        print("[INFO] Video stream released.")
    def read(self):
    # If we're disconnected, try to reconnect
        if not self.is_connected:
            if not self.open():
                time.sleep(config.RECONNECT_DELAY)
                return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            print("[WARNING] Frame read failed. Reconnecting...")

            self.release()
            time.sleep(config.RECONNECT_DELAY)

            return False, None

        return True, frame
