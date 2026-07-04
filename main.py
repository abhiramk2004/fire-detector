"""
main.py

Fire Detection Service

Pipeline:

HTTP Stream
    ↓
Preprocess
    ↓
OpenVINO Inference
    ↓
Postprocess
    ↓
Draw Results
    ↓
Display
"""

import cv2

import config
from video import VideoStream
from preprocess import preprocess
from detector import FireDetector
from postprocess import postprocess
from utils import FPSCounter, draw_detections, draw_fps


def main():

    print("=" * 60)
    print(" Fire Detection Service ")
    print("=" * 60)

    # -------------------------------------------------
    # Initialize modules
    # -------------------------------------------------

    video = VideoStream()
    detector = FireDetector()
    fps_counter = FPSCounter()

    # -------------------------------------------------
    # Open camera
    # -------------------------------------------------

    if not video.open():
        print("[ERROR] Unable to open video stream.")
        return

    print("[INFO] Starting detection...")

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------

    try:

        while True:

            ret, frame = video.read()

            if not ret:
                continue

            # -----------------------------
            # Preprocess
            # -----------------------------

            input_tensor, scale, pad = preprocess(frame)

            # -----------------------------
            # Inference
            # -----------------------------

            output = detector.infer(input_tensor)

            # -----------------------------
            # Postprocess
            # -----------------------------

            detections = postprocess(
                output,
                frame.shape,
                scale,
                pad
            )

            # -----------------------------
            # Draw
            # -----------------------------

            draw_detections(frame, detections)

            fps = fps_counter.update()
            draw_fps(frame, fps)

            # -----------------------------
            # Display
            # -----------------------------

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:

        print("[INFO] Shutting down...")

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
