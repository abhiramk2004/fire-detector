import os
import cv2

from detector import FireDetector
from preprocess import preprocess
from postprocess import postprocess
from utils import draw_detections

IMAGE_DIR = "test_image"


def main():

    detector = FireDetector()

    images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not images:
        print("No images found.")
        return

    for image_name in images:

        path = os.path.join(IMAGE_DIR, image_name)

        frame = cv2.imread(path)

        if frame is None:
            continue

        tensor, scale, pad = preprocess(frame)

        output = detector.infer(tensor)

        detections = postprocess(
            output,
            frame.shape,
            scale,
            pad
        )

        draw_detections(frame, detections)

        print("\n", "=" * 60)
        print(image_name)
        print("=" * 60)

        if detections:

            for det in detections:

                print(
                    f"{det.label:<6}"
                    f" Confidence: {det.confidence:.3f}"
                )

        else:

            print("No detections.")

        cv2.imshow("Fire Detector Test", frame)

        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
