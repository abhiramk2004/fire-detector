"""
postprocess.py

Post-processing for Fire Detection.

Responsibilities:
- Decode model output
- Confidence filtering
- Convert bounding boxes
- Undo letterboxing
- Apply NMS
- Return Detection objects
"""

from dataclasses import dataclass

import cv2
import numpy as np
import config


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


def load_labels():
    """Load class labels from labels.txt"""

    with open(config.LABELS_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


LABELS = load_labels()


def postprocess(output, original_shape, scale, pad):
    """
    Convert raw model output into Detection objects.

    Args:
        output: Raw OpenVINO output
        original_shape: Original frame shape
        scale: Resize scale from preprocessing
        pad: (pad_x, pad_y)

    Returns:
        List[Detection]
    """

    h, w = original_shape[:2]

    pad_x, pad_y = pad

    predictions = output[0].T

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:

        x, y, bw, bh, conf, cls = pred

        if conf < config.CONFIDENCE_THRESHOLD:
            continue

        # XYWH → XYXY
        x1 = x - bw / 2
        y1 = y - bh / 2
        x2 = x + bw / 2
        y2 = y + bh / 2

        # Remove padding
        x1 -= pad_x
        x2 -= pad_x
        y1 -= pad_y
        y2 -= pad_y

        # Scale back
        x1 /= scale
        x2 /= scale
        y1 /= scale
        y2 /= scale

        # Clip
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        boxes.append([
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1)
        ])

        scores.append(float(conf))
        class_ids.append(int(cls))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        config.CONFIDENCE_THRESHOLD,
        config.NMS_THRESHOLD
    )

    detections = []

    if len(indices) > 0:

        for idx in indices.flatten():

            x, y, bw, bh = boxes[idx]

            detections.append(
                Detection(
                    class_id=class_ids[idx],
                    label=LABELS[class_ids[idx]],
                    confidence=scores[idx],
                    bbox=(x, y, x + bw, y + bh)
                )
            )

    return detections
