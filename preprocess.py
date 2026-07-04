"""
preprocess.py

Image preprocessing for OpenVINO Fire Detection.

Responsibilities:
- Letterbox resize while preserving aspect ratio
- Convert BGR -> RGB
- Normalize image
- Convert HWC -> CHW
- Add batch dimension
"""

import cv2
import numpy as np
import config


def letterbox(image,
              new_shape=(config.INPUT_WIDTH, config.INPUT_HEIGHT),
              color=(114, 114, 114)):
    """
    Resize image while preserving aspect ratio using letterboxing.

    Returns:
        image      : Letterboxed image
        scale      : Resize scale
        (dw, dh)   : Padding applied (left/right, top/bottom)
    """

    h, w = image.shape[:2]

    new_w, new_h = new_shape

    # Compute resize scale
    scale = min(new_w / w, new_h / h)

    # Compute resized dimensions
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    # Resize image
    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Compute padding
    dw = new_w - resized_w
    dh = new_h - resized_h

    dw /= 2
    dh /= 2

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # Add border
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, (left, top)


def preprocess(frame):
    """
    Convert OpenCV frame into OpenVINO input tensor.

    Returns:
        input_tensor
        scale
        pad
    """

    image, scale, pad = letterbox(frame)

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # uint8 -> float32
    image = image.astype(np.float32)

    # Normalize
    image /= 255.0

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # CHW -> NCHW
    image = np.expand_dims(image, axis=0)

    # Ensure contiguous memory
    image = np.ascontiguousarray(image)

    return image, scale, pad
