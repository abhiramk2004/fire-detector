"""
config.py

Central configuration for the Fire Detection Service.

This file contains only configuration constants.
No application logic should be placed here.
"""

# ==========================================================
# Camera Configuration
# ==========================================================

# HTTP camera stream URL
# Replace <camera-ip> with the actual IP address later.
STREAM_URL = "http://192.168.10.100:5000/video"


# ==========================================================
# Model Configuration
# ==========================================================

# OpenVINO model (.xml)
MODEL_PATH = "models/openvino/fire.xml"

# OpenVINO inference device
# Options: "CPU", "AUTO"
DEVICE = "CPU"


# ==========================================================
# Input Configuration
# ==========================================================

# Model input size
INPUT_WIDTH = 480
INPUT_HEIGHT = 480


# ==========================================================
# Detection Configuration
# ==========================================================

# Minimum confidence required to keep a detection
CONFIDENCE_THRESHOLD = 0.30

# Non-Maximum Suppression threshold
NMS_THRESHOLD = 0.45


# ==========================================================
# Display Configuration
# ==========================================================

WINDOW_NAME = "Fire Detection"

# Show FPS on output window
SHOW_FPS = True

# Thickness of bounding boxes
BOX_THICKNESS = 2

# Font scale used for labels
FONT_SCALE = 0.6


# ==========================================================
# Labels
# ==========================================================

LABELS_FILE = "labels.txt"


# ==========================================================
# Video Stream Configuration
# ==========================================================

# Delay (seconds) before attempting to reconnect
RECONNECT_DELAY = 2

# OpenCV capture buffer size
BUFFER_SIZE = 1


# ==========================================================
# Performance
# ==========================================================

# Target display resolution
DISPLAY_WIDTH = 704
DISPLAY_HEIGHT = 480
