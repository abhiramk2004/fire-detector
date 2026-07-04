# Fire Detection Service

A standalone AI microservice for real-time **Fire** and **Smoke** detection from an HTTP video stream using **OpenVINO Runtime** on Intel CPUs.

This service is designed as a modular component of a larger AI surveillance platform. It operates independently and does not depend on Frigate or any external AI inference service.

---

# Features

* Real-time Fire and Smoke detection
* CPU-only inference using OpenVINO Runtime
* HTTP camera stream support
* Automatic stream reconnection
* Modular project architecture
* Bounding box visualization
* Confidence score display
* FPS monitoring
* Easily extendable for Docker, MQTT, and notifications

---

# Current Architecture

```text
                        HTTP Camera Stream
                                │
                                ▼
                      +-------------------+
                      |   video.py        |
                      | OpenCV Capture    |
                      +-------------------+
                                │
                                ▼
                      +-------------------+
                      | preprocess.py     |
                      | Letterbox Resize  |
                      | RGB Conversion    |
                      | Normalization     |
                      +-------------------+
                                │
                                ▼
                      +-------------------+
                      | detector.py       |
                      | OpenVINO Runtime  |
                      | Fire Model        |
                      +-------------------+
                                │
                                ▼
                      +-------------------+
                      | postprocess.py    |
                      | Decode Output     |
                      | Confidence Filter |
                      | NMS               |
                      +-------------------+
                                │
                                ▼
                      +-------------------+
                      | utils.py          |
                      | Draw Boxes & FPS  |
                      +-------------------+
                                │
                                ▼
                           Display Output
```

---

# Repository Structure

```text
fire-detector/
│
├── main.py                 # Application entry point
├── config.py               # Central configuration
├── video.py                # HTTP video stream handling
├── preprocess.py           # Image preprocessing
├── detector.py             # OpenVINO inference
├── postprocess.py          # Decode detections
├── utils.py                # Drawing utilities & FPS
│
├── models/
│   └── openvino/
│       ├── fire.xml
│       └── fire.bin
│
├── labels.txt
├── requirements.txt
└── README.md
```

---

# Detection Pipeline

```text
HTTP Camera
      │
      ▼
OpenCV Video Capture
      │
      ▼
Preprocessing
  • Letterbox Resize
  • BGR → RGB
  • Normalization
      │
      ▼
OpenVINO Runtime
      │
      ▼
YOLOv8 Fire/Smoke Model
      │
      ▼
Postprocessing
  • Confidence Filtering
  • NMS
  • Bounding Boxes
      │
      ▼
Visualization
      │
      ▼
Output Display
```

---

# Configuration

Most runtime settings can be adjusted from **`config.py`** without modifying the application code.

Examples include:

### Camera

* HTTP camera URL
* Stream source
* Reconnection delay
* Buffer size

### Model

* OpenVINO model path (`fire.xml`)
* Inference device (`CPU`, `AUTO`, etc.)
* Labels file

### Detection

* Confidence threshold
* NMS threshold

### Input

* Model input resolution
* Display resolution

### Display

* Window title
* FPS overlay
* Bounding box thickness
* Font scale

These settings make it easy to adapt the service for different cameras, models, or deployment environments.

---

# Model Information

**Architecture**

* YOLOv8 Nano

**Inference Backend**

* OpenVINO Runtime

**Input**

* RGB
* Float32
* 480 × 480

**Output**

```
(1, 6, 4725)
```

Each prediction contains:

```
[x_center,
 y_center,
 width,
 height,
 confidence,
 class_id]
```

Classes:

| ID | Label |
| -- | ----- |
| 0  | Smoke |
| 1  | Fire  |

---

# Running the Service

## 1. Create a virtual environment

```bash
python3 -m venv convert-env
source convert-env/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure the camera URL

Edit `config.py` and update:

```python
STREAM_URL = "http://<camera-ip>:5000/video"
```

## 4. Place the OpenVINO model

```
models/
└── openvino/
    ├── fire.xml
    └── fire.bin
```

## 5. Start the detector

```bash
python main.py
```

---

# Current Status

### Completed

* HTTP video streaming
* Automatic stream reconnection
* Image preprocessing
* OpenVINO inference
* Fire detection
* Smoke detection
* Bounding box visualization
* FPS monitoring
* End-to-end pipeline validation

The detector has been successfully validated on sample fire, smoke, and normal images, with correct detection and bounding box rendering.

---

# Future Improvements

Possible enhancements include:

* Docker containerization
* Docker Compose deployment
* MQTT event publishing
* Frigate integration
* Email/SMS/Telegram notifications
* Multi-camera support
* REST API for detections
* Detection logging and event history
* Snapshot and video clip generation
* Performance benchmarking
* INT8 quantization for faster CPU inference
* Support for alternative detection models
* Hardware acceleration on Intel iGPU/NPU where available

---

# Notes

The project is intentionally modular. Each component has a single responsibility, making it easy to replace the detection model, change the video source, or integrate additional services without affecting the overall architecture.
