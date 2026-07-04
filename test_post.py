from video import VideoStream
from preprocess import preprocess
from detector import FireDetector
from postprocess import postprocess

video = VideoStream()
detector = FireDetector()

video.open()

ret, frame = video.read()

tensor, scale, pad = preprocess(frame)

output = detector.infer(tensor)

detections = postprocess(
    output,
    frame.shape,
    scale,
    pad
)

print("\nDetections")

print("-" * 40)

for d in detections:
    print(d)

video.release()
