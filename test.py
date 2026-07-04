from video import VideoStream
from preprocess import preprocess
from detector import FireDetector
import numpy as np

video = VideoStream()
detector = FireDetector()

video.open()

ret, frame = video.read()

tensor, scale, pad = preprocess(frame)

output = detector.infer(tensor)

print(output.shape)

# Remove batch dimension
pred = output[0]

print(pred.shape)

# Print first 10 predictions
np.set_printoptions(suppress=True, precision=4)

for i in range(10):
    print(f"\nPrediction {i}")
    print(pred[:, i])

video.release()
