from video import VideoStream
from preprocess import preprocess
from detector import FireDetector

video = VideoStream()
detector = FireDetector()

if not video.open():
    print("Could not connect to stream.")
    exit()

ret, frame = video.read()

if not ret:
    print("Could not read frame.")
    exit()

input_tensor, scale, pad = preprocess(frame)

output = detector.infer(input_tensor)

print("\n========== Inference Results ==========")
print("Input Tensor Shape :", input_tensor.shape)
print("Output Shape       :", output.shape)
print("Output Type        :", output.dtype)
print("Output Min         :", output.min())
print("Output Max         :", output.max())

video.release()
