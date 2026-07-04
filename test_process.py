from video import VideoStream
from preprocess import preprocess

video = VideoStream()

if not video.open():
    print("Failed to open stream.")
    exit()

ret, frame = video.read()

if not ret:
    print("Failed to read frame.")
    exit()

tensor, scale, pad = preprocess(frame)

print("Original Shape :", frame.shape)
print("Tensor Shape   :", tensor.shape)
print("Tensor Type    :", tensor.dtype)
print("Scale          :", scale)
print("Padding        :", pad)

video.release()
