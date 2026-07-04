from video import VideoStream
import cv2

video = VideoStream()

while True:
    ret, frame = video.read()

    if not ret:
        continue

    cv2.imshow("Video Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
