import numpy as np
import cv2
import os

cap = cv2.VideoCapture("../bunny.mp4")  # Replace with your video file
file_size = os.path.getsize("../bunny.mp4")  # Size in bytes

print(f"Video file size: {file_size / (1024 * 1024):.2f} MB")  # Convert to MB
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

def play_video_from_frames(frames, fps=30):
    window_name = "Video Playback"
    cv2.namedWindow(window_name)  # Create a named window

    for frame in frames:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break  # Exit if the window is closed manually

        cv2.imshow(window_name, frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break  # Exit if 'q' is pressed

    cv2.destroyAllWindows()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # Stop if the video ends or cannot be read
#
#     cv2.imshow("Video", frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
#         break

frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)  # Each frame is a NumPy array

frames = np.array(frames)
print(frames.shape)
print(frames.dtype)

num_frames, height, width, _ = frames.shape

print(f'{(num_frames*height*width)//(1024*1024)} MB')
# play_video_from_frames(frames)

cap.release()
# cv2.destroyAllWindows()