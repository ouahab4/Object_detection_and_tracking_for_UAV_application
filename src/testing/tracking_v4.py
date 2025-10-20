from ultralytics import YOLO
import cv2
import os

# Load the pretrained YOLOv11s model
model = YOLO("yolo11n.pt")

# Output video path
output_path = "/home/pi/YOLO/v11_visdrone/cpu/webcam_output.mp4"

# Try opening webcam with different indices and backends
for index in [0, 1]:
    for backend in [None, cv2.CAP_V4L2]:
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam opened on /dev/video{index} with backend: {backend}")
            break
    if cap.isOpened():
        break
else:
    print("Error: Could not open any webcam. Check connection, permissions, or drivers.")
    exit()

# Set webcam properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30

print(f"Webcam resolution: {width}x{height} @ {fps} FPS")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    print("Error: Could not create output video file.")
    cap.release()
    exit()

frame_count = 0
skip_factor = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break

    frame_count += 1
    if frame_count % skip_factor != 0:
        out.write(frame)
        continue

    # Perform inference
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to {output_path}")