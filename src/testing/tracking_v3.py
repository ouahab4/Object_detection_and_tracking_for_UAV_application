from ultralytics import YOLO
import cv2
import os

# Load the pretrained YOLOv11s model
#model = YOLO("yolo11s.torchscript")  #with ncnn 
model = YOLO("best.pt")

# Path to your input video
#video_path = "/home/pi/YOLO/House_Drone_Shot.mp4"  
#output_path = "/home/pi/YOLO/v11_visdrone/cpu/House_Drone_Shot_out.mp4"


video_path = "/home/pi/YOLO/House_Drone_Shot.mp4"
output_path = "/home/pi/YOLO/v11_visdrone/cpu/parking_garage_out.mp4"


# Open the video.
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
skip_factor = 5  # Process every 5th frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % skip_factor != 0:
        out.write(frame)  # Write original frame
        continue

    # Resize the frame to a smaller resolution (e.g., 640x360)
    frame = cv2.resize(frame, (640, 380))  # (width, height)

    # Perform inference
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the frame with bounding boxes in real-time
    cv2.imshow("YOLOv11 Detection", annotated_frame)

    # Write the frame to the output video
    out.write(annotated_frame)

    # Press 'q' to quit the video display early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to {output_path}")
