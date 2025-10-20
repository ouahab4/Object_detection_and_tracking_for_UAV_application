import cv2
import math
from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')

current_results = None
selected_id = None
frame_center = None
tracker = None
tracking = False
current_frame = None

cap = cv2.VideoCapture('./21.mp4')
cv2.namedWindow('YOLO Object Tracking')

def mouse_callback(event, x, y, flags, param):
    global selected_id, current_results, tracker, tracking, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_id = None
        tracking = False
        if current_results is not None and current_frame is not None:
            for box in current_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2 and box.id is not None:
                    selected_id = int(box.id.item())
                    print(f"Selected Object ID: {selected_id}")
                    
                    #tracker initialization
                    tracker = cv2.TrackerCSRT_create()
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    tracker.init(current_frame, bbox)
                    tracking = True
                    break

cv2.setMouseCallback('YOLO Object Tracking', mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    current_frame = frame.copy() 

    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    results = model.track(frame, persist=True)
    current_results = results
    annotated_frame = results[0].plot()

    cv2.circle(annotated_frame, frame_center, 5, (0, 0, 255), -1)

    if tracking and tracker is not None:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            obj_center = (x + w // 2, y + h // 2)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.line(annotated_frame, frame_center, obj_center, (255, 0, 0), 2)

            dx = obj_center[0] - frame_center[0]
            dy = obj_center[1] - frame_center[1]
            angle = math.degrees(math.atan2(dy, dx))
            angle = angle if angle >= 0 else angle + 360

            distance_pixels = math.sqrt(dx**2 + dy**2)
            rel_x = dx / (width / 2)
            rel_y = dy / (height / 2)

            cv2.putText(annotated_frame, f"Tracked (CSRT): {selected_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Angle: {angle:.1f}°", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Distance: {distance_pixels:.1f} px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Rel X: {rel_x:.2f}, Rel Y: {rel_y:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if abs(rel_x) > 0.1:
                direction = "RIGHT" if rel_x > 0 else "LEFT"
                cv2.putText(annotated_frame, f"Turn {direction}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(annotated_frame, "Tracking lost", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            tracking = False

    cv2.imshow('YOLO Object Tracking', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
