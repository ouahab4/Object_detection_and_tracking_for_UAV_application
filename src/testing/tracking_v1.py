import cv2
import math
from ultralytics import YOLO

model = YOLO('runs/detect/train5/weights/best.pt')

current_results = None
selected_id = None
frame_center = None

def mouse_callback(event, x, y, flags, param):
    global selected_id, current_results
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_id = None
        if current_results is not None:
            for box in current_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2 and box.id is not None:
                    selected_id = int(box.id.item())
                    print(f"Selected Object ID: {selected_id}")
                    break

def print_tracking_info(selected_id, distance_pixels, angle, dx, dy, rel_x, rel_y, turn_direction=None):
    print(f"\n--- Tracking Info for Object ID {selected_id} ---")
    print(f"Distance Vector (dx, dy): ({dx:.1f}, {dy:.1f}) pixels")
    print(f"Euclidean Distance: {distance_pixels:.1f} pixels")
    print(f"Angle: {angle:.1f} degrees (0° is right, 90° is down)")
    print(f"Relative Position: Rel X: {rel_x:.2f}, Rel Y: {rel_y:.2f}")
    if turn_direction:
        print(f"Drone Action: TURN {turn_direction}")
    else:
        print("Drone Action: HOLD/ON TARGET")
    print("---------------------------------------------")

cap = cv2.VideoCapture('./21.mp4')
cv2.namedWindow('YOLO Object Tracking')
cv2.setMouseCallback('YOLO Object Tracking', mouse_callback)

last_printed_id = None
last_printed_turn = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    
    results = model.track(frame, persist=True)
    current_results = results

    
    annotated_frame = results[0].plot()

    cv2.circle(annotated_frame, frame_center, 5, (0, 0, 255), -1)

    if selected_id is not None:
        for box in results[0].boxes:
            if box.id is not None and int(box.id.item()) == selected_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                cv2.line(annotated_frame, frame_center, obj_center, (255, 0, 0), 2)

                #angle
                dx = obj_center[0] - frame_center[0]
                dy = obj_center[1] - frame_center[1]
                angle = math.degrees(math.atan2(dy, dx))
                angle = angle if angle >= 0 else angle + 360  

                #estimatemated distance 
                distance_pixels = math.sqrt(dx**2 + dy**2)

                #relative position
                rel_x = dx / (width / 2)
                rel_y = dy / (height / 2)

               
                cv2.putText(annotated_frame, f"Tracked: {selected_id}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Angle: {angle:.1f}°", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Distance: {distance_pixels:.1f} px", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Rel X: {rel_x:.2f}, Rel Y: {rel_y:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"dx: {dx:.1f}, dy: {dy:.1f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

                #drone control logic
                turn_direction = None
                if abs(rel_x) > 0.1:  #if significantly off-center horizontally
                    turn_direction = "RIGHT" if rel_x > 0 else "LEFT"
                    cv2.putText(annotated_frame, f"Turn {turn_direction}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(annotated_frame, "On Target", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                #only print to terminal if tracking object or turn direction changes
                if (last_printed_id != selected_id) or (last_printed_turn != turn_direction):
                    print_tracking_info(selected_id, distance_pixels, angle, dx, dy, rel_x, rel_y, turn_direction)
                    last_printed_id = selected_id
                    last_printed_turn = turn_direction

                break

    cv2.imshow('YOLO Object Tracking', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()