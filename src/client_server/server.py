import cv2
from picamera2 import Picamera2
import socket
import struct
import time
import threading

# Initialize PiCamera2
picam2 = Picamera2()
width, height = 480, 360
camera_config = picam2.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
picam2.configure(camera_config)
picam2.start()

# Camera properties
fps = 30
frame_delay = 1.0 / fps

# Thread-safe flag
running = True
video_sequence_number = 0

# UDP socket setup
video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_address = ('192.168.158.195', 9999)  # PC's IP and video port

def video_thread():
    global running, video_sequence_number
    try:
        while running:
            start_time = time.time()
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Serialize frame
            success, frame_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not success:
                print(f"Video packet {video_sequence_number}: Failed to encode frame")
                continue
            frame_data = frame_encoded.tobytes()
            if not frame_data:
                print(f"Video packet {video_sequence_number}: Empty frame data")
                continue

            timestamp = int(time.time() * 1000)
            header = struct.pack(">LBBLLQ", video_sequence_number, 1, 0, len(frame_data), 0, timestamp)
            packet = header + frame_data
            if len(packet) > 65507:
                print(f"Video packet {video_sequence_number} too large: {len(packet)} bytes")
                continue

            video_socket.sendto(packet, client_address)
            print(f"Sent video packet {video_sequence_number}: frame_size={len(frame_data)}, timestamp={timestamp}")
            video_sequence_number = (video_sequence_number + 1) % 0xFFFFFFFF

            elapsed = time.time() - start_time
            time.sleep(max(0, frame_delay - elapsed))
    finally:
        running = False

def command_thread():
    global running
    video_socket.settimeout(0.1)
    while running:
        try:
            data, addr = video_socket.recvfrom(1024)
            cmd = data.decode()
            print(f"Received command: {cmd} from {addr}")
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Command error: {e}")

def cleanup():
    global running
    running = False
    picam2.stop()
    video_socket.close()

if __name__ == "__main__":
    try:
        threading.Thread(target=video_thread, daemon=True).start()
        threading.Thread(target=command_thread, daemon=True).start()
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
