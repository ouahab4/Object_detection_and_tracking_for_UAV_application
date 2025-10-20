import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import socket
import struct
import threading
import queue
import time
import os
from ultralytics import YOLO
import math

# Constants for UI styling - Grey theme
PRIMARY_COLOR = "#333333"        # Dark grey
SECONDARY_COLOR = "#444444"      # Medium grey
ACCENT_COLOR = "#666666"         # Light grey
TEXT_COLOR = "#e0e0e0"           # Light grey text
BUTTON_HOVER = "#555555"         # Hover color
DANGER_COLOR = "#800000"         # Dark red
SUCCESS_COLOR = "#006600"        # Dark green
TRACKING_COLOR = "#FF6600"       # Orange for tracking
FONT_FAMILY = "Segoe UI"
FONT_SIZE = 12
LARGE_FONT_SIZE = 14

frame_queue = queue.Queue(maxsize=10)
box_queue = queue.Queue(maxsize=1)
tracking_queue = queue.Queue(maxsize=1)
running = True

def start_video_client(host='0.0.0.0', port=9999):
    global running
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.bind((host, port))
    client_socket.settimeout(0.5)
    print(f"Video client listening on {host}:{port}")
    last_sequence = -1
    timeout_count = 0
    max_timeouts = 5  # Number of timeouts before considering disconnected

    try:
        while running:
            try:
                packet, addr = client_socket.recvfrom(65507)
                timeout_count = 0  # Reset timeout count on successful receive
                if len(packet) < 22:
                    continue
                sequence_number, packet_type, reserved, frame_size, boxes_size, timestamp = struct.unpack(">LBBLLQ", packet[:22])
                current_time = int(time.time() * 1000)
                latency = current_time - timestamp - 3605000  # Adjust for timezone difference
                if packet_type != 1 or sequence_number <= last_sequence:
                    continue
                if len(packet) < 22 + frame_size:
                    continue
                last_sequence = sequence_number
                frame_data = packet[22:22 + frame_size]
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if frame is None:
                    continue
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not frame_queue.full():
                    frame_queue.put(frame)
                if YOLOApp.instance:
                    YOLOApp.instance.latency = latency
                    YOLOApp.instance.connection_status.config(text="Connected")
            except socket.timeout:
                timeout_count += 1
                if timeout_count >= max_timeouts and YOLOApp.instance:
                    YOLOApp.instance.connection_status.config(text="Disconnected")
                continue
            except Exception as e:
                print(f"Video packet error: {e}")
    finally:
        client_socket.close()

def draw_boxes_and_tracking(frame, boxes, tracking_info=None):
    frame_copy = frame.copy()
    height, width = frame_copy.shape[:2]
    frame_center = (width // 2, height // 2)
    
    # Draw frame center
    cv2.circle(frame_copy, frame_center, 2, (255, 0, 0), 1)
    cv2.circle(frame_copy, frame_center, 5, (255, 0, 80), 2)
    
    # Draw detection boxes
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box['xyxy']]
        label = f"{box['label']} {box['conf']:.2f}"
        
        # Different color for tracked object
        color = (255, 200, 0) if tracking_info and box.get('id') == tracking_info.get('selected_id') else (0, 255, 0)
        #thickness = 2 if tracking_info and box.get('id') == tracking_info.get('selected_id') else 1
        thickness = 1


        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Clean, small text for labels
        cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add ID if available - small and clean
        if 'id' in box and box['id'] is not None:
            cv2.putText(frame_copy, f"ID:{box['id']}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw tracking line (but no text overlay)
    if tracking_info and tracking_info.get('selected_id') is not None:
        tracked_box = tracking_info.get('tracked_box')
        if tracked_box:
            x1, y1, x2, y2 = tracked_box
            obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Draw clean line from center to tracked object
            cv2.line(frame_copy, frame_center, obj_center, (255, 102, 0), 1)
            
            # Just a small marker at the object center
            cv2.circle(frame_copy, obj_center, 3, (255, 102, 0), 1)
    
    return frame_copy

def detection_and_tracking_thread():
    global running
    while running:
        if not frame_queue.empty() and YOLOApp.instance and YOLOApp.instance.detection_enabled and YOLOApp.instance.yolo_model:
            try:
                frame = frame_queue.get_nowait()
                try:
                    # Use tracking if enabled, otherwise use detection
                    if YOLOApp.instance.tracking_enabled:
                        results = YOLOApp.instance.yolo_model.track(frame, persist=True)
                    else:
                        results = YOLOApp.instance.yolo_model(frame)
                    
                    boxes = []
                    tracking_info = None
                    
                    for result in results:
                        for box in result.boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            if conf >= 0.6:
                                label = YOLOApp.instance.yolo_model.names[cls]
                                box_data = {
                                    'xyxy': xyxy,
                                    'label': label,
                                    'conf': conf
                                }
                                
                                # Add ID if tracking is enabled and ID exists
                                if YOLOApp.instance.tracking_enabled and hasattr(box, 'id') and box.id is not None:
                                    box_data['id'] = int(box.id.item())
                                    
                                    # Check if this is the selected object for tracking
                                    if (YOLOApp.instance.selected_tracking_id is not None and 
                                        box_data['id'] == YOLOApp.instance.selected_tracking_id):
                                        
                                        # Calculate tracking information
                                        x1, y1, x2, y2 = map(int, xyxy)
                                        height, width = frame.shape[:2]
                                        frame_center = (width // 2, height // 2)
                                        obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                        
                                        dx = obj_center[0] - frame_center[0]
                                        dy = obj_center[1] - frame_center[1]
                                        angle = math.degrees(math.atan2(dy, dx))
                                        angle = angle if angle >= 0 else angle + 360
                                        distance = math.sqrt(dx**2 + dy**2)
                                        rel_x = dx / (width / 2)
                                        rel_y = dy / (height / 2)
                                        
                                        turn_direction = None
                                        if abs(rel_x) > 0.1:
                                            turn_direction = "RIGHT" if rel_x > 0 else "LEFT"
                                        
                                        tracking_info = {
                                            'selected_id': box_data['id'],
                                            'tracked_box': (x1, y1, x2, y2),
                                            'dx': dx,
                                            'dy': dy,
                                            'angle': angle,
                                            'distance': distance,
                                            'rel_x': rel_x,
                                            'rel_y': rel_y,
                                            'turn_direction': turn_direction
                                        }
                                
                                boxes.append(box_data)
                    
                    # Update queues only if there is valid data
                    if boxes and not box_queue.full():
                        while not box_queue.empty():
                            box_queue.get_nowait()
                        box_queue.put(boxes)
                    
                    if tracking_info and not tracking_queue.full():
                        while not tracking_queue.empty():
                            tracking_queue.get_nowait()
                        tracking_queue.put(tracking_info)
                    
                except Exception as e:
                    print(f"Object detection/tracking error: {e}")
            except queue.Empty:
                pass
        time.sleep(0.01)  # Keep the fast processing rate

class ModernButton(ttk.Button):
    """Custom styled button for consistent look"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.style = ttk.Style()
        self.style.configure('Modern.TButton', 
                           font=(FONT_FAMILY, FONT_SIZE),
                           padding=8,
                           background=ACCENT_COLOR,
                           foreground=TEXT_COLOR,
                           bordercolor=ACCENT_COLOR,
                           focuscolor=SECONDARY_COLOR)
        self.configure(style='Modern.TButton')

class YOLOApp(tk.Tk):
    instance = None
    
    def __init__(self):
        super().__init__()
        YOLOApp.instance = self
        self.title("Ground Control Station - Object Tracking")
        self.geometry("1280x900")
        self.configure(bg=PRIMARY_COLOR)
        
        # Initialize variables
        self.selected_model = tk.StringVar(value="YOLOv11n")
        self.detection_enabled = True
        self.tracking_enabled = False
        self.selected_tracking_id = None
        self.last_boxes = []
        self.last_tracking_info = None
        self.latency = 0
        self.fps = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.yolo_model = None
        self.last_frame_time = time.time()
        
        # Canvas scaling factors for click detection
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Setup UI components first to ensure widgets exist
        self.setup_styles()
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.setup_status_bar()  # Moved before load_yolo_model
        self.setup_video_display()
        self.setup_control_panel()
        
        # Load YOLO model after status bar (model_label) is created
        self.load_yolo_model()
        
        # Network setup
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_address = ('192.168.247.238', 9999)
        
        # Start network thread for video and detection
        threading.Thread(target=start_video_client, daemon=True).start()
        threading.Thread(target=detection_and_tracking_thread, daemon=True).start()

        # Start UI update loop - faster updates
        self.update_frame()
        self.update_stats()
        
        # Placeholder image
        self.setup_placeholder()

    def load_yolo_model(self):
        """Load the selected YOLO model from specified paths"""
        model_name = self.selected_model.get()
        model_paths = {
            "YOLOv11n": "v11n/yolov11n_50e_8bs_visdrone6/weights/best.pt",
            "YOLOv11s": "v11_visdrone/cpu/best.pt",
        }
        model_path = model_paths.get(model_name)
        if model_path:
            if not os.path.exists(model_path):
                print(f"Error: Model file {model_path} does not exist")
                if hasattr(self, 'model_label'):
                    self.model_label.config(text=f"Model: {model_name} (Not Found)")
                self.yolo_model = None
                return
            try:
                self.yolo_model = YOLO(model_path)
                print(f"Loaded YOLO model: {model_name} from {model_path}")
                if hasattr(self, 'model_label'):
                    self.model_label.config(text=f"Model: {model_name}")
            except Exception as e:
                print(f"Error loading YOLO model {model_name} from {model_path}: {e}")
                if hasattr(self, 'model_label'):
                    self.model_label.config(text=f"Model: {model_name} (Failed)")
                self.yolo_model = None
        else:
            self.yolo_model = None
            if hasattr(self, 'model_label'):
                self.model_label.config(text=f"Model: {model_name}")
            print(f"No model loaded for selection: {model_name}")
    
    def setup_styles(self):
        """Configure all custom styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame styles
        style.configure('TFrame', background=PRIMARY_COLOR)
        style.configure('Dark.TFrame', background=SECONDARY_COLOR)
        
        # Button styles
        style.configure('Modern.TButton', 
                       font=(FONT_FAMILY, FONT_SIZE),
                       padding=8,
                       background=ACCENT_COLOR,
                       foreground=TEXT_COLOR,
                       bordercolor=ACCENT_COLOR)
        style.map('Modern.TButton',
                 background=[('active', BUTTON_HOVER), ('disabled', SECONDARY_COLOR)],
                 foreground=[('disabled', TEXT_COLOR)])
        
        # Toggle button styles
        style.configure('Disable.TButton',
                       font=(FONT_FAMILY, FONT_SIZE),
                       padding=8,
                       background=DANGER_COLOR,
                       foreground=TEXT_COLOR)
        style.map('Disable.TButton',
                 background=[('active', '#a00000')])

        style.configure('Enable.TButton',
                       font=(FONT_FAMILY, FONT_SIZE),
                       padding=8,
                       background=SUCCESS_COLOR,
                       foreground=TEXT_COLOR)
        style.map('Enable.TButton',
                 background=[('active', '#008000')])
        
        # Tracking button styles
        style.configure('Tracking.TButton',
                       font=(FONT_FAMILY, FONT_SIZE),
                       padding=8,
                       background=TRACKING_COLOR,
                       foreground=TEXT_COLOR)
        style.map('Tracking.TButton',
                 background=[('active', '#ff8833')])
        
        # Combobox style
        style.configure('Modern.TCombobox',
                      font=(FONT_FAMILY, FONT_SIZE),
                      padding=5,
                      fieldbackground=SECONDARY_COLOR,
                      foreground=TEXT_COLOR,
                      background=SECONDARY_COLOR)
        style.map('Modern.TCombobox',
                 fieldbackground=[('readonly', SECONDARY_COLOR)],
                 selectbackground=[('readonly', ACCENT_COLOR)],
                 selectforeground=[('readonly', TEXT_COLOR)])
        
        # Label styles
        style.configure('Title.TLabel',
                      font=(FONT_FAMILY, 16, 'bold'),
                      background=PRIMARY_COLOR,
                      foreground=TEXT_COLOR)
        style.configure('Status.TLabel',
                      font=(FONT_FAMILY, 10),
                      background=SECONDARY_COLOR,
                      foreground=TEXT_COLOR)
        style.configure('Tracking.TLabel',
                      font=(FONT_FAMILY, 11),
                      background=SECONDARY_COLOR,
                      foreground=TRACKING_COLOR)
    
    def setup_video_display(self):
        """Set up the video display canvas"""
        self.video_frame = ttk.Frame(self.main_container, style='Dark.TFrame')
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.canvas_width = 1200
        self.canvas_height = 500
        
        self.canvas = tk.Canvas(self.video_frame, 
                               width=self.canvas_width, 
                               height=self.canvas_height, 
                               bg=SECONDARY_COLOR,
                               highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Title label
        title_label = ttk.Label(self.video_frame, 
                              text="LIVE VIDEO FEED - Click on objects to track", 
                              style='Title.TLabel')
        title_label.pack(side=tk.TOP, pady=(5, 0))
        
        # Bind click events for object selection
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_resize)
    
    def setup_control_panel(self):
        """Set up the control panel with buttons and controls"""
        control_frame = ttk.Frame(self.main_container, style='Dark.TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Gimbal control section
        gimbal_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        gimbal_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Label(gimbal_frame, 
                 text="Gimbal Controls", 
                 style='Title.TLabel').pack(side=tk.TOP)
        
        button_frame = ttk.Frame(gimbal_frame, style='Dark.TFrame')
        button_frame.pack(side=tk.TOP, pady=5)
        
        # Gimbal control buttons
        controls = [
            ("Pan Left", "pan_left"), 
            ("Pan Right", "pan_right"),
            ("Tilt Up", "tilt_up"), 
            ("Tilt Down", "tilt_down"),
            ("Center", "center_gimbal")
        ]
        
        for i, (text, cmd) in enumerate(controls):
            btn = ModernButton(button_frame, 
                             text=text,
                             command=lambda c=cmd: self.send_command(c))
            btn.grid(row=0, column=i, padx=5)
        
        # Detection controls section
        detection_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        detection_frame.pack(side=tk.LEFT, padx=20, pady=5)
        
        ttk.Label(detection_frame, 
                 text="Detection Settings", 
                 style='Title.TLabel').pack(side=tk.TOP)
        
        detection_buttons = ttk.Frame(detection_frame, style='Dark.TFrame')
        detection_buttons.pack(side=tk.TOP, pady=5)
        
        # Toggle detection button
        self.toggle_button = ttk.Button(detection_buttons, 
                                       text="Disable Detection", 
                                       command=self.toggle_detection,
                                       style='Disable.TButton')
        self.toggle_button.pack(side=tk.LEFT, padx=5)
        
        # Toggle tracking button
        self.tracking_button = ttk.Button(detection_buttons, 
                                         text="Enable Tracking", 
                                         command=self.toggle_tracking,
                                         style='Enable.TButton')
        self.tracking_button.pack(side=tk.LEFT, padx=5)
        
        # Clear tracking button
        self.clear_tracking_button = ttk.Button(detection_buttons, 
                                               text="Clear Track", 
                                               command=self.clear_tracking,
                                               style='Modern.TButton')
        self.clear_tracking_button.pack(side=tk.LEFT, padx=5)
        
        # Model selector
        self.model_selector = ttk.Combobox(detection_frame, 
                                         textvariable=self.selected_model, 
                                         values=["YOLOv11n", "YOLOv11s"], 
                                         state="readonly",
                                         style='Modern.TCombobox')
        self.model_selector.pack(side=tk.TOP, pady=5)
        self.model_selector.bind("<<ComboboxSelected>>", self.model_changed)
        
        # Tracking info display section
        self.tracking_info_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        self.tracking_info_frame.pack(side=tk.LEFT, padx=20, pady=5)
        
        ttk.Label(self.tracking_info_frame, 
                 text="Tracking Info", 
                 style='Title.TLabel').pack(side=tk.TOP)
        
        # Main tracking info
        self.tracking_info_label = ttk.Label(self.tracking_info_frame, 
                                           text="No tracking active", 
                                           style='Tracking.TLabel')
        self.tracking_info_label.pack(side=tk.TOP, pady=2)
        
        # Direction info
        self.direction_label = ttk.Label(self.tracking_info_frame, 
                                       text="", 
                                       style='Status.TLabel')
        self.direction_label.pack(side=tk.TOP, pady=2)
        
        # Position info
        self.position_label = ttk.Label(self.tracking_info_frame, 
                                      text="", 
                                      style='Status.TLabel')
        self.position_label.pack(side=tk.TOP, pady=2)
    
    def setup_status_bar(self):
        """Set up the status bar at the bottom"""
        status_frame = ttk.Frame(self.main_container, style='Dark.TFrame')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status indicators
        self.connection_status = ttk.Label(status_frame, 
                                         text="Connected", 
                                         style='Status.TLabel')
        self.connection_status.pack(side=tk.LEFT, padx=10)
        
        self.latency_label = ttk.Label(status_frame, 
                                     text="Latency: 0ms", 
                                     style='Status.TLabel')
        self.latency_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = ttk.Label(status_frame, 
                                 text="FPS: 0", 
                                 style='Status.TLabel')
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.model_label = ttk.Label(status_frame, 
                                   text="Model: YOLOv11n", 
                                   style='Status.TLabel')
        self.model_label.pack(side=tk.LEFT, padx=10)
        
        self.tracking_status_label = ttk.Label(status_frame, 
                                             text="Tracking: OFF", 
                                             style='Status.TLabel')
        self.tracking_status_label.pack(side=tk.LEFT, padx=10)
        
        # Close button
        close_btn = ModernButton(status_frame, 
                               text="Exit", 
                               command=self.destroy)
        close_btn.pack(side=tk.RIGHT, padx=10)
    
    def setup_placeholder(self):
        """Create placeholder image for initial display"""
        self.placeholder_image = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        cv2.putText(self.placeholder_image, "Waiting for video stream...", 
                   (self.canvas_width//4, self.canvas_height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.putText(self.placeholder_image, "Enable tracking and click on objects to track them", 
         #          (self.canvas_width//4 - 50, self.canvas_height//2 + 40),
          #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self.placeholder_image = cv2.cvtColor(self.placeholder_image, cv2.COLOR_BGR2RGB)
        self.display_placeholder()
    
    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            self.toggle_button.config(text="Disable Detection", style='Disable.TButton')
        else:
            self.toggle_button.config(text="Enable Detection", style='Enable.TButton')
            self.last_boxes = []  # Clear boxes when detection is disabled
            self.last_tracking_info = None
        print(f"Detection {'enabled' if self.detection_enabled else 'disabled'}")
    
    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled
        if self.tracking_enabled:
            self.tracking_button.config(text="Disable Tracking", style='Disable.TButton')
            self.tracking_status_label.config(text="Tracking: ON (Click object)")
        else:
            self.tracking_button.config(text="Enable Tracking", style='Enable.TButton')
            self.tracking_status_label.config(text="Tracking: OFF")
            self.selected_tracking_id = None
            self.last_tracking_info = None
            # Clear tracking info display
            self.tracking_info_label.config(text="No tracking active")
            self.direction_label.config(text="")
            self.position_label.config(text="")
        print(f"Tracking {'enabled' if self.tracking_enabled else 'disabled'}")
    
    def clear_tracking(self):
        self.selected_tracking_id = None
        self.last_tracking_info = None
        # Clear tracking info display
        self.tracking_info_label.config(text="No tracking active")
        self.direction_label.config(text="")
        self.position_label.config(text="")
        print("Tracking cleared")
    
    def on_canvas_click(self, event):
        """Handle canvas click events for object selection"""
        if not self.tracking_enabled or not self.last_boxes:
            return
        
        # Convert canvas coordinates to frame coordinates
        canvas_x = event.x
        canvas_y = event.y
        
        # Account for scaling and offset
        frame_x = int((canvas_x - self.offset_x) / self.scale_x)
        frame_y = int((canvas_y - self.offset_y) / self.scale_y)
        
        # Check which box was clicked
        self.selected_tracking_id = None
        for box in self.last_boxes:
            if 'id' in box and box['id'] is not None:
                x1, y1, x2, y2 = map(int, box['xyxy'])
                if x1 <= frame_x <= x2 and y1 <= frame_y <= y2:
                    self.selected_tracking_id = box['id']
                    print(f"Selected object ID {self.selected_tracking_id} for tracking")
                    break
        
        if self.selected_tracking_id is None:
            print(f"No trackable object found at click position ({frame_x}, {frame_y})")
    
    def model_changed(self, event=None):
        model = self.selected_model.get()
        self.model_label.config(text=f"Model: {model}")
        print(f"Model changed to: {model}")
        self.load_yolo_model()
    
    def send_command(self, command):
        try:
            self.client_socket.sendto(command.encode(), self.server_address)
            print(f"Sent command: {command}")
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def on_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.display_placeholder()
    
    def resize_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height
        canvas_aspect = self.canvas_width / self.canvas_height
        
        if aspect_ratio > canvas_aspect:
            new_width = self.canvas_width
            new_height = int(self.canvas_width / aspect_ratio)
        else:
            new_height = self.canvas_height
            new_width = int(self.canvas_height * aspect_ratio)
        
        # Calculate scaling factors and offsets for click detection
        self.scale_x = new_width / frame_width
        self.scale_y = new_height / frame_height
        self.offset_x = (self.canvas_width - new_width) // 2
        self.offset_y = (self.canvas_height - new_height) // 2
            
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        canvas[self.offset_y:self.offset_y+new_height, self.offset_x:self.offset_x+new_width] = resized_frame
        return canvas
    
    def display_placeholder(self):
        display_frame = self.resize_frame(self.placeholder_image)
        image = Image.fromarray(display_frame)
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, image=photo, anchor="center")
        self.canvas.image = photo
    
    def update_frame(self):
        current_time = time.time()
        try:
            if not frame_queue.empty():
                # Skip older frames to keep display real-time
                while frame_queue.qsize() > 1:
                    frame_queue.get_nowait()
                frame = frame_queue.get_nowait()
                self.frame_count += 1
                self.last_frame_time = current_time
                
                # Update boxes and tracking info if available
                if not box_queue.empty():
                    self.last_boxes = box_queue.get_nowait()
                
                if not tracking_queue.empty():
                    self.last_tracking_info = tracking_queue.get_nowait()
                
                # Draw boxes and tracking if detection is enabled
                if self.detection_enabled and self.last_boxes:
                    display_frame = draw_boxes_and_tracking(frame, self.last_boxes, self.last_tracking_info)
                else:
                    display_frame = frame
                    # Still draw frame center if tracking is enabled
                    if self.tracking_enabled:
                        height, width = display_frame.shape[:2]
                        frame_center = (width // 2, height // 2)
                        cv2.circle(display_frame, frame_center, 5, (255, 0, 0), -1)
                        cv2.circle(display_frame, frame_center, 20, (255, 0, 0), 2)
                
                # Resize and display frame
                display_frame = self.resize_frame(display_frame)
                image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image=image)
                self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, image=photo, anchor="center")
                self.canvas.image = photo
                
                # Update tracking status
                if self.selected_tracking_id is not None:
                    self.tracking_status_label.config(text=f"Tracking: ID {self.selected_tracking_id}")
                    # Update tracking info display
                    if self.last_tracking_info and self.last_tracking_info.get('selected_id') == self.selected_tracking_id:
                        info = self.last_tracking_info
                        self.tracking_info_label.config(text=f"Tracking ID {info['selected_id']} (Dist: {info['distance']:.1f}px)")
                        self.direction_label.config(text=f"Angle: {info['angle']:.1f}° {'RIGHT' if info['rel_x'] > 0.1 else 'LEFT' if info['rel_x'] < -0.1 else 'ON TARGET'}")
                        self.position_label.config(text=f"Rel Pos: ({info['rel_x']:.2f}, {info['rel_y']:.2f})")
                    else:
                        self.tracking_info_label.config(text="No tracking active")
                        self.direction_label.config(text="")
                        self.position_label.config(text="")
                elif self.tracking_enabled:
                    self.tracking_status_label.config(text="Tracking: ON (Click object)")
                    self.tracking_info_label.config(text="No object selected")
                    self.direction_label.config(text="")
                    self.position_label.config(text="")
                else:
                    self.tracking_status_label.config(text="Tracking: OFF")
                    self.tracking_info_label.config(text="No tracking active")
                    self.direction_label.config(text="")
                    self.position_label.config(text="")
                
            elif current_time - self.last_frame_time > 2.0:  # Show placeholder if no frames for 2 seconds
                self.display_placeholder()
                self.last_boxes = []  # Clear boxes when showing placeholder
                self.last_tracking_info = None
                self.tracking_info_label.config(text="No tracking active")
                self.direction_label.config(text="")
                self.position_label.config(text="")
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Frame update error: {e}")
        
        # Update latency display
        self.latency_label.config(text=f"Latency: {self.latency}ms")
        self.after(10, self.update_frame)  # Increased update frequency for smoother rendering
    
    def update_stats(self):
        """Update performance statistics in the status bar"""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.frame_count = 0
            self.last_fps_update = current_time
        
        # Print tracking info to console if tracking an object
        if (self.last_tracking_info and 
            self.selected_tracking_id is not None and 
            self.last_tracking_info.get('selected_id') == self.selected_tracking_id):
            
            info = self.last_tracking_info
            print(f"\n--- Tracking Info for Object ID {info['selected_id']} ---")
            print(f"Distance Vector (dx, dy): ({info['dx']:.1f}, {info['dy']:.1f}) pixels")
            print(f"Euclidean Distance: {info['distance']:.1f} pixels")
            print(f"Angle: {info['angle']:.1f} degrees (0° is right, 90° is down)")
            print(f"Relative Position: Rel X: {info['rel_x']:.2f}, Rel Y: {info['rel_y']:.2f}")
            if info['turn_direction']:
                print(f"Drone Action: TURN {info['turn_direction']}")
            else:
                print("Drone Action: HOLD/ON TARGET")
            print("---------------------------------------------")
        
        self.after(1000, self.update_stats)
    
    def destroy(self):
        global running
        running = False
        self.client_socket.close()
        super().destroy()

if __name__ == "__main__":
    try:
        app = YOLOApp()
        app.mainloop()
    except KeyboardInterrupt:
        running = False