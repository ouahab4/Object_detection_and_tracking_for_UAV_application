# Aerial Surveillance Gimbal System for UAV Defence & Security applications

## Overview

A hardware-software system for real-time object detection and tracking using a Raspberry Pi-controlled gimbal. Detects and tracks targets (e.g., vehicles, pedestrians, tanks, armed vehicles, soldiers) in aerial imagery, with integration to a ground control UI.

<div align="center">

 | HyperDet GPU Demo |
 | :---: |
| ![HyperDet GPU Demo](media/model_demo.gif) |
 | *Best model running on GPU with real-time inference* |

</div>

This project was developed as part of a security drone internship at Electronic Systems Research and Development Unit (Feb-Jun 2025), combining AI vision with mechanical control.

## Features
- **Object Detection**: Uses YOLOv8 small or HyperDet CNN models, pre-trained on ImageNet and fine-tuned on Roboflow dataset for custom classes.
- **Tracking**: Implements CSRT and DeepSort for robust, occlusion-resistant tracking.
- **Hardware Integration**: Raspberry Pi 5 with Pi Camera 2 on Storm32 gimbal; sends motor signals for physical tracking.
- **Streaming & UI**: UDP socket video stream to PC; GroundControl UI with sensor panels, error handling, and interactive object selection.
- **Performance**: Optimized for edge devices with Movidius Neural Compute Stick.

## System Architecture
![System Diagram](media/2025-09-25_23-33.png)
*System overview showing the complete pipeline*

- **Raspberry Pi Side**: Captures frames, controls gimbal via signals, streams video/data via UDP.
- **PC Side**: Receives stream, runs inference, displays UI, allows operator to select/track objects and what models to use.
  
  *P.S: the inference runing unit could be configures to be either the raspberry + NCS 2, PC's CPU or GPU, in this demonstration the CPU was used to obtain the average results*


## 📁 Repository Structure
```
Object_detection_and_tracking_for_UAV_application/
│
├── 📄 README.md           # Main project documentation
├── 📁 docs/               # Additional documentation
│   └── project_report.pdf    # academic report
├── 📁 src/                # Source code
│   ├── 📁 client_server/          # clent server configuration
│       ├── server.py                # Raspberry side: It streams video frames and detection results via UDP, separate threads for frame, inference, and communication.
│       ├── client_version2.py       # PC side: ground control station with a Tkinter GUI that receives UDP video streams and performs YOLO inference. features dual         │       │                            tracking modes (YOLO's built-in, and OpenCV's CSRT), interactive object selection via mouse clicks, and gimbal control commands.
│       └── client_version1.py       # same as above, Simplified unified pipeline using only YOLO's persist tracking for lower CPU/GPU overhead
│   ├── 📁 training/               # Training scripts for many architectures (Yolovx, Faster R-CNN, Hybreddet)
│       ├── faster_rcnn_train.ipynb  
│   ├── 📁 data_format_converter/   
│       ├── visdrone_process.py      #loads and processes VisDrone dataset sequences + handles image and annotation files 
│       └── yolov8_formatter.py      # processes XML annotations to yolo format
│   ├── 📁 testing/                # this has testing files for different tracking techniques with different models  
│       ├── tracking_v1.py           #YOLO for real-time tracking, object selection via mouse click, displays tracking data (distance, angle, position) 
│       ├── tracking_v2.py           #with CSRT tracker for real-time tracking
│       ├── tracking_v3.py           #detection on a video, processes every 5th frame, resizes frames to 640x380, and saves the output with bounding boxes to video file
│       ├── tracking_v4.py           #real-time detection on webcam feed, resolution to 640x480 at 30 FPS, processes every x (here x=3) frame for efficiency
│       └── detection_test.py        # runs a model on a test image and visualizes detections
├── 📁 models/             # Trained model weights
│   ├── yolov11s_visdrone.pt       # yolov11 small trained on visdrone, 50 epochs, batch size = 8  
│   └── hyperdet_cnn_best.pt
├── 📁 media/              # Demo + screenshots + components materials...
│   └── ...
└── 📁 datasets/           # Dataset info (links only)
    └── dataset_sources.md
```

### Hardware Setup
<div align="center">

| | |
| :---: | :---: |
| **Main Controller**<br>Raspberry Pi 4 or 5<br>![Raspberry Pi](media/raspberry_pi.png) | **AI Accelerator**<br>Intel Movidius NCS 2<br>![Intel Movidius](media/ncs.png) |
| **Gimbal System**<br>3-axis gimbal<br>![3-axis gimbal](media/gimbal1.png) | **Gimbal Controller**<br>Storm32 BGCC<br>![Storm32 BGCC](media/gimbal2.png) |
| **Camera**<br>Pi Camera v2<br>![Pi Camera](media/pi_camera.png) | **Communication**<br>UDP socket streaming |

</div>

## Results & Metrics
- Accuracy: 90%+ on custom dataset (mAP@0.5 ~0.85 for YOLOv8s).
- FPS: ~5 FPS inference on Raspberry Pi + Movidus NCS 2, 5-12 FPS on CPU (intel I3 11th gen), and 40 FPS on Nvidia RTX 4060 8G
- Screenshots:
  - UI Dashboard: ![UI Screenshot](media/ui-screenshot.png)
  - Detection output: ![Detection Example](images/detection-screenshot.png)
  - Gimbal position adjustments data: to be sent from the processing unit (PC's CPU/GPU, or Raspberry Pi + NCS) to the gimbal controller ![Detection Example](images/detection-screenshot.png)



## Installation
1. Clone the repo: `git clone https://github.com/ouahab4/Object_detection_and_tracking_for_UAV_application.git`
2. Install dependencies: `pip install -r requirements.txt` (Includes ultralytics, opencv-python, deepsort-realtime, etc.)
3. Download models: Place YOLOv8s.pt and HyperDet weights in `/models/` (links in models/README.md).
4. Hardware setup: Connect Pi Camera to Raspberry Pi 5, gimbal to Storm32 controller, Movidius stick for acceleration.

## Usage
- **Run on Raspberry Pi (Detection & Gimbal Control)**: `python gimbal_tracker.py --model yolov8s.pt --tracker csrt`
- **Run PC UI (Ground Control)**: `python ground_control_ui.py --ip 192.168.247.196 --port 5000`
- Select object in UI to initiate tracking; gimbal adjusts in real-time.


