# ========================= LICENSE INFORMATION =========================

# This code is developed by Sergey Korneyev from AVStream (https://avstream.ru).
# It is distributed under an open-source license, allowing for free use, modification, 
# and distribution, provided that this notice and attribution to the original author 
# are preserved.

# This code utilizes FFmpeg, an open-source multimedia framework licensed under the LGPL or GPL.
# FFmpeg is a trademark of Fabrice Bellard and is maintained by the FFmpeg team.
# For more details on FFmpeg's licensing terms, please refer to https://ffmpeg.org/legal.html.

# This code also uses YOLOv8 by Ultralytics, an open-source object detection model.
# YOLOv8 and Ultralytics' models are distributed under the GPL-3.0 License.
# For more details on YOLOv8's licensing, please refer to https://github.com/ultralytics/ultralytics.

# By using this code, you acknowledge that you comply with the terms and conditions of both
# FFmpeg's and YOLOv8's licenses, as well as the open-source license provided by Sergey Korneyev.
# Redistribution of this script with modifications should maintain these notices to respect the 
# original work and licenses of the authors.

# Author: Sergey Korneyev, AVStream
# Website: https://avstream.ru
# Telegram: https://t.me/avstream


import socket
import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

# ========================= CONFIGURABLE PARAMETERS =========================

# Camera parameters
CAMERA_IP = '192.168.1.45'
CAMERA_PORT = 1259
RTSP_URL = 'rtsp://192.168.1.45:554/live/av1'  # RTSP URL for the video stream

# VISCA control parameters
VISCA_STOP = b'\x81\x01\x06\x01\x00\x00\x03\x03\xff'
MIN_SPEED = 3  # Minimum speed for smooth movement
MAX_SPEED = 12  # Maximum speed for fast movement
THRESHOLD = 20  # Threshold for stabilization

# Video stream parameters
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
VIDEO_FPS = 15
VIDEO_BITRATE = '300k'  # Video bitrate, e.g., '300k' for 300 kbps

# Focus point parameters for tracking
FOCUS_POINT_Y_FACTOR = 0.2  # Vertical position of the focus point (20% from the top of the screen)

# Yolo8 parameters
MODEL_PATH = "yolov8n.pt"  # Path to the YOLO8 model

# Toggle visualization (set to True to display points and rectangle)
SHOW_VISUALS = True

# ===========================================================================

# VISCA command function for different speeds and directions
def get_visca_command(pan_speed, tilt_speed, pan_direction, tilt_direction):
    pan_speed_byte = bytes([pan_speed])
    tilt_speed_byte = bytes([tilt_speed])
    pan_direction_byte = bytes([int(pan_direction, 16)])
    tilt_direction_byte = bytes([int(tilt_direction, 16)])
    command = b'\x81\x01\x06\x01' + pan_speed_byte + tilt_speed_byte + pan_direction_byte + tilt_direction_byte + b'\xff'
    print(f"Sending command: {command}")
    return command

# YOLO8 model initialization
model = YOLO(MODEL_PATH)

def send_command(s, command):
    try:
        s.send(command)
        print(f"Command sent: {command}")
    except Exception as e:
        print(f"Error sending command: {e}")

def control_camera_to_center_person(s, target_position, person_center):
    x_diff = person_center[0] - target_position[0]
    y_diff = person_center[1] - target_position[1]

    # Calculate speed based on offset
    pan_speed = min(MAX_SPEED, max(MIN_SPEED, int(abs(x_diff) / 10)))
    tilt_speed = min(MAX_SPEED, max(MIN_SPEED, int(abs(y_diff) / 10)))

    # Determine directions considering the threshold
    pan_direction = '02' if x_diff > THRESHOLD else '01' if x_diff < -THRESHOLD else '03'
    tilt_direction = '02' if y_diff > THRESHOLD else '01' if y_diff < -THRESHOLD else '03'

    # Stop camera if the object is within the threshold
    if pan_direction == '03' and tilt_direction == '03':
        send_command(s, VISCA_STOP)
    else:
        command = get_visca_command(pan_speed, tilt_speed, pan_direction, tilt_direction)
        send_command(s, command)

def display_and_track_rtsp_stream():
    ffmpeg_command = [
        'ffmpeg',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-rtsp_transport', 'tcp',
        '-i', RTSP_URL,
        '-vf', f'scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}',
        '-r', str(VIDEO_FPS),
        '-b:v', VIDEO_BITRATE,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-an', '-',
    ]

    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=10 ** 8)
    target_position = (VIDEO_WIDTH // 2, int(VIDEO_HEIGHT * FOCUS_POINT_Y_FACTOR))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("Connecting to camera...")
        s.connect((CAMERA_IP, CAMERA_PORT))
        print("Camera connected.")

        cv2.namedWindow("RTSP Video Stream")

        while True:
            frame_bytes = process.stdout.read(VIDEO_WIDTH * VIDEO_HEIGHT * 3)
            if not frame_bytes:
                print("Video stream interrupted.")
                break

            frame = np.frombuffer(frame_bytes, np.uint8).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3))
            frame_copy = frame.copy()

            # Obtain detections using Yolo8
            results = model(frame_copy)
            persons = []

            # Find all people in the frame and determine head region
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls == 0:  # 0 - "Person" class in Yolo8
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Determine head coordinates (upper third of the body)
                    head_y = y1 + (y2 - y1) // 3
                    head_center = ((x1 + x2) // 2, int(y1 + 0.3 * (head_y - y1)))
                    persons.append(head_center)

                    # Visualization block (can be commented out)
                    if SHOW_VISUALS:
                        # Draw green rectangle around the person
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw blue point on the head center
                        cv2.circle(frame_copy, head_center, 5, (255, 0, 0), -1)
                        # Draw red target point on the screen
                        cv2.circle(frame_copy, target_position, 5, (0, 0, 255), -1)

            # If there is a person in the frame, find the one closest to the target point
            if persons:
                closest_person = min(persons,
                                     key=lambda p: abs(p[0] - target_position[0]) + abs(p[1] - target_position[1]))
                control_camera_to_center_person(s, target_position, closest_person)
            else:
                # If there is no person in the frame, stop the camera
                send_command(s, VISCA_STOP)

            cv2.imshow("RTSP Video Stream", frame_copy)

            # Exit on 'ESC' key press
            if cv2.waitKey(1) & 0xFF == 27:
                send_command(s, VISCA_STOP)
                print("Exiting tracking.")
                break

    process.stdout.close()
    process.wait()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_and_track_rtsp_stream()

