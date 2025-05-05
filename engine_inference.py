import cv2
import time
import os
from ultralytics import YOLO

# Task definition
task = "detect"

# Model path
model_path = "/home/mcw/Karthick/shopable-ads/yolov8_int8.engine"

# Load TensorRT engine model
model = YOLO(model_path)

# Load video
video_path = "/home/mcw/Karthick/shopable-ads/sunglasses1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
output_path = "processed_output.mp4"

# VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

frame_count = 0
total_detections = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, imgsz=1280)[0]

    # Draw results
    annotated_frame = results.plot()

    # Write to output video
    out.write(annotated_frame)

    # Count detections
    num_detections = len(results.boxes)
    total_detections += num_detections
    frame_count += 1

end_time = time.time()
fps = frame_count / (end_time - start_time)

# Release resources
cap.release()
out.release()

# Get model file size in megabytes
model_size_bytes = os.path.getsize(model_path)
model_size_mb = model_size_bytes / (1024 * 1024)

# Final report
print(f"\n📊 Final Report")
print(f"Task: {task}")
print(f"Model: {os.path.basename(model_path)}")
print(f"Model Size: {model_size_mb:.1f} MB")
print(f"Frames Processed: {frame_count}")
print(f"Total Detections: {total_detections}")
print(f"FPS: {fps:.2f}")
print(f"Saved video to: {output_path}")
