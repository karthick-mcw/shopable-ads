from flask import Flask, render_template_string, send_file
import cv2
import threading
import time
import numpy as np
import os
import logging
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration: update these paths to your video and engine files ---
VIDEO_FILE = '/home/mcw/Karthick/shopable-ads/suitcase.mp4'
ENGINE_PATH = '/home/mcw/Karthick/shopable-ads/yolov8_int8.engine'

# Class definitions and corresponding product links
definitions = {
    'classes': ['headphone', 'suitcase', 'sunglasses', 'watch'],
    'links': {
        'headphone': 'https://www.google.com',
        'suitcase':  'https://www.nike.com',
        'sunglasses':'https://www.puma.com',
        'watch':      'https://www.apple.com',
    }
}

# Global variables to store video metadata and detections
video_metadata = {
    'fps': 30.0,  # Default fallback
    'total_frames': 0,
    'duration': 0,
    'width': 1280,
    'height': 480
}
all_detections = {}
processed = False

# Initialize YOLO model
try:
    # Remove any specific size expectations by using the default configuration
    model = YOLO(ENGINE_PATH)
    logger.info(f"Successfully loaded YOLO model from {ENGINE_PATH}")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

# Initialize tracker
tracker = DeepSort(max_age=5)
track_classes = {}  # Class ID tracker

# Process all frames at startup
def process_video():
    global all_detections, video_metadata, processed
    
    # Get video metadata first
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {VIDEO_FILE}")
        processed = True
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0 or fps > 120:
        fps = 30.0  # Default fallback
        
    duration = total_frames / fps if fps > 0 else 0
    
    video_metadata = {
        'fps': fps,
        'total_frames': total_frames,
        'duration': duration,
        'width': width,
        'height': height
    }
    
    logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
    
    # Process frames
    for frame_idx in range(min(total_frames, 5000)):  # Limit to max 5000 frames
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        try:
            if model is None:
                # Skip if model failed to load
                frame_objects = []
            else:
                # NEW: Let YOLO handle the resizing internally
                # The imgsz parameter will be passed to the model to specify input dimensions
                results = model(frame, conf=0.4, imgsz=1280)[0]
            
            # Format detections for tracker
            detections = []
            for box in results.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                
                # Skip tiny detections
                if w < 20 or h < 20:
                    continue
                    
                conf = float(box.conf)
                class_id = int(box.cls)
                
                # Add to detection list for tracker
                detections.append(([int(x1), int(y1), int(w), int(h)], conf, class_id))
            
            # Update tracker
            frame_objects = []
            if model is None:
                # Skip detection if model failed to load
                pass
            elif detections:
                tracks = tracker.update_tracks(detections, frame=frame)
                
                # Process tracked objects
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    track_id = track.track_id
                    
                    # Get or update class ID
                    if hasattr(track, 'det_class'):
                        class_id = track.det_class
                        track_classes[track_id] = class_id
                    elif track_id in track_classes:
                        class_id = track_classes[track_id]
                    else:
                        continue
                        
                    if class_id >= len(definitions['classes']):
                        continue
                        
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    class_name = definitions['classes'][class_id]
                    
                    # Store detection data
                    frame_objects.append({
                        'id': track_id,
                        'box': [x1, y1, x2 - x1, y2 - y1],
                        'class': class_name,
                        'link': definitions['links'][class_name],
                        'confidence': float(track.get_det_conf()) if hasattr(track, 'get_det_conf') and track.get_det_conf() is not None else 0.8
                    })
            
            # Store frame detections
            all_detections[frame_idx] = frame_objects
            
            # Log progress
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
                
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
            all_detections[frame_idx] = []
    
    logger.info(f"Finished processing video. Processed {len(all_detections)} frames")
    cap.release()
    processed = True

def create_app():
    # Start processing thread
    processing_thread = threading.Thread(target=process_video, daemon=True)
    processing_thread.start()
    
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        # Create updated HTML with inline data
        available_classes = definitions['classes']
        
        # Generate a compact JSON string of the first 100 frames for immediate display
        initial_detections = {}
        for i in range(min(100, len(all_detections))):
            if i in all_detections:
                initial_detections[i] = all_detections[i]
        
        return render_template_string(
            HTML_PAGE.replace(
                '/* INJECTED_DATA */', 
                f'''
                const AVAILABLE_CLASSES = {available_classes};
                const VIDEO_METADATA = {video_metadata};
                const INITIAL_DETECTIONS = {initial_detections};
                const PROCESSING_DONE = {str(processed).lower()};
                '''
            )
        )

    @app.route('/video_file')
    def video_file():
        return send_file(VIDEO_FILE, mimetype='video/mp4')
        
    @app.route('/get_frame_data/<int:frame_idx>')
    def get_frame_data(frame_idx):
        # Simple function to get a specific frame's data if it wasn't in the initial batch
        if frame_idx in all_detections:
            return {'detections': all_detections[frame_idx]}
        else:
            return {'detections': []}

    return app

# HTML Page with inline data support
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shoppable Ads</title>
  <style>
    body { 
      font-family: Arial, sans-serif; 
      margin: 20px; 
      background-color: #f5f5f5;
    }
    h1 {
      color: #333;
      text-align: center;
    }
    .controls { 
      margin-bottom: 15px; 
      text-align: center;
    }
    .controls button {
      background-color: #4CAF50;
      border: none;
      color: white;
      padding: 10px 20px;
      text-align: center;
      text-decoration: none;
      display: inline-block;
      font-size: 16px;
      margin: 4px 2px;
      cursor: pointer;
      border-radius: 4px;
    }
    .toggle-switch {
      display: inline-block;
      margin-left: 20px;
    }
    .toggle-switch label {
      display: inline-block;
      vertical-align: middle;
      margin-right: 10px;
    }
    .toggle-switch input {
      height: 0;
      width: 0;
      visibility: hidden;
    }
    .toggle-switch .switch {
      cursor: pointer;
      width: 60px;
      height: 30px;
      display: inline-block;
      border-radius: 100px;
      position: relative;
      background: #ddd;
      vertical-align: middle;
    }
    .toggle-switch .switch:after {
      content: '';
      position: absolute;
      top: 3px;
      left: 3px;
      width: 24px;
      height: 24px;
      background: #fff;
      border-radius: 90px;
      transition: 0.3s;
    }
    .toggle-switch input:checked + .switch {
      background: #4CAF50;
    }
    .toggle-switch input:checked + .switch:after {
      left: calc(100% - 3px);
      transform: translateX(-100%);
    }
    .video-container { 
      position: relative; 
      display: block;
      margin: 0 auto; 
      max-width: 1000px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    video { 
      display: block; 
      width: 100%; 
    }
    canvas { 
      position: absolute; 
      top: 0; 
      left: 0; 
      cursor: pointer;
    }
    .tooltip {
      position: absolute;
      padding: 5px;
      background: rgba(0,0,0,0.7);
      color: white;
      border-radius: 3px;
      pointer-events: none;
      z-index: 100;
      font-size: 12px;
    }
    .status {
      text-align: center;
      margin-top: 10px;
      color: #666;
    }
    .frame-display {
      text-align: center;
      margin-top: 5px;
      font-size: 14px;
      color: #666;
    }
    .filter-container {
      max-width: 1000px;
      margin: 15px auto;
      text-align: center;
      padding: 10px;
      background-color: #f9f9f9;
      border: 1px solid #ddd;
      border-radius: 4px;
    }
    .filter-container input {
      padding: 8px;
      width: 70%;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 14px;
    }
    .available-classes {
      margin-top: 8px;
      font-size: 12px;
      color: #666;
    }
    .video-info {
      max-width: 1000px;
      margin: 5px auto;
      text-align: center;
      font-size: 12px;
      color: #666;
    }
  </style>
</head>
<body>
  <h1>SHOPPABLE ADS</h1>
  
  <div class="filter-container">
    <div>
      <label for="class-filter">Filter Classes (comma separated):</label>
      <input type="text" id="class-filter" placeholder="e.g. sunglasses, watch">
      <button id="apply-filter">Apply Filter</button>
    </div>
    <div class="available-classes">
      Available classes: <span id="available-classes">loading...</span>
    </div>
  </div>
  
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <div class="toggle-switch">
      <label for="show-boxes">Show Boxes:</label>
      <input type="checkbox" id="show-boxes" checked />
      <label for="show-boxes" class="switch"></label>
    </div>
  </div>
  <div class="video-container">
    <video id="player" controls>
      <source src="/video_file" type="video/mp4" />
    </video>
    <canvas id="overlay"></canvas>
    <div id="tooltip" class="tooltip" style="display:none"></div>
  </div>
  <div class="status" id="status">Ready</div>
  <div class="frame-display" id="frame-display">Frame: 0</div>
  <div class="video-info" id="video-info">Loading video information...</div>
  
  <script>
    // Injected data from server - will be populated when page loads
    /* INJECTED_DATA */
    
    // DOM elements
    const player = document.getElementById('player');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');
    const showBoxesToggle = document.getElementById('show-boxes');
    const statusDisplay = document.getElementById('status');
    const frameDisplay = document.getElementById('frame-display');
    const videoInfoDisplay = document.getElementById('video-info');
    const classFilterInput = document.getElementById('class-filter');
    const availableClassesSpan = document.getElementById('available-classes');
    
    // State variables
    let detections = [];
    let showBoxes = true;
    let colorCache = {};
    let currentFrame = 0;
    let requestedFrame = 0;
    let filteredClasses = [];
    let cachedDetections = INITIAL_DETECTIONS || {};
    let videoFPS = VIDEO_METADATA.fps || 30;
    let totalFrames = VIDEO_METADATA.total_frames || 0;
    let animationId = null;
    
    // Update video info display
    videoInfoDisplay.textContent = `Video: ${totalFrames} frames at ${videoFPS.toFixed(2)} FPS`;
    
    // Update available classes
    if (AVAILABLE_CLASSES) {
      availableClassesSpan.textContent = AVAILABLE_CLASSES.join(', ');
    }
    
    // Generate consistent colors for tracking IDs
    function getColor(id) { 
      if (!colorCache[id]) {
        colorCache[id] = `hsl(${(id * 31) % 360},70%,60%)`; 
      }
      return colorCache[id]; 
    }
    
    // Resize canvas to match video dimensions
    function resizeCanvas() { 
      canvas.width = player.clientWidth; 
      canvas.height = player.clientHeight; 
      drawDetections();
    }
    
    // Get current frame number from video time
    function getCurrentFrame() {
      if (!player.duration) return 0;
      return Math.round(player.currentTime * videoFPS);
    }
    
    // Get detection data for current frame
    async function getDetectionsForFrame(frameIdx) {
      // If we already have this frame cached, use it
      if (frameIdx in cachedDetections) {
        return cachedDetections[frameIdx];
      }
      
      // Otherwise fetch from server
      try {
        const response = await fetch(`/get_frame_data/${frameIdx}`);
        const data = await response.json();
        
        // Cache the result
        cachedDetections[frameIdx] = data.detections;
        return data.detections;
      } catch (error) {
        console.error("Error fetching frame data:", error);
        return [];
      }
    }
    
    // Update detections for current frame
    async function updateDetections() {
      const frameIdx = getCurrentFrame();
      
      // Update current frame display
      currentFrame = frameIdx;
      frameDisplay.textContent = `Frame: ${currentFrame} / ${totalFrames}`;
      
      // Get detections for this frame
      detections = await getDetectionsForFrame(frameIdx);
      
      // Apply filters if needed
      if (filteredClasses.length > 0) {
        detections = detections.filter(d => filteredClasses.includes(d.class));
      }
      
      // Update status
      statusDisplay.textContent = detections.length > 0 ? 
        `${detections.length} objects detected` : 
        "No detections in current frame";
      
      // Draw detections
      drawDetections();
    }
    
    // Draw bounding boxes and labels
    function drawDetections() {
      if (!showBoxes) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }
      
      const sx = player.clientWidth / VIDEO_METADATA.width;
      const sy = player.clientHeight / VIDEO_METADATA.height;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      detections.forEach(d => {
        const [x, y, w, h] = d.box;
        const X = x * sx;
        const Y = y * sy;
        const W = w * sx;
        const H = h * sy;
        const color = getColor(d.id);
        
        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(X, Y, W, H);
        
        // Add fill
        ctx.fillStyle = color.replace('hsl', 'hsla').replace(')', ', 0.2)');
        ctx.fillRect(X, Y, W, H);
        
        // Draw label
        const text = `${d.class} #${d.id}`;
        ctx.font = '16px Arial';
        const textWidth = ctx.measureText(text).width + 10;
        ctx.fillStyle = color;
        ctx.fillRect(X, Y - 25, textWidth, 25);
        
        ctx.fillStyle = 'white';
        ctx.fillText(text, X + 5, Y - 7);
      });
    }
    
    // Start animation loop for updating detections while playing
    function startUpdateLoop() {
      if (animationId) cancelAnimationFrame(animationId);
      
      let lastProcessedFrame = -1;
      
      function update() {
        const currentFrameIdx = getCurrentFrame();
        
        // Only fetch if frame changed
        if (currentFrameIdx !== lastProcessedFrame) {
          updateDetections();
          lastProcessedFrame = currentFrameIdx;
        }
        
        animationId = requestAnimationFrame(update);
      }
      
      animationId = requestAnimationFrame(update);
    }
    
    // Stop animation loop
    function stopUpdateLoop() {
      if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
      }
    }
    
    // Toggle box display
    showBoxesToggle.addEventListener('change', () => {
      showBoxes = showBoxesToggle.checked;
      drawDetections();
    });
    
    // Handle object clicks
    canvas.addEventListener('click', (e) => {
      if (!showBoxes) return;
      
      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;
      const sx = player.clientWidth / VIDEO_METADATA.width;
      const sy = player.clientHeight / VIDEO_METADATA.height;
      
      // Check if click is on an object
      for (const d of detections) {
        const [x, y, w, h] = d.box;
        const X = x * sx;
        const Y = y * sy;
        const W = w * sx;
        const H = h * sy;
        
        if (clickX >= X && clickX <= X + W && clickY >= Y && clickY <= Y + H) {
          window.open(d.link, '_blank');
          return;
        }
      }
    });
    
    // Show tooltip on hover
    canvas.addEventListener('mousemove', (e) => {
      if (!showBoxes) return;
      
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const sx = player.clientWidth / VIDEO_METADATA.width;
      const sy = player.clientHeight / VIDEO_METADATA.height;
      
      let hovering = false;
      
      for (const d of detections) {
        const [x, y, w, h] = d.box;
        const X = x * sx;
        const Y = y * sy;
        const W = w * sx;
        const H = h * sy;
        
        if (mouseX >= X && mouseX <= X + W && mouseY >= Y && mouseY <= Y + H) {
          tooltip.style.display = 'block';
          tooltip.style.left = `${e.clientX}px`;
          tooltip.style.top = `${e.clientY - 30}px`;
          tooltip.textContent = `${d.class} (ID: ${d.id}) - Click to shop`;
          
          canvas.style.cursor = 'pointer';
          hovering = true;
          break;
        }
      }
      
      if (!hovering) {
        tooltip.style.display = 'none';
        canvas.style.cursor = 'default';
      }
    });
    
    // Hide tooltip when mouse leaves
    canvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });
    
    // Apply class filter
    document.getElementById('apply-filter').addEventListener('click', () => {
      const filterText = classFilterInput.value.trim();
      filteredClasses = filterText ? 
        filterText.split(',').map(c => c.trim().toLowerCase()).filter(c => c) : 
        [];
      
      statusDisplay.textContent = `Filtering: ${filteredClasses.join(', ') || 'No filter'}`;
      updateDetections();
    });
    
    // Video event handlers
    player.addEventListener('loadedmetadata', () => {
      resizeCanvas();
      updateDetections();
    });
    
    player.addEventListener('play', startUpdateLoop);
    player.addEventListener('pause', () => {
      stopUpdateLoop();
      updateDetections();
    });
    player.addEventListener('seeking', updateDetections);
    
    // Play/pause buttons
    document.getElementById('play').addEventListener('click', () => player.play());
    document.getElementById('pause').addEventListener('click', () => player.pause());
    
    // Handle window resize
    window.addEventListener('resize', resizeCanvas);
    
    // Initialize
    resizeCanvas();
    updateDetections();
    
    // Show processing status
    if (!PROCESSING_DONE) {
      statusDisplay.textContent = "Processing video in background...";
      
      // Check processing status every 5 seconds
      const checkInterval = setInterval(() => {
        fetch('/get_frame_data/0')
          .then(response => response.json())
          .then(() => {
            updateDetections();
          });
      }, 5000);
    }
  </script>
</body>
</html>
'''

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)