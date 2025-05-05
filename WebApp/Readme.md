
# YOLO Detection Web Application

This application provides a web interface for object detection and tracking using YOLO and DeepSORT.

## Prerequisites

- Python 3.7+
- Node.js and npm
- TensorRT
- CUDA and pyCUDA
- OpenCV

## Backend Setup

1. Install Python dependencies:
```
pip install flask flask-cors opencv-python numpy tensorrt pycuda deep-sort-realtime
```

2. Place your model.engine file in the root directory.

3. Place your video.mp4 file in the root directory.

4. Update the custom_classes list in app.py with your 4 custom classes.

5. Update the class_links dictionary in app.py with the URLs associated with your classes.

6. Start the backend server:
```
python app.py
```

## Frontend Setup (Optional React App)

1. Navigate to the frontend directory:
```
cd frontend
```

2. Install Node.js dependencies:
```
npm install
```

3. Start the React development server:
```
npm start
```

## Usage

1. Open a web browser and navigate to:
   - http://localhost:5000 (for the Flask-served HTML version)
   - http://localhost:3000 (for the React app version)

2. Click "Start Detection" to begin processing the video.

3. Click on the bounding boxes of detected objects to open the associated links.

4. Click "Stop Detection" to stop processing.

## Note

The app provides two frontend options:
1. A simple HTML/JavaScript version served directly by Flask
2. A React application with more advanced features

You can use either depending on your requirements.