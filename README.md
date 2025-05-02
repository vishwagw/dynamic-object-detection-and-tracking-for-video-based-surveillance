# Dynamic object detection for video surveillance

This repository contains the AI applications that have been built for moving/dynamic object detection and tracking.
The appilcations are in a range from simple detectors to advanced trackers.

1. Simple detectors:
File/s: simple_detector.py
they are not using advanced machine learning algorithms. 
Libraries:
 * OpenCV
 * Python
 * Numpy

The key steps involve capturing video from a camera, detecting motion (usually through background subtraction or frame differencing), and highlighting detected objects.

* Video Capture: Initializes the camera or reads a video file using OpenCV's cv2.VideoCapture.
* Background Subtractor: Uses cv2.createBackgroundSubtractorMOG2 to detect motion by comparing frames.
* Noise Reduction: Applies cv2.medianBlur to remove noise from the foreground mask.
* Contour Detection: Detects contours (moving objects) in the foreground mask.
* Bounding Boxes: Draws rectangles around detected objects to highlight them.
* Display: Shows the video feed with bounding boxes and the foreground mask.
* Exit Condition: Stops the script when the user presses 'q'.

2. Using pre-trained models:
File/s: v4_detector.py, v7_detect.py, v7_detect_track.py, v8_detector.py
for further accurate results, these detectors use pre-trained models. 
These pre-trained models are deep learning algorithms.
The most famous algorithm is YOLO. 
For this project I am using different versions of YOLO (v4, v7 and v8) for better result generation.
We can also use SSD(single shot detector) and Faster-RNN deep learnin models as well.
This is supervised learning approach with training data.
Libraries/framework:
 * OpenCV
 * Numpy
 * ultralytics 
 * yolo

weigh files: 
V4:
* yolov4.weights
* yolov3.cfg
* coconames
V7:
* yolov7n.pt
V8:
* yolov8n.pt

3. Unsupersives learning:
This more advanced approach than supervised learning in deep learning algorithms. 
We are using unflitered/untrained data for direct detection.
This application has following features:
* Unsupervised Learning Approach: Uses background subtraction, KMeans clustering, and motion pattern analysis to detect and track moving objects
* Advanced Motion Detection: Combines background subtraction with optical flow analysis
* Temporal Analysis: Tracks motion patterns over time to identify consistent movement
* Real-time Processing: Optimized for video surveillance applications
* Flexible Input Sources: Works with webcams, video files, and IP cameras
* Detection Storage: Option to save frames containing detected motion

work process:
* Background Subtraction: Uses an adaptive MOG2 background subtractor to identify moving objects
* Motion Vector Analysis: Calculates optical flow between frames to understand movement direction
* Unsupervised Clustering: Uses KMeans to group similar motions without predefined categories
* Motion Pattern Recognition: Analyzes movement patterns over time to identify meaningful activity

