# using YOLOv8 model:
import cv2
import numpy as np
from ultralytics import YOLO

def load_yolo_model(model_path="yolov8n.pt"):
    """
    Load YOLOv8 model
    """
    return YOLO(model_path)

def process_video(model_path="yolov8n.pt", video_source="input1.mp4", conf_threshold=0.5):
    """
    Process video with YOLOv8 model for object detection
    """
    # Load YOLOv8 model
    model = load_yolo_model(model_path)
    
    # Initialize video capture
    # For live detection: video_capture = cv2.VideoCapture(0)
    # For video file detection:
    video_capture = cv2.VideoCapture(video_source)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video stream")
            break
            
        # Resize frame for display (original will be used for detection)
        display_frame = cv2.resize(frame, (640, 480))
        
        # Perform detection
        results = model(frame)
        
        # Process and display results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get confidence score
                confidence = float(box.conf)
                
                if confidence >= conf_threshold:
                    # Get bounding box coordinates (normalized)
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class name
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    
                    # Scale coordinates for display frame
                    x1_display, y1_display = int(x1 * 0.5), int(y1 * 0.5)
                    x2_display, y2_display = int(x2 * 0.5), int(y2 * 0.5)
                    
                    # Draw bounding box and label
                    cv2.rectangle(display_frame, (x1_display, y1_display), (x2_display, y2_display), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1_display, y1_display - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow("YOLOv8 Video Surveillance", display_frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # Use YOLOv8 pretrained model - you can use any YOLOv8 model:
    # yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), 
    # yolov8l.pt (large), or yolov8x.pt (xlarge)
    model_path = "yolov8n.pt"  # Change to your specific model path if needed
    
    # Video source - can be camera (0) or video file
    video_source = "input2.mp4"  # Change to your video file path
    
    # Run detection
    process_video(model_path, video_source)

if __name__ == "__main__":
    main()