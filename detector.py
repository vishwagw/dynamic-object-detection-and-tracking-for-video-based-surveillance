# detecting dynamic objects:
# detecting in a video frame:

# importing libraries
import cv2
import numpy as np
import argparse
import time

# detecting dynamic objects:
def detect_motion(video_path, sensitivity=20, min_area=500, display=True, output_path=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output video writer if requested
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    print("Motion detection started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:  # Update FPS every 10 frames
            end_time = time.time()
            fps_display = 10 / (end_time - start_time)
            start_time = end_time
        
        # Make a copy of the original frame for drawing on
        output_frame = frame.copy()
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Remove shadows (gray pixels) and apply some noise reduction
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour (moving object)
        motion_detected = False
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter out small contours
            if area < min_area:
                continue
                
            motion_detected = True
            
            # Draw bounding box around the moving object
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_frame, f"Area: {int(area)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add status text to frame
        status = "Motion Detected" if motion_detected else "No Motion"
        cv2.putText(output_frame, f"Status: {status}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(output_frame, f"FPS: {fps_display:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Save the output frame if requested
        if out is not None:
            out.write(output_frame)
        
        # Display the frame if requested
        if display:
            cv2.imshow('Motion Detection', output_frame)
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames.")

def main():
    parser = argparse.ArgumentParser(description='Motion Detection in Video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--sensitivity', type=int, default=20, help='Motion detection sensitivity (lower = more sensitive)')
    parser.add_argument('--min-area', type=int, default=500, help='Minimum area to consider as motion')
    parser.add_argument('--no-display', action='store_true', help="Don't display the video")
    parser.add_argument('--output', type=str, default=None, help='Path to save output video')
    
    args = parser.parse_args()
    
    detect_motion(
        args.video,
        sensitivity=args.sensitivity,
        min_area=args.min_area,
        display=not args.no_display,
        output_path=args.output
    )

if __name__ == "__main__":
    main()

