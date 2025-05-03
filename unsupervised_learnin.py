# using unsupervised learning for obeject detection:
import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse
import time
import os
from datetime import datetime

class MovingObjectDetector:
    def __init__(self, bg_history=500, bg_threshold=400, learning_rate=0.01, 
                 min_contour_area=500, blur_size=21, morph_kernel_size=5, 
                 kmeans_clusters=3):
        """
        Initialize the moving object detector with unsupervised learning approach
        
        Parameters:
        -----------
        bg_history: int
            History length for background subtractor
        bg_threshold: int
            Threshold for background subtractor
        learning_rate: float
            Learning rate for background model update
        min_contour_area: int
            Minimum contour area to be considered as a valid object
        blur_size: int
            Size of Gaussian blur kernel
        morph_kernel_size: int
            Size of morphological operations kernel
        kmeans_clusters: int
            Number of clusters for KMeans unsupervised learning
        """
        # Background subtractor - main unsupervised component
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=bg_history, 
            varThreshold=bg_threshold, 
            detectShadows=False
        )
        
        self.learning_rate = learning_rate
        self.min_contour_area = min_contour_area
        self.blur_size = blur_size
        self.morph_kernel_size = morph_kernel_size
        self.kmeans_clusters = kmeans_clusters
        
        # Kernel for morphological operations
        self.kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        
        # Initialize KMeans for unsupervised clustering of motion vectors
        self.kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42)
        
        # For optical flow calculation
        self.prev_frame = None
        self.prev_gray = None
        
        # For storing motion history
        self.motion_history = []
    
    def preprocess_frame(self, frame):
        """Apply preprocessing to frame before detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        return blurred
    
    def detect_motion(self, frame):
        """Detect motion using background subtraction"""
        # Apply the background subtractor
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Apply morphological operations to clean up the mask
        opening = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
        
        # Dilate to fill in holes
        dilated = cv2.dilate(closing, self.kernel, iterations=2)
        
        return dilated
    
    def calculate_optical_flow(self, gray):
        """Calculate optical flow between consecutive frames"""
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Calculate optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Update previous frame
        self.prev_gray = gray
        
        return flow
    
    def cluster_motion_vectors(self, flow):
        """Cluster motion vectors using KMeans"""
        if flow is None:
            return None, None
            
        # Downsample flow for performance
        h, w = flow.shape[:2]
        step = 16  # Sample every 16 pixels
        
        # Extract points and their flow vectors
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        
        # Filter out small movements
        magnitude = np.sqrt(fx*fx + fy*fy)
        mask = magnitude > 1.0
        
        if np.sum(mask) < 10:  # Not enough significant motion vectors
            return None, None
            
        # Get points with significant motion
        points = np.vstack([x[mask], y[mask]]).T
        vectors = np.vstack([fx[mask], fy[mask]]).T
        
        if len(points) < self.kmeans_clusters:
            return points, None  # Not enough points for clustering
            
        # Apply KMeans clustering to motion vectors
        try:
            clusters = self.kmeans.fit_predict(vectors)
            return points, clusters
        except:
            return points, None
    
    def find_contours(self, mask):
        """Find contours in the mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
        
        return valid_contours
    
    def draw_results(self, frame, contours, points=None, clusters=None):
        """Draw detection results on the frame"""
        result = frame.copy()
        
        # Draw contours
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # Label each contour
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"Object {i+1}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw clustered motion vectors if available
        if points is not None and clusters is not None:
            colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), 
                      (255, 0, 255), (0, 255, 255)]
                      
            for i, (point, cluster) in enumerate(zip(points, clusters)):
                color = colors[cluster % len(colors)]
                cv2.circle(result, tuple(point), 5, color, -1)
        
        # Add timestamp
        cv2.putText(result, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        return result
    
    def update_motion_history(self, contours):
        """Update motion history for temporal analysis"""
        if len(contours) > 0:
            # Store centroid of each contour
            centroids = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
            
            self.motion_history.append(centroids)
            
            # Keep history limited to prevent memory issues
            if len(self.motion_history) > 30:  # 30 frames history
                self.motion_history.pop(0)
    
    def analyze_motion_patterns(self):
        """Analyze motion patterns over time (unsupervised approach)"""
        if len(self.motion_history) < 10:  # Need enough history
            return None
            
        # Flatten all centroids
        all_centroids = []
        for centroids in self.motion_history:
            all_centroids.extend(centroids)
            
        if len(all_centroids) < self.kmeans_clusters:
            return None
            
        # Apply KMeans to find motion patterns
        try:
            centroids_array = np.array(all_centroids)
            clusters = self.kmeans.fit_predict(centroids_array)
            
            # Analyze cluster properties (e.g., size, density)
            cluster_info = {}
            for i in range(self.kmeans_clusters):
                points_in_cluster = centroids_array[clusters == i]
                if len(points_in_cluster) > 0:
                    # Calculate cluster center
                    center = np.mean(points_in_cluster, axis=0)
                    # Calculate spread/density
                    spread = np.mean(np.linalg.norm(points_in_cluster - center, axis=1))
                    cluster_info[i] = {
                        "center": center,
                        "size": len(points_in_cluster),
                        "spread": spread
                    }
            
            return cluster_info
        except:
            return None
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Preprocess the frame
        processed = self.preprocess_frame(frame)
        
        # Detect motion using background subtraction
        motion_mask = self.detect_motion(processed)
        
        # Calculate optical flow
        flow = self.calculate_optical_flow(processed)
        
        # Find contours in the motion mask
        contours = self.find_contours(motion_mask)
        
        # Update motion history
        self.update_motion_history(contours)
        
        # Analyze motion patterns
        motion_patterns = self.analyze_motion_patterns()
        
        # Cluster motion vectors using KMeans
        points, clusters = self.cluster_motion_vectors(flow)
        
        # Draw results
        result = self.draw_results(frame, contours, points, clusters)
        
        return result, motion_mask, contours, motion_patterns


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Moving Object Detection for Video Surveillance')
    parser.add_argument('--input', type=str, default='0', 
                        help='Path to input video file or camera index (default: 0)')
    parser.add_argument('--output', type=str, default='', 
                        help='Path to output video file (default: no output)')
    parser.add_argument('--display', action='store_true', 
                        help='Display video frames in window')
    parser.add_argument('--min-area', type=int, default=500, 
                        help='Minimum contour area to consider (default: 500)')
    parser.add_argument('--clusters', type=int, default=3, 
                        help='Number of clusters for KMeans (default: 3)')
    parser.add_argument('--save-detections', action='store_true', 
                        help='Save frames with detections')
    parser.add_argument('--detection-dir', type=str, default='detections', 
                        help='Directory to save detection frames (default: detections)')
    
    args = parser.parse_args()
    
    # Initialize the detector
    detector = MovingObjectDetector(
        min_contour_area=args.min_area,
        kmeans_clusters=args.clusters
    )
    
    # Open video capture
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output is specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Create directory for detections if needed
    if args.save_detections:
        os.makedirs(args.detection_dir, exist_ok=True)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        frame_count += 1
        
        # Process the frame
        result, mask, contours, patterns = detector.process_frame(frame)
        
        # Write to output video if specified
        if writer is not None:
            writer.write(result)
            
        # Save frames with detections if specified
        if args.save_detections and len(contours) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(args.detection_dir, f"detection_{timestamp}.jpg")
            cv2.imwrite(filename, result)
        
        # Display the result if specified
        if args.display:
            cv2.imshow('Motion Detection', result)
            cv2.imshow('Motion Mask', mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Print motion analysis occasionally
        if frame_count % 30 == 0 and patterns is not None:
            print(f"Frame {frame_count}: Motion patterns detected:")
            for cluster_id, info in patterns.items():
                print(f"  Cluster {cluster_id}: {info['size']} points, "
                      f"center at ({info['center'][0]:.1f}, {info['center'][1]:.1f}), "
                      f"spread {info['spread']:.1f}")
    
    # Calculate and print performance metrics
    elapsed_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count / elapsed_time:.2f}")
    
    # Release resources
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()