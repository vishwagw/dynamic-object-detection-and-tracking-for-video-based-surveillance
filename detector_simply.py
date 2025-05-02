# simple detection without deep learning
# import the necessary libraries
import cv2
import numpy as np

# function for main program:
def main():
    # Initialize video capture (0 for primary camera, or replace with video file path)
    #video_capture = cv2.VideoCapture(0)
    # for video file, use: video_capture = cv2.VideoCapture('path_to_video.mp4')
    video_capture = cv2.VideoCapture('./input2.mp4')

    # Initialize the background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()

    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Apply the background subtractor to detect motion
        fg_mask = back_sub.apply(frame)

        # Optional: Improve the mask by removing noise
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Find contours of the moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Ignore small contours to filter out noise
            if cv2.contourArea(contour) < 500:
                continue

            # Draw bounding boxes around the detected objects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the original frame with bounding boxes
        cv2.imshow('Video Surveillance', frame)

        # Display the foreground mask (optional)
        cv2.imshow('Foreground Mask', fg_mask)

        # Exit when 'q' is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

# initialize the program:
if __name__ == "__main__":
    main()


