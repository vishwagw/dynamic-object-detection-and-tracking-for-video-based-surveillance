# using YOLO deep learning model:
import cv2
import numpy as np

def load_yolo_model():
    # Load YOLO model configuration and weights
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    # Load COCO dataset class labels
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    return net, classes

def detect_objects(frame, net, output_layers):
    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    # Perform forward pass to get detections
    return net.forward(output_layers)

def main():
    # Load YOLO model and classes
    net, classes = load_yolo_model()
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Initialize video capture
    # for live detection:
    #video_capture = cv2.VideoCapture(0)  # Replace 0 with video file path if needed
    # for video file detection:
    video_capture = cv2.VideoCapture("input2.mp4")  # Replace with your video file path

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)  # Resize frame for faster processing

        # Detect objects
        height, width, _ = frame.shape
        detections = detect_objects(frame, net, output_layers)

        # Draw bounding boxes for detected objects
        for detection in detections:
            for obj in detection:
                scores = obj[5:]  # Class scores start at index 5
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Filter by confidence threshold
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("YOLO Video Surveillance", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()