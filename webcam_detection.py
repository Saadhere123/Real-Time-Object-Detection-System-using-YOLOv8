from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("Real-Time Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()