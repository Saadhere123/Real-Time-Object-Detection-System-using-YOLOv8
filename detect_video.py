from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Video path
video_path = "input/video.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Video writer settings
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "output/detected_video.mp4",
    fourcc,
    fps,
    (width, height)
)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO prediction
    results = model(frame)

    # Annotated frame
    annotated_frame = results[0].plot()

    # Write frame
    out.write(annotated_frame)

    # Display
    cv2.imshow("YOLO Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()

print("✅ Video Detection Completed")