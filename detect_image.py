from ultralytics import YOLO
import cv2
import pandas as pd

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load image
image_path = "input/test.jpg"

# Run detection
results = model(image_path)

# Read image
image = cv2.imread(image_path)

# CSV data list
data = []

# Process detections
for r in results:
    boxes = r.boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        # Save data
        data.append([class_name, confidence])

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

        # Label text
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

# Save output image
output_path = "output/detected_image.jpg"
cv2.imwrite(output_path, image)

# Save CSV logs
df = pd.DataFrame(data, columns=["Object", "Confidence"])
df.to_csv("output/detection_logs.csv", index=False)

print("✅ Detection Completed")
print(f"Output Saved: {output_path}")

# Display image
cv2.imshow("YOLO Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()