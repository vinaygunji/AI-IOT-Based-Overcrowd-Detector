from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

capacity = 5

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    people_count = 0

    # Count only people
    for box in results[0].boxes:
        class_id = int(box.cls[0])

        if class_id == 0:
            people_count += 1

    output = results[0].plot()

    # Crowd level
    percentage = (people_count / capacity) * 100

    if percentage < 80:
        status = "NORMAL"
        alert = "NO ALERT"

    elif percentage <= 100:
        status = "WARNING"
        alert = "PREPARE"

    else:
        status = "OVERCROWDED"
        alert = "ALERT SENT"

    # Display information
    cv2.putText(
        output,
        f"People Detected: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"Capacity: {capacity}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"Status: {status}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"IoT Alert: {alert}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("AI + IoT Overcrowd Detector", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()