from ultralytics import YOLO
import cv2
from heatmap import make_heatmap
from arduino_sender import send_risk_to_arduino
from alerts import send_email_alert

MODEL_PATH = "yolov8n.pt"   # use pretrained model

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, conf=0.3)

    persons = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            clsid = int(box.cls[0])

            # person class
            if clsid == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    count = len(persons)

    # if count < 5:
    #     risk = "LOW"
    # elif count < 10:
    #     risk = "MEDIUM"
    # else:
    #     risk = "HIGH"

    if count >= 1:
        risk = "HIGH"
    else:
        risk = "LOW"

    if risk == "HIGH":
        print("🔥 HIGH RISK DETECTED")
        send_email_alert(count)


    heatmap_img = make_heatmap(frame, persons, grid=(3,3))
    overlay = cv2.addWeighted(frame, 0.7, heatmap_img, 0.3, 0)

    cv2.putText(overlay, f"Count: {count} | Risk: {risk}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Crowd Monitor", overlay)

    send_risk_to_arduino(risk)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
