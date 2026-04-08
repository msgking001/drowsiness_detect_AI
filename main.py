import cv2
import numpy as np
import winsound

# ---------------- BEEP FUNCTION ---------------- #
def alert_beep():
    frequency = 2000   # Hz (sound pitch)
    duration = 500     # ms (0.5 second)
    winsound.Beep(frequency, duration)

# ---------------- MODEL PATH ---------------- #
prototxt = r"C:\Users\DELL USER\projects\drowsiness_detect_system\deploy.prototxt.txt"
model = r"C:\Users\DELL USER\projects\drowsiness_detect_system\res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0)

closed_frames = 0
FRAME_LIMIT = 15

# 🔥 Cooldown control (important)
alert_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Clamp coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face = frame[y1:y2, x1:x2]

            if face is None or face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )

            eyes = eye_cascade.detectMultiScale(gray)

            if len(eyes) == 0:
                closed_frames += 1
            else:
                closed_frames = 0
                alert_active = False  # reset alert when eyes open

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---------------- ALERT ---------------- #
    if closed_frames > FRAME_LIMIT:
        cv2.putText(frame, "DROWSY ALERT!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if not alert_active:
            alert_beep()
            alert_active = True  # prevent continuous spam

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()