import cv2
import numpy as np

# ---- INPUT / OUTPUT ----
VIDEO_IN  = "Aluminum_Ingot.mp4"
VIDEO_OUT = "output_detected.mp4"

cap = cv2.VideoCapture(VIDEO_IN)

if not cap.isOpened():
    raise RuntimeError("Video not opening. Congrats.")

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

# ---- PROCESS EACH FRAME ----

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. HSV conversion
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Aluminium color mask (tune if needed)
    lower = np.array([0, 0, 160])
    upper = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 3. Morph cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4. Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 5. Filter + draw

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / float(bh)

        if 0.8 < aspect_ratio < 3.5:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame, "Aluminium",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    # 6. Show + save
    out.write(frame)
    cv2.imshow("Detection", frame)

    delay = int(1000/fps)
    if cv2.waitKey(delay) & 0xFF == 27:  # ESC
        break

# ---- CLEANUP ----
cap.release()
out.release()
cv2.destroyAllWindows()
