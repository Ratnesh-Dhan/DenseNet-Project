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

# ---- BIG DISPLAY WINDOW ----
# cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Detection", 1600, 900)

kernel = np.ones((5,5), np.uint8)

# ---- PROCESS EACH FRAME ----
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Shiny aluminium mask (low S, high V)
    al_lower = np.array([0, 0, 170])
    al_upper = np.array([180, 50, 255])
    metal_mask = cv2.inRange(hsv, al_lower, al_upper)

    metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
    metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)

    # 3. Dark separation lines (low V)
    dark_lower = np.array([0, 0, 0])
    # dark_upper = np.array([180, 255, 70])
    # dark_upper = np.array([80, 70, 60])
    dark_upper = np.array([93, 77, 73])
    dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)

    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.dilate(dark_mask, kernel, iterations=2)

    # 4. Split touching ingots
    split_mask = cv2.subtract(metal_mask, dark_mask)

    # 5. Find contours on split mask
    contours, _ = cv2.findContours(
        split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 6. Filter + draw
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / float(bh)

        if 0.7 < aspect_ratio < 3.2:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Aluminium",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    # 7. Show + save
    out.write(frame)
    cv2.imshow("Detection", frame)

    delay = int(1000 / fps)
    if cv2.waitKey(delay) & 0xFF == 27:
        break

# ---- CLEANUP ----
cap.release()
out.release()
cv2.destroyAllWindows()
