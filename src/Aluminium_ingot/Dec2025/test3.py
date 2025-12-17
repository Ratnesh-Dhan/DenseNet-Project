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

    # 1. Convert to grayscale for better edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply CLAHE to normalize lighting variations
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. Threshold to get bright aluminum regions
    _, thresh = cv2.threshold(enhanced, 140, 255, cv2.THRESH_BINARY)
    
    # 4. Morphological operations to separate ingots
    # Use rectangular kernel to emphasize horizontal/vertical structures
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # Open to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # 5. Edge detection to find boundaries
    edges = cv2.Canny(opened, 50, 150)
    
    # Dilate edges slightly to connect gaps
    kernel_dilate = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
    
    # 6. Find contours
    contours, _ = cv2.findContours(
        edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 7. Filter and draw with better criteria
    ingot_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by area (adjust based on your video resolution)
        if area < 1000 or area > 50000:
            continue

        # Get bounding rectangle
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Calculate aspect ratio
        aspect_ratio = bw / float(bh) if bh > 0 else 0
        
        # Filter by aspect ratio - ingots are wider than tall
        if aspect_ratio < 1.2 or aspect_ratio > 5.0:
            continue
        
        # Calculate extent (area vs bounding box area) for rectangularity
        rect_area = bw * bh
        extent = area / rect_area if rect_area > 0 else 0
        
        # Ingots should be fairly rectangular
        if extent < 0.5:
            continue
        
        # Additional brightness check in original image
        roi = gray[y:y+bh, x:x+bw]
        mean_brightness = np.mean(roi)
        
        if mean_brightness < 100:  # Should be bright
            continue
        
        ingot_count += 1
        
        # Draw detection
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(
            frame, f"Ingot {ingot_count}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    # Display total count
    cv2.putText(
        frame, f"Total Ingots: {ingot_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 8. Show + save
    out.write(frame)
    cv2.imshow("Detection", frame)
    cv2.imshow("Processed", edges_dilated)  # Debug view

    delay = int(1000/fps) if fps > 0 else 30
    if cv2.waitKey(delay) & 0xFF == 27:  # ESC
        break

# ---- CLEANUP ----
cap.release()
out.release()
cv2.destroyAllWindows()