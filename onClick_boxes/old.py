import cv2
import matplotlib.pyplot as plt

# Load image using OpenCV
image = cv2.imread(r'D:\NML ML Works\Coal photomicrographs\0.1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Store click positions
clicks = []

# Event handler
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        square_size = 50
        top_left = (x - square_size // 2, y - square_size // 2)
        bottom_right = (x + square_size // 2, y + square_size // 2)
        clicks.append((top_left, bottom_right))
        redraw()

# Redraw image with squares
def redraw():
    img_copy = image.copy()
    for tl, br in clicks:
        cv2.rectangle(img_copy, tl, br, (255, 0, 0), 2)
    ax.imshow(img_copy)
    fig.canvas.draw()

# Setup matplotlib window
fig, ax = plt.subplots()
ax.imshow(image)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
