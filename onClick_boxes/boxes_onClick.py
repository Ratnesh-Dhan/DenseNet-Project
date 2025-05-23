import cv2
import matplotlib.pyplot as plt
import os

# Load image using OpenCV
image = cv2.imread(r'D:\NML ML Works\Coal photomicrographs\0.1.jpg')  # Replace with your image path
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for matplotlib display

# Store click positions
clicks = []
square_size = 25
half_size = square_size // 2
save_counter = 1

# Create output directory
output_dir = 'squares'
os.makedirs(output_dir, exist_ok=True)

# Event handler
def onclick(event):
    global save_counter
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)

        # Define bounding box coordinates
        top_left_x = max(x - half_size, 0)
        top_left_y = max(y - half_size, 0)
        bottom_right_x = min(x + half_size, image.shape[1])
        bottom_right_y = min(y + half_size, image.shape[0])

        top_left = (top_left_x, top_left_y)
        bottom_right = (bottom_right_x, bottom_right_y)
        clicks.append((top_left, bottom_right))

        # Save the cropped region as PNG
        cropped = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        save_path = os.path.join(output_dir, f'square_{save_counter}.png')
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
            print(f"Saved: {save_path}")
            save_counter += 1
        else:
            while(os.path.exists(os.path.join(output_dir, f'square_{save_counter}.png'))):
                save_counter += 1
            cv2.imwrite(os.path.join(output_dir, f"Square_{save_counter}.png"), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
            print(f"Saved: {os.path.join(output_dir, f'Square_{save_counter}.png')}")
            save_counter += 1

        redraw()

# Redraw image with squares
def redraw():
    img_copy = image.copy()
    for tl, br in clicks:
        cv2.rectangle(img_copy, tl, br, (255, 0, 0), 2)
    ax.clear()
    ax.imshow(img_copy)
    fig.canvas.draw()

# Setup matplotlib window
fig, ax = plt.subplots()
ax.imshow(image)
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
