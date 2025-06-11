import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display multiple images
def show_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 1. Create a blank black image
image = np.zeros((200, 200), dtype=np.uint8)

# 2. Define two triangles touching at their tips
triangle1 = np.array([[50, 150], [100, 100], [50, 50]], np.int32)
triangle2 = np.array([[150, 50], [100, 100], [150, 150]], np.int32)

# 3. Draw both triangles filled in white
cv2.drawContours(image, [triangle1], 0, 255, -1)
cv2.drawContours(image, [triangle2], 0, 255, -1)

# 4. Erode to break the connection
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(image, kernel, iterations=1)

# 5. Dilate to restore shape (without reconnecting)
separated = cv2.dilate(eroded, kernel, iterations=1)

# 6. Show results
show_images(
    [image, eroded, separated],
    ['Original (Touching Triangles)', 'After Erosion', 'After Erosion + Dilation']
)

# 7. Optional: Count components
num_labels, labels = cv2.connectedComponents(separated)
print(f"Number of separated objects: {num_labels - 1}")
