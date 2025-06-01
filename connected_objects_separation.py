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

# 1. Create a synthetic image with two connected circles
image = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(image, (70, 100), 40, 255, -1)
cv2.circle(image, (120, 100), 40, 255, -1)

# 2. Apply erosion to separate them
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
eroded = cv2.erode(image, kernel, iterations=1)

# 3. Then dilate to recover the original size without reconnecting
separated = cv2.dilate(eroded, kernel, iterations=1)

# 4. Display results
show_images(
    [image, eroded, separated],
    ['Original (Connected Objects)', 'After Erosion', 'After Erosion + Dilation (Separated)']
)

# 5. Optional: Count separated components
num_labels, labels = cv2.connectedComponents(separated)
print(f"Number of separated objects: {num_labels - 1}")  # Subtract background