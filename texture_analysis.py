import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 1. Load a textured image (e.g. cameraman.tif)
img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("cameraman.tif not found")

# 2. Choose a structuring element for texture scale
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# 3. Smooth background via opening
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 4. Extract texture component (original minus opened)
texture = cv2.subtract(img, opened)

# 5. Compute morphological gradient (edge emphasis)
dilated = cv2.dilate(img, kernel)
eroded  = cv2.erode(img, kernel)
gradient = cv2.subtract(dilated, eroded)

# 6. Display results
show_images(
    [img, opened, texture, gradient],
    ['Original Image',
     'After Opening (Background)',
     'Texture Component',
     'Morphological Gradient']
)
