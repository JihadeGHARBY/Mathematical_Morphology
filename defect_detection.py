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

# 1. Load an metal sheet image
img = cv2.imread('metal_sheet.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("metal_sheet.png not found")

# 2. Choose a structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

# 3. White top-hat to highlight bright defects
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 4. Threshold to get a binary defect mask
_, mask = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)

# 5. Clean small specks with a 5×5 opening
clean_mask = cv2.morphologyEx(mask,
                              cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

# 6. Overlay defects in red on the original
overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
overlay[clean_mask == 255] = [0, 0, 255]

# 7. Display all steps
show_images(
    [img, tophat, clean_mask, overlay],
    ['Original Image',
     'White Top-Hat Result',
     'Binary Defect Mask',
     'Defects Highlighted']
)
