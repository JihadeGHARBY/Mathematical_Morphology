# segmentation_preprocessing_cameraman.py

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

# 1. Load cameraman.tif in grayscale
img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("cameraman.tif not found")

# 2. Denoise: opening then closing with a 5×5 ellipse
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

# 3. Sure background via dilation
sure_bg = cv2.dilate(cleaned, kernel, iterations=3)

# 4. Sure foreground via distance transform + threshold
dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)

# 5. Unknown region = bg minus fg
unknown = cv2.subtract(sure_bg, sure_fg)

# 6. Marker labeling
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1            # background = 1
markers[unknown == 255] = 0      # unknown = 0

# 7. Watershed (need a color version for display)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
markers_ws = cv2.watershed(img_color, markers.copy())
# mark boundaries in red
overlay = img_color.copy()
overlay[markers_ws == -1] = [255, 0, 0]

# 8. Show all stages
show_images(
    [img, cleaned, sure_bg, sure_fg, unknown, overlay],
    ['Original Gray',
     'Denoised (Open+Close)',
     'Sure Background',
     'Sure Foreground',
     'Unknown Region',
     'Watershed Result']
)
