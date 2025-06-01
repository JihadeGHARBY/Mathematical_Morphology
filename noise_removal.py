import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display multiple images side by side
def show_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 1. Load the image in grayscale
image = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("The image 'cameraman.tif' was not found.")

# 2. Add salt and pepper noise
def add_salt_and_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    num_salt = int(amount * img.size)
    num_pepper = int(amount * img.size)

    # Add white pixels (salt)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    # Add black pixels (pepper)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

noisy_image = add_salt_and_pepper_noise(image)

# 3. Define structuring element (3x3 square)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 4. Apply morphological opening to remove white noise (salt)
opened = cv2.morphologyEx(noisy_image, cv2.MORPH_OPEN, kernel)

# 5. Apply morphological closing to remove black noise (pepper)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# 6. Show results
show_images(
    [image, noisy_image, opened, cleaned],
    ['Original Image', 'Noisy Image', 'After Opening', 'After Opening + Closing']
)
