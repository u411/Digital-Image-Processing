import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# Read image
image_path = 'img.jpg'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Add noise: 1/4 is 0, 1/3 is 255
noisy_image = gray_image.copy()
num_pixels = gray_image.shape[0] * gray_image.shape[1]
num_noise_0 = int(num_pixels * 0.25)
num_noise_255 = int(num_pixels * 0.33)

# Random choose 
coords_0 = (np.random.randint(0, gray_image.shape[0], num_noise_0), np.random.randint(0, gray_image.shape[1], num_noise_0))
coords_255 = (np.random.randint(0, gray_image.shape[0], num_noise_255), np.random.randint(0, gray_image.shape[1], num_noise_255))

noisy_image[coords_0] = 0
noisy_image[coords_255] = 255

# Add "NCHU" and "CSE" noise
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(noisy_image, 'NCHU', (10, 50), font, 1, 0, 1, cv2.LINE_AA)
cv2.putText(noisy_image, 'CSE', (200, 50), font, 1, 255, 1, cv2.LINE_AA)
text_mask = (noisy_image != gray_image)

# Manual median filter implementation
def median_filter(image, kernel_size=3):
    padded_image = np.pad(image, kernel_size // 2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
            median_value = np.median(neighborhood)
            filtered_image[i, j] = median_value

    return filtered_image

# Median filter
median_filtered_img = median_filter(noisy_image, 3)
median_filtered_img[text_mask] = gray_image[text_mask]

# Switch median filter
def switch_median_filter(image, kernel_size=3, threshold=20):
    padded_image = np.pad(image, kernel_size // 2, mode='constant', constant_values=0)
    result = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size]
            median_value = np.median(neighborhood)
            diff = np.abs(image[i, j] - median_value)
            if diff > threshold:
                result[i, j] = median_value
            else:
                result[i, j] = image[i, j]

    return result

switch_median_filtered_img = switch_median_filter(noisy_image)

# Function to calculate PSNR
def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:  # If MSE is zero, the PSNR is infinite
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Calculate PSNR
psnr_median = calculate_psnr(gray_image, median_filtered_img)
psnr_switch_median = calculate_psnr(gray_image, switch_median_filtered_img)

# Convert images to color for display
noisy_image_color = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2BGR)
median_filtered_img_color = cv2.cvtColor(median_filtered_img, cv2.COLOR_GRAY2BGR)
switch_median_filtered_img_color = cv2.cvtColor(switch_median_filtered_img, cv2.COLOR_GRAY2BGR)

# Show images
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Impulse Noise Image')
plt.imshow(noisy_image_color)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Median Filtered Image')
plt.imshow(median_filtered_img_color)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Switch Median Filtered Image')
plt.imshow(switch_median_filtered_img_color)
plt.axis('off')

plt.show()

print(f'PSNR of Median Filtered Image: {psnr_median:.2f} dB')
print(f'PSNR of Switch Median Filtered Image: {psnr_switch_median:.2f} dB')
