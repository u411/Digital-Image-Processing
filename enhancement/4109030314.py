import cv2
import numpy as np
from matplotlib import pyplot as plt

# 計算 PSNR 的函數
def calculate_psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:  # 如果 MSE 為零，表示完全相同，PSNR 無限大
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# 讀取圖像
image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

# 步驟1: Sobel濾波器
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# 步驟2: 高斯濾波去噪
gaussian_blurred = cv2.GaussianBlur(image, (3, 3), 0)

# 步驟3: 正規化和乘積
normalized_blurred = cv2.normalize(gaussian_blurred, None, 0, 1, cv2.NORM_MINMAX)
result = cv2.multiply(normalized_blurred, sobel_magnitude.astype(np.float32), dtype=cv2.CV_32F)

# 步驟4: 和原始圖像相加
final_result = cv2.add(image, result.astype(np.uint8))

# 步驟5: Laplacian圖像增強
laplacian_mask = np.array([[-1, -1, -1], 
                           [-1,  8, -1], 
                           [-1, -1, -1]])
laplacian_enhanced = cv2.filter2D(image, -1, laplacian_mask)
laplacian_enhanced_image = cv2.add(image, laplacian_enhanced)

# 計算 PSNR
psnr_final = calculate_psnr(image, final_result)
psnr_laplacian = calculate_psnr(image, laplacian_enhanced_image)



# 顯示結果
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.imshow(sobel_magnitude, cmap='gray'), plt.title('Sobel Magnitude')
plt.subplot(2, 2, 3), plt.imshow(final_result, cmap='gray'), plt.title('Sobel Enhanced Image')
plt.subplot(2, 2, 4), plt.imshow(laplacian_enhanced_image, cmap='gray'), plt.title('Laplacian Enhanced Image')
plt.show()
print(f"PSNR between original and sobel_enhanced_image: {psnr_final} dB")
print(f"PSNR between original and laplacian_enhanced_image: {psnr_laplacian} dB")


