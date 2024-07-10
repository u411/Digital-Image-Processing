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
image = plt.imread('img.jpg')
if image.ndim == 3:  # 如果是彩色圖像，轉換為灰度圖像
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# 手動實現卷積函數
def convolve(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    output = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return output

# 步驟1: Sobel濾波器
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

sobel_y_kernel = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

sobel_x = convolve(image, sobel_x_kernel)
sobel_y = convolve(image, sobel_y_kernel)
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_magnitude = (255 * sobel_magnitude / np.max(sobel_magnitude)).astype(np.uint8)

# 步驟2: 高斯濾波去噪
gaussian_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]]) / 16.0

gaussian_blurred = convolve(image, gaussian_kernel)

# 步驟3: 正規化和乘積
normalized_blurred = gaussian_blurred / 255.0
result = normalized_blurred * sobel_magnitude

# 步驟4: 和原始圖像相加
final_result = image + result
final_result = np.clip(final_result, 0, 255).astype(np.uint8)

# 步驟1: Laplacian圖像增強
laplacian_mask = np.array([[-1, -1, -1], 
                           [-1,  8, -1], 
                           [-1, -1, -1]])

laplacian_enhanced = convolve(image, laplacian_mask)
laplacian_enhanced_image = image + laplacian_enhanced
laplacian_enhanced_image = np.clip(laplacian_enhanced_image, 0, 255).astype(np.uint8)

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
