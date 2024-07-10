import cv2
import numpy as np
import matplotlib.pyplot as plt

def boundary_extraction(img):
    # 二值化影像
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 定義結構元素
    kernel = np.ones((3, 3), np.uint8)
    
    # 侵蝕操作 
    h, w = binary.shape
    erosion = np.zeros((h, w), np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            if binary[i, j] == 255:
                if all(binary[i-1:i+2, j-1:j+2].flatten() == kernel.flatten() * 255):
                    erosion[i, j] = 255
    
    # 邊界提取
    boundary = binary - erosion
    
    return boundary

def manual_flood_fill(img, seed_point, new_value):
    h, w = img.shape
    fill_value = img[seed_point[1], seed_point[0]]
    to_fill = [(seed_point[0], seed_point[1])]

    while to_fill:
        x, y = to_fill.pop()
        if img[y, x] == fill_value:
            img[y, x] = new_value
            if x > 0 and img[y, x-1] == fill_value:
                to_fill.append((x - 1, y))
            if x < w - 1 and img[y, x+1] == fill_value:
                to_fill.append((x + 1, y))
            if y > 0 and img[y-1, x] == fill_value:
                to_fill.append((x, y - 1))
            if y < h - 1 and img[y+1, x] == fill_value:
                to_fill.append((x, y + 1))

def region_filling(img):
    # 二值化影像
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # 取反以獲得背景
    binary_inv = 255 - binary
    
    # 創建種子影像
    h, w = binary.shape
    filled = binary.copy()
    
    # 中心點作為種子點
    center = (0 // 2, 0 // 2)
    
    # 手動填充
    manual_flood_fill(filled, center, 255)
    
    # 填充内部空心圆区域
    filled = 255 - filled
    
    return filled

def process_images(image_path_1, image_path_2):
    # 讀取影像1 (侵蝕操作)
    img1 = cv2.imread(image_path_1)
    if img1 is None:
        print("Error: Image 1 not found or unable to load.")
        return
    
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 讀取影像2 (區域填充)
    img2 = cv2.imread(image_path_2)
    if img2 is None:
        print("Error: Image 2 not found or unable to load.")
        return
    
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 邊界提取
    boundary = boundary_extraction(img1)


    # 區域填充
    filled = region_filling(img2)
    
    # 顯示結果
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title('Original Image 1')
    plt.imshow(img1, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title('Boundary Extracted')
    plt.imshow(boundary, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title('Original Image 2')
    plt.imshow(img2, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.title('Region Filled ')
    plt.imshow(filled, cmap='gray')
    plt.show()

# 使用範例
process_images('img1.png', 'img2.png')
