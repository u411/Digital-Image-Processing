import cv2
import numpy as np

points_l = []
points_r = []
points_t = []
points_s = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[1].append((x, y))
        cv2.circle(param[0], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", param[0])

def bilinear_interpolation(img, x, y):
    h, w = img.shape[:2]
    
    if x < 0 or y < 0 or x >= w or y >= h:
        return np.zeros(3)

    x1 = int(x)
    y1 = int(y)
    x2 = min(x1 + 1, w - 1)
    y2 = min(y1 + 1, h - 1)

    r1 = (x2 - x) * img[y1, x1] + (x - x1) * img[y1, x2]
    r2 = (x2 - x) * img[y2, x1] + (x - x1) * img[y2, x2]
    
    return (y2 - y) * r1 + (y - y1) * r2

def inverse_map(panorama, img, M):
    inv_M = cv2.invertAffineTransform(M)
    h, w = panorama.shape[:2]
    result = np.zeros_like(panorama)

    for y in range(h):
        for x in range(w):
            src_point = np.dot(inv_M, np.array([x, y, 1]))
            src_x, src_y = src_point[0], src_point[1]

            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                result[y, x] = bilinear_interpolation(img, src_x, src_y)

    return result

def main():
    # 讀取兩張圖像
    img_l = cv2.imread('img/Ri.jpg')
    img_r = cv2.imread('img/Mi.jpg')
    img_t = cv2.imread('img/Le.jpg')

    # 選取左圖的對應點
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img_l)
    cv2.setMouseCallback("Image", mouse_callback, [img_l, points_l])
    while len(points_l) < 3:
        cv2.waitKey(100)
    cv2.destroyAllWindows()

    # 選取右圖的對應點
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img_r)
    cv2.setMouseCallback("Image", mouse_callback, [img_r, points_r])
    while len(points_r) < 3:
        cv2.waitKey(100)
    cv2.destroyAllWindows()

    # 計算從右圖到左圖的仿射變換矩陣
    M = cv2.getAffineTransform(np.float32(points_r[:3]), np.float32(points_l[:3]))

    # 確定全景圖像的尺寸
    panorama_height = max(img_l.shape[0], img_r.shape[0])
    panorama_width = img_l.shape[1] + img_r.shape[1]

    # 創建一個空的全景圖像
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    # 將左圖拷貝到全景圖像中
    panorama[:img_l.shape[0], :img_l.shape[1]] = img_l

    # 將右圖逆向映射到左圖的坐標空間中
    inv_warp_img_r = inverse_map(panorama, img_r, M)
    
    # 在重疊區域將右圖像素放置到全景圖像中
    for y in range(panorama.shape[0]):
        for x in range(panorama.shape[1]):
            if not np.all(inv_warp_img_r[y, x] == 0):
                panorama[y, x] = inv_warp_img_r[y, x]

    
    # 選取左圖的對應點
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", panorama)
    cv2.setMouseCallback("Image", mouse_callback, [panorama, points_s])
    while len(points_s) < 3:
        cv2.waitKey(100)
    cv2.destroyAllWindows()

    # 選取右圖的對應點
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img_t)
    cv2.setMouseCallback("Image", mouse_callback, [img_t, points_t])
    while len(points_t) < 3:
        cv2.waitKey(100)
    cv2.destroyAllWindows()

    M = cv2.getAffineTransform(np.float32(points_t[:3]), np.float32(points_s[:3]))
    
    # 將右圖逆向映射到左圖的坐標空間中
    inv_warp_img_t = inverse_map(panorama, img_t, M)
    
    # 在重疊區域將右圖像素放置到全景圖像中
    for y in range(panorama.shape[0]):
        for x in range(panorama.shape[1]):
            if not np.all(inv_warp_img_t[y, x] == 0):
                panorama[y, x] = inv_warp_img_t[y, x]


    # 顯示拼接後的全景圖像
    cv2.namedWindow('Panorama', cv2.WINDOW_NORMAL)
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("img/panorama.jpg", panorama)

if __name__ == "__main__":
    main()
