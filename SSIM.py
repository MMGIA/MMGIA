from skimage.metrics import structural_similarity as ssim
import cv2

# 加载两张图片（以灰度模式读取）
image1 = cv2.imread("D:\datasets\img5.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("D:\datasets\img6.jpg", cv2.IMREAD_GRAYSCALE)

# 检查图片尺寸是否一致
if image1.shape != image2.shape:
    print("两张图片的尺寸不同，无法计算 SSIM。请调整为相同尺寸。")
else:
    # 计算 SSIM 值
    ssim_value, _ = ssim(image1, image2, full=True)
    print(f"两张图片的 SSIM 值为: {ssim_value}")
