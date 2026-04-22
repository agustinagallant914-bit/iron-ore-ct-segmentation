import os
import cv2
import numpy as np
from tqdm import tqdm


def slice_data(img_dir, label_dir, save_dir, patch_size=512, stride=448):
    """
    针对TIF格式的离线切片脚本：
    1. 支持中文路径读取
    2. 提取4通道 (RGB + 梯度)
    3. 过滤背景块以优化训练效率
    """
    img_save = os.path.join(save_dir, "images")
    mask_save = os.path.join(save_dir, "masks")
    os.makedirs(img_save, exist_ok=True)
    os.makedirs(mask_save, exist_ok=True)

    # 获取所有tif文件
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    print(f"发现 {len(img_names)} 张TIF图像，开始切片...")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    count = 0

    for name in tqdm(img_names):
        img_path = os.path.join(img_dir, name)
        label_path = os.path.join(label_dir, f"seg_{name}")
        if not os.path.exists(label_path): continue

        # 鲁棒读取 TIF
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or label is None: continue

        # 1. 图像预处理 (与 1.py 逻辑保持一致)
        img_clahe = clahe.apply(img)
        img_smooth = cv2.GaussianBlur(img_clahe, (5, 5), 0)
        sobelx = cv2.Sobel(img_smooth, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        combined = np.concatenate([img_rgb, gradient[..., np.newaxis]], axis=-1)

        # 2. 标签映射
        mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        mask[(label[..., 2] > 150) & (label[..., 1] < 50) & (label[..., 0] < 50)] = 1  # 红
        mask[(label[..., 1] > 150) & (label[..., 2] < 50) & (label[..., 0] < 50)] = 2  # 绿
        mask[(label[..., 0] > 150) & (label[..., 1] < 50) & (label[..., 2] < 50)] = 3  # 蓝

        # 3. 动态滑动窗口切片
        h, w = img.shape
        # 确保整除或覆盖边缘，这里通过填充逻辑或控制滑动
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                img_patch = combined[y:y + patch_size, x:x + patch_size]
                mask_patch = mask[y:y + patch_size, x:x + patch_size]

                # 4. 样本平衡：保留包含目标的切片，仅保留10%的背景块
                if np.any(mask_patch > 0) or np.random.random() < 0.1:
                    np.save(os.path.join(img_save, f"{count}.npy"), img_patch)
                    cv2.imwrite(os.path.join(mask_save, f"{count}.png"), mask_patch)
                    count += 1

    print(f"预处理完成！共生成 {count} 个有效切片，保存在 {save_dir}")


if __name__ == "__main__":
    # 执行前请确保 yuantu 和 biaozhu 文件夹路径正确
    slice_data("yuantu", "biaozhu", "local_patches")