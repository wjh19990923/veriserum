import os
import time
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置保存校准后图像的文件夹
output_folder = r'D:\veriserum_calibrated'
os.makedirs(output_folder, exist_ok=True)

# 校准函数
def calibrate_and_save(idx):
    try:
        # 每个子进程单独加载数据集实例
        from veriserum_dataset import Veriserum_calibrated  # 放在函数内避免主进程过多内存占用
        dataset = Veriserum_calibrated(calibration_on_time=True)

        # 设置校准后图像的输出路径
        bs_save_path = os.path.join(output_folder, f'calibrated_bs_{idx:06d}.png')
        fs_save_path = os.path.join(output_folder, f'calibrated_fs_{idx:06d}.png')

        # 检查校准后图像是否已经存在
        if os.path.exists(bs_save_path) and os.path.exists(fs_save_path):
            print(f"Images for index {idx} already exist, skipping calibration.")
            return idx  # 直接返回已存在的索引

        # 获取图像名称和路径
        img_name = dataset.get_img_name(idx)
        img_path_bs = os.path.join(dataset.image_folder, img_name[0])
        img_path_fs = os.path.join(dataset.image_folder, img_name[1])

        # 加载和处理图像
        bit_num = 8
        target_img_bs = np.asarray(Image.open(img_path_bs)).astype("float32") / (2 ** bit_num - 1)
        target_img_fs = np.asarray(Image.open(img_path_fs)).astype("float32") / (2 ** bit_num - 1)

        new_size = (1664, 1600)
        target_img_bs = cv2.resize(target_img_bs, new_size)
        target_img_fs = cv2.resize(target_img_fs, new_size)

        if target_img_bs.max() > 0 and target_img_fs.max() > 0:
            target_img_bs /= target_img_bs.max()
            target_img_fs /= target_img_fs.max()

        bs_corrected, fs_corrected = dataset.get_calibrated_images(idx, target_img_bs, target_img_fs)

        # 保存校准后的图像
        cv2.imwrite(bs_save_path, (bs_corrected * 255).astype(np.uint8))
        cv2.imwrite(fs_save_path, (fs_corrected * 255).astype(np.uint8))

        return idx  # 返回已处理的索引
    except Exception as e:
        print(f"Error processing image {idx}: {e}")
        return None

# 主程序
if __name__ == '__main__':
    start_time = time.time()
    num_workers = max(1, os.cpu_count() - 2)  # 减少进程数
    total_images = 1000  # 设置需要处理的图像数量

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(calibrate_and_save, idx) for idx in range(1, total_images + 1)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calibrating images"):
            future.result()  # 确保捕获到任何错误

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
