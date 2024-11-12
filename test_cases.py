import os
import unittest
import time

import torch
import numpy as np
from matplotlib import pyplot as plt

from veriserum_dataset import Veriserum_calibrated

DTYPE_TORCH = torch.float32

from torch.utils.data import Sampler, DataLoader
import random
from sampler import NonObsoleteSampler


class TestVeriserumDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 初始化数据集实例
        cls.dataset = Veriserum_calibrated()

    def test_length(self):
        # 检查数据集长度
        self.assertEqual(len(self.dataset), 110990)

    def test_get_item_structure(self):
        # 检查 __getitem__ 的输出格式
        sample = self.dataset[1]
        self.assertIn('image', sample)
        self.assertIn('pose', sample)
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertEqual(sample['image'].shape[2], 3)  # 检查是否为 3 通道图像
        self.assertIsInstance(sample['pose'], torch.Tensor)

    def test_image_normalization(self):
        # 测试图像是否归一化到 [0, 1] 范围内
        sample = self.dataset[1]
        image = sample['image']
        self.assertTrue((image >= 0).all() and (image <= 1).all())

    def test_pose_data(self):
        # 测试姿态数据的格式
        sample = self.dataset[1]
        pose = sample['pose']
        self.assertEqual(pose.shape, (6,))
        self.assertTrue(torch.is_tensor(pose))
        self.assertEqual(pose.dtype, DTYPE_TORCH)

    def test_calibration_data(self):
        # 测试校准数据的提取
        idx = 1
        bs_cal, fs_cal = self.dataset.get_source_intensifier_calibration(idx)
        self.assertIsInstance(bs_cal, dict)
        self.assertIsInstance(fs_cal, dict)
        self.assertIn("cal_mm_per_pxl", bs_cal)
        self.assertIn("cal_focal_length", fs_cal)

    def test_get_img_name(self):
        # 测试图像命名是否正确
        img_name = self.dataset.get_img_name(123)
        self.assertEqual(img_name, ('calibrated_bs_000123.jpg', 'calibrated_fs_000123.jpg'))

    def test_show_pose_and_images(self):
        # 打印姿态并显示图像
        sample = self.dataset[1]
        pose = sample['pose']
        print("Pose:", pose)

        # 显示图像
        bs_img = sample['image'][:, :, 0]
        fs_img = sample['image'][:, :, 1]

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(bs_img, cmap="gray")
        plt.title("BS Image")

        plt.subplot(1, 2, 2)
        plt.imshow(fs_img, cmap="gray")
        plt.title("FS Image")

        plt.show()

    # 在 test_dataloader_speed 测试中使用 NonZeroSampler
    # def test_dataloader_speed(self):
    #     sampler = NonObsoleteSampler(self.dataset)
    #     dataloader = DataLoader(self.dataset, batch_size=16, sampler=sampler, num_workers=4)
    #     num_batches = 1
    #     start_time = time.time()
    #
    #     for i, batch in enumerate(dataloader):
    #         if i >= num_batches:
    #             break
    #
    #     end_time = time.time()
    #     print(f"Time to load {num_batches} batches: {end_time - start_time:.2f} seconds")

    # def test_calibrated_images(self):
    #     self.calibrated_image_folder=rf'D:\veriserum_calibrated'
    #     # 筛选出所有 obsolete=0 的 id
    #     valid_ids = self.dataset.veriserum_poses[self.dataset.veriserum_poses['obsolete'] == 0]['id']
    #
    #     missing_files = []
    #     for idx in valid_ids:
    #         bs_file = os.path.join(self.calibrated_image_folder, f'calibrated_bs_{idx:06d}.jpg')
    #         fs_file = os.path.join(self.calibrated_image_folder, f'calibrated_fs_{idx:06d}.jpg')
    #
    #         # 检查 BS 和 FS 文件是否都存在
    #         if not (os.path.exists(bs_file) and os.path.exists(fs_file)):
    #             missing_files.append(idx)
    #
    #     # 检查是否有缺失的文件
    #     if missing_files:
    #         missing_count = len(missing_files)
    #         print(f"Missing {missing_count} calibrated images for IDs: {missing_files}")
    #     else:
    #         print("All required calibrated images are present.")
    #
    #     # 断言所有图像文件存在
    #     self.assertEqual(len(missing_files), 0, f"Missing calibrated images for IDs: {missing_files}")


if __name__ == '__main__':
    unittest.main()
