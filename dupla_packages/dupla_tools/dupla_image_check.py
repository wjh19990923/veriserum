import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import mean_squared_error
from dupla_discal.disCal import GridCalibration_HoughROI_PassThrough_PointExtractor, \
    GridCalibration_HoughROI_PassThrough_PointExtractor


def check_grid_image(magpose2_image, magpose3_image, f=1):
    # magpose2_image = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos2\CamA_6182_Y20240314H143052_24538_-2_-2.jpg'
    # magpose3_image = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos3\CamA_6218_Y20240314H161620_24647_MagPos3_50cm_0cm_-2.jpg'
    # magpose2_image = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos3\CamA_6232_Y20240314H161840_24675_MagPos3_50cm_0cm\frame_0.jpg'
    # magpose3_image = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos3\CamA_6232_Y20240314H161840_24675_MagPos3_50cm_0cm\frame_1.jpg'
    images = []
    filenames = []
    files = [magpose2_image, magpose3_image]
    # 加载图像
    for file in files:
        print(f"Loading {file}")
        testfile = Path(file)
        calImg = np.asarray(Image.open(testfile).convert('L'))  # 转换为灰度图
        images.append(calImg)
        filenames.append(file[-12:])

    # 计算平均图像
    images = np.array(images)
    avg_img = np.mean(images, axis=0)

    # 计算每张图片与平均图片的MSE
    mse_values = []
    for image in images:
        diff = (image - avg_img) ** 2
        mse_values.append(np.sqrt(diff.mean()))
    print(mse_values)
    # 可视化MSE
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(mse_values)), mse_values, tick_label=filenames)
    plt.xlabel('Image File')
    plt.ylabel('RMSE with Average Image')
    plt.title('RMSE of Each Image Compared to Average Image')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos4 vs. magpos8\Intensity_diff_pose_{f}.png')

    # 可视化平均图像
    plt.figure()
    plt.imshow(avg_img, cmap='gray')
    plt.title('Average Image')
    plt.colorbar()
    plt.savefig(rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos4 vs. magpos8\averaged_img_pose_{f}.png')


def check_discal_on_grid_roi(magpose2_image, magpose3_image, f=1):
    # magpose2_image = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos3\CamA_6232_Y20240314H161840_24675_MagPos3_50cm_0cm\frame_0.jpg'
    # magpose3_image = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos3\CamA_6232_Y20240314H161840_24675_MagPos3_50cm_0cm\frame_1.jpg'
    # magpose2_image = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos2\CamA_6182_Y20240314H143052_24538_-2_-2.jpg'
    # magpose3_image = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos3\CamA_6218_Y20240314H161620_24647_MagPos3_50cm_0cm_-2.jpg'
    check_grid_image(magpose2_image, magpose3_image, f=f)
    images = []
    # filenames = []
    rois_all = []
    files = [magpose2_image, magpose3_image]
    # 加载图像
    for file in files:

        print(f"Loading {file}")
        testfile = Path(file)
        calImg = np.asarray(Image.open(testfile).convert('L'))  # 转换为灰度图
        images.append(calImg)
        # filenames.append(file[-12:])

        """fit from real calibration image, use cartesian polynomials"""
        # plt.figure()
        # plt.imshow(calImg)
        # plt.show()
        # optimised these values to agree pretty well with the image
        gc = GridCalibration_HoughROI_PassThrough_PointExtractor(
            polynomialN=1, regularisation=False
        )
        gc.low_threshold = 20
        gc.high_threshold = 25
        gc.debug = False
        rois = gc.give_rois(calImg)
        # Using hasattr to check if 'rois' attribute exists in the 'gc' object
        if hasattr(gc, 'rois'):
            print(f'ROIS: {gc.rois}')
        rois_all.append(rois)
        print(f'ROIS collected for {file}')

    # compare centre distance of rois
    rois1 = rois_all[0]
    rois2 = rois_all[1]
    rois1_centers = []
    rois2_centers = []
    assert len(rois1) == len(rois2)
    for i in range(len(rois1)):
        xMinPxl = rois1[i].xMinPxl
        xMaxPxl = rois1[i].xMaxPxl
        yMinPxl = rois1[i].yMinPxl
        yMaxPxl = rois1[i].yMaxPxl

        center1 = np.array([(xMinPxl + xMaxPxl) / 2, (yMinPxl + yMaxPxl) / 2])
        rois1_centers.append(center1)

        xMinPxl = rois2[i].xMinPxl
        xMaxPxl = rois2[i].xMaxPxl
        yMinPxl = rois2[i].yMinPxl
        yMaxPxl = rois2[i].yMaxPxl

        center2 = np.array([(xMinPxl + xMaxPxl) / 2, (yMinPxl + yMaxPxl) / 2])
        rois2_centers.append(center2)

    rois1_centers = np.sort(rois1_centers, axis=0)
    rois2_centers = np.sort(rois2_centers, axis=0)
    distances = []
    for i in range(len(rois1)):
        distance = np.linalg.norm(rois1_centers[i] - rois2_centers[i])
        distances.append(distance)
    print(f'individual distances are {distances}')
    # Plotting percentile plot
    percentiles = np.linspace(0, 100, len(distances))
    sorted_distances = np.sort(distances)
    plt.figure()
    plt.plot(percentiles, sorted_distances)
    plt.xlabel('Percentile')
    plt.ylabel('Distance (per pixel)')
    plt.title('Percentile Plot of Error Distances')
    plt.grid(True)
    # plt.show()
    plt.savefig(rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos4 vs. magpos8\ROI_error_pose_{f}.png')
    plt.close()  # 确保释放内存


# img1 = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos3\CamA_6218_Y20240314H161620_24647_MagPos3_50cm_0cm_-2.jpg'
# img2 = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-15\magpos7\CamA_6363_Y20240315H143754_24936_MagPos7_50cm_0cm_-2.jpg'

path1 = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos4'
path2 = rf'P:\Projects\CMB_dual_plane_measurements\2024-03-15\magpos8'
cine_files1 = [f for f in os.listdir(path1) if f.endswith('.jpg')]
cine_files2 = [f for f in os.listdir(path2) if f.endswith('.jpg')]

for f in range(36):
    file1 = os.path.join(path1, cine_files1[f])
    file2 = os.path.join(path2, cine_files2[f])
    check_discal_on_grid_roi(file1, file2, f=f)
