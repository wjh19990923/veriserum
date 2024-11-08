import numpy as np
import pycine
import matplotlib.pyplot as plt
import cv2
import os
import IPython

from pycine.color import color_pipeline, resize
from pycine.file import read_header

from pycine.raw import read_frames


# exporting cine images using pycine, having trouble in bit conversion so that output image is different from original
def store_frames(cine_file, start_frame=1, target_folder=None, filename=None):
    # 使用pycine读取Cine文件的头信息

    raw_images, setup, bpp = read_frames(cine_file, start_frame=start_frame)
    # frame = next(raw_images)  # 使用next来获取第一帧
    # breakpoint()
    # rgb_images = (color_pipeline(raw_image, setup=setup, bpp=bpp) for raw_image in raw_images)

    for i, rgb_image in enumerate(raw_images):
        frame = start_frame + i
        print(rgb_image.min(), rgb_image.max())
        print(rgb_image.dtype)
        rgb_image = ((rgb_image / rgb_image.max()) * 255).astype(np.uint8)
        # breakpoint()
        new_folder = os.path.join(target_folder, os.path.splitext(filename)[0])
        os.makedirs(new_folder, exist_ok=True)
        image_name = rf'{os.path.splitext(filename)}_frame_{i}.jpg'
        cv2.imwrite(os.path.join(new_folder, image_name), rgb_image)


for num in [3, 4]:
    target_folder = rf"P:\Projects\CMB_dual_plane_measurements\2024-03-14\magpos{num}"
    for file in os.listdir(target_folder):
        if file.endswith('cine'):
            print(file)
            cine_file = os.path.join(target_folder, file)
            header = read_header(cine_file)
            count = header["cinefileheader"].ImageCount
            print(count)
            store_frames(cine_file, start_frame=1, target_folder=target_folder, filename=file)
