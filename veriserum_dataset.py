import os
import re
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.spatial.transform import Rotation as R
from torch.utils.tensorboard import SummaryWriter
from dupla_packages.dupla_tools.dupla_classes import DistortionCalibration

import cv2
import pandas as pd

DTYPE_TORCH = torch.float32


class Veriserum_calibrated(Dataset):
    def __init__(self, transform=None, anatomy='femur', calibration_on_time=False,data_length=None):
        self.original_image_folder = r'D:\veriserum_collection_compressed'
        self.calibrated_image_folder = rf'D:\veriserum_calibrated'

        self.image_folder = self.original_image_folder if calibration_on_time else self.calibrated_image_folder
        self.sqlite3db = rf'veriserumSqlite3_refined.db'
        self.veriserum_poses = pd.read_csv('csv_files/veriserum_pose_reloaded_refined_all.csv')
        self.ids = self.veriserum_poses['id'][0:110990]  # decide the length of dataset we use
        self.transform = transform
        self.calibration_on_time = calibration_on_time
        self.anatomy = anatomy
        self.data_length=data_length

        self.veriserum_discal = pd.read_csv(r'csv_files/distortion_calibration_reloaded.csv')
        self.veriserum_sical = pd.read_csv(r'csv_files/source_int_calibration_reloaded.csv')

    def __len__(self):
        if self.data_length is None:
            total_length = len(self.veriserum_poses)
        else:
            total_length=self.data_length
        return total_length

    def __getitem__(self, idx):
        img_name = self.get_img_name(idx)
        img_path_bs = os.path.join(self.image_folder, img_name[0])
        img_path_fs = os.path.join(self.image_folder, img_name[1])

        # 加载图像
        target_img_bs = np.asarray(Image.open(img_path_bs)).astype("float32") / 255.0
        target_img_fs = np.asarray(Image.open(img_path_fs)).astype("float32") / 255.0

        if target_img_bs.max() == 0 or target_img_fs.max() == 0:
            print(f"Image max value is 0 for {img_name}")
        else:
            target_img_bs /= target_img_bs.max()
            target_img_fs /= target_img_fs.max()

        if self.calibration_on_time:
            target_img_bs, target_img_fs = self.get_calibrated_images(idx, target_img_bs, target_img_fs)

        # 将图像堆叠为 3 通道并应用变换
        combined_image = torch.tensor(np.stack([target_img_bs, target_img_fs, np.zeros_like(target_img_bs)], axis=2))
        # combined_image = Image.fromarray((combined_image * 255).astype(np.uint8))
        if self.transform:
            combined_image = self.transform(combined_image)

        pose = self.get_pose_veriserum(idx)
        return {'image': combined_image, 'pose': pose}

    def get_calibrated_images(self, idx, target_img_bs, target_img_fs):
        df = self.veriserum_poses
        distortion_calibration_id = df[df['id'] == idx][f'distortion_calibration_id'].values[0]
        discal = DistortionCalibration()
        discal_dict = {'id': self.veriserum_discal['id'][distortion_calibration_id - 1],
                       'calibration_trial_id_bs': self.veriserum_discal['calibration_trial_id_bs'][
                           distortion_calibration_id - 1],
                       'calibration_trial_id_fs': self.veriserum_discal['calibration_trial_id_fs'][
                           distortion_calibration_id - 1],
                       'method': 'whatever',
                       'distortion_result_fs': self.veriserum_discal['distortion_result_fs'][
                           distortion_calibration_id - 1],
                       'distortion_result_bs': self.veriserum_discal['distortion_result_bs'][
                           distortion_calibration_id - 1]
                       }
        discal.load_from_dict(discal_dict)

        bs_corrected = discal.transform_bs(target_img_bs, outputResolution=1600, outputSize=360)
        fs_corrected = discal.transform_fs(target_img_fs, outputResolution=1600, outputSize=360)
        new_size = (512, 512)
        bs_corrected = cv2.resize(bs_corrected, new_size)
        fs_corrected = cv2.resize(fs_corrected, new_size)
        return bs_corrected, fs_corrected

    def get_img_name(self, idx):
        # Convert the ID number to a string, padding with zeros to ensure it is 6 digits
        id_str = str(idx).zfill(6)
        # Convert the string to bytes
        if self.calibration_on_time:
            return f'bs_{id_str}.jpg', f'fs_{id_str}.jpg'
        else:
            return f'calibrated_bs_{id_str}.jpg', f'calibrated_fs_{id_str}.jpg'

    def get_measurement_id(self, idx):
        df = self.veriserum_poses
        measurement_id = df[df['id'] == idx][f'measurement_id'].values[0]
        return measurement_id

    def get_anatomy_id(self, idx):
        df = self.veriserum_poses
        anatomy_id = df[df['id'] == idx][f'anatomy_id'].values[0]
        return anatomy_id

    def get_update_timestamp(self, idx):
        df = self.veriserum_poses
        update_timestamp = df[df['id'] == idx][f'update_timestamp'].values[0]
        return update_timestamp

    def get_obsolete(self, idx):
        df = self.veriserum_poses
        obsolete = df[df['id'] == idx][f'obsolete'].values[0]
        return obsolete

    def get_git_hash(self, idx):
        df = self.veriserum_poses
        git_hash = df[df['id'] == idx][f'git_hash'].values[0]
        return git_hash

    def get_git_repo(self, idx):
        df = self.veriserum_poses
        git_repo = df[df['id'] == idx][f'git_repo'].values[0]
        return git_repo

    def get_source_int_calibration_id(self, idx):
        df = self.veriserum_poses
        source_int_calibration_id = df[df['id'] == idx][f'source_int_calibration_id'].values[0]
        return source_int_calibration_id

    def get_distortion_calibration_id(self, idx):
        df = self.veriserum_poses
        distortion_calibration_id = df[df['id'] == idx][f'distortion_calibration_id'].values[0]
        return distortion_calibration_id

    def get_anatomy_path(self, idx):
        df = self.veriserum_poses
        anatomy_id = df[df['id'] == idx][f'anatomy_id'].values[0]
        id_str = str(anatomy_id).zfill(6)
        anatomy_stl = f'ana_{id_str}.stl'
        # anatomy_nii = f'ana_{id_str}_HU10000_filled.nii'

        return f'veriserum_anatomies/{anatomy_stl}'

    def get_source_intensifier_calibration(self, idx):
        # Convert the ID number to a string, padding with zeros to ensure it is 6 digits
        # load the calibration
        df = self.veriserum_poses
        source_int_calibration_id = df[df['id'] == idx][f'source_int_calibration_id'].values[0]
        cal_file_bs = {
            "cal_mm_per_pxl": 360 / 1600,
            "cal_principalp_x": self.veriserum_sical['principalp_h_bs'][source_int_calibration_id - 1],
            "cal_principalp_y": self.veriserum_sical['principalp_v_bs'][source_int_calibration_id - 1],
            "cal_focal_length": self.veriserum_sical['sid_bs'][source_int_calibration_id - 1],
        }
        cal_file_fs = {
            "cal_mm_per_pxl": 360 / 1600,
            "cal_principalp_x": self.veriserum_sical['principalp_h_fs'][source_int_calibration_id - 1],
            "cal_principalp_y": self.veriserum_sical['principalp_v_fs'][source_int_calibration_id - 1],
            "cal_focal_length": self.veriserum_sical['sid_fs'][source_int_calibration_id - 1],
        }
        return cal_file_bs, cal_file_fs

    def get_pose_veriserum(self, idx):
        # check if ID exists in DataFrame
        df = self.veriserum_poses
        anatomy = self.anatomy
        if idx in df['id'].values:
            # if exists，return the corresponding sourcepath
            # need to check if the euler angle in flumatch is ZXY

            pose_veriserum = np.array([
                df[df['id'] == idx][f'tx'].values[0],
                df[df['id'] == idx][f'ty'].values[0],
                df[df['id'] == idx][f'tz'].values[0],
                df[df['id'] == idx][f'rz'].values[0],
                df[df['id'] == idx][f'rx'].values[0],
                df[df['id'] == idx][f'ry'].values[0],
            ], dtype=float)
            # breakpoint()
            # calibration_settings = np.array([
            #     df[df['id'] == id][f'cal_focal_length'].values[0],
            #     df[df['id'] == id][f'cal_mm_per_pxl'].values[0],
            #     df[df['id'] == id][f'cal_principalp_x'].values[0],
            #     df[df['id'] == id][f'cal_principalp_y'].values[0],
            # ], dtype=np.float32)
            # Convert Euler angles to rotation matrix using ZXY convention
            # rotation_matrix = R.from_euler('ZXY', euler_angles, degrees=True).as_matrix()

            return torch.tensor(pose_veriserum.astype(np.float32), dtype=DTYPE_TORCH)
        else:
            # if not exists, raise an error
            raise ValueError(f"ID {idx} does not exist in the data.")
