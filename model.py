import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation
from torchvision.utils import make_grid
from torchvision.models import resnext50_32x4d

from losses import GeodesicLossOrtho6d, gcc_loss, ncc_loss
from pytorch_msssim import SSIM
from dupla_renderers.pytorch3d import AnatomyCT, Camera, CTRenderer, STLRenderer, Scene
from pytorch3d.transforms import rotation_6d_to_matrix, euler_angles_to_matrix, matrix_to_rotation_6d

DTYPE_TORCH = torch.float32


class InterIntensifierCalibration:
    def __init__(self):
        pass

    def load_default_values(self):
        self.bs_pos = np.array((0.0, 0, 0))
        self.bs_normal_vec = np.array((0.0, 0, 1))
        self.bs_vertical_vec = np.array((0.0, 1, 0))
        self.fs_normal_vec = np.dot(
            Rotation.from_rotvec(np.deg2rad(70) * np.array([0, -1, 0])).as_matrix(),
            self.bs_normal_vec,
        )
        self.fs_pos = np.array((360.0, 0, 250))  # veriserum inter-plane setting
        self.fs_vertical_vec = np.array((0.0, 1, 0))

    def load_from_dict(self, d):
        self.bs_pos = d["bs_pos"]
        self.bs_normal_vec = d["bs_normal_vec"]
        self.bs_vertical_vec = d["bs_vertical_vec"]
        self.fs_pos = d["fs_pos"]
        self.fs_vertical_vec = d["fs_vertical_vec"]
        self.fs_normal_vec = d["fs_normal_vec"]


class PoseEstimationFineTuneModel(pl.LightningModule):
    def __init__(self, anatomy_path, learning_rate=1e-3, freeze_pretrained=True, loss_type='gcc', translation_range=10,
                 rotation_range=10, freeze=None):
        super(PoseEstimationFineTuneModel, self).__init__()
        self.learning_rate = learning_rate
        self.loss_type = loss_type

        # 姿态估计网络：ResNeXt50
        self.resnext = resnext50_32x4d(weights="DEFAULT")
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, 9)  # 输出为 9D 姿态参数

        # 损失函数
        self.translation_loss = nn.SmoothL1Loss()
        self.rotation_loss = GeodesicLossOrtho6d()

        self.translation_range = translation_range  # 平移噪声范围 (单位: mm)
        self.rotation_range = rotation_range  # 旋转噪声范围 (单位: 度)
        self.use_render_p = 1.0
        self.use_render_loss = True

        self.mse_loss = nn.MSELoss()
        # 可选：绑定解剖学和渲染器
        if anatomy_path is not None:
            self.anatomy_ct = AnatomyCT.load_data(anatomy_path)
            renderer, their_world_to_yours = self._initialize_renderer_and_scene()
            self.renderer = renderer
            self.their_world_to_yours = their_world_to_yours.to(self.device)
        if freeze is not None:
            self._freeze_partial_model(self.resnext, freeze_ratio=freeze)
            print(f'freeze layers ratio:{freeze}')

    def _freeze_partial_model(self, model, freeze_ratio):
        """freeze part of the model"""
        total_layers = len(list(model.parameters()))
        freeze_layers = int(total_layers * freeze_ratio)

        for i, param in enumerate(model.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
            else:
                break

    def forward(self, x):
        """
        前向传播：
        1. 使用预训练的 AutoEncoder 解码输入图像。
        2. 将解码后的图像作为输入传递到 ResNeXt。
        """

        pose = self.resnext(x)  # 姿态预测 [B, 9]
        return pose

    def add_noise_to_tmat(self, tmat):
        """
        对变换矩阵 (4x4) 添加噪声，包括随机平移和随机旋转。
        """
        # 添加平移噪声
        translation_noise = np.random.uniform(-self.translation_range, self.translation_range, 3)
        translation_noise = torch.tensor(translation_noise, dtype=torch.float32, device=tmat.device)
        # breakpoint()
        tmat[:, :3, 3] += translation_noise

        # 添加旋转噪声
        rotation_angle = np.random.uniform(0, np.deg2rad(self.rotation_range))
        rotation_axis = np.random.uniform(-1, 1, 3)
        rotation_axis /= np.linalg.norm(rotation_axis)  # 归一化轴
        rotation_matrix = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()

        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32, device=tmat.device)
        tmat[:, :3, :3] = torch.matmul(rotation_matrix, tmat[:, :3, :3])

        return tmat

    def _initialize_renderer_and_scene(self):
        Intensifiers = InterIntensifierCalibration()
        Intensifiers.load_default_values()

        cal_pixel_size = 0.225
        cal_principal_point_h_bs = 2.72
        cal_principal_point_v_bs = -6.12
        cal_focal_length_bs = 1860.54

        cal_principal_point_h_fs = -10.1351
        cal_principal_point_v_fs = 2.02703
        cal_focal_length_fs = 1851.35

        camera_1 = Camera(
            "camera_1",
            screen_center_poses=Intensifiers.bs_pos,
            screen_normals=Intensifiers.bs_normal_vec,
            screen_verticals=Intensifiers.bs_vertical_vec,
            screen_sizes_h=1600 * cal_pixel_size,
            screen_sizes_v=1600 * cal_pixel_size,
            principal_points_h=cal_principal_point_h_bs,
            principal_points_v=cal_principal_point_v_bs,
            focal_lengths=cal_focal_length_bs,
        )

        camera_2 = Camera(
            "camera_2",
            screen_center_poses=Intensifiers.fs_pos,
            screen_normals=Intensifiers.fs_normal_vec,
            screen_verticals=Intensifiers.fs_vertical_vec,
            screen_sizes_h=1600 * cal_pixel_size,
            screen_sizes_v=1600 * cal_pixel_size,
            principal_points_h=cal_principal_point_h_fs,
            principal_points_v=cal_principal_point_v_fs,
            focal_lengths=cal_focal_length_fs,
        )
        their_world_to_yours = torch.tensor(
            [
                [1, 0, 0, cal_principal_point_h_bs],
                [0, 1, 0, cal_principal_point_v_bs],
                [0, 0, 1, cal_focal_length_bs],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )[None]

        scene_sipla = Scene()
        scene_sipla.add_anatomies(self.anatomy_ct)
        scene_sipla.add_cameras(camera_1)
        scene_sipla.add_cameras(camera_2)

        renderer = CTRenderer(device="cuda")
        renderer.bind_scene(scene_sipla)

        return renderer, their_world_to_yours

    def compute_loss(self, images, decoded_image):
        """
        根据 loss_type 计算损失。
        Args:
            images (torch.Tensor): 输入图像 [B, C, H, W]
            decoded_image (torch.Tensor): 解码图像 [B, C, H, W]
        Returns:
            torch.Tensor: 总损失
        """

        batch_size = images.shape[0]

        if self.loss_type == "gcc":
            total_loss = 0.0
            for i in range(batch_size):
                image_curr = images[i, 0, :, :].unsqueeze(0)  # [1, H, W]
                decoded_curr = decoded_image[i, 0, :, :].unsqueeze(0)  # [1, H, W]
                loss = gcc_loss(image_curr, decoded_curr)
                total_loss += loss
            total_loss /= batch_size  # 对批次损失求平均
        elif self.loss_type == "mse":
            total_loss = 0.0
            total_loss = self.mse_loss(decoded_image, images)  # MSE 损失
        else:
            total_loss = 0.0
            for i in range(batch_size):
                image_curr = images[i, 0, :, :].unsqueeze(0)  # [1, H, W]
                decoded_curr = decoded_image[i, 0, :, :].unsqueeze(0)  # [1, H, W]
                loss = gcc_loss(image_curr, decoded_curr)
                total_loss += loss
            total_loss /= batch_size  # 对批次损失求平均
            total_loss = self.mse_loss(decoded_image, images) + total_loss
        return total_loss

    def calculate_gcc_loss(self):
        pass

    def pose_to_tmat(self, target_pose):
        device = target_pose.device

        # Extract translation vectors
        femur_translation = target_pose[:3].view(3).to(device)

        # Extract rotation in 6D
        femur_rotation_6d = target_pose[3:9].view(1, 6).to(device)

        # Convert 6D rotation representation to 3x3 rotation matrices
        femur_rotation_matrix = rotation_6d_to_matrix(femur_rotation_6d)

        # Create 4x4 transformation matrices using torch.stack to keep grad tracking
        tmat = torch.stack([
            torch.cat([femur_rotation_matrix[0][0], femur_translation[0].view(1)]),
            torch.cat([femur_rotation_matrix[0][1], femur_translation[1].view(1)]),
            torch.cat([femur_rotation_matrix[0][2], femur_translation[2].view(1)]),
            torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32)
        ])
        # Ensure matrices are of shape [1, 4, 4]
        tmat = tmat.unsqueeze(0)
        return tmat

    def tmat_to_pose(self, tmat):
        """
        将 4x4 变换矩阵转换为 9D pose。
        """
        rotation_matrix = tmat[:, :3, :3]
        translation = tmat[:, :3, 3]

        # 将 rotation_matrix 转为 6D 表示
        rotation_6d = matrix_to_rotation_6d(rotation_matrix)
        # breakpoint()
        # 拼接 translation 和 rotation_6d
        pose_9d = torch.cat([translation.squeeze(), rotation_6d.squeeze()], dim=0)

        return pose_9d

    def training_step(self, batch, batch_idx):
        images, target_pose = batch['image'], batch['pose']
        batch_size = images.size(0)
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs
        # 初始化存储
        inputs = []
        updated_target_pose = []
        # 0.5 概率选择 rendered image

        use_rendered = torch.rand(1).item() <= self.use_render_p
        if use_rendered:
            for i in range(batch_size):
                # 将 target_pose 转换为 4x4 变换矩阵
                tmat = self.pose_to_tmat(target_pose[i])

                # 添加噪声到 tmat
                noised_tmat = self.add_noise_to_tmat(tmat.clone())

                # 将噪声后的 tmat 渲染为虚拟 X-ray 图像
                rendered_image_bs, rendered_image_fs = self.generate_virtual_xray_with_grad(
                    tmat=noised_tmat,
                    img_resolution_width=images[i].shape[-1],
                    img_resolution_height=images[i].shape[-1],
                    binary=False
                )
                rendered_image_bs = 1 - rendered_image_bs[0]
                rendered_image_fs = 1 - rendered_image_fs[0]
                # breakpoint()
                # 拼接 rendered images
                rendered_image = torch.stack(
                    [rendered_image_bs, rendered_image_fs, torch.zeros_like(rendered_image_bs)], dim=0)

                # 将 noised_tmat 转回 9D pose
                noised_pose = self.tmat_to_pose(noised_tmat)

                # 更新输入和目标 pose
                inputs.append(rendered_image)
                updated_target_pose.append(noised_pose)

            # 合并 batch
            inputs = torch.stack(inputs, dim=0)
            updated_target_pose = torch.stack(updated_target_pose, dim=0)
        else:
            # 使用原始图像和 pose
            inputs = images.clone()  # 确保类型一致
            updated_target_pose = target_pose.clone()

        # 通过网络进行预测
        pred_pose = self(inputs)

        # 分离平移和旋转部分
        pred_translation, pred_rotation = pred_pose[:, :3], pred_pose[:, 3:]
        target_translation, target_rotation = updated_target_pose[:, :3], updated_target_pose[:, 3:]

        # 计算损失
        loss_translation = self.translation_loss(pred_translation, target_translation)
        loss_rotation = self.rotation_loss(pred_rotation, target_rotation)
        # loss_structural = self.compute_loss(images, decoded_image)  # Structural-Aware Loss
        if self.use_render_loss:
            gcc_loss_value = 0
            # add NCC loss
            for i in range(len(pred_pose)):
                pred_tmat = self.pose_to_tmat(pred_pose[i])

                # 生成虚拟X-ray图像
                rendered_image_bs, rendered_image_fs = self.generate_virtual_xray_with_grad(
                    tmat=pred_tmat,
                    img_resolution_width=images[i].shape[-1],
                    img_resolution_height=images[i].shape[-1], binary=False)
                rendered_image_bs = 1 - rendered_image_bs
                rendered_image_fs = 1 - rendered_image_fs

                # 将原始图像与生成图像进行NCC计算
                background_image_tensor = images[i].clone().detach().unsqueeze(dim=0).to(dtype=torch.float32)

                # gcc_loss_value += gcc_loss(rendered_image_bs, background_image_tensor[:, 0, :, :]) + gcc_loss(
                #     rendered_image_fs,
                #     background_image_tensor[
                #     :, 1, :, :])
                gcc_loss_value += ncc_loss(rendered_image_bs, background_image_tensor[:, 0, :, :]) + ncc_loss(
                    rendered_image_fs,
                    background_image_tensor[
                    :, 1, :, :])
                # 对所有样本的NCC损失求平均
            gcc_loss_value = gcc_loss_value / len(pred_pose)

            loss = loss_translation + loss_rotation + gcc_loss_value
        else:
            loss = loss_translation + loss_rotation
        # 记录损失
        self.log('train_loss', loss)
        self.log('train_translation_loss', loss_translation)
        self.log('train_rotation_loss', loss_rotation)
        if self.use_render_loss:
            self.log('train_rendered_loss', gcc_loss_value)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target_pose = batch['image'], batch['pose']
        batch_size = images.size(0)
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs
        # 初始化存储
        inputs = []
        updated_target_pose = []
        if self.use_render_p > 0.99:
            for i in range(batch_size):
                # 将 target_pose 转换为 4x4 变换矩阵
                tmat = self.pose_to_tmat(target_pose[i])

                # 添加噪声到 tmat
                noised_tmat = self.add_noise_to_tmat(tmat.clone())

                # 将噪声后的 tmat 渲染为虚拟 X-ray 图像
                rendered_image_bs, rendered_image_fs = self.generate_virtual_xray_with_grad(
                    tmat=noised_tmat,
                    img_resolution_width=images[i].shape[-1],
                    img_resolution_height=images[i].shape[-1],
                    binary=False
                )
                rendered_image_bs = 1 - rendered_image_bs[0]
                rendered_image_fs = 1 - rendered_image_fs[0]
                # breakpoint()
                # 拼接 rendered images
                rendered_image = torch.stack(
                    [rendered_image_bs, rendered_image_fs, torch.zeros_like(rendered_image_bs)], dim=0)

                # 将 noised_tmat 转回 9D pose
                noised_pose = self.tmat_to_pose(noised_tmat)

                # 更新输入和目标 pose
                inputs.append(rendered_image)
                updated_target_pose.append(noised_pose)

            # 合并 batch
            inputs = torch.stack(inputs, dim=0)
            updated_target_pose = torch.stack(updated_target_pose, dim=0)
        else:
            # 使用原始图像和 pose
            inputs = images.clone()  # 确保类型一致
            updated_target_pose = target_pose.clone()

        # 通过网络进行预测
        pred_pose = self(inputs)

        # 分离平移和旋转部分
        pred_translation, pred_rotation = pred_pose[:, :3], pred_pose[:, 3:]
        target_translation, target_rotation = updated_target_pose[:, :3], updated_target_pose[:, 3:]

        loss_translation = self.translation_loss(pred_translation, target_translation)
        loss_rotation = self.rotation_loss(pred_rotation, target_rotation)
        # loss_structural = self.compute_loss(images, decoded_image)  # Structural-Aware Loss
        gcc_loss_value = 0
        pred_synth_bs = []
        pred_synth_fs = []
        for i in range(len(pred_pose)):
            pred_tmat = self.pose_to_tmat(pred_pose[i])

            # 生成虚拟X-ray图像
            rendered_image_bs, rendered_image_fs = self.generate_virtual_xray_with_grad(
                tmat=pred_tmat,
                img_resolution_width=images[i].shape[-1],
                img_resolution_height=images[i].shape[-1], binary=False)
            rendered_image_bs = 1 - rendered_image_bs
            rendered_image_fs = 1 - rendered_image_fs

            # 将原始图像与生成图像进行NCC计算
            background_image_tensor = images[i].clone().detach().unsqueeze(dim=0).to(dtype=torch.float32)

            # gcc_loss_value += gcc_loss(rendered_image_bs, background_image_tensor[:, 0, :, :]) + gcc_loss(
            #     rendered_image_fs,
            #     background_image_tensor[
            #     :, 1, :, :])
            gcc_loss_value += ncc_loss(rendered_image_bs, background_image_tensor[:, 0, :, :]) + ncc_loss(
                rendered_image_fs,
                background_image_tensor[
                :, 1, :, :])
            pred_synth_bs.append(torch.tensor(rendered_image_bs).unsqueeze(0))  # (1, 1, H, W)
            pred_synth_fs.append(torch.tensor(rendered_image_fs).unsqueeze(0))  # (1, 1, H, W)
            # 对所有样本的NCC损失求平均
        pred_synth_bs = torch.cat(pred_synth_bs, dim=0)  # (B, 1, H, W)
        pred_synth_fs = torch.cat(pred_synth_fs, dim=0)  # (B, 1, H, W)
        gcc_loss_value = gcc_loss_value / len(pred_pose)
        if self.use_render_loss:
            val_loss = loss_translation + loss_rotation + gcc_loss_value
        else:
            val_loss = loss_translation + loss_rotation
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_translation_loss', loss_translation, prog_bar=False)
        self.log('val_rotation_loss', loss_rotation, prog_bar=False)
        if self.use_render_loss:
            self.log('val_rendered_loss', gcc_loss_value, prog_bar=False)
        # Log images (log only the first batch)
        if batch_idx == 0:
            # Make grids
            batch_size = inputs.shape[0]
            indices = torch.randperm(batch_size)[:4] if batch_size > 8 else torch.arange(batch_size)
            input_images_bs = inputs[indices, 0:1, :, :]
            input_images_fs = inputs[indices, 1:2, :, :]
            # decoded_images = decoded_image[indices, 0:1, :, :]
            # breakpoint()
            input_images_grid_bs = make_grid(input_images_bs, nrow=2, normalize=True, value_range=(0, 1))
            input_images_grid_fs = make_grid(input_images_fs, nrow=2, normalize=True, value_range=(0, 1))
            # decoded_images_grid = make_grid(decoded_images, nrow=2, normalize=True, value_range=(0, 1))

            pred_images_grid_bs = make_grid(pred_synth_bs, nrow=2, normalize=True, value_range=(0, 1))
            pred_images_grid_fs = make_grid(pred_synth_fs, nrow=2, normalize=True, value_range=(0, 1))
            # Add images to TensorBoard
            self.logger.experiment.add_image('val_input_images_bs', input_images_grid_bs, self.current_epoch,
                                             dataformats="CHW")
            self.logger.experiment.add_image('val_input_images_fs', input_images_grid_fs, self.current_epoch,
                                             dataformats="CHW")

            self.logger.experiment.add_image('val_pred_images_bs', pred_images_grid_bs, self.current_epoch,
                                             dataformats="CHW")
            self.logger.experiment.add_image('val_pred_images_fs', pred_images_grid_fs, self.current_epoch,
                                             dataformats="CHW")
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def generate_virtual_xray_with_grad(self, tmat, img_resolution_width, img_resolution_height,
                                        binary=False):
        self.set_model_matrix(self.anatomy_ct, tmat, self.their_world_to_yours)
        # breakpoint()
        try:
            foreground_efficient_renderer_1 = self.renderer.render_efficient_memory(
                0, img_resolution_width, img_resolution_height, binary=binary
            )[:, :, :]
            foreground_efficient_renderer_1.requires_grad_(True)
            foreground_efficient_renderer_2 = self.renderer.render_efficient_memory(
                1, img_resolution_width, img_resolution_height, binary=binary
            )[:, :, :]
            foreground_efficient_renderer_2.requires_grad_(True)
        except torch._C._LinAlgError as e:
            print(f"Matrix inversion failed: {e}")
            print(f"Camera transformation matrix: {self.cam.tmats}")
            breakpoint()
        return foreground_efficient_renderer_1, foreground_efficient_renderer_2

    def set_model_matrix(self, ct_data, tmat, their_world_to_yours):
        rotation_about_z = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=DTYPE_TORCH,
        )[None].to(tmat.device)
        ct_data.set_model_matrix(torch.matmul(tmat, rotation_about_z), is_yours=True,
                                 theirs_to_yours=their_world_to_yours)
