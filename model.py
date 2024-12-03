import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torchvision.models import resnext50_32x4d

from losses import GeodesicLossOrtho6d, gcc_loss, ncc
from pytorch_msssim import SSIM
from dupla_renderers.pytorch3d import AnatomyCT, Camera, CTRenderer, STLRenderer, Scene
from pytorch3d.transforms import rotation_6d_to_matrix, euler_angles_to_matrix, matrix_to_rotation_6d

DTYPE_TORCH = torch.float32


class PoseEstimationFineTuneModel(pl.LightningModule):
    def __init__(self, anatomy_path,autoencoder_checkpoint, learning_rate=1e-3, freeze_pretrained=True, loss_type='gcc'):
        super(PoseEstimationFineTuneModel, self).__init__()
        self.learning_rate = learning_rate
        self.loss_type = loss_type

        # 加载预训练的 AutoEncoder 模型
        self.autoencoder = EncoderDecoderSelfSupervised()
        checkpoint = torch.load(autoencoder_checkpoint)
        self.autoencoder.load_state_dict(checkpoint['state_dict'])

        # 冻结 AutoEncoder 权重（可选）
        if freeze_pretrained:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        # 姿态估计网络：ResNeXt50
        self.resnext = resnext50_32x4d(weights="DEFAULT")
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, 9)  # 输出为 9D 姿态参数

        # 损失函数
        self.translation_loss = nn.SmoothL1Loss()
        self.rotation_loss = GeodesicLossOrtho6d()
        self.mse_loss = nn.MSELoss()
        # 可选：绑定解剖学和渲染器
        if anatomy_path is not None:
            self.anatomy_ct = AnatomyCT.load_data(anatomy_path)
            renderer, cam, their_world_to_yours = self._initialize_renderer_and_scene()
            self.renderer = renderer
            self.cam = cam
            self.their_world_to_yours = their_world_to_yours.to(self.device)
    def forward(self, x):
        """
        前向传播：
        1. 使用预训练的 AutoEncoder 解码输入图像。
        2. 将解码后的图像作为输入传递到 ResNeXt。
        """
        with torch.no_grad():  # 确保 autoencoder 不更新
            decoded_image = self.autoencoder(x)  # 解码图像 [B, C, H, W]

        pose = self.resnext(decoded_image)  # 姿态预测 [B, 9]
        return pose, decoded_image

    def _initialize_renderer_and_scene(self):
        cal_pixel_size = 0.225
        cal_principal_point_h = 2.72
        cal_principal_point_v = -6.12
        cal_focal_length = 1860.54

        cam = Camera(
            "camera_1",
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            1600 * cal_pixel_size,
            1600 * cal_pixel_size,
            cal_principal_point_h,
            cal_principal_point_v,
            cal_focal_length,
        )

        their_world_to_yours = torch.tensor(
            [
                [1, 0, 0, cal_principal_point_h],
                [0, 1, 0, cal_principal_point_v],
                [0, 0, 1, cal_focal_length],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )[None]

        scene_sipla = Scene()
        scene_sipla.add_anatomies(self.anatomy_ct)
        scene_sipla.add_cameras(cam)

        renderer = CTRenderer(device="cuda")
        renderer.bind_scene(scene_sipla)

        return renderer, cam, their_world_to_yours

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

    def training_step(self, batch, batch_idx):
        images, target_pose = batch['image'], batch['pose']
        pred_pose, decoded_image = self(images)

        # 分割 9D 表示为平移和旋转
        pred_translation, pred_rotation = pred_pose[:, :3], pred_pose[:, 3:]
        target_translation, target_rotation = target_pose[:, :3], target_pose[:, 3:]

        # 计算损失
        loss_translation = self.translation_loss(pred_translation, target_translation)
        loss_rotation = self.rotation_loss(pred_rotation, target_rotation)
        loss_structural = self.compute_loss(images, decoded_image)  # Structural-Aware Loss

        gcc_loss_value = 0
        # add NCC loss
        for i in range(len(pred_pose)):
            pred_tmat = self.pose_to_tmat(pred_pose[i])

            # 生成虚拟X-ray图像
            rendered_image = 1 - self.generate_virtual_xray_with_grad(
                tmat=pred_tmat,
                img_resolution_width=images[i].shape[-1],
                img_resolution_height=images[i].shape[-1], binary=False)

            # 将原始图像与生成图像进行NCC计算
            background_image_tensor = images[i].clone().detach().unsqueeze(dim=0).to(dtype=torch.float32)

            gcc_loss_value += gcc_loss(rendered_image, background_image_tensor[:, 0, :, :])

            # 对所有样本的NCC损失求平均
        gcc_loss_value = gcc_loss_value / len(pred_pose)

        loss = loss_translation + loss_rotation + gcc_loss_value
        # 记录损失
        self.log('train_loss', loss)
        self.log('train_translation_loss', loss_translation)
        self.log('train_rotation_loss', loss_rotation)
        self.log('train_structural_loss', loss_structural)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target_pose = batch['image'], batch['pose']
        pred_pose, decoded_image = self(images)

        pred_translation, pred_rotation = pred_pose[:, :3], pred_pose[:, 3:]
        target_translation, target_rotation = target_pose[:, :3], target_pose[:, 3:]

        loss_translation = self.translation_loss(pred_translation, target_translation)
        loss_rotation = self.rotation_loss(pred_rotation, target_rotation)
        loss_structural = self.compute_loss(images, decoded_image)  # Structural-Aware Loss
        gcc_loss_value = 0
        pred_synth = []
        for i in range(len(pred_pose)):
            pred_tmat = self.pose_to_tmat(pred_pose[i])

            # 生成虚拟X-ray图像
            rendered_image = 1 - self.generate_virtual_xray_with_grad(
                tmat=pred_tmat,
                img_resolution_width=images[i].shape[-1],
                img_resolution_height=images[i].shape[-1], binary=False)

            # 将原始图像与生成图像进行NCC计算
            background_image_tensor = images[i].clone().detach().unsqueeze(dim=0).to(dtype=torch.float32)

            gcc_loss_value += gcc_loss(rendered_image, background_image_tensor[:, 0, :, :])
            pred_synth.append(torch.tensor(rendered_image).unsqueeze(0))  # (1, 1, H, W)
            # 对所有样本的NCC损失求平均
        pred_synth = torch.cat(pred_synth, dim=0)  # (B, 1, H, W)
        gcc_loss_value = gcc_loss_value / len(pred_pose)

        val_loss = loss_translation + loss_rotation + gcc_loss_value

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_translation_loss', loss_translation, prog_bar=True)
        self.log('val_rotation_loss', loss_rotation, prog_bar=True)
        self.log('val_structural_loss', loss_structural, prog_bar=True)
        # Log images (log only the first batch)
        if batch_idx == 0:
            # Make grids
            batch_size = images.shape[0]
            indices = torch.randperm(batch_size)[:4] if batch_size > 8 else torch.arange(batch_size)
            input_images = images[indices, 0:1, :, :]
            decoded_images = decoded_image[indices, 0:1, :, :]
            # breakpoint()
            input_images_grid = make_grid(input_images, nrow=2, normalize=True, value_range=(0, 1))
            decoded_images_grid = make_grid(decoded_images, nrow=2, normalize=True, value_range=(0, 1))

            pred_images_grid = make_grid(pred_synth, nrow=2, normalize=True, value_range=(0, 1))

            # Add images to TensorBoard
            self.logger.experiment.add_image('val_input_images', input_images_grid, self.current_epoch,
                                             dataformats="CHW")
            self.logger.experiment.add_image('val_decoded_images', decoded_images_grid, self.current_epoch,
                                             dataformats="CHW")
            self.logger.experiment.add_image('val_pred_images', pred_images_grid, self.current_epoch, dataformats="CHW")

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def generate_virtual_xray_with_grad(self, tmat, img_resolution_width, img_resolution_height,
                                        binary=False):
        self.set_model_matrix(self.anatomy_ct, tmat, self.their_world_to_yours)
        # breakpoint()
        try:
            foreground_efficient_renderer = self.renderer.render_efficient_memory(
                0, img_resolution_width, img_resolution_height, binary=binary
            )[:, :, :]
            foreground_efficient_renderer.requires_grad_(True)
        except torch._C._LinAlgError as e:
            print(f"Matrix inversion failed: {e}")
            print(f"Camera transformation matrix: {self.cam.tmats}")
            breakpoint()
        return foreground_efficient_renderer

    def set_model_matrix(self, ct_data, tmat, their_world_to_yours):
        rotation_about_z = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=DTYPE_TORCH,
        )[None].to(tmat.device)
        ct_data.set_model_matrix(torch.matmul(tmat, rotation_about_z), is_yours=True,
                                 theirs_to_yours=their_world_to_yours)


import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms.functional import gaussian_blur
from pytorch_msssim import SSIM
from losses import gcc_loss
import torch.nn.functional as F


class EncoderDecoderSelfSupervised(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, loss_type="gcc", noise_std=0.0):
        """
        使用完整的 DeepLabv3 模型作为 AutoEncoder。
        Args:
            learning_rate (float): 学习率。
            loss_type (str): 损失函数类型 ("gcc", "mse", "ssim")。
            noise_std (float): 添加到输入图像的高斯噪声标准差。
        """
        super(EncoderDecoderSelfSupervised, self).__init__()
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.noise_std = noise_std

        # 加载预训练的 DeepLabv3 模型
        self.deeplab = deeplabv3_resnet50(weights="DEFAULT")

        # 修改 classifier 的最后一层以适应 AutoEncoder 的输出
        self.deeplab.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)
        self.deeplab.aux_classifier = None  # 移除辅助分类器
        self.sigmoid = nn.Sigmoid()  # 将输出限制在 [0, 1]

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)

    def add_noise(self, images):
        """
        添加高斯噪声到输入图像。
        Args:
            images (torch.Tensor): 输入图像，形状为 [B, C, H, W]。
        Returns:
            torch.Tensor: 加入噪声的图像。
        """
        noise = torch.randn_like(images) * self.noise_std
        return images + noise

    def forward(self, x):
        """
        前向传播，直接使用 DeepLabv3 的结构。
        Args:
            x (torch.Tensor): 输入图像 [B, C, H, W]。
        Returns:
            torch.Tensor: 解码后的图像 [B, C, H, W]。
        """
        output = self.deeplab(x)["out"]  # DeepLabv3 的输出
        if output.size(-1) == 1 and output.size(-2) == 1:
            output = F.adaptive_avg_pool2d(output, (1, 1))  # 全局平均池化
        return self.sigmoid(output)  # 限制输出范围在 [0, 1]

    def compute_loss(self, images, decoded_image):
        """
        根据 loss_type 计算损失。
        Args:
            images (torch.Tensor): Ground truth 图像 [B, C, H, W]。
            decoded_image (torch.Tensor): 解码图像 [B, C, H, W]。
        Returns:
            torch.Tensor: 总损失。
        """
        if self.loss_type == "gcc":
            total_loss_gcc = 0.0
            for i in range(images.size(0)):
                loss = gcc_loss(images[i, 0:1, :, :], decoded_image[i, 0:1, :, :]) + gcc_loss(images[i, 1:2, :, :],
                                                                                              decoded_image[i, 1:2, :,
                                                                                              :])
                total_loss_gcc += loss
            total_loss_gcc /= images.size(0)
            total_loss = total_loss_gcc
        elif self.loss_type == "mse":
            total_loss = self.mse_loss(decoded_image, images)
        elif self.loss_type == "ssim":
            total_loss = 1 - self.ssim_loss(decoded_image, images)  # 最大化结构相似性
        elif self.loss_type == "combine":
            total_loss_mse = self.mse_loss(decoded_image, images)
            total_loss_gcc = 0.0
            for i in range(images.size(0)):
                # breakpoint()
                loss = gcc_loss(images[i, 0:1, :, :], decoded_image[i, 0:1, :, :]) + gcc_loss(images[i, 1:2, :, :],
                                                                                              decoded_image[i, 1:2, :,
                                                                                              :])
                total_loss_gcc += loss
            total_loss_gcc /= images.size(0)
            total_loss = total_loss_gcc + total_loss_mse
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return total_loss

    def training_step(self, batch, batch_idx):
        images = batch['image']  # Ground truth 图像 [B, C, H, W]
        noisy_images = self.add_noise(images)  # 添加噪声
        decoded_image = self(noisy_images)  # 解码噪声图像
        total_loss = self.compute_loss(images, decoded_image)  # 计算损失
        self.log('train_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']  # Ground truth 图像 [B, C, H, W]
        noisy_images = self.add_noise(images)  # 添加噪声
        decoded_image = self(noisy_images)  # 解码噪声图像
        total_loss = self.compute_loss(images, decoded_image)  # 计算损失

        if batch_idx == 0:
            self.log_images(noisy_images[0], decoded_image[0])

        self.log('val_loss', total_loss, prog_bar=True)
        return total_loss

    def log_images(self, input_image, output_image):
        """
        在 TensorBoard 中记录输入和输出图像。
        Args:
            input_image (torch.Tensor): 输入图像 [C, H, W]。
            output_image (torch.Tensor): 输出图像 [C, H, W]。
        """
        input_image_np = input_image.permute(1, 2, 0).detach().cpu().numpy()
        output_image_np = output_image.permute(1, 2, 0).detach().cpu().numpy()

        self.logger.experiment.add_image(
            "Input Image bs", input_image_np[:, :, 0:1], self.current_epoch, dataformats="HWC"
        )
        self.logger.experiment.add_image(
            "Decoded Image bs", output_image_np[:, :, 0:1], self.current_epoch, dataformats="HWC"
        )
        self.logger.experiment.add_image(
            "Input Image fs", input_image_np[:, :, 1:2], self.current_epoch, dataformats="HWC"
        )
        self.logger.experiment.add_image(
            "Decoded Image fs", output_image_np[:, :, 1:2], self.current_epoch, dataformats="HWC"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
