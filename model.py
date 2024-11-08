import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MeanSquaredError


class PoseEstimationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(PoseEstimationModel, self).__init__()
        self.learning_rate = learning_rate

        # 使用预训练的 ResNeXt50 模型
        # model.py 中的 ResNeXt 模型定义
        self.feature_extractor = models.resnext50_32x4d(weights="ResNeXt50_32X4D_Weights.DEFAULT")

        num_features = self.feature_extractor.fc.in_features

        # 替换全连接层，以适应姿态估计输出
        self.feature_extractor.fc = nn.Linear(num_features, 256)
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 6)  # 输出6个姿态参数 (3 translations, 3 rotations)
        )

        # 损失函数
        self.criterion = nn.MSELoss()
        self.metric = MeanSquaredError()

    def forward(self, x):
        # 输入特征提取器并通过回归头
        features = self.feature_extractor(x)
        pose = self.regressor(features)
        return pose

    def training_step(self, batch, batch_idx):
        images, target_pose = batch['image'], batch['pose']
        pred_pose = self(images)  # 输入已是 [batch_size, 3, H, W] 形状
        loss = self.criterion(pred_pose, target_pose)

        # 记录训练损失和指标
        self.log('train_loss', loss)
        self.log('train_mse', self.metric(pred_pose, target_pose))
        return loss

    def validation_step(self, batch, batch_idx):
        images, target_pose = batch['image'], batch['pose']
        pred_pose = self(images)
        val_loss = self.criterion(pred_pose, target_pose)

        # 记录验证损失和指标
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_mse', self.metric(pred_pose, target_pose), prog_bar=True)
        return val_loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
