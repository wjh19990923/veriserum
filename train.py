# train.py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from model import PoseEstimationModel  # 导入模型
from veriserum_dataset import Veriserum_calibrated  # 假设数据集在 veriserum_dataset.py 文件中
from sampler import NonObsoleteSampler  # 假设采样器在 sampler.py 文件中
import torchvision.transforms as transforms

if __name__ == '__main__':
    # 定义 transform
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    dataset = Veriserum_calibrated(transform=train_transform)
    print('Dataset loaded')

    # 筛选有效样本并拆分为训练集和验证集
    sampler = NonObsoleteSampler(dataset)
    valid_indices = sampler.valid_indices
    train_size = int(0.8 * len(valid_indices))
    train_indices, val_indices = valid_indices[:train_size], valid_indices[train_size:]

    # 使用数据加载器
    train_loader = DataLoader(
        Veriserum_calibrated(transform=train_transform),
        batch_size=4,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=1
    )
    val_loader = DataLoader(
        Veriserum_calibrated(transform=val_transform),
        batch_size=4,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=1
    )
    # 创建模型
    model = PoseEstimationModel(learning_rate=1e-3)

    # 定义模型保存的回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    # 使用 PyTorch Lightning 的 Trainer 进行训练
    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        # num_sanity_val_steps=0  # 禁用 Sanity Check，避免验证集加载问题
    )

    # 训练模型
    trainer.fit(model, train_loader, val_loader)
