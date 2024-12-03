# train.py
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from model import PoseEstimationFineTuneModel, EncoderDecoderSelfSupervised  # 导入模型
from veriserum_dataset import Veriserum_calibrated  # 假设数据集在 veriserum_dataset.py 文件中
from sampler import NonObsoleteSampler  # 假设采样器在 sampler.py 文件中
import torchvision.transforms as transforms
from pytorch_lightning.loggers import TensorBoardLogger


def main(loss_type='combine'):
    # set random seed
    seed = 923
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    sampler = NonObsoleteSampler(dataset, length=1000, anatomy_id=1)
    valid_indices = sampler.valid_indices
    print(len(valid_indices))
    train_size = int(0.8 * len(valid_indices))
    train_indices, val_indices = valid_indices[:train_size], valid_indices[train_size:]

    # 使用数据加载器
    train_loader = DataLoader(
        Veriserum_calibrated(transform=train_transform),
        batch_size=4,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=1,
        drop_last=True
    )
    val_loader = DataLoader(
        Veriserum_calibrated(transform=val_transform),
        batch_size=4,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=1,
        drop_last=True
    )
    # 创建模型

    # pretrain_model = EncoderDecoderSelfSupervised(learning_rate=1e-3, loss_type=loss_type)

    # 初始化新模型
    pretrained_path = rf'C:\Users\Public\Public Dupla\veriserum\checkpoints_pretrained\pretrain_gcc_epoch=199-val_loss=0.3543.ckpt'

    model = PoseEstimationFineTuneModel(autoencoder_checkpoint=pretrained_path, learning_rate=1e-3, loss_type='mse',
                                           anatomy_path=rf'D:\veriserum_anatomies\ana_000001_HU10000_filled.nii')
    # 加载预训练的 Encoder 和 Decoder 权重

    # 定义模型保存的回调
    logger_pretrain = TensorBoardLogger("pretrain_logs", name=f"pretrain_{loss_type}_loss")
    checkpoint_callback_pretrain = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints_pretrained',
        filename=f'pretrain_{loss_type}_' + '{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )
    logger_actual = TensorBoardLogger("estimation_logs2")
    checkpoint_callback_actual = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints_actual',
        filename=f'estimation' + '{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )
    # 使用 PyTorch Lightning 的 Trainer 进行训练
    trainer = pl.Trainer(
        max_epochs=250,
        logger=logger_actual,
        callbacks=[checkpoint_callback_actual],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=1,
        check_val_every_n_epoch=5,
        # num_sanity_val_steps=0  # 禁用 Sanity Check，避免验证集加载问题
    )

    # 训练模型
    # trainer.fit(pretrain_model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    # main(loss_type='mse')
    # main(loss_type='gcc')
    main(loss_type='combine')
    # main(loss_type='ssim')
