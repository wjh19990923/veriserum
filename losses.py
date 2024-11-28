import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.modules.module import Module

def rotation_6d_to_matrix(rotation_6d):
    """
    Converts a 6D rotation representation to a 3x3 rotation matrix.
    Args:
        rotation_6d (torch.Tensor): Tensor of shape (N, 6), where each row represents
            the first two columns of a 3x3 rotation matrix in 6D representation.
    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3), where each slice along the batch dimension
            is a valid rotation matrix.
    """
    # Split the 6D input into two vectors
    x_raw = rotation_6d[:, 0:3]
    y_raw = rotation_6d[:, 3:6]

    # Normalize the first vector
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)

    # Make the second vector orthogonal to the first
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    # Ensure orthogonality by recomputing the second vector
    y = torch.cross(z, x, dim=-1)

    # Stack the vectors to form the rotation matrix
    rotation_matrix = torch.stack((x, y, z), dim=-1)  # Shape: (N, 3, 3)
    return rotation_matrix


class GeodesicLossOrtho6d(nn.Module):
    r"""Creates a criterion that measures the distance between rotation matrices, which is
    useful for pose estimation problems.
    The distance ranges from 0 to :math:`pi`.
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices and:
    "Metrics for 3D Rotations: Comparison and Analysis" (https://link.springer.com/article/10.1007/s10851-009-0161-2).
    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.
    The loss can be described as:
    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)
    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``
        radius (bool, optional): Whether to output the result in radians (default: True).
            If False, the result will be in degrees.
    Shape:
        - Input: Shape :math:`(N, 6)`.
        - Target: Shape :math:`(N, 6)`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`. Otherwise, scalar.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean", radius=False) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.radius = radius

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_m = rotation_6d_to_matrix(pred)
        target_m = rotation_6d_to_matrix(target)
        R_diffs = pred_m @ target_m.permute(0, 2, 1)
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))

        if not self.radius:
            dists = torch.rad2deg(dists)  # Convert to degrees

        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()


class PointMatchingLoss(nn.Module):
    def __init__(self, num_points=1024):
        super(PointMatchingLoss, self).__init__()
        self.num_points = num_points
        self.points = self.generate_random_points(self.num_points)

    def generate_random_points(self, num_points):
        """
        Generates random unit vectors on the surface of a sphere (radius=1).
        :param num_points: Number of points to generate.
        :return: A tensor of shape (num_points, 3) representing the points.
        """
        # Generate random points in spherical coordinates
        phi = torch.rand(num_points) * 2 * np.pi  # Uniform distribution [0, 2*pi]
        costheta = torch.rand(num_points) * 2 - 1  # Uniform distribution [-1, 1]
        u = torch.rand(num_points)  # Uniform distribution [0, 1]

        # Convert spherical coordinates to Cartesian coordinates
        theta = torch.acos(costheta)
        r = u ** (1/3)

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        points = torch.stack((x, y, z), dim=-1)  # Shape: (num_points, 3)
        return points

    def forward(self, pred_transform, gt_transform):
        """
        Computes the point matching loss between predicted and ground truth transformations.
        :param pred_transform: Predicted transformation, shape (B, 12) for 3 translation + 6D rotation.
        :param gt_transform: Ground truth transformation, shape (B, 12) for 3 translation + 6D rotation.
        :return: Scalar loss.
        """
        # Unpack the transformations
        pred_translation = pred_transform[:, :3]
        pred_rotation_6d = pred_transform[:, 3:9]
        gt_translation = gt_transform[:, :3]
        gt_rotation_6d = gt_transform[:, 3:9]

        # Convert 6D rotation representations to 3x3 matrices
        pred_rotation_matrix = rotation_6d_to_matrix(pred_rotation_6d)
        gt_rotation_matrix = rotation_6d_to_matrix(gt_rotation_6d)

        # Apply the transformations to the points
        pred_transformed_points = torch.einsum('bij,jk->bik', pred_rotation_matrix, self.points.T) + pred_translation.unsqueeze(-1)
        gt_transformed_points = torch.einsum('bij,jk->bik', gt_rotation_matrix, self.points.T) + gt_translation.unsqueeze(-1)

        # Compute the point-wise Euclidean distance between the transformed points
        loss = torch.mean(torch.norm(pred_transformed_points - gt_transformed_points, dim=1))

        return loss



import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch_similarity.modules import GradientCorrelationLoss2d, NormalizedCrossCorrelation

def gcc_loss(image_curr_bs: torch.Tensor, target_image_curr_bs: torch.Tensor, mask=None, dilation_kernel_size=5) -> torch.Tensor:
    # Gradient Correlation Loss implementation
    # breakpoint()
    image1 = torch.reshape(image_curr_bs, (1, 1, image_curr_bs.shape[1], image_curr_bs.shape[2]))
    image_ref1 = torch.reshape(target_image_curr_bs, (1, 1, target_image_curr_bs.shape[1], target_image_curr_bs.shape[2]))

    if mask is None:
        loss_function = GradientCorrelationLoss2d(grad_method="sobel", return_map=False).cuda()
        return loss_function(image1, image_ref1)
    else:
        loss_function = GradientCorrelationLoss2d(grad_method="sobel", return_map=True).cuda()
        gcc, gcc_map = loss_function(image1, image_ref1)

        # Resize the mask if sizes don't match
        if mask.shape[-2:] != gcc_map.shape[-2:]:
            # breakpoint()
            mask = F.interpolate(mask.unsqueeze(1), size=gcc_map.shape[-2:]).squeeze(1)

        # Define dilation kernel
        dilation_kernel = torch.ones(1, 1, dilation_kernel_size, dilation_kernel_size, device=mask.device)

        # Dilate the mask using 2D convolution
        mask = F.conv2d(mask.unsqueeze(0), dilation_kernel, padding=dilation_kernel_size//2).squeeze(0)

        # Binarize the mask (e.g., set threshold at 0.5, or adjust based on your needs)
        mask = torch.where(mask > 0.5, torch.tensor(1.0, device=mask.device), torch.tensor(0.0, device=mask.device))

        # Apply the dilated mask to the gcc_map
        gcc_weighted = (gcc_map * mask).sum()
        return 1 - gcc_weighted


import torch

def ncc(x, y, return_map=False, reduction='mean', eps=1e-8):
    assert x.requires_grad or y.requires_grad, "At least one input must require gradients."
    """
    计算 N 维的标准化互相关 (NCC)。

    Args:
        x (torch.Tensor): 输入张量1 (BHW)。
        y (torch.Tensor): 输入张量2 (BHW)。
        return_map (bool): 是否返回 NCC map。
        reduction (str): 指定输出的操作 ('mean' 或 'sum')。
        eps (float): 防止数值不稳定的 epsilon 值。

    Returns:
        torch.Tensor: NCC 相似度值。
        torch.Tensor: NCC map (如果 return_map 为 True)。
    """
    # breakpoint()
    # 获取形状信息
    shape = x.shape
    b = shape[0]

    # 重塑为 (B, N)
    x = x.view(b, -1)
    y = y.view(b, -1)

    # 计算均值
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # 去除均值
    x = x - x_mean
    y = y - y_mean

    # 逐元素相乘和平方操作
    dev_xy = torch.mul(x, y)
    dev_xx = torch.mul(x, x)
    dev_yy = torch.mul(y, y)

    # 分别计算 dev_xx 和 dev_yy 的和
    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    # 计算 NCC
    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt(torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])
    # print(f"NCC requires_grad: {ncc.requires_grad}, grad_fn: {ncc.grad_fn}")
    # 根据 reduction 参数进行归约
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError(f"不支持的 reduction 类型: {reduction}")

    # 返回结果
    if not return_map:
        return ncc

    return ncc, ncc_map