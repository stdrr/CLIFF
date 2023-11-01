import torch
import torch.nn as nn
import torch.nn.functional as F



def MAE(pred:torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
    """
    Compute the mean absolute error between two tensors.

    Args:
        pred (torch.Tensor): predicted tensor (mesh vertices, keypoints, volumes, etc.); shape (number_of_frames, )
        gt (torch.Tensor): ground truth tensor (mesh vertices, keypoints, volumes, etc.)

    Returns:
        torch.Tensor: mean absolute error between the two tensors
    """
    return F.l1_loss(pred, gt, reduction='mean')


def MSE(pred:torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error between two tensors.

    Args:
        pred (torch.Tensor): predicted tensor (mesh vertices, keypoints, volumes, etc.)
        gt (torch.Tensor): ground truth tensor (mesh vertices, keypoints, volumes, etc.)

    Returns:
        torch.Tensor: mean squared error between the two tensors
    """
    return F.mse_loss(pred, gt, reduction='mean')


def NAE(pred:torch.Tensor, gt:torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized absolute error between two tensors.

    Args:
        pred (torch.Tensor): predicted tensor (mesh vertices, keypoints, volumes, etc.)
        gt (torch.Tensor): ground truth tensor (mesh vertices, keypoints, volumes, etc.)

    Returns:
        torch.Tensor: normalized absolute error between the two tensors
    """
    return torch.mean(F.l1_loss(pred, gt, reduction='none') / gt)


def PPMAE(pred:torch.Tensor, gt_vols:torch.Tensor, gt_counts:torch.Tensor) -> torch.Tensor:
    """
    Compute the per-person mean absolute error between two tensors.

    Args:
        pred (torch.Tensor): predicted tensor (mesh vertices, keypoints, volumes, etc.)
        gt (torch.Tensor): ground truth tensor (mesh vertices, keypoints, volumes, etc.)

    Returns:
        torch.Tensor: per-person mean absolute error between the two tensors
    """
    return torch.mean(F.l1_loss(pred, gt_vols, reduction='none') / gt_counts)


