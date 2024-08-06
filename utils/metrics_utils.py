import os.path

import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, mask, size_average)

def _ssim(img1, img2, window, window_size, channel, mask=None, size_average=True):
    if mask is not None:
        img1 = img1 * mask
        img2 = img2 * mask
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # if mask is None:
    #     mask = torch.ones_like(img1)[0]
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ==========================
# Depth Prediction Metrics
# Refernece
# - zioulis2018omnidepth
# ==========================
eps = 1e-7


def abs_rel_error(pred, gt, mask):
    """Compute absolute relative difference error"""
    return np.mean(np.abs(pred[mask > 0] - gt[mask > 0]) / np.maximum(np.abs(gt[mask > 0]),
                                                                      np.full_like(gt[mask > 0], eps)))


def abs_rel_error_map(pred, gt, mask):
    """ per pixels' absolute relative difference.

    Parameters @see delta_inlier_ratio_map
    :return: invalid pixel is NaN
    """
    are_map = np.zeros_like(pred)
    are_map[mask > 0] = np.abs(pred[mask > 0] - gt[mask > 0]) / gt[mask > 0]
    are_map[mask <= 0] = np.nan
    return are_map


def sq_rel_error(pred, gt, mask):
    """Compute squared relative difference error"""
    return np.mean((pred[mask > 0] - gt[mask > 0]) ** 2 / np.maximum(np.abs(gt[mask > 0]),
                                                                     np.full_like(gt[mask > 0], eps)))


def sq_rel_error_map(pred, gt, mask):
    """ squared relative difference error map.
    Parameters @see delta_inlier_ratio_map
    """
    are_map = np.zeros_like(pred)
    are_map[mask > 0] = (pred[mask > 0] - gt[mask > 0]) ** 2 / gt[mask > 0]
    are_map[mask <= 0] = np.nan
    return are_map


def mean_absolute_error(pred, gt, mask):
    """Mean absolute error"""
    return np.mean(np.abs(pred[mask > 0] - gt[mask > 0]))


def lin_rms_sq_error(pred, gt, mask):
    """Compute the linear RMS error except the final square-root step"""
    return np.mean((pred[mask > 0] - gt[mask > 0]) ** 2)


def lin_rms_sq_error_map(pred, gt, mask):
    """ Each pixel RMS.
    """
    lin_rms_map = np.zeros_like(pred)
    lin_rms_map[mask > 0] = (pred[mask > 0] - gt[mask > 0]) ** 2
    lin_rms_map[mask <= 0] = np.nan
    return lin_rms_map


def log_rms_sq_error(pred, gt, mask):
    """Compute the log RMS error except the final square-root step"""
    # if np.any(pred[mask] < 0) or np.any(gt[mask] < 0):
    #     log.error("The disparity map has negative value! The metric log will generate NaN")

    mask = (mask > 0) & (pred > eps) & (gt > eps)  # Compute a mask of valid values
    return np.mean((np.log10(pred[mask]) - np.log10(gt[mask])) ** 2)


def log_rms_sq_error_map(pred, gt, mask):
    """ Each pixel log RMS.
    Parameters @see delta_inlier_ratio_map
    """
    # if np.any(pred[mask] < 0) or np.any(gt[mask] < 0):
    #     log.error("The disparity map has negative value! The metric log will generate NaN")
    mask = (mask > 0) & (pred > eps) & (gt > eps)  # Compute a mask of valid values

    log_rms_map = np.zeros_like(pred)
    log_rms_map[mask > 0] = (np.log10(pred[mask > 0]) - np.log10(gt[mask > 0])) ** 2 / gt[mask > 0]
    log_rms_map[mask <= 0] = np.nan
    return log_rms_map


def log_rms_scale_invariant(pred, gt, mask):
    """ scale-invariant log RMSE.
    """
    # if np.any(pred[mask] < 0) or np.any(gt[mask] < 0):
        # log.error("The disparity map has negative value! The metric log will generate NaN")

    alpha_depth = np.mean(np.log(pred[mask > 0]) - np.log(gt[mask > 0]))
    log_rms_scale_inv = np.mean(np.log(pred[mask > 0]) - np.log(gt[mask > 0]) + alpha_depth)
    return log_rms_scale_inv


def log_rms_scale_invariant_map(pred, gt, mask):
    """ Each pixel scale invariant log RMS.
    Parameters @see delta_inlier_ratio_map
    """
    # if np.any(pred[mask] < 0) or np.any(gt[mask] < 0):
        # log.error("The disparity map has negative value! The metric log will generate NaN")

    log_rms_map = np.zeros_like(pred)
    alpha_depth = np.mean(np.log(pred[mask > 0]) - np.log(gt[mask > 0]))
    log_rms_map[mask > 0] = np.log(pred[mask > 0]) - np.log(gt[mask > 0]) + alpha_depth
    log_rms_map[mask <= 0] = np.nan
    return log_rms_map


def delta_inlier_ratio(pred, gt, mask, degree=1):
    """Compute the delta inlier rate to a specified degree (def: 1)"""
    return np.mean(np.maximum(pred[mask > 0] / gt[mask > 0], gt[mask > 0] / pred[mask > 0]) < (1.25 ** degree))


def delta_inlier_ratio_map(pred, gt, mask, degree=1):
    """ Get the δ < 1.25^degree map.

    Get the δ map, if pixels less than thr is 1, larger is 0, invalid is -1.

    :param pred: predict disparity map, [height, width]
    :type pred: numpy
    :param gt: ground truth disparity map, [height, width]
    :type gt: numpy
    :param mask: If the mask is greater than 0 the pixel is available, otherwise it's invalided.
    :type mask: numpy
    :param degree: The exponent of 1.24, defaults to 1
    :type degree: int, optional
    :return: The δ map, [height, width]
    :rtype: numpy
    """
    delta_max = np.maximum(pred[mask > 0] / gt[mask > 0], gt[mask > 0] / pred[mask > 0])

    delta_map = np.zeros_like(delta_max)
    delta_less = delta_max < (1.25 ** degree)
    delta_map[delta_less] = 1

    delta_larger = delta_max >= (1.25 ** degree)
    delta_map[delta_larger] = 0

    delta_map_all = np.zeros_like(pred)
    delta_map_all[mask > 0] = delta_map
    delta_map_all[mask <= 0] = -1
    return delta_map_all
