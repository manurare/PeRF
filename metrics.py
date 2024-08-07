import argparse
import glob
import os
from PIL import Image
import tqdm
import json

import numpy as np
import torch

from utils.metrics_utils import psnr, ssim, abs_rel_error, lin_rms_sq_error, delta_inlier_ratio
from utils.utils import read_dpt

parser = argparse.ArgumentParser()
parser.add_argument("--exp_renders_dir", required=True, default=None)
parser.add_argument("--gt_renders_dir", required=True, default=None)
args = parser.parse_args()

import lpips
lpips_alex = lpips.LPIPS(net='alex') # best forward scores
lpips_alex.to("cuda")

files = glob.glob(os.path.join(args.exp_renders_dir, "image_*.png"))
numfiles = len(files)

pred_rgb_files_format = os.path.join(args.exp_renders_dir, "image_{}.png")
pred_depth_files_format = os.path.join(args.exp_renders_dir, "distance_{}.npy")
gt_rgb_files_format = os.path.join(args.gt_renders_dir, "{:05}_rgb.png")
gt_depth_files_format = os.path.join(args.gt_renders_dir, "{:05}_depth.dpt")

metrics = {"psnr": [], "ssim": [], "lpips": [], "absrel": [], "rms": [], "delta1": [], "delta2": [], "delta3": []}

for i in tqdm.tqdm(range(numfiles)):
    with torch.no_grad():
        pred_rgb = np.array(Image.open(pred_rgb_files_format.format(i))) / 255.
        gt_rgb = np.array(Image.open(gt_rgb_files_format.format(i))) / 255.
        pred_rgb = torch.from_numpy(pred_rgb).permute(2, 0, 1).type(torch.float32).contiguous().to("cuda")[None]
        gt_rgb = torch.from_numpy(gt_rgb).permute(2, 0, 1).type(torch.float32).contiguous().to("cuda")[None]

        psnr_ = psnr(pred_rgb, gt_rgb).cpu().item()
        ssim_ = ssim(pred_rgb, gt_rgb).cpu().item()
        lpips_ = lpips_alex(pred_rgb, gt_rgb).cpu().item()

    # Depth
    pred_depth = np.load(pred_depth_files_format.format(i)).squeeze()
    gt_depth = read_dpt(gt_depth_files_format.format(i)).squeeze()

    mask = gt_depth > 0

    abs_rel = abs_rel_error(pred_depth, gt_depth, mask)
    rms = lin_rms_sq_error(pred_depth, gt_depth, mask)
    delta1 = delta_inlier_ratio(pred_depth, gt_depth, mask, 1)
    delta2 = delta_inlier_ratio(pred_depth, gt_depth, mask, 2)
    delta3 = delta_inlier_ratio(pred_depth, gt_depth, mask, 3)

    metrics["psnr"].append(psnr_)
    metrics["ssim"].append(ssim_)
    metrics["lpips"].append(lpips_)
    metrics["absrel"].append(abs_rel)
    metrics["rms"].append(rms)
    metrics["delta1"].append(delta1)
    metrics["delta2"].append(delta2)
    metrics["delta3"].append(delta3)

means = {k:np.mean(v) for k,v in metrics.items()}
print(",".join(["exp_name"] + [*means.keys()]))
print(",".join(map(str, [*means.values()])))

results_file = os.path.join(args.exp_renders_dir, os.pardir, "results.json")
with open(results_file, 'w') as fp:
    json.dump(means, fp, indent=True)