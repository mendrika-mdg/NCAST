import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter, maximum_filter
import scipy.ndimage as nd

lead_time = sys.argv[1]

base_dir = (
    f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/"
    f"ensemble/t{lead_time}"
)

output_dir = "/home/users/mendrika/NCAST/Output/evaluation/ncast/fss/ensemble"
os.makedirs(output_dir, exist_ok=True)

SEEDS = [10, 20, 30, 40, 50]

PIXEL_SIZE_KM = 3
GT_FILTER_SIZE = 25
MIN_CORES = None
windows = [3, 9, 25, 49, 81, 121]

case_type = (
    "all_scenes"
    if MIN_CORES is None
    else f"min_{MIN_CORES}_cores"
)

def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

def compute_fss(pred, obs, window):
    pred = np.clip(pred, 0, 1)
    obs = np.clip(obs, 0, 1)

    f_pred = uniform_filter(pred, size=window, mode="constant")
    f_obs = uniform_filter(obs, size=window, mode="constant")

    num = np.mean((f_pred - f_obs) ** 2)
    den = np.mean(f_pred ** 2 + f_obs ** 2)

    return 1 - num / (den + 1e-8)

all_files = []

for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

all_files = sorted(all_files)

print(f"Lead time: t+{lead_time}")
print(f"Case type: {case_type}")
print(f"Found {len(all_files)} files")

fss_raw = {
    w: {
        "mean": [],
        "persistence": [],
        "members": {seed: [] for seed in SEEDS},
    }
    for w in windows
}

fss_smooth = {
    w: {
        "mean": [],
        "persistence": [],
        "members": {seed: [] for seed in SEEDS},
    }
    for w in windows
}

n_skipped = 0
n_loaded = 0

for file_path in tqdm(all_files, desc="Computing ensemble FSS"):
    try:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    n_loaded += 1

    gt_raw = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    persistence_raw = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    members = np.nan_to_num(data["members"].cpu().numpy().astype(np.float32))
    mean_pred = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))

    gt_raw_bin = (gt_raw > 0).astype(np.float32)
    persistence_raw = (persistence_raw > 0).astype(np.float32)

    if MIN_CORES is not None:
        if count_cores(gt_raw_bin) < MIN_CORES:
            n_skipped += 1
            continue

    gt_smooth = maximum_filter(gt_raw_bin, size=GT_FILTER_SIZE)
    pers_smooth = maximum_filter(persistence_raw, size=GT_FILTER_SIZE)

    for w in windows:
        fss_raw[w]["mean"].append(compute_fss(mean_pred, gt_raw_bin, w))
        fss_raw[w]["persistence"].append(compute_fss(persistence_raw, gt_raw_bin, w))

        fss_smooth[w]["mean"].append(compute_fss(mean_pred, gt_smooth, w))
        fss_smooth[w]["persistence"].append(compute_fss(pers_smooth, gt_smooth, w))

        for i, seed in enumerate(SEEDS):
            member_pred = members[i]

            fss_raw[w]["members"][seed].append(
                compute_fss(member_pred, gt_raw_bin, w)
            )

            fss_smooth[w]["members"][seed].append(
                compute_fss(member_pred, gt_smooth, w)
            )

rows = []

print(f"Loaded {n_loaded} files")

if MIN_CORES is None:
    print("Using all scenes")
else:
    print(f"Skipped {n_skipped} scenes with fewer than {MIN_CORES} cores")

print(
    "\nwindow_px | scale_km | raw_mean | raw_member_mean | raw_member_std | "
    "smooth_mean | smooth_member_mean | smooth_member_std | persistence"
)

for w in windows:
    nominal_km = w * PIXEL_SIZE_KM

    raw_mean = np.nanmean(fss_raw[w]["mean"])
    raw_persistence = np.nanmean(fss_raw[w]["persistence"])

    smooth_mean = np.nanmean(fss_smooth[w]["mean"])
    smooth_persistence = np.nanmean(fss_smooth[w]["persistence"])

    raw_member_scores = np.array([
        np.nanmean(fss_raw[w]["members"][seed])
        for seed in SEEDS
    ])

    smooth_member_scores = np.array([
        np.nanmean(fss_smooth[w]["members"][seed])
        for seed in SEEDS
    ])

    raw_member_mean = np.nanmean(raw_member_scores)
    raw_member_std = np.nanstd(raw_member_scores)

    smooth_member_mean = np.nanmean(smooth_member_scores)
    smooth_member_std = np.nanstd(smooth_member_scores)

    print(
        f"{w:>9} | {nominal_km:>8.1f} | "
        f"{raw_mean:>8.4f} | {raw_member_mean:>15.4f} | {raw_member_std:>14.4f} | "
        f"{smooth_mean:>11.4f} | {smooth_member_mean:>18.4f} | "
        f"{smooth_member_std:>17.4f} | {smooth_persistence:>11.4f}"
    )

    row = {
        "lead_time": lead_time,
        "case_type": case_type,
        "window_px": w,
        "nominal_scale_km": nominal_km,
        "raw_ensemble_mean": raw_mean,
        "raw_member_mean": raw_member_mean,
        "raw_member_std": raw_member_std,
        "raw_persistence": raw_persistence,
        "smooth_ensemble_mean": smooth_mean,
        "smooth_member_mean": smooth_member_mean,
        "smooth_member_std": smooth_member_std,
        "smooth_persistence": smooth_persistence,
        "n_files_total": len(all_files),
        "n_files_loaded": n_loaded,
        "n_cases_used": len(fss_raw[w]["mean"]),
        "n_cases_skipped": n_skipped,
    }

    for seed, score in zip(SEEDS, raw_member_scores):
        row[f"raw_seed{seed}"] = score

    for seed, score in zip(SEEDS, smooth_member_scores):
        row[f"smooth_seed{seed}"] = score

    rows.append(row)

output_csv = os.path.join(
    output_dir,
    f"fss_{case_type}_t{lead_time}.csv"
)

pd.DataFrame(rows).to_csv(output_csv, index=False)

print(f"\nSaved FSS to {output_csv}")