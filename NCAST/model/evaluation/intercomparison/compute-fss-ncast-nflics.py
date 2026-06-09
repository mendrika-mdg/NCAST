import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter, maximum_filter
import scipy.ndimage as nd

model_name = sys.argv[1]
lead_time = sys.argv[2]

base_dir = (
    f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/"
    f"{model_name}/t{lead_time}"
)

output_dir = (
    f"/home/users/mendrika/NCAST/Output/evaluation/ncast/fss/{model_name}"
)
os.makedirs(output_dir, exist_ok=True)

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

filtered_files = all_files

print(f"Model: {model_name}")
print(f"Lead time: t+{lead_time}")
print(f"Case type: {case_type}")
print(f"Found {len(filtered_files)} files")

fss_raw = {
    w: {"model": [], "persistence": [], "nflics": []}
    for w in windows
}

fss_smooth = {
    w: {"model": [], "persistence": [], "nflics": []}
    for w in windows
}

n_skipped = 0
n_loaded = 0
n_missing_nflics = 0

for file_path in tqdm(filtered_files, desc="Computing FSS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    if "nflics" not in data:
        n_missing_nflics += 1
        continue

    n_loaded += 1

    gt_raw = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    model = np.nan_to_num(data["pred"].cpu().numpy().astype(np.float32))
    persistence_raw = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    nflics_data = data["nflics"]
    if isinstance(nflics_data, torch.Tensor):
        nflics = np.nan_to_num(nflics_data.cpu().numpy().astype(np.float32))
    else:
        nflics = np.nan_to_num(np.asarray(nflics_data, dtype=np.float32))

    gt_raw_bin = (gt_raw > 0).astype(np.float32)
    persistence_raw = (persistence_raw > 0).astype(np.float32)

    if MIN_CORES is not None:
        if count_cores(gt_raw_bin) < MIN_CORES:
            n_skipped += 1
            continue

    gt_smooth = maximum_filter(gt_raw_bin, size=GT_FILTER_SIZE)
    pers_smooth = maximum_filter(persistence_raw, size=GT_FILTER_SIZE)

    for w in windows:
        fss_raw[w]["model"].append(compute_fss(model, gt_raw_bin, w))
        fss_raw[w]["persistence"].append(compute_fss(persistence_raw, gt_raw_bin, w))
        fss_raw[w]["nflics"].append(compute_fss(nflics, gt_raw_bin, w))

        fss_smooth[w]["model"].append(compute_fss(model, gt_smooth, w))
        fss_smooth[w]["persistence"].append(compute_fss(pers_smooth, gt_smooth, w))
        fss_smooth[w]["nflics"].append(compute_fss(nflics, gt_smooth, w))

rows = []

print(f"Loaded {n_loaded} files")

if n_missing_nflics > 0:
    print(f"Skipped {n_missing_nflics} files without NFLICS")

if MIN_CORES is None:
    print("Using all scenes")
else:
    print(f"Skipped {n_skipped} scenes with fewer than {MIN_CORES} cores")

print(
    "\nwindow_px | scale_km | raw_model | raw_persistence | raw_nflics | "
    "smooth_model | smooth_persistence | smooth_nflics"
)

for w in windows:
    nominal_km = w * PIXEL_SIZE_KM

    r_mod = np.nanmean(fss_raw[w]["model"])
    r_per = np.nanmean(fss_raw[w]["persistence"])
    r_nfl = np.nanmean(fss_raw[w]["nflics"])

    s_mod = np.nanmean(fss_smooth[w]["model"])
    s_per = np.nanmean(fss_smooth[w]["persistence"])
    s_nfl = np.nanmean(fss_smooth[w]["nflics"])

    print(
        f"{w:>9} | {nominal_km:>8.1f} | "
        f"{r_mod:>9.4f} | {r_per:>15.4f} | {r_nfl:>10.4f} | "
        f"{s_mod:>12.4f} | {s_per:>18.4f} | {s_nfl:>13.4f}"
    )

    rows.append({
        "model_name": model_name,
        "lead_time": lead_time,
        "case_type": case_type,
        "window_px": w,
        "nominal_scale_km": nominal_km,
        "raw_model": r_mod,
        "raw_persistence": r_per,
        "raw_nflics": r_nfl,
        "smooth_model": s_mod,
        "smooth_persistence": s_per,
        "smooth_nflics": s_nfl,
        "n_files_total": len(filtered_files),
        "n_files_loaded": n_loaded,
        "n_files_missing_nflics": n_missing_nflics,
        "n_cases_used": len(fss_raw[w]["model"]),
        "n_cases_skipped": n_skipped,
    })

output_csv = os.path.join(
    output_dir,
    f"fss_{case_type}_t{lead_time}.csv"
)

pd.DataFrame(rows).to_csv(output_csv, index=False)

print(f"\nSaved FSS to {output_csv}")