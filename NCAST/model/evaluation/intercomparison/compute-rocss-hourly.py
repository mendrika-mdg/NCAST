import os
import sys
import torch
import numpy as np
import scipy.ndimage as nd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

lead_time = sys.argv[1]
target_hour = sys.argv[2]

base_dir = f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/ncast-nflics/t{lead_time}"
output_dir = "/home/users/mendrika/NCAST/Output/evaluation/ncast/rocss"
os.makedirs(output_dir, exist_ok=True)

H, W = 512, 512
L_pixels = 25

y_all = []
pred_all = []
nflics_all = []
pers_all = []


def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()

    return np.asarray(x)


def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")

    if len(parts) < 3:
        return None

    hh = parts[2].replace(".pt", "")[:2]

    if hh.isdigit():
        return hh

    return None


all_files = []

for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

filtered = [p for p in all_files if extract_hour(p) == target_hour]

print(f"Found {len(filtered)} files at hour={target_hour} UTC")

for file_path in tqdm(filtered, desc="Collecting all pixels"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    gt = np.nan_to_num(to_numpy(data["gt"]).astype(np.float32))
    gt0 = np.nan_to_num(to_numpy(data["gt0"]).astype(np.float32))
    pred = np.nan_to_num(to_numpy(data["pred"]).astype(np.float32))
    nflics = np.nan_to_num(to_numpy(data["nflics"]).astype(np.float32))

    if gt.shape != (H, W):
        continue

    if gt0.shape != (H, W):
        continue

    if pred.shape != (H, W):
        continue

    if nflics.shape != (H, W):
        continue

    gt = np.clip(gt, 0, 1)
    gt0 = np.clip(gt0, 0, 1)
    pred = np.clip(pred, 0, 1)
    nflics = np.clip(nflics, 0, 1)

    gt_smooth = nd.maximum_filter(gt, size=L_pixels).astype(np.float32)
    pers_smooth = nd.maximum_filter(gt0, size=L_pixels).astype(np.float32)

    y_all.append((gt_smooth.reshape(-1) > 0).astype(np.int8))
    pred_all.append(pred.reshape(-1))
    nflics_all.append(nflics.reshape(-1))
    pers_all.append(pers_smooth.reshape(-1))

if len(y_all) == 0:
    raise RuntimeError(f"No valid files found for hour={target_hour} and lead_time={lead_time}")

y_all = np.concatenate(y_all)
pred_all = np.concatenate(pred_all)
nflics_all = np.concatenate(nflics_all)
pers_all = np.concatenate(pers_all)

print(f"Total samples: {y_all.size:,}")
print(f"Event pixels: {int(y_all.sum()):,}")
print(f"Event fraction: {y_all.mean():.6f}")

if np.all(y_all == 0) or np.all(y_all == 1):
    raise RuntimeError("AUC cannot be computed because the target contains only one class")

auc_model = roc_auc_score(y_all, pred_all)
auc_nflics = roc_auc_score(y_all, nflics_all)
auc_persistence = roc_auc_score(y_all, pers_all)

rocss_vs_nflics = 1.0 - (auc_nflics / auc_model)
rocss_vs_persistence = 1.0 - (auc_persistence / auc_model)

out_file = os.path.join(
    output_dir,
    f"rocss_hour_{target_hour}_t{lead_time}_smooth{L_pixels}.npz"
)

np.savez(
    out_file,
    lead_time=lead_time,
    target_hour=target_hour,
    L_pixels=L_pixels,
    auc_model=auc_model,
    auc_nflics=auc_nflics,
    auc_persistence=auc_persistence,
    rocss_vs_nflics=rocss_vs_nflics,
    rocss_vs_persistence=rocss_vs_persistence,
    n_samples=y_all.size,
    n_events=int(y_all.sum()),
    event_fraction=float(y_all.mean()),
)

print(f"Saved hourly ROCSS result to {out_file}")
print(f"AUC model: {auc_model:.4f}")
print(f"AUC NFLICS: {auc_nflics:.4f}")
print(f"AUC persistence: {auc_persistence:.4f}")
print(f"ROCSS vs NFLICS: {rocss_vs_nflics:.4f}")
print(f"ROCSS vs persistence: {rocss_vs_persistence:.4f}")