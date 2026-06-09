import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import scipy.ndimage as nd

lead_time = sys.argv[1]
target_hour = sys.argv[2]

base_dir = f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/ncast-nflics/t{lead_time}"
output_dir = "/home/users/mendrika/NCAST/Output/evaluation/ncast/bss"
os.makedirs(output_dir, exist_ok=True)

H, W = 512, 512
L_pixels = 25

bs_model_sum = np.zeros((H, W), dtype=np.float64)
bs_nflics_sum = np.zeros((H, W), dtype=np.float64)
bs_pers_sum = np.zeros((H, W), dtype=np.float64)
count = np.zeros((H, W), dtype=np.int32)

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

for file_path in tqdm(filtered, desc="Accumulating pixelwise BSS"):
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

    bs_model_sum += (pred - gt_smooth) ** 2
    bs_nflics_sum += (nflics - gt_smooth) ** 2
    bs_pers_sum += (pers_smooth - gt_smooth) ** 2

    count += 1

if np.all(count == 0):
    raise RuntimeError(f"No valid files found for hour={target_hour} and lead_time={lead_time}")

mask = count > 0

bs_model_mean = np.full((H, W), np.nan, dtype=np.float32)
bs_nflics_mean = np.full((H, W), np.nan, dtype=np.float32)
bs_pers_mean = np.full((H, W), np.nan, dtype=np.float32)

bs_model_mean[mask] = bs_model_sum[mask] / count[mask]
bs_nflics_mean[mask] = bs_nflics_sum[mask] / count[mask]
bs_pers_mean[mask] = bs_pers_sum[mask] / count[mask]

bss_vs_nflics = np.full((H, W), np.nan, dtype=np.float32)
bss_vs_persistence = np.full((H, W), np.nan, dtype=np.float32)

valid_nflics = (bs_nflics_mean > 1e-12) & mask
valid_persistence = (bs_pers_mean > 1e-12) & mask

bss_vs_nflics[valid_nflics] = 1.0 - (
    bs_model_mean[valid_nflics] / bs_nflics_mean[valid_nflics]
)

bss_vs_persistence[valid_persistence] = 1.0 - (
    bs_model_mean[valid_persistence] / bs_pers_mean[valid_persistence]
)

out_bss_nflics = os.path.join(
    output_dir,
    f"bss_vs_nflics_pixelwise_hour_{target_hour}_t{lead_time}_smooth{L_pixels}.npy"
)

out_bss_persistence = os.path.join(
    output_dir,
    f"bss_vs_persistence_pixelwise_hour_{target_hour}_t{lead_time}_smooth{L_pixels}.npy"
)

out_bs_model = os.path.join(
    output_dir,
    f"bs_model_pixelwise_hour_{target_hour}_t{lead_time}_smooth{L_pixels}.npy"
)

out_bs_nflics = os.path.join(
    output_dir,
    f"bs_nflics_pixelwise_hour_{target_hour}_t{lead_time}_smooth{L_pixels}.npy"
)

out_bs_persistence = os.path.join(
    output_dir,
    f"bs_persistence_pixelwise_hour_{target_hour}_t{lead_time}_smooth{L_pixels}.npy"
)

np.save(out_bss_nflics, bss_vs_nflics)
np.save(out_bss_persistence, bss_vs_persistence)

np.save(out_bs_model, bs_model_mean)
np.save(out_bs_nflics, bs_nflics_mean)
np.save(out_bs_persistence, bs_pers_mean)

print(f"Saved BSS against NFLICS to {out_bss_nflics}")
print(f"Saved BSS against persistence to {out_bss_persistence}")
print(f"Saved model BS to {out_bs_model}")
print(f"Saved NFLICS BS to {out_bs_nflics}")
print(f"Saved persistence BS to {out_bs_persistence}")