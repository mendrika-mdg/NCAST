import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# lead time not important because inputs are identical for all the future lead times
LEAD_TIME = "1"

BASE_DIR   = "/work/scratch-nopw2/mendrika/OB/ncast/raw"
SHARDS_DIR = f"{BASE_DIR}/shards/t{LEAD_TIME}/train_t{LEAD_TIME}"
SAVE_DIR   = f"/home/users/mendrika/NCAST/NCAST/scaler"

os.makedirs(SAVE_DIR, exist_ok=True)

# [month_sin, month_cos, tod_sin, tod_cos, lat, lon, lat_min, lat_max, lon_min, lon_max, tir, size, mask]

# feature index for mask
MASK_COL_INDEX = 12

# columns to scale
COLS_TO_SCALE = range(4, 12)

scaler = StandardScaler()

# list all train shards
all_shards = sorted(
    [os.path.join(SHARDS_DIR, f) for f in os.listdir(SHARDS_DIR) if f.endswith(".pt")]
)

if not all_shards:
    raise FileNotFoundError(f"No shard files found in {SHARDS_DIR}")

print(f"Computing scaling parameters from {len(all_shards)} training shards (LT{LEAD_TIME})...")

n_real_total = 0

# loop through all shards
for shard_path in tqdm(all_shards, desc="Processing shards"):
    data = torch.load(shard_path, map_location="cpu")
    X = data["X"].numpy()

    # flatten to (B*5*50, 13)
    flat = X.reshape(-1, X.shape[-1])

    # select real cores
    real = flat[flat[:, MASK_COL_INDEX] == 1]
    if real.shape[0] == 0:
        continue

    # update scaler
    scaler.partial_fit(real[:, COLS_TO_SCALE])
    n_real_total += real.shape[0]

print(f"Total real cores processed: {n_real_total:,}")

# save statistics
stats = {
    "mean": scaler.mean_,
    "scale": scaler.scale_,
    "cols": COLS_TO_SCALE,
    "n_real_cores": n_real_total
}

out_path = os.path.join(SAVE_DIR, f"scaler_realcores.pt")
torch.save(stats, out_path)

print(f"Scaling parameters saved to {out_path}")
print("Done.")
