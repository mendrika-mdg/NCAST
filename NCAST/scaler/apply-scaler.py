import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# check args
if len(sys.argv) != 3:
    print("Usage: python apply_scaler_to_shards.py <partition> <lead_time>")
    sys.exit(1)

# read args
PARTITION = sys.argv[1]
LEAD_TIME = sys.argv[2]

# directories
SCALER_PATH = "/home/users/mendrika/NCAST/NCAST/scaler/scaler_realcores.pt"

BASE_DIR = "/work/scratch-nopw2/mendrika/OB/ncast/raw"
SHARDS_DIR = f"{BASE_DIR}/shards/t{LEAD_TIME}/{PARTITION}_t{LEAD_TIME}"

SAVE_DIR = f"/work/scratch-nopw2/mendrika/OB/ncast/preprocessed/shards/t{LEAD_TIME}/{PARTITION}_t{LEAD_TIME}"
os.makedirs(SAVE_DIR, exist_ok=True)

# column indices
MASK_COL_INDEX = 12
COLS_TO_SCALE = range(4, 12)

# load scaler (computed using real cores only)
scaler = torch.load(SCALER_PATH, weights_only=False)
mean   = np.asarray(scaler["mean"])
scale  = np.asarray(scaler["scale"])

print(f"Loaded scaler from {SCALER_PATH}")
print(f"Applying to partition={PARTITION.upper()}, lead_time={LEAD_TIME}h")

# list shards
all_shards = sorted(
    [os.path.join(SHARDS_DIR, f) for f in os.listdir(SHARDS_DIR) if f.endswith('.pt')]
)
if not all_shards:
    raise FileNotFoundError(f"No shards found in {SHARDS_DIR}")

print(f"Found {len(all_shards)} shard files to process...")

# loop through shards
for shard_path in tqdm(all_shards, desc=f"Scaling {PARTITION}_t{LEAD_TIME}"):
    data = torch.load(shard_path, map_location="cpu")
    X = data["X"].numpy()

    # flatten
    flat = X.reshape(-1, X.shape[-1])

    # scale all cores (real + padded) using stats from real cores
    flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale

    # reshape
    X_scaled = flat.reshape(X.shape)

    # save scaled shard
    scaled_data = {
        "X": torch.tensor(X_scaled, dtype=torch.float32),
        "Y": data["Y"],
        "ID": data["ID"],
    }

    out_path = os.path.join(SAVE_DIR, os.path.basename(shard_path))
    torch.save(scaled_data, out_path)

print(f"Finished scaling {PARTITION}_t{LEAD_TIME}. Scaled files saved to {SAVE_DIR}")
