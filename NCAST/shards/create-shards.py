import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Config
PARTITION = sys.argv[1]
LEAD_TIME = sys.argv[2]

BASE_DIR = "/work/scratch-nopw2/mendrika/OB/ncast/raw"
RAW_INPUT_DIR  = f"{BASE_DIR}/inputs_t0"
RAW_TARGET_DIR = f"{BASE_DIR}/targets_t{LEAD_TIME}"

SPLIT_FILE = f"/home/users/mendrika/NCAST/NCAST/data-split/{PARTITION}_files.txt"

# shards within raw folder
SHARDS_DIR = f"{BASE_DIR}/shards/t{LEAD_TIME}/{PARTITION}_t{LEAD_TIME}"
os.makedirs(SHARDS_DIR, exist_ok=True)

# Load list
with open(SPLIT_FILE) as f:
    files = [line.strip() for line in f if line.strip()]

FILES_PER_SHARD = 1000
print(f"Creating shards for partition={PARTITION.upper()}, lead time={LEAD_TIME}h")
print(f"Total input files: {len(files):,}")

# Buffers
shard_inputs, shard_targets, shard_ids = [], [], []
shard_index = 0

# Main loop
for i, fpath in enumerate(tqdm(files, desc="Sharding inputs and targets")):
    try:
        # Load input
        data_in = torch.load(fpath, map_location="cpu")
        if "input_tensor" not in data_in:
            print(f"Missing keys in {fpath}, skipping file.")
            continue

        x = data_in["input_tensor"]         # 5 x 50 x 13
        nowcast_id = data_in.get("nowcast_origin", os.path.basename(fpath))

        # Convert to NumPy
        x = x.detach().cpu().numpy()

        # Load target
        fname = os.path.basename(fpath).replace("input-", "target-")
        target_path = os.path.join(RAW_TARGET_DIR, fname)
        if not os.path.exists(target_path):
            print(f"Missing target file: {target_path}")
            continue

        data_out = torch.load(target_path, map_location="cpu")
        y = data_out.get("data", None)
        if y is None:
            print(f"Missing 'data' key in {target_path}")
            continue
        if y.ndim != 2:         # 512 x 512
            y = y.squeeze()

        y = y.detach().cpu().numpy().astype(np.uint8)

        # Append
        shard_inputs.append(x)
        shard_targets.append(y)
        shard_ids.append(nowcast_id)

        # Save shard
        if len(shard_inputs) >= FILES_PER_SHARD or (i + 1) == len(files):
            shard_path = os.path.join(SHARDS_DIR, f"shard_{shard_index:03d}.pt")
            torch.save({
                "X": torch.tensor(np.stack(shard_inputs), dtype=torch.float32),
                "Y": torch.tensor(np.stack(shard_targets), dtype=torch.uint8),
                "ID": shard_ids
            }, shard_path)

            print(f"Saved shard_{shard_index:03d} ({len(shard_inputs)} samples) → {shard_path}")
            shard_index += 1
            shard_inputs, shard_targets, shard_ids = [], [], []

    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        continue

print(f"Finished creating {PARTITION.upper()} shards for LT{LEAD_TIME}.")
