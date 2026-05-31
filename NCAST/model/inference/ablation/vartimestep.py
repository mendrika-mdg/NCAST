import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# model path
sys.path.append("/home/users/mendrika/NCAST/NCAST/model/ablation")
from timestep import Core2MapModel

# arguments
LEAD_TIME = int(sys.argv[1])
YEAR = sys.argv[2]
MONTH = sys.argv[3]
HOUR = sys.argv[4]
ABLATION_NAME = sys.argv[5]

TIMESTEP_INDICES = {
    "t0": [4],
    "t0_tminus1h": [2, 4],
    "t0_tminus1h_tminus2h": [0, 2, 4],
    "all": [0, 1, 2, 3, 4],
}

if ABLATION_NAME not in TIMESTEP_INDICES:
    raise ValueError(f"Unknown temporal ablation: {ABLATION_NAME}")

selected_steps = TIMESTEP_INDICES[ABLATION_NAME]

# paths
SCALER_PATH = "/home/users/mendrika/NCAST/NCAST/scaler/scaler_realcores.pt"
INPUT_ROOT = "/work/scratch-nopw2/mendrika/OB/ncast/raw/inputs_t0"
OUTPUT_BASE = f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/vartimestep/{ABLATION_NAME}/t{LEAD_TIME}"

os.makedirs(OUTPUT_BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"{YEAR}{MONTH}", f"{HOUR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# device
DEVICE = torch.device("cpu")

# threads
num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
torch.set_num_threads(num_threads)
print(f"Running on CPU with {num_threads} threads")

# load model
MODEL_ROOT = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/ncast/checkpoints/temporal/t{LEAD_TIME}/{ABLATION_NAME}/seed1998"
CKPT_PATH = os.path.join(MODEL_ROOT, "best-ncast.ckpt")

if not os.path.exists(CKPT_PATH):
    raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")

model = Core2MapModel.load_from_checkpoint(
    CKPT_PATH,
    map_location=DEVICE
)

model.eval()

print(f"Loaded NCAST temporal ablation model: {CKPT_PATH}")
print(f"Using timestep indices: {selected_steps}")

# prediction
def predict(model, x):
    with torch.no_grad():
        pred = torch.sigmoid(model(x)).squeeze(0).squeeze(0)
    return pred

# scaler
COLS_TO_SCALE = range(4, 12)

scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
mean = np.asarray(scaler["mean"])
scale = np.asarray(scaler["scale"])

# regex
pattern = re.compile(r"input-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.pt$")

def load_ncast_input(year, month, day, hour, minute):
    path = f"{INPUT_ROOT}/input-{year}{month}{day}_{hour}{minute}.pt"
    return torch.load(path)

def load_output(year, month, day, hour, minute, lead_time):
    gt_path = f"/work/scratch-nopw2/mendrika/OB/ncast/raw/targets_t{lead_time}/target-{year}{month}{day}_{hour}{minute}.pt"
    pers_path = f"/work/scratch-nopw2/mendrika/OB/ncast/raw/targets_t0/target-{year}{month}{day}_{hour}{minute}.pt"

    gt = torch.load(gt_path)["data"].numpy()
    persistence = torch.load(pers_path)["data"].numpy()

    return gt, persistence

# discover input files
input_files = []

for f in sorted(os.listdir(INPUT_ROOT)):
    m = pattern.match(f)

    if m:
        y, mo, d, h, mi = m.groups()

        if y == YEAR and mo == MONTH and h == HOUR:
            input_files.append((y, mo, d, h, mi))

print(f"Detected {len(input_files)} inputs for {YEAR}-{MONTH} {HOUR} UTC")

# inference loop
for year, month, day, hour, minute in tqdm(input_files, desc="Predicting"):
    try:
        data = load_ncast_input(year, month, day, hour, minute)

        gt, persistence = load_output(year, month, day, hour, minute, LEAD_TIME)

        X = data["input_tensor"].clone()

        X_np = X.numpy()

        flat = X_np.reshape(-1, X_np.shape[-1])

        flat[:, COLS_TO_SCALE] = (
            flat[:, COLS_TO_SCALE] - mean
        ) / scale

        X_scaled = torch.tensor(
            flat.reshape(X_np.shape),
            dtype=torch.float32
        )

        X_scaled = X_scaled[selected_steps, :, :]

        input_scaled = X_scaled.unsqueeze(0).to(DEVICE)

        pred = predict(model, input_scaled)

        out_file = os.path.join(
            OUTPUT_DIR,
            f"pred_{year}{month}{day}_{hour}{minute}.pt"
        )

        torch.save({
            "pred": pred.cpu(),
            "gt": torch.tensor(gt),
            "gt0": torch.tensor(persistence),
            "checkpoint": CKPT_PATH,
            "ablation_name": ABLATION_NAME,
            "timestep_indices": selected_steps,
        }, out_file)

    except Exception as e:
        print(f"Skipping {year}-{month}-{day} {hour}:{minute}: {e}")

print("All NCAST temporal ablation nowcasts completed on CPU.")