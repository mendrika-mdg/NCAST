import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from netCDF4 import Dataset

import warnings
warnings.filterwarnings("ignore")

# model path
sys.path.append("/home/users/mendrika/NCAST/NCAST/model/base")
from ncast import Core2MapModel

# arguments
LEAD_TIME = int(sys.argv[1])
YEAR = sys.argv[2]
MONTH = sys.argv[3]
HOUR = sys.argv[4]

best_configs = {
    "1": {"lr": 2e-4, "dropout_p": 0.2, "pos_weight": 25.0, "alpha": 0.3},
    "3": {"lr": 1e-4, "dropout_p": 0.3, "pos_weight": 25.0, "alpha": 0.3},
    "6": {"lr": 1e-4, "dropout_p": 0.3, "pos_weight": 25.0, "alpha": 0.3},
}

cfg = best_configs[str(LEAD_TIME)]

# paths
MODEL_ROOT = "/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/ncast/checkpoints/tuning"

RUN_DIR = os.path.join(
    MODEL_ROOT,
    f"t{LEAD_TIME}_lr{cfg['lr']}_do{cfg['dropout_p']}_pw{cfg['pos_weight']}_a{cfg['alpha']}"
)

SCALER_PATH = "/home/users/mendrika/NCAST/NCAST/scaler/scaler_realcores.pt"
INPUT_ROOT = "/work/scratch-nopw2/mendrika/OB/ncast/raw/inputs_t0"
OUTPUT_BASE = f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/ncast-nflics/t{LEAD_TIME}"

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
CKPT_PATH = os.path.join(RUN_DIR, "best-ncast.ckpt")

if not os.path.exists(CKPT_PATH):
    raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")

model = Core2MapModel.load_from_checkpoint(
    CKPT_PATH,
    map_location=DEVICE
)

model.eval()

print(f"Loaded NCAST model: {CKPT_PATH}")

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

def load_nflics_nowcast(year, month, day, hour, minute, lead_time):
    base = f"/gws/ssde/j25b/swift/nflics_nowcasts/{year}/{month}/{day}/{hour}{minute}"
    path = f"{base}/Nowcast_{year}{month}{day}{hour}{minute}_000.nc"

    with Dataset(path, mode="r") as ds:
        nflics = ds["Probability"][lead_time, :, :]

    return nflics

NFLICS_ymin, NFLICS_ymax = 249, 764
NFLICS_xmin, NFLICS_xmax = 101, 597

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

        nflics = load_nflics_nowcast(year, month, day, hour, minute, LEAD_TIME)
        nflics = np.asarray(nflics, dtype=float) / 100.0
        nflics[nflics <= 0] = 0.0
        nflics[np.isnan(nflics)] = 0.0
        nflics = np.clip(nflics, 0.0, 1.0)

        nflics_crop = nflics[
            NFLICS_ymin:NFLICS_ymax + 1,
            NFLICS_xmin:NFLICS_xmax + 1,
        ]

        scale_y = 512 / nflics_crop.shape[0]
        scale_x = 512 / nflics_crop.shape[1]
        nflics_512 = zoom(nflics_crop, (scale_y, scale_x), order=1)

        X = data["input_tensor"].clone()

        X_np = X.numpy()

        flat = X_np.reshape(-1, X_np.shape[-1])

        flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale

        X_scaled = torch.tensor(flat.reshape(X_np.shape), dtype=torch.float32)

        input_scaled = X_scaled.unsqueeze(0).to(DEVICE)

        pred = predict(model, input_scaled)

        out_file = os.path.join(
            OUTPUT_DIR,
            f"pred_{year}{month}{day}_{hour}{minute}.pt"
        )

        torch.save({
            "pred": pred.cpu(),
            "nflics": torch.tensor(nflics_512, dtype=torch.float32),
            "gt": torch.tensor(gt),
            "gt0": torch.tensor(persistence),
            "checkpoint": CKPT_PATH,
        }, out_file)

    except Exception as e:
        print(f"Skipping {year}-{month}-{day} {hour}:{minute}: {e}")

print("All NCAST nowcasts completed on CPU.")