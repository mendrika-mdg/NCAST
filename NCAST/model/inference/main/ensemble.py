import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training")
from ncast import Core2MapModel

LEAD_TIME = sys.argv[1]
YEAR = sys.argv[2]
MONTH = sys.argv[3]
HOUR = sys.argv[4]

SEEDS = [10, 20, 30, 40, 50]

ENSEMBLE_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/checkpoints/WS/transformer/t{LEAD_TIME}"

SCALER_PATH = "/home/users/mendrika/NCAST/NCAST/scaler/scaler_realcores.pt"
INPUT_ROOT = "/work/scratch-nopw2/mendrika/OB/ncast/raw/inputs_t0"
OUTPUT_BASE = f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/ensemble/t{LEAD_TIME}"

OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"{YEAR}{MONTH}", f"{HOUR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
torch.set_num_threads(num_threads)
print(f"Running on CPU with {num_threads} threads")

ckpts = []

for seed in SEEDS:
    ckpt_path = os.path.join(
        ENSEMBLE_DIR,
        f"seed{seed}",
        "best-core2map.ckpt"
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpts.append(ckpt_path)

print("Checkpoints:")
for path in ckpts:
    print(path)

models = []

for path in ckpts:
    model = Core2MapModel.load_from_checkpoint(path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()
    models.append(model)

print(f"Loaded {len(models)} ensemble models on CPU")

def ensemble_predict(models, x):
    preds = []

    with torch.no_grad():
        for model in models:
            pred = torch.sigmoid(model(x)).squeeze(0).squeeze(0)
            preds.append(pred.cpu())

    members = torch.stack(preds, dim=0)
    mean_pred = members.mean(dim=0)
    var_pred = members.var(dim=0, unbiased=False)

    return members, mean_pred, var_pred

COLS_TO_SCALE = range(4, 12)

scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
mean = np.asarray(scaler["mean"])
scale = np.asarray(scaler["scale"])

pattern = re.compile(r"input-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.pt$")

def load_ncast_input(year, month, day, hour, minute):
    path = f"{INPUT_ROOT}/input-{year}{month}{day}_{hour}{minute}.pt"
    return torch.load(path, map_location="cpu")

def load_output(year, month, day, hour, minute, lead_time):
    gt_path = f"/work/scratch-nopw2/mendrika/OB/ncast/raw/targets_t{lead_time}/target-{year}{month}{day}_{hour}{minute}.pt"
    pers_path = f"/work/scratch-nopw2/mendrika/OB/ncast/raw/targets_t0/target-{year}{month}{day}_{hour}{minute}.pt"

    gt = torch.load(gt_path, map_location="cpu")["data"]
    persistence = torch.load(pers_path, map_location="cpu")["data"]

    return gt, persistence

input_files = []

for filename in sorted(os.listdir(INPUT_ROOT)):
    match = pattern.match(filename)

    if match:
        y, mo, d, h, mi = match.groups()

        if y == YEAR and mo == MONTH and h == HOUR:
            input_files.append((y, mo, d, h, mi))

print(f"Detected {len(input_files)} inputs for {YEAR}-{MONTH} {HOUR} UTC")

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

        input_scaled = X_scaled.unsqueeze(0).to(DEVICE)

        members, mean_pred, var_pred = ensemble_predict(models, input_scaled)

        out_file = os.path.join(
            OUTPUT_DIR,
            f"pred_{year}{month}{day}_{hour}{minute}.pt"
        )

        torch.save(
            {
                "members": members,
                "mean": mean_pred,
                "var": var_pred,
                "gt": gt.cpu(),
                "gt0": persistence.cpu(),
                "seeds": SEEDS,
                "checkpoint_paths": ckpts,
            },
            out_file,
        )

    except Exception as e:
        print(f"Skipping {year}-{month}-{day} {hour}:{minute}: {e}")

print("All ensemble nowcasts completed on CPU.")