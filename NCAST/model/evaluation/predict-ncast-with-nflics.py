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
sys.path.append("/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training")
from ncast import Core2MapModel

# arguments
LEAD_TIME = int(sys.argv[1])  # use int consistently
YEAR  = sys.argv[2]           # strings for path components
MONTH = sys.argv[3]
HOUR  = sys.argv[4]

# paths
ENSEMBLE_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/checkpoints/WS/transformer/t{LEAD_TIME}"
SCALER_PATH  = "/home/users/mendrika/Object-Based-LSTMConv/outputs/scaler/scaler_realcores.pt"
INPUT_ROOT   = "/work/scratch-nopw2/mendrika/OB/raw/inputs_t0"
OUTPUT_BASE  = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-nflics-full-corrected/t{LEAD_TIME}"

os.makedirs(OUTPUT_BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"{YEAR}{MONTH}", f"{HOUR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# device
DEVICE = torch.device("cpu")

# threads
num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
torch.set_num_threads(num_threads)
print(f"Running on CPU with {num_threads} threads")

# load ensemble models
ckpts = []
for root, _, files in os.walk(ENSEMBLE_DIR):
    for f in files:
        if f.endswith(".ckpt"):
            ckpts.append(os.path.join(root, f))
ckpts = sorted(ckpts)
if not ckpts:
    raise RuntimeError(f"No checkpoints in {ENSEMBLE_DIR}")

models = []
for path in ckpts:
    model = Core2MapModel.load_from_checkpoint(path, map_location=DEVICE)
    model.eval()
    models.append(model)

print(f"Loaded {len(models)} ensemble models on CPU")

# ensemble inference
def ensemble_predict(models, x):
    preds = []
    with torch.no_grad():
        for model in models:
            pred = torch.sigmoid(model(x)).squeeze(0).squeeze(0)
            preds.append(pred)
    preds = torch.stack(preds)
    return preds.mean(dim=0), preds.var(dim=0)

# scaler
MASK_COL_INDEX = 12
COLS_TO_SCALE = range(4, 12)
scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
mean = np.asarray(scaler["mean"])
scale = np.asarray(scaler["scale"])

# regex
pattern = re.compile(r"input-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.pt$")

def load_zcast_input(year, month, day, hour, minute):
    path = f"{INPUT_ROOT}/input-{year}{month}{day}_{hour}{minute}.pt"
    return torch.load(path)

def load_output(year, month, day, hour, minute, lead_time):
    gt_path   = f"/work/scratch-nopw2/mendrika/OB/raw/targets_t{lead_time}/target-{year}{month}{day}_{hour}{minute}.pt"
    pers_path = f"/work/scratch-nopw2/mendrika/OB/raw/targets_t0/target-{year}{month}{day}_{hour}{minute}.pt"
    gt = torch.load(gt_path)["data"].numpy()
    persistence = torch.load(pers_path)["data"].numpy()
    return gt, persistence

def load_NFLICS_nowcast(year, month, day, hour, minute, lead_time):
    base = f"/gws/ssde/j25b/swift/nflics_nowcasts/{year}/{month}/{day}/{hour}{minute}"
    file = f"{base}/Nowcast_{year}{month}{day}{hour}{minute}_000.nc"
    # lead_time is the index in the Probability dimension
    return Dataset(file, mode="r")["Probability"][lead_time, :, :]

# WA domain bounds
lat_top = 19.99001
lat_bottom = -2.018712
lon_left = -23.0
lon_right = 32.021806

ny, nx = 776, 1748
nflics_lats = np.linspace(lat_bottom, lat_top, ny)
nflics_lons = np.linspace(lon_left, lon_right, nx)
nflics_lon2d, nflics_lat2d = np.meshgrid(nflics_lons, nflics_lats)

# crop bounds
LAT_MIN = 5.033134
LAT_MAX = 19.701021
LON_MIN = -19.831186
LON_MAX = -4.1754155

mask = (
    (nflics_lat2d >= LAT_MIN) &
    (nflics_lat2d <= LAT_MAX) &
    (nflics_lon2d >= LON_MIN) &
    (nflics_lon2d <= LON_MAX)
)

rows = np.where(mask.any(axis=1))[0]
cols = np.where(mask.any(axis=0))[0]
row_min, row_max = rows[0], rows[-1]
col_min, col_max = cols[0], cols[-1]

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
        data = load_zcast_input(year, month, day, hour, minute)
        gt, persistence = load_output(year, month, day, hour, minute, LEAD_TIME)

        nf = load_NFLICS_nowcast(year, month, day, hour, minute, LEAD_TIME)
    
        nf[nf <= 0] = 0.0
        nf = nf / 100.0
        nf[np.isnan(nf)] = 0.0
        nf = np.clip(nf, 0.0, 1.0)

        nf_crop = nf[row_min:row_max+1, col_min:col_max+1]
        nf_512  = zoom(nf_crop, (512 / nf_crop.shape[0], 512 / nf_crop.shape[1]), order=1)

        X = data["input_tensor"].clone()
        X_np = X.numpy()
        flat = X_np.reshape(-1, X_np.shape[-1])
        flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale
        X_scaled = torch.tensor(flat.reshape(X_np.shape), dtype=torch.float32)
        input_scaled = X_scaled.unsqueeze(0).to(DEVICE)

        mean_pred, var_pred = ensemble_predict(models, input_scaled)

        out_file = os.path.join(
            OUTPUT_DIR,
            f"pred_{year}{month}{day}_{hour}{minute}.pt"
        )

        torch.save({
            "mean": mean_pred.cpu(),
            "var": var_pred.cpu(),
            "nflics": nf_512,
            "gt": torch.tensor(gt),
            "gt0": torch.tensor(persistence),
        }, out_file)

    except Exception as e:
        print(f"Skipping {year}-{month}-{day} {hour}:{minute}: {e}")

print("All ensemble nowcasts completed on CPU.")
