import os
import sys
import torch
import numpy as np
from tqdm import tqdm

model_name = sys.argv[1]
lead_time = sys.argv[2]

base_dir = (
    f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/"
    f"{model_name}/t{lead_time}"
)

output_dir = (
    f"/home/users/mendrika/NCAST/Output/evaluation/ncast/auc/{model_name}"
)
os.makedirs(output_dir, exist_ok=True)

thresholds = np.linspace(0, 1, 101)

TP_model = np.zeros_like(thresholds, dtype=np.int64)
FP_model = np.zeros_like(thresholds, dtype=np.int64)
TN_model = np.zeros_like(thresholds, dtype=np.int64)
FN_model = np.zeros_like(thresholds, dtype=np.int64)

TP_pers = np.zeros_like(thresholds, dtype=np.int64)
FP_pers = np.zeros_like(thresholds, dtype=np.int64)
TN_pers = np.zeros_like(thresholds, dtype=np.int64)
FN_pers = np.zeros_like(thresholds, dtype=np.int64)

TP_nflics = np.zeros_like(thresholds, dtype=np.int64)
FP_nflics = np.zeros_like(thresholds, dtype=np.int64)
TN_nflics = np.zeros_like(thresholds, dtype=np.int64)
FN_nflics = np.zeros_like(thresholds, dtype=np.int64)

all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

print(f"Found {len(all_files)} files")

for file_path in tqdm(all_files, desc=f"Streaming ROC for t+{lead_time}"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    gt = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    pred = np.nan_to_num(data["pred"].cpu().numpy().astype(np.float32))
    pers = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))
    nflics = np.nan_to_num(data["nflics"].cpu().numpy().astype(np.float32))

    gt = (gt.reshape(-1) > 0).astype(np.int8)
    pred = pred.reshape(-1)
    pers = pers.reshape(-1)
    nflics = nflics.reshape(-1)

    for i, th in enumerate(thresholds):
        pred_bin = pred >= th
        TP_model[i] += np.sum((pred_bin == 1) & (gt == 1))
        FP_model[i] += np.sum((pred_bin == 1) & (gt == 0))
        TN_model[i] += np.sum((pred_bin == 0) & (gt == 0))
        FN_model[i] += np.sum((pred_bin == 0) & (gt == 1))

        pers_bin = pers >= th
        TP_pers[i] += np.sum((pers_bin == 1) & (gt == 1))
        FP_pers[i] += np.sum((pers_bin == 1) & (gt == 0))
        TN_pers[i] += np.sum((pers_bin == 0) & (gt == 0))
        FN_pers[i] += np.sum((pers_bin == 0) & (gt == 1))

        nflics_bin = nflics >= th
        TP_nflics[i] += np.sum((nflics_bin == 1) & (gt == 1))
        FP_nflics[i] += np.sum((nflics_bin == 1) & (gt == 0))
        TN_nflics[i] += np.sum((nflics_bin == 0) & (gt == 0))
        FN_nflics[i] += np.sum((nflics_bin == 0) & (gt == 1))

TPR_model = TP_model / (TP_model + FN_model + 1e-12)
FPR_model = FP_model / (FP_model + TN_model + 1e-12)

TPR_pers = TP_pers / (TP_pers + FN_pers + 1e-12)
FPR_pers = FP_pers / (FP_pers + TN_pers + 1e-12)

TPR_nflics = TP_nflics / (TP_nflics + FN_nflics + 1e-12)
FPR_nflics = FP_nflics / (FP_nflics + TN_nflics + 1e-12)

idx_model = np.argsort(FPR_model)
idx_pers = np.argsort(FPR_pers)
idx_nflics = np.argsort(FPR_nflics)

auc_model = np.trapz(TPR_model[idx_model], FPR_model[idx_model])
auc_pers = np.trapz(TPR_pers[idx_pers], FPR_pers[idx_pers])
auc_nflics = np.trapz(TPR_nflics[idx_nflics], FPR_nflics[idx_nflics])

np.save(os.path.join(output_dir, f"roc_thresholds_t{lead_time}.npy"), thresholds)

np.save(os.path.join(output_dir, f"roc_fpr_model_t{lead_time}.npy"), FPR_model)
np.save(os.path.join(output_dir, f"roc_tpr_model_t{lead_time}.npy"), TPR_model)

np.save(os.path.join(output_dir, f"roc_fpr_persistence_t{lead_time}.npy"), FPR_pers)
np.save(os.path.join(output_dir, f"roc_tpr_persistence_t{lead_time}.npy"), TPR_pers)

np.save(os.path.join(output_dir, f"roc_fpr_nflics_t{lead_time}.npy"), FPR_nflics)
np.save(os.path.join(output_dir, f"roc_tpr_nflics_t{lead_time}.npy"), TPR_nflics)

np.save(os.path.join(output_dir, f"auc_model_t{lead_time}.npy"), auc_model)
np.save(os.path.join(output_dir, f"auc_persistence_t{lead_time}.npy"), auc_pers)
np.save(os.path.join(output_dir, f"auc_nflics_t{lead_time}.npy"), auc_nflics)

print("\nROC arrays saved in:", output_dir)
print(f"AUC model: {auc_model:.4f}")
print(f"AUC persistence: {auc_pers:.4f}")
print(f"AUC NFLICS: {auc_nflics:.4f}")
print("Done.")