import os
import sys
import torch
import numpy as np
from tqdm import tqdm

lead_time = sys.argv[1]

base_dir = f"/work/scratch-nopw2/mendrika/OB/ncast/evaluation/nowcasts/ensemble/t{lead_time}"
output_dir = "/home/users/mendrika/NCAST/Output/evaluation/ncast/auc/ensemble"

os.makedirs(output_dir, exist_ok=True)

seeds = [10, 20, 30, 40, 50]
thresholds = np.linspace(0, 1, 101)

n_members = len(seeds)

TP_members = np.zeros((n_members, len(thresholds)), dtype=np.int64)
FP_members = np.zeros((n_members, len(thresholds)), dtype=np.int64)
TN_members = np.zeros((n_members, len(thresholds)), dtype=np.int64)
FN_members = np.zeros((n_members, len(thresholds)), dtype=np.int64)

TP_mean = np.zeros_like(thresholds, dtype=np.int64)
FP_mean = np.zeros_like(thresholds, dtype=np.int64)
TN_mean = np.zeros_like(thresholds, dtype=np.int64)
FN_mean = np.zeros_like(thresholds, dtype=np.int64)

TP_pers = np.zeros_like(thresholds, dtype=np.int64)
FP_pers = np.zeros_like(thresholds, dtype=np.int64)
TN_pers = np.zeros_like(thresholds, dtype=np.int64)
FN_pers = np.zeros_like(thresholds, dtype=np.int64)

all_files = []

for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

all_files = sorted(all_files)

print(f"Found {len(all_files)} files")

for file_path in tqdm(all_files, desc=f"Streaming ensemble ROC for t+{lead_time}"):
    try:
        data = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Skipping unreadable file {file_path}: {e}")
        continue

    gt = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    gt0 = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    members = np.nan_to_num(data["members"].cpu().numpy().astype(np.float32))
    mean_pred = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))

    gt = (gt.reshape(-1) > 0).astype(np.int8)
    gt0 = gt0.reshape(-1)
    mean_pred = mean_pred.reshape(-1)

    members = members.reshape(members.shape[0], -1)

    for i, th in enumerate(thresholds):
        mean_bin = mean_pred >= th

        TP_mean[i] += np.sum((mean_bin == 1) & (gt == 1))
        FP_mean[i] += np.sum((mean_bin == 1) & (gt == 0))
        TN_mean[i] += np.sum((mean_bin == 0) & (gt == 0))
        FN_mean[i] += np.sum((mean_bin == 0) & (gt == 1))

        pers_bin = gt0 >= th

        TP_pers[i] += np.sum((pers_bin == 1) & (gt == 1))
        FP_pers[i] += np.sum((pers_bin == 1) & (gt == 0))
        TN_pers[i] += np.sum((pers_bin == 0) & (gt == 0))
        FN_pers[i] += np.sum((pers_bin == 0) & (gt == 1))

        for m in range(n_members):
            member_bin = members[m] >= th

            TP_members[m, i] += np.sum((member_bin == 1) & (gt == 1))
            FP_members[m, i] += np.sum((member_bin == 1) & (gt == 0))
            TN_members[m, i] += np.sum((member_bin == 0) & (gt == 0))
            FN_members[m, i] += np.sum((member_bin == 0) & (gt == 1))

TPR_mean = TP_mean / (TP_mean + FN_mean + 1e-12)
FPR_mean = FP_mean / (FP_mean + TN_mean + 1e-12)

TPR_pers = TP_pers / (TP_pers + FN_pers + 1e-12)
FPR_pers = FP_pers / (FP_pers + TN_pers + 1e-12)

TPR_members = TP_members / (TP_members + FN_members + 1e-12)
FPR_members = FP_members / (FP_members + TN_members + 1e-12)

idx_mean = np.argsort(FPR_mean)
idx_pers = np.argsort(FPR_pers)

auc_mean = np.trapz(TPR_mean[idx_mean], FPR_mean[idx_mean])
auc_pers = np.trapz(TPR_pers[idx_pers], FPR_pers[idx_pers])

auc_members = np.zeros(n_members, dtype=np.float32)

for m in range(n_members):
    idx = np.argsort(FPR_members[m])
    auc_members[m] = np.trapz(TPR_members[m, idx], FPR_members[m, idx])

np.save(os.path.join(output_dir, f"roc_thresholds_t{lead_time}.npy"), thresholds)

np.save(os.path.join(output_dir, f"roc_fpr_mean_t{lead_time}.npy"), FPR_mean)
np.save(os.path.join(output_dir, f"roc_tpr_mean_t{lead_time}.npy"), TPR_mean)
np.save(os.path.join(output_dir, f"auc_mean_t{lead_time}.npy"), auc_mean)

np.save(os.path.join(output_dir, f"roc_fpr_members_t{lead_time}.npy"), FPR_members)
np.save(os.path.join(output_dir, f"roc_tpr_members_t{lead_time}.npy"), TPR_members)
np.save(os.path.join(output_dir, f"auc_members_t{lead_time}.npy"), auc_members)

np.save(os.path.join(output_dir, f"roc_fpr_persistence_t{lead_time}.npy"), FPR_pers)
np.save(os.path.join(output_dir, f"roc_tpr_persistence_t{lead_time}.npy"), TPR_pers)
np.save(os.path.join(output_dir, f"auc_persistence_t{lead_time}.npy"), auc_pers)

print("\nROC arrays saved in:", output_dir)
print(f"AUC ensemble mean: {auc_mean:.4f}")
print(f"AUC persistence: {auc_pers:.4f}")

for seed, auc in zip(seeds, auc_members):
    print(f"AUC seed{seed}: {auc:.4f}")

print(f"AUC member mean ± std: {auc_members.mean():.4f} ± {auc_members.std(ddof=0):.4f}")
print("Done.")