import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_fss(preds, targets, window=9):
    pool = nn.AvgPool2d(window, 1, window//2)
    p = pool(preds)
    t = pool(targets)
    mse = torch.mean((p - t)**2)
    ref = torch.mean(p**2) + torch.mean(t**2)
    return (1 - mse/(ref + 1e-8)).clamp(0.0, 1.0)

class SpatiallyEnhancedLoss(nn.Module):
    def __init__(self, window_size=9, pos_weight=25.0, alpha=0.3):
        super().__init__()
        self.pool = nn.AvgPool2d(window_size, 1, window_size//2)
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = self.bce(logits, targets)
        fss = F.mse_loss(self.pool(probs), self.pool(targets))
        return self.alpha*bce + (1 - self.alpha)*fss

class ShardDataset(IterableDataset):
    def __init__(self, shard_dir, split_by_rank=True, split_by_worker=True):
        super().__init__()
        self.shard_dir = shard_dir
        self.split_by_rank = split_by_rank
        self.split_by_worker = split_by_worker
        self.files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))
        if not self.files:
            raise RuntimeError("No shards found")

    def _rankinfo(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __iter__(self):
        rank, world = self._rankinfo()
        worker = torch.utils.data.get_worker_info()

        files = list(self.files)

        if self.split_by_rank and world > 1:
            n = len(files)
            size = max(1, n // world)
            files = files[rank * size:(rank + 1) * size]

        if worker and self.split_by_worker:
            files = files[worker.id::worker.num_workers]

        while True:
            random.shuffle(files)
            for f in files:
                d = torch.load(os.path.join(self.shard_dir, f), map_location="cpu")
                X, Y = d["X"], d["Y"]
                for i in range(X.shape[0]):
                    Xi = X[i].float()
                    Yi = Y[i].unsqueeze(0).float()   # correct shape
                    yield Xi, Yi

class SimpleDecoder(nn.Module):
    def __init__(self, embed_dim, out_hw=(512, 512), dropout_p=0.2):
        super().__init__()
        self.out_hw = out_hw
        ch = [embed_dim, 512, 256, 128, 64, 32]
        layers = []
        for c1, c2 in zip(ch[:-1], ch[1:]):
            layers += [
                nn.ConvTranspose2d(c1, c2, 4, 2, 1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_p)
            ]
        self.up = nn.Sequential(*layers)
        self.final = nn.Conv2d(ch[-1], 1, 1)

    def forward(self, x):
        x = self.up(x)
        if x.shape[-2:] != self.out_hw:
            x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.final(x)

class Core2MapModel(pl.LightningModule):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4,
                 lr=1e-4, dropout_p=0.2, pos_weight=25.0, alpha=0.3):
        super().__init__()
        self.save_hyperparameters()

        self.in_proj = nn.Sequential(
            nn.Linear(13, embed_dim),
            nn.Dropout(dropout_p)
        )

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.map_proj = nn.Linear(embed_dim, embed_dim*16*16)
        self.decoder = SimpleDecoder(embed_dim,
                                     out_hw=(512, 512),
                                     dropout_p=dropout_p)

        self.criterion = SpatiallyEnhancedLoss(window_size=9,
                                               pos_weight=pos_weight,
                                               alpha=alpha)
        self.val_auc = BinaryAUROC()
        self.mask_col = 12

    def forward(self, x):
        b, t, c, f = x.shape
        mask = (x[..., self.mask_col] <= 0)
        x = x.view(b, t*c, f)
        mask = mask.view(b, t*c)

        x = self.in_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        valid = (~mask).float().unsqueeze(-1)
        pooled = (x*valid).sum(1) / valid.sum(1).clamp_min(1.0)

        z = self.map_proj(pooled).view(b, -1, 16, 16)
        return self.decoder(z)

    def training_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        x, y = (t.to(self.device) for t in batch)
        preds = torch.sigmoid(self(x))
        for w in [3, 5, 9]:
            self.log(f"val_fss_{w}", compute_fss(preds, y, w),
                     on_epoch=True, prog_bar=(w == 9), sync_dist=True)
        self.val_auc.update(preds.flatten(), y.flatten().int())

    def on_validation_epoch_end(self):
        self.log("val_auc", self.val_auc.compute(), prog_bar=True, sync_dist=True)
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.hparams.lr)

def main():
    torch.set_float32_matmul_precision("high")

    lead = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    set_seed(seed)
    pl.seed_everything(seed, workers=True)

    base = f"/work/scratch-nopw2/mendrika/OB/preprocessed/t{lead}"
    train_dir = f"{base}/train_t{lead}"
    val_dir = f"{base}/val_t{lead}"

    ckpt = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/checkpoints/WS/transformer/t{lead}/seed{seed}"
    os.makedirs(ckpt, exist_ok=True)

    train_dl = DataLoader(
        ShardDataset(train_dir, True, True),
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

    val_dl = DataLoader(
        ShardDataset(val_dir, split_by_rank=False, split_by_worker=True),
        batch_size=1,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )

    model = Core2MapModel()

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu",
        devices=4,
        strategy="ddp",
        precision="16-mixed",
        logger=WandbLogger(project="WS_transformer",
                           name=f"t{lead}_seed{seed}"),
        log_every_n_steps=5,
        limit_val_batches=300,
        limit_train_batches=3000,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt,
                filename="best-core2map",
                monitor="val_auc",
                mode="max",
                save_top_k=1
            ),
            EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=5,
                min_delta=0.001
            )
        ]
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"Training complete for seed {seed}")

if __name__ == "__main__":
    main()
