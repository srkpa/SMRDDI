from typing import Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from packaging import version
from torch import optim
from torchmetrics import (
    Accuracy,
    MetricCollection,
    Precision,
    Recall,
    F1Score,
    AveragePrecision,
    AUROC,
)

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")


class SiamNet(pl.LightningModule):
    def __init__(
        self,
        base_network,
        projection_head,
        lr: float,
        weight_decay,
        max_epochs=100,
        num_classes: int = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.base_network = base_network
        self.projection_head = projection_head
        self.metrics = MetricCollection(
            {
                "acc": Accuracy(),
                "macro_precision": Precision(num_classes=num_classes, average="macro"),
                "macro_recall": Recall(num_classes=num_classes, average="macro"),
                "macro_F1": F1Score(num_classes=num_classes, average="macro"),
                "weighted_recall": Recall(num_classes=num_classes, average="weighted"),
                "weighted_F1": F1Score(num_classes=num_classes, average="weighted"),
                "weighted_precision": Precision(
                    num_classes=num_classes, average="weighted"
                ),
                "AUPRC": AveragePrecision(num_classes=num_classes),
                "AUROC": AUROC(num_classes=num_classes),
            },
            prefix="_",
        )

    def forward(self, data) -> torch.Tensor:
        data = torch.cat(data, dim=0)
        feats = torch.hstack(torch.split(self.base_network(data), data.shape[0] // 2))
        pred = self.projection_head(feats)
        return F.log_softmax(pred, dim=1)

    def _calculate_loss(self, batch, mode="train"):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log_dict(
            {f"{mode}_loss": loss, f"{mode}_acc": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._calculate_loss(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._calculate_loss(batch, "val")

    def test_step(self, batch, batch_idx, test_idx=0):
        # self._calculate_loss(batch, mode="test")
        data, target = batch
        output = self(data)
        self.metrics.update(output, target)

    def test_epoch_end(self, outputs):
        self.log_dict(self.metrics.compute())
        self.metrics.reset()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(self.hparams.max_epochs * 0.6),
                int(self.hparams.max_epochs * 0.8),
            ],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]


class SimCLR(pl.LightningModule):
    def __init__(
        self,
        base_network,
        projection_head,
        lr,
        temperature,
        weight_decay,
        max_epochs=500,
        save_dir=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"
        self.base_network = base_network
        self.projection_head = projection_head
        self.backup = save_dir

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train", sync=True):
        mols, _ = batch
        mols = torch.cat(mols, dim=0)
        # Encode all drugs
        feats = self.projection_head(self.base_network(mols))
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        # Logging loss
        self.log(f"{mode}_loss", nll, sync_dist=sync)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],
                cos_sim.masked_fill(pos_mask, -9e15),
            ],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(f"{mode}_acc_top1", (sim_argsort == 0).float().mean(), sync_dist=sync)
        self.log(f"{mode}_acc_top5", (sim_argsort < 5).float().mean(), sync_dist=sync)
        self.log(f"{mode}_acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=sync)

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        x, _ = batch
        x = torch.cat(x, dim=0)
        return self.base_network(x)
