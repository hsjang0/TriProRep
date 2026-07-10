import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributed as dist
from typing import Optional

from models.backbones.net import Net
from models.backbones.schedulers import esm_style_scheduler
from utils import cknna_score


class ProteinModel(pl.LightningModule):
    """Lightning module for Stage 1 single-chain pretraining."""

    def __init__(
        self,
        max_residues: int = 512,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_dim: int = 384,
        decoder_depth: int = 8,
        decoder_heads: int = 6,
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        total_training_steps: Optional[int] = None,
        pad_token_id: int = 2,
        seq_vocab_size: int = 21,
        bb_vocab_size: int = 21,
        fa_vocab_size: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.total_training_steps = total_training_steps
        self.learning_rate = learning_rate if learning_rate is not None else 4e-4
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.pad_token_id = pad_token_id

        self.test_encoder_embeddings = []
        self.test_boltz_embeddings = []

        self.model = Net(
            max_residues=max_residues,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            seq_vocab_size=seq_vocab_size,
            bb_vocab_size=bb_vocab_size,
            fa_vocab_size=fa_vocab_size,
        )

        self.nparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(
        self,
        input_seq: torch.Tensor,
        input_bb: torch.Tensor,
        attention_mask: torch.Tensor,
        input_fa: Optional[torch.Tensor] = None,
        chain_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_seq.transpose(0, 1),
            input_bb.transpose(0, 1),
            attention_mask,
            input_fa.transpose(0, 1) if input_fa is not None else None,
            chain_ids=chain_ids,
            position_ids=position_ids,
        )

    def _run_step(self, batch):
        """Shared forward + loss computation for train/val."""
        input_seq = batch["input_seq"]
        input_bb = batch["input_bb"]
        input_fa = batch.get("input_fa", None)
        target_seq = batch["target_seq"]
        target_bb = batch["target_bb"]
        target_fa = batch.get("target_fa", None)
        attention_mask = batch["attention_mask"]
        chain_ids = batch.get("chain_ids", None)
        position_ids = batch.get("position_ids", None)
        seqlens = batch.get("seqlens", None)

        logits = self.model(
            input_seq.transpose(0, 1),
            input_bb.transpose(0, 1),
            attention_mask,
            input_fa.transpose(0, 1) if input_fa is not None else None,
            chain_ids=chain_ids,
            position_ids=position_ids,
            seqlens=seqlens,
        )  # [L, B, decoder_dim]

        logits = logits.transpose(0, 1)  # [B, L, decoder_dim]

        pred_logits_seq, pred_logits_bb, pred_logits_fa = self.model.apply_lm_head(logits)

        pred_logits_seq = pred_logits_seq.reshape(-1, pred_logits_seq.size(-1))
        pred_logits_bb = pred_logits_bb.reshape(-1, pred_logits_bb.size(-1))
        if pred_logits_fa is not None:
            pred_logits_fa = pred_logits_fa.reshape(-1, pred_logits_fa.size(-1))

        target_flat_seq = target_seq.reshape(-1)
        target_flat_bb = target_bb.reshape(-1)
        if target_fa is not None:
            target_flat_fa = target_fa.reshape(-1)

        loss_seq = self.criterion(pred_logits_seq, target_flat_seq)
        loss_bb = self.criterion(pred_logits_bb, target_flat_bb)
        if target_fa is not None and pred_logits_fa is not None:
            loss_fa = self.criterion(pred_logits_fa, target_flat_fa)
        else:
            loss_fa = 0.0

        loss = loss_seq + loss_bb + loss_fa
        return loss, loss_seq, loss_bb, loss_fa, pred_logits_seq, pred_logits_bb, pred_logits_fa, target_flat_seq, target_flat_bb

    def training_step(self, batch, batch_idx):
        loss, loss_seq, loss_bb, loss_fa, *_ = self._run_step(batch)
        self.log("train_loss_seq", loss_seq, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss_bb", loss_bb, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss_fa", loss_fa, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("nparams", self.nparams, on_step=True, on_epoch=False, prog_bar=False,
                 logger=True, batch_size=1, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_seq, loss_bb, loss_fa, pred_logits_seq, pred_logits_bb, pred_logits_fa, target_flat_seq, target_flat_bb = self._run_step(batch)

        self.log("val_loss_seq", loss_seq, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_loss_bb", loss_bb, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_loss_fa", loss_fa, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Accuracy
        preds = pred_logits_seq.argmax(dim=-1)
        mask = target_flat_seq != self.pad_token_id
        if mask.sum() > 0:
            accuracy = (preds[mask] == target_flat_seq[mask]).float().mean()
            self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        preds = pred_logits_bb.argmax(dim=-1)
        mask = target_flat_bb != self.pad_token_id
        if mask.sum() > 0:
            accuracy = (preds[mask] == target_flat_bb[mask]).float().mean()
            self.log("val_acc_bb", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        target_seq = batch["target_seq"]
        target_bb = batch["target_bb"]
        target_fa = batch.get("target_fa", None)
        attention_mask = batch["attention_mask"]
        boltz_embeddings = batch["boltz_s"]
        valid_boltz_positions = (boltz_embeddings != 0).any(dim=-1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoder_embeddings = self.model(
                target_seq.transpose(0, 1),
                target_bb.transpose(0, 1),
                attention_mask,
                target_fa.transpose(0, 1) if target_fa is not None else None,
                ret_embeddings=True,
            )

        encoder_embeddings = encoder_embeddings.float()
        encoder_embeddings = (encoder_embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
        encoder_embeddings = encoder_embeddings[valid_boltz_positions]
        boltz_embeddings = boltz_embeddings[valid_boltz_positions]

        self.test_encoder_embeddings.append(encoder_embeddings.detach().cpu())
        self.test_boltz_embeddings.append(boltz_embeddings.detach().cpu())
        return {}
    def _gather_across_ranks(self, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor

        world_size = dist.get_world_size()
        local_count = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.long)
        counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(counts, local_count)
        max_count = int(torch.stack(counts).max().item())

        if max_count == 0:
            return tensor

        pad_rows = max_count - tensor.size(0)
        if pad_rows > 0:
            pad_shape = (pad_rows, *tensor.shape[1:])
            tensor = torch.cat(
                [tensor, torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)],
                dim=0,
            )

        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)

        pieces = [g[: int(c.item())] for g, c in zip(gathered, counts)]
        return torch.cat(pieces, dim=0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = esm_style_scheduler(
            optimizer,
            trainer=self.trainer,
            explicit_total_steps=self.total_training_steps,
            warmup_steps=self.warmup_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
