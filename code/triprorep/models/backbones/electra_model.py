import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional

from models.backbones.net import Net
from models.backbones.schedulers import esm_style_scheduler


class ELECTRAProteinModel(pl.LightningModule):
    """ELECTRA-style replaced token detection for multi-modal protein pretraining.

    A small generator produces plausible replacement tokens at masked positions.
    The full-size discriminator classifies every position as real or replaced.
    Generator and discriminator share embedding tables.
    After pretraining, only the discriminator encoder is kept.
    """

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
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,  # override to 0.95 at scale (Qwen/Llama/PaLM default)
        warmup_steps: int = 2000,
        total_training_steps: Optional[int] = None,
        pad_token_id: int = 2,
        seq_vocab_size: int = 21,
        bb_vocab_size: int = 21,
        fa_vocab_size: int = 0,
        # ELECTRA-specific
        generator_depth: int = 4,
        generator_decoder_depth: int = 2,
        disc_loss_weight: float = 50.0,
        generator_lr_multiplier: float = 1.0,
        generator_temperature: float = 1.0,
        exclude_original_in_sampling: bool = True,
        # Stability (recommended for 3B+ to avoid attention-logit blowup)
        use_qk_norm: bool = False,
        attn_logit_softcap: float = 0.0,  # Gemma-style soft cap (e.g. 50.0); 0 disables
        # Attention temperature multiplier on Q (default None → 1/sqrt(head_dim)).
        relax_temperature_scaling: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.total_training_steps = total_training_steps
        self.learning_rate = learning_rate if learning_rate is not None else 4e-4
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.warmup_steps = warmup_steps
        self.pad_token_id = pad_token_id
        self.disc_loss_weight = disc_loss_weight
        self.generator_lr_multiplier = generator_lr_multiplier
        self.generator_temperature = generator_temperature
        self.exclude_original_in_sampling = exclude_original_in_sampling
        self.mask_token_id = pad_token_id + 1  # 3

        # --- Discriminator (full model) ---
        self.discriminator = Net(
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
            use_qk_norm=use_qk_norm,
            attn_logit_softcap=attn_logit_softcap,
            relax_temperature_scaling=relax_temperature_scaling,
        )

        # --- Generator (small model, shared embeddings) ---
        self.generator = Net(
            max_residues=max_residues,
            embed_dim=embed_dim,
            encoder_depth=generator_depth,
            encoder_heads=encoder_heads,
            decoder_dim=decoder_dim,
            decoder_depth=generator_decoder_depth,
            decoder_heads=decoder_heads,
            seq_vocab_size=seq_vocab_size,
            bb_vocab_size=bb_vocab_size,
            fa_vocab_size=fa_vocab_size,
            use_qk_norm=use_qk_norm,
            attn_logit_softcap=attn_logit_softcap,
            relax_temperature_scaling=relax_temperature_scaling,
        )
        # Share embedding tables: generator uses discriminator's embeddings
        self.generator.seq_embedding = self.discriminator.seq_embedding
        self.generator.bb_embedding = self.discriminator.bb_embedding
        if fa_vocab_size > 4:
            self.generator.fa_embedding = self.discriminator.fa_embedding
        self.generator.fuse_layer = self.discriminator.fuse_layer
        self.generator.chain_embedding = self.discriminator.chain_embedding

        # Discriminator's MLM heads are ACTIVE — used for corrective MLM objective
        # (predict original tokens from corrupted input at ALL positions)
        self.fa_vocab_size = fa_vocab_size

        # --- Losses ---
        self.criterion_mlm = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        # Corrective MLM: predict original tokens at ALL non-pad positions
        self.criterion_corrective = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        self.nparams_disc = sum(
            p.numel() for p in self.discriminator.parameters() if p.requires_grad
        )
        # Generator-only params (exclude shared embeddings)
        shared_ids = {id(p) for p in self.discriminator.parameters()}
        self.nparams_gen_only = sum(
            p.numel() for p in self.generator.parameters()
            if p.requires_grad and id(p) not in shared_ids
        )
        print(
            f"[ELECTRA] disc_params={self.nparams_disc:,}, "
            f"gen_only_params={self.nparams_gen_only:,}, "
            f"disc_loss_weight={disc_loss_weight}, "
            f"generator_depth={generator_depth}/{generator_decoder_depth}, "
            f"generator_temperature={generator_temperature}"
        )

    def _sample_excluding_original(
        self,
        original: torch.Tensor,
        gen_logits: torch.Tensor,
        mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        """Replace masked positions with samples from the generator.

        If self.exclude_original_in_sampling is True, the original token's
        logit is set to -inf before sampling, guaranteeing every replacement
        differs from the original (replaced_frac == masking probability exactly).
        If False, vanilla ELECTRA behavior: sample from full distribution;
        the generator may reproduce the original token, in which case the
        discriminator sees an uncorrupted position.

        Args:
            original: [B, L] original token IDs
            gen_logits: [B, L, V] generator logits
            mask_bool: [B, L] True at masked positions

        Returns:
            replaced: [B, L] tokens sampled at masked positions
        """
        replaced = original.clone()
        if mask_bool.any():
            logits = gen_logits[mask_bool].clone()  # [N_masked, V]
            if self.exclude_original_in_sampling:
                orig_ids = original[mask_bool]  # [N_masked]
                logits[torch.arange(len(orig_ids), device=logits.device), orig_ids] = float("-inf")
            probs = F.softmax(logits.float() / self.generator_temperature, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            replaced[mask_bool] = sampled
        return replaced

    def forward(
        self,
        input_seq: torch.Tensor,
        input_bb: torch.Tensor,
        attention_mask: torch.Tensor,
        input_fa: Optional[torch.Tensor] = None,
        chain_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        ret_embeddings: bool = False,
        ret_hidden_at: Optional[int] = None,
    ):
        """Forward through discriminator encoder (for downstream use)."""
        return self.discriminator(
            input_seq.transpose(0, 1),
            input_bb.transpose(0, 1),
            attention_mask,
            input_fa.transpose(0, 1) if input_fa is not None else None,
            chain_ids=chain_ids,
            position_ids=position_ids,
            ret_embeddings=ret_embeddings,
            ret_hidden_at=ret_hidden_at,
        )

    def training_step(self, batch, batch_idx):
        input_seq = batch["input_seq"]       # [B, L] masked
        input_bb = batch["input_bb"]         # [B, L] masked
        input_fa = batch.get("input_fa", None)
        target_seq = batch["target_seq"]     # [B, L] original tokens at masked pos, pad elsewhere
        target_bb = batch["target_bb"]
        target_fa = batch.get("target_fa", None)
        original_seq = batch["original_seq"]  # [B, L] fully original (no masking, no padding)
        original_bb = batch["original_bb"]
        original_fa = batch.get("original_fa", None)
        attention_mask = batch["attention_mask"]
        chain_ids = batch.get("chain_ids", None)
        position_ids = batch.get("position_ids", None)
        seqlens = batch.get("seqlens", None)
        # Per-modality independent masks
        mask_seq = batch["mask_seq"]         # [B, L] True at seq-masked positions
        mask_bb = batch["mask_bb"]           # [B, L] True at bb-masked positions
        mask_fa = batch.get("mask_fa", None) # [B, L] True at fa-masked positions

        # ---- 1. Generator forward on masked input ----
        gen_out = self.generator(
            input_seq.transpose(0, 1),
            input_bb.transpose(0, 1),
            attention_mask,
            input_fa.transpose(0, 1) if input_fa is not None else None,
            chain_ids=chain_ids,
            position_ids=position_ids,
            seqlens=seqlens,
        )  # [L, B, decoder_dim]
        gen_out = gen_out.transpose(0, 1)  # [B, L, decoder_dim]
        gen_logits_seq, gen_logits_bb, gen_logits_fa = self.generator.apply_lm_head(gen_out)

        # Generator MLM loss (only at masked positions per modality)
        loss_gen_seq = self.criterion_mlm(
            gen_logits_seq.reshape(-1, gen_logits_seq.size(-1)),
            target_seq.reshape(-1),
        )
        loss_gen_bb = self.criterion_mlm(
            gen_logits_bb.reshape(-1, gen_logits_bb.size(-1)),
            target_bb.reshape(-1),
        )
        loss_gen_fa = torch.tensor(0.0, device=input_seq.device)
        if target_fa is not None and gen_logits_fa is not None:
            loss_gen_fa = self.criterion_mlm(
                gen_logits_fa.reshape(-1, gen_logits_fa.size(-1)),
                target_fa.reshape(-1),
            )
        loss_gen = loss_gen_seq + loss_gen_bb + loss_gen_fa

        # ---- 2. Sample guaranteed-different replacements (no gradient) ----
        with torch.no_grad():
            replaced_seq = self._sample_excluding_original(original_seq, gen_logits_seq, mask_seq)
            replaced_bb = self._sample_excluding_original(original_bb, gen_logits_bb, mask_bb)
            replaced_fa = None
            if original_fa is not None and gen_logits_fa is not None and mask_fa is not None:
                replaced_fa = self._sample_excluding_original(original_fa, gen_logits_fa, mask_fa)

        # ---- 3. Discriminator forward on corrupted input ----
        disc_out = self.discriminator(
            replaced_seq.transpose(0, 1),
            replaced_bb.transpose(0, 1),
            attention_mask,
            replaced_fa.transpose(0, 1) if replaced_fa is not None else None,
            chain_ids=chain_ids,
            position_ids=position_ids,
            seqlens=seqlens,
        ).transpose(0, 1)  # [B, L, decoder_dim]

        # ---- 4. Corrective MLM: predict original tokens at ALL positions ----
        corr_logits_seq, corr_logits_bb, corr_logits_fa = self.discriminator.apply_lm_head(disc_out)

        # Targets: original tokens everywhere, pad at padding positions
        corr_target_seq = original_seq.clone()
        corr_target_bb = original_bb.clone()
        corr_target_seq[~attention_mask] = self.pad_token_id
        corr_target_bb[~attention_mask] = self.pad_token_id

        loss_corr_seq = self.criterion_corrective(
            corr_logits_seq.reshape(-1, corr_logits_seq.size(-1)),
            corr_target_seq.reshape(-1),
        )
        loss_corr_bb = self.criterion_corrective(
            corr_logits_bb.reshape(-1, corr_logits_bb.size(-1)),
            corr_target_bb.reshape(-1),
        )
        loss_corr_fa = torch.tensor(0.0, device=input_seq.device)
        if original_fa is not None and corr_logits_fa is not None:
            corr_target_fa = original_fa.clone()
            corr_target_fa[~attention_mask] = self.pad_token_id
            loss_corr_fa = self.criterion_corrective(
                corr_logits_fa.reshape(-1, corr_logits_fa.size(-1)),
                corr_target_fa.reshape(-1),
            )
        loss_corr = loss_corr_seq + loss_corr_bb + loss_corr_fa

        # ---- 5. Total loss ----
        loss = loss_gen + self.disc_loss_weight * loss_corr

        # ---- Metrics ----
        with torch.no_grad():
            valid_mask = attention_mask.float()
            pred_seq_ids = corr_logits_seq.argmax(dim=-1)
            # Overall corrective accuracy
            corr_acc_seq = ((pred_seq_ids == original_seq).float() * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            # Accuracy at corrupted positions specifically
            if mask_seq.any():
                corr_acc_at_replaced = (pred_seq_ids[mask_seq] == original_seq[mask_seq]).float().mean()
            else:
                corr_acc_at_replaced = torch.tensor(0.0)
            # Binary disc accuracy (monitoring only)
            labels_seq = (replaced_seq != original_seq).float()
            replaced_frac = labels_seq[attention_mask.bool()].mean() if attention_mask.any() else torch.tensor(0.0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_gen", loss_gen, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss_corr", loss_corr, on_step=True, on_epoch=True, prog_bar=False)
        self.log("corr_acc_seq", corr_acc_seq, on_step=True, prog_bar=True)
        self.log("corr_acc_at_replaced", corr_acc_at_replaced, on_step=True, prog_bar=False)
        self.log("replaced_frac", replaced_frac, on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        input_seq = batch["input_seq"]
        input_bb = batch["input_bb"]
        input_fa = batch.get("input_fa", None)
        target_seq = batch["target_seq"]
        target_bb = batch["target_bb"]
        target_fa = batch.get("target_fa", None)
        original_seq = batch["original_seq"]
        original_bb = batch["original_bb"]
        original_fa = batch.get("original_fa", None)
        attention_mask = batch["attention_mask"]
        chain_ids = batch.get("chain_ids", None)
        position_ids = batch.get("position_ids", None)
        seqlens = batch.get("seqlens", None)
        mask_seq = batch["mask_seq"]
        mask_bb = batch["mask_bb"]
        mask_fa = batch.get("mask_fa", None)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Generator
            gen_out = self.generator(
                input_seq.transpose(0, 1),
                input_bb.transpose(0, 1),
                attention_mask,
                input_fa.transpose(0, 1) if input_fa is not None else None,
                chain_ids=chain_ids,
                position_ids=position_ids,
                seqlens=seqlens,
            ).transpose(0, 1)
            gen_logits_seq, gen_logits_bb, gen_logits_fa = self.generator.apply_lm_head(gen_out)

            # Sample guaranteed-different replacements
            replaced_seq = self._sample_excluding_original(original_seq, gen_logits_seq, mask_seq)
            replaced_bb = self._sample_excluding_original(original_bb, gen_logits_bb, mask_bb)
            replaced_fa = None
            if original_fa is not None and gen_logits_fa is not None and mask_fa is not None:
                replaced_fa = self._sample_excluding_original(original_fa, gen_logits_fa, mask_fa)

            # Discriminator corrective MLM
            disc_out = self.discriminator(
                replaced_seq.transpose(0, 1),
                replaced_bb.transpose(0, 1),
                attention_mask,
                replaced_fa.transpose(0, 1) if replaced_fa is not None else None,
                chain_ids=chain_ids,
                position_ids=position_ids,
                seqlens=seqlens,
            ).transpose(0, 1)

        corr_logits_seq, _, _ = self.discriminator.apply_lm_head(disc_out)
        corr_target_seq = original_seq.clone()
        corr_target_seq[~attention_mask] = self.pad_token_id
        loss_corr = self.criterion_corrective(
            corr_logits_seq.reshape(-1, corr_logits_seq.size(-1)),
            corr_target_seq.reshape(-1),
        )
        pred_seq_ids = corr_logits_seq.argmax(dim=-1)
        valid_mask = attention_mask.float()
        corr_acc = ((pred_seq_ids == original_seq).float() * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        self.log("val_loss_corr", loss_corr, on_epoch=True, sync_dist=True)
        self.log("val_corr_acc", corr_acc, on_epoch=True, sync_dist=True)
    def configure_optimizers(self):
        # Separate param groups: discriminator + shared embeddings at base LR,
        # generator-only params at multiplied LR
        shared_ids = {id(p) for p in self.discriminator.parameters()}

        disc_params = list(self.discriminator.parameters())

        gen_only_params = [
            p for p in self.generator.parameters()
            if p.requires_grad and id(p) not in shared_ids
        ]

        param_groups = [
            {"params": disc_params, "lr": self.learning_rate},
            {"params": gen_only_params, "lr": self.learning_rate * self.generator_lr_multiplier},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.weight_decay,
        )
        print(f"[ELECTRA] AdamW betas=({self.adam_beta1}, {self.adam_beta2}), weight_decay={self.weight_decay}")
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
