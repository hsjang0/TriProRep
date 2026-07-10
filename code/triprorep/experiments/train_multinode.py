import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import argparse
import torch
from models import build_backbone
from dataloader import build_datamodule


class CompileModelCallback(Callback):
    """Compile inner model(s) with torch.compile (dynamic=False, static shapes).

    Static shapes are guaranteed by the dataloader (fixed seq/patch lengths)
    so no recompilation is needed. EC linear probing was removed, which was
    the only path that required per-layer / dynamic=True compile.
    """

    def __init__(self, attrs=None):
        super().__init__()
        self.attrs = attrs or ["model"]

    def _compile_attr(self, pl_module, attr):
        inner = getattr(pl_module, attr, None)
        if inner is None or hasattr(inner, "_orig_mod"):
            return
        setattr(pl_module, attr, torch.compile(inner, dynamic=False))

    def on_fit_start(self, trainer, pl_module):
        # ELECTRA pretraining (has both discriminator and generator).
        if hasattr(pl_module, "discriminator") and hasattr(pl_module, "generator"):
            self._compile_attr(pl_module, "discriminator")
            self._compile_attr(pl_module, "generator")
            return
        # Indep-apo + ELECTRA self-sup variant: main encoder/decoder under
        # self.model + a separate generator.
        if hasattr(pl_module, "model") and hasattr(pl_module, "generator"):
            self._compile_attr(pl_module, "model")
            self._compile_attr(pl_module, "generator")
            return
        # Default: compile self.model (or whatever attrs were configured).
        for attr in self.attrs:
            self._compile_attr(pl_module, attr)


def main():
    parser = argparse.ArgumentParser(description="Train protein structure model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    
    args = parser.parse_args()
    
    if args.config:
        cfg = OmegaConf.load(args.config)
        if args.overrides:
            overrides = OmegaConf.from_dotlist(args.overrides)
            cfg = OmegaConf.merge(cfg, overrides)
    else:
        cfg = OmegaConf.create({})

    model_cfg = OmegaConf.to_container(cfg.get("model", {}), resolve=True)
    training_cfg = OmegaConf.to_container(cfg.get("training", {}), resolve=True)
    data_cfg = OmegaConf.to_container(cfg.get("data", {}), resolve=True)
    lightning_cfg = OmegaConf.to_container(cfg.get("lightning", {}), resolve=True)
    
    wandb_name = lightning_cfg.get("wandb_name")
    wandb_dir = lightning_cfg.get("wandb_dir")
    os.makedirs(wandb_dir, exist_ok=True)
    
    data_name = data_cfg.get("name", "ProteinDataModule")
    data_params = {k: v for k, v in data_cfg.items() if k != "name"}
    data_params.update({
        "batch_size": training_cfg.get("batch_size"),
        "num_workers": training_cfg.get("num_workers"),
    })
    datamodule = build_datamodule(data_name, **data_params)
    
    backbone_name = model_cfg.get("name", "ProteinModel")
    backbone_params = {k: v for k, v in model_cfg.items() if k != "name"}
    backbone_args = {
        **backbone_params,
        "learning_rate": training_cfg.get("learning_rate"),
        "weight_decay": training_cfg.get("weight_decay"),
        "warmup_steps": training_cfg.get("warmup_steps"),
        "total_training_steps": training_cfg.get("total_training_steps", None),
        "pad_token_id": datamodule.pad_token_id,
        "seq_vocab_size": datamodule.get_seq_vocab_size(),
        "bb_vocab_size": datamodule.get_bb_vocab_size(),
        "fa_vocab_size": datamodule.get_fa_vocab_size(),
    }
    # Optional optimizer betas (Qwen/Llama/PaLM use β2=0.95 at scale)
    if "adam_beta1" in training_cfg:
        backbone_args["adam_beta1"] = training_cfg["adam_beta1"]
    if "adam_beta2" in training_cfg:
        backbone_args["adam_beta2"] = training_cfg["adam_beta2"]
    model = build_backbone(backbone_name, **backbone_args)


    checkpoint_callback = ModelCheckpoint(
        dirpath=lightning_cfg.get("checkpoint_dir"),
        filename="step={step}",
        every_n_train_steps=lightning_cfg.get("save_every_n_steps"),
        save_top_k=-1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    # Conditional torch.compile
    if lightning_cfg.get("compile", False):
        callbacks.append(CompileModelCallback())

    # Logger
    logger = WandbLogger(
        project=lightning_cfg.get("wandb_project", "PLM_after325"),
        name=wandb_name,
        entity="ahngroup",
        save_dir=wandb_dir
    )
    
    # Trainer
    trainer_kwargs = {
        "max_steps": training_cfg.get("max_steps", training_cfg.get("total_training_steps", -1)),
        "max_epochs": -1,
        "devices": lightning_cfg.get("gpus"),
        "accelerator": "gpu",
        "precision": lightning_cfg.get("precision"),
        "accumulate_grad_batches": training_cfg.get("accumulate_grad_batches"),
        "callbacks": callbacks,
        "logger": logger,
        "strategy": lightning_cfg.get("strategy", "ddp"),
        "gradient_clip_val": training_cfg.get("gradient_clip_val"),
        "gradient_clip_algorithm": "norm",
        "num_nodes": lightning_cfg.get("nodes", 1),
        #"compile": True
    }

    # TokenBatchSampler already shards across ranks internally — tell Lightning
    # not to wrap the custom batch_sampler with DistributedSampler.
    if data_cfg.get("dynamic_batch_tokens") is not None:
        trainer_kwargs["use_distributed_sampler"] = False
    
    # Disable validation loop only if explicitly requested via config.
    if lightning_cfg.get("disable_val_loop", False):
        trainer_kwargs["limit_val_batches"] = 0



    ckpt_path = lightning_cfg.get("resume_from_model_only", None)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        # Strip _orig_mod. prefix from compiled model checkpoints
        # + remap legacy representation head keys
        state_dict = {
            k.replace("_orig_mod.", "").replace("rep_proj.", "rep_loss_fn.proj."): v
            for k, v in state_dict.items()
        }

        # Remap ELECTRA discriminator → target model prefix.
        # ELECTRA: discriminator.encoder.* → ComplexIndepApoModel: model.encoder.*
        # ELECTRA: discriminator.decoder.* → model.decoder.*  etc.
        target_keys = set(model.state_dict().keys())
        has_discriminator = any(k.startswith("discriminator.") for k in state_dict)
        has_model_prefix = any(k.startswith("model.") for k in target_keys)
        if has_discriminator and has_model_prefix:
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("discriminator."):
                    new_k = "model." + k[len("discriminator."):]
                    remapped[new_k] = v
                else:
                    remapped[k] = v
            state_dict = remapped
            print(f"Remapped {sum(1 for k in remapped if k.startswith('model.'))} "
                  f"keys from discriminator.* → model.*")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Loaded weights only.")
        if missing:
            print(f"missing: {len(missing)} keys (showing 10): {missing[:10]}")
        if unexpected:
            print(f"unexpected: {len(unexpected)} keys (showing 10): {unexpected[:10]}")
        # Report how many keys were loaded per top-level prefix so we can
        # see at a glance whether model/generator/etc. got their weights.
        loaded_prefixes = {}
        loaded_keys = set(state_dict.keys()) - set(missing)
        for k in loaded_keys:
            pfx = k.split(".", 1)[0]
            loaded_prefixes[pfx] = loaded_prefixes.get(pfx, 0) + 1
        print(f"loaded by prefix: {loaded_prefixes}")

    # Patch on_load_checkpoint to strip _orig_mod. prefix from compiled model checkpoints
    _orig_on_load = model.on_load_checkpoint
    def _patched_on_load(checkpoint):
        sd = checkpoint.get("state_dict", {})
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        checkpoint["state_dict"] = cleaned
        _orig_on_load(checkpoint)
    model.on_load_checkpoint = _patched_on_load

    trainer = pl.Trainer(**trainer_kwargs)
    print("Start training,")
    trainer.fit(model, datamodule, ckpt_path=lightning_cfg.get("resume_from_checkpoint"))


if __name__ == "__main__":
    main()
