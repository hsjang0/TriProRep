import torch
from typing import Optional


def compute_total_training_steps(
    trainer,
    explicit_total_steps: Optional[int],
    fallback_steps: int = 500_000,
) -> int:
    """
    Resolve total training steps with precedence:
    1) If trainer has an epoch-based plan, use its estimated stepping batches.
    2) Otherwise, if trainer.max_steps is set, use that.
    3) Otherwise use explicit_total_steps if provided.
    4) Fallback to fallback_steps.
    """
    if trainer is not None:
        estimated = getattr(trainer, "estimated_stepping_batches", None)
        if estimated is not None and estimated > 0:
            return int(estimated)

        max_steps = getattr(trainer, "max_steps", None)
        if max_steps is not None and max_steps > 0:
            return int(max_steps)

    if explicit_total_steps is not None:
        return max(int(explicit_total_steps), 1)

    return max(int(fallback_steps), 1)


def esm_style_scheduler(
    optimizer: torch.optim.Optimizer,
    trainer,
    explicit_total_steps: Optional[int],
    warmup_steps: int = 2000,
    end_lr_scale: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    ESM-style learning-rate schedule.
    Epoch count overrides step count: if trainer knows epochs, we derive steps/epoch.
    - Linear warmup to peak over `warmup_steps`.
    - Linear decay to `end_lr_scale` of the peak across 90% of total steps.
    - Flat at `end_lr_scale` afterwards.
    """
    total_steps = compute_total_training_steps(trainer, explicit_total_steps)
    warmup_steps = max(int(warmup_steps), 1)
    decay_end_step = int(0.9 * total_steps)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / float(warmup_steps)
        if step <= decay_end_step:
            decay_progress = (step - warmup_steps) / max(decay_end_step - warmup_steps, 1)
            return 1.0 - decay_progress * (1.0 - end_lr_scale)
        return end_lr_scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

