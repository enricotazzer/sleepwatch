"""Phase 2b, method C: fine-tune the population GRU on one person's labelled earlier nights.

This uses the earlier nights' expert labels, so it is an upper bound (a real user has no EEG).
A copy of the population model is trained on the person's nights as one full batch for a fixed
number of epochs (there is nothing to early-stop on), with an L2 penalty that pulls the weights
back toward the population model instead of weight decay. A stage missing from those nights is
no problem: the model keeps predicting all five stages, and the penalty limits the drift.

The grid point (learning rate, epochs, trained layers) is chosen per fold by running the same
procedure on the fold's validation subjects, whom the population GRU never fitted on.
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from sleepwatch.models.gru import IGNORE, N_CLASSES, GRUConfig, Net, collate


class FinetuneSetting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float
    epochs: int
    layers: Literal["output", "all"]


class FinetuneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid: list[FinetuneSetting] = [
        FinetuneSetting(lr=lr, epochs=epochs, layers=layers)
        for lr in (1e-4, 1e-3)
        for epochs in (5, 20)
        for layers in ("output", "all")
    ]
    l2_to_population: float = 0.01


def finetune(
    model: Net, items, setting: FinetuneSetting, cfg: FinetuneConfig, gru: GRUConfig, seed: int
) -> Net:
    """A fine-tuned copy of ``model``; ``model`` itself is left unchanged."""
    tuned = copy.deepcopy(model)
    torch.manual_seed(seed)
    trainable = tuned.out if setting.layers == "output" else tuned
    for param in tuned.parameters():
        param.requires_grad_(False)
    params = dict(trainable.named_parameters())
    for param in params.values():
        param.requires_grad_(True)
    anchor = {name: param.detach().clone() for name, param in params.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=setting.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    device = next(tuned.parameters()).device
    inputs, y = collate(items, device)
    tuned.train()
    for _ in range(setting.epochs):
        optimizer.zero_grad()
        loss = loss_fn(tuned(*inputs).reshape(-1, N_CLASSES), y.reshape(-1))
        penalty = sum(((p - anchor[name]) ** 2).sum() for name, p in params.items())
        (loss + cfg.l2_to_population * penalty).backward()
        nn.utils.clip_grad_norm_(params.values(), gru.clip)
        optimizer.step()
    tuned.eval()
    return tuned
