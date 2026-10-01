"""Phase 2b, method B: a GRU conditioned on a learned summary of the person's earlier nights.

Each earlier night (its preprocessed epoch features, no labels) is encoded into a short vector:
a per-epoch linear layer, then the mean and SD over the night, then a linear projection. The
person summary is the average over all of that person's strictly earlier nights. It is appended,
with a has-context flag, to every epoch's input of the staging GRU (``gru.Net``). Everything is
trained end to end on training subjects; a test person contributes only unlabelled nights.

Against memorizing training subjects through the summary: dropout on the summary and, during
training, each earlier night is dropped with probability ``night_drop`` (a night can lose all of
its context, like a first night).

``context_shuffled`` is the control: the same model given another person's nights (``donors``).
"""

from __future__ import annotations

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from sleepwatch.models.gru import IGNORE, GRUConfig, Net, _pad


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder: int = 32
    dim: int = 16
    summary_dropout: float = 0.3
    night_drop: float = 0.3


class ContextNet(nn.Module):
    def __init__(self, n_inputs: int, cfg: GRUConfig, context: ContextConfig):
        super().__init__()
        self.context = context
        self.encode = nn.Sequential(nn.Linear(n_inputs, context.encoder), nn.GELU())
        self.project = nn.Linear(2 * context.encoder, context.dim)
        self.drop = nn.Dropout(context.summary_dropout)
        self.net = Net(n_inputs + context.dim + 1, cfg)

    def summaries(self, cx: torch.Tensor, clen: torch.Tensor) -> torch.Tensor:
        """One vector per context night: projected mean and SD of the encoded epochs."""
        mask = torch.arange(cx.shape[1], device=cx.device)[None] < clen.to(cx.device)[:, None]
        mask = mask.unsqueeze(-1).float()
        h = self.encode(cx) * mask
        count = mask.sum(dim=1).clamp(min=1)
        mean = h.sum(dim=1) / count
        var = (((h - mean[:, None]) * mask) ** 2).sum(dim=1) / count
        return self.project(torch.cat([mean, (var + 1e-6).sqrt()], dim=-1))

    def forward(self, x, lengths, cx, clen, owner):
        n = x.shape[0]
        person = torch.zeros(n, self.context.dim, device=x.device)
        weight = torch.zeros(n, device=x.device)
        if len(owner):
            z = self.summaries(cx, clen)
            keep = torch.ones(len(owner), device=x.device)
            if self.training and self.context.night_drop > 0:
                keep = (torch.rand(len(owner), device=x.device) >= self.context.night_drop).float()
            weight = weight.index_add(0, owner, keep)
            person = person.index_add(0, owner, z * keep[:, None])
            person = person / weight.clamp(min=1)[:, None]
        has = (weight > 0).float()[:, None]
        extra = torch.cat([self.drop(person), has], dim=-1)
        extra = extra[:, None].expand(-1, x.shape[1], -1)
        return self.net(torch.cat([x, extra], dim=-1), lengths)


def collate(items, device):
    """Inputs for a batch of ``(key, x, y, context)`` items; ``context`` is a list of arrays."""
    x, y, lengths = _pad(items)
    context = [c for item in items for c in item[3]]
    owner = torch.tensor([i for i, item in enumerate(items) for _ in item[3]], dtype=torch.long)
    if context:
        cx, _, clen = _pad([(None, c, np.full(len(c), IGNORE)) for c in context])
    else:
        cx, clen = torch.zeros(0, 1, x.shape[2]), torch.zeros(0, dtype=torch.long)
    return (x.to(device), lengths, cx.to(device), clen, owner.to(device)), y.to(device)


def donors(subjects: list[str], seed: int) -> dict[str, str]:
    """A fixed random pairing in which nobody is their own donor."""
    subjects = sorted(subjects)
    if len(subjects) < 2:
        raise ValueError("need at least two subjects to pair")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(subjects))
        if (order != np.arange(len(subjects))).all():
            return {s: subjects[i] for s, i in zip(subjects, order, strict=True)}


def with_context(items, nights: dict, ranks: dict, donor: dict[str, str] | None = None):
    """Attach each night's context: the arrays of the person's strictly earlier nights.

    ``nights`` maps ``(subject, night)`` to its preprocessed array and ``ranks`` maps it to the
    night's rank within the subject. With ``donor``, night ``r`` instead gets the donor's first
    ``r - 1`` nights (or as many as the donor has).
    """
    by_subject: dict[str, list] = {}
    for key in sorted(nights, key=lambda k: (k[0], ranks[k])):
        by_subject.setdefault(key[0], []).append(key)
    out = []
    for key, x, y in items:
        subject, rank = key[0], ranks[key]
        source = by_subject[donor[subject]] if donor else by_subject[subject]
        earlier = source[: rank - 1]
        out.append((key, x, y, [nights[k] for k in earlier]))
    return out
