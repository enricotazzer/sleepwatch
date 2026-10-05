"""Structured summary of a flagged night: the input for the Phase 4 follow-up agent.

It says when, in which sleep stages and by how much a night differed from the person's own
earlier nights. It carries no interpretation and no cause.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from sleepwatch.constants import EPOCH_S

DISCLAIMER = (
    "Research prototype, not a medical device. A flag means the night differed from this "
    "person's earlier nights; it does not say why."
)


class ChannelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str  # hr_window, hr_night, frag or onset
    value: float  # z for hr_*; z against earlier nights for frag and onset
    tail_probability: float
    fired: bool


class AnomalySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    night: int
    date_local: str  # date the recording started (US Eastern wall clock)
    baseline_nights: int = Field(ge=0)
    top_channel: str
    channels: list[ChannelResult]
    window_start_local: str | None = None  # highest-HR 30-min window (clock time)
    window_end_local: str | None = None
    window_hours_since_onset: float | None = None
    window_stage_share: dict[str, float] = {}  # predicted stages in that window
    hr_excess_bpm: float | None = None  # mean HR above the personal expectation in the window
    wake_bouts_per_hour: float | None = None
    onset_latency_min: float | None = None
    data_quality: list[str] = []
    disclaimer: str = DISCLAIMER


def clock(rec_start_local: str, epoch: int) -> str:
    """Wall-clock time of an epoch's start, from the recording's local start time."""
    start = datetime.strptime(rec_start_local, "%Y-%m-%d %H:%M:%S")
    return (start + timedelta(seconds=EPOCH_S * epoch)).strftime("%Y-%m-%d %H:%M")
