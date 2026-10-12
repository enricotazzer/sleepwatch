"""Plain-language text the person sees: the observed facts, the opening and the closing.

All of it is written by code from the ``AnomalySummary``, so numbers are never left to the LLM.
It says what differed from the person's earlier nights, never why (docs/decisions.md 49).
"""

from __future__ import annotations

from datetime import date

from sleepwatch.anomaly.summary import AnomalySummary

STAGE_WORDS = {
    "Wake": "awake",
    "N1": "in light sleep",
    "N2": "in light sleep",
    "N3": "in deep sleep",
    "REM": "in REM sleep",
}
# anomaly.summary.DISCLAIMER, addressed to the person
DISCLAIMER = (
    "This is a research prototype, not a medical device. A flag means this night differed from "
    "your earlier nights; it does not say why."
)

OPEN_QUESTION = "Was anything about your day or evening different from usual?"
CLOSING = (
    "Thank you, that's all I wanted to ask. Your answers are saved with this night. " + DISCLAIMER
)
REFERRAL = (
    "I can't help with symptoms like that. If you have chest pain, have fainted or have trouble "
    "breathing, please contact a healthcare professional or emergency services now."
)


def _hhmm(local: str) -> str:
    return local[-5:]  # "YYYY-MM-DD HH:MM"


def _night_name(date_local: str) -> str:
    day = date.fromisoformat(date_local)
    return f"the night of {day.strftime('%A')} {day.day} {day.strftime('%B')}"


def _main_stage(share: dict[str, float]) -> str | None:
    words: dict[str, float] = {}
    for stage, part in share.items():
        if stage in STAGE_WORDS:
            words[STAGE_WORDS[stage]] = words.get(STAGE_WORDS[stage], 0.0) + part
    if not words:
        return None
    word, part = max(words.items(), key=lambda item: item[1])
    return word if part >= 0.5 else None


def describe(summary: AnomalySummary) -> str:
    """What was observed, in a few plain sentences, without z-scores, p-values or causes."""
    fired = {c.channel for c in summary.channels if c.fired}
    nights = summary.baseline_nights
    earlier = f"your {nights} earlier recorded night{'s' if nights != 1 else ''}"
    lines = [f"Your watch recorded {_night_name(summary.date_local)} as different from {earlier}."]
    if "hr_night" in fired:
        lines.append(
            "Across the night as a whole, your heart rate was higher than your earlier nights "
            "would suggest."
        )
    has_window = summary.window_start_local is not None and summary.window_end_local is not None
    if fired & {"hr_window", "hr_night"} and has_window and summary.hr_excess_bpm is not None:
        span = f"{_hhmm(summary.window_start_local)} and {_hhmm(summary.window_end_local)}"
        lines.append(
            f"The biggest difference was between {span}, when your heart rate was about "
            f"{round(summary.hr_excess_bpm)} beats per minute higher than expected for that "
            "part of the night."
        )
        stage = _main_stage(summary.window_stage_share)
        if stage:
            lines.append(f"The watch estimated you were mostly {stage} then.")
    if "frag" in fired and summary.wake_bouts_per_hour is not None:
        lines.append(
            "You woke briefly more often than on your earlier nights (the watch estimated about "
            f"{summary.wake_bouts_per_hour:.1f} awakenings per hour of sleep)."
        )
    if "onset" in fired and summary.onset_latency_min is not None:
        lines.append(
            f"It took about {round(summary.onset_latency_min)} minutes from the start of the "
            "recording to fall asleep, longer than on your earlier nights."
        )
    for note in summary.data_quality:
        lines.append(f"Note on the data: {note}.")
    return " ".join(lines)


def opening(summary: AnomalySummary) -> str:
    return (
        f"{describe(summary)}\n\n{DISCLAIMER}\n\n"
        "If you're happy to, I'll ask a few short questions about that day, and your answers "
        "will be saved with this night. You can skip any question.\n\n"
        f"{OPEN_QUESTION}"
    )
