"""Checks on generated messages before the person sees them (docs/decisions.md 49).

The agent may ask about factors but must not say or imply what caused the night, and must not
diagnose or advise. A generated question that trips a pattern is regenerated once and then
replaced by ``FALLBACK``. The patterns are deliberately broad: a false hit costs one regeneration.
"""

from __future__ import annotations

import re

CAUSAL = re.compile(
    r"\b(caus\w*|due to|because of|result(ed|s)? (of|from)|led to|lead(s|ing)? to|"
    r"responsible for|trigger\w*|reason for|that'?s why|which is why|explain(s|ed)? (the|this|"
    r"that|your|why)|why (your|you)|linked to|associated with|contribut\w*|affect(ed|s|ing)?|"
    r"impact(ed|s|ing)?|(main|primary) (factor|reason)|factor (in|behind)|role in|played a "
    r"(part|role))\b",
    re.IGNORECASE,
)
# The opening states the observation; a generated question must not bring it up again.
OBSERVATION = re.compile(r"\b(heart ?rate|heartbeat|pulse|sleep onset)\b", re.IGNORECASE)
DIAGNOSIS = re.compile(
    r"\b(diagnos\w*|disorder|apno?ea|insomnia|arrhythmi\w*|atrial|a-?fib|tachycardi\w*|"
    r"bradycardi\w*|hypertension|treat(ment|ed)?|prescri\w*|you (should|must|need to|ought to) "
    r"(see|take|stop|avoid|reduce|cut|try|consult|talk)|see a doctor)\b",
    re.IGNORECASE,
)
RED_FLAGS = re.compile(
    r"\b(chest pains?|chest tightness|faint(ed|ing)?|passed out|blacked out|short(ness)? of "
    r"breath|(can'?t|couldn'?t|could not|trouble|difficulty) breath(e|ing))\b",
    re.IGNORECASE,
)
FALLBACK = "Is there anything else about that day or evening you'd like to mention?"


def problems(text: str) -> list[str]:
    """Causal wording, mentions of the observation and diagnostic phrases in a generated
    question, as ``kind: phrase``."""
    found = [f"causal: {m.group(0)}" for m in CAUSAL.finditer(text)]
    found += [f"observation: {m.group(0)}" for m in OBSERVATION.finditer(text)]
    found += [f"diagnosis: {m.group(0)}" for m in DIAGNOSIS.finditer(text)]
    return found


def red_flag(text: str) -> bool:
    """Whether the person mentions a symptom that needs a referral, not more questions."""
    return RED_FLAGS.search(text) is not None
