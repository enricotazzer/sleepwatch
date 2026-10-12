"""What the follow-up agent stores: the factors a person reported about the day before a flagged
night, as validated JSON (docs/decisions.md 49).

``confidence`` is how sure the *person* was ("I think I had a coffee" is ``unsure``), not the
model's confidence. ``quote`` holds the person's own words, so every factor can be checked against
the conversation.
"""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

Checklist = Literal[
    "caffeine",
    "alcohol",
    "late_exercise",
    "late_meal",
    "stress",
    "illness",
    "schedule_change",
]
Factor = Literal[
    "caffeine",
    "alcohol",
    "late_exercise",
    "late_meal",
    "stress",
    "illness",
    "schedule_change",
    "other",
]
Timing = Literal[
    "morning",
    "afternoon",
    "evening",
    "before_bed",
    "during_night",
    "earlier_days",
    "unspecified",
]
Confidence = Literal["certain", "unsure"]

CHECKLIST: tuple[str, ...] = get_args(Checklist)
FACTORS: tuple[str, ...] = get_args(Factor)
TIMINGS: tuple[str, ...] = get_args(Timing)


class ReportedFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: Factor
    detail: str = Field(min_length=1, max_length=200)  # e.g. "two espressos"
    timing: Timing = "unspecified"
    confidence: Confidence
    quote: str = Field(min_length=1, max_length=400)  # the person's own words


class FollowUpResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factors: list[ReportedFactor] = []
    denied: list[Checklist] = []  # factors the person ruled out
    nothing_unusual: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> FollowUpResult:
        named = [f.factor for f in self.factors if f.factor != "other"]
        if len(named) != len(set(named)):
            raise ValueError("a factor other than 'other' is listed twice; merge them")
        if set(named) & set(self.denied):
            raise ValueError("a factor is both reported and denied")
        if len(self.denied) != len(set(self.denied)):
            raise ValueError("a denied factor is listed twice")
        if self.nothing_unusual and self.factors:
            raise ValueError("nothing_unusual is true but factors are listed")
        return self


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["agent", "person"]
    text: str
    # agent: opening, question, fallback (guard replaced the question), closing or referral
    kind: Literal["opening", "question", "fallback", "closing", "referral", "answer"]
    drafts: list[str] = []  # generated texts the guard rejected, in order
    guard: list[list[str]] = []  # what the guard found in each rejected draft
    seconds: float | None = None


class FollowUpRecord(BaseModel):
    """One finished follow-up: the conversation and what was extracted from it."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    agent: str
    model: str | None  # the model that extracted (and asked, for an LLM agent)
    prompt_version: str
    turns: list[Turn]
    result: FollowUpResult
    extraction_attempts: int
    extraction_error: str | None = None
    extraction_seconds: float | None = None
    extraction_raw: list[str] = []  # the model's output on each attempt
    extraction_repairs: list[str] = []  # what code fixed in the parsed output (agent.to_result)


def inline_schema(model: type[BaseModel], require_all: bool = False) -> dict:
    """JSON schema with every ``$ref`` resolved in place, for backends that constrain output.

    ``require_all`` makes every property required, so a constrained model cannot skip fields
    that have defaults (such as ``timing``)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            out = {k: resolve(v) for k, v in node.items()}
            if require_all and "properties" in out:
                out["required"] = list(out["properties"])
                for prop in out["properties"].values():
                    prop.pop("default", None)
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)
