"""The follow-up conversation and the extraction of reported factors (docs/decisions.md 49).

Code fixes the frame: the opening (facts, disclaimer, one open question), the closing, the
question limit, the guard and the red-flag referral. The LLM writes only the follow-up questions
and, in a separate call after the conversation, the list of things the person mentioned, which
code turns into the ``FollowUpResult``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sleepwatch.agent import guard
from sleepwatch.agent.backend import ChatBackend, Messages
from sleepwatch.agent.render import CLOSING, OPEN_QUESTION, REFERRAL, describe, opening
from sleepwatch.agent.schema import (
    Confidence,
    Factor,
    FollowUpRecord,
    FollowUpResult,
    ReportedFactor,
    Timing,
    Turn,
    inline_schema,
)
from sleepwatch.anomaly.summary import AnomalySummary

PROMPT_VERSION = "v6"
MAX_QUESTIONS = 5  # the templated open question included
EXTRACTION_RETRIES = 2
QUESTION_TOKENS = 150  # output caps: a runaway generation must not fill the context
EXTRACTION_TOKENS = 1200

QUESTION_PROMPT = """\
You are the follow-up assistant of a sleep research app. The person's watch recorded a night \
that differed from their earlier nights. The app has already shown them what was observed and \
asked an open question about their day. Your job is to ask a few short follow-up questions so \
the person can report anything about that day that was different. You only collect what they \
say.

What the app showed them:
{facts}

Rules:
- Write one short, friendly question (one or two sentences).
- Never say or hint what caused the night, and never connect anything the person says to their \
heart rate or sleep. Do not mention the observation in your questions.
- Never diagnose, give health advice or name medical conditions.
- Ask neutrally ("Did you ...?", "Was there ...?"). Never suggest an answer ("You probably ...", \
"Could it have been ...?").
- If the person mentions something, you may ask once when it happened or how sure they are.
- Never ask again about something the person has already answered or ruled out, and don't ask \
about details of their ordinary routine.
- Areas you may ask about if the person has not mentioned them, two in one question if natural: \
caffeine, alcohol, exercise late in the day, a late or heavy meal, stress, feeling unwell, a \
change in schedule or bedtime.
- To ask for anything else, ask whether anything else was different that day. Never ask what \
"affected" or "caused" anything.
- End the conversation only when the person has said there is nothing more, or the areas are \
covered.

Reply in JSON: {{"message": "<your next question>", "end": false}}, or \
{{"message": "", "end": true}} to end."""

NEXT = (
    "The conversation so far:\n\n{transcript}\n\nWrite the assistant's next question. You can "
    "ask at most {remaining} more question{s}."
)
CORRECTION = (
    " Your last draft was not allowed ({found}). Write a neutral question that does not suggest a "
    "cause, connect anything to the night, or mention any medical matter."
)

EXTRACTION_PROMPT = """\
You turn a conversation into structured data. List every factor the PERSON talked about \
regarding the day before the night, using only the person's own words.

Factor categories:
- caffeine: coffee, tea, energy drinks, cola
- alcohol
- late_exercise: physical exercise or sport in the late afternoon or evening
- late_meal: a late or heavy meal or snack
- stress: worry, conflict, pressure
- illness: feeling unwell, infection, pain, a vaccine, medicines
- schedule_change: going to bed or getting up at a different time than usual, working \
different hours than usual, travel across time zones
- other: anything else the person says was different that fits none of the categories above

For each thing the person talked about, give:
- status: "happened" if the person says it happened and was different from a normal day; \
"did_not_happen" if the person says it did not happen ("no coffee"); "usual" if the person \
mentions it as part of a normal day ("dinner at the usual time", "coffee as I always do")
- detail: a short description in plain words
- timing: morning, afternoon, evening, before_bed, during_night, earlier_days (the days before), \
or unspecified if the person did not say
- confidence: "certain" if the person said it plainly, "unsure" if they hedged ("I think", \
"maybe", "not sure")
- quote: the person's exact words about it, a short span copied character for character

Rules:
- Only list what the person talked about. Leave out categories that never came up.
- Words like "as usual" or "otherwise a normal day" apply only to the things they describe. \
Anything else the person reported as different still happened: in "I went to a concert, \
otherwise the day was normal", the concert happened.
- Hedging does not make something usual: "I think I had a few beers" reports something that \
happened, with confidence "unsure".
- Never add anything the person did not say, however likely it seems. What the assistant said \
is not evidence.
- If the person had nothing unusual to report, return an empty list."""


class Mention(BaseModel):
    """What the extraction model returns per thing the person talked about. Code turns these into
    the ``FollowUpResult``: "happened" becomes a factor, "did_not_happen" a denial, "usual" is
    dropped."""

    model_config = ConfigDict(extra="forbid")

    factor: Factor
    status: Literal["happened", "did_not_happen", "usual"]
    detail: str = Field(max_length=120)
    timing: Timing
    confidence: Confidence
    quote: str = Field(max_length=200)  # the length limits also bound the generation


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mentions: list[Mention] = Field(max_length=10)


def to_result(extraction: Extraction) -> tuple[FollowUpResult, list[str]]:
    """The ``FollowUpResult`` and the repairs code made: an exact repeat (same factor and words)
    is dropped, a repeated category keeps its first entry, a category both reported and denied
    stays reported, "other" cannot be denied, and ``nothing_unusual`` is true exactly when no
    factor was reported."""
    factors: list[ReportedFactor] = []
    denied: list[str] = []
    repairs: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for m in extraction.mentions:
        key = (m.factor, m.status, " ".join(m.quote.lower().split()))
        if key in seen:
            repairs.append(f"exact repeat of {m.factor} dropped")
            continue
        seen.add(key)
        if m.status == "happened":
            if m.factor != "other" and any(f.factor == m.factor for f in factors):
                repairs.append(f"repeated {m.factor} dropped")
                continue
            try:
                factors.append(
                    ReportedFactor(
                        factor=m.factor,
                        detail=m.detail.strip() or m.factor,
                        timing=m.timing,
                        confidence=m.confidence,
                        quote=m.quote.strip(),
                    )
                )
            except ValidationError:
                repairs.append(f"invalid {m.factor} dropped")
        elif m.status == "did_not_happen":
            if m.factor == "other":
                repairs.append("denied 'other' dropped")
            elif m.factor not in denied:
                denied.append(m.factor)
    reported = {f.factor for f in factors}
    if overlap := [d for d in denied if d in reported]:
        repairs.append(f"reported and denied, kept as reported: {', '.join(overlap)}")
        denied = [d for d in denied if d not in reported]
    result = FollowUpResult(factors=factors, denied=denied, nothing_unusual=not factors)
    return result, repairs


class Step(BaseModel):
    message: str
    end: bool


STEP_SCHEMA = inline_schema(Step)
RESULT_SCHEMA = inline_schema(Extraction, require_all=True)


def transcript(turns: list[Turn]) -> str:
    names = {"agent": "Assistant", "person": "Person"}
    return "\n\n".join(f"{names[t.role]}: {t.text}" for t in turns)


def answers_transcript(turns: list[Turn]) -> str:
    """What the extraction model sees: the questions and the answers. The opening is reduced to
    its open question, so the description of the night cannot be taken for the person's words."""
    shown = [
        t.model_copy(update={"text": OPEN_QUESTION}) if t.kind == "opening" else t
        for t in turns
        if t.role == "person" or t.kind in ("opening", "question", "fallback")
    ]
    return transcript(shown)


def questions_asked(turns: list[Turn]) -> int:
    return sum(t.kind in ("opening", "question", "fallback") for t in turns)


class LLMAgent:
    """Follow-up questions written by an LLM, checked by the guard."""

    def __init__(self, backend: ChatBackend, seed: int = 0, temperature: float = 0.0) -> None:
        self.backend = backend
        self.name = backend.model
        self.seed = seed
        self.temperature = temperature

    def messages(
        self, summary: AnomalySummary, turns: list[Turn], remaining: int, found: list[str] | None
    ) -> Messages:
        """The rules as the system message and the conversation as a transcript in one user
        message, which small models follow more reliably than replayed turns."""
        note = NEXT.format(
            transcript=transcript(turns), remaining=remaining, s="s" if remaining != 1 else ""
        )
        if found:
            note += CORRECTION.format(found="; ".join(found))
        return [
            {"role": "system", "content": QUESTION_PROMPT.format(facts=describe(summary))},
            {"role": "user", "content": note},
        ]

    def next_question(
        self, summary: AnomalySummary, turns: list[Turn], remaining: int
    ) -> Turn | None:
        """The next question, or ``None`` to end. Two rejected drafts give ``guard.FALLBACK``."""
        drafts: list[str] = []
        hits: list[list[str]] = []
        seconds = 0.0
        for _ in range(2):
            reply = self.backend.chat(
                self.messages(summary, turns, remaining, hits[-1] if hits else None),
                schema=STEP_SCHEMA,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=QUESTION_TOKENS,
            )
            seconds += reply.seconds
            try:
                step = Step.model_validate_json(reply.text)
            except ValidationError:
                drafts.append(reply.text)
                hits.append(["invalid: not a {message, end} object"])
                continue
            if step.end:
                return None
            found = guard.problems(step.message) or (
                [] if step.message.strip() else ["invalid: empty message"]
            )
            if not found:
                return Turn(
                    role="agent",
                    kind="question",
                    text=step.message.strip(),
                    drafts=drafts,
                    guard=hits,
                    seconds=seconds,
                )
            drafts.append(step.message)
            hits.append(found)
        return Turn(
            role="agent",
            kind="fallback",
            text=guard.FALLBACK,
            drafts=drafts,
            guard=hits,
            seconds=seconds,
        )


class ChecklistAgent:
    """The baseline: the same opening, then fixed neutral questions, in order."""

    QUESTIONS = (
        "Did you have any caffeine or alcohol that day? If so, roughly when?",
        "Did you exercise, or eat a large or late meal, later in the day?",
        "Were you feeling stressed or unwell?",
        "Was your schedule different from usual, for example a later bedtime, a shift or travel?",
    )

    name = "checklist"

    def next_question(
        self, summary: AnomalySummary, turns: list[Turn], remaining: int
    ) -> Turn | None:
        asked = questions_asked(turns) - 1  # the opening is not on the list
        if asked >= len(self.QUESTIONS):
            return None
        return Turn(role="agent", kind="question", text=self.QUESTIONS[asked], seconds=0.0)


def extract(
    backend: ChatBackend, turns: list[Turn], seed: int = 0, temperature: float = 0.0
) -> tuple[FollowUpResult, int, str | None, float, list[str], list[str]]:
    """``(result, attempts, error, seconds, raw outputs, repairs)``. Output that does not parse is
    sent back with the error, up to ``EXTRACTION_RETRIES`` times; after that the result is empty
    and ``error`` set."""
    messages: Messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": answers_transcript(turns)},
    ]
    seconds = 0.0
    error = None
    raw: list[str] = []
    retry: Messages = []
    for attempt in range(1, EXTRACTION_RETRIES + 2):
        reply = backend.chat(
            messages + retry,
            schema=RESULT_SCHEMA,
            temperature=temperature,
            seed=seed + attempt - 1,  # a repeated seed would repeat a runaway output
            max_tokens=EXTRACTION_TOKENS,
        )
        seconds += reply.seconds
        raw.append(reply.text)
        try:
            extraction = Extraction.model_validate_json(reply.text)
        except ValidationError as err:
            error = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'output'}: {e['msg']}" for e in err.errors()
            )
            # only the error goes back: a runaway output would fill the context
            retry = [
                {
                    "role": "user",
                    "content": f"Your previous answer was invalid ({error[:300]}). Reply again "
                    "with valid JSON only.",
                }
            ]
            continue
        result, repairs = to_result(extraction)
        return result, attempt, None, seconds, raw, repairs
    return FollowUpResult(nothing_unusual=False), EXTRACTION_RETRIES + 1, error, seconds, raw, []


class Conversation:
    """One follow-up in progress. ``converse`` runs one to the end; the evaluation advances many
    together, one model at a time, so models are not swapped in and out of memory each turn."""

    def __init__(
        self, summary: AnomalySummary, case_id: str, max_questions: int = MAX_QUESTIONS
    ) -> None:
        self.summary = summary
        self.case_id = case_id
        self.max_questions = max_questions
        self.turns = [Turn(role="agent", kind="opening", text=opening(summary), seconds=0.0)]
        self.finished = False

    @property
    def remaining(self) -> int:
        return self.max_questions - questions_asked(self.turns)

    def answer(self, text: str, seconds: float | None = None) -> None:
        """Add the person's reply; end with a referral on a red flag or at the question limit."""
        self.turns.append(Turn(role="person", kind="answer", text=text, seconds=seconds))
        if guard.red_flag(text):
            self.turns.append(Turn(role="agent", kind="referral", text=REFERRAL, seconds=0.0))
            self.finished = True
        elif self.remaining <= 0:
            self.ask(None)

    def ask(self, question: Turn | None) -> None:
        """Add the agent's next question, or close the conversation if it is ``None``."""
        if question is None:
            self.turns.append(Turn(role="agent", kind="closing", text=CLOSING, seconds=0.0))
            self.finished = True
        else:
            self.turns.append(question)

    def record(self, agent_name: str, extractor: ChatBackend, seed: int = 0) -> FollowUpRecord:
        """Extract the reported factors from the finished conversation."""
        result, attempts, error, seconds, raw, repairs = extract(extractor, self.turns, seed=seed)
        return FollowUpRecord(
            case_id=self.case_id,
            agent=agent_name,
            model=extractor.model,
            prompt_version=PROMPT_VERSION,
            turns=self.turns,
            result=result,
            extraction_attempts=attempts,
            extraction_error=error,
            extraction_seconds=seconds,
            extraction_raw=raw,
            extraction_repairs=repairs,
        )


def converse(
    agent: LLMAgent | ChecklistAgent,
    extractor: ChatBackend,
    summary: AnomalySummary,
    answer: Callable[[list[Turn]], str],
    case_id: str,
    max_questions: int = MAX_QUESTIONS,
    seed: int = 0,
) -> FollowUpRecord:
    """Run one follow-up: ``answer(turns)`` gives the person's reply to the last agent turn."""
    conversation = Conversation(summary, case_id, max_questions)
    while not conversation.finished:
        start = time.perf_counter()
        text = answer(conversation.turns)
        conversation.answer(text, time.perf_counter() - start)
        if not conversation.finished:
            conversation.ask(
                agent.next_question(summary, conversation.turns, conversation.remaining)
            )
    return conversation.record(agent.name, extractor, seed)
