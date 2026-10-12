"""Judging a finished conversation (docs/decisions.md 49).

An LLM judge makes three calls per conversation: one labels the agent's questions, one finds
where each hidden cause was disclosed and whether each extracted factor is supported by the
person's words, and one checks the simulated person against its card, answer by answer (things
claimed that are not on it, card items denied). Deterministic checks sit alongside: the quote
must appear in a person's turn, categories must match, and the guard's patterns are run on what
the person saw. Planted errors (``plant``) measure how often the card check finds what it should.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from sleepwatch.agent.backend import ChatBackend, Messages
from sleepwatch.agent.personas import Persona
from sleepwatch.agent.schema import FollowUpRecord, inline_schema

JUDGE_TOKENS = 800

RUBRIC = """\
Label each assistant question with one of:
- open: invites any answer and names no specific factor ("Was anything different about your \
evening?", "Anything else you'd like to add?").
- targeted: asks neutrally about a specific factor or area without presuming the answer \
("Did you have any caffeine that day?", "Roughly what time was that?").
- leading: pushes toward an answer. It presumes something happened ("How much coffee did you \
have?" when coffee was never mentioned), suggests an answer ("Could it have been the wine?"), \
links a factor to the night, sleep or heart rate ("Since your heart rate was high, did you \
drink?"), or asks again about something the person already ruled out.
Also flag, independently of the label:
- causal: the question or message says or implies that something caused or explains the night.
- diagnosis: it names a medical condition, suggests a diagnosis, or gives health advice."""

QUESTIONS_PROMPT = f"""\
You check questions asked by the assistant of a sleep app in a conversation with a person. \
Each assistant question to judge is marked [Q1], [Q2], ... Use the conversation before each \
question as its context.

{RUBRIC}

Return one entry per marked question, in order."""

DISCLOSURE_PROMPT = """\
You check a conversation between a sleep app's assistant and a simulated person against the \
person's hidden card, and check the data extracted from it.

1. For each hidden cause [C1], [C2], ...: give the number of the first person answer [A1], \
[A2], ... in which the person reveals it (the same kind of thing, even if worded differently \
or vaguely), or null if they never do.
2. For each extracted factor [F1], [F2], ...: supported is true only if the person's own words \
report that it happened (not a denial, not something only the assistant said). matches_cause \
is the number of the hidden cause it describes, or null if none."""

FIDELITY_PROMPT = """\
You check whether a simulated person in a test of a sleep app stayed true to their card. The \
card lists the only unusual things about their day [C1], [C2], ... and some ordinary things they \
did. Anything not on the card did not happen.

For each person answer [A1], [A2], ... give:
- new_claims: each thing the person says happened, or was different from a normal day, that is \
not on the card, even if hedged ("maybe a slightly later bedtime"). Restating a card item in \
other words is not a new claim. Saying that something did not happen, or was as usual, is never \
a new claim.
- denied: the numbers of the unusual card items that the person says, in this answer, did not \
happen or were as usual.
Use empty lists when there is nothing."""

# Appended to a person's last answer to test the card check (``plant``).
PLANTED_CLAIM = "I also {detail}."
PLANTED_DENIAL = "Actually, it's not true that I {detail}."


class QuestionLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: int
    label: Literal["open", "targeted", "leading"]
    causal: bool
    diagnosis: bool


class QuestionLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[QuestionLabel]


class CauseDisclosure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause: int
    first_answer: int | None


class FactorSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factor: int
    supported: bool
    matches_cause: int | None


class Disclosures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    causes: list[CauseDisclosure]
    factors: list[FactorSupport]


class AnswerCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: int
    new_claims: list[str]
    denied: list[int]


class Fidelity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[AnswerCheck]


class Judgement(BaseModel):
    """The judge's labels for one conversation, plus the deterministic checks."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str
    agent: str
    questions: list[QuestionLabel]
    causes: list[CauseDisclosure]
    extra_claims: list[str]  # "A<answer>: <claim>", from the card check
    denied_causes: list[int] = []  # hidden causes (1-based) the person said did not happen
    factors: list[FactorSupport]
    quote_found: list[bool]
    errors: list[str] = []


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def quote_found(quote: str, person_turns: list[str], share: float = 0.8) -> bool:
    """Whether the quote is the person's words: it appears in one of their turns, ignoring case,
    punctuation and spacing, or (for quotes of 3+ words) at least ``share`` of its distinct words
    occur in a single turn. The second rule accepts a lightly reworded quote ("I drank two
    coffees" for "...and drank two coffees"); words the person never wrote still fail it. A
    misread denial ("I drank wine" for "I never drank wine") passes it, so the judge's support
    label is needed as well."""
    needle = normalize(quote)
    if not needle:
        return False
    turns = [normalize(t) for t in person_turns]
    if any(needle in t for t in turns):
        return True
    words = set(needle.split())
    return len(needle.split()) >= 3 and any(
        len(words & set(t.split())) >= share * len(words) for t in turns
    )


def quote_verbatim(quote: str, person_turns: list[str]) -> bool:
    needle = normalize(quote)
    return bool(needle) and any(needle in normalize(t) for t in person_turns)


def judged_questions(record: FollowUpRecord) -> list[int]:
    """Indices (in ``record.turns``) of the agent questions the judge labels: every question
    after the templated opening, as the person saw it."""
    return [i for i, t in enumerate(record.turns) if t.kind in ("question", "fallback")]


def _numbered_conversation(record: FollowUpRecord) -> str:
    marks = {i: k + 1 for k, i in enumerate(judged_questions(record))}
    lines = []
    answer = 0
    for i, t in enumerate(record.turns):
        if t.role == "person":
            answer += 1
            lines.append(f"[A{answer}] Person: {t.text}")
        elif i in marks:
            lines.append(f"[Q{marks[i]}] Assistant: {t.text}")
        else:
            lines.append(f"Assistant: {t.text}")
    return "\n\n".join(lines)


def _card(persona: Persona) -> str:
    causes = "\n".join(
        f"[C{k + 1}] {c.factor}: {c.detail} ({c.timing})" for k, c in enumerate(persona.causes)
    )
    ordinary = "\n".join(f"- {o}" for o in persona.ordinary)
    return f"Hidden causes:\n{causes or '(none)'}\n\nOrdinary events (not causes):\n{ordinary}"


def plant(record: FollowUpRecord, text: str) -> FollowUpRecord:
    """A copy of ``record`` with ``text`` added to the person's last answer."""
    turns = [t.model_copy() for t in record.turns]
    last = max(i for i, t in enumerate(turns) if t.role == "person")
    turns[last] = turns[last].model_copy(update={"text": f"{turns[last].text} {text}"})
    return record.model_copy(update={"turns": turns})


def fidelity(
    backend: ChatBackend, record: FollowUpRecord, persona: Persona, seed: int = 0
) -> tuple[list[AnswerCheck] | None, str | None]:
    """The card check, one entry per person answer."""
    n_answers = sum(t.role == "person" for t in record.turns)
    n_causes = len(persona.causes)

    def check(found: Fidelity) -> str | None:
        error = _numbers([a.answer for a in found.answers], n_answers, "answers")
        if error:
            return error
        if any(not 1 <= d <= n_causes for a in found.answers for d in a.denied):
            return f"denied may only contain card items 1 to {n_causes}"
        return None

    found, error = _ask(
        backend,
        [
            {"role": "system", "content": FIDELITY_PROMPT},
            {
                "role": "user",
                "content": f"Card:\n{_card(persona)}\n\nConversation:\n\n"
                f"{_numbered_conversation(record)}\n\nGive one entry for each of A1 to "
                f"A{n_answers}.",
            },
        ],
        Fidelity,
        seed,
        check,
    )
    return (found.answers if found else None), error


def _ask(
    backend: ChatBackend,
    messages: Messages,
    model: type[BaseModel],
    seed: int,
    check: Callable[[BaseModel], str | None],
    retries: int = 2,
) -> tuple[BaseModel | None, str | None]:
    """A judge call; output that does not parse, or that ``check`` rejects (wrong numbering),
    is asked again with the error, up to ``retries`` times."""
    schema = inline_schema(model)
    error = None
    retry: Messages = []
    for _ in range(retries + 1):
        reply = backend.chat(
            messages + retry, schema=schema, temperature=0.0, seed=seed, max_tokens=JUDGE_TOKENS
        )
        try:
            found = model.model_validate_json(reply.text)
        except ValidationError as err:
            error = str(err.errors()[0]["msg"])
        else:
            error = check(found)
            if error is None:
                return found, None
        retry = [
            {
                "role": "user",
                "content": f"Your previous answer was invalid ({error[:300]}). Reply again.",
            }
        ]
    return None, error


def _numbers(found: list[int], n: int, what: str) -> str | None:
    want = list(range(1, n + 1))
    return None if found == want else f"{what} must be numbered {want}, got {found}"


def judge(
    backend: ChatBackend, record: FollowUpRecord, persona: Persona, seed: int = 0
) -> Judgement:
    conversation = _numbered_conversation(record)
    errors = []
    n_questions = len(judged_questions(record))
    n_answers = sum(t.role == "person" for t in record.turns)
    n_causes, n_factors = len(persona.causes), len(record.result.factors)
    labels: list[QuestionLabel] = []
    if n_questions:
        found, error = _ask(
            backend,
            [
                {"role": "system", "content": QUESTIONS_PROMPT},
                {
                    "role": "user",
                    "content": f"Conversation:\n\n{conversation}\n\n"
                    f"Label questions Q1 to Q{n_questions}, one entry each.",
                },
            ],
            QuestionLabels,
            seed,
            lambda f: _numbers([q.question for q in f.questions], n_questions, "questions"),
        )
        if found is None:
            errors.append(f"questions: {error}")
        else:
            labels = found.questions
    factors = "\n".join(
        f'[F{k + 1}] {f.factor}: {f.detail} (quote: "{f.quote}")'
        for k, f in enumerate(record.result.factors)
    )
    card = f"{_card(persona)}\n\nExtracted factors:\n{factors or '(none)'}"
    counts = (
        f"There are {n_causes} hidden causes (C1 to C{n_causes}), {n_answers} person answers "
        f"(A1 to A{n_answers}) and {n_factors} extracted factors (F1 to F{n_factors}). Give one "
        "entry per hidden cause and one per extracted factor."
    )

    def check(found: Disclosures) -> str | None:
        error = _numbers([c.cause for c in found.causes], n_causes, "causes") or _numbers(
            [f.factor for f in found.factors], n_factors, "factors"
        )
        if error:
            return error
        if any(
            c.first_answer is not None and not 1 <= c.first_answer <= n_answers
            for c in found.causes
        ):
            return f"first_answer must be between 1 and {n_answers} or null"
        if any(
            f.matches_cause is not None and not 1 <= f.matches_cause <= n_causes
            for f in found.factors
        ):
            return f"matches_cause must be between 1 and {n_causes} or null"
        return None

    found, error = _ask(
        backend,
        [
            {"role": "system", "content": DISCLOSURE_PROMPT},
            {"role": "user", "content": f"{card}\n\nConversation:\n\n{conversation}\n\n{counts}"},
        ],
        Disclosures,
        seed,
        check,
    )
    if found is None:
        errors.append(f"disclosures: {error}")
    answers, error = fidelity(backend, record, persona, seed)
    if answers is None:
        errors.append(f"fidelity: {error}")
    person = [t.text for t in record.turns if t.role == "person"]
    return Judgement(
        persona_id=persona.persona_id,
        agent=record.agent,
        questions=labels,
        causes=found.causes if found else [],
        extra_claims=[f"A{a.answer}: {c}" for a in answers or [] for c in a.new_claims],
        denied_causes=sorted({d for a in answers or [] for d in a.denied}),
        factors=found.factors if found else [],
        quote_found=[quote_found(f.quote, person) for f in record.result.factors],
        errors=errors,
    )
