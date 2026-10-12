"""Phase 4: the follow-up agent, simulated people, the judge's checks and the metrics."""

import json

import httpx
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sleepwatch.agent import guard
from sleepwatch.agent.agent import (
    EXTRACTION_TOKENS,
    MAX_QUESTIONS,
    RESULT_SCHEMA,
    ChecklistAgent,
    Conversation,
    Extraction,
    LLMAgent,
    answers_transcript,
    converse,
    extract,
    questions_asked,
    to_result,
)
from sleepwatch.agent.backend import AgentSettings, ScriptedBackend
from sleepwatch.agent.cases import AgentCase, select
from sleepwatch.agent.evaluate import (
    AgentEvalConfig,
    AgentSpec,
    compute_metrics,
    run_conversations,
    run_judge,
    run_probes,
    start_run,
)
from sleepwatch.agent.judge import (
    CauseDisclosure,
    FactorSupport,
    Judgement,
    QuestionLabel,
    judge,
    judged_questions,
    plant,
    quote_found,
    quote_verbatim,
)
from sleepwatch.agent.labels import agreement, sample
from sleepwatch.agent.metrics import counts, summarize
from sleepwatch.agent.personas import (
    CAUSE_COUNTS,
    CAUSES,
    OTHER_CAUSES,
    STYLES,
    TOPIC_PATTERNS,
    HiddenCause,
    Persona,
    SimulatedUser,
    make_personas,
    unlocked,
)
from sleepwatch.agent.render import CLOSING, DISCLAIMER, OPEN_QUESTION, REFERRAL, describe, opening
from sleepwatch.agent.schema import FollowUpRecord, FollowUpResult, ReportedFactor, Turn
from sleepwatch.anomaly.summary import AnomalySummary, ChannelResult

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def make_summary(**changes) -> AnomalySummary:
    fields = {
        "subject": "S01",
        "night": 3,
        "date_local": "2022-03-28",
        "baseline_nights": 2,
        "top_channel": "hr_window",
        "channels": [
            ChannelResult(channel="hr_window", value=4.81, tail_probability=0.008, fired=True),
            ChannelResult(channel="hr_night", value=1.2, tail_probability=0.2, fired=False),
            ChannelResult(channel="frag", value=-0.5, tail_probability=0.6, fired=False),
            ChannelResult(channel="onset", value=-0.3, tail_probability=0.5, fired=False),
        ],
        "window_start_local": "2022-03-28 23:04",
        "window_end_local": "2022-03-28 23:34",
        "window_hours_since_onset": 0.36,
        "window_stage_share": {"N3": 0.8, "N2": 0.2},
        "hr_excess_bpm": 19.75,
        "wake_bouts_per_hour": 0.13,
        "onset_latency_min": 18.0,
    }
    return AnomalySummary(**(fields | changes))


def factor(name="caffeine", quote="two coffees at 4", **changes) -> ReportedFactor:
    fields = {
        "factor": name,
        "detail": "coffee",
        "timing": "afternoon",
        "confidence": "certain",
        "quote": quote,
    }
    return ReportedFactor(**(fields | changes))


def step(message="Did you have any caffeine that day?", end=False) -> str:
    return json.dumps({"message": message, "end": end})


EMPTY = json.dumps({"mentions": []})


def mention(name="caffeine", status="happened", quote="two coffees at 4", **changes) -> dict:
    fields = {
        "factor": name,
        "status": status,
        "detail": "coffee",
        "timing": "afternoon",
        "confidence": "certain",
        "quote": quote,
    }
    return fields | changes


# --- schema --------------------------------------------------------------------------------------


def test_result_schema_rules():
    FollowUpResult(factors=[factor(), factor("other"), factor("other")], denied=["alcohol"])
    with pytest.raises(ValidationError, match="listed twice"):
        FollowUpResult(factors=[factor(), factor()])
    with pytest.raises(ValidationError, match="reported and denied"):
        FollowUpResult(factors=[factor()], denied=["caffeine"])
    with pytest.raises(ValidationError, match="nothing_unusual"):
        FollowUpResult(factors=[factor()], nothing_unusual=True)
    with pytest.raises(ValidationError):
        FollowUpResult(denied=["other"])  # only checklist factors can be denied
    with pytest.raises(ValidationError):
        ReportedFactor.model_validate(factor().model_dump() | {"cause": "the night"})
    with pytest.raises(ValidationError):
        factor(confidence="0.9")


def test_model_facing_schema_is_inline_and_requires_every_field():
    text = json.dumps(RESULT_SCHEMA)
    assert "$ref" not in text and "$defs" not in text
    item = RESULT_SCHEMA["properties"]["mentions"]["items"]
    assert set(item["required"]) == {"factor", "status", "detail", "timing", "confidence", "quote"}
    assert RESULT_SCHEMA["required"] == ["mentions"]


def test_mentions_become_a_valid_result_and_repairs_are_logged():
    extraction = Extraction(
        mentions=[
            mention(),
            mention(quote="another coffee"),  # repeated category
            mention("alcohol", "did_not_happen", quote="no wine"),
            mention("caffeine", "did_not_happen", quote="no tea"),  # contradicts the report
            mention("late_meal", "usual", quote="dinner at the usual time"),
            mention("other", "did_not_happen", quote="no noise"),
            mention("other", detail="nap", quote="a long nap"),
            mention("other", detail="noise", quote="loud neighbours"),
        ]
    )
    result, repairs = to_result(extraction)
    assert [f.factor for f in result.factors] == ["caffeine", "other", "other"]
    assert result.factors[0].quote == "two coffees at 4"
    assert result.denied == ["alcohol"] and not result.nothing_unusual
    assert len(repairs) == 3
    repeated, repairs = to_result(Extraction(mentions=[mention("other", quote="a nap")] * 3))
    assert len(repeated.factors) == 1 and repairs == ["exact repeat of other dropped"] * 2
    empty, repairs = to_result(Extraction(mentions=[mention(status="usual")]))
    assert empty.factors == [] and empty.nothing_unusual and repairs == []


# --- what the person sees ------------------------------------------------------------------------


def test_describe_states_facts_without_scores_or_causes():
    text = describe(make_summary())
    assert "Monday 28 March" in text and "2 earlier recorded nights" in text
    assert "23:04 and 23:34" in text and "about 20 beats per minute" in text
    assert "mostly in deep sleep" in text
    assert "4.81" not in text and "0.008" not in text and "z" not in text.split()
    assert not guard.CAUSAL.search(text) and not guard.DIAGNOSIS.search(text)
    night = make_summary(
        channels=[
            ChannelResult(channel="hr_window", value=1, tail_probability=0.3, fired=False),
            ChannelResult(channel="hr_night", value=1, tail_probability=0.3, fired=False),
            ChannelResult(channel="frag", value=3, tail_probability=0.01, fired=True),
            ChannelResult(channel="onset", value=3, tail_probability=0.01, fired=True),
        ],
        data_quality=["implausible heart-rate stretch: 32 min at up to 172 bpm"],
    )
    text = describe(night)
    assert "beats per minute" not in text  # the HR window is only described if an HR channel fired
    assert "0.1 awakenings per hour" in text and "about 18 minutes" in text
    assert "Note on the data: implausible heart-rate stretch" in text


def test_opening_has_facts_disclaimer_and_one_open_question():
    text = opening(make_summary())
    assert text.startswith(describe(make_summary())) and DISCLAIMER in text
    assert text.endswith(OPEN_QUESTION) and text.count("?") == 1
    assert DISCLAIMER in CLOSING


def test_guard_patterns():
    assert guard.problems("Could the wine have caused this?")
    assert guard.problems("That might be due to stress.")
    assert guard.problems("This looks like insomnia.")
    assert guard.problems("You should see a doctor about it.")
    assert guard.problems("Was the argument the primary factor in your sleep onset?")
    assert guard.problems("Did anything raise your heart rate?")[0].startswith("observation")
    assert guard.problems("Was there a late meal that might have affected your energy levels?")
    for ok in [
        "Did you have any caffeine that day?",
        "Was there anything else different about your evening?",
        "Roughly what time did you finish dinner?",
    ]:
        assert guard.problems(ok) == [], ok
    assert guard.red_flag("I had chest pain and nearly fainted")
    assert guard.red_flag("I couldn't breathe properly")
    assert not guard.red_flag("my heart was racing before the exam")


# --- the conversation ----------------------------------------------------------------------------


def test_llm_agent_asks_at_most_the_limit_and_closes():
    agent_backend = ScriptedBackend(lambda m, s: step() if s and "end" in json.dumps(s) else EMPTY)
    answers = iter(f"answer {i}" for i in range(10))
    record = converse(
        LLMAgent(agent_backend), agent_backend, make_summary(), lambda t: next(answers), "c"
    )
    kinds = [t.kind for t in record.turns]
    assert kinds[0] == "opening" and kinds[-1] == "closing"
    assert questions_asked(record.turns) == MAX_QUESTIONS
    assert sum(t.role == "person" for t in record.turns) == MAX_QUESTIONS
    assert record.result.nothing_unusual and record.extraction_attempts == 1
    system = agent_backend.calls[0][0][0]["content"]
    assert describe(make_summary()) in system  # the facts, rendered by code


def test_llm_agent_may_end_early():
    backend = ScriptedBackend([step(end=True), EMPTY])
    record = converse(LLMAgent(backend), backend, make_summary(), lambda t: "nothing", "c")
    assert [t.kind for t in record.turns] == ["opening", "answer", "closing"]


def test_guard_regenerates_once_then_falls_back():
    bad = step("Could the coffee have caused your high heart rate?")
    backend = ScriptedBackend([bad, step(), bad, bad, step(end=True), EMPTY])
    record = converse(LLMAgent(backend), backend, make_summary(), lambda t: "ok", "c")
    regenerated, fallback = record.turns[2], record.turns[4]
    assert regenerated.kind == "question" and regenerated.drafts == [json.loads(bad)["message"]]
    assert regenerated.guard[0][0].startswith("causal")
    assert fallback.kind == "fallback" and fallback.text == guard.FALLBACK
    assert len(fallback.drafts) == 2
    note = backend.calls[1][0][-1]["content"]
    assert "not allowed" in note  # the regeneration is told what was wrong
    asked = [t.text for t in record.turns if t.kind in ("question", "fallback")]
    assert all(not guard.problems(text) for text in asked)


def test_red_flag_gets_a_referral_and_ends():
    backend = ScriptedBackend([EMPTY])
    record = converse(LLMAgent(backend), backend, make_summary(), lambda t: "I had chest pain", "c")
    assert record.turns[-1].kind == "referral" and record.turns[-1].text == REFERRAL
    assert len(backend.calls) == 1  # only the extraction


def test_checklist_agent_asks_its_questions_in_order():
    backend = ScriptedBackend([EMPTY])
    record = converse(ChecklistAgent(), backend, make_summary(), lambda t: "no", "c")
    asked = [t.text for t in record.turns if t.kind == "question"]
    assert asked == list(ChecklistAgent.QUESTIONS) and record.agent == "checklist"


def test_extraction_retries_unparseable_output_then_gives_up():
    good = json.dumps({"mentions": [mention()]})
    turns = [Turn(role="person", kind="answer", text="two coffees at 4")]
    result, attempts, error, _, raw, repairs = extract(ScriptedBackend(["{}", good]), turns)
    assert attempts == 2 and error is None and result.factors[0].factor == "caffeine"
    assert raw == ["{}", good] and repairs == []
    backend = ScriptedBackend(["not json"] * 3)
    result, attempts, error, _, _, _ = extract(backend, turns)
    assert attempts == 3 and error and result.factors == [] and not result.nothing_unusual
    assert "invalid" in backend.calls[1][0][-1]["content"]
    assert all(m["content"] != "not json" for m in backend.calls[1][0])  # output not echoed
    assert backend.max_tokens == [EXTRACTION_TOKENS] * 3


def test_extraction_sees_only_the_questions_and_answers():
    summary = make_summary()
    turns = [
        Turn(role="agent", kind="opening", text=opening(summary)),
        Turn(role="person", kind="answer", text="two coffees at 4"),
        Turn(role="agent", kind="question", text="Anything else?"),
        Turn(role="person", kind="answer", text="No"),
        Turn(role="agent", kind="closing", text=CLOSING),
    ]
    shown = answers_transcript(turns)
    assert shown.startswith(f"Assistant: {OPEN_QUESTION}") and "Anything else?" in shown
    assert describe(summary) not in shown and DISCLAIMER not in shown
    backend = ScriptedBackend([json.dumps({"mentions": [mention()]})])
    extract(backend, turns)
    assert backend.calls[0][0][-1]["content"] == shown


def test_conversations_advanced_together_match_one_at_a_time():
    def agent_reply(messages, schema):
        answers = sum(m["role"] == "user" for m in messages)
        return step(f"Question {answers}?", end=answers >= 3)

    def person(turns):
        return f"reply to: {turns[-1].text[-12:]}"

    one = converse(
        LLMAgent(ScriptedBackend(agent_reply)),
        ScriptedBackend([EMPTY]),
        make_summary(),
        person,
        "c",
    )
    together = [Conversation(make_summary(), "c") for _ in range(3)]
    agent = LLMAgent(ScriptedBackend(agent_reply))
    while active := [c for c in together if not c.finished]:
        for c in active:
            c.answer(person(c.turns))
        for c in active:
            if not c.finished:
                c.ask(agent.next_question(c.summary, c.turns, c.remaining))
    for c in together:
        assert [t.text for t in c.turns] == [t.text for t in one.turns]


# --- simulated people ----------------------------------------------------------------------------


def test_personas_are_seeded_stratified_and_spread_over_cases():
    cases = [f"case{i}" for i in range(4)]
    people = make_personas("test", 5, seed=1, case_ids=cases)
    assert people == make_personas("test", 5, seed=1, case_ids=cases)
    assert people != make_personas("test", 5, seed=2, case_ids=cases)
    assert len(people) == len(STYLES) * len(CAUSE_COUNTS) * 5
    cells = pd.Series([(p.style, len(p.causes)) for p in people]).value_counts()
    assert (cells == 5).all() and len(cells) == 12
    for p in people:
        named = [c.factor for c in p.causes if c.factor != "other"]
        assert len(named) == len(set(named))
        assert sum(c.factor == "other" for c in p.causes) <= 1
        assert all(c.certain == (p.style != "vague") for c in p.causes)
        assert len(p.ordinary) == 2
    assert pd.Series([p.case_id for p in people]).value_counts().tolist() == [15] * 4
    share = np.mean([c.factor == "other" for p in people for c in p.causes])
    assert 0.05 < share < 0.4
    dev = make_personas("dev", 1, seed=1, case_ids=cases)
    assert not {p.persona_id for p in dev} & {p.persona_id for p in people}


def test_simulated_user_sees_its_card_and_the_agent_as_the_other_side():
    person = make_personas("dev", 1, seed=0, case_ids=["c"])[5]
    backend = ScriptedBackend(["  hi  "])
    turns = [Turn(role="agent", kind="opening", text="Was anything different?")]
    assert SimulatedUser(person, backend)(turns) == "hi"
    messages, schema = backend.calls[0]
    assert schema is None and person.card() in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Was anything different?"}


def test_reticent_people_reveal_causes_only_when_asked():
    assert set(TOPIC_PATTERNS) == set(CAUSES) | {kind for _, kind, _ in OTHER_CAUSES}
    person = Persona(
        persona_id="r",
        style="reticent",
        causes=[
            HiddenCause(factor="alcohol", detail="wine", timing="evening", certain=True),
            HiddenCause(factor="stress", detail="argued", timing="evening", certain=True),
            HiddenCause(factor="other", detail="nap", timing="afternoon", certain=True, kind="nap"),
        ],
        ordinary=["read a book", "tidied the flat"],
        case_id="c",
        seed=0,
    )

    def after(*questions):
        turns = [
            Turn(role="agent", kind="opening", text="facts... heart rate. Anything different?")
        ]
        for q in questions:
            turns += [
                Turn(role="person", kind="answer", text="..."),
                Turn(role="agent", kind="question", text=q),
            ]
        return unlocked(person, turns)

    assert after() == [0]  # the opening unlocks one
    assert after("Were you feeling stressed?") == [0, 1]
    assert after("Was your schedule different from usual?") == [0]  # named topic, none held
    assert after("Was anything else different that day?") == [0, 1]
    assert after("Did you take a nap?", "Anything else?") == [0, 2, 1]
    user = SimulatedUser(person, ScriptedBackend(["ok"]))
    card = user.messages([Turn(role="agent", kind="opening", text="Anything different?")])
    assert "wine" in card[0]["content"] and "argued" not in card[0]["content"]


# --- cases ---------------------------------------------------------------------------------------


def test_case_selection_uses_each_subject_once():
    rows = []
    for s in range(20):
        for version, channel in [
            ("clean", "hr_window"),
            ("onset+60", "onset"),
            ("frag8", "frag"),
            ("hr+10_30", "hr_window"),
            ("frag8_wrongchannel", "hr_window"),
        ]:
            rows.append(
                {
                    "source": "gru",
                    "baseline": "personal",
                    "subject": f"S{s:02d}",
                    "night": 3,
                    "version": version.replace("_wrongchannel", ""),
                    "top_channel": channel,
                    "flagged": version != "clean" or s < 2,
                }
            )
    results = pd.DataFrame(rows)
    picked = select(results, seed=0, n_test=3, n_dev=1)
    assert picked["subject"].is_unique
    assert (picked.loc[picked["kind"] == "real", "split"] == "test").all()
    assert len(picked[picked["kind"] == "real"]) == 2
    assert (picked.groupby(["kind", "split"]).size().drop("real") == [1, 3] * 3).all()
    frag = picked[picked["kind"] == "frag"]
    assert (frag["top_channel"] == "frag").all()  # the flag must come from the injected channel
    assert picked.equals(select(results, seed=0, n_test=3, n_dev=1))
    with pytest.raises(ValueError, match="subjects available"):
        select(results, seed=0, n_test=6, n_dev=2)


# --- judge and metrics ---------------------------------------------------------------------------


def test_quote_check_ignores_case_punctuation_and_spacing():
    said = ["Yes - I had TWO coffees, at 4pm!", "no wine"]
    assert quote_found("two coffees at 4pm", said)
    assert quote_found("I had two  coffees", said)
    assert not quote_found("three coffees", said)
    assert not quote_found("...", said)
    # lightly reworded: 80% of the words in one turn
    assert quote_found("I drank two coffees at 4pm", ["Yes - I had two coffees at 4pm"])
    assert not quote_found("I drank some red wine", ["no wine for me"])
    assert not quote_verbatim("I drank two coffees at 4pm", ["Yes - I had two coffees at 4pm"])


def make_record(agent="a", factors=(), guarded=False) -> FollowUpRecord:
    q2 = Turn(
        role="agent",
        kind="question",
        text="Any caffeine?",
        drafts=["Did coffee cause it?"] if guarded else [],
        guard=[["causal: cause"]] if guarded else [],
    )
    turns = [
        Turn(role="agent", kind="opening", text="facts... Was anything different?"),
        Turn(role="person", kind="answer", text="I had a big argument with my partner"),
        q2,
        Turn(role="person", kind="answer", text="Yes, two coffees at 4pm"),
        Turn(role="agent", kind="question", text="You probably drank wine too?"),
        Turn(role="person", kind="answer", text="No wine"),
        Turn(role="agent", kind="closing", text=CLOSING),
    ]
    return FollowUpRecord(
        case_id="c",
        agent=agent,
        model="m",
        prompt_version="v",
        turns=turns,
        result=FollowUpResult(factors=list(factors)),
        extraction_attempts=1,
    )


def make_persona(pid="p1") -> Persona:
    return Persona(
        persona_id=pid,
        style="forthcoming",
        causes=[
            HiddenCause(factor="stress", detail="argued", timing="evening", certain=True),
            HiddenCause(factor="caffeine", detail="coffee", timing="afternoon", certain=True),
            HiddenCause(factor="alcohol", detail="wine", timing="evening", certain=True),
        ],
        ordinary=["read a book", "called my parents"],
        case_id="c",
        seed=0,
    )


def make_judgement(record, pid="p1", supported=(True, True, True)) -> Judgement:
    person = [t.text for t in record.turns if t.role == "person"]
    return Judgement(
        persona_id=pid,
        agent=record.agent,
        questions=[
            QuestionLabel(question=1, label="targeted", causal=False, diagnosis=False),
            QuestionLabel(question=2, label="leading", causal=False, diagnosis=False),
        ],
        causes=[
            CauseDisclosure(cause=1, first_answer=1),
            CauseDisclosure(cause=2, first_answer=2),
            CauseDisclosure(cause=3, first_answer=None),
        ],
        extra_claims=[],
        factors=[
            FactorSupport(factor=k + 1, supported=s, matches_cause=None)
            for k, s in enumerate(supported[: len(record.result.factors)])
        ],
        quote_found=[quote_found(f.quote, person) for f in record.result.factors],
    )


def test_counts_for_one_conversation():
    record = make_record(
        factors=[
            factor("stress", quote="big argument with my partner", timing="evening"),
            factor("caffeine", quote="two coffees at 4pm", timing="evening"),
            factor("alcohol", quote="had some wine"),  # not in the person's words
        ],
        guarded=True,
    )
    row = counts(record, make_judgement(record), make_persona())
    assert judged_questions(record) == [2, 4]
    assert row["hidden"] == 3 and row["matched"] == 2 and row["factors_matched"] == 2
    assert row["factors"] == 3 and row["factors_unsupported"] == 1  # quote not found
    assert row["leading"] == 1 and row["leading_before_guard"] == 2 and row["guarded"] == 1
    assert row["disclosed"] == 2 and row["disclosed_at_1"] == 1 and row["disclosed_at_2"] == 1
    assert row["matched_disclosed"] == 2 and row["timing_agree"] == 1
    assert row["confidence_agree"] == 2 and row["no_cause"] == 0


def test_a_judged_match_captures_across_categories():
    record = make_record(
        factors=[factor("illness", quote="big argument with my partner")],  # filed oddly
    )
    judgement = make_judgement(record)
    judgement.factors[0].matches_cause = 1  # the judge: this is the argument (stress)
    row = counts(record, judgement, make_persona())
    assert row["matched"] == 1 and row["matched_strict"] == 0 and row["factors_matched"] == 1


def test_summary_rates_are_pooled_and_paired():
    rows = []
    for pid in ["p1", "p2", "p3", "p4"]:
        record = make_record(factors=[factor("stress", quote="big argument with my partner")])
        for agent in ["ref", "same"]:
            record = record.model_copy(update={"agent": agent})
            rows.append(counts(record, make_judgement(record, pid), make_persona(pid)))
    table = pd.DataFrame(rows)
    out = summarize(table, "ref", n_boot=200, seed=0)
    ref = out["agents"]["ref"]
    assert ref["cause_recall"]["value"] == pytest.approx(1 / 3)
    assert ref["leading_question_rate"]["value"] == 0.5
    assert ref["turns_to_elicit_median"]["value"] == 1.0
    assert ref["disclosed_by_turn"][:2] == pytest.approx([1 / 3, 2 / 3])
    diff = out["differences"]["same"]["cause_recall"]
    assert diff["value"] == 0 and diff["lo"] == 0 and diff["hi"] == 0  # identical agents


def test_judge_parses_labels_and_rejects_wrong_numbering():
    record = make_record(factors=[factor("stress", quote="big argument with my partner")])
    labels = {
        "questions": [
            {"question": 1, "label": "targeted", "causal": False, "diagnosis": False},
            {"question": 2, "label": "leading", "causal": False, "diagnosis": False},
        ]
    }
    disclosures = {
        "causes": [
            {"cause": 1, "first_answer": 1},
            {"cause": 2, "first_answer": 2},
            {"cause": 3, "first_answer": None},
        ],
        "factors": [{"factor": 1, "supported": True, "matches_cause": 1}],
    }
    card = {
        "answers": [
            {"answer": 1, "new_claims": [], "denied": []},
            {"answer": 2, "new_claims": ["a late bedtime"], "denied": []},
            {"answer": 3, "new_claims": [], "denied": [3]},
        ]
    }
    replies = [json.dumps(labels), json.dumps(disclosures), json.dumps(card)]
    backend = ScriptedBackend(replies)
    found = judge(backend, record, make_persona())
    assert found.errors == [] and [q.label for q in found.questions] == ["targeted", "leading"]
    assert found.quote_found == [True]
    assert found.extra_claims == ["A2: a late bedtime"] and found.denied_causes == [3]
    prompt = backend.calls[0][0][1]["content"]
    assert "[Q1] Assistant: Any caffeine?" in prompt and "[A1] Person:" in prompt
    assert "[C3] alcohol: wine" in backend.calls[2][0][1]["content"]
    wrong = disclosures | {"causes": disclosures["causes"][:2]}
    backend = ScriptedBackend([json.dumps(labels), *[json.dumps(wrong)] * 3, json.dumps(card)])
    found = judge(backend, record, make_persona())
    assert found.errors and found.causes == [] and found.questions and found.denied_causes
    bad_card = {"answers": [{"answer": 1, "new_claims": [], "denied": [4]}]}  # no cause 4
    backend = ScriptedBackend([*replies[:2], *[json.dumps(bad_card)] * 3])
    found = judge(backend, record, make_persona())
    assert found.errors[0].startswith("fidelity") and found.extra_claims == []


def test_card_check_counts_and_planting():
    record = make_record(factors=[factor("stress", quote="big argument with my partner")])
    judgement = make_judgement(record)
    judgement.extra_claims, judgement.denied_causes = ["A2: a late bedtime"], [1]
    row = counts(record, judgement, make_persona())
    assert row["drift"] == row["denial"] == row["fidelity_judged_with_causes"] == 1
    judgement.errors = ["fidelity: answers must be numbered [1, 2, 3], got [1]"]
    row = counts(record, judgement, make_persona())
    assert row["fidelity_judged"] == row["drift"] == row["denial"] == 0
    assert row["matched"] == 1  # the other calls still count
    planted = plant(record, "I also had a whisky.")
    answers = [t.text for t in planted.turns if t.role == "person"]
    assert (
        answers[-1] == "No wine I also had a whisky."
        and answers[:2] == [t.text for t in record.turns if t.role == "person"][:2]
    )
    assert record.turns[-2].text == "No wine"  # the original is unchanged


# --- the evaluation run --------------------------------------------------------------------------


def scripted_world(model: str):
    """Agents ask two questions then end; people mention coffee; extraction reports it; the
    judge marks everything targeted and supported."""

    def reply(messages, schema):
        text = json.dumps(schema) if schema else ""
        if schema is None:  # a simulated person
            return "I had two coffees at 4pm"
        if '"message"' in text:
            asked = sum(m["role"] == "assistant" for m in messages)
            return step("Anything else?", end=asked >= 3)
        if '"mentions"' in text:
            return json.dumps({"mentions": [mention(quote="two coffees at 4pm")]})
        n_q = messages[-1]["content"].count("[Q")
        if '"questions"' in text:
            return json.dumps(
                {
                    "questions": [
                        {
                            "question": k + 1,
                            "label": "targeted",
                            "causal": False,
                            "diagnosis": False,
                        }
                        for k in range(n_q)
                    ]
                }
            )
        content = messages[-1]["content"]
        n_c = content.count("[C")
        n_f = content.count("[F")
        if '"new_claims"' in text:  # the card check notices planted errors in the last answer
            n_a = content.count("[A")
            last = content.rsplit("[A", 1)[-1]
            return json.dumps(
                {
                    "answers": [
                        {
                            "answer": k + 1,
                            "new_claims": ["planted"] if k + 1 == n_a and "I also" in last else [],
                            "denied": list(range(1, n_c + 1))
                            if k + 1 == n_a and "not true" in last
                            else [],
                        }
                        for k in range(n_a)
                    ]
                }
            )
        return json.dumps(
            {
                "causes": [{"cause": k + 1, "first_answer": 1} for k in range(n_c)],
                "factors": [
                    {"factor": k + 1, "supported": True, "matches_cause": None} for k in range(n_f)
                ],
            }
        )

    return ScriptedBackend(reply, model=model)


def test_evaluation_run_resumes_and_scores(tmp_path):
    summary = make_summary()
    cases = [
        AgentCase(
            case_id=f"S{k}-3-clean",
            split="dev",
            kind="real",
            subject=f"S{k}",
            night=3,
            version="clean",
            summary=summary,
        )
        for k in range(2)
    ]
    cfg = AgentEvalConfig(
        name="agent_t",
        split="dev",
        persona_seed=3,
        personas_per_cell=1,
        agents=[
            AgentSpec(name="m1", model="m1"),
            AgentSpec(name="checklist", kind="checklist", model="m1"),
        ],
        reference="m1",
        batch_size=5,
        n_boot=50,
    )
    personas = make_personas("dev", 1, 3, [c.case_id for c in cases])
    run_dir = start_run(cfg, tmp_path, cases, personas, ollama={})
    run_conversations(cfg, run_dir, cases, personas[:4], scripted_world, log=lambda _: None)
    run_conversations(cfg, run_dir, cases, personas, scripted_world, log=lambda _: None)
    lines = (run_dir / "conversations.jsonl").read_text().splitlines()
    keys = [(json.loads(x)["agent"], json.loads(x)["persona_id"]) for x in lines]
    assert len(keys) == len(set(keys)) == 2 * len(personas)  # resumed without repeats
    run_judge(cfg, run_dir, personas, scripted_world, log=lambda _: None)
    metrics = compute_metrics(cfg, run_dir)
    assert metrics["complete"]
    m1, checklist = metrics["agents"]["m1"], metrics["agents"]["checklist"]
    assert m1["conversations"] == checklist["conversations"] == len(personas)
    caffeine_personas = sum(any(c.factor == "caffeine" for c in p.causes) for p in personas)
    hidden = sum(len(p.causes) for p in personas)
    assert m1["cause_recall"]["num"] == caffeine_personas and m1["cause_recall"]["den"] == hidden
    assert m1["leading_question_rate"]["value"] == 0
    assert m1["no_cause_factor_rate"]["value"] == 1.0  # coffee reported on ordinary days too
    assert m1["simulator_drift_rate"]["value"] == 0 and m1["simulator_denial_rate"]["den"] > 0
    assert (run_dir / "counts.parquet").is_file()
    probes = metrics["judge_probes"]
    with_causes = sum(bool(p.causes) for p in personas)
    assert probes["claim"] == {
        "n": len(personas),
        "detected": len(personas),
        "rate": 1.0,
        "errors": 0,
    }
    assert probes["denial"]["n"] == with_causes and probes["denial"]["rate"] == 1.0
    before = (run_dir / "judge_probes.jsonl").read_text()
    run_probes(cfg, run_dir, personas, scripted_world, log=lambda _: None)
    assert (run_dir / "judge_probes.jsonl").read_text() == before  # resumed: nothing repeated

    sheet = sample(run_dir, n_questions=4, n_factors=4, seed=0)
    questions = pd.read_csv(sheet / "questions.csv")
    assert "agent" not in questions.columns and "label" in questions.columns
    assert questions["label"].isna().all()
    questions["label"], questions["causal"], questions["diagnosis"] = "targeted", "n", "n"
    questions.loc[0, "label"] = "leading"
    questions.to_csv(sheet / "questions.csv", index=False)
    factors = pd.read_csv(sheet / "factors.csv").assign(supported="y")
    factors.to_csv(sheet / "factors.csv", index=False)
    result = agreement(run_dir)
    assert result["questions"] == 4 and result["factors"] == 4
    assert result["kappa_leading"] == 0  # the judge said targeted throughout
    assert np.isnan(result["kappa_supported"]) and result["agreement_supported"] == 1
    assert (
        result["human_leading_rate"]["m1"]["n"] + result["human_leading_rate"]["checklist"]["n"]
        == 4
    )
    assert json.loads((sheet / "agreement.json").read_text())["factors"] == 4


# --- live backend (optional) ---------------------------------------------------------------------


def _ollama_has(model: str) -> bool:
    try:
        tags = httpx.get(f"{AgentSettings().ollama_url}/api/tags", timeout=2).json()["models"]
    except (httpx.HTTPError, KeyError, ValueError):
        return False
    return any(m["name"] == model for m in tags)


@pytest.mark.llm
@pytest.mark.skipif(not _ollama_has("gemma3:4b"), reason="needs Ollama with gemma3:4b")
def test_live_extraction_returns_valid_json():
    from sleepwatch.agent.backend import OllamaBackend

    turns = [
        Turn(role="agent", kind="opening", text=OPEN_QUESTION),
        Turn(role="person", kind="answer", text="I had two large coffees around 4 pm."),
    ]
    result, attempts, error, _, _, _ = extract(OllamaBackend("gemma3:4b"), turns, seed=1)
    assert error is None and attempts <= 3
    assert isinstance(result, FollowUpResult)
