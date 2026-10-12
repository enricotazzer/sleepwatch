"""Phase 4 metrics (docs/decisions.md 49): counts per conversation, then pooled rates with 95%
CIs from a bootstrap over personas, paired across agents.

A hidden cause is *captured* by an extracted factor that is supported (its quote is the person's
words and the judge says those words report it) and that the judge matches to the cause or that
has the cause's checklist category, one factor per cause. *Strict* recall also needs the right
category. Conversations whose judge call failed drop out of the rates that call feeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sleepwatch.agent.judge import Judgement, judged_questions, quote_verbatim
from sleepwatch.agent.personas import Persona
from sleepwatch.agent.schema import FollowUpRecord

MAX_TURNS = 5
# metric: (numerator, denominator); every rate is pooled, sum(num) / sum(den).
RATES = {
    "cause_recall": ("matched", "hidden"),
    "cause_precision": ("factors_matched", "factors"),
    "hallucinated_factor_rate": ("factors_unsupported", "factors"),
    "leading_question_rate": ("leading", "questions_judged"),
    "leading_before_guard_rate": ("leading_before_guard", "questions_judged"),
    "causal_statement_rate": ("causal", "questions_judged"),
    "diagnosis_statement_rate": ("diagnosis", "questions_judged"),
    "no_cause_factor_rate": ("no_cause_with_factor", "no_cause"),
    "timing_agreement": ("timing_agree", "matched"),
    "confidence_agreement": ("confidence_agree", "matched"),
    "disclosure_rate": ("disclosed", "hidden"),
    "extraction_recall_disclosed": ("matched_disclosed", "disclosed"),
    "strict_recall": ("matched_strict", "hidden"),
    "category_accuracy": ("matched_strict", "matched"),
    "recall_checklist": ("matched_checklist", "hidden_checklist"),
    "recall_other": ("matched_other", "hidden_other"),
    "schema_valid_first_try": ("valid_first", "conversations"),
    "extraction_failed": ("extraction_failed", "conversations"),
    "extraction_repaired": ("repaired", "conversations"),
    "verbatim_quote_rate": ("quotes_verbatim", "factors"),
    "guard_trigger_rate": ("guarded", "questions"),
    "simulator_drift_rate": ("drift", "fidelity_judged"),
    "simulator_denial_rate": ("denial", "fidelity_judged_with_causes"),
}
PRIMARY = (
    "cause_recall",
    "cause_precision",
    "hallucinated_factor_rate",
    "leading_question_rate",
)


def counts(record: FollowUpRecord, judgement: Judgement, persona: Persona) -> dict:
    """Numerators and denominators of every rate for one conversation."""
    hidden = persona.causes
    factors = record.result.factors
    asked = judged_questions(record)
    q_ok = not any(e.startswith("questions") for e in judgement.errors)
    d_ok = not any(e.startswith("disclosures") for e in judgement.errors)
    f_ok = not any(e.startswith("fidelity") for e in judgement.errors)
    row = {
        "persona_id": persona.persona_id,
        "agent": record.agent,
        "style": persona.style,
        "n_causes": len(hidden),
        "conversations": 1,
        "questions": len(asked),
        "guarded": sum(bool(record.turns[i].guard) for i in asked),
        "valid_first": int(record.extraction_attempts == 1 and record.extraction_error is None),
        "extraction_failed": int(record.extraction_error is not None),
        "repaired": int(bool(record.extraction_repairs)),
        "questions_judged": len(asked) if q_ok else 0,
        "disclosures_judged": int(d_ok),
        "fidelity_judged": int(f_ok),
        "fidelity_judged_with_causes": int(f_ok and bool(hidden)),
        "drift": int(f_ok and bool(judgement.extra_claims)),
        "denial": int(f_ok and bool(judgement.denied_causes)),
        "judge_errors": len(judgement.errors),
    }
    leading = before = causal = diagnosis = 0
    if q_ok:
        for label, i in zip(judgement.questions, asked, strict=True):
            is_leading = label.label == "leading"
            rejected = any(
                h.startswith(("causal", "observation", "diagnosis"))
                for hits in record.turns[i].guard
                for h in hits
            )
            leading += is_leading
            before += is_leading or rejected
            causal += label.causal
            diagnosis += label.diagnosis
    row |= {
        "leading": leading,
        "leading_before_guard": before,
        "causal": causal,
        "diagnosis": diagnosis,
    }
    zero = dict.fromkeys(
        [
            "hidden",
            "hidden_checklist",
            "hidden_other",
            "matched",
            "matched_checklist",
            "matched_other",
            "matched_strict",
            "matched_disclosed",
            "disclosed",
            "factors",
            "factors_matched",
            "factors_unsupported",
            "quotes_verbatim",
            "timing_agree",
            "confidence_agree",
            "no_cause",
            "no_cause_with_factor",
        ]
        + [f"disclosed_at_{k}" for k in range(1, MAX_TURNS + 1)],
        0,
    )
    row |= zero
    if not d_ok:
        return row
    supported = [
        s.supported and found
        for s, found in zip(judgement.factors, judgement.quote_found, strict=True)
    ]
    # A supported factor captures a hidden cause the judge matches it to, or one with the same
    # checklist category; each cause and each factor is used once, in the factors' order.
    captured: dict[int, int] = {}  # hidden cause index -> factor index
    for k, (factor, support) in enumerate(zip(factors, judgement.factors, strict=True)):
        if not supported[k]:
            continue
        candidates = [support.matches_cause - 1] if support.matches_cause is not None else []
        candidates += [
            c
            for c, cause in enumerate(hidden)
            if cause.factor == factor.factor and cause.factor != "other"
        ]
        for c in candidates:
            if c not in captured:
                captured[c] = k
                break
    disclosed_at = {c.cause - 1: c.first_answer for c in judgement.causes}
    for c, cause in enumerate(hidden):
        is_other = cause.factor == "other"
        row["hidden"] += 1
        row["hidden_other" if is_other else "hidden_checklist"] += 1
        at = disclosed_at.get(c)
        if at is not None:
            row["disclosed"] += 1
            row[f"disclosed_at_{min(at, MAX_TURNS)}"] += 1
        if c in captured:
            factor = factors[captured[c]]
            row["matched"] += 1
            row["matched_strict"] += factor.factor == cause.factor
            row["matched_other" if is_other else "matched_checklist"] += 1
            row["matched_disclosed"] += at is not None
            row["timing_agree"] += factor.timing == cause.timing
            row["confidence_agree"] += (factor.confidence == "certain") == cause.certain
    row["factors"] = len(factors)
    row["factors_matched"] = len(captured)
    row["factors_unsupported"] = sum(not s for s in supported)
    person = [t.text for t in record.turns if t.role == "person"]
    row["quotes_verbatim"] = sum(quote_verbatim(f.quote, person) for f in factors)
    row["no_cause"] = int(not hidden)
    row["no_cause_with_factor"] = int(not hidden and bool(factors))
    return row


def _ci(boot: np.ndarray) -> tuple[float, float]:
    finite = boot[np.isfinite(boot)]
    if not len(finite):
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def _median_turn(disclosed: np.ndarray) -> np.ndarray:
    """Median of the turn of first disclosure from per-turn counts (``... x MAX_TURNS``)."""
    total = disclosed.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cumulative = disclosed.cumsum(axis=-1) / total
    median = (cumulative >= 0.5).argmax(axis=-1) + 1.0
    return np.where(total[..., 0] > 0, median, np.nan)


def summarize(table: pd.DataFrame, reference: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Pooled rates per agent with 95% CIs and paired differences against ``reference``.

    One bootstrap resample draws personas with replacement and is shared by every agent, so the
    differences are paired."""
    personas = sorted(table["persona_id"].unique())
    agents = list(dict.fromkeys(table["agent"]))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(personas), np.full(len(personas), 1 / len(personas)), n_boot)
    turn_cols = [f"disclosed_at_{k}" for k in range(1, MAX_TURNS + 1)]
    numeric = [c for c in table.columns if c not in ("persona_id", "agent", "style")]
    per_agent = {
        a: table[table["agent"] == a].groupby("persona_id")[numeric].sum().reindex(personas)
        for a in agents
    }
    out: dict = {"personas": len(personas), "n_boot": n_boot, "reference": reference}
    boots: dict[str, dict[str, np.ndarray]] = {}
    agents_out = {}
    for a, frame in per_agent.items():
        frame = frame.fillna(0)
        res: dict = {"conversations": int(frame["conversations"].sum())}
        boots[a] = {}
        for metric, (num, den) in RATES.items():
            n, d = frame[num].to_numpy(float), frame[den].to_numpy(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                value = n.sum() / d.sum() if d.sum() else float("nan")
                boot = (weights @ n) / (weights @ d)
            boots[a][metric] = boot
            lo, hi = _ci(boot)
            res[metric] = {"value": value, "lo": lo, "hi": hi, "num": n.sum(), "den": d.sum()}
        disclosed = frame[turn_cols].to_numpy(float)
        hidden = frame["hidden"].to_numpy(float)
        median = _median_turn(disclosed.sum(axis=0))
        boot = _median_turn(weights @ disclosed)
        boots[a]["turns_to_elicit_median"] = boot
        lo, hi = _ci(boot)
        res["turns_to_elicit_median"] = {"value": float(median), "lo": lo, "hi": hi}
        res["disclosed_by_turn"] = [
            float(disclosed[:, :k].sum() / hidden.sum()) if hidden.sum() else float("nan")
            for k in range(1, MAX_TURNS + 1)
        ]
        res["judge_errors"] = int(frame["judge_errors"].sum())
        agents_out[a] = res
    out["agents"] = agents_out
    if reference in boots:
        diffs = {}
        for a in agents:
            if a == reference:
                continue
            diffs[a] = {}
            for metric in [*RATES, "turns_to_elicit_median"]:
                boot = boots[a][metric] - boots[reference][metric]
                lo, hi = _ci(boot)
                value = agents_out[a][metric]["value"] - agents_out[reference][metric]["value"]
                diffs[a][metric] = {"value": value, "lo": lo, "hi": hi}
        out["differences"] = diffs
    out["strata"] = {by: strata(table, by) for by in ("style", "n_causes")}
    return out


def strata(table: pd.DataFrame, by: str) -> dict:
    """Point values of the primary rates per agent and level of ``by``."""
    out: dict = {}
    for (agent, level), frame in table.groupby(["agent", by], sort=False):
        row = {}
        for metric in (*PRIMARY, "disclosure_rate", "recall_other"):
            num, den = RATES[metric]
            d = frame[den].sum()
            row[metric] = float(frame[num].sum() / d) if d else float("nan")
        row["conversations"] = int(frame["conversations"].sum())
        out.setdefault(agent, {})[str(level)] = row
    return out
