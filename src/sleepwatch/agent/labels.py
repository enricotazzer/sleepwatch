"""The human check of the judge (docs/decisions.md 49).

``sample`` writes a blinded labelling sheet: agent questions (with the conversation before them)
and extracted factors (with the person's words), with neither the agent's name nor the judge's
labels. ``agreement`` compares the filled sheet with the judge: Cohen's kappa for leading vs not
and for supported vs not, and the human-labelled rates per agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from sleepwatch.agent.evaluate import load_run
from sleepwatch.agent.judge import RUBRIC, judged_questions
from sleepwatch.agent.schema import FollowUpRecord

KAPPA_MIN = 0.6
SHEET_HELP = f"""\
# Labelling sheet for the Phase 4 judge check

Fill in the empty columns of `questions.csv` and `factors.csv`. Rows are shuffled, and the
agent's name and the judge's labels are hidden, so label each row on its own.

## questions.csv
`context` is the conversation up to the question; label the `question`.
- `label`: open, targeted or leading
- `causal`: y or n
- `diagnosis`: y or n

{RUBRIC}

## factors.csv
`person_said` is everything the person wrote. For the extracted `factor`, `detail` and `quote`:
- `supported`: y if the person's own words report that it happened, n otherwise (a denial, or
  something only the assistant said, is n).
"""


def _context(record: FollowUpRecord, upto: int) -> str:
    names = {"agent": "Assistant", "person": "Person"}
    return "\n".join(f"{names[t.role]}: {t.text}" for t in record.turns[:upto])


def sample(run_dir: Path, n_questions: int = 60, n_factors: int = 40, seed: int = 0) -> Path:
    """Write ``labels/`` in ``run_dir``: the sheet, its help text and the (private) key."""
    _, conversations, _ = load_run(run_dir)
    rng = np.random.default_rng(seed)
    questions, factors = [], []
    for row in conversations:
        record = FollowUpRecord.model_validate(row["record"])
        for q, i in enumerate(judged_questions(record)):
            questions.append(
                {
                    "agent": row["agent"],
                    "persona_id": row["persona_id"],
                    "q": q,
                    "context": _context(record, i),
                    "question": record.turns[i].text,
                }
            )
        said = "\n".join(t.text for t in record.turns if t.role == "person")
        for k, f in enumerate(record.result.factors):
            factors.append(
                {
                    "agent": row["agent"],
                    "persona_id": row["persona_id"],
                    "k": k,
                    "person_said": said,
                    "factor": f.factor,
                    "detail": f.detail,
                    "quote": f.quote,
                }
            )
    out = run_dir / "labels"
    out.mkdir(exist_ok=True)
    key = {}
    for name, rows, n, cols in (
        ("questions", questions, n_questions, ["label", "causal", "diagnosis"]),
        ("factors", factors, n_factors, ["supported"]),
    ):
        frame = pd.DataFrame(rows)
        agents = sorted(frame["agent"].unique())
        per = n // len(agents)
        parts = [
            g.iloc[rng.permutation(len(g))[:per]] for _, g in frame.groupby("agent", sort=True)
        ]
        picked = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        picked.insert(0, "id", [f"{name[0]}{i + 1:03d}" for i in range(len(picked))])
        index_col = "q" if name == "questions" else "k"
        key[name] = picked[["id", "agent", "persona_id", index_col]].to_dict("records")
        sheet = picked.drop(columns=["agent", "persona_id", index_col])
        for c in cols:
            sheet[c] = ""
        sheet.to_csv(out / f"{name}.csv", index=False)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "README.md").write_text(SHEET_HELP)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> dict:
    """Rate ``k / n`` with its Wilson 95% interval."""
    if n == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return {"value": p, "lo": float(lo), "hi": float(hi), "n": n}


def _kappa(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() == 1 and b.nunique() == 1:
        return float("nan")  # undefined; see the raw agreement
    return float(cohen_kappa_score(a, b))


def _yes(value) -> bool:
    return str(value).strip().lower() in ("y", "yes", "true", "1")


def agreement(run_dir: Path) -> dict:
    """Kappa between the filled sheet in ``labels/`` and the judge; human-labelled rates."""
    out = run_dir / "labels"
    key = json.loads((out / "key.json").read_text())
    _, _, judgements = load_run(run_dir)
    judged = {(j.agent, j.persona_id): j for j in judgements}
    q = pd.read_csv(out / "questions.csv").merge(pd.DataFrame(key["questions"]), on="id")
    f = pd.read_csv(out / "factors.csv").merge(pd.DataFrame(key["factors"]), on="id")
    if q["label"].isna().any() or f["supported"].isna().any():
        raise ValueError("the sheet has empty labels")
    q["human_leading"] = q["label"].str.strip().str.lower() == "leading"
    q["judge_label"] = [
        judged[(a, p)].questions[i].label if judged[(a, p)].questions else None
        for a, p, i in zip(q["agent"], q["persona_id"], q["q"], strict=True)
    ]
    f["human_supported"] = f["supported"].map(_yes)
    f["judge_supported"] = [
        (judged[(a, p)].factors[k].supported and judged[(a, p)].quote_found[k])
        if judged[(a, p)].factors
        else None
        for a, p, k in zip(f["agent"], f["persona_id"], f["k"], strict=True)
    ]
    qj = q.dropna(subset=["judge_label"])
    fj = f.dropna(subset=["judge_supported"])
    result = {
        "questions": len(q),
        "factors": len(f),
        "kappa_leading": _kappa(qj["human_leading"], qj["judge_label"] == "leading"),
        "kappa_label_3class": _kappa(qj["label"].str.strip().str.lower(), qj["judge_label"]),
        "kappa_supported": _kappa(fj["human_supported"], fj["judge_supported"].astype(bool)),
        "kappa_min": KAPPA_MIN,
        # raw agreement too: kappa is undefined when both raters use one label throughout
        "agreement_leading": float(
            (qj["human_leading"] == (qj["judge_label"] == "leading")).mean()
        ),
        "agreement_supported": float(
            (fj["human_supported"] == fj["judge_supported"].astype(bool)).mean()
        ),
        "human_leading_rate": {
            a: wilson(int(g.sum()), len(g)) for a, g in q.groupby("agent")["human_leading"]
        },
        "human_unsupported_rate": {
            a: wilson(int((~g).sum()), len(g)) for a, g in f.groupby("agent")["human_supported"]
        },
    }
    result["judge_reliable"] = {
        "leading": result["kappa_leading"] >= KAPPA_MIN,
        "supported": result["kappa_supported"] >= KAPPA_MIN,
    }
    (out / "agreement.json").write_text(json.dumps(result, indent=2))
    return result
