"""Phase 4 evaluation with simulated people (docs/decisions.md 49).

``run_conversations`` writes one line per (agent, persona) to ``conversations.jsonl`` as it
goes, and ``run_judge`` one line per conversation to ``judgements.jsonl`` and then one per
planted error to ``judge_probes.jsonl``, so an interrupted run continues where it stopped.
``compute_metrics`` reads only those files.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from sleepwatch.agent import cases as cs
from sleepwatch.agent.agent import PROMPT_VERSION, ChecklistAgent, Conversation, LLMAgent
from sleepwatch.agent.backend import ChatBackend, OllamaBackend
from sleepwatch.agent.judge import PLANTED_CLAIM, PLANTED_DENIAL, Judgement, fidelity, judge, plant
from sleepwatch.agent.metrics import counts, summarize
from sleepwatch.agent.personas import CAUSES, Persona, SimulatedUser, make_personas
from sleepwatch.agent.schema import FollowUpRecord
from sleepwatch.provenance import git_revision, timestamp

BackendFactory = Callable[[str], ChatBackend]


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # as reported: the model tag, or "checklist"
    kind: Literal["llm", "checklist"] = "llm"
    model: str  # asks (llm) and extracts


class AgentEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    split: Literal["dev", "test"]
    anomaly_run: str = "anomaly_screened/20261006T141555Z"
    case_seed: int = 49
    persona_seed: int
    personas_per_cell: int
    agents: list[AgentSpec]
    reference: str = "gemma3:4b"
    simulator: str = "qwen3.5:9b"
    judge: str = "qwen3.5:9b"
    seed: int = 42  # agents and judge
    persona_temperature: float = 0.7
    max_questions: int = 5
    num_ctx: int = 8192
    batch_size: int = 20  # conversations advanced together (speed only, see run_conversations)
    n_boot: int = 2000
    n_probes: int = 20  # conversations given planted errors to test the judge's card check
    limit: int | None = None  # first personas only, for quick checks; recorded as partial

    @classmethod
    def from_yaml(cls, path: Path) -> AgentEvalConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def models(self) -> list[str]:
        return list(dict.fromkeys([*(a.model for a in self.agents), self.simulator, self.judge]))


def load_cases(cfg: AgentEvalConfig, results_dir: Path, processed_dir: Path) -> list[cs.AgentCase]:
    run_dir = results_dir / cfg.anomaly_run
    features = yaml.safe_load((run_dir / "config.yaml").read_text()).get("features", "v1")
    quality = pd.read_csv(processed_dir / f"quality_{features}.csv")
    return cs.build(run_dir, quality, cfg.case_seed)


def personas_for(cfg: AgentEvalConfig, cases: list[cs.AgentCase]) -> list[Persona]:
    ids = [c.case_id for c in cases if c.split == cfg.split]
    people = make_personas(cfg.split, cfg.personas_per_cell, cfg.persona_seed, ids)
    return people[: cfg.limit] if cfg.limit else people


def start_run(
    cfg: AgentEvalConfig,
    results_dir: Path,
    cases: list[cs.AgentCase],
    personas: list[Persona],
    ollama: dict,
) -> Path:
    run_dir = results_dir / cfg.name / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    )
    (run_dir / "cases.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in cases], indent=1)
    )
    (run_dir / "personas.json").write_text(
        json.dumps([p.model_dump(mode="json") for p in personas], indent=1)
    )
    info = {
        "created": timestamp(),
        "git": git_revision(),
        "prompt_version": PROMPT_VERSION,
        "partial": cfg.limit is not None,
        "ollama": ollama,
        "cases": {s: sum(c.split == s for c in cases) for s in ("dev", "test")},
        "personas": len(personas),
    }
    (run_dir / "run.json").write_text(json.dumps(info, indent=2))
    return run_dir


def check_resume(cfg: AgentEvalConfig, run_dir: Path) -> None:
    """A run continues only with the same config and the same commit."""
    saved = AgentEvalConfig.model_validate(yaml.safe_load((run_dir / "config.yaml").read_text()))
    if saved != cfg:
        raise ValueError(f"{run_dir} was started with a different config")
    started = json.loads((run_dir / "run.json").read_text())["git"]
    now = git_revision()
    if started["commit"] != now["commit"] or now["dirty"] != started["dirty"]:
        raise ValueError(f"{run_dir} was started at another commit or tree state")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, row: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def run_conversations(
    cfg: AgentEvalConfig,
    run_dir: Path,
    cases: list[cs.AgentCase],
    personas: list[Persona],
    backend: BackendFactory,
    log: Callable[[str], None] = print,
) -> None:
    """Each agent talks to every persona not yet in ``conversations.jsonl``.

    Up to ``cfg.batch_size`` conversations advance together: every person answers, then the
    agent asks, so each model stays loaded for a whole step instead of being swapped in and out of
    memory every turn. Every call is seeded and sees only its own conversation, so the batch size
    sets the speed, not what is said. A batch is saved when its extractions are done."""
    path = run_dir / "conversations.jsonl"
    done = {(r["agent"], r["persona_id"]) for r in _read_jsonl(path)}
    by_case = {c.case_id: c for c in cases}
    simulator = backend(cfg.simulator)
    for spec in cfg.agents:
        extractor = backend(spec.model)
        agent = LLMAgent(extractor, seed=cfg.seed) if spec.kind == "llm" else ChecklistAgent()
        todo = [p for p in personas if (spec.name, p.persona_id) not in done]
        for b in range(0, len(todo), cfg.batch_size):
            batch = todo[b : b + cfg.batch_size]
            talks = {
                p.persona_id: (
                    Conversation(by_case[p.case_id].summary, p.case_id, cfg.max_questions),
                    SimulatedUser(p, simulator, temperature=cfg.persona_temperature),
                )
                for p in batch
            }
            while active := [c for c, _ in talks.values() if not c.finished]:
                for conversation, user in talks.values():
                    if not conversation.finished:
                        start = time.perf_counter()
                        text = user(conversation.turns)
                        conversation.answer(text, time.perf_counter() - start)
                for conversation in active:
                    if not conversation.finished:
                        conversation.ask(
                            agent.next_question(
                                conversation.summary, conversation.turns, conversation.remaining
                            )
                        )
            for persona_id, (conversation, _) in talks.items():
                record = conversation.record(spec.name, extractor, seed=cfg.seed)
                _append(
                    path,
                    {
                        "agent": spec.name,
                        "persona_id": persona_id,
                        "record": record.model_dump(mode="json"),
                    },
                )
            log(f"{spec.name}: {min(b + cfg.batch_size, len(todo))}/{len(todo)} conversations")


def run_judge(
    cfg: AgentEvalConfig,
    run_dir: Path,
    personas: list[Persona],
    backend: BackendFactory,
    log: Callable[[str], None] = print,
) -> None:
    path = run_dir / "judgements.jsonl"
    done = {(r["agent"], r["persona_id"]) for r in _read_jsonl(path)}
    by_id = {p.persona_id: p for p in personas}
    judge_backend = backend(cfg.judge)
    rows = _read_jsonl(run_dir / "conversations.jsonl")
    for k, row in enumerate(rows):
        if (row["agent"], row["persona_id"]) in done:
            continue
        record = FollowUpRecord.model_validate(row["record"])
        found = judge(judge_backend, record, by_id[row["persona_id"]], seed=cfg.seed)
        _append(path, found.model_dump(mode="json"))
        log(f"judged {k + 1}/{len(rows)} {row['agent']} {row['persona_id']}")
    run_probes(cfg, run_dir, personas, backend, log=log)


def run_probes(
    cfg: AgentEvalConfig,
    run_dir: Path,
    personas: list[Persona],
    backend: BackendFactory,
    log: Callable[[str], None] = print,
) -> None:
    """Planted errors for the judge's card check. In ``cfg.n_probes`` of the reference agent's
    conversations, the person's last answer gets a claim that is not on the card (a checklist
    cause from a category the card does not have) and, if the card has causes, a denial of one
    of them. A probe is detected when the card check reports it for that answer."""
    path = run_dir / "judge_probes.jsonl"
    done = {(r["persona_id"], r["kind"]) for r in _read_jsonl(path)}
    by_id = {p.persona_id: p for p in personas}
    rows = [r for r in _read_jsonl(run_dir / "conversations.jsonl") if r["agent"] == cfg.reference]
    rng = np.random.default_rng(cfg.seed)
    plan = []  # drawn in full before skipping finished probes, so a resumed run plants the same
    for k in rng.permutation(len(rows))[: cfg.n_probes]:
        persona = by_id[rows[k]["persona_id"]]
        absent = [f for f in CAUSES if f not in {c.factor for c in persona.causes}]
        detail = CAUSES[absent[rng.integers(len(absent))]][rng.integers(4)][0]
        plan.append((rows[k], persona, "claim", PLANTED_CLAIM.format(detail=detail), None))
        if persona.causes:
            c = int(rng.integers(len(persona.causes)))
            text = PLANTED_DENIAL.format(detail=persona.causes[c].detail)
            plan.append((rows[k], persona, "denial", text, c + 1))
    judge_backend = backend(cfg.judge)
    for n, (row, persona, kind, text, cause) in enumerate(plan):
        if (persona.persona_id, kind) in done:
            continue
        record = plant(FollowUpRecord.model_validate(row["record"]), text)
        answers, error = fidelity(judge_backend, record, persona, seed=cfg.seed)
        detected = None
        if answers is not None:
            last = answers[-1]
            detected = bool(last.new_claims) if kind == "claim" else cause in last.denied
        _append(
            path,
            {
                "persona_id": persona.persona_id,
                "agent": row["agent"],
                "kind": kind,
                "planted": text,
                "cause": cause,
                "detected": detected,
                "error": error if answers is None else None,
            },
        )
        log(f"probe {n + 1}/{len(plan)} {kind} {persona.persona_id}: detected={detected}")


def probe_summary(run_dir: Path) -> dict:
    """Detection rate of the planted errors, per kind; failed judge calls are left out."""
    rows = pd.DataFrame(_read_jsonl(run_dir / "judge_probes.jsonl"))
    out = {}
    for kind in ("claim", "denial"):
        sub = rows[rows["kind"] == kind] if len(rows) else rows
        judged = sub[sub["detected"].notna()] if len(sub) else sub
        n = len(judged)
        found = int(judged["detected"].astype(bool).sum()) if n else 0
        out[kind] = {
            "n": n,
            "detected": found,
            "rate": found / n if n else float("nan"),
            "errors": len(sub) - n,
        }
    return out


def load_run(run_dir: Path) -> tuple[list[Persona], list[dict], list[Judgement]]:
    personas = [Persona(**p) for p in json.loads((run_dir / "personas.json").read_text())]
    conversations = _read_jsonl(run_dir / "conversations.jsonl")
    judgements = [Judgement(**j) for j in _read_jsonl(run_dir / "judgements.jsonl")]
    return personas, conversations, judgements


def count_table(run_dir: Path) -> pd.DataFrame:
    personas, conversations, judgements = load_run(run_dir)
    by_id = {p.persona_id: p for p in personas}
    judged = {(j.agent, j.persona_id): j for j in judgements}
    rows = []
    for row in conversations:
        key = (row["agent"], row["persona_id"])
        if key not in judged:
            raise ValueError(f"conversation {key} has not been judged")
        record = FollowUpRecord.model_validate(row["record"])
        rows.append(counts(record, judged[key], by_id[row["persona_id"]]))
    return pd.DataFrame(rows)


def compute_metrics(cfg: AgentEvalConfig, run_dir: Path) -> dict:
    table = count_table(run_dir)
    expected = len(cfg.agents) * len(json.loads((run_dir / "personas.json").read_text()))
    metrics = summarize(table, cfg.reference, n_boot=cfg.n_boot, seed=cfg.seed)
    metrics["complete"] = len(table) == expected
    metrics["judge_probes"] = probe_summary(run_dir)
    table.to_parquet(run_dir / "counts.parquet", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def record_index(cfg: AgentEvalConfig, run_dir: Path, results_dir: Path, metrics: dict) -> None:
    info = json.loads((run_dir / "run.json").read_text())
    reference = metrics["agents"].get(cfg.reference, {})
    row = {
        "name": cfg.name,
        "run": run_dir.name,
        "commit": (info["git"]["commit"] or "")[:8],
        "dirty": info["git"]["dirty"],
        "partial": info["partial"],
        "complete": metrics["complete"],
        "prompt_version": info["prompt_version"],
        "reference_recall": round(reference.get("cause_recall", {}).get("value", float("nan")), 3),
    }
    index = results_dir / "agent_index.csv"
    pd.DataFrame([row]).to_csv(index, mode="a", header=not index.exists(), index=False)


def ollama_backends(cfg: AgentEvalConfig) -> BackendFactory:
    cache: dict[str, ChatBackend] = {}

    def make(model: str) -> ChatBackend:
        if model not in cache:
            cache[model] = OllamaBackend(model, num_ctx=cfg.num_ctx)
        return cache[model]

    return make
