"""Phase 2b models: the context-conditioned GRU and per-person fine-tuning."""

import numpy as np
import pytest
import torch

from sleepwatch.models.context import ContextConfig, ContextNet, collate, donors, with_context
from sleepwatch.models.data import ALL_PRIOR
from sleepwatch.models.experiment import _finetune_jobs
from sleepwatch.models.finetune import FinetuneConfig, FinetuneSetting, finetune
from sleepwatch.models.gru import IGNORE, GRUConfig, Net, predict

GRU = GRUConfig(hidden=8)


def night(n, seed, labels=(0, 2, 3, 4)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 4)).astype(np.float32), rng.choice(labels, n)


def nights_table(subjects=("A", "B", "C"), count=4):
    nights, ranks = {}, {}
    for s_i, subject in enumerate(subjects):
        for rank in range(1, count + 1):
            key = (subject, 10 + rank)  # night numbers differ from ranks on purpose
            nights[key] = night(5 + rank, 10 * s_i + rank)[0]
            ranks[key] = rank
    return nights, ranks


def test_donors_never_pair_a_subject_with_itself():
    pairing = donors(["A", "B", "C", "D"], seed=3)
    assert sorted(pairing) == sorted(pairing.values()) == ["A", "B", "C", "D"]
    assert all(s != d for s, d in pairing.items())
    assert pairing == donors(["D", "C", "B", "A"], seed=3)


def test_context_is_the_strictly_earlier_nights():
    nights, ranks = nights_table()
    items = [(k, nights[k], np.zeros(len(nights[k]), int)) for k in nights]
    own = {item[0]: item[3] for item in with_context(items, nights, ranks)}
    assert own[("A", 11)] == []
    assert len(own[("A", 13)]) == 2
    assert all(c is nights[("A", 11 + i)] for i, c in enumerate(own[("A", 13)]))
    pairing = {"A": "B", "B": "C", "C": "A"}
    swapped = {item[0]: item[3] for item in with_context(items, nights, ranks, pairing)}
    assert all(c is nights[("B", 11 + i)] for i, c in enumerate(swapped[("A", 14)]))
    assert len(swapped[("A", 14)]) == 3 and swapped[("A", 11)] == []


def test_each_night_only_sees_its_own_context():
    torch.manual_seed(0)
    net = ContextNet(4, GRU, ContextConfig(encoder=6, dim=3)).eval()
    nights, ranks = nights_table()
    items = [(k, nights[k], np.full(len(nights[k]), IGNORE)) for k in nights]
    items = with_context(items, nights, ranks)
    together = predict(net, items, collate_fn=collate)
    alone = [predict(net, [item], collate_fn=collate)[0] for item in items]
    for a, b in zip(together, alone, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)
    # The summary is an average, so the order of the earlier nights doesn't matter.
    key, x, y, context = items[-1]
    reordered = predict(net, [(key, x, y, context[::-1])], collate_fn=collate)[0]
    np.testing.assert_allclose(reordered, together[-1], atol=1e-6)


def test_context_changes_the_prediction():
    torch.manual_seed(0)
    net = ContextNet(4, GRU, ContextConfig(encoder=6, dim=3)).eval()
    x, y = night(9, 1)
    without = predict(net, [(("A", 1), x, y, [])], collate_fn=collate)[0]
    with_ = predict(net, [(("A", 1), x, y, [night(7, 2)[0]])], collate_fn=collate)[0]
    assert not np.allclose(without, with_)


@pytest.mark.parametrize("layers", ["output", "all"])
def test_finetune_leaves_the_population_model_unchanged(layers):
    torch.manual_seed(0)
    model = Net(4, GRU).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x, y = night(12, 3, labels=(2, 4))  # only two stages present
    setting = FinetuneSetting(lr=1e-2, epochs=5, layers=layers)
    tuned = finetune(model, [(("A", 1), x, y)], setting, FinetuneConfig(), GRU, seed=0)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])
    changed = {k for k, v in tuned.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and (layers == "all" or all(k.startswith("out.") for k in changed))
    proba = predict(tuned, [(("A", 2), *night(6, 4))])[0]
    assert proba.shape == (6, 5) and np.allclose(proba.sum(axis=1), 1)  # still 5 stages
    again = finetune(model, [(("A", 1), x, y)], setting, FinetuneConfig(), GRU, seed=0)
    for k, v in tuned.state_dict().items():
        assert torch.equal(v, again.state_dict()[k])  # deterministic


def test_finetune_jobs_train_only_on_nights_before_the_scored_ones():
    order = {s: [(s, n) for n in range(1, 7)] for s in ("A", "B")}
    order["C"] = [("C", 1), ("C", 2)]
    eval_keys = {(s, n) for s in ("A", "B") for n in range(4, 7)}
    jobs = _finetune_jobs(order, ["A"], eval_keys, [1, 3])
    assert (1, [("A", 1)], [("A", 4), ("A", 5), ("A", 6)]) in jobs
    assert (3, order["A"][:3], [("A", 4), ("A", 5), ("A", 6)]) in jobs
    assert (ALL_PRIOR, order["A"][:5], [("A", 6)]) in jobs
    for _, train, targets in jobs:
        assert max(n for _, n in train) < min(n for _, n in targets)
    other = _finetune_jobs(order, ["A"], eval_keys, [3], donor={"A": "C"})
    assert (3, order["C"], [("A", 4), ("A", 5), ("A", 6)]) in other  # the donor has only 2
    assert all(s == "C" for _, train, _ in other for s, _ in train)
