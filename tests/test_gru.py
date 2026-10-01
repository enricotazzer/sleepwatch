import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sleepwatch.models.gru import IGNORE, GRUConfig, Net, predict, reverse_within

LENGTHS = torch.tensor([7, 3, 5])


def padded_batch(n_inputs=4, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(len(LENGTHS), int(LENGTHS.max()), n_inputs)
    for i, n in enumerate(LENGTHS):
        x[i, n:] = 99.0  # padding must never leak into real steps
    return x


def test_reverse_within_flips_real_steps_only():
    x = torch.arange(len(LENGTHS) * 7, dtype=torch.float).reshape(len(LENGTHS), 7, 1)
    out = reverse_within(x, LENGTHS)
    for i, n in enumerate(LENGTHS):
        assert torch.equal(out[i, :n], x[i, :n].flip(0))
        assert torch.equal(out[i, n:], x[i, n:])
    assert torch.equal(reverse_within(out, LENGTHS), x)


@pytest.mark.parametrize("layers", [1, 2])
def test_net_equals_packed_bidirectional_gru(layers):
    cfg = GRUConfig(hidden=6, layers=layers)
    net = Net(4, cfg).eval()
    ref = nn.GRU(6, 6, num_layers=layers, batch_first=True, bidirectional=True).eval()
    with torch.no_grad():
        for k in range(layers):
            for name, param in net.fwd[k].named_parameters():
                getattr(ref, name.replace("l0", f"l{k}")).copy_(param)
            for name, param in net.bwd[k].named_parameters():
                getattr(ref, name.replace("l0", f"l{k}") + "_reverse").copy_(param)
        x = padded_batch()
        packed = pack_padded_sequence(net.inp(x), LENGTHS, batch_first=True, enforce_sorted=False)
        hidden, _ = pad_packed_sequence(ref(packed)[0], batch_first=True)
        expected, got = net.out(hidden), net(x, LENGTHS)
    for i, n in enumerate(LENGTHS):
        torch.testing.assert_close(got[i, :n], expected[i, :n], atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_matches_cpu():
    torch.manual_seed(0)
    net = Net(4, GRUConfig(hidden=8)).eval()
    x = padded_batch()
    items = [((0, i), x[i, :n].numpy(), np.full(int(n), IGNORE)) for i, n in enumerate(LENGTHS)]
    cpu = predict(net, items)
    mps = predict(net.to("mps"), items)
    for a, b in zip(cpu, mps, strict=True):
        assert a.shape == b.shape
        assert abs(a - b).max() < 1e-5
