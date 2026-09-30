import numpy as np
import pandas as pd

from sleepwatch.constants import N2, WAKE
from sleepwatch.data.label_timing import best_lags, label_agreement_by_lag, wake_separation_by_lag


def night_with_wake_bouts(label_delay: int, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Activity is high during true wake bouts; labels are recorded ``label_delay`` epochs late."""
    rng = np.random.default_rng(seed)
    truth = np.full(n, N2)
    for start in rng.choice(np.arange(10, n - 20), 12, replace=False):
        truth[start : start + rng.integers(2, 8)] = WAKE
    activity = np.where(truth == WAKE, 1.0, 0.0) + rng.normal(0, 0.1, n)
    labels = np.roll(truth, label_delay)  # label at k describes the truth at k - delay
    return pd.DataFrame(
        {"subject": "S", "night": 1, "expert": labels, "dreem": truth, "activity": activity}
    )


def test_aligned_labels_peak_at_lag_zero():
    sep = wake_separation_by_lag(night_with_wake_bouts(0), "expert", "activity")
    assert best_lags(sep, "separation").iloc[0] == 0


def test_late_labels_are_detected_as_a_negative_lag():
    sep = wake_separation_by_lag(night_with_wake_bouts(2), "expert", "activity")
    assert best_lags(sep, "separation").iloc[0] == -2


def test_agreement_lag_between_label_sequences():
    agree = label_agreement_by_lag(night_with_wake_bouts(2), "expert", "dreem")
    assert best_lags(agree, "agreement").iloc[0] == -2
    assert agree.set_index("lag").loc[-2, "agreement"] > 0.98
