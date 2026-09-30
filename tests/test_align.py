import numpy as np
import pandas as pd

from sleepwatch.data.align import crop, epoch_grid, epoch_index, resample

R = 1_000_000.0


def test_epoch_boundaries():
    t = R + np.array([-0.001, 0.0, 29.999, 30.0, 59.999, 60.0])
    assert epoch_index(t, R).tolist() == [-1, 0, 0, 1, 1, 2]


def test_matches_the_dataset_formula():
    t = R + np.random.default_rng(0).uniform(0, 3600, 1000)
    one_based = np.floor((t - R) / 30) + 1  # formula from the dataset README
    assert np.array_equal(epoch_index(t, R) + 1, one_based)


def test_crop_is_half_open():
    df = pd.DataFrame({"t": R + np.array([-1.0, 0.0, 59.0, 60.0])})
    assert crop(df, R, R + 60)["t"].tolist() == [R, R + 59.0]


def test_grid_reshapes_to_one_row_per_epoch():
    grid = epoch_grid(R, n_epochs=3, hz=2)
    assert grid.shape == (180,)
    assert np.array_equal(epoch_index(grid, R).reshape(3, -1), np.repeat([[0], [1], [2]], 60, 1))
    assert grid[0] == R + 0.25


def test_resample_interpolates_but_never_bridges_gaps():
    t = np.array([0.0, 2.0, 4.0, 40.0, 42.0])
    v = np.array([0.0, 2.0, 4.0, 40.0, 42.0])
    grid = np.array([-1.0, 1.0, 3.0, 20.0, 41.0, 43.0])
    out = resample(t, v, grid, max_gap=5.0)
    assert np.allclose(out[[1, 2, 4]], [1.0, 3.0, 41.0])
    assert np.isnan(out[[0, 3, 5]]).all()  # before data, inside the 36 s gap, after data
