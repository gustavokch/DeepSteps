import numpy as np
from build_dataset import rows_to_dataset


def test_rows_to_dataset_stacks_and_shapes():
    rows = [np.zeros(32), np.ones(32)]
    ds = rows_to_dataset(rows)
    assert ds.shape == (2, 32)
    assert ds[1].sum() == 32
