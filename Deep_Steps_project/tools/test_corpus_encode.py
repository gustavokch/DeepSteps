import numpy as np
from corpus_encode import encode_onsets


def test_single_bar_onsets_to_32dim():
    # onsets in samples; dur = total length in samples. One bar.
    # Place onsets exactly on 16th-note boundaries -> substep offset 0.5 (centered).
    dur = 48 * 4 * 1  # per_quarter_note(48) * 4 * bar_length(1) PPQN units as "samples"
    # 4 onsets on steps 0,4,8,12 exactly on the grid
    sixteenth = dur / 16
    onsets = np.array([0, 4, 8, 12]) * sixteenth
    vec = encode_onsets(onsets.astype(int), int(dur))
    assert vec.shape == (32,)
    onehot, substeps = vec[:16], vec[16:]
    assert list(onehot[[0, 4, 8, 12]]) == [1, 1, 1, 1]
    assert onehot.sum() == 4
    # on-grid onset -> substep distance 0 -> normalized (0+6)/12 = 0.5
    for i in (0, 4, 8, 12):
        assert abs(substeps[i] - 0.5) < 1e-6
    # empty steps carry substep 0
    assert substeps[1] == 0
