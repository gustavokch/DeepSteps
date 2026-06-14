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


def test_off_grid_substep_not_half():
    # dur=192 -> ppqn_timebase=1, sixteenths=12. Onset 5 samples past step 4 (=48).
    # r = round(53/12) = 4 (in range). ss = (53 - 48)//1 = 5 -> (5+6)/12 = 11/12.
    dur = 192
    onset = 4 * 12 + 5  # 53
    vec = encode_onsets(np.array([onset]), dur)
    onehot, substeps = vec[:16], vec[16:]
    assert onehot[4] == 1
    assert onehot.sum() == 1
    expected = (5 + 6) / 12  # 0.9166666...
    assert abs(expected - 0.5) > 1e-6  # genuinely off-grid, not the centered 0.5
    assert abs(substeps[4] - expected) < 1e-9


def test_duplicate_step_dropped():
    # dur=192, sixteenths=12. Two onsets that both round to step 4:
    # round(48/12)=4 and round(50/12)=round(4.1667)=4 -> second is a duplicate.
    dur = 192
    onsets = np.array([4 * 12, 4 * 12 + 2])  # [48, 50]
    vec = encode_onsets(onsets, dur)
    onehot = vec[:16]
    assert onehot[4] == 1
    # the duplicate is dropped: step 4 counted exactly once
    assert onehot.sum() == 1


def test_out_of_range_step_excluded():
    # dur=192, sixteenths=12. Step 0 (=0) plus an onset rounding to step 16 (=192).
    # round(192/12)=16, which is NOT < 16, so it is excluded from the 16-len onehot.
    dur = 192
    onsets = np.array([0, 16 * 12])  # [0, 192]
    vec = encode_onsets(onsets, dur)
    onehot = vec[:16]
    assert onehot[0] == 1
    # only the in-range onset is set; the step-16 onset does not affect the onehot
    assert onehot.sum() == 1
