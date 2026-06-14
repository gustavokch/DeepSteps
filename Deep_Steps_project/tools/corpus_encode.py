"""Pure, importable port of the single-bar onset->32dim encoding.

Extracted from ``Deep_Steps_project/bin/data/Process_corpus.py`` so the
onset -> vector math can be tested without audio libraries. The 32-dim
representation (16 onset one-hots + 16 substep timing offsets) must match the
original exactly: the autoencoder trains on it and the Rust decoder must
reproduce the same patterns.

``bar_length`` is fixed to 1 here (16-step model); the multi-bar split in the
original is out of scope.
"""

import numpy as np

PER_QUARTER_NOTE = 48
SIXTEENTHS_DIV = 16  # bar_length = 1


def encode_onsets(onsets, dur):
    """onsets: int sample positions within one bar. dur: bar length in samples.
    Returns a 32-dim vector [onset_onehot[16], substep_offset[16]]."""
    timebase = PER_QUARTER_NOTE * 4          # 192 PPQN per bar
    num_ppqn = timebase                       # bar_length = 1
    ppqn_timebase = round(dur / num_ppqn)
    sixteenths = round(dur / SIXTEENTHS_DIV)

    onsets = np.round(onsets).astype(int)

    # round onsets to nearest 16th, drop duplicates landing on the same step
    onset_points_rounded = []
    previous = None
    keep = np.ones(len(onsets), dtype=bool)
    for i, onset in enumerate(onsets):
        r = int(round(onset / sixteenths))
        if r != previous:
            onset_points_rounded.append(r)
            previous = r
        else:
            keep[i] = False
    onsets = onsets[keep]

    ppqn_onsets = [int(o // ppqn_timebase) * ppqn_timebase for o in onsets]

    onehot = np.zeros(SIXTEENTHS_DIV)
    for o in onset_points_rounded:
        if o < SIXTEENTHS_DIV:
            onehot[o] = 1

    # substep = signed PPQN distance from nearest 16th, clamped [-6,6], -> (x+6)/12
    substeps = []
    nearest = [int(round(o / sixteenths)) * sixteenths for o in onsets]
    for f, c in zip(ppqn_onsets, nearest):
        ss = (f - c) // ppqn_timebase
        substeps.append((ss + 6) / 12)

    substeps_full = []
    j = 0
    for v in onehot:
        if v == 1 and j < len(substeps):
            substeps_full.append(substeps[j])
            j += 1
        else:
            substeps_full.append(0)

    return np.concatenate((onehot, np.array(substeps_full)))
