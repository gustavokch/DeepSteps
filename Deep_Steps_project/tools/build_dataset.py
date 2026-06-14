"""Audio-corpus -> dataset.csv builder (thin librosa wrapper).

For each audio file: librosa onset detection -> ``encode_onsets`` -> one
32-dim row. Rows are stacked into an ``(n_files, 32)`` matrix and saved as CSV.

This is the OPTIONAL real-audio path. The committed autoencoder weights use a
synthetic dataset, so this module is not required to run against a real corpus.
Only ``rows_to_dataset`` (the row-assembly) is unit-tested; the librosa glue is
thin I/O whose math is already covered by ``test_corpus_encode.py``.

Usage:
    uv run python build_dataset.py "corpus/*.wav" dataset.csv
"""

import sys
import glob
import numpy as np
import librosa

from corpus_encode import encode_onsets


def rows_to_dataset(rows):
    """Stack a list of 32-dim row vectors into an (n_files, 32) matrix."""
    return np.vstack(rows)


def onsets_for_file(path):
    """Return (onset sample positions, total length in samples) for one audio file."""
    y, sr = librosa.load(path, sr=None, mono=True)  # native samplerate
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
    return np.asarray(onsets), len(y)


def build(corpus_glob):
    """Encode every file matching ``corpus_glob`` into a stacked dataset matrix."""
    rows = []
    for path in sorted(glob.glob(corpus_glob)):
        onsets, dur = onsets_for_file(path)
        if len(onsets) == 0:
            continue
        rows.append(encode_onsets(onsets, dur))
    return rows_to_dataset(rows)


if __name__ == "__main__":
    ds = build(sys.argv[1])  # e.g. "corpus/*.wav"
    np.savetxt(sys.argv[2], ds, delimiter=",")
    print(f"wrote {ds.shape} -> {sys.argv[2]}")
