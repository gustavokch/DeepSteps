# make_synth_dataset.py  — placeholder corpus: random 16-step patterns + grid substeps
import sys, numpy as np
rng = np.random.default_rng(42)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
onehot = (rng.random((n, 16)) > 0.6).astype(float)
substeps = np.where(onehot == 1, rng.uniform(0.3, 0.7, (n, 16)), 0.0)
ds = np.concatenate((onehot, substeps), axis=1)
np.savetxt(sys.argv[1], ds, delimiter=",")
print(f"wrote {ds.shape} -> {sys.argv[1]}")
