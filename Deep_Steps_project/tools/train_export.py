import os
import sys
import json
import numpy as np

# AE_init.py lives in ../bin/data relative to this file. Importing it runs the
# whole module top-to-bottom, including `os.chdir("./data")` (which would raise
# FileNotFoundError when imported from tools/) and a module-level Autoencoder()
# construction + OSC client setup. Neutralize the chdir for the duration of the
# import so it is cwd-independent.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "../bin/data"))

_real_chdir = os.chdir
os.chdir = lambda *a, **k: None
try:
    import AE_init
    from AE_init import Autoencoder
finally:
    os.chdir = _real_chdir


class _NullClient:
    """Swallows the OSC /loss messages NeuralNetwork.fit emits each epoch."""

    def send_message(self, *a, **k):
        pass


def build_autoencoder():
    AE_init.client = _NullClient()  # silence the OSC /loss calls in fit()
    return Autoencoder()


def _relu(x):
    return np.where(x >= 0, x, 0)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def export_decoder(ae):
    """Walk decoder.layers, emit ordered op list mirroring forward_pass(training=False)."""
    ops = []
    for layer in ae.decoder.layers:
        name = type(layer).__name__
        if name == "Dense":
            ops.append({"op": "dense",
                        "W": layer.W.tolist(),
                        "b": layer.w0.tolist()})       # w0 shape (1, n)
        elif name == "Activation":
            ops.append({"op": layer.activation_name})  # 'relu' | 'sigmoid'
        elif name == "BatchNormalization":
            # Running stats are populated on the first forward pass. If they are
            # still None the net was never run, so the export would be broken --
            # fail loudly rather than emit null running stats.
            if layer.running_mean is None or layer.running_var is None:
                raise ValueError(
                    "BatchNormalization running stats are None -- the decoder "
                    "was never run forward. Train (fit) before exporting.")
            ops.append({"op": "bn",
                        "gamma": layer.gamma.tolist(),
                        "beta": layer.beta.tolist(),
                        "running_mean": layer.running_mean.tolist(),
                        "running_var": layer.running_var.tolist(),
                        "eps": layer.eps})
        else:
            raise ValueError(f"unexpected decoder layer {name}")
    return {"latent_dim": ae.latent_dim, "input_dim": ae.input_dim, "ops": ops}


def forward_from_export(export, z):
    x = np.asarray(z, dtype=float).reshape(1, -1)
    for op in export["ops"]:
        k = op["op"]
        if k == "dense":
            x = x.dot(np.array(op["W"])) + np.array(op["b"])
        elif k == "relu":
            x = _relu(x)
        elif k == "sigmoid":
            x = _sigmoid(x)
        elif k == "bn":
            mean = np.array(op["running_mean"])
            var = np.array(op["running_var"])
            g = np.array(op["gamma"])
            b = np.array(op["beta"])
            eps = op["eps"]
            x = g * ((x - mean) / np.sqrt(var + eps)) + b
        else:
            raise ValueError(k)
    return x[0]


def reference_vectors(ae, n=8, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.random(4)
        y = ae.decoder.predict(z.reshape(1, -1))[0]
        out.append({"latent": z.tolist(), "output": y.tolist()})
    return out


if __name__ == "__main__":
    dataset_csv, weights_out, refs_out = sys.argv[1], sys.argv[2], sys.argv[3]
    ds = np.loadtxt(dataset_csv, delimiter=",")
    ae = build_autoencoder()
    ae.autoencoder.fit(ds, ds, n_epochs=200, batch_size=16)
    json.dump(export_decoder(ae), open(weights_out, "w"))
    json.dump(reference_vectors(ae), open(refs_out, "w"))
    print(f"exported {weights_out} + {refs_out}")
