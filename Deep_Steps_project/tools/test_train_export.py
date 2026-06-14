import numpy as np
from train_export import build_autoencoder, export_decoder, forward_from_export


def test_export_roundtrip_matches_python_decoder():
    rng = np.random.default_rng(0)
    ds = (rng.random((64, 32)) > 0.5).astype(float)  # dummy bars
    ae = build_autoencoder()
    ae.autoencoder.fit(ds, ds, n_epochs=3, batch_size=16)
    export = export_decoder(ae)
    # three random latents must match the live Python decoder within eps
    for _ in range(3):
        z = rng.random((1, 4))
        py = ae.decoder.predict(z)[0]
        js = forward_from_export(export, z[0])
        assert np.allclose(py, js, atol=1e-6)


def test_export_op_order():
    ae = build_autoencoder()
    ds = (np.random.default_rng(0).random((32, 32)) > 0.5).astype(float)
    ae.autoencoder.fit(ds, ds, n_epochs=2, batch_size=16)
    export = export_decoder(ae)
    ops = [op["op"] for op in export["ops"]]
    assert ops == ["dense", "relu", "bn", "dense", "relu", "bn", "dense", "sigmoid"]
    assert export["latent_dim"] == 4
    assert export["input_dim"] == 32


def test_export_rejects_none_running_stats():
    import pytest
    ae = build_autoencoder()  # untrained: BN running stats are None
    with pytest.raises(Exception):
        export_decoder(ae)
