//! Serializable op-list model export, byte-compatible with `weights/decoder.json`
//! and the Python `train_export.py` output. A model trained in-app serializes to
//! exactly the format `decoder::Decoder::from_json_str` already consumes, so the
//! audio-thread inference path needs no changes to run an in-app-trained model.
//!
//! The encoder is exported in the same op-list shape; at runtime it is loaded as
//! a `Decoder` too (the forward pass is a generic layer chain — only `generate()`
//! assumes a 32-dim output, which the encode path does not call).

use serde::{Deserialize, Serialize};

use crate::autoencoder::{ActKind, Autoencoder, LayerView};
use crate::decoder::Decoder;

/// One layer in the exported op list. Field names/tags match `decoder.rs`'s
/// private `Op` and `train_export.py` exactly (note `b` is nested 1xN).
#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "op")]
pub enum ExportOp {
    #[serde(rename = "dense")]
    #[allow(non_snake_case)]
    Dense { W: Vec<Vec<f64>>, b: Vec<Vec<f64>> },
    #[serde(rename = "relu")]
    Relu,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "bn")]
    Bn {
        gamma: Vec<f64>,
        beta: Vec<f64>,
        running_mean: Vec<f64>,
        running_var: Vec<f64>,
        eps: f64,
    },
}

/// A full exported model (`{latent_dim, input_dim, ops}`), the persistence and
/// hot-swap unit.
#[derive(Serialize, Deserialize, Clone)]
pub struct ModelExport {
    pub latent_dim: usize,
    pub input_dim: usize,
    pub ops: Vec<ExportOp>,
}

/// Both halves of a trained autoencoder, serialized into DAW state (`#[persist]`).
#[derive(Serialize, Deserialize, Clone)]
pub struct TrainedModel {
    pub decoder: ModelExport,
    pub encoder: ModelExport,
}

#[derive(Debug)]
pub enum ExportError {
    /// A BatchNorm layer was exported before any forward pass populated its
    /// running stats (mirrors the Python guard in `train_export.py`).
    UntrainedBatchNorm,
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::UntrainedBatchNorm => write!(
                f,
                "BatchNorm running stats are None -- train (fit) before exporting"
            ),
        }
    }
}

fn export_range(
    ae: &Autoencoder,
    range: std::ops::Range<usize>,
) -> Result<ModelExport, ExportError> {
    let mut ops = Vec::new();
    for i in range {
        match ae.layer_export(i) {
            LayerView::Dense { n_in, n_out, w, b } => {
                // Reshape row-major w[i*n_out+j] -> nested [n_in][n_out]; bias 1xN.
                let mut rows = Vec::with_capacity(n_in);
                for r in 0..n_in {
                    rows.push(w[r * n_out..(r + 1) * n_out].to_vec());
                }
                ops.push(ExportOp::Dense { W: rows, b: vec![b.to_vec()] });
            }
            LayerView::Activation(ActKind::Relu) => ops.push(ExportOp::Relu),
            LayerView::Activation(ActKind::Sigmoid) => ops.push(ExportOp::Sigmoid),
            LayerView::BatchNorm { gamma, beta, eps, running_mean, running_var } => {
                let (rm, rv) = match (running_mean, running_var) {
                    (Some(m), Some(v)) => (m, v),
                    _ => return Err(ExportError::UntrainedBatchNorm),
                };
                ops.push(ExportOp::Bn {
                    gamma: gamma.to_vec(),
                    beta: beta.to_vec(),
                    running_mean: rm.to_vec(),
                    running_var: rv.to_vec(),
                    eps,
                });
            }
        }
    }
    Ok(ModelExport { latent_dim: ae.latent_dim, input_dim: ae.input_dim, ops })
}

/// Export the decoder half (latent -> 32-dim), matching `train_export.py`.
pub fn export_decoder(ae: &Autoencoder) -> Result<ModelExport, ExportError> {
    export_range(ae, ae.decoder_range())
}

/// Export the encoder half (32-dim -> latent).
pub fn export_encoder(ae: &Autoencoder) -> Result<ModelExport, ExportError> {
    export_range(ae, ae.encoder_range())
}

/// Build a runnable `Decoder` from an export by round-tripping through the same
/// JSON the baked weights use, so there is a single forward-pass implementation.
pub fn to_decoder(export: &ModelExport) -> Result<Decoder, serde_json::Error> {
    let s = serde_json::to_string(export)?;
    Decoder::from_json_str(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Train tiny, export the decoder, rebuild a `Decoder`, and confirm its
    /// forward matches the in-memory autoencoder's own decoder forward. This
    /// ties the training net (autoencoder.rs) to the inference path (decoder.rs)
    /// through the persisted JSON format.
    #[test]
    fn export_decoder_roundtrips_through_decoder() {
        let data = [[0.0f32; 32], [1.0f32; 32], {
            let mut v = [0.0f32; 32];
            v[2] = 1.0;
            v[16] = 0.5;
            v
        }];
        let mut ae = Autoencoder::new(7);
        ae.fit(&data, 50, 3, 1, |_, _| true); // populate BN running stats

        let export = export_decoder(&ae).expect("decoder should export after fit");
        let dec = to_decoder(&export).expect("export must parse as Decoder");

        for seed in 0..5u64 {
            // Deterministic test latents.
            let z = [
                ((seed * 7 + 1) % 11) as f64 / 11.0,
                ((seed * 13 + 3) % 11) as f64 / 11.0,
                ((seed * 5 + 2) % 11) as f64 / 11.0,
                ((seed * 3 + 9) % 11) as f64 / 11.0,
            ];
            let a = ae.decode(&z);
            let b = dec.forward(&z);
            assert_eq!(b.len(), 32);
            for i in 0..32 {
                assert!((a[i] - b[i]).abs() < 1e-9, "op {i}: {} vs {}", a[i], b[i]);
            }
        }
    }

    /// Encoder export is also a valid runnable op chain (32-dim in -> 4-dim out).
    #[test]
    fn export_encoder_matches_encode() {
        let data = [[0.0f32; 32], [1.0f32; 32]];
        let mut ae = Autoencoder::new(3);
        ae.fit(&data, 30, 2, 1, |_, _| true);

        let export = export_encoder(&ae).unwrap();
        let enc = to_decoder(&export).unwrap();

        let mut x = [0.0f64; 32];
        x[0] = 1.0;
        x[16] = 0.5;
        let z_ref = ae.encode(&x);
        let z_run = enc.forward(&x);
        assert_eq!(z_run.len(), 4);
        for i in 0..4 {
            assert!((z_ref[i] - z_run[i]).abs() < 1e-9, "{} vs {}", z_ref[i], z_run[i]);
        }
    }

    /// Exporting before any forward pass fails loudly (untrained BN).
    #[test]
    fn export_untrained_bn_errors() {
        let ae = Autoencoder::new(1);
        assert!(matches!(
            export_decoder(&ae),
            Err(ExportError::UntrainedBatchNorm)
        ));
    }
}
