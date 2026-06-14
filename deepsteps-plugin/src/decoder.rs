use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "op")]
enum Op {
    #[serde(rename = "dense")]
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

#[derive(Deserialize)]
pub struct Decoder {
    pub latent_dim: usize,
    pub input_dim: usize,
    ops: Vec<Op>,
}

impl Decoder {
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn forward(&self, latent: &[f64]) -> Vec<f64> {
        let mut x = latent.to_vec();
        for op in &self.ops {
            x = match op {
                Op::Dense { W, b } => {
                    let n_out = W[0].len();
                    let mut y = vec![0.0; n_out];
                    for j in 0..n_out {
                        let mut acc = b[0][j];
                        for i in 0..x.len() {
                            acc += x[i] * W[i][j];
                        }
                        y[j] = acc;
                    }
                    y
                }
                Op::Relu => x.iter().map(|&v| if v >= 0.0 { v } else { 0.0 }).collect(),
                Op::Sigmoid => x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
                Op::Bn {
                    gamma,
                    beta,
                    running_mean,
                    running_var,
                    eps,
                } => (0..x.len())
                    .map(|i| {
                        gamma[i] * ((x[i] - running_mean[i]) / (running_var[i] + eps).sqrt())
                            + beta[i]
                    })
                    .collect(),
            };
        }
        x
    }

    /// steps[16] thresholded >0.5, substeps[16] raw — mirrors receive_latent().
    pub fn generate(&self, latent: &[f64]) -> ([bool; 16], [f64; 16]) {
        let out = self.forward(latent);
        let mut steps = [false; 16];
        let mut substeps = [0.0; 16];
        for i in 0..16 {
            steps[i] = out[i] > 0.5;
            substeps[i] = out[16 + i];
        }
        (steps, substeps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_matches_reference_vectors() {
        let dec = Decoder::from_json_str(include_str!("../weights/decoder.json")).unwrap();
        let refs: Vec<RefVec> =
            serde_json::from_str(include_str!("../weights/reference_vectors.json")).unwrap();
        let mut worst = 0.0f64;
        for r in &refs {
            let out = dec.forward(&r.latent);
            assert_eq!(out.len(), 32);
            for (a, b) in out.iter().zip(r.output.iter()) {
                let err = (a - b).abs();
                worst = worst.max(err);
                assert!(err < 1e-5, "mismatch {a} vs {b}");
            }
        }
        println!("worst absolute error across reference vectors: {worst:e}");
    }
    #[derive(serde::Deserialize)]
    struct RefVec {
        latent: Vec<f64>,
        output: Vec<f64>,
    }
}
