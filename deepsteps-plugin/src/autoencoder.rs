//! From-scratch autoencoder training, ported from the upstream Python
//! `Deep_Steps_project/bin/data/AE_init.py` (itself adapted from
//! ML-From-Scratch). Pure f64, no external ML deps, no nih-plug deps so it is
//! unit-testable in isolation.
//!
//! Architecture (input 32, latent 4):
//!   encoder: Dense(32->16) Relu BN  Dense(16->8) Relu BN  Dense(8->4)
//!   decoder: Dense(4->8)  Relu BN  Dense(8->16) Relu BN  Dense(16->32) Sigmoid
//!
//! Two deliberate fixes over the Python original (see project plan, decision 2):
//!   * Adam bias correction uses a real per-parameter step counter `t`
//!     (`1 - b1^t`), where the Python divides by the constant `1 - b1`.
//!   * Batches are shuffled each epoch (the Python iterates sequentially).
//!
//! The exported op list (see `model_ops.rs`) feeds `decoder.rs` unchanged.

/// Minimal row-major dense matrix used for whole-batch forward/backward.
/// Dimensions are tiny (<=32) so naive loops are fine.
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }
    #[inline]
    fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }
    fn col_mean(&self) -> Vec<f64> {
        let mut m = vec![0.0; self.cols];
        for r in 0..self.rows {
            for (c, mc) in m.iter_mut().enumerate() {
                *mc += self.at(r, c);
            }
        }
        for v in &mut m {
            *v /= self.rows as f64;
        }
        m
    }
    /// Population variance (ddof=0), matching numpy `np.var`.
    fn col_var(&self, mean: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0; self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let d = self.at(r, c) - mean[c];
                v[c] += d * d;
            }
        }
        for x in &mut v {
            *x /= self.rows as f64;
        }
        v
    }
}

// ---------------------------------------------------------------------------
// PRNG: SplitMix64 -> deterministic init + shuffle, no `rand` dependency.
// ---------------------------------------------------------------------------
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point.
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f64 in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // Top 53 bits -> [0,1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform f64 in [-limit, limit).
    #[inline]
    fn uniform(&mut self, limit: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * limit
    }
    /// Uniform integer in [0, n).
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Adam (fixed bias correction).
// ---------------------------------------------------------------------------
#[derive(Clone)]
struct Adam {
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: u32,
}

impl Adam {
    fn new(lr: f64, b1: f64, b2: f64, n: usize) -> Self {
        Adam { lr, b1, b2, eps: 1e-8, m: vec![0.0; n], v: vec![0.0; n], t: 0 }
    }
    /// In-place `w -= lr * m_hat / (sqrt(v_hat) + eps)` with real bias correction.
    fn step(&mut self, w: &mut [f64], grad: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.b1.powi(self.t as i32);
        let bc2 = 1.0 - self.b2.powi(self.t as i32);
        for i in 0..w.len() {
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * grad[i];
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            w[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Layers.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq)]
pub enum ActKind {
    Relu,
    Sigmoid,
}

enum Layer {
    Dense {
        n_in: usize,
        n_out: usize,
        w: Vec<f64>, // row-major (n_in x n_out): w[i*n_out + j]
        b: Vec<f64>, // (n_out)
        w_opt: Adam,
        b_opt: Adam,
        x_cache: Mat,
    },
    Activation {
        kind: ActKind,
        in_cache: Mat,
    },
    BatchNorm {
        gamma: Vec<f64>,
        beta: Vec<f64>,
        eps: f64,
        momentum: f64,
        running_mean: Option<Vec<f64>>,
        running_var: Option<Vec<f64>>,
        g_opt: Adam,
        b_opt: Adam,
        x_centered: Mat,
        stddev_inv: Vec<f64>,
    },
}

impl Layer {
    fn dense(n_in: usize, n_out: usize, lr: f64, b1: f64, b2: f64, rng: &mut Rng) -> Layer {
        // Xavier uniform per fan-in: U(-1/sqrt(n_in), +1/sqrt(n_in)); bias zeros.
        let limit = 1.0 / (n_in as f64).sqrt();
        let mut w = vec![0.0; n_in * n_out];
        for x in &mut w {
            *x = rng.uniform(limit);
        }
        Layer::Dense {
            n_in,
            n_out,
            w,
            b: vec![0.0; n_out],
            w_opt: Adam::new(lr, b1, b2, n_in * n_out),
            b_opt: Adam::new(lr, b1, b2, n_out),
            x_cache: Mat::zeros(0, 0),
        }
    }

    fn bn(dim: usize, momentum: f64, lr: f64, b1: f64, b2: f64) -> Layer {
        Layer::BatchNorm {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps: 0.01,
            momentum,
            running_mean: None,
            running_var: None,
            g_opt: Adam::new(lr, b1, b2, dim),
            b_opt: Adam::new(lr, b1, b2, dim),
            x_centered: Mat::zeros(0, 0),
            stddev_inv: vec![0.0; dim],
        }
    }

    fn forward(&mut self, x: &Mat, training: bool) -> Mat {
        match self {
            Layer::Dense { n_in, n_out, w, b, x_cache, .. } => {
                debug_assert_eq!(x.cols, *n_in);
                *x_cache = x.clone();
                let mut y = Mat::zeros(x.rows, *n_out);
                for r in 0..x.rows {
                    for j in 0..*n_out {
                        let mut acc = b[j];
                        for i in 0..*n_in {
                            acc += x.at(r, i) * w[i * *n_out + j];
                        }
                        y.set(r, j, acc);
                    }
                }
                y
            }
            Layer::Activation { kind, in_cache } => {
                *in_cache = x.clone();
                let mut y = Mat::zeros(x.rows, x.cols);
                for idx in 0..x.data.len() {
                    let v = x.data[idx];
                    y.data[idx] = match kind {
                        ActKind::Relu => {
                            if v >= 0.0 {
                                v
                            } else {
                                0.0
                            }
                        }
                        ActKind::Sigmoid => 1.0 / (1.0 + (-v).exp()),
                    };
                }
                y
            }
            Layer::BatchNorm {
                gamma,
                beta,
                eps,
                momentum,
                running_mean,
                running_var,
                x_centered,
                stddev_inv,
                ..
            } => {
                // Running stats initialize on the first forward (matches Python).
                if running_mean.is_none() {
                    *running_mean = Some(x.col_mean());
                    let m = running_mean.as_ref().unwrap();
                    *running_var = Some(x.col_var(m));
                }
                let (mean, var) = if training {
                    let bmean = x.col_mean();
                    let bvar = x.col_var(&bmean);
                    // running = momentum*running + (1-momentum)*batch
                    let rm = running_mean.as_mut().unwrap();
                    let rv = running_var.as_mut().unwrap();
                    for c in 0..rm.len() {
                        rm[c] = *momentum * rm[c] + (1.0 - *momentum) * bmean[c];
                        rv[c] = *momentum * rv[c] + (1.0 - *momentum) * bvar[c];
                    }
                    (bmean, bvar)
                } else {
                    (running_mean.clone().unwrap(), running_var.clone().unwrap())
                };

                *x_centered = Mat::zeros(x.rows, x.cols);
                for c in 0..x.cols {
                    stddev_inv[c] = 1.0 / (var[c] + *eps).sqrt();
                }
                let mut out = Mat::zeros(x.rows, x.cols);
                for r in 0..x.rows {
                    for c in 0..x.cols {
                        let xc = x.at(r, c) - mean[c];
                        x_centered.set(r, c, xc);
                        let x_norm = xc * stddev_inv[c];
                        out.set(r, c, gamma[c] * x_norm + beta[c]);
                    }
                }
                out
            }
        }
    }

    fn backward(&mut self, grad: &Mat) -> Mat {
        match self {
            Layer::Dense { n_in, n_out, w, b, w_opt, b_opt, x_cache } => {
                let rows = grad.rows;
                // grad_w[i,j] = sum_r x[r,i]*grad[r,j]; grad_b[j] = sum_r grad[r,j]
                let mut grad_w = vec![0.0; *n_in * *n_out];
                let mut grad_b = vec![0.0; *n_out];
                for r in 0..rows {
                    for j in 0..*n_out {
                        let g = grad.at(r, j);
                        grad_b[j] += g;
                        for i in 0..*n_in {
                            grad_w[i * *n_out + j] += x_cache.at(r, i) * g;
                        }
                    }
                }
                // accum_out = grad @ W^T, using the pre-update W.
                let mut out = Mat::zeros(rows, *n_in);
                for r in 0..rows {
                    for i in 0..*n_in {
                        let mut acc = 0.0;
                        for j in 0..*n_out {
                            acc += grad.at(r, j) * w[i * *n_out + j];
                        }
                        out.set(r, i, acc);
                    }
                }
                w_opt.step(w, &grad_w);
                b_opt.step(b, &grad_b);
                out
            }
            Layer::Activation { kind, in_cache } => {
                let mut out = Mat::zeros(grad.rows, grad.cols);
                for idx in 0..grad.data.len() {
                    let x = in_cache.data[idx];
                    let d = match kind {
                        ActKind::Relu => {
                            if x >= 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        ActKind::Sigmoid => {
                            let s = 1.0 / (1.0 + (-x).exp());
                            s * (1.0 - s)
                        }
                    };
                    out.data[idx] = grad.data[idx] * d;
                }
                out
            }
            Layer::BatchNorm {
                gamma, beta, g_opt, b_opt, x_centered, stddev_inv, ..
            } => {
                let bs = grad.rows as f64;
                let cols = grad.cols;
                let gamma_old = gamma.clone();

                // grad_gamma[c] = sum_r grad*x_norm ; grad_beta[c] = sum_r grad
                let mut grad_gamma = vec![0.0; cols];
                let mut grad_beta = vec![0.0; cols];
                // per-column sums needed by the batchnorm backward formula
                let mut sum_grad = vec![0.0; cols];
                let mut sum_grad_xc = vec![0.0; cols];
                for r in 0..grad.rows {
                    for c in 0..cols {
                        let g = grad.at(r, c);
                        let xc = x_centered.at(r, c);
                        let x_norm = xc * stddev_inv[c];
                        grad_gamma[c] += g * x_norm;
                        grad_beta[c] += g;
                        sum_grad[c] += g;
                        sum_grad_xc[c] += g * xc;
                    }
                }
                g_opt.step(gamma, &grad_gamma);
                b_opt.step(beta, &grad_beta);

                let mut out = Mat::zeros(grad.rows, cols);
                for r in 0..grad.rows {
                    for c in 0..cols {
                        let g = grad.at(r, c);
                        let xc = x_centered.at(r, c);
                        let val = (1.0 / bs)
                            * gamma_old[c]
                            * stddev_inv[c]
                            * (bs * g
                                - sum_grad[c]
                                - xc * stddev_inv[c] * stddev_inv[c] * sum_grad_xc[c]);
                        out.set(r, c, val);
                    }
                }
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Autoencoder: encoder layers + decoder layers in one chain, split at enc_len.
// ---------------------------------------------------------------------------
pub struct Autoencoder {
    layers: Vec<Layer>,
    enc_len: usize,
    pub input_dim: usize,
    pub latent_dim: usize,
}

/// Adam/optimizer hyperparameters, matching AE_init.py `Adam(lr=0.01, b1=0.5)`.
const LR: f64 = 0.01;
const B1: f64 = 0.5;
const B2: f64 = 0.999;
const BN_MOMENTUM: f64 = 0.8;

impl Autoencoder {
    pub fn new(seed: u64) -> Self {
        let input_dim = 32;
        let latent_dim = 4;
        let mut rng = Rng::new(seed);
        let relu = || Layer::Activation { kind: ActKind::Relu, in_cache: Mat::zeros(0, 0) };

        // Encoder 32 -> 16 -> 8 -> 4. (Each `dense` borrows `rng` only for its
        // own call, so the sequential `&mut rng` uses don't overlap.)
        let mut layers = vec![
            Layer::dense(32, 16, LR, B1, B2, &mut rng),
            relu(),
            Layer::bn(16, BN_MOMENTUM, LR, B1, B2),
            Layer::dense(16, 8, LR, B1, B2, &mut rng),
            relu(),
            Layer::bn(8, BN_MOMENTUM, LR, B1, B2),
            Layer::dense(8, 4, LR, B1, B2, &mut rng),
        ];
        let enc_len = layers.len();

        // Decoder 4 -> 8 -> 16 -> 32.
        layers.extend([
            Layer::dense(4, 8, LR, B1, B2, &mut rng),
            relu(),
            Layer::bn(8, BN_MOMENTUM, LR, B1, B2),
            Layer::dense(8, 16, LR, B1, B2, &mut rng),
            relu(),
            Layer::bn(16, BN_MOMENTUM, LR, B1, B2),
            Layer::dense(16, 32, LR, B1, B2, &mut rng),
            Layer::Activation { kind: ActKind::Sigmoid, in_cache: Mat::zeros(0, 0) },
        ]);

        Autoencoder { layers, enc_len, input_dim, latent_dim }
    }

    fn forward_range(&mut self, x: &Mat, start: usize, end: usize, training: bool) -> Mat {
        let mut out = x.clone();
        for l in start..end {
            out = self.layers[l].forward(&out, training);
        }
        out
    }

    /// Full reconstruction forward over the whole chain.
    fn forward_all(&mut self, x: &Mat, training: bool) -> Mat {
        let n = self.layers.len();
        self.forward_range(x, 0, n, training)
    }

    fn backward_all(&mut self, grad: &Mat) {
        let mut g = grad.clone();
        for l in (0..self.layers.len()).rev() {
            g = self.layers[l].backward(&g);
        }
    }

    /// Train on `data` (each row a 32-dim sample). `on_epoch(epoch, avg_loss)`
    /// is called after every epoch; returning `false` cancels training.
    /// Returns the number of completed epochs.
    ///
    /// Note: the backward pass sums (does not average) the per-element loss
    /// gradient over a batch's rows, matching the Python original. So the
    /// weight-gradient magnitude scales with `batch`, and `batch` acts as a
    /// secondary learning-rate knob — Adam's per-parameter normalisation absorbs
    /// most of this, but larger batches still train slightly more aggressively.
    pub fn fit(
        &mut self,
        data: &[[f32; 32]],
        epochs: usize,
        batch: usize,
        seed: u64,
        mut on_epoch: impl FnMut(usize, f64) -> bool,
    ) -> usize {
        let n = data.len();
        if n == 0 {
            return 0;
        }
        let batch = batch.max(1);
        let mut rng = Rng::new(seed ^ 0xD1B5_4A32_D192_ED03);
        let mut order: Vec<usize> = (0..n).collect();

        for epoch in 0..epochs {
            // Fisher-Yates shuffle each epoch (fix vs Python's sequential batches).
            for i in (1..n).rev() {
                let j = rng.below(i + 1);
                order.swap(i, j);
            }

            let mut epoch_loss = 0.0;
            let mut n_batches = 0;
            let mut start = 0;
            while start < n {
                let end = (start + batch).min(n);
                let rows = end - start;
                // Build batch matrix (target == input for an autoencoder).
                let mut x = Mat::zeros(rows, 32);
                for (r, &idx) in order[start..end].iter().enumerate() {
                    for (c, &val) in data[idx].iter().enumerate() {
                        x.set(r, c, val as f64);
                    }
                }
                let recon = self.forward_all(&x, true);
                // MSE 0.5*(y-yhat)^2 averaged over all elements (loss report).
                let mut s = 0.0;
                let mut grad = Mat::zeros(rows, 32);
                for idx in 0..x.data.len() {
                    let d = x.data[idx] - recon.data[idx];
                    s += 0.5 * d * d;
                    grad.data[idx] = -d; // -(y - yhat), elementwise, not averaged
                }
                epoch_loss += s / (x.data.len() as f64);
                self.backward_all(&grad);

                n_batches += 1;
                start = end;
            }
            let avg = epoch_loss / n_batches as f64;
            if !on_epoch(epoch, avg) {
                return epoch + 1;
            }
        }
        epochs
    }

    /// Encode a single 32-dim sample to a 4-dim latent (BN in inference mode).
    pub fn encode(&mut self, x: &[f64; 32]) -> [f64; 4] {
        let mut m = Mat::zeros(1, 32);
        m.data.copy_from_slice(x);
        let out = self.forward_range(&m, 0, self.enc_len, false);
        let mut z = [0.0; 4];
        z.copy_from_slice(&out.data[..4]);
        z
    }

    /// Decode a 4-dim latent to a 32-dim output through the decoder layers
    /// (BN in inference mode). Mirrors the exported `Decoder::forward`; used by
    /// tests to check export parity.
    pub fn decode(&mut self, z: &[f64; 4]) -> [f64; 32] {
        let mut m = Mat::zeros(1, 4);
        m.data.copy_from_slice(z);
        let out = self.forward_range(&m, self.enc_len, self.layers.len(), false);
        let mut y = [0.0; 32];
        y.copy_from_slice(&out.data[..32]);
        y
    }

    /// Index range of decoder layers, for the exporter (`model_ops.rs`).
    pub fn decoder_range(&self) -> std::ops::Range<usize> {
        self.enc_len..self.layers.len()
    }
    /// Index range of encoder layers, for the exporter.
    pub fn encoder_range(&self) -> std::ops::Range<usize> {
        0..self.enc_len
    }
    /// Read-only layer accessor for the exporter.
    pub(crate) fn layer_export(&self, i: usize) -> LayerView<'_> {
        match &self.layers[i] {
            Layer::Dense { n_in, n_out, w, b, .. } => {
                LayerView::Dense { n_in: *n_in, n_out: *n_out, w, b }
            }
            Layer::Activation { kind, .. } => LayerView::Activation(*kind),
            Layer::BatchNorm { gamma, beta, eps, running_mean, running_var, .. } => {
                LayerView::BatchNorm {
                    gamma,
                    beta,
                    eps: *eps,
                    running_mean: running_mean.as_deref(),
                    running_var: running_var.as_deref(),
                }
            }
        }
    }
}

/// Borrowed view of a layer's trained parameters, used by `model_ops::export_*`.
pub(crate) enum LayerView<'a> {
    Dense { n_in: usize, n_out: usize, w: &'a [f64], b: &'a [f64] },
    Activation(ActKind),
    BatchNorm {
        gamma: &'a [f64],
        beta: &'a [f64],
        eps: f64,
        running_mean: Option<&'a [f64]>,
        running_var: Option<&'a [f64]>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adam with real bias correction differs from the buggy constant-divisor
    /// form on the very first step. Locks in the fix (decision 2).
    #[test]
    fn adam_bias_correction_is_applied() {
        let mut a = Adam::new(0.01, 0.5, 0.999, 1);
        let mut w = [0.0];
        a.step(&mut w, &[1.0]);
        // Correct: m=0.5, m_hat=0.5/(1-0.5)=1.0; v=0.001, v_hat=0.001/(1-0.999)=1.0;
        // w -= 0.01 * 1.0 / (1.0 + 1e-8) ~= -0.01.
        assert!((w[0] + 0.01).abs() < 1e-6, "got {}", w[0]);
    }

    /// The real gradient-correctness gate: a tiny fixed dataset must be driven
    /// to near-zero reconstruction loss. Only correct backprop converges here.
    #[test]
    fn overfits_tiny_dataset() {
        let data = [
            [0.0f32; 32],
            {
                let mut v = [0.0f32; 32];
                v[0] = 1.0;
                v[5] = 1.0;
                v[16] = 0.5;
                v
            },
            {
                let mut v = [0.0f32; 32];
                for (i, slot) in v.iter_mut().enumerate().take(16) {
                    *slot = (i % 2) as f32;
                }
                v
            },
            {
                // First 16 steps on, no substeps.
                let mut v = [0.0f32; 32];
                for x in v.iter_mut().take(16) {
                    *x = 1.0;
                }
                v
            },
        ];
        let mut ae = Autoencoder::new(42);
        let mut last = f64::INFINITY;
        ae.fit(&data, 1500, 4, 7, |_, loss| {
            last = loss;
            true
        });
        assert!(last < 1e-3, "did not converge: final loss {last}");

        // Reconstruction (infer mode) should match each input closely.
        let mut x = Mat::zeros(data.len(), 32);
        for (r, row) in data.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                x.set(r, c, val as f64);
            }
        }
        let recon = ae.forward_all(&x, false);
        let mut worst = 0.0f64;
        for idx in 0..x.data.len() {
            worst = worst.max((x.data[idx] - recon.data[idx]).abs());
        }
        assert!(worst < 0.1, "infer-mode reconstruction off by {worst}");
    }

    /// Cancellation: returning false from the callback stops early.
    #[test]
    fn fit_can_be_cancelled() {
        let data = [[0.0f32; 32], [1.0f32; 32]];
        let mut ae = Autoencoder::new(1);
        let done = ae.fit(&data, 100, 2, 1, |e, _| e < 4);
        assert_eq!(done, 5, "should stop after epoch 4 returns false");
    }

    /// Shuffle is deterministic given a seed; different seeds diverge.
    #[test]
    fn shuffle_is_seed_deterministic() {
        let data: Vec<[f32; 32]> = (0..20)
            .map(|i| {
                let mut v = [0.0f32; 32];
                v[i % 32] = 1.0;
                v
            })
            .collect();
        let run = |seed: u64| {
            let mut ae = Autoencoder::new(99);
            let mut losses = Vec::new();
            ae.fit(&data, 5, 4, seed, |_, l| {
                losses.push(l);
                true
            });
            losses
        };
        assert_eq!(run(123), run(123), "same seed must reproduce");
        assert_ne!(run(123), run(456), "different seeds should diverge");
    }
}
