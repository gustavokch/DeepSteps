//! Runtime training state + background-thread driver.
//!
//! Training must never run on the audio thread. It runs on nih-plug's background
//! thread (via `Plugin::task_executor` + `AsyncExecutor::execute_background`),
//! reading/writing the `Arc<TrainShared>` shared with the audio and GUI threads.
//!
//! The audio thread only ever does a wait-free `model.load()` (an `ArcSwap`) and
//! a few `Relaxed` atomic reads — it never locks a `Mutex`. The dataset and
//! trained-model `Mutex`es are touched solely by the GUI and background threads.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use arc_swap::ArcSwap;

use crate::autoencoder::Autoencoder;
use crate::decoder::Decoder;
use crate::model_ops::{self, TrainedModel};

/// A background task. Kept `Copy`/heap-free per nih-plug's `BackgroundTask`
/// contract; all data flows through `TrainShared`.
#[derive(Clone, Copy)]
pub enum Task {
    /// Train an autoencoder on the current dataset and hot-swap the result.
    Train { epochs: usize, batch: usize, seed: u64 },
    /// Decode + onset-detect the files queued in `pending_paths`, appending each
    /// to the dataset.
    IngestAudio,
}

/// Current background-operation status, for the GUI. Distinct from whether a
/// trained model exists (that is tracked by `model_generation > 0`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrainStatus {
    Idle,
    Ingesting,
    Running,
    Done,
    Cancelled,
    Error,
}

impl TrainStatus {
    fn to_u8(self) -> u8 {
        match self {
            TrainStatus::Idle => 0,
            TrainStatus::Ingesting => 1,
            TrainStatus::Running => 2,
            TrainStatus::Done => 3,
            TrainStatus::Cancelled => 4,
            TrainStatus::Error => 5,
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => TrainStatus::Ingesting,
            2 => TrainStatus::Running,
            3 => TrainStatus::Done,
            4 => TrainStatus::Cancelled,
            5 => TrainStatus::Error,
            _ => TrainStatus::Idle,
        }
    }
}

/// State shared between the audio thread, the GUI thread, and the background
/// training thread.
pub struct TrainShared {
    /// Accumulated training samples. GUI/background only.
    pub dataset: Mutex<Vec<[f32; 32]>>,
    /// Audio files queued for ingestion (the GUI can't pass a `Vec<PathBuf>`
    /// through the `Copy` task enum, so it parks them here). GUI/background only.
    pub pending_paths: Mutex<Vec<PathBuf>>,

    status: AtomicU8,
    /// Completed epochs (for the progress bar).
    pub epoch: AtomicUsize,
    pub total_epochs: AtomicUsize,
    /// Latest epoch loss, as `f64::to_bits` (display only).
    last_loss_bits: AtomicU64,
    /// GUI sets this to request the running training stop after the next epoch.
    pub cancel: AtomicBool,
    /// Bumped on every model swap so the audio thread knows to regenerate.
    pub model_generation: AtomicU64,

    /// The live decoder the audio thread runs. Hot-swapped on training finish.
    pub model: ArcSwap<Decoder>,
    /// The live encoder for the "encode pattern -> latent" feature. `None` until
    /// a model is trained/restored (the baked default ships no encoder).
    pub encoder: ArcSwap<Option<Decoder>>,
    /// Persisted trained model (shared with the `#[persist]` param field).
    pub trained_model: Arc<Mutex<Option<TrainedModel>>>,
}

impl TrainShared {
    pub fn new(
        initial_decoder: Decoder,
        trained_model: Arc<Mutex<Option<TrainedModel>>>,
    ) -> Self {
        TrainShared {
            dataset: Mutex::new(Vec::new()),
            pending_paths: Mutex::new(Vec::new()),
            status: AtomicU8::new(TrainStatus::Idle.to_u8()),
            epoch: AtomicUsize::new(0),
            total_epochs: AtomicUsize::new(0),
            last_loss_bits: AtomicU64::new(0),
            cancel: AtomicBool::new(false),
            model_generation: AtomicU64::new(0),
            model: ArcSwap::from_pointee(initial_decoder),
            encoder: ArcSwap::from_pointee(None),
            trained_model,
        }
    }

    pub fn status(&self) -> TrainStatus {
        TrainStatus::from_u8(self.status.load(Relaxed))
    }
    fn set_status(&self, s: TrainStatus) {
        self.status.store(s.to_u8(), Relaxed);
    }
    pub fn last_loss(&self) -> f64 {
        f64::from_bits(self.last_loss_bits.load(Relaxed))
    }
    pub fn dataset_len(&self) -> usize {
        self.dataset.lock().map(|d| d.len()).unwrap_or(0)
    }
    pub fn has_trained_model(&self) -> bool {
        self.model_generation.load(Relaxed) > 0
    }
    /// Encode a 32-dim pattern to a latent using the live encoder, if one exists.
    pub fn encode(&self, x: &[f64]) -> Option<[f64; 4]> {
        let guard = self.encoder.load();
        let enc = guard.as_ref().as_ref()?;
        let out = enc.forward(x);
        if out.len() < 4 {
            return None;
        }
        Some([out[0], out[1], out[2], out[3]])
    }
}

/// Build the background task executor closure. Called once by `Plugin::task_executor`.
pub fn executor(train: Arc<TrainShared>) -> Box<dyn Fn(Task) + Send> {
    Box::new(move |task| match task {
        Task::Train { epochs, batch, seed } => run_training(&train, epochs, batch, seed),
        Task::IngestAudio => run_ingest(&train),
    })
}

fn run_training(train: &Arc<TrainShared>, epochs: usize, batch: usize, seed: u64) {
    train.cancel.store(false, Relaxed);
    train.epoch.store(0, Relaxed);
    train.total_epochs.store(epochs, Relaxed);
    train.set_status(TrainStatus::Running);

    let data = match train.dataset.lock() {
        Ok(d) => d.clone(),
        Err(_) => {
            train.set_status(TrainStatus::Error);
            return;
        }
    };
    if data.is_empty() {
        train.set_status(TrainStatus::Error);
        return;
    }

    let mut ae = Autoencoder::new(seed);
    let completed = ae.fit(&data, epochs, batch, seed, |epoch, loss| {
        train.epoch.store(epoch + 1, Relaxed);
        train.last_loss_bits.store(loss.to_bits(), Relaxed);
        !train.cancel.load(Relaxed)
    });

    // Export + swap even if cancelled: a partially trained net is still a valid
    // model (BN running stats are populated after the first forward).
    let dec_exp = model_ops::export_decoder(&ae);
    let enc_exp = model_ops::export_encoder(&ae);
    match (dec_exp, enc_exp) {
        (Ok(dec_exp), Ok(enc_exp)) => {
            if let Ok(dec) = model_ops::to_decoder(&dec_exp) {
                train.model.store(Arc::new(dec));
            }
            if let Ok(enc) = model_ops::to_decoder(&enc_exp) {
                train.encoder.store(Arc::new(Some(enc)));
            }
            if let Ok(mut slot) = train.trained_model.lock() {
                *slot = Some(TrainedModel { decoder: dec_exp, encoder: enc_exp });
            }
            train.model_generation.fetch_add(1, Relaxed);
            train.set_status(if completed < epochs {
                TrainStatus::Cancelled
            } else {
                TrainStatus::Done
            });
        }
        _ => train.set_status(TrainStatus::Error),
    }
}

fn run_ingest(train: &Arc<TrainShared>) {
    let prev = train.status();
    train.set_status(TrainStatus::Ingesting);
    let paths: Vec<PathBuf> = match train.pending_paths.lock() {
        Ok(mut q) => std::mem::take(&mut *q),
        Err(_) => Vec::new(),
    };
    for p in paths {
        match crate::audio::file_to_sample(&p) {
            Ok(Some(v)) => {
                if let Ok(mut d) = train.dataset.lock() {
                    d.push(v);
                }
            }
            Ok(None) => nih_plug::nih_log!("DeepSteps: no onsets in {p:?}, skipped"),
            Err(e) => nih_plug::nih_log!("DeepSteps: ingest failed for {p:?}: {e}"),
        }
    }
    // Restore the prior status (e.g. keep showing Done) unless training is active.
    train.set_status(if prev == TrainStatus::Running {
        TrainStatus::Running
    } else {
        TrainStatus::Idle
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::encode_grid;

    fn shared() -> Arc<TrainShared> {
        let slot = Arc::new(Mutex::new(None));
        Arc::new(TrainShared::new(Decoder::empty(), slot))
    }

    /// End-to-end: dataset -> Train task -> hot-swap + persistence slot filled.
    #[test]
    fn train_task_swaps_and_persists() {
        let train = shared();
        // A few distinct patterns captured from "grids".
        {
            let mut d = train.dataset.lock().unwrap();
            for i in 0..8u16 {
                let mask = i.wrapping_mul(37) | 1;
                let mut ss = [0.0f64; 16];
                ss[(i % 16) as usize] = 0.5;
                d.push(encode_grid(mask, &ss));
            }
        }
        assert_eq!(train.model_generation.load(Relaxed), 0);
        assert!(!train.has_trained_model());

        // Run the executor inline (no real background thread needed for the test).
        executor(train.clone())(Task::Train { epochs: 40, batch: 4, seed: 1 });

        assert_eq!(train.status(), TrainStatus::Done);
        assert_eq!(train.model_generation.load(Relaxed), 1);
        assert!(train.has_trained_model());
        // Persistence slot is filled with both halves.
        assert!(train.trained_model.lock().unwrap().is_some());
        // Encoder is live and produces a 4-dim latent.
        let x: Vec<f64> = encode_grid(0b1010_1010_1010_1010, &[0.5; 16])
            .iter()
            .map(|&v| v as f64)
            .collect();
        let z = train.encode(&x).expect("encoder available after training");
        assert!(z.iter().all(|v| v.is_finite()));
        // The swapped decoder runs and yields a 32-element pattern.
        let dec = train.model.load();
        let out = dec.forward(&[z[0], z[1], z[2], z[3]]);
        assert_eq!(out.len(), 32);
    }

    /// Training with an empty dataset reports an error and swaps nothing.
    #[test]
    fn train_empty_dataset_errors() {
        let train = shared();
        executor(train.clone())(Task::Train { epochs: 5, batch: 2, seed: 1 });
        assert_eq!(train.status(), TrainStatus::Error);
        assert_eq!(train.model_generation.load(Relaxed), 0);
        assert!(!train.has_trained_model());
    }
}
