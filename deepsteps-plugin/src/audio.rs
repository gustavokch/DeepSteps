//! Audio-file ingestion for the dataset builder: decode a file to mono f32
//! (symphonia), detect onsets with a spectral-flux + adaptive-peak-pick
//! detector (rustfft), then encode to a 32-dim training sample.
//!
//! This is deliberately NOT a librosa clone (project plan, decision 2): the
//! file-derived dataset is approximate. The detector is a single self-contained
//! function with exposed constants for tuning.

use std::path::Path;

use rustfft::{num_complex::Complex, FftPlanner};

use crate::dataset::encode_onsets;

const FRAME: usize = 1024;
const HOP: usize = 512;
/// Adaptive-threshold moving-average window, in frames.
const THRESH_WIN: usize = 8;
/// Threshold = mean(window) * THRESH_MULT + THRESH_DELTA (on normalized flux).
const THRESH_MULT: f32 = 1.5;
const THRESH_DELTA: f32 = 0.04;
/// Minimum gap between detected onsets, in frames (refractory period).
const MIN_GAP_FRAMES: usize = 3;

#[derive(Debug)]
pub enum DecodeError {
    Io(String),
    Unsupported(String),
    Decode(String),
    Empty,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Io(s) => write!(f, "io error: {s}"),
            DecodeError::Unsupported(s) => write!(f, "unsupported audio: {s}"),
            DecodeError::Decode(s) => write!(f, "decode error: {s}"),
            DecodeError::Empty => write!(f, "decoded audio was empty"),
        }
    }
}

/// Decode an audio file to mono f32 samples + sample rate. Channels are
/// averaged. Supports the formats enabled in symphonia's Cargo features.
pub fn decode_audio(path: &Path) -> Result<(Vec<f32>, u32), DecodeError> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| DecodeError::Io(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| DecodeError::Unsupported(e.to_string()))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| DecodeError::Unsupported("no decodable track".into()))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44_100);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| DecodeError::Unsupported(e.to_string()))?;

    let mut mono: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(DecodeError::Decode(e.to_string())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = *audio_buf.spec();
                let channels = spec.channels.count().max(1);
                if sample_buf.is_none() {
                    sample_buf =
                        Some(SampleBuffer::<f32>::new(audio_buf.capacity() as u64, spec));
                }
                let buf = sample_buf.as_mut().unwrap();
                buf.copy_interleaved_ref(audio_buf);
                // Downmix interleaved channels to mono.
                for frame in buf.samples().chunks(channels) {
                    let sum: f32 = frame.iter().copied().sum();
                    mono.push(sum / channels as f32);
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue, // skip bad packet
            Err(e) => return Err(DecodeError::Decode(e.to_string())),
        }
    }

    if mono.is_empty() {
        return Err(DecodeError::Empty);
    }
    Ok((mono, sample_rate))
}

/// Detect onset sample positions in mono audio via spectral flux + adaptive
/// peak picking. Returns positions in samples, ascending.
pub fn detect_onsets(mono: &[f32], _sr: u32) -> Vec<usize> {
    if mono.len() < FRAME + HOP {
        return Vec::new();
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FRAME);

    // Hann window.
    let window: Vec<f32> = (0..FRAME)
        .map(|n| {
            let w = (std::f32::consts::PI * n as f32 / (FRAME as f32 - 1.0)).sin();
            w * w
        })
        .collect();

    let n_frames = (mono.len() - FRAME) / HOP + 1;
    let bins = FRAME / 2;
    let mut prev_mag = vec![0.0f32; bins];
    let mut flux = vec![0.0f32; n_frames];
    let mut scratch = vec![Complex::new(0.0f32, 0.0); FRAME];

    for (t, f) in flux.iter_mut().enumerate() {
        let start = t * HOP;
        for i in 0..FRAME {
            scratch[i] = Complex::new(mono[start + i] * window[i], 0.0);
        }
        fft.process(&mut scratch);
        let mut sf = 0.0f32;
        for k in 0..bins {
            let mag = scratch[k].norm();
            let d = mag - prev_mag[k];
            if d > 0.0 {
                sf += d;
            }
            prev_mag[k] = mag;
        }
        *f = sf;
    }

    // Normalize flux to [0,1].
    let max = flux.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return Vec::new();
    }
    for v in &mut flux {
        *v /= max;
    }

    // Adaptive peak pick: local maximum, above moving-average threshold, with a
    // refractory gap.
    let mut onsets = Vec::new();
    let mut last_onset: Option<usize> = None;
    for t in 1..n_frames - 1 {
        let lo = t.saturating_sub(THRESH_WIN);
        let hi = (t + THRESH_WIN + 1).min(n_frames);
        let mean: f32 = flux[lo..hi].iter().copied().sum::<f32>() / (hi - lo) as f32;
        let thresh = mean * THRESH_MULT + THRESH_DELTA;

        let is_peak = flux[t] > flux[t - 1] && flux[t] >= flux[t + 1] && flux[t] > thresh;
        if is_peak {
            if let Some(prev) = last_onset {
                if t - prev < MIN_GAP_FRAMES {
                    // Keep the stronger of the two close peaks.
                    if flux[t] > flux[prev] {
                        onsets.pop();
                        onsets.push(t * HOP);
                        last_onset = Some(t);
                    }
                    continue;
                }
            }
            onsets.push(t * HOP);
            last_onset = Some(t);
        }
    }
    onsets
}

/// Decode a file and encode it to a single 32-dim training sample. The whole
/// file is treated as one bar (matching `corpus_encode.py`'s `bar_length=1`).
/// Returns `Ok(None)` when the file decodes but has no detectable onsets.
pub fn file_to_sample(path: &Path) -> Result<Option<[f32; 32]>, DecodeError> {
    let (mono, sr) = decode_audio(path)?;
    let onsets = detect_onsets(&mono, sr);
    if onsets.is_empty() {
        return Ok(None);
    }
    let onsets_i64: Vec<i64> = onsets.iter().map(|&o| o as i64).collect();
    Ok(Some(encode_onsets(&onsets_i64, mono.len() as i64)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic click train should yield onsets near the click positions.
    #[test]
    fn detects_click_train() {
        let sr = 44_100u32;
        let len = 44_100; // 1 second
        let mut sig = vec![0.0f32; len];
        let clicks = [5_000usize, 16_000, 28_000, 39_000];
        for &c in &clicks {
            // Short decaying burst so a frame sees a clear energy rise.
            for i in 0..256 {
                if c + i < len {
                    let env = 1.0 - (i as f32 / 256.0);
                    // broadband-ish: alternate sign
                    sig[c + i] = if i % 2 == 0 { env } else { -env };
                }
            }
        }
        let onsets = detect_onsets(&sig, sr);
        assert!(!onsets.is_empty(), "no onsets detected");

        // Every real click should have a detected onset within ~1.5 hops.
        let tol = (HOP as f32 * 1.5) as usize + FRAME; // detection lags by ~a frame
        for &c in &clicks {
            let found = onsets.iter().any(|&o| o.abs_diff(c) <= tol);
            assert!(found, "no onset near click {c}; got {onsets:?}");
        }
        // Should not produce a flood of spurious onsets.
        assert!(onsets.len() <= clicks.len() + 2, "too many onsets: {onsets:?}");
    }

    #[test]
    fn silence_has_no_onsets() {
        let onsets = detect_onsets(&vec![0.0f32; 44_100], 44_100);
        assert!(onsets.is_empty());
    }

    #[test]
    fn short_signal_is_safe() {
        assert!(detect_onsets(&[0.0f32; 100], 44_100).is_empty());
    }
}
