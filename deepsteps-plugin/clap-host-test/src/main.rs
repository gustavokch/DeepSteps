//! Headless CLAP host that dlopens the SHIPPED `deepsteps-plugin.clap` and verifies
//! per-scale pitch quantization end-to-end.
//!
//! For each of the 14 scales it: instantiates a fresh plugin, sets the `Scale` /
//! `Key` params + a spread of 16 note pitches (one per pitch class) via CLAP
//! param-value events, drives a *playing* host transport over one full 4-beat bar,
//! collects the emitted NoteOn events, and asserts every emitted pitch's pitch
//! class is a member of that scale's table (the quantization contract) — and that
//! it equals the reference snap-down of one of the input pitches.
//!
//! This exercises the real artifact: the bundled `.clap` shared object, its CLAP
//! ABI, the nih-plug wrapper, the param plumbing, and the sequencer/quantizer.

use std::ffi::{c_char, c_void, CString};

use clap_sys::entry::clap_plugin_entry;
use clap_sys::events::{
    clap_event_header, clap_event_note, clap_event_param_value, clap_event_transport,
    clap_input_events, clap_output_events, CLAP_CORE_EVENT_SPACE_ID, CLAP_EVENT_NOTE_ON,
    CLAP_EVENT_PARAM_VALUE, CLAP_TRANSPORT_HAS_BEATS_TIMELINE, CLAP_TRANSPORT_HAS_TEMPO,
    CLAP_TRANSPORT_IS_PLAYING,
};
use clap_sys::ext::params::{clap_param_info, clap_plugin_params, CLAP_EXT_PARAMS};
use clap_sys::factory::plugin_factory::{clap_plugin_factory, CLAP_PLUGIN_FACTORY_ID};
use clap_sys::fixedpoint::CLAP_BEATTIME_FACTOR;
use clap_sys::host::clap_host;
use clap_sys::plugin::clap_plugin;
use clap_sys::process::{clap_process, CLAP_PROCESS_ERROR};
use clap_sys::version::CLAP_VERSION;

const PLUGIN_ID: &str = "dev.gruber.deepsteps";
const SAMPLE_RATE: f64 = 48000.0;
const BLOCK: u32 = 4096;
const TEMPO: f64 = 120.0;
/// Internal min of the note Pitch IntParam (params.rs: `IntRange { min: 24, .. }`).
/// nih-plug zero-bases IntParam CLAP values, so CLAP value = midi_note - PITCH_MIN.
const PITCH_MIN: usize = 24;

/// The 14 scale tables, mirroring `sequencer.rs`. Index = the `ScaleParam` variant
/// index (declaration order), which is the value the CLAP enum param takes.
const SCALES: &[(&str, &[i32])] = &[
    ("Chromatic", &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ("Pentatonic Major", &[0, 2, 4, 7, 9]),
    ("Pentatonic Minor", &[0, 3, 5, 7, 10]),
    ("Major", &[0, 2, 4, 5, 7, 9, 11]),
    ("Natural Minor", &[0, 2, 3, 5, 7, 8, 10]),
    ("Harmonic Minor", &[0, 2, 3, 5, 7, 8, 11]),
    ("Melodic Minor", &[0, 2, 3, 5, 7, 9, 11]),
    ("Dorian", &[0, 2, 3, 5, 7, 9, 10]),
    ("Phrygian", &[0, 1, 3, 5, 7, 8, 10]),
    ("Lydian", &[0, 2, 4, 6, 7, 9, 11]),
    ("Mixolydian", &[0, 2, 4, 5, 7, 9, 10]),
    ("Locrian", &[0, 1, 3, 5, 6, 8, 10]),
    ("Blues", &[0, 3, 5, 6, 7, 10]),
    ("Whole Tone", &[0, 2, 4, 6, 8, 10]),
];

/// Reference snap-down (matches `sequencer::quantize`): snap pitch class DOWN to the
/// nearest table member, keep octave, add key.
fn quantize_ref(note: i32, table: &[i32], key: i32) -> i32 {
    let pc = note.rem_euclid(12);
    let octave = note - pc;
    let snapped = table.iter().rev().copied().find(|&d| d <= pc).unwrap_or(table[0]);
    octave + snapped + key
}

fn read_cstr(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// --- host-side event list callbacks (C ABI) -------------------------------------

unsafe extern "C" fn in_size(list: *const clap_input_events) -> u32 {
    let v = &*((*list).ctx as *const Vec<clap_event_param_value>);
    v.len() as u32
}
unsafe extern "C" fn in_get(
    list: *const clap_input_events,
    index: u32,
) -> *const clap_event_header {
    let v = &*((*list).ctx as *const Vec<clap_event_param_value>);
    &v[index as usize].header as *const clap_event_header
}

/// Collected NoteOn keys (pitches) pushed by the plugin during process().
unsafe extern "C" fn out_try_push(
    list: *const clap_output_events,
    event: *const clap_event_header,
) -> bool {
    if (*event).type_ == CLAP_EVENT_NOTE_ON {
        let note = &*(event as *const clap_event_note);
        let out = &mut *((*list).ctx as *mut Vec<i16>);
        out.push(note.key);
    }
    true
}

fn param_event(param_id: u32, value: f64) -> clap_event_param_value {
    clap_event_param_value {
        header: clap_event_header {
            size: std::mem::size_of::<clap_event_param_value>() as u32,
            time: 0,
            space_id: CLAP_CORE_EVENT_SPACE_ID,
            type_: CLAP_EVENT_PARAM_VALUE,
            flags: 0,
        },
        param_id,
        cookie: std::ptr::null_mut(),
        note_id: -1,
        port_index: -1,
        channel: -1,
        key: -1,
        value,
    }
}

/// Run one full bar for a given scale index; returns the emitted NoteOn pitches.
unsafe fn run_scale(
    plugin: *const clap_plugin,
    params: *const clap_plugin_params,
    scale_id: u32,
    key_id: u32,
    pitch_ids: &[u32],
    scale_index: f64,
) -> Vec<i16> {
    // Build the param-value events: scale, key=0, and a spread of note pitches.
    let mut events = vec![param_event(scale_id, scale_index), param_event(key_id, 0.0)];
    for (i, &pid) in pitch_ids.iter().enumerate() {
        // nih-plug zero-bases IntParam CLAP values: the Pitch param's CLAP range is
        // 0..72, i.e. `midi_note - PITCH_MIN`. So to set actual MIDI pitch 60+i we
        // send (60+i) - PITCH_MIN. 60..=75 covers every pitch class (0..11 then 0..3),
        // so some firing steps always land off-scale and must snap.
        events.push(param_event(pid, (60 + i - PITCH_MIN) as f64));
    }

    let in_events = clap_input_events {
        ctx: &events as *const _ as *mut c_void,
        size: Some(in_size),
        get: Some(in_get),
    };
    let mut collected: Vec<i16> = Vec::new();
    let out_events = clap_output_events {
        ctx: &mut collected as *mut _ as *mut c_void,
        try_push: Some(out_try_push),
    };

    // Flush the param changes before processing so they're applied up front.
    if let Some(flush) = (*params).flush {
        flush(plugin, &in_events, &out_events);
    }

    let beats_per_sample = TEMPO / 60.0 / SAMPLE_RATE;
    let beats_per_block = BLOCK as f64 * beats_per_sample;

    let mut pos_beats = 0.0_f64;
    let mut steady: i64 = 0;
    // One bar = 4 beats (16 steps). Run a hair past 4.0 to include the last step.
    while pos_beats < 4.0 {
        let transport = clap_event_transport {
            header: clap_event_header {
                size: std::mem::size_of::<clap_event_transport>() as u32,
                time: 0,
                space_id: CLAP_CORE_EVENT_SPACE_ID,
                type_: 0,
                flags: 0,
            },
            flags: CLAP_TRANSPORT_HAS_TEMPO
                | CLAP_TRANSPORT_HAS_BEATS_TIMELINE
                | CLAP_TRANSPORT_IS_PLAYING,
            song_pos_beats: (pos_beats * CLAP_BEATTIME_FACTOR as f64).round() as i64,
            song_pos_seconds: 0,
            tempo: TEMPO,
            tempo_inc: 0.0,
            loop_start_beats: 0,
            loop_end_beats: 0,
            loop_start_seconds: 0,
            loop_end_seconds: 0,
            bar_start: 0,
            bar_number: 0,
            tsig_num: 4,
            tsig_denom: 4,
        };
        // Only feed the param events on the first block.
        let empty_in = clap_input_events {
            ctx: &Vec::<clap_event_param_value>::new() as *const _ as *mut c_void,
            size: Some(in_size),
            get: Some(in_get),
        };
        let process = clap_process {
            steady_time: steady,
            frames_count: BLOCK,
            transport: &transport,
            audio_inputs: std::ptr::null(),
            audio_outputs: std::ptr::null_mut(),
            audio_inputs_count: 0,
            audio_outputs_count: 0,
            in_events: &empty_in,
            out_events: &out_events,
        };
        let status = ((*plugin).process.unwrap())(plugin, &process);
        if status == CLAP_PROCESS_ERROR {
            panic!("process() returned CLAP_PROCESS_ERROR");
        }
        pos_beats += beats_per_block;
        steady += BLOCK as i64;
    }
    collected
}

fn main() {
    let clap_path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../target/bundled/deepsteps-plugin.clap").to_string()
    });
    println!("Loading: {clap_path}\n");

    unsafe {
        let lib = libloading::Library::new(&clap_path).expect("dlopen .clap");
        let entry: libloading::Symbol<*const clap_plugin_entry> =
            lib.get(b"clap_entry\0").expect("clap_entry symbol");
        let entry = &**entry;

        let path_c = CString::new(clap_path.as_str()).unwrap();
        assert!((entry.init.unwrap())(path_c.as_ptr()), "entry.init failed");

        let factory = (entry.get_factory.unwrap())(CLAP_PLUGIN_FACTORY_ID.as_ptr())
            as *const clap_plugin_factory;
        assert!(!factory.is_null(), "no plugin factory");

        // Minimal host. The plugin queries extensions via get_extension; returning
        // null for all is fine (nih-plug degrades gracefully).
        let host = clap_host {
            clap_version: CLAP_VERSION,
            host_data: std::ptr::null_mut(),
            name: c"scale-test".as_ptr(),
            vendor: c"DeepSteps".as_ptr(),
            url: c"".as_ptr(),
            version: c"0".as_ptr(),
            get_extension: Some(host_get_extension),
            request_restart: Some(host_noop),
            request_process: Some(host_noop),
            request_callback: Some(host_noop),
        };

        let id_c = CString::new(PLUGIN_ID).unwrap();
        let mut fails = 0u32;

        for (scale_index, (name, table)) in SCALES.iter().enumerate() {
            let plugin =
                ((*factory).create_plugin.unwrap())(factory, &host, id_c.as_ptr());
            assert!(!plugin.is_null(), "create_plugin returned null");
            assert!(((*plugin).init.unwrap())(plugin), "plugin.init failed");

            let params =
                ((*plugin).get_extension.unwrap())(plugin, CLAP_EXT_PARAMS.as_ptr())
                    as *const clap_plugin_params;
            assert!(!params.is_null(), "no params extension");

            // Enumerate params: find Scale, Key, and the 16 note Pitch ids.
            let (mut scale_id, mut key_id) = (None, None);
            let mut pitch_ids: Vec<u32> = Vec::new();
            let count = ((*params).count.unwrap())(plugin);
            for i in 0..count {
                let mut info: clap_param_info = std::mem::zeroed();
                if !((*params).get_info.unwrap())(plugin, i, &mut info) {
                    continue;
                }
                match read_cstr(&info.name).as_str() {
                    "Scale" => scale_id = Some(info.id),
                    "Key" => key_id = Some(info.id),
                    "Pitch" => pitch_ids.push(info.id),
                    _ => {}
                }
            }
            let scale_id = scale_id.expect("Scale param not found");
            let key_id = key_id.expect("Key param not found");
            assert_eq!(pitch_ids.len(), 16, "expected 16 note Pitch params");

            assert!(
                ((*plugin).activate.unwrap())(plugin, SAMPLE_RATE, 1, BLOCK),
                "activate failed"
            );
            assert!(((*plugin).start_processing.unwrap())(plugin), "start_processing failed");

            let notes = run_scale(
                plugin,
                params,
                scale_id,
                key_id,
                &pitch_ids,
                scale_index as f64,
            );

            ((*plugin).stop_processing.unwrap())(plugin);
            ((*plugin).deactivate.unwrap())(plugin);
            ((*plugin).destroy.unwrap())(plugin);

            // Verify: every emitted pitch is in-scale, and equals a reference snap.
            let expected: Vec<i32> = (60..76).map(|p| quantize_ref(p, table, 0)).collect();
            let mut bad = Vec::new();
            for &n in &notes {
                let pc = (n as i32).rem_euclid(12);
                let in_scale = table.contains(&pc);
                let is_ref = expected.contains(&(n as i32));
                if !in_scale || !is_ref {
                    bad.push(n);
                }
            }
            let mut pcs: Vec<i32> =
                notes.iter().map(|&n| (n as i32).rem_euclid(12)).collect();
            pcs.sort_unstable();
            pcs.dedup();
            let ok = bad.is_empty() && !notes.is_empty();
            if !ok {
                fails += 1;
            }
            println!(
                "{:<16} value={:>2}  notes={:>2}  pcs={:?}  table={:?}  {}",
                name,
                scale_index,
                notes.len(),
                pcs,
                table,
                if ok {
                    "PASS".to_string()
                } else if notes.is_empty() {
                    "FAIL (no notes emitted)".to_string()
                } else {
                    format!("FAIL (out-of-scale: {bad:?})")
                }
            );
        }

        (entry.deinit.unwrap())();
        println!();
        if fails == 0 {
            println!("ALL 14 SCALES PASS");
        } else {
            eprintln!("{fails} scale(s) FAILED");
            std::process::exit(1);
        }
    }
}

unsafe extern "C" fn host_get_extension(
    _host: *const clap_host,
    _id: *const c_char,
) -> *const c_void {
    std::ptr::null()
}
unsafe extern "C" fn host_noop(_host: *const clap_host) {}
