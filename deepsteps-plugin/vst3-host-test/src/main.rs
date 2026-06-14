//! Headless VST3 host that dlopens the SHIPPED `deepsteps-plugin.vst3` and verifies
//! per-scale pitch quantization end-to-end.
//!
//! For each of the 14 scales it: instantiates a fresh component, sets the `Scale` /
//! `Key` params + a spread of 16 note pitches (one per pitch class) via VST3 input
//! parameter changes, drives a *playing* host transport (a `ProcessContext`) over one
//! full 4-beat bar, collects the emitted NoteOn events, and asserts every emitted
//! pitch's pitch class is a member of that scale's table (the quantization contract)
//! — and that it equals the reference snap-down of one of the input pitches.
//!
//! This exercises the real artifact: the bundled `.vst3` shared object, its COM ABI,
//! the nih-plug wrapper, the param plumbing, and the sequencer/quantizer. It mirrors
//! the sibling `clap-host-test` harness exactly, just feeding a VST3 host instead.
#![allow(clippy::missing_safety_doc)]

use std::ffi::c_void;

// The `vst3-com` dependency provides the `vst3_com::` crate path the `#[VST3(...)]`
// co_class macro expands to; it is referenced only by macro-generated code.
use vst3_sys::base::{kResultOk, tresult, IPluginBase, IPluginFactory};
use vst3_sys::utils::StaticVstPtr;
use vst3_sys::vst::{
    BusDirections, Event, IAudioProcessor, IComponent, IEditController, IEventList,
    IParamValueQueue, IParameterChanges, MediaTypes, NoteOnEvent, ProcessContext, ProcessData,
    ProcessSetup,
};
use vst3_sys::{ComInterface, VstPtr, VST3};

const SAMPLE_RATE: f64 = 48000.0;
const BLOCK: i32 = 4096;
const TEMPO: f64 = 120.0;

// VST3 ProcessContext::state flags (not exposed by the bindings — SDK values).
const K_PLAYING: u32 = 1 << 1; // 2
const K_PROJECT_TIME_MUSIC_VALID: u32 = 1 << 9; // 512
const K_TEMPO_VALID: u32 = 1 << 10; // 1024
const TRANSPORT_STATE: u32 = K_PLAYING | K_PROJECT_TIME_MUSIC_VALID | K_TEMPO_VALID; // 1538

const K_NOTE_ON_EVENT: u16 = 0; // EventTypes::kNoteOnEvent

/// The 14 scale tables, mirroring `sequencer.rs`. Index = the `ScaleParam` variant
/// index (declaration order), which is the plain value the Scale param takes.
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

/// Decode a VST3 `String128` (`[i16; 128]` UTF-16) into a `String`.
fn read_utf16(buf: &[i16]) -> String {
    let units: Vec<u16> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u16).collect();
    String::from_utf16_lossy(&units)
}

// --- host-side COM callback objects ------------------------------------------------
//
// These are real VST3 COM objects (built with the same `#[VST3(implements(..))]` macro
// the plugin examples use). `Box::into_raw` yields a `*mut Self` whose first member is
// the vtable-ptr-ptr, so it transmutes cleanly into a `StaticVstPtr<dyn I>` field of
// `ProcessData`. We keep the boxes alive for the whole run and free them afterwards.

/// One parameter's value queue: a single point (offset 0, normalized value).
#[VST3(implements(IParamValueQueue))]
struct ParamQueue {
    id: u32,
    value: f64,
}

impl ParamQueue {
    fn new(id: u32, value: f64) -> Box<Self> {
        Self::allocate(id, value)
    }
}

impl IParamValueQueue for ParamQueue {
    unsafe fn get_parameter_id(&self) -> u32 {
        self.id
    }
    unsafe fn get_point_count(&self) -> i32 {
        1
    }
    unsafe fn get_point(&self, index: i32, sample_offset: *mut i32, value: *mut f64) -> tresult {
        if index != 0 {
            return 1; // kResultFalse
        }
        *sample_offset = 0;
        *value = self.value;
        kResultOk
    }
    unsafe fn add_point(&self, _sample_offset: i32, _value: f64, _index: *mut i32) -> tresult {
        1 // kResultFalse — host-built, read-only
    }
}

/// The container of all input parameter queues for one process() run.
#[VST3(implements(IParameterChanges))]
struct ParamChanges {
    // Raw COM pointers to our ParamQueue objects; we own them and free them in Drop.
    queues: Vec<*mut ParamQueue>,
}

impl ParamChanges {
    fn new(params: &[(u32, f64)]) -> Box<Self> {
        let queues = params
            .iter()
            .map(|&(id, value)| Box::into_raw(ParamQueue::new(id, value)))
            .collect();
        Self::allocate(queues)
    }
}

impl IParameterChanges for ParamChanges {
    unsafe fn get_parameter_count(&self) -> i32 {
        self.queues.len() as i32
    }
    unsafe fn get_parameter_data(&self, index: i32) -> StaticVstPtr<dyn IParamValueQueue> {
        if index < 0 || index as usize >= self.queues.len() {
            return std::mem::transmute::<*mut c_void, StaticVstPtr<dyn IParamValueQueue>>(
                std::ptr::null_mut(),
            );
        }
        let ptr = self.queues[index as usize];
        std::mem::transmute::<*mut ParamQueue, StaticVstPtr<dyn IParamValueQueue>>(ptr)
    }
    unsafe fn add_parameter_data(
        &self,
        _id: *const u32,
        _index: *mut i32,
    ) -> StaticVstPtr<dyn IParamValueQueue> {
        std::mem::transmute::<*mut c_void, StaticVstPtr<dyn IParamValueQueue>>(std::ptr::null_mut())
    }
}

/// Output event list: records every emitted NoteOn pitch into `collected`.
#[VST3(implements(IEventList))]
struct OutEvents {
    collected: std::cell::RefCell<Vec<i16>>,
}

impl OutEvents {
    fn new() -> Box<Self> {
        Self::allocate(std::cell::RefCell::new(Vec::new()))
    }
}

impl IEventList for OutEvents {
    unsafe fn get_event_count(&self) -> i32 {
        self.collected.borrow().len() as i32
    }
    unsafe fn get_event(&self, _index: i32, _e: *mut Event) -> tresult {
        1 // kResultFalse — not needed by the plugin
    }
    unsafe fn add_event(&self, e: *mut Event) -> tresult {
        let ev = &*e;
        if ev.type_ == K_NOTE_ON_EVENT {
            let note: NoteOnEvent = ev.event.note_on;
            self.collected.borrow_mut().push(note.pitch);
        }
        kResultOk
    }
}

/// Type for the exported `GetPluginFactory` symbol.
type GetFactoryFn = unsafe extern "C" fn() -> *mut c_void;
/// Type for the Linux-only exported `ModuleEntry` / `ModuleExit` symbols.
type ModuleEntryFn = unsafe extern "C" fn(*mut c_void) -> bool;
type ModuleExitFn = unsafe extern "C" fn() -> bool;

/// Build a `VstPtr` from a raw interface pointer returned by `create_instance` /
/// `query_interface` (it already carries a refcount of 1, so we own it).
unsafe fn owned<I: ComInterface + ?Sized>(ptr: *mut c_void) -> Option<VstPtr<I>> {
    VstPtr::owned(ptr as *mut *mut _)
}

/// queryInterface helper on a raw COM pointer.
unsafe fn query<I: ComInterface + ?Sized>(unknown: *mut c_void) -> Option<VstPtr<I>> {
    if unknown.is_null() {
        return None;
    }
    let mut obj = std::ptr::null_mut::<c_void>();
    // The first member of any COM object is its vtable; query_interface lives at slot 0.
    let vptr = unknown as *mut *mut *mut unsafe extern "C" fn();
    let query_interface: unsafe extern "C" fn(
        *mut c_void,
        *const vst3_sys::IID,
        *mut *mut c_void,
    ) -> tresult = std::mem::transmute(*(*vptr));
    let res = query_interface(unknown, &I::IID as *const _, &mut obj);
    if res == kResultOk && !obj.is_null() {
        owned(obj)
    } else {
        None
    }
}

/// Run one full 4-beat bar for a given scale; returns the emitted NoteOn pitches.
unsafe fn run_scale(
    processor: &VstPtr<dyn IAudioProcessor>,
    scale_id: u32,
    key_id: u32,
    pitch_ids: &[u32],
    scale_norm: f64,
    key_norm: f64,
    pitch_norms: &[f64],
) -> Vec<i16> {
    // Build the param list: Scale, Key, and the 16 note pitches.
    let mut params: Vec<(u32, f64)> = vec![(scale_id, scale_norm), (key_id, key_norm)];
    for (&pid, &norm) in pitch_ids.iter().zip(pitch_norms.iter()) {
        params.push((pid, norm));
    }

    let in_changes = ParamChanges::new(&params);
    let out_events = OutEvents::new();
    // Raw pointers for the ProcessData fields; we keep the boxes alive below.
    let in_changes_ptr = Box::into_raw(in_changes);
    let out_events_ptr = Box::into_raw(out_events);

    let mut context = ProcessContext {
        state: TRANSPORT_STATE,
        sample_rate: SAMPLE_RATE,
        tempo: TEMPO,
        time_sig_num: 4,
        time_sig_den: 4,
        project_time_music: 0.0,
        ..Default::default()
    };

    let beats_per_sample = TEMPO / 60.0 / SAMPLE_RATE;
    let beats_per_block = BLOCK as f64 * beats_per_sample;

    // Helpers to (re)build the `StaticVstPtr` fields — they are not `Copy` for `dyn`,
    // so we reconstruct them from the raw pointers each iteration.
    let null_changes = || -> StaticVstPtr<dyn IParameterChanges> {
        unsafe {
            std::mem::transmute::<*mut c_void, StaticVstPtr<dyn IParameterChanges>>(
                std::ptr::null_mut(),
            )
        }
    };
    let null_events = || -> StaticVstPtr<dyn IEventList> {
        unsafe {
            std::mem::transmute::<*mut c_void, StaticVstPtr<dyn IEventList>>(std::ptr::null_mut())
        }
    };

    let mut pos_beats = 0.0_f64;
    let mut first = true;
    // One bar = 4 beats (16 steps). Mirror the CLAP harness loop exactly.
    while pos_beats < 4.0 {
        context.project_time_music = pos_beats;

        let input_param_changes = if first {
            std::mem::transmute::<*mut ParamChanges, StaticVstPtr<dyn IParameterChanges>>(
                in_changes_ptr,
            )
        } else {
            null_changes()
        };
        let output_events =
            std::mem::transmute::<*mut OutEvents, StaticVstPtr<dyn IEventList>>(out_events_ptr);

        let mut data = ProcessData {
            process_mode: 0,
            symbolic_sample_size: 0,
            num_samples: BLOCK,
            num_inputs: 0,
            num_outputs: 0,
            inputs: std::ptr::null_mut(),
            outputs: std::ptr::null_mut(),
            // Only feed the param changes on the first block (like the CLAP flush).
            input_param_changes,
            output_param_changes: null_changes(),
            input_events: null_events(),
            output_events,
            context: &mut context,
        };

        let status = processor.process(&mut data);
        assert_eq!(status, kResultOk, "process() returned non-OK: {status}");

        pos_beats += beats_per_block;
        first = false;
    }

    // Read the collected pitches, then free our COM objects.
    let collected = (*out_events_ptr).collected.borrow().clone();

    // Free the ParamQueue children, then the containers.
    let in_changes = Box::from_raw(in_changes_ptr);
    for &q in &in_changes.queues {
        drop(Box::from_raw(q));
    }
    drop(in_changes);
    drop(Box::from_raw(out_events_ptr));

    collected
}

fn main() {
    let vst3_path = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../target/bundled/deepsteps-plugin.vst3/Contents/x86_64-linux/deepsteps-plugin.so"
        )
        .to_string()
    });
    println!("Loading: {vst3_path}\n");

    unsafe {
        let lib = libloading::Library::new(&vst3_path).expect("dlopen .vst3 .so");

        // Linux: must call ModuleEntry(handle) before GetPluginFactory.
        let module_entry: libloading::Symbol<ModuleEntryFn> =
            lib.get(b"ModuleEntry\0").expect("ModuleEntry symbol");
        // nih-plug's ModuleEntry ignores its argument; pass a non-null dummy handle.
        let dummy_handle = &lib as *const _ as *mut c_void;
        assert!(module_entry(dummy_handle), "ModuleEntry returned false");

        let get_factory: libloading::Symbol<GetFactoryFn> =
            lib.get(b"GetPluginFactory\0").expect("GetPluginFactory symbol");
        let factory_raw = get_factory();
        assert!(!factory_raw.is_null(), "GetPluginFactory returned null");
        let factory: VstPtr<dyn IPluginFactory> =
            owned(factory_raw).expect("wrap factory");

        // Find the Audio Module Class CID.
        let class_count = factory.count_classes();
        let mut audio_cid = None;
        for i in 0..class_count {
            let mut info: vst3_sys::base::PClassInfo = std::mem::zeroed();
            if factory.get_class_info(i, &mut info) != kResultOk {
                continue;
            }
            let category: Vec<u8> =
                info.category.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            if String::from_utf8_lossy(&category) == "Audio Module Class" {
                audio_cid = Some(info.cid);
            }
        }
        let audio_cid = audio_cid.expect("no Audio Module Class found");

        let mut fails = 0u32;

        for (scale_index, (name, table)) in SCALES.iter().enumerate() {
            // --- create a fresh component instance per scale ---
            let mut comp_raw = std::ptr::null_mut::<c_void>();
            assert_eq!(
                factory.create_instance(
                    &audio_cid as *const _,
                    &<dyn IComponent as ComInterface>::IID as *const _,
                    &mut comp_raw,
                ),
                kResultOk,
                "create_instance(IComponent) failed"
            );
            assert!(!comp_raw.is_null(), "component is null");
            let component: VstPtr<dyn IComponent> = owned(comp_raw).expect("wrap component");

            assert_eq!(
                component.initialize(std::ptr::null_mut()),
                kResultOk,
                "component.initialize failed"
            );

            // --- controller: same object, or a separate one ---
            let controller: VstPtr<dyn IEditController> = match query(component.as_ptr()
                as *mut c_void)
            {
                Some(c) => c,
                None => {
                    let mut ctrl_cid: vst3_sys::IID = std::mem::zeroed();
                    assert_eq!(
                        component.get_controller_class_id(&mut ctrl_cid),
                        kResultOk,
                        "get_controller_class_id failed"
                    );
                    let mut ctrl_raw = std::ptr::null_mut::<c_void>();
                    assert_eq!(
                        factory.create_instance(
                            &ctrl_cid as *const _,
                            &<dyn IEditController as ComInterface>::IID as *const _,
                            &mut ctrl_raw,
                        ),
                        kResultOk,
                        "create_instance(IEditController) failed"
                    );
                    let ctrl: VstPtr<dyn IEditController> =
                        owned(ctrl_raw).expect("wrap controller");
                    assert_eq!(
                        ctrl.initialize(std::ptr::null_mut()),
                        kResultOk,
                        "controller.initialize failed"
                    );
                    ctrl
                }
            };

            // --- audio processor ---
            let processor: VstPtr<dyn IAudioProcessor> =
                query(component.as_ptr() as *mut c_void).expect("query IAudioProcessor");

            // --- enumerate params: Scale, Key, 16 Pitch ---
            let (mut scale_id, mut key_id) = (None, None);
            let mut pitch_ids: Vec<u32> = Vec::new();
            let pcount = controller.get_parameter_count();
            for i in 0..pcount {
                let mut info: vst3_sys::vst::ParameterInfo = std::mem::zeroed();
                if controller.get_parameter_info(i, &mut info) != kResultOk {
                    continue;
                }
                match read_utf16(&info.title).as_str() {
                    "Scale" => scale_id = Some(info.id),
                    "Key" => key_id = Some(info.id),
                    "Pitch" => pitch_ids.push(info.id),
                    _ => {}
                }
            }
            let scale_id = scale_id.expect("Scale param not found");
            let key_id = key_id.expect("Key param not found");
            assert_eq!(pitch_ids.len(), 16, "expected 16 note Pitch params");

            // Normalize plain values via the controller (don't hand-roll).
            let scale_norm = controller.plain_param_to_normalized(scale_id, scale_index as f64);
            let key_norm = controller.plain_param_to_normalized(key_id, 0.0);
            let pitch_norms: Vec<f64> = pitch_ids
                .iter()
                .enumerate()
                .map(|(i, &pid)| controller.plain_param_to_normalized(pid, (60 + i) as f64))
                .collect();

            // --- activate buses (audio + event, both directions, every index) ---
            for media in [MediaTypes::kAudio as i32, MediaTypes::kEvent as i32] {
                for dir in [BusDirections::kInput as i32, BusDirections::kOutput as i32] {
                    let n = component.get_bus_count(media, dir);
                    for idx in 0..n {
                        component.activate_bus(media, dir, idx, 1);
                    }
                }
            }

            // --- setup processing ---
            let setup = ProcessSetup {
                process_mode: 0,
                symbolic_sample_size: 0,
                max_samples_per_block: BLOCK,
                sample_rate: SAMPLE_RATE,
            };
            assert_eq!(
                processor.setup_processing(&setup),
                kResultOk,
                "setup_processing failed"
            );
            assert_eq!(component.set_active(1), kResultOk, "set_active failed");
            assert_eq!(processor.set_processing(1), kResultOk, "set_processing failed");

            let notes = run_scale(
                &processor,
                scale_id,
                key_id,
                &pitch_ids,
                scale_norm,
                key_norm,
                &pitch_norms,
            );

            processor.set_processing(0);
            component.set_active(0);
            component.terminate();
            // controller.terminate() only matters if it's a separate object; calling it
            // on the shared object is harmless for nih-plug. Drop releases everything.
            drop(controller);
            drop(processor);
            drop(component);

            // --- verify: every emitted pitch in-scale and equals a reference snap ---
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

        // Tear down the module.
        if let Ok(module_exit) = lib.get::<ModuleExitFn>(b"ModuleExit\0") {
            module_exit();
        }

        println!();
        if fails == 0 {
            println!("ALL 14 SCALES PASS");
        } else {
            eprintln!("{fails} scale(s) FAILED");
            std::process::exit(1);
        }
    }
}
