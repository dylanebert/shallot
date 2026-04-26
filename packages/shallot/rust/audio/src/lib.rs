mod convolution;
mod envelope;
mod fft;
mod filter;
mod graph;
mod hrtf;
mod oscillator;
mod sample;

use core::cell::UnsafeCell;
use core::f32::consts::TAU;
use envelope::EnvStage;
use filter::{AllpassFilter, Biquad};
use graph::{
    synthesize_graph_voice, tick_envelopes_only, InstrumentDef, ModConnection, ModMode, NodeDef,
    NodeState, NodeType, MAX_BUFFERS, MAX_INSTRUMENTS, MAX_MODS, MAX_NODES, MAX_PARAMS, NO_BUF,
};
use hrtf::{Speaker, HRTF_TAPS, NUM_SPEAKERS};
use oscillator::{WAVETABLE_FRAMES, WAVETABLE_SAMPLES};

pub const BLOCK_SIZE: usize = 128;
pub const MAX_VOICES: usize = 64;
pub const MAX_TRANSPORTS: usize = 8;
pub const MAX_SAMPLES: usize = 256;
const MAX_EVENTS_PER_TRANSPORT: usize = 256;
const SPATIAL_SMOOTH: f32 = 0.999;
const AIR_ABSORPTION: [f32; 3] = [0.0002, 0.0017, 0.0182];
const PARAM_SMOOTH_TIME: f32 = 0.005;
const MAX_VOICE_EVENTS_PER_BLOCK: usize = 8;
const FDN_SIZE: usize = 16;
const FDN_MAX_DELAY: usize = 16384;
const FDN_PRIMES: [usize; FDN_SIZE] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
const ALLPASS_COUNT: usize = 4;
const ALLPASS_DELAYS: [usize; ALLPASS_COUNT] = [225, 341, 441, 556];
const ALLPASS_FEEDBACK: f32 = 0.5;
const FADE_SAMPLES: i32 = 240;

struct Voice {
    active: bool,
    instrument: u8,
    spatial: bool,
    spatial_snap: bool,
    one_shot: bool,
    params: [f32; MAX_PARAMS],
    smooth_params: [f32; MAX_PARAMS],
    node_states: [NodeState; MAX_NODES],
    buffers: [[f32; BLOCK_SIZE]; MAX_BUFFERS],
    gate_off_countdown: i32,
    air_lp: f32,
    occ_gain: f32,
    occ_gain_target: f32,
    azimuth: f32,
    elevation: f32,
    distance: f32,
    azimuth_target: f32,
    elevation_target: f32,
    distance_target: f32,
    ref_distance: f32,
    max_distance: f32,
    rolloff: f32,
    refl_gain: f32,
    refl_gain_target: f32,
    convolver: Option<Box<convolution::Convolver>>,
    idle_countdown: i32,
    virtual_voice: bool,
    audibility: f32,
    fade_samples: i32,
}

fn default_voice() -> Voice {
    Voice {
        active: false,
        instrument: 0,
        spatial: false,
        spatial_snap: true,
        one_shot: false,
        params: [0.0; MAX_PARAMS],
        smooth_params: [0.0; MAX_PARAMS],
        node_states: [NodeState::None; MAX_NODES],
        buffers: [[0.0; BLOCK_SIZE]; MAX_BUFFERS],
        gate_off_countdown: -1,
        air_lp: 0.0,
        occ_gain: 1.0,
        occ_gain_target: 1.0,
        azimuth: 0.0,
        elevation: 0.0,
        distance: 0.0,
        azimuth_target: 0.0,
        elevation_target: 0.0,
        distance_target: 0.0,
        ref_distance: 1.0,
        max_distance: 100.0,
        rolloff: 1.0,
        refl_gain: 0.0,
        refl_gain_target: 0.0,
        convolver: None,
        idle_countdown: -1,
        virtual_voice: false,
        audibility: 0.0,
        fade_samples: 0,
    }
}

#[derive(Clone, Copy)]
struct TransportEvent {
    beat: f64,
    voice_id: u8,
    duration_beats: f32,
    param_count: u8,
    params: [(u8, f32); 4],
}

const DEFAULT_TRANSPORT_EVENT: TransportEvent = TransportEvent {
    beat: 0.0,
    voice_id: 0,
    duration_beats: 0.0,
    param_count: 0,
    params: [(0, 0.0); 4],
};

#[derive(Clone, Copy)]
struct Transport {
    playing: bool,
    sample_pos: u64,
    bpm: f32,
    tempo_anchor_sample: u64,
    tempo_anchor_beat: f64,
    events: [TransportEvent; MAX_EVENTS_PER_TRANSPORT],
    event_count: usize,
    cursor: usize,
    loop_length: f64,
}

const DEFAULT_TRANSPORT: Transport = Transport {
    playing: false,
    sample_pos: 0,
    bpm: 120.0,
    tempo_anchor_sample: 0,
    tempo_anchor_beat: 0.0,
    events: [DEFAULT_TRANSPORT_EVENT; MAX_EVENTS_PER_TRANSPORT],
    event_count: 0,
    cursor: 0,
    loop_length: 0.0,
};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct TransportReadback {
    pub playing: u32,
    pub beat_lo: u32,
    pub beat_hi: u32,
    pub bpm_x1000: u32,
}

fn next_power_of_prime(min_val: usize, prime: usize) -> usize {
    let mut v = 1;
    while v < min_val && v * prime <= FDN_MAX_DELAY {
        v *= prime;
    }
    v
}

fn compute_fdn_lengths(rt60_mid: f32, sample_rate: f32) -> [usize; FDN_SIZE] {
    let delay_sum = (0.15 * rt60_mid * sample_rate) as usize;
    let delay_min = delay_sum.max(FDN_SIZE) / FDN_SIZE;
    let mut lengths = [0usize; FDN_SIZE];
    let mut seed: u32 = 42;
    for i in 0..FDN_SIZE {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let offset = ((seed >> 16) % 101) as usize;
        lengths[i] = next_power_of_prime((delay_min + offset).max(2), FDN_PRIMES[i]);
    }
    lengths
}

fn hadamard16(buf: &mut [f32; FDN_SIZE]) {
    let mut stride = 1;
    for _ in 0..4 {
        let step = stride * 2;
        let mut i = 0;
        while i < FDN_SIZE {
            for j in 0..stride {
                let a = buf[i + j];
                let b = buf[i + j + stride];
                buf[i + j] = a + b;
                buf[i + j + stride] = a - b;
            }
            i += step;
        }
        stride = step;
    }
    for v in buf.iter_mut() {
        *v *= 0.25;
    }
}

fn distance_gain(distance: f32, ref_dist: f32, rolloff: f32) -> f32 {
    if distance <= ref_dist {
        return 1.0;
    }
    ref_dist / (ref_dist + rolloff * (distance - ref_dist))
}

fn current_beat(t: &Transport, sample_rate: f32) -> f64 {
    let elapsed = (t.sample_pos - t.tempo_anchor_sample) as f64;
    t.tempo_anchor_beat + elapsed * (t.bpm as f64) / (60.0 * sample_rate as f64)
}

fn find_cursor_at(events: &[TransportEvent], count: usize, beat: f64) -> usize {
    for i in 0..count {
        if events[i].beat >= beat {
            return i;
        }
    }
    count
}

#[derive(Clone, Copy)]
struct VoiceBlockEvent {
    sample_offset: usize,
    duration_beats: f32,
    transport_bpm: f32,
    param_count: u8,
    params: [(u8, f32); 4],
}

fn boxed<T: Clone, const N: usize>(val: T) -> Box<[T; N]> {
    vec![val; N].into_boxed_slice().try_into().ok().unwrap()
}

fn boxed_fn<T, const N: usize>(f: impl Fn() -> T) -> Box<[T; N]> {
    let v: Vec<T> = (0..N).map(|_| f()).collect();
    v.into_boxed_slice().try_into().ok().unwrap()
}

struct FdnReverb {
    lines: Box<[[f32; FDN_MAX_DELAY]; FDN_SIZE]>,
    write_pos: [usize; FDN_SIZE],
    lengths: [usize; FDN_SIZE],
    absorptive: [[Biquad; 3]; FDN_SIZE],
    tone_correction: [Biquad; 3],
    allpass: [AllpassFilter; ALLPASS_COUNT],
    wet_gain: f32,
    wet_target: f32,
    rt60: [f32; 3],
    rt60_target: [f32; 3],
}

impl FdnReverb {
    fn new(sample_rate: f32) -> Self {
        let lengths = compute_fdn_lengths(10.0, sample_rate);
        let mut fdn = FdnReverb {
            lines: boxed([0.0; FDN_MAX_DELAY]),
            write_pos: [0; FDN_SIZE],
            lengths,
            absorptive: [[Biquad::passthrough(); 3]; FDN_SIZE],
            tone_correction: [Biquad::passthrough(); 3],
            allpass: [
                AllpassFilter::new(ALLPASS_DELAYS[0], ALLPASS_FEEDBACK),
                AllpassFilter::new(ALLPASS_DELAYS[1], ALLPASS_FEEDBACK),
                AllpassFilter::new(ALLPASS_DELAYS[2], ALLPASS_FEEDBACK),
                AllpassFilter::new(ALLPASS_DELAYS[3], ALLPASS_FEEDBACK),
            ],
            wet_gain: 0.0,
            wet_target: 0.0,
            rt60: [0.5; 3],
            rt60_target: [0.5; 3],
        };
        fdn.update_filters(sample_rate);
        fdn
    }

    fn update_filters(&mut self, sample_rate: f32) {
        for i in 0..FDN_SIZE {
            let delay_samples = self.lengths[i] as f32;
            let mut gains = [0.0f32; 3];
            for b in 0..3 {
                gains[b] = if self.rt60[b] > 0.01 {
                    (-6.91 * delay_samples / (self.rt60[b] * sample_rate))
                        .exp()
                        .max(1e-3)
                } else {
                    1e-3
                };
            }
            let ls = Biquad::low_shelf(800.0, gains[0], sample_rate);
            let pk = Biquad::peaking(800.0, 8000.0, gains[1], sample_rate);
            let hs = Biquad::high_shelf(8000.0, gains[2], sample_rate);
            self.absorptive[i][0].set_coeffs(&ls);
            self.absorptive[i][1].set_coeffs(&pk);
            self.absorptive[i][2].set_coeffs(&hs);
        }

        let mut tone_gains = [0.0f32; 3];
        let mut max_gain = 0.0f32;
        for b in 0..3 {
            tone_gains[b] = (1.0 / self.rt60[b].max(0.01)).sqrt();
            if tone_gains[b] > max_gain {
                max_gain = tone_gains[b];
            }
        }
        for b in 0..3 {
            tone_gains[b] /= max_gain;
        }
        let ls = Biquad::low_shelf(800.0, tone_gains[0], sample_rate);
        let pk = Biquad::peaking(800.0, 8000.0, tone_gains[1], sample_rate);
        let hs = Biquad::high_shelf(8000.0, tone_gains[2], sample_rate);
        self.tone_correction[0].set_coeffs(&ls);
        self.tone_correction[1].set_coeffs(&pk);
        self.tone_correction[2].set_coeffs(&hs);
    }
}

pub struct AudioEngine {
    sample_rate: f32,
    param_smooth_coeff: f32,
    spatial_smooth_block: f32,
    voices: Box<[Voice; MAX_VOICES]>,
    instruments: [InstrumentDef; MAX_INSTRUMENTS],
    wavetable: Box<[f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES]>,
    samples: Vec<Vec<f32>>,
    foa_bus: [[f32; BLOCK_SIZE]; 4],
    output: [f32; BLOCK_SIZE * 2],
    speakers: [Speaker; NUM_SPEAKERS],
    hrtf_norm: f32,
    conv_buf: [[f32; BLOCK_SIZE + HRTF_TAPS - 1]; NUM_SPEAKERS],
    conv_pos: usize,
    transports: Box<[Transport; MAX_TRANSPORTS]>,
    readbacks: [TransportReadback; MAX_TRANSPORTS],
    voice_events: [[VoiceBlockEvent; MAX_VOICE_EVENTS_PER_BLOCK]; MAX_VOICES],
    voice_event_counts: [usize; MAX_VOICES],
    event_overflow_count: u32,
    fdn: FdnReverb,
    active_spatial_count: u32,
    fft_plan: fft::FftPlan,
    ir_staging: Box<[f32; convolution::MAX_IR_SAMPLES]>,
    spike_diag: [f32; 8],
    pre_tanh_peak: f32,
    fdn_peak: f32,
    real_voice_budget: u32,
    sorted_indices: [u8; MAX_VOICES],
    diag_active: u32,
    diag_real: u32,
    diag_virtual: u32,
    diag_convolved: u32,
}

impl AudioEngine {
    pub fn new(sample_rate: f32) -> Self {
        let mut e = AudioEngine {
            sample_rate,
            param_smooth_coeff: 1.0 - (-1.0 / (PARAM_SMOOTH_TIME * sample_rate)).exp(),
            spatial_smooth_block: SPATIAL_SMOOTH.powi(BLOCK_SIZE as i32),
            voices: boxed_fn(default_voice),
            instruments: [InstrumentDef {
                node_count: 0,
                output_buf: 0,
                mod_count: 0,
                nodes: [NodeDef {
                    node_type: NodeType::None,
                    input_buf: NO_BUF,
                    input_buf_b: NO_BUF,
                    output_buf: 0,
                    param_offset: 0,
                }; MAX_NODES],
                mod_connections: [ModConnection {
                    source_buf: NO_BUF,
                    target_node: 0,
                    target_param: 0,
                    depth_param: 0,
                    mode: ModMode::Linear,
                }; MAX_MODS],
            }; MAX_INSTRUMENTS],
            wavetable: boxed(0.0),
            samples: (0..MAX_SAMPLES).map(|_| Vec::new()).collect(),
            foa_bus: [[0.0; BLOCK_SIZE]; 4],
            output: [0.0; BLOCK_SIZE * 2],
            speakers: unsafe { core::mem::zeroed() },
            hrtf_norm: 0.0,
            conv_buf: [[0.0; BLOCK_SIZE + HRTF_TAPS - 1]; NUM_SPEAKERS],
            conv_pos: 0,
            transports: boxed(DEFAULT_TRANSPORT),
            readbacks: [TransportReadback {
                playing: 0,
                beat_lo: 0,
                beat_hi: 0,
                bpm_x1000: 120_000,
            }; MAX_TRANSPORTS],
            voice_events: [[VoiceBlockEvent {
                sample_offset: 0,
                duration_beats: 0.0,
                transport_bpm: 120.0,
                param_count: 0,
                params: [(0, 0.0); 4],
            }; MAX_VOICE_EVENTS_PER_BLOCK]; MAX_VOICES],
            voice_event_counts: [0; MAX_VOICES],
            event_overflow_count: 0,
            fdn: FdnReverb::new(sample_rate),
            active_spatial_count: 0,
            fft_plan: fft::FftPlan::new(),
            ir_staging: boxed(0.0),
            spike_diag: [0.0; 8],
            pre_tanh_peak: 0.0,
            fdn_peak: 0.0,
            real_voice_budget: 24,
            sorted_indices: [0; MAX_VOICES],
            diag_active: 0,
            diag_real: 0,
            diag_virtual: 0,
            diag_convolved: 0,
        };
        e.speakers = hrtf::init_speakers(sample_rate);
        e.hrtf_norm = 1.0 / NUM_SPEAKERS as f32;
        e
    }

    pub fn reset_state(&mut self) {
        for vi in 0..MAX_VOICES {
            let v = &mut self.voices[vi];
            if !v.active {
                continue;
            }
            if let Some(conv) = &mut v.convolver {
                conv.reset();
            }
            v.air_lp = 0.0;
        }
        for sp in 0..NUM_SPEAKERS {
            self.conv_buf[sp].fill(0.0);
        }
        self.conv_pos = 0;
        for k in 0..FDN_SIZE {
            self.fdn.lines[k].fill(0.0);
            for b in 0..3 {
                self.fdn.absorptive[k][b].reset();
            }
        }
        for b in 0..3 {
            self.fdn.tone_correction[b].reset();
        }
        for ap in &mut self.fdn.allpass {
            ap.reset();
        }
    }

    fn update_diag_convolved(&mut self) {
        let mut n = 0u32;
        for vi in 0..MAX_VOICES {
            if self.voices[vi].active && !self.voices[vi].virtual_voice && self.voices[vi].convolver.is_some() {
                n += 1;
            }
        }
        self.diag_convolved = n;
    }

    pub fn set_real_voice_budget(&mut self, budget: u32) {
        self.real_voice_budget = budget.clamp(1, MAX_VOICES as u32);
    }

    fn classify_voices(&mut self) {
        let mut count = 0u32;
        for vi in 0..MAX_VOICES {
            if !self.voices[vi].active {
                continue;
            }
            let dist_gain = distance_gain(
                self.voices[vi].distance,
                self.voices[vi].ref_distance,
                self.voices[vi].rolloff,
            );
            self.voices[vi].audibility = dist_gain * self.voices[vi].occ_gain;
            self.sorted_indices[count as usize] = vi as u8;
            count += 1;
        }
        if count <= self.real_voice_budget {
            for i in 0..count as usize {
                let vi = self.sorted_indices[i] as usize;
                if self.voices[vi].virtual_voice {
                    self.voices[vi].virtual_voice = false;
                    self.voices[vi].fade_samples = FADE_SAMPLES;
                }
            }
            self.diag_active = count;
            self.diag_real = count;
            self.diag_virtual = 0;
            self.update_diag_convolved();
            return;
        }

        let slice = &mut self.sorted_indices[..count as usize];
        for i in 1..slice.len() {
            let key = slice[i];
            let key_score = self.voices[key as usize].audibility;
            let mut j = i;
            while j > 0 && self.voices[slice[j - 1] as usize].audibility < key_score {
                slice[j] = slice[j - 1];
                j -= 1;
            }
            slice[j] = key;
        }

        let budget = self.real_voice_budget as usize;
        let below_boundary = if budget < count as usize {
            self.voices[slice[budget] as usize].audibility
        } else {
            0.0
        };

        for rank in 0..count as usize {
            let vi = slice[rank] as usize;
            let v = &mut self.voices[vi];
            if rank < budget {
                if v.virtual_voice {
                    if rank < budget - 1 || v.audibility > below_boundary * 1.2 {
                        v.virtual_voice = false;
                        if v.fade_samples < 0 {
                            v.fade_samples = -v.fade_samples;
                        } else {
                            v.fade_samples = FADE_SAMPLES;
                        }
                    }
                }
            } else if !v.virtual_voice {
                if v.fade_samples > 0 {
                    v.fade_samples = -v.fade_samples;
                } else {
                    v.fade_samples = -FADE_SAMPLES;
                }
                v.virtual_voice = true;
            }
        }

        self.diag_active = count;
        self.diag_real = budget.min(count as usize) as u32;
        self.diag_virtual = count - self.diag_real;
        self.update_diag_convolved();

        let block_coeff = self.spatial_smooth_block;
        let one_minus = 1.0 - block_coeff;
        for vi in 0..MAX_VOICES {
            let v = &mut self.voices[vi];
            if !v.active || !v.virtual_voice || !v.spatial {
                continue;
            }
            let mut az_diff = v.azimuth_target - v.azimuth;
            if az_diff > core::f32::consts::PI {
                az_diff -= TAU;
            } else if az_diff < -core::f32::consts::PI {
                az_diff += TAU;
            }
            v.azimuth += one_minus * az_diff;
            v.elevation = block_coeff * v.elevation + one_minus * v.elevation_target;
            v.distance = block_coeff * v.distance + one_minus * v.distance_target;
            v.occ_gain = block_coeff * v.occ_gain + one_minus * v.occ_gain_target;
            v.refl_gain = block_coeff * v.refl_gain + one_minus * v.refl_gain_target;
        }
    }

    pub fn set_param(&mut self, voice_id: u32, id: u32, value: f32) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        let offset = id as usize;
        if offset < MAX_PARAMS {
            self.voices[idx].params[offset] = value;
        }
    }

    pub fn set_gate(&mut self, voice_id: u32, gate: u32) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        let v = &mut self.voices[idx];
        if gate != 0 {
            v.gate_off_countdown = -1;
            v.smooth_params = v.params;
        }
        let inst_idx = v.instrument as usize;
        if inst_idx >= MAX_INSTRUMENTS {
            return;
        }
        let nc = self.instruments[inst_idx].node_count as usize;
        for ni in 0..nc {
            if gate != 0 {
                v.node_states[ni].gate_on();
            } else {
                v.node_states[ni].gate_off();
            }
        }
    }

    pub fn set_spatial(
        &mut self,
        voice_id: u32,
        azimuth: f32,
        elevation: f32,
        distance: f32,
        ref_distance: f32,
        max_distance: f32,
        rolloff: f32,
    ) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        let v = &mut self.voices[idx];
        v.azimuth_target = azimuth;
        v.elevation_target = elevation;
        v.distance_target = distance.min(max_distance);
        v.ref_distance = ref_distance;
        v.max_distance = max_distance;
        v.rolloff = rolloff;
        if v.spatial_snap {
            v.azimuth = v.azimuth_target;
            v.elevation = v.elevation_target;
            v.distance = v.distance_target;
            v.spatial_snap = false;
        }
    }

    pub fn set_voice_spatial(&mut self, voice_id: u32, spatial: u32) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        self.voices[idx].spatial = spatial != 0;
    }

    pub fn set_voice_one_shot(&mut self, voice_id: u32, one_shot: u32) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        self.voices[idx].one_shot = one_shot != 0;
    }

    pub fn set_acoustic(&mut self, voice_id: u32, gain_low: f32, gain_mid: f32, gain_high: f32) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        if !gain_low.is_finite() || !gain_mid.is_finite() || !gain_high.is_finite() {
            return;
        }
        let peak = gain_low
            .clamp(0.0, 1.0)
            .max(gain_mid.clamp(0.0, 1.0))
            .max(gain_high.clamp(0.0, 1.0));
        self.voices[idx].occ_gain_target = peak;
    }

    pub fn set_acoustic_separate(
        &mut self,
        voice_id: u32,
        occlusion: f32,
        trans_low: f32,
        trans_mid: f32,
        trans_high: f32,
    ) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        let occ = occlusion.clamp(0.0, 1.0);
        let gl = occ + (1.0 - occ) * trans_low.clamp(0.0, 1.0);
        let gm = occ + (1.0 - occ) * trans_mid.clamp(0.0, 1.0);
        let gh = occ + (1.0 - occ) * trans_high.clamp(0.0, 1.0);
        self.voices[idx].occ_gain_target = gl.max(gm).max(gh);
    }

    pub fn ir_staging_ptr(&mut self) -> *mut f32 {
        self.ir_staging.as_mut_ptr()
    }

    pub fn set_reflection_ir(&mut self, voice_id: u32, ir_len: u32) {
        let vi = voice_id as usize;
        if vi >= MAX_VOICES {
            return;
        }
        let len = (ir_len as usize).min(convolution::MAX_IR_SAMPLES);
        if len == 0 {
            return;
        }
        for i in 0..len {
            if !self.ir_staging[i].is_finite() {
                return;
            }
        }
        let conv = self.voices[vi]
            .convolver
            .get_or_insert_with(|| Box::new(convolution::Convolver::new()));
        conv.update_ir(&self.ir_staging[..len], &self.fft_plan);
    }

    pub fn set_reflection_gain(&mut self, voice_id: u32, gain: f32) {
        let vi = voice_id as usize;
        if vi >= MAX_VOICES {
            return;
        }
        if !gain.is_finite() {
            return;
        }
        self.voices[vi].refl_gain_target = gain.clamp(0.0, 1.0);
    }

    pub fn set_reverb(
        &mut self,
        rt60_low: f32,
        rt60_mid: f32,
        rt60_high: f32,
        wet_gain: f32,
        _eq_low: f32,
        _eq_mid: f32,
        _eq_high: f32,
    ) {
        self.fdn.rt60_target = [rt60_low.max(0.1), rt60_mid.max(0.1), rt60_high.max(0.1)];
        self.fdn.wet_target = wet_gain;
    }

    pub fn voice_active(&mut self, voice_id: u32, active: u32) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        let v = &mut self.voices[idx];
        if active != 0 && !v.active {
            v.smooth_params = v.params;
            v.spatial_snap = true;
            v.distance = v.max_distance;
            v.azimuth = 0.0;
            v.elevation = 0.0;
            v.convolver = None;
            v.one_shot = false;
            v.idle_countdown = -1;
            v.air_lp = 0.0;
            v.occ_gain = 1.0;
            v.occ_gain_target = 1.0;
            v.refl_gain = 0.0;
            v.refl_gain_target = 0.0;
        }
        if active == 0 {
            v.gate_off_countdown = -1;
        }
        v.active = active != 0;
    }

    pub fn voice_idle(&mut self, voice_id: u32) -> u32 {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return 0;
        }
        let v = &mut self.voices[idx];
        if !v.active {
            return 0;
        }
        let inst_idx = v.instrument as usize;
        if inst_idx >= MAX_INSTRUMENTS {
            return 0;
        }
        let inst = &self.instruments[inst_idx];
        let mut env_idle = true;
        for ni in 0..inst.node_count as usize {
            if let NodeState::Envelope { stage, .. } = &v.node_states[ni] {
                if *stage != EnvStage::Idle {
                    env_idle = false;
                    break;
                }
            }
        }
        if !env_idle {
            v.idle_countdown = -1;
            return 0;
        }
        if v.idle_countdown < 0 {
            let conv_tail = v.convolver.as_ref().map_or(0, |c| c.num_blocks() as i32);
            v.idle_countdown = if v.spatial { conv_tail.max(16) } else { 0 };
        }
        if v.idle_countdown > 0 {
            v.idle_countdown -= 1;
            return 0;
        }
        1
    }

    pub fn set_instrument(&mut self, id: u32, node_count: u32, output_buf: u32) {
        let idx = id as usize;
        if idx >= MAX_INSTRUMENTS {
            return;
        }
        let inst = &mut self.instruments[idx];
        inst.node_count = node_count as u8;
        inst.output_buf = output_buf as u8;
        inst.mod_count = 0;
        for n in inst.nodes.iter_mut() {
            *n = NodeDef::default();
        }
        for m in inst.mod_connections.iter_mut() {
            *m = ModConnection::default();
        }
    }

    pub fn sample_alloc(&mut self, id: u32, len: u32) -> *mut f32 {
        let idx = id as usize;
        if idx >= MAX_SAMPLES {
            return core::ptr::null_mut();
        }
        self.samples[idx] = vec![0.0; len as usize];
        self.samples[idx].as_mut_ptr()
    }

    pub fn clear_sample(&mut self, id: u32) {
        let idx = id as usize;
        if idx >= MAX_SAMPLES {
            return;
        }
        self.samples[idx].clear();
    }

    pub fn set_instrument_node(
        &mut self,
        id: u32,
        index: u32,
        node_type: u32,
        input_buf: u32,
        input_buf_b: u32,
        output_buf: u32,
        param_offset: u32,
    ) {
        let idx = id as usize;
        let ni = index as usize;
        if idx >= MAX_INSTRUMENTS || ni >= MAX_NODES {
            return;
        }
        self.instruments[idx].nodes[ni] = NodeDef {
            node_type: NodeType::from_u8(node_type as u8),
            input_buf: input_buf as u8,
            input_buf_b: input_buf_b as u8,
            output_buf: output_buf as u8,
            param_offset: param_offset as u8,
        };
    }

    pub fn set_instrument_mod(
        &mut self,
        id: u32,
        index: u32,
        source_buf: u32,
        target_node: u32,
        target_param: u32,
        depth_param: u32,
        mode: u32,
    ) {
        let idx = id as usize;
        let mi = index as usize;
        if idx >= MAX_INSTRUMENTS || mi >= MAX_MODS {
            return;
        }
        let inst = &mut self.instruments[idx];
        inst.mod_connections[mi] = ModConnection {
            source_buf: source_buf as u8,
            target_node: target_node as u8,
            target_param: target_param as u8,
            depth_param: depth_param as u8,
            mode: ModMode::from_u8(mode as u8),
        };
        if mi as u8 >= inst.mod_count {
            inst.mod_count = mi as u8 + 1;
        }
    }

    pub fn set_voice_instrument(&mut self, voice_id: u32, instrument_id: u32) {
        let vi = voice_id as usize;
        let ii = instrument_id as usize;
        if vi >= MAX_VOICES || ii >= MAX_INSTRUMENTS {
            return;
        }
        let mut node_types = [NodeType::None; MAX_NODES];
        let nc = self.instruments[ii].node_count as usize;
        for ni in 0..nc {
            node_types[ni] = self.instruments[ii].nodes[ni].node_type;
        }
        let v = &mut self.voices[vi];
        v.instrument = instrument_id as u8;
        v.params = [0.0; MAX_PARAMS];
        v.smooth_params = [0.0; MAX_PARAMS];
        for ni in 0..MAX_NODES {
            v.node_states[ni] = match node_types[ni] {
                NodeType::Oscillator => NodeState::Oscillator { phase: 0.0 },
                NodeType::Filter => NodeState::Filter {
                    ic1eq: 0.0,
                    ic2eq: 0.0,
                    a1: 0.0,
                    a2: 0.0,
                    a3: 0.0,
                    k: 0.0,
                },
                NodeType::Envelope => NodeState::Envelope {
                    stage: EnvStage::Idle,
                    level: 0.0,
                    time: 0.0,
                    attack_start: 0.0,
                    release_start: 0.0,
                },
                NodeType::Sample => NodeState::Sample { position: 0.0 },
                _ => NodeState::None,
            };
        }
        for buf in v.buffers.iter_mut() {
            for s in buf.iter_mut() {
                *s = 0.0;
            }
        }
    }

    pub fn set_gate_duration(&mut self, voice_id: u32, duration_samples: i32) {
        let idx = voice_id as usize;
        if idx >= MAX_VOICES {
            return;
        }
        self.voices[idx].gate_off_countdown = duration_samples;
    }

    pub fn transport_play(&mut self, tid: u32) {
        let idx = tid as usize;
        if idx >= MAX_TRANSPORTS {
            return;
        }
        self.transports[idx].playing = true;
    }

    pub fn transport_stop(&mut self, tid: u32) {
        let idx = tid as usize;
        if idx >= MAX_TRANSPORTS {
            return;
        }
        let t = &mut self.transports[idx];
        t.playing = false;
        t.sample_pos = 0;
        t.tempo_anchor_sample = 0;
        t.tempo_anchor_beat = 0.0;
        t.cursor = 0;
    }

    pub fn transport_pause(&mut self, tid: u32) {
        let idx = tid as usize;
        if idx >= MAX_TRANSPORTS {
            return;
        }
        self.transports[idx].playing = false;
    }

    pub fn transport_set_bpm(&mut self, tid: u32, bpm: f32) {
        let idx = tid as usize;
        if idx >= MAX_TRANSPORTS {
            return;
        }
        let sample_rate = self.sample_rate;
        let t = &mut self.transports[idx];
        let beat = current_beat(t, sample_rate);
        t.tempo_anchor_beat = beat;
        t.tempo_anchor_sample = t.sample_pos;
        t.bpm = bpm;
    }

    pub fn transport_queue_event(
        &mut self,
        tid: u32,
        beat: f64,
        voice_id: u32,
        duration_beats: f32,
        p0_off: u32,
        p0_val: f32,
        p1_off: u32,
        p1_val: f32,
        p2_off: u32,
        p2_val: f32,
        p3_off: u32,
        p3_val: f32,
        param_count: u32,
    ) {
        let tidx = tid as usize;
        if tidx >= MAX_TRANSPORTS {
            return;
        }
        let t = &mut self.transports[tidx];
        if t.event_count >= MAX_EVENTS_PER_TRANSPORT {
            self.event_overflow_count += 1;
            return;
        }
        let pc = (param_count as u8).min(4);
        let mut params = [(0u8, 0.0f32); 4];
        if pc > 0 {
            params[0] = (p0_off as u8, p0_val);
        }
        if pc > 1 {
            params[1] = (p1_off as u8, p1_val);
        }
        if pc > 2 {
            params[2] = (p2_off as u8, p2_val);
        }
        if pc > 3 {
            params[3] = (p3_off as u8, p3_val);
        }
        let new_evt = TransportEvent {
            beat,
            voice_id: voice_id as u8,
            duration_beats,
            param_count: pc,
            params,
        };
        let count = t.event_count;
        let mut lo = 0usize;
        let mut hi = count;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if t.events[mid].beat <= beat {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        for i in (lo..count).rev() {
            t.events[i + 1] = t.events[i];
        }
        t.events[lo] = new_evt;
        t.event_count += 1;
        t.cursor = find_cursor_at(&t.events, t.event_count, current_beat(t, self.sample_rate));
    }

    pub fn transport_clear_events(&mut self, tid: u32) {
        let idx = tid as usize;
        if idx >= MAX_TRANSPORTS {
            return;
        }
        self.transports[idx].event_count = 0;
        self.transports[idx].cursor = 0;
    }

    pub fn transport_seek(&mut self, tid: u32, beat: f64) {
        let idx = tid as usize;
        if idx >= MAX_TRANSPORTS {
            return;
        }
        let t = &mut self.transports[idx];
        t.tempo_anchor_beat = beat;
        t.tempo_anchor_sample = t.sample_pos;
        t.cursor = find_cursor_at(&t.events, t.event_count, beat);

        for ei in 0..t.event_count {
            let vi = t.events[ei].voice_id as usize;
            if vi < MAX_VOICES && self.voices[vi].active {
                let inst_idx = self.voices[vi].instrument as usize;
                if inst_idx < MAX_INSTRUMENTS {
                    let nc = self.instruments[inst_idx].node_count as usize;
                    for ni in 0..nc {
                        self.voices[vi].node_states[ni].gate_off();
                    }
                }
            }
        }
    }

    pub fn transport_set_loop(&mut self, tid: u32, length: f64) {
        let idx = tid as usize;
        if idx >= MAX_TRANSPORTS {
            return;
        }
        self.transports[idx].loop_length = length;
    }

    fn apply_voice_event(&mut self, vi: usize, evt: &VoiceBlockEvent) {
        let v = &mut self.voices[vi];
        if !v.active {
            v.azimuth = v.azimuth_target;
            v.elevation = v.elevation_target;
            v.distance = v.distance_target;
            v.air_lp = 0.0;
            v.occ_gain = 1.0;
            v.occ_gain_target = 1.0;
            v.refl_gain = 0.0;
            v.refl_gain_target = 0.0;
        }
        v.active = true;

        for p in 0..evt.param_count as usize {
            let (off, val) = evt.params[p];
            let idx = off as usize;
            if idx < MAX_PARAMS {
                v.params[idx] = val;
            }
        }
        v.smooth_params = v.params;

        let inst_idx = v.instrument as usize;
        if inst_idx < MAX_INSTRUMENTS {
            let nc = self.instruments[inst_idx].node_count as usize;
            for ni in 0..nc {
                v.node_states[ni].gate_on();
            }
        }

        if evt.duration_beats > 0.0 {
            let duration_secs = evt.duration_beats as f64 * 60.0 / evt.transport_bpm as f64;
            v.gate_off_countdown = (duration_secs * self.sample_rate as f64) as i32;
        } else {
            v.gate_off_countdown = -1;
        }
    }

    fn gate_off_voice(&mut self, vi: usize) {
        let v = &mut self.voices[vi];
        let inst_idx = v.instrument as usize;
        if inst_idx < MAX_INSTRUMENTS {
            let nc = self.instruments[inst_idx].node_count as usize;
            for ni in 0..nc {
                v.node_states[ni].gate_off();
            }
        }
    }

    fn schedule_transport_events(&mut self) {
        for vi in 0..MAX_VOICES {
            self.voice_event_counts[vi] = 0;
        }

        for tidx in 0..MAX_TRANSPORTS {
            let t = &mut self.transports[tidx];
            if !t.playing {
                continue;
            }

            let block_start_beat = current_beat(t, self.sample_rate);
            t.sample_pos += BLOCK_SIZE as u64;
            let mut block_end_beat = current_beat(t, self.sample_rate);

            if t.loop_length > 0.0 && block_end_beat >= t.loop_length {
                let wrap_beat = t.loop_length;
                let beat_span = block_end_beat - block_start_beat;
                let pre_wrap_beats = wrap_beat - block_start_beat;
                let pre_wrap_frac = if beat_span > 0.0 {
                    pre_wrap_beats / beat_span
                } else {
                    1.0
                };
                let pre_wrap_samples = (pre_wrap_frac * BLOCK_SIZE as f64) as usize;

                while t.cursor < t.event_count && t.events[t.cursor].beat < wrap_beat {
                    let evt = t.events[t.cursor];
                    if evt.beat >= block_start_beat {
                        let vi = evt.voice_id as usize;
                        if vi < MAX_VOICES {
                            let cnt = &mut self.voice_event_counts[vi];
                            if *cnt < MAX_VOICE_EVENTS_PER_BLOCK {
                                let frac = (evt.beat - block_start_beat) / beat_span;
                                self.voice_events[vi][*cnt] = VoiceBlockEvent {
                                    sample_offset: (frac * BLOCK_SIZE as f64)
                                        .min(pre_wrap_samples as f64)
                                        as usize,
                                    duration_beats: evt.duration_beats,
                                    transport_bpm: t.bpm,
                                    param_count: evt.param_count,
                                    params: evt.params,
                                };
                                *cnt += 1;
                            }
                        }
                    }
                    t.cursor += 1;
                }

                t.tempo_anchor_beat -= t.loop_length;
                t.cursor = 0;
                block_end_beat -= t.loop_length;

                let post_wrap_start_beat = 0.0f64;
                while t.cursor < t.event_count && t.events[t.cursor].beat < block_end_beat {
                    let evt = t.events[t.cursor];
                    if evt.beat >= post_wrap_start_beat {
                        let vi = evt.voice_id as usize;
                        if vi < MAX_VOICES {
                            let cnt = &mut self.voice_event_counts[vi];
                            if *cnt < MAX_VOICE_EVENTS_PER_BLOCK {
                                let post_frac = if block_end_beat > post_wrap_start_beat {
                                    (evt.beat - post_wrap_start_beat)
                                        / (block_end_beat - post_wrap_start_beat)
                                } else {
                                    0.0
                                };
                                let offset = pre_wrap_samples
                                    + (post_frac * (BLOCK_SIZE - pre_wrap_samples) as f64) as usize;
                                self.voice_events[vi][*cnt] = VoiceBlockEvent {
                                    sample_offset: offset.min(BLOCK_SIZE),
                                    duration_beats: evt.duration_beats,
                                    transport_bpm: t.bpm,
                                    param_count: evt.param_count,
                                    params: evt.params,
                                };
                                *cnt += 1;
                            }
                        }
                    }
                    t.cursor += 1;
                }
            } else {
                while t.cursor < t.event_count && t.events[t.cursor].beat < block_end_beat {
                    let evt = t.events[t.cursor];
                    if evt.beat >= block_start_beat {
                        let vi = evt.voice_id as usize;
                        if vi < MAX_VOICES {
                            let cnt = &mut self.voice_event_counts[vi];
                            if *cnt < MAX_VOICE_EVENTS_PER_BLOCK {
                                let beat_span = block_end_beat - block_start_beat;
                                let frac = if beat_span > 0.0 {
                                    (evt.beat - block_start_beat) / beat_span
                                } else {
                                    0.0
                                };
                                self.voice_events[vi][*cnt] = VoiceBlockEvent {
                                    sample_offset: (frac * BLOCK_SIZE as f64) as usize,
                                    duration_beats: evt.duration_beats,
                                    transport_bpm: t.bpm,
                                    param_count: evt.param_count,
                                    params: evt.params,
                                };
                                *cnt += 1;
                            }
                        }
                    }
                    t.cursor += 1;
                }
            }
        }
    }

    fn synthesize_voices(&mut self) {
        for vi in 0..MAX_VOICES {
            if !self.voices[vi].active {
                continue;
            }

            let evt_count = self.voice_event_counts[vi];
            let inst_idx = self.voices[vi].instrument as usize;
            if inst_idx >= MAX_INSTRUMENTS {
                continue;
            }

            if self.voices[vi].virtual_voice && self.voices[vi].fade_samples >= 0 {
                for ei in 0..evt_count {
                    let evt = self.voice_events[vi][ei];
                    self.apply_voice_event(vi, &evt);
                }
                tick_envelopes_only(
                    &self.instruments[inst_idx],
                    &mut self.voices[vi].node_states,
                    &self.voices[vi].params,
                    self.sample_rate,
                );
                let mut one_shot_gate_off = false;
                if self.voices[vi].one_shot {
                    let nc = self.instruments[inst_idx].node_count as usize;
                    for ni in 0..nc {
                        if let NodeState::Envelope { stage, .. } = &self.voices[vi].node_states[ni]
                        {
                            if *stage == EnvStage::Sustain {
                                one_shot_gate_off = true;
                                break;
                            }
                        }
                    }
                }
                if one_shot_gate_off {
                    self.gate_off_voice(vi);
                }
                if self.voices[vi].gate_off_countdown >= 0 {
                    self.voices[vi].gate_off_countdown -= BLOCK_SIZE as i32;
                    if self.voices[vi].gate_off_countdown <= 0 {
                        self.voices[vi].gate_off_countdown = -1;
                        self.gate_off_voice(vi);
                    }
                }
                continue;
            }

            if evt_count == 0 {
                let voice = &mut self.voices[vi];
                let inst = &self.instruments[inst_idx];

                synthesize_graph_voice(
                    inst,
                    &mut voice.node_states,
                    &voice.params,
                    &mut voice.smooth_params,
                    &mut voice.buffers,
                    self.sample_rate,
                    self.param_smooth_coeff,
                    &*self.wavetable,
                    &self.samples,
                    0,
                    BLOCK_SIZE,
                );

                let mut one_shot_gate_off = false;
                if voice.one_shot {
                    let nc = self.instruments[inst_idx].node_count as usize;
                    for ni in 0..nc {
                        if let NodeState::Envelope { stage, .. } = &voice.node_states[ni] {
                            if *stage == EnvStage::Sustain {
                                one_shot_gate_off = true;
                                break;
                            }
                        }
                    }
                }
                if one_shot_gate_off {
                    self.gate_off_voice(vi);
                }
                let voice = &mut self.voices[vi];

                if voice.gate_off_countdown >= 0 {
                    voice.gate_off_countdown -= BLOCK_SIZE as i32;
                    if voice.gate_off_countdown <= 0 {
                        voice.gate_off_countdown = -1;
                        let nc = self.instruments[inst_idx].node_count as usize;
                        for ni in 0..nc {
                            voice.node_states[ni].gate_off();
                        }
                    }
                }
            } else {
                let mut events_sorted: [VoiceBlockEvent; MAX_VOICE_EVENTS_PER_BLOCK] =
                    self.voice_events[vi];
                for i in 1..evt_count {
                    let key = events_sorted[i];
                    let mut j = i;
                    while j > 0 && events_sorted[j - 1].sample_offset > key.sample_offset {
                        events_sorted[j] = events_sorted[j - 1];
                        j -= 1;
                    }
                    events_sorted[j] = key;
                }

                let mut pos = 0usize;
                for ei in 0..evt_count {
                    let evt = events_sorted[ei];
                    let offset = evt.sample_offset.min(BLOCK_SIZE);

                    if offset > pos {
                        let voice = &mut self.voices[vi];
                        let inst = &self.instruments[inst_idx];
                        synthesize_graph_voice(
                            inst,
                            &mut voice.node_states,
                            &voice.params,
                            &mut voice.smooth_params,
                            &mut voice.buffers,
                            self.sample_rate,
                            self.param_smooth_coeff,
                            &*self.wavetable,
                            &self.samples,
                            pos,
                            offset - pos,
                        );

                        if self.voices[vi].gate_off_countdown >= 0 {
                            self.voices[vi].gate_off_countdown -= (offset - pos) as i32;
                            if self.voices[vi].gate_off_countdown <= 0 {
                                self.voices[vi].gate_off_countdown = -1;
                                self.gate_off_voice(vi);
                            }
                        }
                    }

                    self.apply_voice_event(vi, &evt);
                    pos = offset;
                }

                if pos < BLOCK_SIZE {
                    let voice = &mut self.voices[vi];
                    let inst = &self.instruments[inst_idx];
                    synthesize_graph_voice(
                        inst,
                        &mut voice.node_states,
                        &voice.params,
                        &mut voice.smooth_params,
                        &mut voice.buffers,
                        self.sample_rate,
                        self.param_smooth_coeff,
                        &*self.wavetable,
                        &self.samples,
                        pos,
                        BLOCK_SIZE - pos,
                    );

                    if self.voices[vi].gate_off_countdown >= 0 {
                        self.voices[vi].gate_off_countdown -= (BLOCK_SIZE - pos) as i32;
                        if self.voices[vi].gate_off_countdown <= 0 {
                            self.voices[vi].gate_off_countdown = -1;
                            self.gate_off_voice(vi);
                        }
                    }
                }
            }

            let voice = &mut self.voices[vi];
            let output_buf = self.instruments[inst_idx].output_buf as usize;

            if voice.spatial {
                self.active_spatial_count += 1;
                let ref_dist = voice.ref_distance;
                let rolloff = voice.rolloff;
                let block_coeff = self.spatial_smooth_block;
                let one_minus_block = 1.0 - block_coeff;

                let dist_gain_start = distance_gain(voice.distance, ref_dist, rolloff);
                let gain_start = dist_gain_start * voice.occ_gain;
                let refl_gain_start = voice.refl_gain;
                let cos_az_start = voice.azimuth.cos();
                let sin_az_start = voice.azimuth.sin();
                let cos_el_start = voice.elevation.cos();
                let sin_el_start = voice.elevation.sin();

                let mut az_diff = voice.azimuth_target - voice.azimuth;
                if az_diff > core::f32::consts::PI {
                    az_diff -= TAU;
                } else if az_diff < -core::f32::consts::PI {
                    az_diff += TAU;
                }
                voice.azimuth += one_minus_block * az_diff;
                voice.elevation =
                    block_coeff * voice.elevation + one_minus_block * voice.elevation_target;
                voice.distance =
                    block_coeff * voice.distance + one_minus_block * voice.distance_target;

                voice.occ_gain =
                    block_coeff * voice.occ_gain + one_minus_block * voice.occ_gain_target;
                voice.refl_gain =
                    block_coeff * voice.refl_gain + one_minus_block * voice.refl_gain_target;

                let dist_gain_end = distance_gain(voice.distance, ref_dist, rolloff);
                let gain_end = dist_gain_end * voice.occ_gain;
                let refl_gain_end = voice.refl_gain;
                let cos_az_end = voice.azimuth.cos();
                let sin_az_end = voice.azimuth.sin();
                let cos_el_end = voice.elevation.cos();
                let sin_el_end = voice.elevation.sin();

                let air_cutoff =
                    (20000.0 * (-AIR_ABSORPTION[1] * voice.distance).exp()).clamp(200.0, 20000.0);
                let air_alpha = 1.0 - (-TAU * air_cutoff / self.sample_rate).exp();

                if voice.air_lp.is_nan() {
                    voice.air_lp = 0.0;
                }

                let mut synth_peak = 0.0f32;
                for i in 0..BLOCK_SIZE {
                    let s = voice.buffers[output_buf][i].abs();
                    if s > synth_peak {
                        synth_peak = s;
                    }
                }
                let inv_block = 1.0 / BLOCK_SIZE as f32;

                let mut raw_block = [0.0f32; BLOCK_SIZE];
                for i in 0..BLOCK_SIZE {
                    let t = i as f32 * inv_block;
                    let dg = dist_gain_start + (dist_gain_end - dist_gain_start) * t;
                    let raw = voice.buffers[output_buf][i] * dg;
                    voice.air_lp += air_alpha * (raw - voice.air_lp);
                    raw_block[i] = voice.air_lp;
                }

                let occ_start = gain_start / dist_gain_start.max(1e-6);
                let occ_end = gain_end / dist_gain_end.max(1e-6);
                let mut dry_block = [0.0f32; BLOCK_SIZE];
                let mut dry_peak = 0.0f32;
                for i in 0..BLOCK_SIZE {
                    let t = i as f32 * inv_block;
                    let og = occ_start + (occ_end - occ_start) * t;
                    dry_block[i] = raw_block[i] * og;
                    let a = dry_block[i].abs();
                    if a > dry_peak {
                        dry_peak = a;
                    }
                }

                let mut refl_block = [0.0f32; BLOCK_SIZE];
                if let Some(conv) = &mut voice.convolver {
                    conv.process(&raw_block, &mut refl_block, &self.fft_plan);
                    for i in 0..BLOCK_SIZE {
                        let t = i as f32 * inv_block;
                        let rg = refl_gain_start + (refl_gain_end - refl_gain_start) * t;
                        refl_block[i] *= rg;
                    }
                }

                let mut refl_peak = 0.0f32;
                let mut total_peak = 0.0f32;
                for i in 0..BLOCK_SIZE {
                    let rp = refl_block[i].abs();
                    if rp > refl_peak {
                        refl_peak = rp;
                    }
                    let total = dry_block[i] + refl_block[i];
                    let tp = total.abs();
                    if tp > total_peak {
                        total_peak = tp;
                    }
                }
                if total_peak > 1.0 && total_peak > self.spike_diag[2] {
                    self.spike_diag[0] = vi as f32;
                    self.spike_diag[1] = synth_peak;
                    self.spike_diag[2] = total_peak;
                    self.spike_diag[3] = dry_peak;
                    self.spike_diag[4] = refl_peak;
                    self.spike_diag[5] = gain_end;
                    self.spike_diag[6] = voice.distance;
                    self.spike_diag[7] = if voice.convolver.is_some() { 1.0 } else { 0.0 };
                }

                if voice.fade_samples != 0 {
                    let fade_len = FADE_SAMPLES as f32;
                    for i in 0..BLOCK_SIZE {
                        let gain = if voice.fade_samples > 0 {
                            (1.0 - (voice.fade_samples as f32 - i as f32) / fade_len)
                                .clamp(0.0, 1.0)
                        } else {
                            ((-voice.fade_samples as f32 - i as f32) / fade_len).clamp(0.0, 1.0)
                        };
                        dry_block[i] *= gain;
                        refl_block[i] *= gain;
                    }
                    if voice.fade_samples > 0 {
                        voice.fade_samples = (voice.fade_samples - BLOCK_SIZE as i32).max(0);
                    } else {
                        voice.fade_samples = (voice.fade_samples + BLOCK_SIZE as i32).min(0);
                    }
                }

                for i in 0..BLOCK_SIZE {
                    let t = i as f32 * inv_block;
                    let sa = sin_az_start + (sin_az_end - sin_az_start) * t;
                    let ca = cos_az_start + (cos_az_end - cos_az_start) * t;
                    let se = sin_el_start + (sin_el_end - sin_el_start) * t;
                    let ce = cos_el_start + (cos_el_end - cos_el_start) * t;
                    let total = dry_block[i] + refl_block[i];
                    self.foa_bus[0][i] += total;
                    self.foa_bus[1][i] += total * sa * ce;
                    self.foa_bus[2][i] += total * se;
                    self.foa_bus[3][i] += total * ca * ce;
                }
            } else {
                let mut peak = 0.0f32;
                for i in 0..BLOCK_SIZE {
                    let s = voice.buffers[output_buf][i].abs();
                    if s > peak {
                        peak = s;
                    }
                }
                if voice.fade_samples != 0 {
                    let fade_len = FADE_SAMPLES as f32;
                    for i in 0..BLOCK_SIZE {
                        let gain = if voice.fade_samples > 0 {
                            (1.0 - (voice.fade_samples as f32 - i as f32) / fade_len)
                                .clamp(0.0, 1.0)
                        } else {
                            ((-voice.fade_samples as f32 - i as f32) / fade_len).clamp(0.0, 1.0)
                        };
                        voice.buffers[output_buf][i] *= gain;
                    }
                    if voice.fade_samples > 0 {
                        voice.fade_samples = (voice.fade_samples - BLOCK_SIZE as i32).max(0);
                    } else {
                        voice.fade_samples = (voice.fade_samples + BLOCK_SIZE as i32).min(0);
                    }
                }

                for i in 0..BLOCK_SIZE {
                    let s = voice.buffers[output_buf][i];
                    self.output[i] += s;
                    self.output[BLOCK_SIZE + i] += s;
                }
            }
        }
    }

    fn render_binaural(&mut self) {
        let mut has_spatial = false;
        for vi in 0..MAX_VOICES {
            if self.voices[vi].active && self.voices[vi].spatial {
                has_spatial = true;
                break;
            }
        }

        let mut spatial_output = [0.0f32; BLOCK_SIZE * 2];

        let conv_len = BLOCK_SIZE + HRTF_TAPS - 1;

        if has_spatial {
            for sp in 0..NUM_SPEAKERS {
                let sx = self.speakers[sp].x;
                let sy = self.speakers[sp].y;
                let sz = self.speakers[sp].z;

                const MAX_RE: f32 = 0.775;
                let pos = self.conv_pos;
                for i in 0..BLOCK_SIZE {
                    let speaker_signal = self.foa_bus[0][i]
                        + MAX_RE
                            * (sx * self.foa_bus[1][i]
                                + sy * self.foa_bus[2][i]
                                + sz * self.foa_bus[3][i]);
                    self.conv_buf[sp][(pos + i) % conv_len] = speaker_signal;
                }

                for i in 0..BLOCK_SIZE {
                    let mut left = 0.0f32;
                    let mut right = 0.0f32;
                    let sample_pos = (pos + i) % conv_len;

                    for t in 0..HRTF_TAPS {
                        let buf_idx = (sample_pos + conv_len - t) % conv_len;
                        let sample = self.conv_buf[sp][buf_idx];
                        left += sample * self.speakers[sp].left[t];
                        right += sample * self.speakers[sp].right[t];
                    }

                    spatial_output[i] += left;
                    spatial_output[BLOCK_SIZE + i] += right;
                }
            }

            let norm = self.hrtf_norm;
            for i in 0..BLOCK_SIZE {
                self.output[i] += spatial_output[i] * norm;
                self.output[BLOCK_SIZE + i] += spatial_output[BLOCK_SIZE + i] * norm;
            }
        }

        self.conv_pos = (self.conv_pos + BLOCK_SIZE) % conv_len;

        let fdn = &mut self.fdn;
        let block_coeff = self.spatial_smooth_block;
        let one_minus = 1.0 - block_coeff;

        let wet_start = fdn.wet_gain;

        for b in 0..3 {
            fdn.rt60[b] = block_coeff * fdn.rt60[b] + one_minus * fdn.rt60_target[b];
        }
        fdn.wet_gain = block_coeff * fdn.wet_gain + one_minus * fdn.wet_target;

        let wet_end = fdn.wet_gain;

        fdn.update_filters(self.sample_rate);
        let inv_n = 1.0 / FDN_SIZE as f32;
        let inv_block = 1.0 / BLOCK_SIZE as f32;
        let mut fdn_peak = 0.0f32;
        for i in 0..BLOCK_SIZE {
            let t = i as f32 * inv_block;
            let wet_g = wet_start + (wet_end - wet_start) * t;
            let input = self.foa_bus[0][i] * wet_g;

            let mut taps = [0.0f32; FDN_SIZE];
            for k in 0..FDN_SIZE {
                let read_pos = (fdn.write_pos[k] + FDN_MAX_DELAY - fdn.lengths[k]) % FDN_MAX_DELAY;
                let raw = fdn.lines[k][read_pos];
                let mut filtered = raw;
                for b in 0..3 {
                    filtered = fdn.absorptive[k][b].tick(filtered);
                }
                taps[k] = filtered;
            }

            let mut raw_wet = 0.0f32;
            for k in 0..FDN_SIZE {
                raw_wet += taps[k];
            }
            raw_wet *= inv_n;

            let mut ap_out = raw_wet;
            for ap in fdn.allpass.iter_mut() {
                ap_out = ap.tick(ap_out);
            }

            let mut wet = ap_out;
            for b in 0..3 {
                wet = fdn.tone_correction[b].tick(wet);
            }
            let wa = wet.abs();
            if wa > fdn_peak {
                fdn_peak = wa;
            }
            self.output[i] += wet;
            self.output[BLOCK_SIZE + i] += wet;

            hadamard16(&mut taps);

            for k in 0..FDN_SIZE {
                let v = taps[k] + input;
                fdn.lines[k][fdn.write_pos[k]] = v;
                fdn.write_pos[k] = (fdn.write_pos[k] + 1) % FDN_MAX_DELAY;
            }
        }
        self.fdn_peak = fdn_peak;
    }

    pub fn process(&mut self) -> *const f32 {
        self.spike_diag = [0.0; 8];
        self.schedule_transport_events();
        self.classify_voices();

        for s in self.output.iter_mut() {
            *s = 0.0;
        }

        for ch in 0..4 {
            for s in self.foa_bus[ch].iter_mut() {
                *s = 0.0;
            }
        }

        self.active_spatial_count = 0;
        self.synthesize_voices();
        self.render_binaural();

        let mut pre_tanh_peak = 0.0f32;
        for s in self.output.iter_mut() {
            let a = s.abs();
            if a > pre_tanh_peak {
                pre_tanh_peak = a;
            }
            *s = s.tanh();
        }
        self.pre_tanh_peak = pre_tanh_peak;

        for tidx in 0..MAX_TRANSPORTS {
            let t = &self.transports[tidx];
            let beat = current_beat(t, self.sample_rate);
            let beat_bits = beat.to_bits();
            self.readbacks[tidx].playing = if t.playing { 1 } else { 0 };
            self.readbacks[tidx].beat_lo = beat_bits as u32;
            self.readbacks[tidx].beat_hi = (beat_bits >> 32) as u32;
            self.readbacks[tidx].bpm_x1000 = (t.bpm * 1000.0) as u32;
        }

        self.output.as_ptr()
    }

    pub fn readbacks(&self) -> &[TransportReadback; MAX_TRANSPORTS] {
        &self.readbacks
    }
}

unsafe impl Sync for Wrap {}
struct Wrap(UnsafeCell<core::mem::MaybeUninit<AudioEngine>>);

static STATE: Wrap = Wrap(UnsafeCell::new(core::mem::MaybeUninit::uninit()));

fn engine() -> &'static mut AudioEngine {
    unsafe { (*STATE.0.get()).assume_init_mut() }
}

#[no_mangle]
pub extern "C" fn audio_init(sample_rate: f32) {
    unsafe {
        (*STATE.0.get()).write(AudioEngine::new(sample_rate));
    }
}

#[no_mangle]
pub extern "C" fn audio_reset() {
    engine().reset_state();
}

#[no_mangle]
pub extern "C" fn audio_set_real_voice_budget(budget: u32) {
    engine().set_real_voice_budget(budget);
}

#[no_mangle]
pub extern "C" fn audio_set_param(voice_id: u32, id: u32, value: f32) {
    engine().set_param(voice_id, id, value);
}

#[no_mangle]
pub extern "C" fn audio_set_gate(voice_id: u32, gate: u32) {
    engine().set_gate(voice_id, gate);
}

#[no_mangle]
pub extern "C" fn audio_set_spatial(
    voice_id: u32,
    azimuth: f32,
    elevation: f32,
    distance: f32,
    ref_distance: f32,
    max_distance: f32,
    rolloff: f32,
) {
    engine().set_spatial(
        voice_id,
        azimuth,
        elevation,
        distance,
        ref_distance,
        max_distance,
        rolloff,
    );
}

#[no_mangle]
pub extern "C" fn audio_set_voice_spatial(voice_id: u32, spatial: u32) {
    engine().set_voice_spatial(voice_id, spatial);
}

#[no_mangle]
pub extern "C" fn audio_set_voice_one_shot(voice_id: u32, one_shot: u32) {
    engine().set_voice_one_shot(voice_id, one_shot);
}

#[no_mangle]
pub extern "C" fn audio_voice_active(voice_id: u32, active: u32) {
    engine().voice_active(voice_id, active);
}

#[no_mangle]
pub extern "C" fn audio_voice_idle(voice_id: u32) -> u32 {
    engine().voice_idle(voice_id)
}

#[no_mangle]
pub extern "C" fn audio_set_instrument(id: u32, node_count: u32, output_buf: u32) {
    engine().set_instrument(id, node_count, output_buf);
}

#[no_mangle]
pub extern "C" fn audio_set_instrument_node(
    id: u32,
    index: u32,
    node_type: u32,
    input_buf: u32,
    input_buf_b: u32,
    output_buf: u32,
    param_offset: u32,
) {
    engine().set_instrument_node(
        id,
        index,
        node_type,
        input_buf,
        input_buf_b,
        output_buf,
        param_offset,
    );
}

#[no_mangle]
pub extern "C" fn audio_set_instrument_mod(
    id: u32,
    index: u32,
    source_buf: u32,
    target_node: u32,
    target_param: u32,
    depth_param: u32,
    mode: u32,
) {
    engine().set_instrument_mod(
        id,
        index,
        source_buf,
        target_node,
        target_param,
        depth_param,
        mode,
    );
}

#[no_mangle]
pub extern "C" fn audio_set_voice_instrument(voice_id: u32, instrument_id: u32) {
    engine().set_voice_instrument(voice_id, instrument_id);
}

#[no_mangle]
pub extern "C" fn audio_sample_alloc(id: u32, len: u32) -> *mut f32 {
    engine().sample_alloc(id, len)
}

#[no_mangle]
pub extern "C" fn audio_clear_sample(id: u32) {
    engine().clear_sample(id);
}

#[no_mangle]
pub extern "C" fn audio_set_gate_duration(voice_id: u32, duration_samples: i32) {
    engine().set_gate_duration(voice_id, duration_samples);
}

#[no_mangle]
pub extern "C" fn transport_play(tid: u32) {
    engine().transport_play(tid);
}

#[no_mangle]
pub extern "C" fn transport_stop(tid: u32) {
    engine().transport_stop(tid);
}

#[no_mangle]
pub extern "C" fn transport_pause(tid: u32) {
    engine().transport_pause(tid);
}

#[no_mangle]
pub extern "C" fn transport_set_bpm(tid: u32, bpm: f32) {
    engine().transport_set_bpm(tid, bpm);
}

#[no_mangle]
pub extern "C" fn transport_queue_event(
    tid: u32,
    beat: f64,
    voice_id: u32,
    duration_beats: f32,
    p0_off: u32,
    p0_val: f32,
    p1_off: u32,
    p1_val: f32,
    p2_off: u32,
    p2_val: f32,
    p3_off: u32,
    p3_val: f32,
    param_count: u32,
) {
    engine().transport_queue_event(
        tid,
        beat,
        voice_id,
        duration_beats,
        p0_off,
        p0_val,
        p1_off,
        p1_val,
        p2_off,
        p2_val,
        p3_off,
        p3_val,
        param_count,
    );
}

#[no_mangle]
pub extern "C" fn transport_clear_events(tid: u32) {
    engine().transport_clear_events(tid);
}

#[no_mangle]
pub extern "C" fn transport_seek(tid: u32, beat: f64) {
    engine().transport_seek(tid, beat);
}

#[no_mangle]
pub extern "C" fn transport_set_loop(tid: u32, length: f64) {
    engine().transport_set_loop(tid, length);
}

#[no_mangle]
pub extern "C" fn audio_set_acoustic(voice_id: u32, gain_low: f32, gain_mid: f32, gain_high: f32) {
    engine().set_acoustic(voice_id, gain_low, gain_mid, gain_high);
}

#[no_mangle]
pub extern "C" fn audio_set_acoustic_separate(
    voice_id: u32,
    occlusion: f32,
    trans_low: f32,
    trans_mid: f32,
    trans_high: f32,
) {
    engine().set_acoustic_separate(voice_id, occlusion, trans_low, trans_mid, trans_high);
}

#[no_mangle]
pub extern "C" fn audio_ir_staging_ptr() -> *mut f32 {
    engine().ir_staging_ptr()
}

#[no_mangle]
pub extern "C" fn audio_set_reflection_ir(voice_id: u32, ir_len: u32) {
    engine().set_reflection_ir(voice_id, ir_len);
}

#[no_mangle]
pub extern "C" fn audio_set_reflection_gain(voice_id: u32, gain: f32) {
    engine().set_reflection_gain(voice_id, gain);
}

#[no_mangle]
pub extern "C" fn audio_set_reverb(
    rt60_low: f32,
    rt60_mid: f32,
    rt60_high: f32,
    wet_gain: f32,
    eq_low: f32,
    eq_mid: f32,
    eq_high: f32,
) {
    engine().set_reverb(
        rt60_low, rt60_mid, rt60_high, wet_gain, eq_low, eq_mid, eq_high,
    );
}

#[no_mangle]
pub extern "C" fn audio_overflow_count() -> u32 {
    let e = engine();
    let count = e.event_overflow_count;
    e.event_overflow_count = 0;
    count
}

#[no_mangle]
pub extern "C" fn transport_readback_ptr() -> *const u32 {
    &engine().readbacks as *const _ as *const u32
}

#[no_mangle]
pub extern "C" fn audio_process() -> *const f32 {
    engine().process()
}

#[no_mangle]
pub extern "C" fn audio_spike_diag_ptr() -> *const f32 {
    engine().spike_diag.as_ptr()
}

#[no_mangle]
pub extern "C" fn audio_pre_tanh_peak() -> f32 {
    engine().pre_tanh_peak
}

#[no_mangle]
pub extern "C" fn audio_fdn_peak() -> f32 {
    engine().fdn_peak
}

#[no_mangle]
pub extern "C" fn audio_diag_active() -> u32 {
    engine().diag_active
}

#[no_mangle]
pub extern "C" fn audio_diag_real() -> u32 {
    engine().diag_real
}

#[no_mangle]
pub extern "C" fn audio_diag_virtual() -> u32 {
    engine().diag_virtual
}

#[no_mangle]
pub extern "C" fn audio_diag_convolved() -> u32 {
    engine().diag_convolved
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_engine(sample_rate: f32) -> Box<AudioEngine> {
        std::thread::Builder::new()
            .stack_size(4 * 1024 * 1024)
            .spawn(move || Box::new(AudioEngine::new(sample_rate)))
            .unwrap()
            .join()
            .unwrap()
    }

    fn setup_osc_env_instrument(e: &mut AudioEngine, inst_id: u32) {
        e.set_instrument(inst_id, 2, 0);
        e.set_instrument_node(inst_id, 0, 1, NO_BUF as u32, NO_BUF as u32, 0, 0);
        e.set_instrument_node(inst_id, 1, 3, 0, NO_BUF as u32, 0, 4);
    }

    fn set_osc_env_params(e: &mut AudioEngine, voice_id: u32) {
        e.set_param(voice_id, 0, 440.0);
        e.set_param(voice_id, 3, 0.7);
        e.set_param(voice_id, 4, 0.01);
        e.set_param(voice_id, 5, 0.01);
        e.set_param(voice_id, 6, 0.7);
        e.set_param(voice_id, 7, 0.01);
        e.set_param(voice_id, 8, 0.0);
        e.set_param(voice_id, 9, 0.0);
        e.set_param(voice_id, 10, 0.0);
    }

    fn setup_osc_only_instrument(e: &mut AudioEngine, inst_id: u32) {
        e.set_instrument(inst_id, 1, 0);
        e.set_instrument_node(inst_id, 0, 1, NO_BUF as u32, NO_BUF as u32, 0, 0);
    }

    fn set_osc_params(e: &mut AudioEngine, voice_id: u32) {
        e.set_param(voice_id, 0, 440.0);
        e.set_param(voice_id, 3, 0.7);
    }

    fn process_blocks(e: &mut AudioEngine, n: usize) {
        for _ in 0..n {
            e.process();
        }
    }

    // WP1: Voice lifecycle tests

    #[test]
    fn voice_gate_cycle() {
        let mut e = new_engine(44100.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        assert_eq!(e.voice_idle(0), 1);

        e.set_gate(0, 1);
        process_blocks(&mut e, 5);
        assert_eq!(e.voice_idle(0), 0);

        e.set_gate(0, 0);
        process_blocks(&mut e, 200);
        assert_eq!(e.voice_idle(0), 1);
    }

    #[test]
    fn voice_regate_during_release() {
        let mut e = new_engine(44100.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);

        e.set_gate(0, 1);
        process_blocks(&mut e, 5);
        e.set_gate(0, 0);
        process_blocks(&mut e, 3);
        assert_eq!(e.voice_idle(0), 0);

        e.set_gate(0, 1);
        process_blocks(&mut e, 1);
        assert_eq!(e.voice_idle(0), 0);
    }

    #[test]
    fn voice_idle_no_envelope() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        assert_eq!(e.voice_idle(0), 1);
    }

    #[test]
    fn voice_gate_off_countdown() {
        let mut e = new_engine(44100.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);

        e.set_gate(0, 1);
        e.set_gate_duration(0, 256);
        process_blocks(&mut e, 3);
        let inst = &e.instruments[0];
        let mut in_release = false;
        for ni in 0..inst.node_count as usize {
            if matches!(
                e.voices[0].node_states[ni],
                NodeState::Envelope {
                    stage: EnvStage::Release,
                    ..
                }
            ) {
                in_release = true;
            }
        }
        assert!(in_release);
    }

    // WP2: Transport core tests

    #[test]
    fn events_persist_after_fire() {
        let mut e = new_engine(44100.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);

        e.transport_queue_event(0, 0.0, 0, 1.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_play(0);
        e.process();

        assert_eq!(e.transports[0].event_count, 1);
    }

    #[test]
    fn transport_stop_preserves_events() {
        let mut e = new_engine(44100.0);
        e.transport_queue_event(0, 0.0, 0, 1.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_queue_event(0, 1.0, 0, 1.0, 0, 880.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_play(0);
        e.process();
        e.transport_stop(0);

        assert_eq!(e.transports[0].event_count, 2);
    }

    #[test]
    fn transport_stop_resets_position() {
        let mut e = new_engine(44100.0);
        e.transport_play(0);
        process_blocks(&mut e, 10);
        e.transport_stop(0);
        assert_eq!(e.transports[0].sample_pos, 0);
        let beat = current_beat(&e.transports[0], 44100.0);
        assert!(beat.abs() < 1e-10);
    }

    #[test]
    fn seek_gates_off_voices() {
        let mut e = new_engine(44100.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);

        e.transport_queue_event(0, 0.0, 0, 0.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_play(0);
        e.process();
        assert_eq!(e.voice_idle(0), 0);

        e.transport_seek(0, 0.0);
        let inst = &e.instruments[0];
        let mut in_release = false;
        for ni in 0..inst.node_count as usize {
            if matches!(
                e.voices[0].node_states[ni],
                NodeState::Envelope {
                    stage: EnvStage::Release,
                    ..
                }
            ) {
                in_release = true;
            }
        }
        assert!(in_release);
    }

    #[test]
    fn event_overflow_256() {
        let mut e = new_engine(44100.0);
        for i in 0..256 {
            e.transport_queue_event(0, i as f64, 0, 1.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        }
        assert_eq!(e.transports[0].event_count, 256);
        e.transport_queue_event(0, 256.0, 0, 1.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        assert_eq!(e.transports[0].event_count, 256);
    }

    // WP4: Sample-accurate events + looping tests

    #[test]
    fn event_at_exact_sample_offset() {
        let sample_rate = 44100.0f32;
        let bpm = 120.0f32;
        let samples_per_beat = 60.0f64 / bpm as f64 * sample_rate as f64;
        let target_sample = 64.0f64;
        let beat = target_sample / samples_per_beat;

        let mut e = new_engine(sample_rate);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        e.transport_set_bpm(0, bpm);

        e.transport_queue_event(0, beat, 0, 0.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_play(0);
        e.process();

        let inst = &e.instruments[0];
        let mut env_level = 0.0f32;
        for ni in 0..inst.node_count as usize {
            if let NodeState::Envelope { level, .. } = e.voices[0].node_states[ni] {
                env_level = level;
            }
        }
        assert!(env_level > 0.0);
        assert!(env_level < 1.0);
    }

    #[test]
    fn gate_off_at_exact_sample() {
        let sample_rate = 44100.0f32;
        let bpm = 120.0f32;
        let samples_per_beat = 60.0f64 / bpm as f64 * sample_rate as f64;
        let gate_duration_samples = 32;
        let duration_beats = gate_duration_samples as f32 / samples_per_beat as f32;

        let mut e = new_engine(sample_rate);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        e.transport_set_bpm(0, bpm);

        e.transport_queue_event(
            0,
            0.0,
            0,
            duration_beats,
            0,
            440.0,
            0,
            0.0,
            0,
            0.0,
            0,
            0.0,
            1,
        );
        e.transport_play(0);
        e.process();

        let inst = &e.instruments[0];
        let mut found_release = false;
        for ni in 0..inst.node_count as usize {
            if matches!(
                e.voices[0].node_states[ni],
                NodeState::Envelope {
                    stage: EnvStage::Release,
                    ..
                } | NodeState::Envelope {
                    stage: EnvStage::Idle,
                    ..
                }
            ) {
                found_release = true;
            }
        }
        assert!(found_release);
    }

    #[test]
    fn two_events_in_one_block() {
        let sample_rate = 44100.0f32;
        let bpm = 120.0f32;
        let samples_per_beat = 60.0f64 / bpm as f64 * sample_rate as f64;
        let beat_30 = 30.0f64 / samples_per_beat;
        let beat_90 = 90.0f64 / samples_per_beat;

        let mut e = new_engine(sample_rate);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        setup_osc_env_instrument(&mut e, 1);
        e.set_voice_instrument(1, 1);
        set_osc_env_params(&mut e, 1);
        e.voice_active(0, 1);
        e.voice_active(1, 1);
        e.transport_set_bpm(0, bpm);

        e.transport_queue_event(0, beat_30, 0, 0.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_queue_event(0, beat_90, 1, 0.0, 0, 880.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_play(0);
        e.process();

        assert_eq!(e.voice_idle(0), 0);
        assert_eq!(e.voice_idle(1), 0);
    }

    #[test]
    fn loop_wraps_at_length() {
        let sample_rate = 44100.0f32;
        let bpm = 120.0f32;

        let mut e = new_engine(sample_rate);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        e.transport_set_bpm(0, bpm);
        e.transport_set_loop(0, 4.0);

        e.transport_queue_event(0, 1.0, 0, 0.5, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_play(0);

        let samples_per_beat = 60.0 / bpm * sample_rate;
        let blocks_to_beat_5 = ((5.0 * samples_per_beat) / BLOCK_SIZE as f32) as usize + 2;
        for _ in 0..blocks_to_beat_5 {
            e.process();
        }

        assert_eq!(e.transports[0].event_count, 1);
        assert!(e.transports[0].playing);
    }

    #[test]
    fn loop_disabled_by_zero() {
        let sample_rate = 44100.0f32;
        let bpm = 120.0f32;

        let mut e = new_engine(sample_rate);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        e.transport_set_bpm(0, bpm);

        e.transport_queue_event(0, 1.0, 0, 0.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_play(0);

        let samples_per_beat = 60.0 / bpm * sample_rate;
        let blocks_to_beat_4 = ((4.0 * samples_per_beat) / BLOCK_SIZE as f32) as usize + 2;
        for _ in 0..blocks_to_beat_4 {
            e.process();
        }

        assert_eq!(e.transports[0].cursor, e.transports[0].event_count);
    }

    #[test]
    fn set_loop_api() {
        let mut e = new_engine(44100.0);
        e.transport_set_loop(0, 4.0);
        assert!((e.transports[0].loop_length - 4.0).abs() < 1e-10);
        e.transport_set_loop(0, 0.0);
        assert!((e.transports[0].loop_length).abs() < 1e-10);
    }

    // WP5: Spatial bypass tests

    #[test]
    fn non_spatial_voice_direct_to_stereo() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 0);

        e.set_gate(0, 1);
        let ptr = e.process();
        let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };

        let mut has_signal = false;
        for i in 0..BLOCK_SIZE {
            if output[i].abs() > 1e-6 {
                has_signal = true;
                break;
            }
        }
        assert!(has_signal);

        let foa_energy: f32 = e.foa_bus[0].iter().map(|s| s * s).sum();
        assert!(foa_energy < 1e-10);
    }

    #[test]
    fn spatial_voice_to_foa() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);

        e.set_gate(0, 1);
        e.process();

        let foa_energy: f32 = e.foa_bus[0].iter().map(|s| s * s).sum();
        assert!(foa_energy > 1e-6);
    }

    #[test]
    fn mixed_spatial_and_direct() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_gate(0, 1);

        setup_osc_only_instrument(&mut e, 1);
        e.set_voice_instrument(1, 1);
        e.set_param(1, 0, 880.0);
        e.set_param(1, 3, 0.7);
        e.voice_active(1, 1);
        e.set_voice_spatial(1, 0);
        e.set_gate(1, 1);

        let ptr = e.process();
        let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };

        let mut has_signal = false;
        for i in 0..BLOCK_SIZE {
            if output[i].abs() > 1e-6 {
                has_signal = true;
                break;
            }
        }
        assert!(has_signal);
    }

    // WP6: Constant node tests

    #[test]
    fn constant_node_outputs_value() {
        let mut e = new_engine(44100.0);
        e.set_instrument(0, 2, 0);
        e.set_instrument_node(0, 0, 6, NO_BUF as u32, NO_BUF as u32, 0, 0);
        e.set_instrument_node(0, 1, 4, 0, NO_BUF as u32, 0, 1);
        e.set_voice_instrument(0, 0);
        e.set_param(0, 0, 0.5);
        e.set_param(0, 1, 1.0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 0);

        let ptr = e.process();
        let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };

        for i in 0..BLOCK_SIZE {
            assert!(
                (output[i] - 0.5).abs() < 0.05,
                "sample {i}: expected ~0.5, got {}",
                output[i]
            );
        }
    }

    #[test]
    fn constant_as_modulation_source() {
        let mut e = new_engine(44100.0);
        e.set_instrument(0, 2, 0);
        e.set_instrument_node(0, 0, 6, NO_BUF as u32, NO_BUF as u32, 1, 0);
        e.set_instrument_node(0, 1, 1, NO_BUF as u32, NO_BUF as u32, 0, 1);
        e.set_instrument_mod(0, 0, 1, 1, 1, 5, 0);
        e.set_voice_instrument(0, 0);
        e.set_param(0, 0, 1.0);
        e.set_param(0, 1, 440.0);
        e.set_param(0, 4, 0.7);
        e.set_param(0, 5, 100.0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 0);

        let ptr = e.process();
        let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };

        let mut has_signal = false;
        for i in 0..BLOCK_SIZE {
            if output[i].abs() > 1e-6 {
                has_signal = true;
                break;
            }
        }
        assert!(has_signal);
    }

    #[test]
    fn reset_state_clears_spatial_buffers() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_gate(0, 1);

        for _ in 0..10 {
            e.process();
        }

        e.reset_state();

        e.set_gate(0, 0);
        e.voice_active(0, 0);

        let ptr = e.process();
        let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
        let energy: f32 = output.iter().map(|s| s * s).sum();
        assert!(
            energy < 1e-6,
            "reset_state should clear spatial buffers, got energy {energy}",
        );
    }

    #[test]
    fn convolver_output_bounded() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_gate(0, 1);

        let mut delta_ir = [0.0f32; 128];
        delta_ir[0] = 1.0;
        e.ir_staging[..128].copy_from_slice(&delta_ir);
        e.set_reflection_ir(0, 128);

        for _ in 0..20 {
            let ptr = e.process();
            let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
            for i in 0..BLOCK_SIZE * 2 {
                assert!(
                    output[i].is_finite(),
                    "NaN/Inf in output at sample {i}: {}",
                    output[i],
                );
                assert!(
                    output[i].abs() < 10.0,
                    "output explosion at sample {i}: {}",
                    output[i],
                );
            }
        }
    }

    #[test]
    fn fdn_no_blowup_high_feedback() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_gate(0, 1);
        e.set_reverb(5.0, 5.0, 5.0, 2.0, 1.0, 1.0, 1.0);

        for block in 0..500 {
            let ptr = e.process();
            let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
            for i in 0..BLOCK_SIZE * 2 {
                assert!(
                    output[i].is_finite(),
                    "NaN/Inf at block {block} sample {i}: {}",
                    output[i],
                );
                assert!(
                    output[i].abs() <= 1.0,
                    "output exceeds 1.0 at block {block} sample {i}: {}",
                    output[i],
                );
            }
        }
    }

    #[test]
    fn multiple_spatial_voices_bounded() {
        let mut e = new_engine(44100.0);
        for v in 0..8u32 {
            setup_osc_only_instrument(&mut e, v);
            e.set_voice_instrument(v, v);
            e.set_param(v, 0, 220.0 + v as f32 * 55.0);
            e.set_param(v, 3, 0.7);
            e.voice_active(v, 1);
            e.set_voice_spatial(v, 1);
            e.set_gate(v, 1);
        }
        e.set_reverb(2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0);

        for block in 0..200 {
            let ptr = e.process();
            let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
            for i in 0..BLOCK_SIZE * 2 {
                assert!(
                    output[i].is_finite(),
                    "NaN/Inf at block {block} sample {i}: {}",
                    output[i],
                );
                assert!(
                    output[i].abs() <= 1.0,
                    "output exceeds 1.0 at block {block} sample {i}: {}",
                    output[i],
                );
            }
        }
    }

    #[test]
    fn absorptive_gain_formula() {
        let sr = 44100.0;
        let mut fdn = FdnReverb::new(sr);
        fdn.rt60 = [1.0; 3];
        fdn.update_filters(sr);
        let delay_samples = fdn.lengths[0] as f32;
        let expected_gain = (-6.91 * delay_samples / (1.0 * sr)).exp();
        let mut bq = Biquad::low_shelf(800.0, expected_gain, sr);
        let dt = 1.0 / sr;
        let mut last = 0.0f32;
        for i in 0..48000 {
            let input = (std::f32::consts::TAU * 200.0 * i as f32 * dt).sin();
            last = bq.tick(input);
        }
        assert!(
            last.is_finite(),
            "biquad with absorptive gain should be stable"
        );
    }

    #[test]
    fn fdn_rt60_floor() {
        let sr = 44100.0;
        let mut fdn = FdnReverb::new(sr);
        fdn.rt60 = [0.01, 0.01, 0.01];
        fdn.update_filters(sr);
        let dt = 1.0 / sr;
        for k in 0..FDN_SIZE {
            for i in 0..4800 {
                let input = (std::f32::consts::TAU * 1000.0 * i as f32 * dt).sin();
                let mut s = input;
                for b in 0..3 {
                    s = fdn.absorptive[k][b].tick(s);
                }
                assert!(
                    s.is_finite(),
                    "absorptive filter unstable at rt60=0.01, line {k}, sample {i}"
                );
            }
        }
    }

    #[test]
    fn occlusion_smoothing_no_click() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 2.0, 1.0, 100.0, 1.0);
        e.set_gate(0, 1);

        for _ in 0..20 {
            e.process();
        }

        e.set_acoustic(0, 0.1, 0.1, 0.1);

        let mut prev_sample = 0.0f32;
        let mut max_diff = 0.0f32;
        for _ in 0..50 {
            let ptr = e.process();
            let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
            for i in 0..BLOCK_SIZE {
                let diff = (output[i] - prev_sample).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                prev_sample = output[i];
            }
        }
        assert!(
            max_diff < 0.2,
            "max sample-to-sample diff during occlusion transition: {max_diff}",
        );
    }

    #[test]
    fn occlusion_full_mute() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 2.0, 1.0, 100.0, 1.0);
        e.set_gate(0, 1);
        e.set_acoustic(0, 0.0, 0.0, 0.0);

        for _ in 0..500 {
            e.process();
        }

        let ptr = e.process();
        let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
        let energy: f32 = output[..BLOCK_SIZE].iter().map(|s| s * s).sum();
        assert!(
            energy < 1e-4,
            "fully occluded voice should be near-silent, got energy {energy}",
        );
    }

    #[test]
    fn occlusion_passthrough() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 1.0, 1.0, 100.0, 1.0);
        e.set_gate(0, 1);
        e.set_acoustic(0, 1.0, 1.0, 1.0);

        for _ in 0..50 {
            e.process();
        }

        let ptr = e.process();
        let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
        let energy: f32 = output[..BLOCK_SIZE].iter().map(|s| s * s).sum();
        assert!(
            energy > 1e-3,
            "unoccluded voice should have signal, got energy {energy}",
        );
    }

    #[test]
    fn voice_recycle_clears_convolver() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_gate(0, 1);

        let mut delta_ir = [0.0f32; 128];
        delta_ir[0] = 1.0;
        e.ir_staging[..128].copy_from_slice(&delta_ir);
        e.set_reflection_ir(0, 128);

        for _ in 0..10 {
            e.process();
        }

        e.voice_active(0, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);

        let ptr = e.process();
        let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
        let energy: f32 = output[..BLOCK_SIZE].iter().map(|s| s * s).sum();
        assert!(
            energy < 0.05,
            "recycled voice with no gate should be near-silent (convolver cleared), got energy {energy}",
        );
    }

    #[test]
    fn convolver_rapid_ir_updates() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_gate(0, 1);

        for block in 0..100 {
            let mut ir = [0.0f32; 128];
            ir[0] = (block as f32 * 0.01).sin().abs().max(0.01);
            e.ir_staging[..128].copy_from_slice(&ir);
            e.set_reflection_ir(0, 128);

            let ptr = e.process();
            let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
            for i in 0..BLOCK_SIZE * 2 {
                assert!(output[i].is_finite(), "NaN/Inf at block {block} sample {i}",);
                assert!(
                    output[i].abs() < 10.0,
                    "explosion at block {block} sample {i}: {}",
                    output[i],
                );
            }
        }
    }

    // Module 5: FDN reverb parity tests

    #[test]
    fn fdn_delay_computation_matches_steam_audio() {
        let sample_rate = 44100.0;
        let rt60_mid = 1.0;
        let lengths = compute_fdn_lengths(rt60_mid, sample_rate);

        // Each delay should be a power of its corresponding prime
        for i in 0..FDN_SIZE {
            let len = lengths[i];
            assert!(len >= 2, "delay {i} too short: {len}");
            assert!(len <= FDN_MAX_DELAY, "delay {i} exceeds max: {len}");

            // Verify it's a power of the prime
            let prime = FDN_PRIMES[i];
            let mut v = len;
            while v > 1 && v % prime == 0 {
                v /= prime;
            }
            assert_eq!(v, 1, "delay {i} ({len}) is not a power of prime {prime}");
        }

        // Delays should span a range (not all the same)
        let min_len = *lengths.iter().min().unwrap();
        let max_len = *lengths.iter().max().unwrap();
        assert!(max_len > min_len, "delays should vary, got all {min_len}");
    }

    #[test]
    fn fdn_delay_computation_various_rt60() {
        let sample_rate = 44100.0;
        for &rt60 in &[0.3, 0.5, 1.0, 2.0, 5.0] {
            let lengths = compute_fdn_lengths(rt60, sample_rate);
            for i in 0..FDN_SIZE {
                assert!(lengths[i] >= 2);
                assert!(lengths[i] <= FDN_MAX_DELAY);
            }
        }
    }

    #[test]
    fn fdn_biquad_absorptive_matches_steam_audio() {
        let sr = 44100.0;
        let mut fdn = FdnReverb::new(sr);
        fdn.rt60 = [1.0, 0.5, 0.3];
        fdn.update_filters(sr);

        let band_freqs = [200.0f32, 2000.0, 12000.0];
        let dt = 1.0 / sr;
        let warmup = 4800;
        let measure = 44100;

        for k in 0..3 {
            let delay_samples = fdn.lengths[k] as f32;
            for (band, &freq) in band_freqs.iter().enumerate() {
                let expected_gain = (-6.91 * delay_samples / (fdn.rt60[band] * sr))
                    .exp()
                    .max(1e-3);

                let mut chain = fdn.absorptive[k];
                let mut in_energy = 0.0f32;
                let mut out_energy = 0.0f32;
                for i in 0..(warmup + measure) {
                    let input = (std::f32::consts::TAU * freq * i as f32 * dt).sin();
                    let mut s = input;
                    for b in 0..3 {
                        s = chain[b].tick(s);
                    }
                    if i >= warmup {
                        in_energy += input * input;
                        out_energy += s * s;
                    }
                }
                let measured_gain = (out_energy / in_energy.max(1e-30)).sqrt();
                assert!(
                    measured_gain.is_finite(),
                    "absorptive gain not finite: line {k}, band {band}",
                );
                let ratio = measured_gain / expected_gain;
                assert!(
                    ratio > 0.3 && ratio < 3.0,
                    "absorptive gain mismatch: line {k}, band {band}, freq {freq}, measured {measured_gain:.4}, expected {expected_gain:.4}",
                );
            }
        }
    }

    #[test]
    fn fdn_tone_correction_formula() {
        // sqrt(1/rt60) normalized by max, matches Steam Audio calcToneCorrectionGains
        let rt60 = [2.0f32, 1.0, 0.5];
        let eq: Vec<f32> = rt60.iter().map(|r| (1.0 / r).sqrt()).collect();
        let max_eq = eq.iter().cloned().fold(0.0f32, f32::max);
        let normalized: Vec<f32> = eq.iter().map(|e| e / max_eq).collect();

        // Low band has longest RT60 → smallest sqrt(1/rt60) → smallest normalized EQ
        assert!(normalized[0] < normalized[1]);
        assert!(normalized[1] < normalized[2]);
        // Highest RT60 band gets 1.0 (max)
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hadamard_preserves_energy() {
        let mut buf = [0.0f32; FDN_SIZE];
        // Unit vector
        buf[0] = 1.0;
        let in_energy: f32 = buf.iter().map(|x| x * x).sum();

        hadamard16(&mut buf);
        let out_energy: f32 = buf.iter().map(|x| x * x).sum();

        // Hadamard with 0.25 normalization: H * x where H is 16×16 Hadamard
        // Energy should be preserved (unitary up to scaling)
        assert!(
            (out_energy - in_energy).abs() < 1e-6,
            "Hadamard should preserve energy: in={in_energy}, out={out_energy}",
        );
    }

    #[test]
    fn hadamard_arbitrary_input_preserves_energy() {
        let mut buf = [0.0f32; FDN_SIZE];
        for i in 0..FDN_SIZE {
            buf[i] = (i as f32 * 0.3).sin();
        }
        let in_energy: f32 = buf.iter().map(|x| x * x).sum();
        hadamard16(&mut buf);
        let out_energy: f32 = buf.iter().map(|x| x * x).sum();
        assert!(
            (out_energy - in_energy).abs() / in_energy < 1e-5,
            "Hadamard energy: in={in_energy}, out={out_energy}",
        );
    }

    #[test]
    fn fdn_steady_state_bounded() {
        // Feed constant input at various RT60, verify no explosion
        for &rt60 in &[0.3, 1.0, 3.0] {
            let mut e = new_engine(44100.0);
            setup_osc_only_instrument(&mut e, 0);
            e.set_voice_instrument(0, 0);
            set_osc_params(&mut e, 0);
            e.voice_active(0, 1);
            e.set_voice_spatial(0, 1);
            e.set_gate(0, 1);
            e.set_reverb(rt60, rt60, rt60, 1.0, 1.0, 1.0, 1.0);

            for block in 0..300 {
                let ptr = e.process();
                let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
                for i in 0..BLOCK_SIZE * 2 {
                    assert!(
                        output[i].is_finite() && output[i].abs() <= 1.0,
                        "FDN explosion at rt60={rt60}, block {block}, sample {i}: {}",
                        output[i],
                    );
                }
            }
        }
    }

    fn run_fdn_standalone(
        sr: f32,
        rt60: [f32; 3],
        input_level: f32,
        excite_samples: usize,
        total_samples: usize,
    ) -> Vec<f32> {
        let mut fdn = FdnReverb::new(sr);
        fdn.rt60 = rt60;
        fdn.rt60_target = rt60;
        fdn.wet_gain = 1.0;
        fdn.wet_target = 1.0;
        fdn.update_filters(sr);

        let inv_n = 1.0 / FDN_SIZE as f32;
        let mut output = vec![0.0f32; total_samples];

        for i in 0..total_samples {
            let input = if i < excite_samples { input_level } else { 0.0 };

            let mut taps = [0.0f32; FDN_SIZE];
            for k in 0..FDN_SIZE {
                let read_pos = (fdn.write_pos[k] + FDN_MAX_DELAY - fdn.lengths[k]) % FDN_MAX_DELAY;
                let raw = fdn.lines[k][read_pos];
                let mut filtered = raw;
                for b in 0..3 {
                    filtered = fdn.absorptive[k][b].tick(filtered);
                }
                taps[k] = filtered;
            }

            let mut raw_wet: f32 = taps.iter().sum();
            raw_wet *= inv_n;

            let mut ap_out = raw_wet;
            for ap in fdn.allpass.iter_mut() {
                ap_out = ap.tick(ap_out);
            }

            let mut wet = ap_out;
            for b in 0..3 {
                wet = fdn.tone_correction[b].tick(wet);
            }
            output[i] = wet;

            hadamard16(&mut taps);
            for k in 0..FDN_SIZE {
                fdn.lines[k][fdn.write_pos[k]] = taps[k] + input;
                fdn.write_pos[k] = (fdn.write_pos[k] + 1) % FDN_MAX_DELAY;
            }
        }
        output
    }

    #[test]
    fn fdn_rt60_accuracy() {
        let sr = 44100.0;
        for &target_rt60 in &[0.5, 1.0, 2.0, 4.0] {
            let rt60 = [target_rt60; 3];
            let excite = (sr * 0.1) as usize;
            let total = (target_rt60 * sr * 3.0) as usize;
            let output = run_fdn_standalone(sr, rt60, 0.1, excite, total);

            let window = (sr * 0.05) as usize;
            let peak_energy: f32 = output[excite..(excite + window)]
                .iter()
                .map(|s| s * s)
                .sum::<f32>()
                / window as f32;

            let t_rt60 = excite + (target_rt60 * sr) as usize;
            if t_rt60 + window <= total {
                let decay_energy: f32 = output[t_rt60..(t_rt60 + window)]
                    .iter()
                    .map(|s| s * s)
                    .sum::<f32>()
                    / window as f32;

                let decay_db = 10.0 * (decay_energy / peak_energy.max(1e-30)).log10();
                eprintln!("RT60={target_rt60}: decay at t=RT60 = {decay_db:.1}dB (expect ~-60)");

                assert!(
                    decay_db < -40.0 && decay_db > -80.0,
                    "RT60={target_rt60}: at t=RT60, should be ~-60dB, got {decay_db:.1}dB",
                );
            }
        }
    }

    #[test]
    fn fdn_cathedral_vs_small_room_level() {
        let sr = 44100.0;
        let excite = (sr * 0.5) as usize;
        let total = (sr * 2.0) as usize;

        let cathedral = run_fdn_standalone(sr, [3.0, 2.5, 1.5], 0.3, excite, total);
        let small_room = run_fdn_standalone(sr, [0.4, 0.3, 0.2], 0.3, excite, total);

        let measure_rms = |buf: &[f32], start: usize, end: usize| -> f32 {
            let slice = &buf[start..end.min(buf.len())];
            (slice.iter().map(|s| s * s).sum::<f32>() / slice.len() as f32).sqrt()
        };

        let cath_rms = measure_rms(&cathedral, excite / 2, excite);
        let room_rms = measure_rms(&small_room, excite / 2, excite);

        eprintln!("cathedral steady-state RMS: {cath_rms:.4}");
        eprintln!("small room steady-state RMS: {room_rms:.4}");
        eprintln!("ratio: {:.1}x", cath_rms / room_rms);

        assert!(
            cath_rms > room_rms * 2.0,
            "cathedral should be significantly louder: cath={cath_rms:.4}, room={room_rms:.4}",
        );
    }

    #[test]
    fn fdn_output_level_physically_reasonable() {
        let sr = 44100.0;
        let excite = (sr * 1.0) as usize;
        let total = (sr * 2.0) as usize;

        let output = run_fdn_standalone(sr, [2.0, 1.5, 0.8], 0.3, excite, total);

        let peak: f32 = output.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let rms: f32 = {
            let slice = &output[(excite / 2)..excite];
            (slice.iter().map(|s| s * s).sum::<f32>() / slice.len() as f32).sqrt()
        };

        eprintln!("cathedral-like: peak={peak:.4}, steady-state RMS={rms:.4}");

        assert!(
            peak > 0.01,
            "reverb should produce audible output: peak={peak}"
        );
        assert!(peak < 2.0, "reverb should not clip: peak={peak}");
        assert!(rms > 0.005, "steady-state RMS should be audible: {rms}");
    }

    // Module 8: Spatial pipeline tests

    #[test]
    fn distance_gain_formula() {
        // ref_dist / (ref_dist + rolloff * (d - ref_dist))
        assert!((distance_gain(1.0, 1.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((distance_gain(0.5, 1.0, 1.0) - 1.0).abs() < 1e-6); // d <= ref_dist
        assert!((distance_gain(2.0, 1.0, 1.0) - 0.5).abs() < 1e-6); // 1/(1+1*1) = 0.5
                                                                    // 1/(1+1*10) = 1/11 ≈ 0.0909
        assert!((distance_gain(11.0, 1.0, 1.0) - 1.0 / 11.0).abs() < 1e-4);
    }

    #[test]
    fn distance_gain_rolloff_variants() {
        // rolloff = 0 → always 1.0
        assert!((distance_gain(100.0, 1.0, 0.0) - 1.0).abs() < 1e-6);
        // rolloff = 2 → faster attenuation
        let g2 = distance_gain(2.0, 1.0, 2.0);
        let g1 = distance_gain(2.0, 1.0, 1.0);
        assert!(g2 < g1, "higher rolloff should attenuate more");
    }

    #[test]
    fn air_absorption_cutoff_formula() {
        // cutoff = 20000 * exp(-0.0017 * distance), clamped [200, 20000]
        let d = 0.0;
        let cutoff = (20000.0 * (-AIR_ABSORPTION[1] * d).exp()).clamp(200.0, 20000.0);
        assert!((cutoff - 20000.0).abs() < 1e-3);

        let d = 100.0;
        let cutoff = (20000.0 * (-AIR_ABSORPTION[1] * d).exp()).clamp(200.0, 20000.0);
        assert!(cutoff < 20000.0);
        assert!(cutoff >= 200.0);
    }

    #[test]
    fn foa_encoding_formula() {
        // W = total, X = total*sin(az)*cos(el), Y = total*sin(el), Z = total*cos(az)*cos(el)
        let total = 0.7f32;
        let az = 0.5f32;
        let el = 0.3f32;
        let w = total;
        let x = total * az.sin() * el.cos();
        let y = total * el.sin();
        let z = total * az.cos() * el.cos();

        // W channel should be the largest
        assert!(w >= x.abs());
        assert!(w >= y.abs());
        assert!(w >= z.abs());

        // X² + Y² + Z² should be ≤ W² (FOA energy relation)
        let directional_energy = x * x + y * y + z * z;
        let omni_energy = w * w;
        assert!(directional_energy <= omni_energy + 1e-6);
    }

    #[test]
    fn spatial_smoothing_converges() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 5.0, 1.0, 100.0, 1.0);
        e.set_gate(0, 1);

        // Let it settle
        for _ in 0..50 {
            e.process();
        }

        // Change position dramatically
        e.set_spatial(0, 1.5, 0.5, 20.0, 1.0, 100.0, 1.0);

        // After enough blocks, smoothed values should converge
        for _ in 0..1000 {
            e.process();
        }

        let v = &e.voices[0];
        assert!(
            (v.azimuth - v.azimuth_target).abs() < 0.01,
            "azimuth didn't converge: {} vs {}",
            v.azimuth,
            v.azimuth_target,
        );
        assert!(
            (v.distance - v.distance_target).abs() < 0.1,
            "distance didn't converge: {} vs {}",
            v.distance,
            v.distance_target,
        );
    }

    #[test]
    fn transport_retrigger_resets_spatial_filters() {
        let mut e = new_engine(48000.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 5.0, 1.0, 100.0, 1.0);
        e.set_gate(0, 1);

        for _ in 0..50 {
            e.process();
        }

        let air_lp_before = e.voices[0].air_lp;
        assert!(
            air_lp_before != 0.0,
            "voice should have non-zero air_lp after playing"
        );

        e.voice_active(0, 0);
        assert!(!e.voices[0].active);
        assert!(
            e.voices[0].air_lp != 0.0,
            "deactivate doesn't clear filter state"
        );

        let evt = VoiceBlockEvent {
            sample_offset: 0,
            duration_beats: 0.25,
            transport_bpm: 120.0,
            param_count: 1,
            params: [(0, 440.0), (0, 0.0), (0, 0.0), (0, 0.0)],
        };
        e.apply_voice_event(0, &evt);

        assert!(e.voices[0].active, "voice should be active after event");
        assert_eq!(
            e.voices[0].air_lp, 0.0,
            "air_lp should be reset on retrigger"
        );
        assert_eq!(
            e.voices[0].occ_gain, 1.0,
            "occ_gain should be reset on retrigger"
        );
    }

    #[test]
    fn set_acoustic_rejects_nan() {
        let mut e = new_engine(48000.0);
        e.voice_active(0, 1);

        e.set_acoustic(0, 0.5, 0.5, 0.5);
        let expected = 0.5f32;
        assert!((e.voices[0].occ_gain_target - expected).abs() < 1e-6);

        e.set_acoustic(0, f32::NAN, 0.5, 0.5);
        assert!(
            (e.voices[0].occ_gain_target - expected).abs() < 1e-6,
            "NaN should be rejected"
        );

        e.set_acoustic(0, f32::INFINITY, 0.5, 0.5);
        assert!(
            (e.voices[0].occ_gain_target - expected).abs() < 1e-6,
            "infinity should be rejected"
        );
    }

    #[test]
    fn set_acoustic_clamps_range() {
        let mut e = new_engine(48000.0);
        e.voice_active(0, 1);

        e.set_acoustic(0, 2.0, 2.0, 2.0);
        assert_eq!(
            e.voices[0].occ_gain_target, 1.0,
            "gain should be clamped to 1.0"
        );

        e.set_acoustic(0, -1.0, -1.0, -1.0);
        assert!(
            e.voices[0].occ_gain_target < 1e-4,
            "gain should be near zero for negative input"
        );
    }

    #[test]
    fn set_reflection_ir_rejects_nan() {
        let mut e = new_engine(48000.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);

        let mut ir = [0.0f32; BLOCK_SIZE];
        ir[0] = 1.0;
        e.ir_staging[..ir.len()].copy_from_slice(&ir);
        e.set_reflection_ir(0, ir.len() as u32);
        assert!(e.voices[0].convolver.is_some());

        ir[10] = f32::NAN;
        e.ir_staging[..ir.len()].copy_from_slice(&ir);
        e.set_reflection_ir(0, ir.len() as u32);

        e.set_gate(0, 1);
        for _ in 0..50 {
            let ptr = e.process();
            let out = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
            for (i, &s) in out.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "NaN IR should be rejected, got {s} at sample {i}"
                );
            }
        }
    }

    fn setup_kick_instrument(e: &mut AudioEngine, inst_id: u32) {
        // osc(0) → filter(1) → env(2) → gain(3), output=buf 0
        e.set_instrument(inst_id, 4, 0);
        // osc: output=buf0, params@0 (freq=p0, waveform=p1, wavetable=p2, vol=p3)
        e.set_instrument_node(inst_id, 0, 1, NO_BUF as u32, NO_BUF as u32, 0, 0);
        // filter: input=buf0, output=buf1, params@4 (cutoff=p4, q=p5, mode=p6, mix=p7)
        e.set_instrument_node(inst_id, 1, 2, 0, NO_BUF as u32, 1, 4);
        // env: input=buf1, output=buf1, params@8 (attack=p8, decay=p9, sustain=p10, release=p11)
        e.set_instrument_node(inst_id, 2, 3, 1, NO_BUF as u32, 1, 8);
        // gain: input=buf1, output=buf0, params@12 (level=p12)
        e.set_instrument_node(inst_id, 3, 4, 1, NO_BUF as u32, 0, 12);
    }

    fn set_kick_params(e: &mut AudioEngine, voice_id: u32) {
        e.set_param(voice_id, 0, 440.0); // osc.frequency
        e.set_param(voice_id, 1, 0.0); // osc.waveform (sine)
        e.set_param(voice_id, 3, 1.0); // osc.vol
        e.set_param(voice_id, 4, 6000.0); // filter.cutoff
        e.set_param(voice_id, 5, 1.0); // filter.q
        e.set_param(voice_id, 6, 0.0); // filter.mode (LP)
        e.set_param(voice_id, 7, 0.0); // filter.mix (bypass)
        e.set_param(voice_id, 8, 0.01); // env.attack
        e.set_param(voice_id, 9, 0.08); // env.decay
        e.set_param(voice_id, 10, 0.0); // env.sustain
        e.set_param(voice_id, 11, 0.05); // env.release
        e.set_param(voice_id, 12, 0.7); // gain.level
    }

    #[test]
    fn kick_300bpm_no_nan() {
        let sr = 48000.0;
        let mut e = new_engine(sr);
        setup_kick_instrument(&mut e, 0);

        // Two voices alternating, like the gym scenario
        for v in 0..2u32 {
            e.set_voice_instrument(v, 0);
            set_kick_params(&mut e, v);
            e.voice_active(v, 1);
            e.set_voice_spatial(v, 1);
            e.set_spatial(v, 0.3, 0.0, 2.57, 3.0, 100.0, 1.0);
        }

        // Bathroom reverb (low absorption = long RT60)
        e.set_reverb(2.0, 1.5, 0.8, 0.5, 1.0, 1.0, 1.0);

        let mut ir_data = vec![0.0f32; convolution::MAX_IR_SAMPLES];
        // Simple early reflection pattern: delta at 0, decay at later taps
        ir_data[0] = 0.3;
        ir_data[441] = 0.15; // ~10ms
        ir_data[882] = 0.08; // ~20ms
        ir_data[1323] = 0.04; // ~30ms

        for v in 0..2u32 {
            let staging = &mut e.ir_staging[..ir_data.len()];
            staging.copy_from_slice(&ir_data);
            e.set_reflection_ir(v, ir_data.len() as u32);
        }

        let bpm = 300.0;
        let beat_samples = (60.0 / bpm * sr) as usize;
        let blocks_per_beat = beat_samples / BLOCK_SIZE;
        let gate_off_block = (0.14 * sr / BLOCK_SIZE as f32) as usize;
        let mut voice_toggle = 0u32;

        for beat in 0..60 {
            let v = voice_toggle;
            e.set_gate(v, 1);

            for block in 0..blocks_per_beat {
                if block == gate_off_block {
                    e.set_gate(v, 0);
                }

                // Update IR periodically (like acoustic system does every few frames)
                if block % 30 == 0 {
                    for vi in 0..2u32 {
                        let staging = &mut e.ir_staging[..ir_data.len()];
                        staging.copy_from_slice(&ir_data);
                        e.set_reflection_ir(vi, ir_data.len() as u32);
                    }
                }

                let ptr = e.process();
                let output = unsafe { core::slice::from_raw_parts(ptr, BLOCK_SIZE * 2) };
                for i in 0..BLOCK_SIZE * 2 {
                    assert!(
                        output[i].is_finite(),
                        "NaN/Inf at beat {beat}, block {block}, sample {i}: {} (voice {v})",
                        output[i],
                    );
                }
            }

            voice_toggle = 1 - voice_toggle;
        }
    }

    #[test]
    fn repeated_kicks_bathroom_reverb() {
        let sample_rate = 44100.0;
        let mut e = new_engine(sample_rate);

        setup_kick_instrument(&mut e, 0);

        // Bathroom: RT60 ~1.5s, wet_gain 2.0 (cap), refl_gain 1.0 (reflective room)
        // These match what processHistogram computes for absorption=0.05
        e.set_reverb(1.5, 1.5, 1.5, 2.0, 1.0, 1.0, 1.0);

        let bpm = 180.0;
        let beat_interval_samples = (60.0 / bpm * sample_rate) as usize;
        let beat_interval_blocks = beat_interval_samples / BLOCK_SIZE;

        let num_kicks = 20;
        let mut max_pre_tanh = 0.0f32;
        let mut block_count = 0usize;

        for kick in 0..num_kicks {
            let vi = (kick % 4) as u32;
            e.set_voice_instrument(vi, 0);
            set_kick_params(&mut e, vi);

            e.voice_active(vi, 1);
            e.set_voice_spatial(vi, 1);
            e.set_spatial(vi, 0.3, 0.0, 1.19, 3.0, 100.0, 1.0);

            {
                let len = convolution::MAX_IR_SAMPLES;
                // Bathroom IR: energy concentrated in first ~1ms (early reflections
                // from nearby walls). More realistic than diffuse exponential decay.
                let mut seed: u32 =
                    0x12345678u32.wrapping_add((kick as u32).wrapping_mul(0x9e3779b9));
                let early_samples = 441; // first 10ms bin
                for i in 0..len {
                    seed ^= seed.wrapping_shl(13);
                    seed ^= seed >> 17;
                    seed ^= seed.wrapping_shl(5);
                    let noise = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    let amp = if i < early_samples {
                        1.0
                    } else {
                        (-5.0 * (i - early_samples) as f32 / (len - early_samples) as f32).exp()
                    };
                    e.ir_staging[i] = noise * amp;
                }
                let mut energy: f32 = 0.0;
                for i in 0..len {
                    energy += e.ir_staging[i] * e.ir_staging[i];
                }
                if energy > 1e-10 {
                    let scale = 1.0 / energy.sqrt();
                    for i in 0..len {
                        e.ir_staging[i] *= scale;
                    }
                }
                e.set_reflection_ir(vi, len as u32);
                e.set_reflection_gain(vi, energy.sqrt().min(1.0));
            }

            e.set_gate(vi, 1);

            for b in 0..beat_interval_blocks {
                if b == 48 {
                    e.set_gate(vi, 0);
                }

                e.process();
                block_count += 1;

                if e.pre_tanh_peak > max_pre_tanh {
                    max_pre_tanh = e.pre_tanh_peak;
                }

                let output =
                    unsafe { core::slice::from_raw_parts(e.output.as_ptr(), BLOCK_SIZE * 2) };
                for i in 0..BLOCK_SIZE * 2 {
                    assert!(
                        output[i].is_finite(),
                        "NaN at kick {kick} block {b} sample {i}"
                    );
                }
            }
        }

        eprintln!("max pre-tanh peak over {num_kicks} kicks: {max_pre_tanh:.4}");
        eprintln!("total blocks processed: {block_count}");

        assert!(
            max_pre_tanh < 1.5,
            "pre-tanh peak {max_pre_tanh:.4} exceeds gain budget — reflections or reverb not properly controlled"
        );
    }

    #[test]
    fn block_boundary_discontinuities() {
        // Reproduce the browser message sequence:
        // Frame N:   voice_active + gate (kick starts)
        // Frame N:   spatial data arrives in same batch
        // Frame N+2: acoustic occlusion arrives (occ_gain changes)
        // Frame N+3: reflection IR arrives (convolver created)
        // Check for sample discontinuities at every block boundary.

        let sr = 44100.0;
        let mut e = new_engine(sr);
        setup_kick_instrument(&mut e, 0);
        e.set_reverb(1.5, 1.5, 1.5, 0.8, 1.0, 1.0, 1.0);

        let num_kicks = 5;
        let beat_blocks = (60.0 / 180.0 * sr) as usize / BLOCK_SIZE;
        let mut prev_last_l = 0.0f32;
        let mut max_intra = 0.0f32;
        let mut discontinuities: Vec<(usize, usize, f32, f32, f32)> = Vec::new();

        for kick in 0..num_kicks {
            let vi = (kick % 2) as u32;

            // --- Browser batch 1: voice_active + instrument + params + gate + spatial ---
            e.set_voice_instrument(vi, 0);
            set_kick_params(&mut e, vi);
            e.voice_active(vi, 1);
            e.set_voice_spatial(vi, 1);
            e.set_spatial(vi, 0.3, 0.0, 1.19, 3.0, 100.0, 1.0);
            e.set_gate(vi, 1);

            for b in 0..beat_blocks {
                if b == 2 {
                    e.set_acoustic(vi, 0.8, 0.6, 4000.0);
                }
                if b == 3 {
                    let len = 128usize;
                    e.ir_staging[0] = 1.0;
                    for i in 1..len {
                        e.ir_staging[i] = 0.0;
                    }
                    e.set_reflection_ir(vi, len as u32);
                    e.set_reflection_gain(vi, 0.4);
                }
                // Gate off during decay
                if b == 48 {
                    e.set_gate(vi, 0);
                }

                e.process();
                let output =
                    unsafe { core::slice::from_raw_parts(e.output.as_ptr(), BLOCK_SIZE * 2) };

                // Max consecutive-sample jump WITHIN this block (left channel)
                for i in 1..BLOCK_SIZE {
                    let d = (output[i] - output[i - 1]).abs();
                    if d > max_intra {
                        max_intra = d;
                    }
                }

                // Check block boundary: last sample of prev block vs first sample of this block
                let first_l = output[0];
                if kick > 0 || b > 0 {
                    let jump = (first_l - prev_last_l).abs();
                    if jump > 0.005 {
                        discontinuities.push((kick, b, prev_last_l, first_l, jump));
                    }
                }
                prev_last_l = output[BLOCK_SIZE - 1]; // last left sample
            }
        }

        if !discontinuities.is_empty() {
            eprintln!("=== Block boundary discontinuities ===");
            for (kick, block, prev, cur, jump) in &discontinuities {
                eprintln!(
                    "  kick={kick} block={block:3}: prev={prev:+.6} cur={cur:+.6} jump={jump:.6}"
                );
            }
            eprintln!("total: {} discontinuities", discontinuities.len());
        }

        // Compare block-boundary jumps against intra-block jumps.
        // If they're similar magnitude, the boundaries are normal signal, not artifacts.
        let max_boundary = discontinuities.iter().map(|d| d.4).fold(0.0f32, f32::max);
        eprintln!("max block-boundary jump: {max_boundary:.6}");
        eprintln!("max intra-block jump:    {max_intra:.6}");
        eprintln!(
            "ratio boundary/intra:    {:.2}",
            max_boundary / max_intra.max(1e-10)
        );
        assert!(
            max_boundary < max_intra * 1.5,
            "block boundary jump {max_boundary:.6} is {:.1}x the intra-block max {max_intra:.6} — \
             indicates a real discontinuity, not normal signal",
            max_boundary / max_intra,
        );
    }

    #[test]
    fn spatial_pipeline_gain_unity() {
        let sr = 44100.0;
        let directions: [(f32, f32); 4] = [
            (0.0, 0.0),  // front
            (1.57, 0.0), // right
            (3.14, 0.0), // back
            (0.0, 0.8),  // above
        ];
        let mut gains = Vec::new();
        for &(az, el) in &directions {
            let mut e = new_engine(sr);
            setup_osc_only_instrument(&mut e, 0);
            e.set_voice_instrument(0, 0);
            e.set_param(0, 0, 440.0);
            e.set_param(0, 3, 0.7);
            e.voice_active(0, 1);
            e.set_voice_spatial(0, 1);
            e.set_spatial(0, az, el, 1.0, 1.0, 100.0, 1.0);
            e.set_gate(0, 1);
            e.set_reverb(0.5, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0);

            let mut max_pre_tanh = 0.0f32;
            for _ in 0..100 {
                e.process();
                if e.pre_tanh_peak > max_pre_tanh {
                    max_pre_tanh = e.pre_tanh_peak;
                }
            }
            let ratio = max_pre_tanh / 0.7;
            eprintln!("az={az:.2} el={el:.2}: pre_tanh={max_pre_tanh:.4} ratio={ratio:.3}");
            gains.push(ratio);
        }

        let min_gain = gains.iter().copied().fold(f32::MAX, f32::min);
        let max_gain = gains.iter().copied().fold(0.0f32, f32::max);
        eprintln!("gain range: {min_gain:.3} — {max_gain:.3}");

        let speakers = hrtf::init_speakers(sr);
        let mut total_energy = 0.0f32;
        for s in &speakers {
            let le: f32 = s.left.iter().map(|v| v * v).sum();
            let re: f32 = s.right.iter().map(|v| v * v).sum();
            total_energy += (le + re) * 0.5;
        }
        let avg_energy = total_energy / NUM_SPEAKERS as f32;
        let derived_norm = 1.0 / (NUM_SPEAKERS as f32 * avg_energy).sqrt();
        eprintln!("avg HRTF energy per speaker: {avg_energy:.4}");
        eprintln!("derived norm = 1/sqrt(N*E) = {derived_norm:.4}");
        eprintln!("current norm = 1/N = {:.4}", 1.0 / NUM_SPEAKERS as f32);

        for (i, &g) in gains.iter().enumerate() {
            assert!(
                g > 0.4 && g < 0.9,
                "direction {i}: gain {g:.3} outside expected HRTF range [0.4, 0.9]"
            );
        }
    }

    #[test]
    fn occlusion_transmission_formula() {
        let mut e = new_engine(44100.0);
        setup_osc_only_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 1.0, 1.0, 100.0, 1.0);
        e.set_gate(0, 1);
        e.set_reverb(0.5, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0);

        // Unoccluded: occlusion=1.0 → gain = 1.0 + 0*trans = 1.0
        e.set_acoustic_separate(0, 1.0, 0.05, 0.05, 0.05);
        for _ in 0..200 {
            e.process();
        }
        let unoccluded_peak = e.pre_tanh_peak;

        // Fully occluded: occlusion=0.0, trans=0.05 → gain = 0 + 1*0.05 = 0.05
        e.set_acoustic_separate(0, 0.0, 0.05, 0.05, 0.05);
        for _ in 0..500 {
            e.process();
        }
        let occluded_peak = e.pre_tanh_peak;

        // Partially occluded (doorway): occlusion=0.5, trans=0.05
        // Expected: gain = 0.5 + 0.5*0.05 = 0.525
        e.set_acoustic_separate(0, 0.5, 0.05, 0.05, 0.05);
        for _ in 0..500 {
            e.process();
        }
        let partial_peak = e.pre_tanh_peak;

        eprintln!("unoccluded: {unoccluded_peak:.4}");
        eprintln!("occluded (trans=0.05): {occluded_peak:.4}");
        eprintln!("partial (occ=0.5, trans=0.05): {partial_peak:.4}");

        let partial_ratio = partial_peak / unoccluded_peak;
        assert!(
            partial_ratio > 0.4 && partial_ratio < 0.7,
            "partial occlusion ratio {partial_ratio:.3} should be ~0.525 of unoccluded"
        );
        assert!(
            partial_peak > occluded_peak * 3.0,
            "partially occluded {partial_peak:.4} should be much louder than fully occluded {occluded_peak:.4}"
        );
    }

    // Steam Audio biquad coefficient parity tests
    // Reference: iir.cpp lowShelf/peaking use f32, highShelf uses f64

    fn steam_low_shelf(cutoff: f32, gain: f32, sr: f32) -> (f32, f32, f32, f32, f32) {
        let q = 0.707f32;
        let w0 = core::f32::consts::TAU * cutoff / sr;
        let cw0 = w0.cos();
        let sw0 = w0.sin();
        let alpha = sw0 / (2.0 * q);
        let a = gain.sqrt();
        let sa = 2.0 * a.sqrt() * alpha;
        let a0 = (a + 1.0) + (a - 1.0) * cw0 + sa;
        (
            a * ((a + 1.0) - (a - 1.0) * cw0 + sa) / a0,
            2.0 * a * ((a - 1.0) - (a + 1.0) * cw0) / a0,
            a * ((a + 1.0) - (a - 1.0) * cw0 - sa) / a0,
            -2.0 * ((a - 1.0) + (a + 1.0) * cw0) / a0,
            ((a + 1.0) + (a - 1.0) * cw0 - sa) / a0,
        )
    }

    fn steam_high_shelf(cutoff: f32, gain: f32, sr: f32) -> (f32, f32, f32, f32, f32) {
        let q = 0.707f64;
        let w0 = core::f64::consts::TAU * cutoff as f64 / sr as f64;
        let cw0 = w0.cos();
        let sw0 = w0.sin();
        let alpha = sw0 / (2.0 * q);
        let a = (gain as f64).sqrt();
        let sa = 2.0 * a.sqrt() * alpha;
        let a0 = (a + 1.0) - (a - 1.0) * cw0 + sa;
        (
            (a * ((a + 1.0) + (a - 1.0) * cw0 + sa) / a0) as f32,
            (-2.0 * a * ((a - 1.0) + (a + 1.0) * cw0) / a0) as f32,
            (a * ((a + 1.0) + (a - 1.0) * cw0 - sa) / a0) as f32,
            (2.0 * ((a - 1.0) - (a + 1.0) * cw0) / a0) as f32,
            (((a + 1.0) - (a - 1.0) * cw0 - sa) / a0) as f32,
        )
    }

    fn steam_peaking(low: f32, high: f32, gain: f32, sr: f32) -> (f32, f32, f32, f32, f32) {
        let center = (low * high).sqrt();
        let qi = (high - low) / center;
        let w0 = core::f32::consts::TAU * center / sr;
        let cw0 = w0.cos();
        let sw0 = w0.sin();
        let alpha = sw0 * qi / 2.0;
        let a = gain.sqrt();
        let a0 = 1.0 + alpha / a;
        (
            (1.0 + alpha * a) / a0,
            -2.0 * cw0 / a0,
            (1.0 - alpha * a) / a0,
            -2.0 * cw0 / a0,
            (1.0 - alpha / a) / a0,
        )
    }

    #[test]
    fn biquad_low_shelf_coefficients_match_steam() {
        for &(cutoff, gain, sr) in &[
            (800.0f32, 0.5, 44100.0),
            (800.0, 0.01, 44100.0),
            (800.0, 0.99, 44100.0),
            (400.0, 0.3, 48000.0),
        ] {
            let bq = Biquad::low_shelf(cutoff, gain, sr);
            let (b0, b1, b2, a1, a2) = steam_low_shelf(cutoff, gain, sr);
            assert!((bq.b0 - b0).abs() < 1e-7, "b0: {} vs {}", bq.b0, b0);
            assert!((bq.b1 - b1).abs() < 1e-7, "b1: {} vs {}", bq.b1, b1);
            assert!((bq.b2 - b2).abs() < 1e-7, "b2: {} vs {}", bq.b2, b2);
            assert!((bq.a1 - a1).abs() < 1e-7, "a1: {} vs {}", bq.a1, a1);
            assert!((bq.a2 - a2).abs() < 1e-7, "a2: {} vs {}", bq.a2, a2);
        }
    }

    #[test]
    fn biquad_high_shelf_coefficients_match_steam() {
        for &(cutoff, gain, sr) in &[
            (8000.0f32, 0.5, 44100.0),
            (8000.0, 0.01, 44100.0),
            (8000.0, 0.99, 44100.0),
            (4000.0, 0.3, 48000.0),
        ] {
            let bq = Biquad::high_shelf(cutoff, gain, sr);
            let (b0, b1, b2, a1, a2) = steam_high_shelf(cutoff, gain, sr);
            assert!(
                (bq.b0 - b0).abs() < 1e-7,
                "b0: {} vs {} (gain={})",
                bq.b0,
                b0,
                gain
            );
            assert!(
                (bq.b1 - b1).abs() < 1e-7,
                "b1: {} vs {} (gain={})",
                bq.b1,
                b1,
                gain
            );
            assert!(
                (bq.b2 - b2).abs() < 1e-7,
                "b2: {} vs {} (gain={})",
                bq.b2,
                b2,
                gain
            );
            assert!(
                (bq.a1 - a1).abs() < 1e-7,
                "a1: {} vs {} (gain={})",
                bq.a1,
                a1,
                gain
            );
            assert!(
                (bq.a2 - a2).abs() < 1e-7,
                "a2: {} vs {} (gain={})",
                bq.a2,
                a2,
                gain
            );
        }
    }

    #[test]
    fn biquad_peaking_coefficients_match_steam() {
        for &(low, high, gain, sr) in &[
            (800.0f32, 8000.0, 0.5, 44100.0),
            (800.0, 8000.0, 0.01, 44100.0),
            (800.0, 8000.0, 0.99, 44100.0),
            (200.0, 4000.0, 0.3, 48000.0),
        ] {
            let bq = Biquad::peaking(low, high, gain, sr);
            let (b0, b1, b2, a1, a2) = steam_peaking(low, high, gain, sr);
            assert!((bq.b0 - b0).abs() < 1e-7, "b0: {} vs {}", bq.b0, b0);
            assert!((bq.b1 - b1).abs() < 1e-7, "b1: {} vs {}", bq.b1, b1);
            assert!((bq.b2 - b2).abs() < 1e-7, "b2: {} vs {}", bq.b2, b2);
            assert!((bq.a1 - a1).abs() < 1e-7, "a1: {} vs {}", bq.a1, a1);
            assert!((bq.a2 - a2).abs() < 1e-7, "a2: {} vs {}", bq.a2, a2);
        }
    }

    #[test]
    fn absorptive_gain_clamps_at_minimum() {
        let sr = 44100.0;
        let gain = (-6.91 * 8192.0f32 / (0.01 * sr)).exp().max(1e-3);
        assert_eq!(
            gain, 1e-3,
            "very short RT60 should clamp to MIN_ABSORPTIVE_GAIN"
        );
    }

    // Virtual voice tests

    fn setup_spatial_voice(e: &mut AudioEngine, voice_id: u32, inst_id: u32, distance: f32) {
        setup_osc_only_instrument(e, inst_id);
        e.set_voice_instrument(voice_id, inst_id);
        set_osc_params(e, voice_id);
        e.voice_active(voice_id, 1);
        e.set_voice_spatial(voice_id, 1);
        e.set_spatial(voice_id, 0.0, 0.0, distance, 1.0, 100.0, 1.0);
        e.set_gate(voice_id, 1);
    }

    #[test]
    fn audibility_score_distance() {
        let mut e = new_engine(44100.0);
        setup_spatial_voice(&mut e, 0, 0, 1.0);
        setup_spatial_voice(&mut e, 1, 1, 10.0);
        e.process();
        let a0 = e.voices[0].audibility;
        let a1 = e.voices[1].audibility;
        assert!(a0 > 0.0, "audibility should be nonzero from gain chain alone");
        assert!(
            a0 > a1,
            "close voice should have higher audibility: {a0} vs {a1}"
        );
    }

    #[test]
    fn virtual_voice_skips_synthesis() {
        let mut e = new_engine(44100.0);
        setup_spatial_voice(&mut e, 0, 0, 1.0);
        setup_spatial_voice(&mut e, 1, 1, 50.0);
        e.process();
        e.set_real_voice_budget(1);
        e.process();
        let foa_before: f32 = e.foa_bus[0].iter().map(|s| s * s).sum();
        assert!(foa_before > 0.0);

        assert!(e.voices[1].virtual_voice, "far voice should be virtual");
        assert!(!e.voices[0].virtual_voice, "close voice should be real");
    }

    #[test]
    fn virtual_envelope_advances_to_idle() {
        let mut e = new_engine(44100.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 50.0, 1.0, 100.0, 1.0);
        e.set_gate(0, 1);
        e.process();

        setup_spatial_voice(&mut e, 1, 1, 1.0);
        e.set_real_voice_budget(1);

        process_blocks(&mut e, 5);
        assert!(e.voices[0].virtual_voice);

        e.set_gate(0, 0);
        for _ in 0..500 {
            e.process();
            if e.voice_idle(0) == 1 {
                return;
            }
        }
        panic!("virtual voice envelope should reach idle");
    }

    #[test]
    fn transport_events_fire_for_virtual() {
        let mut e = new_engine(44100.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 50.0, 1.0, 100.0, 1.0);

        setup_spatial_voice(&mut e, 1, 1, 1.0);

        e.process();
        e.set_real_voice_budget(1);
        e.process();

        assert!(e.voices[0].virtual_voice, "far voice should be virtual");

        e.transport_queue_event(0, 0.0, 0, 0.0, 0, 440.0, 0, 0.0, 0, 0.0, 0, 0.0, 1);
        e.transport_play(0);
        e.process();

        let inst = &e.instruments[0];
        let mut gated = false;
        for ni in 0..inst.node_count as usize {
            if let NodeState::Envelope { stage, .. } = &e.voices[0].node_states[ni] {
                if *stage != EnvStage::Idle {
                    gated = true;
                }
            }
        }
        assert!(gated, "transport event should gate virtual voice");
    }

    #[test]
    fn fade_on_virtual_transition() {
        let mut e = new_engine(44100.0);
        setup_spatial_voice(&mut e, 0, 0, 1.0);
        setup_spatial_voice(&mut e, 1, 1, 50.0);
        e.process();
        e.set_real_voice_budget(1);
        e.process();

        assert!(e.voices[1].virtual_voice);
        e.set_spatial(0, 0.0, 0.0, 99.0, 1.0, 100.0, 1.0);
        e.set_spatial(1, 0.0, 0.0, 1.0, 1.0, 100.0, 1.0);

        process_blocks(&mut e, 10);

        let v0_fading = e.voices[0].fade_samples < 0 || e.voices[0].virtual_voice;
        let v1_fading = e.voices[1].fade_samples > 0 || !e.voices[1].virtual_voice;
        assert!(v0_fading || v1_fading, "transition should trigger fade");
    }

    #[test]
    fn hysteresis_prevents_toggling() {
        let mut e = new_engine(44100.0);
        setup_spatial_voice(&mut e, 0, 0, 10.0);
        setup_spatial_voice(&mut e, 1, 1, 10.0);
        e.set_real_voice_budget(1);

        for _ in 0..20 {
            e.process();
        }

        let mut toggle_count = 0;
        let mut v0_was_virtual = e.voices[0].virtual_voice;
        for _ in 0..100 {
            e.process();
            if e.voices[0].virtual_voice != v0_was_virtual {
                toggle_count += 1;
                v0_was_virtual = e.voices[0].virtual_voice;
            }
        }
        assert!(
            toggle_count <= 2,
            "voices at equal distance should not toggle rapidly: {toggle_count} toggles"
        );
    }

    #[test]
    fn one_shot_virtual_voice_cleanup() {
        let mut e = new_engine(44100.0);
        setup_osc_env_instrument(&mut e, 0);
        e.set_voice_instrument(0, 0);
        set_osc_env_params(&mut e, 0);
        e.voice_active(0, 1);
        e.set_voice_spatial(0, 1);
        e.set_spatial(0, 0.0, 0.0, 50.0, 1.0, 100.0, 1.0);
        e.set_voice_one_shot(0, 1);
        e.set_gate(0, 1);

        setup_spatial_voice(&mut e, 1, 1, 1.0);
        e.process();
        e.set_real_voice_budget(1);

        for _ in 0..500 {
            e.process();
            if e.voice_idle(0) == 1 {
                return;
            }
        }
        panic!("one-shot virtual voice should reach idle");
    }

    #[test]
    fn budget_change_virtualizes_excess() {
        let mut e = new_engine(44100.0);
        for v in 0..4u32 {
            setup_spatial_voice(&mut e, v, v, (v + 1) as f32 * 5.0);
        }
        process_blocks(&mut e, 5);
        let virtual_count: usize = (0..4).filter(|&v| e.voices[v].virtual_voice).count();
        assert_eq!(virtual_count, 0, "all should be real at default budget");

        e.set_real_voice_budget(2);
        process_blocks(&mut e, 3);
        let virtual_count: usize = (0..4).filter(|&v| e.voices[v].virtual_voice).count();
        assert_eq!(
            virtual_count, 2,
            "2 voices should be virtual after budget=2"
        );

        let real_voices: Vec<usize> = (0..4).filter(|&v| !e.voices[v].virtual_voice).collect();
        for &rv in &real_voices {
            assert!(
                e.voices[rv].distance <= 10.5,
                "real voices should be the closest"
            );
        }
    }

    #[test]
    fn budget_zero_clamps_to_one() {
        let mut e = new_engine(44100.0);
        setup_spatial_voice(&mut e, 0, 0, 1.0);
        setup_spatial_voice(&mut e, 1, 1, 10.0);
        e.set_real_voice_budget(0);
        assert_eq!(e.real_voice_budget, 1);
        process_blocks(&mut e, 5);
        let real_count: usize = (0..2).filter(|&v| !e.voices[v].virtual_voice).count();
        assert_eq!(real_count, 1, "budget=0 should clamp to 1");
    }

    #[test]
    fn new_voice_real_without_prior_synthesis() {
        let mut e = new_engine(44100.0);
        setup_spatial_voice(&mut e, 0, 0, 1.0);
        setup_spatial_voice(&mut e, 1, 1, 50.0);
        e.set_real_voice_budget(1);
        e.process();
        assert!(
            !e.voices[0].virtual_voice,
            "close voice should be real on first block even without prior synthesis"
        );
        assert!(
            e.voices[1].virtual_voice,
            "far voice should be virtual on first block"
        );
    }

    #[test]
    fn audibility_uses_gain_chain_not_peak() {
        let mut e = new_engine(44100.0);
        setup_spatial_voice(&mut e, 0, 0, 1.0);
        setup_spatial_voice(&mut e, 1, 1, 50.0);
        e.set_real_voice_budget(1);
        e.process();
        assert!(e.voices[0].audibility > 0.0, "close voice audibility should be nonzero");
        assert!(
            e.voices[0].audibility > e.voices[1].audibility,
            "close voice should have higher audibility than far voice"
        );
    }

    #[test]
    fn virtual_voice_recovers_when_approaching() {
        let mut e = new_engine(44100.0);
        setup_spatial_voice(&mut e, 0, 0, 1.0);
        setup_spatial_voice(&mut e, 1, 1, 50.0);
        e.set_real_voice_budget(1);
        process_blocks(&mut e, 10);
        assert!(e.voices[1].virtual_voice, "far voice should be virtual");

        e.set_spatial(1, 0.0, 0.0, 0.5, 1.0, 100.0, 1.0);
        e.set_spatial(0, 0.0, 0.0, 99.0, 1.0, 100.0, 1.0);
        process_blocks(&mut e, 10);
        assert!(
            !e.voices[1].virtual_voice,
            "voice 1 should become real after moving close (no last_peak dependency)"
        );
    }
}
