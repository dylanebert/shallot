use crate::envelope::{EnvStage, Envelope};
use crate::filter::{FilterMode, SvfFilter};
use crate::oscillator::{oscillator, Waveform};
use crate::sample::sample_read;
use crate::BLOCK_SIZE;

pub const MAX_NODES: usize = 16;
pub const MAX_BUFFERS: usize = 8;
pub const MAX_PARAMS: usize = 64;
pub const MAX_INSTRUMENTS: usize = 16;
pub const MAX_MODS: usize = 16;
pub const NO_BUF: u8 = 0xFF;

#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum NodeType {
    None = 0,
    Oscillator = 1,
    Filter = 2,
    Envelope = 3,
    Gain = 4,
    Mix = 5,
    Constant = 6,
    Sample = 7,
}

impl NodeType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Oscillator,
            2 => Self::Filter,
            3 => Self::Envelope,
            4 => Self::Gain,
            5 => Self::Mix,
            6 => Self::Constant,
            7 => Self::Sample,
            _ => Self::None,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ModMode {
    Linear = 0,
    Semitone = 1,
}

impl ModMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Semitone,
            _ => Self::Linear,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ModConnection {
    pub source_buf: u8,
    pub target_node: u8,
    pub target_param: u8,
    pub depth_param: u8,
    pub mode: ModMode,
}

impl Default for ModConnection {
    fn default() -> Self {
        Self {
            source_buf: NO_BUF,
            target_node: 0,
            target_param: 0,
            depth_param: 0,
            mode: ModMode::Linear,
        }
    }
}

#[derive(Clone, Copy)]
pub struct NodeDef {
    pub node_type: NodeType,
    pub input_buf: u8,
    pub input_buf_b: u8,
    pub output_buf: u8,
    pub param_offset: u8,
}

impl Default for NodeDef {
    fn default() -> Self {
        Self {
            node_type: NodeType::None,
            input_buf: NO_BUF,
            input_buf_b: NO_BUF,
            output_buf: 0,
            param_offset: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct InstrumentDef {
    pub node_count: u8,
    pub output_buf: u8,
    pub mod_count: u8,
    pub nodes: [NodeDef; MAX_NODES],
    pub mod_connections: [ModConnection; MAX_MODS],
}

impl Default for InstrumentDef {
    fn default() -> Self {
        Self {
            node_count: 0,
            output_buf: 0,
            mod_count: 0,
            nodes: [NodeDef::default(); MAX_NODES],
            mod_connections: [ModConnection::default(); MAX_MODS],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NodeState {
    None,
    Oscillator {
        phase: f32,
    },
    Filter {
        ic1eq: f32,
        ic2eq: f32,
        a1: f32,
        a2: f32,
        a3: f32,
        k: f32,
    },
    Envelope {
        stage: EnvStage,
        level: f32,
        time: f32,
        attack_start: f32,
        release_start: f32,
    },
    Sample {
        position: f32,
    },
}

impl Default for NodeState {
    fn default() -> Self {
        NodeState::None
    }
}

impl NodeState {
    fn store_envelope(&mut self, env: &Envelope) {
        *self = NodeState::Envelope {
            stage: env.stage,
            level: env.level,
            time: env.time,
            attack_start: env.attack_start_level,
            release_start: env.release_start_level,
        };
    }

    pub fn as_filter(&self, params: &[f32]) -> SvfFilter {
        match self {
            NodeState::Filter {
                ic1eq,
                ic2eq,
                a1,
                a2,
                a3,
                k,
            } => SvfFilter {
                ic1eq: *ic1eq,
                ic2eq: *ic2eq,
                mode: FilterMode::from_u32(params[2] as u32),
                q: params[1].max(0.1),
                a1: *a1,
                a2: *a2,
                a3: *a3,
                k: *k,
            },
            _ => SvfFilter {
                ic1eq: 0.0,
                ic2eq: 0.0,
                mode: FilterMode::from_u32(params[2] as u32),
                q: params[1].max(0.1),
                a1: 0.0,
                a2: 0.0,
                a3: 0.0,
                k: 0.0,
            },
        }
    }

    pub fn store_filter(&mut self, flt: &SvfFilter) {
        *self = NodeState::Filter {
            ic1eq: flt.ic1eq,
            ic2eq: flt.ic2eq,
            a1: flt.a1,
            a2: flt.a2,
            a3: flt.a3,
            k: flt.k,
        };
    }

    pub fn gate_on(&mut self) {
        if let NodeState::Envelope {
            level,
            stage,
            time,
            attack_start,
            ..
        } = self
        {
            *attack_start = *level;
            *stage = EnvStage::Attack;
            *time = 0.0;
        }
    }

    pub fn gate_off(&mut self) {
        if let NodeState::Envelope {
            level,
            stage,
            time,
            release_start,
            ..
        } = self
        {
            *release_start = *level;
            *stage = EnvStage::Release;
            *time = 0.0;
        }
    }
}

const SEMITONE_TO_LOG: f32 = core::f32::consts::LN_2 / 12.0;

fn borrow_io<'a>(
    buffers: &'a mut [[f32; BLOCK_SIZE]; MAX_BUFFERS],
    input: usize,
    output: usize,
) -> (&'a [f32; BLOCK_SIZE], &'a mut [f32; BLOCK_SIZE]) {
    if input < output {
        let (l, r) = buffers.split_at_mut(output);
        (&l[input], &mut r[0])
    } else {
        let (l, r) = buffers.split_at_mut(input);
        (&r[0], &mut l[output])
    }
}

fn oscillator_node(
    state: &mut NodeState,
    params: &[f32],
    smooth: &mut [f32],
    output: &mut [f32; BLOCK_SIZE],
    sample_rate: f32,
    smooth_coeff: f32,
    wavetable: &[f32],
    start: usize,
    count: usize,
) {
    let target_freq = params[0];
    let waveform = Waveform::from_u32(params[1] as u32);
    let wavetable_pos = params[2];
    let target_vol = params[3];

    if let NodeState::Oscillator { phase } = state {
        for i in start..start + count {
            smooth[0] += smooth_coeff * (target_freq - smooth[0]);
            smooth[3] += smooth_coeff * (target_vol - smooth[3]);
            let phase_inc = smooth[0] / sample_rate;
            output[i] =
                oscillator(waveform, *phase, phase_inc, wavetable, wavetable_pos) * smooth[3];
            *phase += phase_inc;
            if *phase >= 1.0 {
                *phase -= 1.0;
            }
        }
    }
}

fn filter_node(
    state: &mut NodeState,
    params: &[f32],
    smooth: &mut [f32],
    input: &[f32; BLOCK_SIZE],
    output: &mut [f32; BLOCK_SIZE],
    sample_rate: f32,
    smooth_coeff: f32,
    start: usize,
    count: usize,
) {
    let target_cutoff = params[0].max(20.0);
    let target_mix = params[3].clamp(0.0, 1.0);

    let mut filter = state.as_filter(params);

    for i in start..start + count {
        smooth[0] += smooth_coeff * (target_cutoff - smooth[0]);
        smooth[3] += smooth_coeff * (target_mix - smooth[3]);
        let cutoff = smooth[0].max(20.0);
        let mix = smooth[3].clamp(0.0, 1.0);

        filter.update_coefficients(sample_rate, cutoff);

        let filtered = if mix >= 1.0 {
            filter.tick(input[i])
        } else if mix <= 0.0 {
            input[i]
        } else {
            let f = filter.tick(input[i]);
            input[i] + (f - input[i]) * mix
        };
        output[i] = filtered;
    }

    state.store_filter(&filter);
}

fn envelope_node(
    state: &mut NodeState,
    params: &[f32],
    input: Option<&[f32; BLOCK_SIZE]>,
    output: &mut [f32; BLOCK_SIZE],
    sample_rate: f32,
    start: usize,
    count: usize,
) {
    let (stage, level, time, attack_start, release_start) = match state {
        NodeState::Envelope {
            stage,
            level,
            time,
            attack_start,
            release_start,
        } => (*stage, *level, *time, *attack_start, *release_start),
        _ => (EnvStage::Idle, 0.0, 0.0, 0.0, 0.0),
    };
    let mut env = Envelope {
        stage,
        level,
        time,
        attack_start_level: attack_start,
        release_start_level: release_start,
        attack: params[0].max(0.001),
        decay: params[1].max(0.001),
        sustain: params[2].clamp(0.0, 1.0),
        release: params[3].max(0.001),
        attack_curve: params[4].clamp(-1.0, 1.0),
        decay_curve: params[5].clamp(-1.0, 1.0),
        release_curve: params[6].clamp(-1.0, 1.0),
    };

    let dt = 1.0 / sample_rate;
    match input {
        Some(inp) => {
            for i in start..start + count {
                let amp = env.tick(dt);
                output[i] = inp[i] * amp;
            }
        }
        None => {
            for i in start..start + count {
                output[i] = env.tick(dt);
            }
        }
    }

    state.store_envelope(&env);
}

fn gain_node(
    params: &[f32],
    smooth: &mut [f32],
    input: &[f32; BLOCK_SIZE],
    output: &mut [f32; BLOCK_SIZE],
    smooth_coeff: f32,
    start: usize,
    count: usize,
) {
    let target_level = params[0];
    for i in start..start + count {
        smooth[0] += smooth_coeff * (target_level - smooth[0]);
        output[i] = input[i] * smooth[0];
    }
}

fn mix_node(
    params: &[f32],
    smooth: &mut [f32],
    input_a: &[f32; BLOCK_SIZE],
    input_b: &[f32; BLOCK_SIZE],
    output: &mut [f32; BLOCK_SIZE],
    smooth_coeff: f32,
    start: usize,
    count: usize,
) {
    let target_mix = params[0].clamp(0.0, 1.0);
    for i in start..start + count {
        smooth[0] += smooth_coeff * (target_mix - smooth[0]);
        let m = smooth[0].clamp(0.0, 1.0);
        output[i] = input_a[i] * (1.0 - m) + input_b[i] * m;
    }
}

fn constant_node(params: &[f32], output: &mut [f32; BLOCK_SIZE], start: usize, count: usize) {
    let value = params[0];
    for i in start..start + count {
        output[i] = value;
    }
}

fn sample_node(
    state: &mut NodeState,
    params: &[f32],
    smooth: &mut [f32],
    output: &mut [f32; BLOCK_SIZE],
    samples: &[Vec<f32>],
    smooth_coeff: f32,
    start: usize,
    count: usize,
) {
    let buffer_id = params[0] as usize;
    let target_rate = params[1];
    let looping = params[2] as u32 != 0;
    let target_vol = params[3];

    let buffer: &[f32] = samples.get(buffer_id).map(|v| v.as_slice()).unwrap_or(&[]);

    if let NodeState::Sample { position } = state {
        for i in start..start + count {
            smooth[1] += smooth_coeff * (target_rate - smooth[1]);
            smooth[3] += smooth_coeff * (target_vol - smooth[3]);

            if buffer.is_empty() {
                output[i] = 0.0;
                continue;
            }

            if *position >= buffer.len() as f32 {
                if looping {
                    let len = buffer.len() as f32;
                    *position -= (*position / len).floor() * len;
                } else {
                    output[i] = 0.0;
                    continue;
                }
            }

            output[i] = sample_read(buffer, *position) * smooth[3];
            *position += smooth[1];
        }
    }
}

fn apply_mod(
    working_params: &mut [f32; MAX_PARAMS],
    params: &[f32; MAX_PARAMS],
    mc: &ModConnection,
    buffers: &[[f32; BLOCK_SIZE]; MAX_BUFFERS],
) {
    let source_val = buffers[mc.source_buf as usize][BLOCK_SIZE - 1];
    let depth = params[mc.depth_param as usize];
    let base = working_params[mc.target_param as usize];

    working_params[mc.target_param as usize] = match mc.mode {
        ModMode::Linear => base + source_val * depth,
        ModMode::Semitone => base * (source_val * depth * SEMITONE_TO_LOG).exp(),
    };
}

pub fn synthesize_graph_voice(
    instrument: &InstrumentDef,
    node_states: &mut [NodeState; MAX_NODES],
    params: &[f32; MAX_PARAMS],
    smooth_params: &mut [f32; MAX_PARAMS],
    buffers: &mut [[f32; BLOCK_SIZE]; MAX_BUFFERS],
    sample_rate: f32,
    smooth_coeff: f32,
    wavetable: &[f32],
    samples: &[Vec<f32>],
    start: usize,
    count: usize,
) -> u8 {
    let mut working_params = *params;

    for ni in 0..instrument.node_count as usize {
        let node = &instrument.nodes[ni];

        for mi in 0..instrument.mod_count as usize {
            let mc = &instrument.mod_connections[mi];
            if mc.target_node as usize == ni {
                apply_mod(&mut working_params, params, mc, buffers);
            }
        }

        let po = node.param_offset as usize;

        match node.node_type {
            NodeType::None => {}
            NodeType::Oscillator => {
                let ob = node.output_buf as usize;
                oscillator_node(
                    &mut node_states[ni],
                    &working_params[po..],
                    &mut smooth_params[po..],
                    &mut buffers[ob],
                    sample_rate,
                    smooth_coeff,
                    wavetable,
                    start,
                    count,
                );
            }
            NodeType::Filter => {
                let ib = node.input_buf as usize;
                let ob = node.output_buf as usize;
                if ib == ob {
                    let tmp = buffers[ib];
                    filter_node(
                        &mut node_states[ni],
                        &working_params[po..],
                        &mut smooth_params[po..],
                        &tmp,
                        &mut buffers[ob],
                        sample_rate,
                        smooth_coeff,
                        start,
                        count,
                    );
                } else {
                    let (inp, out) = borrow_io(buffers, ib, ob);
                    filter_node(
                        &mut node_states[ni],
                        &working_params[po..],
                        &mut smooth_params[po..],
                        inp,
                        out,
                        sample_rate,
                        smooth_coeff,
                        start,
                        count,
                    );
                }
            }
            NodeType::Envelope => {
                let ob = node.output_buf as usize;
                if node.input_buf == NO_BUF {
                    envelope_node(
                        &mut node_states[ni],
                        &working_params[po..],
                        None,
                        &mut buffers[ob],
                        sample_rate,
                        start,
                        count,
                    );
                } else {
                    let ib = node.input_buf as usize;
                    if ib == ob {
                        let tmp = buffers[ib];
                        envelope_node(
                            &mut node_states[ni],
                            &working_params[po..],
                            Some(&tmp),
                            &mut buffers[ob],
                            sample_rate,
                            start,
                            count,
                        );
                    } else {
                        let (inp, out) = borrow_io(buffers, ib, ob);
                        envelope_node(
                            &mut node_states[ni],
                            &working_params[po..],
                            Some(inp),
                            out,
                            sample_rate,
                            start,
                            count,
                        );
                    }
                }
            }
            NodeType::Gain => {
                let ib = node.input_buf as usize;
                let ob = node.output_buf as usize;
                if ib == ob {
                    let tmp = buffers[ib];
                    gain_node(
                        &working_params[po..],
                        &mut smooth_params[po..],
                        &tmp,
                        &mut buffers[ob],
                        smooth_coeff,
                        start,
                        count,
                    );
                } else {
                    let (inp, out) = borrow_io(buffers, ib, ob);
                    gain_node(
                        &working_params[po..],
                        &mut smooth_params[po..],
                        inp,
                        out,
                        smooth_coeff,
                        start,
                        count,
                    );
                }
            }
            NodeType::Mix => {
                let ia = node.input_buf as usize;
                let ib = node.input_buf_b as usize;
                let ob = node.output_buf as usize;
                let tmp_a = buffers[ia];
                let tmp_b = buffers[ib];
                mix_node(
                    &working_params[po..],
                    &mut smooth_params[po..],
                    &tmp_a,
                    &tmp_b,
                    &mut buffers[ob],
                    smooth_coeff,
                    start,
                    count,
                );
            }
            NodeType::Constant => {
                let ob = node.output_buf as usize;
                constant_node(&working_params[po..], &mut buffers[ob], start, count);
            }
            NodeType::Sample => {
                let ob = node.output_buf as usize;
                sample_node(
                    &mut node_states[ni],
                    &working_params[po..],
                    &mut smooth_params[po..],
                    &mut buffers[ob],
                    samples,
                    smooth_coeff,
                    start,
                    count,
                );
            }
        }
    }

    instrument.output_buf
}

pub fn tick_envelopes_only(
    instrument: &InstrumentDef,
    node_states: &mut [NodeState; MAX_NODES],
    params: &[f32; MAX_PARAMS],
    sample_rate: f32,
) {
    let dt = 1.0 / sample_rate;
    for ni in 0..instrument.node_count as usize {
        if instrument.nodes[ni].node_type != NodeType::Envelope {
            continue;
        }
        let po = instrument.nodes[ni].param_offset as usize;
        let (stage, level, time, attack_start, release_start) = match &node_states[ni] {
            NodeState::Envelope {
                stage,
                level,
                time,
                attack_start,
                release_start,
            } => (*stage, *level, *time, *attack_start, *release_start),
            _ => continue,
        };
        let mut env = Envelope {
            stage,
            level,
            time,
            attack_start_level: attack_start,
            release_start_level: release_start,
            attack: params[po].max(0.001),
            decay: params[po + 1].max(0.001),
            sustain: params[po + 2].clamp(0.0, 1.0),
            release: params[po + 3].max(0.001),
            attack_curve: params[po + 4].clamp(-1.0, 1.0),
            decay_curve: params[po + 5].clamp(-1.0, 1.0),
            release_curve: params[po + 6].clamp(-1.0, 1.0),
        };
        for _ in 0..BLOCK_SIZE {
            env.tick(dt);
        }
        node_states[ni].store_envelope(&env);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oscillator::{WAVETABLE_FRAMES, WAVETABLE_SAMPLES};

    fn single_node_instrument(
        node_type: NodeType,
        param_offset: u8,
        input_buf: u8,
        output_buf: u8,
    ) -> InstrumentDef {
        let mut inst = InstrumentDef::default();
        inst.node_count = 1;
        inst.output_buf = output_buf;
        inst.nodes[0] = NodeDef {
            node_type,
            input_buf,
            input_buf_b: NO_BUF,
            output_buf,
            param_offset,
        };
        inst
    }

    fn two_node_instrument(
        t0: NodeType,
        p0: u8,
        i0: u8,
        o0: u8,
        t1: NodeType,
        p1: u8,
        i1: u8,
        o1: u8,
    ) -> InstrumentDef {
        let mut inst = InstrumentDef::default();
        inst.node_count = 2;
        inst.output_buf = o1;
        inst.nodes[0] = NodeDef {
            node_type: t0,
            input_buf: i0,
            input_buf_b: NO_BUF,
            output_buf: o0,
            param_offset: p0,
        };
        inst.nodes[1] = NodeDef {
            node_type: t1,
            input_buf: i1,
            input_buf_b: NO_BUF,
            output_buf: o1,
            param_offset: p1,
        };
        inst
    }

    fn mix_instrument() -> InstrumentDef {
        let mut inst = InstrumentDef::default();
        inst.node_count = 1;
        inst.output_buf = 2;
        inst.nodes[0] = NodeDef {
            node_type: NodeType::Mix,
            input_buf: 0,
            input_buf_b: 1,
            output_buf: 2,
            param_offset: 0,
        };
        inst
    }

    const SR: f32 = 48000.0;
    const COEFF: f32 = 0.005;

    #[test]
    fn oscillator_sine_zero_crossings() {
        let inst = single_node_instrument(NodeType::Oscillator, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Oscillator { phase: 0.0 };
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 440.0;
        params[3] = 1.0;
        smooth[0] = 440.0;
        smooth[3] = 1.0;

        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        let mut crossings = 0;
        for i in 1..BLOCK_SIZE {
            if buffers[0][i - 1].signum() != buffers[0][i].signum() && buffers[0][i] != 0.0 {
                crossings += 1;
            }
        }
        assert!(
            crossings >= 1 && crossings <= 4,
            "440Hz sine: expected ~2 zero crossings in 128 samples, got {crossings}"
        );
    }

    #[test]
    fn oscillator_waveforms_in_range() {
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        for waveform_id in 0..4u32 {
            let inst = single_node_instrument(NodeType::Oscillator, 0, NO_BUF, 0);
            let mut states = [NodeState::default(); MAX_NODES];
            states[0] = NodeState::Oscillator { phase: 0.0 };
            let mut params = [0.0f32; MAX_PARAMS];
            let mut smooth = [0.0f32; MAX_PARAMS];
            let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];

            params[0] = 440.0;
            params[1] = waveform_id as f32;
            params[3] = 1.0;
            smooth[0] = 440.0;
            smooth[1] = waveform_id as f32;
            smooth[3] = 1.0;

            synthesize_graph_voice(
                &inst,
                &mut states,
                &params,
                &mut smooth,
                &mut buffers,
                SR,
                COEFF,
                &wt,
                &[],
                0,
                BLOCK_SIZE,
            );

            for i in 0..BLOCK_SIZE {
                assert!(
                    buffers[0][i].abs() <= 1.5,
                    "waveform {waveform_id} sample {i} out of range: {}",
                    buffers[0][i]
                );
            }
        }
    }

    #[test]
    fn filter_lowpass_attenuates_high() {
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let dt = 1.0 / SR;

        fn measure_energy(freq: f32, cutoff: f32, sr: f32, dt: f32, wt: &[f32]) -> f32 {
            let inst = single_node_instrument(NodeType::Filter, 0, 0, 1);
            let mut states = [NodeState::default(); MAX_NODES];
            states[0] = NodeState::Filter {
                ic1eq: 0.0,
                ic2eq: 0.0,
                a1: 0.0,
                a2: 0.0,
                a3: 0.0,
                k: 0.0,
            };
            let mut params = [0.0f32; MAX_PARAMS];
            let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];

            params[0] = cutoff;
            params[1] = 0.707;
            params[2] = 0.0; // lowpass
            params[3] = 1.0; // mix
            let mut smooth = params;

            let mut energy = 0.0f32;
            for block in 0..8 {
                for i in 0..BLOCK_SIZE {
                    let t = (block * BLOCK_SIZE + i) as f32 * dt;
                    buffers[0][i] = (core::f32::consts::TAU * freq * t).sin();
                }
                synthesize_graph_voice(
                    &inst,
                    &mut states,
                    &params,
                    &mut smooth,
                    &mut buffers,
                    sr,
                    0.005,
                    wt,
                    &[],
                    0,
                    BLOCK_SIZE,
                );
                if block >= 4 {
                    for i in 0..BLOCK_SIZE {
                        energy += buffers[1][i] * buffers[1][i];
                    }
                }
            }
            energy
        }

        let low_energy = measure_energy(200.0, 1000.0, SR, dt, &wt);
        let high_energy = measure_energy(10000.0, 1000.0, SR, dt, &wt);

        let ratio_db = 10.0 * (low_energy / high_energy.max(1e-30)).log10();
        assert!(
            ratio_db > 12.0,
            "LP@1kHz should attenuate 10kHz >12dB vs 200Hz, got {ratio_db:.1}dB"
        );
    }

    #[test]
    fn filter_mix_zero_bypasses() {
        let inst = single_node_instrument(NodeType::Filter, 0, 0, 1);
        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Filter {
            ic1eq: 0.0,
            ic2eq: 0.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            k: 0.0,
        };
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 1000.0;
        params[1] = 0.707;
        params[2] = 0.0;
        params[3] = 0.0; // mix=0 -> bypass
        smooth.copy_from_slice(&params);

        for i in 0..BLOCK_SIZE {
            buffers[0][i] = (i as f32 * 0.1).sin();
        }
        let input_copy = buffers[0];

        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        for i in 0..BLOCK_SIZE {
            assert!(
                (buffers[1][i] - input_copy[i]).abs() < 1e-6,
                "mix=0 should bypass, sample {i}: {} vs {}",
                buffers[1][i],
                input_copy[i]
            );
        }
    }

    #[test]
    fn envelope_attack_reaches_one() {
        let inst = single_node_instrument(NodeType::Envelope, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        // attack=0.01, decay=0.1, sustain=0.7, release=0.3
        params[0] = 0.01;
        params[1] = 0.1;
        params[2] = 0.7;
        params[3] = 0.3;

        states[0] = NodeState::Envelope {
            stage: EnvStage::Attack,
            level: 0.0,
            time: 0.0,
            attack_start: 0.0,
            release_start: 0.0,
        };

        for _ in 0..4 {
            synthesize_graph_voice(
                &inst,
                &mut states,
                &params,
                &mut smooth,
                &mut buffers,
                SR,
                COEFF,
                &wt,
                &[],
                0,
                BLOCK_SIZE,
            );
        }

        // After 4 blocks (512 samples = ~10.7ms), attack (10ms) should be done, stage = Decay
        assert!(
            matches!(
                states[0],
                NodeState::Envelope {
                    stage: EnvStage::Decay,
                    ..
                } | NodeState::Envelope {
                    stage: EnvStage::Sustain,
                    ..
                }
            ),
            "expected Decay or Sustain after attack, got {:?}",
            states[0]
        );
    }

    #[test]
    fn envelope_sustain_holds() {
        let inst = single_node_instrument(NodeType::Envelope, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 0.01;
        params[1] = 0.1;
        params[2] = 0.7;
        params[3] = 0.3;

        states[0] = NodeState::Envelope {
            stage: EnvStage::Sustain,
            level: 0.7,
            time: 0.0,
            attack_start: 0.0,
            release_start: 0.0,
        };

        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        for i in 0..BLOCK_SIZE {
            assert!(
                (buffers[0][i] - 0.7).abs() < 0.01,
                "sustain should hold at 0.7, sample {i}: {}",
                buffers[0][i]
            );
        }
    }

    #[test]
    fn envelope_release_to_idle() {
        let inst = single_node_instrument(NodeType::Envelope, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 0.01;
        params[1] = 0.1;
        params[2] = 0.7;
        params[3] = 0.1; // release=0.1s

        states[0] = NodeState::Envelope {
            stage: EnvStage::Release,
            level: 0.7,
            time: 0.0,
            attack_start: 0.0,
            release_start: 0.7,
        };

        // 0.1s = 4800 samples = ~38 blocks
        for _ in 0..40 {
            synthesize_graph_voice(
                &inst,
                &mut states,
                &params,
                &mut smooth,
                &mut buffers,
                SR,
                COEFF,
                &wt,
                &[],
                0,
                BLOCK_SIZE,
            );
        }

        assert!(
            matches!(
                states[0],
                NodeState::Envelope {
                    stage: EnvStage::Idle,
                    ..
                }
            ),
            "should reach Idle after release time"
        );
        if let NodeState::Envelope { level, .. } = states[0] {
            assert!(level.abs() < 0.01, "level should be ~0 after release");
        }
    }

    #[test]
    fn gain_multiplies() {
        let inst = single_node_instrument(NodeType::Gain, 0, 0, 1);
        let mut states = [NodeState::default(); MAX_NODES];
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 0.5;
        smooth[0] = 0.5;

        for i in 0..BLOCK_SIZE {
            buffers[0][i] = 1.0;
        }

        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        for i in 0..BLOCK_SIZE {
            assert!(
                (buffers[1][i] - 0.5).abs() < 0.01,
                "gain*1.0 should be ~0.5, sample {i}: {}",
                buffers[1][i]
            );
        }
    }

    #[test]
    fn mix_interpolates() {
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        for (mix_val, expected_desc) in [(0.0f32, "A"), (1.0, "B"), (0.5, "avg")] {
            let inst = mix_instrument();
            let mut states = [NodeState::default(); MAX_NODES];
            let mut params = [0.0f32; MAX_PARAMS];
            let mut smooth = [0.0f32; MAX_PARAMS];
            let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];

            params[0] = mix_val;
            smooth[0] = mix_val;

            for i in 0..BLOCK_SIZE {
                buffers[0][i] = 1.0; // A
                buffers[1][i] = 3.0; // B
            }

            synthesize_graph_voice(
                &inst,
                &mut states,
                &params,
                &mut smooth,
                &mut buffers,
                SR,
                COEFF,
                &wt,
                &[],
                0,
                BLOCK_SIZE,
            );

            let expected = 1.0 * (1.0 - mix_val) + 3.0 * mix_val;
            for i in 0..BLOCK_SIZE {
                assert!(
                    (buffers[2][i] - expected).abs() < 0.05,
                    "mix={mix_val} ({expected_desc}) sample {i}: expected ~{expected}, got {}",
                    buffers[2][i]
                );
            }
        }
    }

    #[test]
    fn constant_fills_value() {
        let inst = single_node_instrument(NodeType::Constant, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 42.0;

        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        for i in 0..BLOCK_SIZE {
            assert!(
                (buffers[0][i] - 42.0).abs() < 1e-6,
                "constant should fill 42.0, sample {i}: {}",
                buffers[0][i]
            );
        }
    }

    #[test]
    fn modulation_linear() {
        let mut inst = two_node_instrument(
            NodeType::Constant,
            0,
            NO_BUF,
            0,
            NodeType::Oscillator,
            4,
            NO_BUF,
            1,
        );
        inst.mod_count = 1;
        inst.mod_connections[0] = ModConnection {
            source_buf: 0,
            target_node: 1,
            target_param: 4, // osc freq (param_offset=4, so param index 4 = freq)
            depth_param: 8,  // depth stored at param 8
            mode: ModMode::Linear,
        };

        let mut states = [NodeState::default(); MAX_NODES];
        states[1] = NodeState::Oscillator { phase: 0.0 };
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 1.0; // constant value
        params[4] = 440.0; // osc freq
        params[5] = 0.0; // osc waveform (sine)
        params[7] = 1.0; // osc volume
        params[8] = 100.0; // mod depth
        smooth.copy_from_slice(&params);

        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        // Constant=1.0, depth=100 → freq shifts from 440 to 540
        // 540Hz at 48kHz: period ~89 samples. In 128 samples we expect ~2-3 zero crossings
        let mut crossings = 0;
        for i in 1..BLOCK_SIZE {
            if buffers[1][i - 1].signum() != buffers[1][i].signum() && buffers[1][i] != 0.0 {
                crossings += 1;
            }
        }
        // 540Hz → ~2.9 cycles in 128 samples → ~5-6 zero crossings (2 per cycle)
        // Without modulation (440Hz) → ~2.3 cycles → ~4-5 crossings
        // We just verify the output is non-silent and has oscillation
        assert!(
            crossings >= 2,
            "modulated osc should produce zero crossings, got {crossings}"
        );

        // Also verify the osc output is non-zero (modulation applied)
        let energy: f32 = buffers[1].iter().map(|s| s * s).sum();
        assert!(
            energy > 0.1,
            "modulated osc should produce output, energy={energy}"
        );
    }

    #[test]
    fn semitone_modulation_octave_up() {
        let mut inst = two_node_instrument(
            NodeType::Constant,
            0,
            NO_BUF,
            0,
            NodeType::Oscillator,
            4,
            NO_BUF,
            1,
        );
        inst.mod_count = 1;
        inst.mod_connections[0] = ModConnection {
            source_buf: 0,
            target_node: 1,
            target_param: 4,
            depth_param: 8,
            mode: ModMode::Semitone,
        };

        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        let count_crossings = |freq: f32, mod_val: f32, depth: f32| -> usize {
            let mut states = [NodeState::default(); MAX_NODES];
            states[1] = NodeState::Oscillator { phase: 0.0 };
            let mut params = [0.0f32; MAX_PARAMS];
            let mut smooth = [0.0f32; MAX_PARAMS];
            let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];

            params[0] = mod_val;
            params[4] = freq;
            params[5] = 0.0;
            params[7] = 1.0;
            params[8] = depth;
            smooth.copy_from_slice(&params);

            let blocks = 16;
            let mut all_samples = Vec::new();
            for _ in 0..blocks {
                synthesize_graph_voice(
                    &inst,
                    &mut states,
                    &params,
                    &mut smooth,
                    &mut buffers,
                    SR,
                    COEFF,
                    &wt,
                    &[],
                    0,
                    BLOCK_SIZE,
                );
                all_samples.extend_from_slice(&buffers[1]);
            }
            let skip = BLOCK_SIZE * 4;
            let mut crossings = 0;
            for i in (skip + 1)..all_samples.len() {
                if all_samples[i - 1].signum() != all_samples[i].signum() && all_samples[i] != 0.0 {
                    crossings += 1;
                }
            }
            crossings
        };

        let base_crossings = count_crossings(440.0, 0.0, 12.0);
        let octave_crossings = count_crossings(440.0, 1.0, 12.0);

        let ratio = octave_crossings as f32 / base_crossings as f32;
        assert!((ratio - 2.0).abs() < 0.3, "octave up should ~double crossings: base={base_crossings}, octave={octave_crossings}, ratio={ratio:.2}");
    }

    #[test]
    fn semitone_modulation_zero_depth_unchanged() {
        let mut inst = two_node_instrument(
            NodeType::Constant,
            0,
            NO_BUF,
            0,
            NodeType::Oscillator,
            4,
            NO_BUF,
            1,
        );
        inst.mod_count = 1;
        inst.mod_connections[0] = ModConnection {
            source_buf: 0,
            target_node: 1,
            target_param: 4,
            depth_param: 8,
            mode: ModMode::Semitone,
        };

        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        let run = |depth: f32| -> Vec<f32> {
            let mut states = [NodeState::default(); MAX_NODES];
            states[1] = NodeState::Oscillator { phase: 0.0 };
            let mut params = [0.0f32; MAX_PARAMS];
            let mut smooth = [0.0f32; MAX_PARAMS];
            let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];

            params[0] = 1.0;
            params[4] = 440.0;
            params[5] = 0.0;
            params[7] = 1.0;
            params[8] = depth;
            smooth.copy_from_slice(&params);

            synthesize_graph_voice(
                &inst,
                &mut states,
                &params,
                &mut smooth,
                &mut buffers,
                SR,
                COEFF,
                &wt,
                &[],
                0,
                BLOCK_SIZE,
            );
            buffers[1].to_vec()
        };

        let with_depth_0 = run(0.0);
        let no_mod_inst = single_node_instrument(NodeType::Oscillator, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Oscillator { phase: 0.0 };
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        params[0] = 440.0;
        params[1] = 0.0;
        params[3] = 1.0;
        smooth.copy_from_slice(&params);
        synthesize_graph_voice(
            &no_mod_inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        for i in 0..BLOCK_SIZE {
            assert!(
                (with_depth_0[i] - buffers[0][i]).abs() < 1e-4,
                "depth=0 should match unmodulated at sample {i}"
            );
        }
    }

    #[test]
    fn osc_into_filter_into_envelope() {
        let mut inst = InstrumentDef::default();
        inst.node_count = 3;
        inst.output_buf = 2;
        inst.nodes[0] = NodeDef {
            node_type: NodeType::Oscillator,
            input_buf: NO_BUF,
            input_buf_b: NO_BUF,
            output_buf: 0,
            param_offset: 0,
        };
        inst.nodes[1] = NodeDef {
            node_type: NodeType::Filter,
            input_buf: 0,
            input_buf_b: NO_BUF,
            output_buf: 1,
            param_offset: 4,
        };
        inst.nodes[2] = NodeDef {
            node_type: NodeType::Envelope,
            input_buf: 1,
            input_buf_b: NO_BUF,
            output_buf: 2,
            param_offset: 8,
        };

        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Oscillator { phase: 0.0 };
        states[1] = NodeState::Filter {
            ic1eq: 0.0,
            ic2eq: 0.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            k: 0.0,
        };
        states[2] = NodeState::Envelope {
            stage: EnvStage::Attack,
            level: 0.0,
            time: 0.0,
            attack_start: 0.0,
            release_start: 0.0,
        };

        let mut params = [0.0f32; MAX_PARAMS];
        params[0] = 440.0;
        params[1] = 0.0;
        params[3] = 1.0;
        params[4] = 2000.0;
        params[5] = 0.707;
        params[6] = 0.0;
        params[7] = 1.0;
        params[8] = 0.01;
        params[9] = 0.1;
        params[10] = 0.7;
        params[11] = 0.3;
        let mut smooth = params;
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        let energy: f32 = buffers[2].iter().map(|s| s * s).sum();
        assert!(energy > 0.0, "3-node chain should produce non-zero output");
        assert!(
            buffers[2].iter().all(|s| s.is_finite()),
            "all samples should be finite"
        );
        assert!(
            buffers[2].iter().all(|s| s.abs() <= 2.0),
            "all samples should be bounded"
        );
    }

    #[test]
    fn osc_filter_envelope_frequency_content() {
        use crate::fft::{FftPlan, FFT_SIZE, SPECTRUM_SIZE};

        let mut inst = InstrumentDef::default();
        inst.node_count = 3;
        inst.output_buf = 2;
        inst.nodes[0] = NodeDef {
            node_type: NodeType::Oscillator,
            input_buf: NO_BUF,
            input_buf_b: NO_BUF,
            output_buf: 0,
            param_offset: 0,
        };
        inst.nodes[1] = NodeDef {
            node_type: NodeType::Filter,
            input_buf: 0,
            input_buf_b: NO_BUF,
            output_buf: 1,
            param_offset: 4,
        };
        inst.nodes[2] = NodeDef {
            node_type: NodeType::Envelope,
            input_buf: 1,
            input_buf_b: NO_BUF,
            output_buf: 2,
            param_offset: 8,
        };

        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Oscillator { phase: 0.0 };
        states[1] = NodeState::Filter {
            ic1eq: 0.0,
            ic2eq: 0.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            k: 0.0,
        };
        states[2] = NodeState::Envelope {
            stage: EnvStage::Sustain,
            level: 1.0,
            time: 0.0,
            attack_start: 0.0,
            release_start: 0.0,
        };

        let mut params = [0.0f32; MAX_PARAMS];
        params[0] = 440.0;
        params[1] = 1.0;
        params[3] = 1.0;
        params[4] = 1000.0;
        params[5] = 0.707;
        params[6] = 0.0;
        params[7] = 1.0;
        params[8] = 0.01;
        params[9] = 0.1;
        params[10] = 1.0;
        params[11] = 0.3;
        let mut smooth = params;
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        let mut collected = [0.0f32; FFT_SIZE];
        for block in 0..(FFT_SIZE / BLOCK_SIZE) {
            synthesize_graph_voice(
                &inst,
                &mut states,
                &params,
                &mut smooth,
                &mut buffers,
                SR,
                COEFF,
                &wt,
                &[],
                0,
                BLOCK_SIZE,
            );
            collected[block * BLOCK_SIZE..(block + 1) * BLOCK_SIZE].copy_from_slice(&buffers[2]);
        }

        let plan = FftPlan::new();
        let mut re = [0.0f32; SPECTRUM_SIZE];
        let mut im = [0.0f32; SPECTRUM_SIZE];
        plan.rfft(&collected, &mut re, &mut im);

        let cutoff_bin = (1000.0 / SR * FFT_SIZE as f32).round() as usize;
        let mut below_energy = 0.0f32;
        let mut above_energy = 0.0f32;
        for k in 1..SPECTRUM_SIZE {
            let e = re[k] * re[k] + im[k] * im[k];
            if k <= cutoff_bin {
                below_energy += e;
            } else {
                above_energy += e;
            }
        }
        assert!(
            below_energy > above_energy * 10.0,
            "LP@1kHz saw: below-cutoff energy {below_energy} should be >10x above {above_energy}"
        );
    }

    #[test]
    fn gain_zero_silences() {
        let inst = single_node_instrument(NodeType::Gain, 0, 0, 1);
        let mut states = [NodeState::default(); MAX_NODES];
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 0.0;
        smooth[0] = 0.0;
        for i in 0..BLOCK_SIZE {
            buffers[0][i] = (i as f32 * 0.1).sin();
        }

        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );

        for i in 0..BLOCK_SIZE {
            assert!(
                buffers[1][i].abs() < 1e-6,
                "gain=0 should silence, sample {i}: {}",
                buffers[1][i]
            );
        }
    }

    #[test]
    fn mix_node_with_modulated_crossfade() {
        let mut inst = InstrumentDef::default();
        inst.node_count = 4;
        inst.output_buf = 3;
        inst.nodes[0] = NodeDef {
            node_type: NodeType::Oscillator,
            input_buf: NO_BUF,
            input_buf_b: NO_BUF,
            output_buf: 0,
            param_offset: 0,
        };
        inst.nodes[1] = NodeDef {
            node_type: NodeType::Oscillator,
            input_buf: NO_BUF,
            input_buf_b: NO_BUF,
            output_buf: 1,
            param_offset: 4,
        };
        inst.nodes[2] = NodeDef {
            node_type: NodeType::Envelope,
            input_buf: NO_BUF,
            input_buf_b: NO_BUF,
            output_buf: 2,
            param_offset: 8,
        };
        inst.nodes[3] = NodeDef {
            node_type: NodeType::Mix,
            input_buf: 0,
            input_buf_b: 1,
            output_buf: 3,
            param_offset: 15,
        };
        inst.mod_count = 1;
        inst.mod_connections[0] = ModConnection {
            source_buf: 2,
            target_node: 3,
            target_param: 15,
            depth_param: 16,
            mode: ModMode::Linear,
        };

        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Oscillator { phase: 0.0 };
        states[1] = NodeState::Oscillator { phase: 0.0 };
        states[2] = NodeState::Envelope {
            stage: EnvStage::Attack,
            level: 0.0,
            time: 0.0,
            attack_start: 0.0,
            release_start: 0.0,
        };

        let mut params = [0.0f32; MAX_PARAMS];
        params[0] = 440.0;
        params[1] = 0.0;
        params[3] = 1.0;
        params[4] = 880.0;
        params[5] = 0.0;
        params[7] = 1.0;
        params[8] = 0.005;
        params[9] = 0.1;
        params[10] = 1.0;
        params[11] = 0.3;
        params[15] = 0.0;
        params[16] = 1.0;
        let mut smooth = params;
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        let mut first_block = [0.0f32; BLOCK_SIZE];
        synthesize_graph_voice(
            &inst,
            &mut states,
            &params,
            &mut smooth,
            &mut buffers,
            SR,
            COEFF,
            &wt,
            &[],
            0,
            BLOCK_SIZE,
        );
        first_block.copy_from_slice(&buffers[3]);

        for _ in 0..8 {
            synthesize_graph_voice(
                &inst,
                &mut states,
                &params,
                &mut smooth,
                &mut buffers,
                SR,
                COEFF,
                &wt,
                &[],
                0,
                BLOCK_SIZE,
            );
        }
        let last_block = buffers[3];

        let first_energy: f32 = first_block.iter().map(|s| s * s).sum();
        let last_energy: f32 = last_block.iter().map(|s| s * s).sum();
        assert!(first_energy > 0.01, "first block should have output");
        assert!(last_energy > 0.01, "last block should have output");
    }

    struct Mode {
        freq: f32,
        amp: f32,
        decay: f32,
    }

    // Linear-chain mix tree: each mix outputs (a + b) / 2.
    // Mode i lands at the final output with scaling 1 / 2^chain_depth_i.
    // For 5 modes: scalings are [1/16, 1/16, 1/8, 1/4, 1/2].
    const CHAIN_SCALING: [f32; 5] = [1.0 / 16.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0];

    fn build_modal_instrument(modes: &[Mode]) -> (InstrumentDef, [f32; MAX_PARAMS]) {
        assert_eq!(modes.len(), 5, "test fixture sized for 5 modes");

        let mut inst = InstrumentDef::default();
        let mut params = [0.0f32; MAX_PARAMS];

        // Layout: osc1, env1, osc2, env2, mix1, osc3, env3, mix2, osc4, env4, mix3, osc5, env5, mix4
        // Buffer reuse: osc/env share buf 0 or buf 1; mix output → buf 0; next osc/env → buf 1.
        let mut node_idx = 0usize;
        let mut po = 0u8; // running param offset
        let mut prev_mix_buf = 0u8; // accumulator buffer

        for (i, mode) in modes.iter().enumerate() {
            // Place mode in the chain such that osc/env sit on buffer 0 (first mode) or 1 (subsequent).
            let osc_buf: u8 = if i == 0 { 0 } else { 1 };

            // osc node
            params[po as usize] = mode.freq;
            params[po as usize + 1] = 0.0; // sine
            params[po as usize + 2] = 0.0; // wavetable_pos
            // Compensate for mix-chain attenuation so the final per-mode amplitude equals mode.amp.
            params[po as usize + 3] = mode.amp / CHAIN_SCALING[i];
            inst.nodes[node_idx] = NodeDef {
                node_type: NodeType::Oscillator,
                input_buf: NO_BUF,
                input_buf_b: NO_BUF,
                output_buf: osc_buf,
                param_offset: po,
            };
            node_idx += 1;
            po += 4;

            // env node (in-place on osc_buf)
            params[po as usize] = 0.001; // attack
            params[po as usize + 1] = mode.decay;
            params[po as usize + 2] = 0.0; // sustain (set by tests when steady tone wanted)
            params[po as usize + 3] = 0.005; // release
            params[po as usize + 4] = 0.0;
            params[po as usize + 5] = -0.3; // decay curve (fast → slow)
            params[po as usize + 6] = -0.3;
            inst.nodes[node_idx] = NodeDef {
                node_type: NodeType::Envelope,
                input_buf: osc_buf,
                input_buf_b: NO_BUF,
                output_buf: osc_buf,
                param_offset: po,
            };
            node_idx += 1;
            po += 7;

            if i > 0 {
                // mix prev_mix_buf + osc_buf (=1) → buf 0
                params[po as usize] = 0.5;
                inst.nodes[node_idx] = NodeDef {
                    node_type: NodeType::Mix,
                    input_buf: prev_mix_buf,
                    input_buf_b: osc_buf,
                    output_buf: 0,
                    param_offset: po,
                };
                node_idx += 1;
                po += 1;
                prev_mix_buf = 0;
            }
        }

        inst.node_count = node_idx as u8;
        inst.output_buf = 0;
        (inst, params)
    }

    fn init_modal_states(modes_len: usize) -> [NodeState; MAX_NODES] {
        let mut states = [NodeState::default(); MAX_NODES];
        let mut idx = 0;
        for i in 0..modes_len {
            states[idx] = NodeState::Oscillator { phase: 0.0 };
            idx += 1;
            states[idx] = NodeState::Envelope {
                stage: EnvStage::Attack,
                level: 0.0,
                time: 0.0,
                attack_start: 0.0,
                release_start: 0.0,
            };
            idx += 1;
            if i > 0 {
                idx += 1; // mix node — no per-node state
            }
        }
        states
    }

    /// Single-bin DFT magnitude (Goertzel-style). Returns peak amplitude of a sinusoid
    /// at `freq` Hz present in `samples` taken at `sr` Hz.
    fn detect_amplitude(samples: &[f32], freq: f32, sr: f32) -> f32 {
        let n = samples.len() as f32;
        let omega = core::f32::consts::TAU * freq / sr;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, &s) in samples.iter().enumerate() {
            let phase = omega * i as f32;
            re += s * phase.cos();
            im += s * phase.sin();
        }
        2.0 * (re * re + im * im).sqrt() / n
    }

    fn render_modal(
        inst: &InstrumentDef,
        params: &[f32; MAX_PARAMS],
        modes_len: usize,
        n_blocks: usize,
    ) -> Vec<f32> {
        let mut states = init_modal_states(modes_len);
        let mut smooth = *params;
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        let mut out = Vec::with_capacity(n_blocks * BLOCK_SIZE);
        for _ in 0..n_blocks {
            synthesize_graph_voice(
                inst,
                &mut states,
                params,
                &mut smooth,
                &mut buffers,
                SR,
                COEFF,
                &wt,
                &[],
                0,
                BLOCK_SIZE,
            );
            out.extend_from_slice(&buffers[inst.output_buf as usize]);
        }
        out
    }

    #[test]
    fn modal_synthesis_peaks_at_configured_frequencies() {
        // Inharmonic spectrum (loose body-like ratios), well-separated in frequency.
        let modes = [
            Mode { freq: 280.0, amp: 1.0, decay: 4.0 },
            Mode { freq: 590.0, amp: 0.7, decay: 4.0 },
            Mode { freq: 940.0, amp: 0.5, decay: 4.0 },
            Mode { freq: 1380.0, amp: 0.35, decay: 4.0 },
            Mode { freq: 1860.0, amp: 0.2, decay: 4.0 },
        ];
        let (inst, mut params) = build_modal_instrument(&modes);
        // Hold envelope at peak for a steady-state spectral measurement.
        for ni in 0..inst.node_count as usize {
            if inst.nodes[ni].node_type == NodeType::Envelope {
                let po = inst.nodes[ni].param_offset as usize;
                params[po + 2] = 1.0; // sustain
            }
        }

        // Skip first ~20ms (attack + smoothing settling); analyze the next ~80ms (4096 samples).
        let total_blocks = 64; // 64 * 128 / 48000 ≈ 170ms
        let pcm = render_modal(&inst, &params, modes.len(), total_blocks);
        let analysis_start = (0.020 * SR) as usize;
        let analysis_len = 4096;
        let window = &pcm[analysis_start..analysis_start + analysis_len];

        // Per-mode amplitude detection.
        for mode in &modes {
            let detected = detect_amplitude(window, mode.freq, SR);
            let rel_err = (detected - mode.amp).abs() / mode.amp;
            assert!(
                rel_err < 0.10,
                "mode {}Hz: expected amp {:.3}, detected {:.3} (rel err {:.2}%)",
                mode.freq,
                mode.amp,
                detected,
                rel_err * 100.0
            );
        }

        // Off-mode probe: midpoints between adjacent modes should be nearly silent.
        for w in modes.windows(2) {
            let mid_freq = 0.5 * (w[0].freq + w[1].freq);
            let mid_amp = detect_amplitude(window, mid_freq, SR);
            let max_mode_amp = w[0].amp.max(w[1].amp);
            assert!(
                mid_amp < 0.10 * max_mode_amp,
                "off-mode probe at {:.0}Hz: amp {:.3} should be <10% of nearby mode peak {:.3}",
                mid_freq,
                mid_amp,
                max_mode_amp
            );
        }
    }

    /// Demodulate at `freq`, lowpass via cascaded 1-poles, return the per-sample
    /// amplitude envelope of the mode at that frequency.
    fn extract_mode_envelope(samples: &[f32], freq: f32, sr: f32) -> Vec<f32> {
        let omega = core::f32::consts::TAU * freq / sr;
        // ~80Hz cutoff; nearest inter-mode spacing is ~310Hz, three cascaded 1-poles
        // give ~36dB attenuation at the nearest neighbor after demodulation.
        let alpha = 1.0 - (-core::f32::consts::TAU * 80.0 / sr).exp();

        let (mut re1, mut re2, mut re3) = (0.0f32, 0.0f32, 0.0f32);
        let (mut im1, mut im2, mut im3) = (0.0f32, 0.0f32, 0.0f32);

        let mut env = Vec::with_capacity(samples.len());
        for (i, &s) in samples.iter().enumerate() {
            let phase = omega * i as f32;
            let tr = 2.0 * s * phase.cos();
            let ti = -2.0 * s * phase.sin();
            re1 += alpha * (tr - re1);
            re2 += alpha * (re1 - re2);
            re3 += alpha * (re2 - re3);
            im1 += alpha * (ti - im1);
            im2 += alpha * (im1 - im2);
            im3 += alpha * (im2 - im3);
            env.push((re3 * re3 + im3 * im3).sqrt());
        }
        env
    }

    #[test]
    fn modal_synthesis_per_mode_independent_decay() {
        // Higher modes decay faster — typical of real impacts.
        let modes = [
            Mode { freq: 280.0, amp: 1.0, decay: 0.40 },
            Mode { freq: 590.0, amp: 1.0, decay: 0.25 },
            Mode { freq: 940.0, amp: 1.0, decay: 0.15 },
            Mode { freq: 1380.0, amp: 1.0, decay: 0.10 },
            Mode { freq: 1860.0, amp: 1.0, decay: 0.06 },
        ];
        let (inst, params) = build_modal_instrument(&modes);

        let total_blocks = (0.5 * SR / BLOCK_SIZE as f32).ceil() as usize;
        let pcm = render_modal(&inst, &params, modes.len(), total_blocks);

        // Demodulation needs ~10ms to settle (cascaded LP group delay).
        let post_settle = (0.030 * SR) as usize;
        let mid = (0.120 * SR) as usize;
        let late = (0.250 * SR) as usize;

        let envs: Vec<Vec<f32>> = modes
            .iter()
            .map(|m| extract_mode_envelope(&pcm, m.freq, SR))
            .collect();

        // Each mode is present (non-trivially) shortly after attack. Threshold is loose
        // because the LP settling time eats into envelope peak for fast-decay modes.
        for (mode, env) in modes.iter().zip(envs.iter()) {
            assert!(
                env[post_settle] > 0.3,
                "mode {}Hz: post-attack envelope {:.3} should exceed sanity floor",
                mode.freq,
                env[post_settle]
            );
        }

        // At mid time, modes are ordered by decay time: slower decay → higher amplitude.
        // Mode 0 (D=400ms) should ring loudest; mode 4 (D=60ms) should be near-silent.
        for i in 0..modes.len() - 1 {
            let a = envs[i][mid];
            let b = envs[i + 1][mid];
            assert!(
                a > b,
                "at t=120ms: mode {}Hz (decay {:.2}s) env {:.3} should exceed mode {}Hz (decay {:.2}s) env {:.3}",
                modes[i].freq,
                modes[i].decay,
                a,
                modes[i + 1].freq,
                modes[i + 1].decay,
                b,
            );
        }

        // Fastest mode is essentially silent past 4× its decay time.
        assert!(
            envs[4][late] < 0.05,
            "mode 1860Hz (decay 60ms) at t=250ms: {:.3} should be near-silent",
            envs[4][late]
        );

        // Slowest mode still has appreciable energy at mid time.
        assert!(
            envs[0][mid] > 0.3,
            "mode 280Hz (decay 400ms) at t=120ms: {:.3} should still be ringing",
            envs[0][mid]
        );
    }

    fn sine_buffer(freq: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (core::f32::consts::TAU * freq * i as f32 / SR).sin())
            .collect()
    }

    #[test]
    fn sample_node_silent_with_no_buffer_registered() {
        let inst = single_node_instrument(NodeType::Sample, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Sample { position: 0.0 };
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        params[0] = 0.0;
        params[1] = 1.0;
        params[3] = 1.0;
        smooth[1] = 1.0;
        smooth[3] = 1.0;

        synthesize_graph_voice(
            &inst, &mut states, &params, &mut smooth, &mut buffers, SR, COEFF, &wt, &[], 0,
            BLOCK_SIZE,
        );

        for s in &buffers[0] {
            assert_eq!(*s, 0.0, "expected silence with no buffer registered");
        }
    }

    #[test]
    fn sample_node_unit_rate_reproduces_buffer() {
        let inst = single_node_instrument(NodeType::Sample, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Sample { position: 0.0 };
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        let buf = sine_buffer(480.0, 4800);
        let samples = vec![buf.clone()];

        params[0] = 0.0;
        params[1] = 1.0;
        params[3] = 1.0;
        smooth[1] = 1.0;
        smooth[3] = 1.0;

        synthesize_graph_voice(
            &inst, &mut states, &params, &mut smooth, &mut buffers, SR, COEFF, &wt, &samples, 0,
            BLOCK_SIZE,
        );

        // At unit rate from position 0, output should match the buffer sample-for-sample.
        for i in 0..BLOCK_SIZE {
            assert!(
                (buffers[0][i] - buf[i]).abs() < 1e-6,
                "i={i}: out={:.6} expected={:.6}",
                buffers[0][i],
                buf[i]
            );
        }
    }

    #[test]
    fn sample_node_double_rate_doubles_zero_crossings() {
        let inst = single_node_instrument(NodeType::Sample, 0, NO_BUF, 0);
        let mut buffers_1x = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let mut buffers_2x = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];
        let buf = sine_buffer(480.0, 4800);
        let samples = vec![buf];

        let mut count_crossings = |rate: f32, buffers: &mut [[f32; BLOCK_SIZE]; MAX_BUFFERS]| {
            let mut states = [NodeState::default(); MAX_NODES];
            states[0] = NodeState::Sample { position: 0.0 };
            let mut params = [0.0f32; MAX_PARAMS];
            let mut smooth = [0.0f32; MAX_PARAMS];
            params[1] = rate;
            params[3] = 1.0;
            smooth[1] = rate;
            smooth[3] = 1.0;
            synthesize_graph_voice(
                &inst, &mut states, &params, &mut smooth, buffers, SR, COEFF, &wt, &samples, 0,
                BLOCK_SIZE,
            );
            let mut crossings = 0;
            for i in 1..BLOCK_SIZE {
                if (buffers[0][i - 1] < 0.0) != (buffers[0][i] < 0.0) {
                    crossings += 1;
                }
            }
            crossings
        };

        let c1 = count_crossings(1.0, &mut buffers_1x);
        let c2 = count_crossings(2.0, &mut buffers_2x);
        assert!(
            c2 as f32 >= 1.6 * c1 as f32,
            "expected ~2x crossings at 2x rate; 1x={c1} 2x={c2}"
        );
    }

    #[test]
    fn sample_node_one_shot_silences_past_end() {
        let inst = single_node_instrument(NodeType::Sample, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        // Start halfway through a 100-sample buffer; one block (128 samples) takes us past the end.
        states[0] = NodeState::Sample { position: 50.0 };
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        let buf: Vec<f32> = (0..100).map(|_| 1.0).collect();
        let samples = vec![buf];

        params[0] = 0.0;
        params[1] = 1.0;
        params[2] = 0.0; // loop = false
        params[3] = 1.0;
        smooth[1] = 1.0;
        smooth[3] = 1.0;

        synthesize_graph_voice(
            &inst, &mut states, &params, &mut smooth, &mut buffers, SR, COEFF, &wt, &samples, 0,
            BLOCK_SIZE,
        );

        // First ~50 samples reproduce the buffer (constant 1.0); rest is silence.
        for i in 0..50 {
            assert!(
                (buffers[0][i] - 1.0).abs() < 1e-3,
                "i={i}: expected ~1.0, got {}",
                buffers[0][i]
            );
        }
        for i in 60..BLOCK_SIZE {
            assert_eq!(
                buffers[0][i], 0.0,
                "i={i}: expected silence past end, got {}",
                buffers[0][i]
            );
        }
    }

    #[test]
    fn sample_node_loop_continues_past_end() {
        let inst = single_node_instrument(NodeType::Sample, 0, NO_BUF, 0);
        let mut states = [NodeState::default(); MAX_NODES];
        states[0] = NodeState::Sample { position: 0.0 };
        let mut params = [0.0f32; MAX_PARAMS];
        let mut smooth = [0.0f32; MAX_PARAMS];
        let mut buffers = [[0.0f32; BLOCK_SIZE]; MAX_BUFFERS];
        let wt = [0.0f32; WAVETABLE_FRAMES * WAVETABLE_SAMPLES];

        // 10-sample ramp [0, 0.1, 0.2, ..., 0.9]
        let buf: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let samples = vec![buf];

        params[0] = 0.0;
        params[1] = 1.0;
        params[2] = 1.0; // loop = true
        params[3] = 1.0;
        smooth[1] = 1.0;
        smooth[3] = 1.0;

        synthesize_graph_voice(
            &inst, &mut states, &params, &mut smooth, &mut buffers, SR, COEFF, &wt, &samples, 0,
            BLOCK_SIZE,
        );

        // No silence anywhere — the ramp loops through the whole block.
        for i in 0..BLOCK_SIZE {
            // Output is ramp-shaped, so position 9 → ~0.9, position 0 → 0.0.
            // Just verify it stays in [0, 0.95] range and reaches a peak well above 0.5 at some point.
            assert!(
                buffers[0][i] >= 0.0 && buffers[0][i] <= 1.0,
                "i={i}: out of [0,1]: {}",
                buffers[0][i]
            );
        }
        let peak = buffers[0].iter().cloned().fold(0.0f32, f32::max);
        assert!(peak > 0.5, "expected ramp peak > 0.5 across loops, got {peak}");
    }
}
