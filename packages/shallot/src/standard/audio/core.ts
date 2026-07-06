import loadAudioWasm from "../../../rust/audio/pkg/shallot_audio.js";
import { byId, getParamPairs, type Instrument } from "./instrument";
import { flushSamples, resetSampleUploads } from "./sample";
import { createWorkletURL } from "./worklet";

// #doc:dev
// The `audio/core` surface is for building sound past the `Sound` happy path: authoring instruments and
// driving voices directly.
//
// ### Custom instruments
//
// An instrument is a DAG of typed DSP nodes — oscillator, filter, envelope, gain, a two-input mix,
// constant, and a PCM sample source — wired by naming each node's input. `instrument(def, name)` compiles
// the graph to a flat kernel eval order and registers it under a name, the same id space `Sound.instrument`
// and `play()` resolve. Naming `volumeParam` / `pitchParams` / `loopParam` in the definition wires the
// params the ECS layer firehoses per voice, so an authored instrument plugs into `Sound.volume` / `.pitch`
// with no extra work. The node kinds and their params are the reference below; the kernel is frozen, so
// this palette is the whole synthesis surface.
//
// ### Direct voices
//
// Bypass ECS for one-off or tightly-timed sound: `alloc` a voice handle, `assign` an instrument, `setParam`
// its params, `gate` it on to sound it, and `free` when done — or mark it `oneShot` and `watchIdle` to free
// it when the envelope completes. `spatialize` routes a voice through the FOA + HRTF path; `polar` +
// `addSpatial` + `flushSpatial` feed it a listener-relative position each frame, which is what the sound
// system does for a positioned `Sound`.

// #doc:dev
// ### The kernel
//
// Synthesis runs in a Rust/WASM kernel (`rust/audio/`) on the AudioWorklet thread. Rust owns per-sample and
// per-block work — oscillators, filters, envelopes, sample playback, the FOA + HRTF spatial render, the FDN
// reverb, mixing, the master limiter — and the DSP hot path never crosses the FFI mid-block. JS owns
// per-event and per-frame work: instrument authoring, voice allocation, and the spatial parameter feed.
//
// The kernel is frozen: each subsystem is grounded in a named reference and gated at a tolerance derived
// from f32 roundoff. Bit-portable formulas are checked against golden vectors; paths whose output depends
// on internal ordering (the reverb tail, the HRTF render) keep property gates instead.
//
// | Subsystem | Reference | Parity gate |
// |---|---|---|
// | State-variable filter | Simper/Cytomic TPT SVF (2013) | captured coefficients + unity-DC / band rolloff |
// | Shelving biquads | RBJ cookbook / Steam Audio `iir.cpp` | captured coefficients + shelf/peak direction |
// | Oscillators (saw / square / triangle) | DaisySP PolyBLEP | captured kernel + spectral rolloff |
// | Sample / wavetable interpolation | Niemitalo 4-point Hermite | captured kernel + polynomial reproduction |
// | FDN reverb | Steam Audio Jot FDN | RT60 decay + energy bounds (ordering isn't bit-portable) |
// | Spatial HRTF | Brown-Duda structural model (1998) | ITD / ILD + contralateral attenuation |
// | FFT | radix-2 real FFT | analytic-DFT vector + round-trip, Parseval |
//
// Deliberate choices, decided rather than provisional:
//
// - **Synthetic Brown-Duda HRTF + FOA** over measured HRIR — ships no impulse-data blob, fits the
//   build-size and procedural-first goals. A measured set is a free internal upgrade later.
// - **tanh soft-clip master limiter** — zero lookahead, the right call for a realtime thread.
// - **Cubic Hermite interpolation** for sample and wavetable reads — band-limits pitch-shift aliasing.
// - **f32 sample stream, f64 only where error compounds** (the transport counter, high-shelf coefficient
//   math, the sample read position) — Web Audio mandates f32 and a 64-voice sum errs by ≈ −108 dB.

const MAX_VOICES = 64;
const SLOT_MASK = 0x7f;
const GEN_MASK = 0xffffff;

/**
 * device-level audio state owned by `AudioPlugin`: the AudioContext + worklet
 * host, the 64-slot voice allocator (a free-list + per-slot generation), and
 * the per-frame message batch. Read through the helper functions, not the
 * fields. The kernel owns all DSP; this owns allocation and the wire — there is
 * no CPU mirror of voice gate/instrument state (the deleted `backend.ts` rot)
 */
export interface Audio {
    ctx: AudioContext | null;
    node: AudioWorkletNode | null;
    /** free voice slots, popped on alloc */
    free: number[];
    /** per-slot generation; bumped on alloc and free so a stale handle no-ops */
    gen: Int32Array;
    /** queued worklet messages, flushed once per frame as one batch */
    queue: object[];
    /** slot → callback fired when the worklet reports that voice idle */
    idle: Map<number, () => void>;
    /** instrument id → topology version last sent (re-sent only on change) */
    sentInstruments: Map<number, number>;
    /** spatial param batch: 7 floats per voice (slot, az, el, dist, ref, max, roll) */
    spatial: Float32Array;
    spatialLen: number;
    resume: (() => void) | null;
    onState: (() => void) | null;
    onVisibility: (() => void) | null;
    onDevice: (() => void) | null;
    heartbeat: ReturnType<typeof setInterval> | null;
    lastHeartbeat: number;
    wasSuspended: boolean;
}

function freeList(): number[] {
    const free: number[] = [];
    for (let i = MAX_VOICES - 1; i >= 0; i--) free.push(i);
    return free;
}

export const Audio: Audio = {
    ctx: null,
    node: null,
    free: freeList(),
    gen: new Int32Array(MAX_VOICES),
    queue: [],
    idle: new Map(),
    sentInstruments: new Map(),
    spatial: new Float32Array(MAX_VOICES * 7),
    spatialLen: 0,
    resume: null,
    onState: null,
    onVisibility: null,
    onDevice: null,
    heartbeat: null,
    lastHeartbeat: 0,
    wasSuspended: false,
};

/** true once the worklet node exists (audio host set up) */
export function started(): boolean {
    return Audio.node !== null;
}

/** true when the AudioContext is running (not suspended awaiting a user gesture) */
export function running(): boolean {
    return Audio.ctx?.state === "running";
}

function reconnect(): void {
    if (!Audio.node || !Audio.ctx) return;
    Audio.node.disconnect();
    Audio.node.connect(Audio.ctx.destination);
}

/**
 * stand up the AudioContext + worklet + WASM kernel and reset the allocator.
 * The context may start suspended (no user gesture yet) — a one-shot
 * pointer/key listener resumes it; `running()` reports false until then
 */
export async function initAudio(): Promise<void> {
    disposeAudio();
    Audio.free = freeList();
    Audio.gen.fill(0);
    Audio.queue.length = 0;
    Audio.idle.clear();
    Audio.sentInstruments.clear();
    Audio.spatialLen = 0;
    resetSampleUploads();

    const ctx = new AudioContext();
    Audio.ctx = ctx;
    if (ctx.state === "suspended") {
        const resume = () => {
            ctx.resume();
            document.removeEventListener("pointerdown", resume);
            document.removeEventListener("keydown", resume);
            Audio.resume = null;
        };
        document.addEventListener("pointerdown", resume);
        document.addEventListener("keydown", resume);
        Audio.resume = resume;
    }

    const wasmBytes = await loadAudioWasm();
    const url = createWorkletURL();
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const node = new AudioWorkletNode(ctx, "synth-processor", { outputChannelCount: [2] });
    node.connect(ctx.destination);
    node.port.postMessage({ type: "init", bytes: wasmBytes });
    Audio.node = node;

    Audio.wasSuspended = ctx.state !== "running";
    Audio.onState = () => {
        if (ctx.state === "running" && Audio.wasSuspended) {
            node.port.postMessage({ type: "reset" });
            reconnect();
        }
        Audio.wasSuspended = ctx.state !== "running";
    };
    ctx.addEventListener("statechange", Audio.onState);

    Audio.onVisibility = () => {
        if (document.visibilityState === "visible") {
            ctx.resume();
            reconnect();
        }
    };
    document.addEventListener("visibilitychange", Audio.onVisibility);

    Audio.onDevice = () => reconnect();
    navigator.mediaDevices?.addEventListener("devicechange", Audio.onDevice);

    node.onprocessorerror = (e) => console.error("audio worklet crashed:", e);
    node.port.onmessage = (e: MessageEvent) => {
        const d = e.data;
        if (d.type === "voice_idle") {
            const cb = Audio.idle.get(d.voiceId);
            if (cb) {
                Audio.idle.delete(d.voiceId);
                cb();
            }
        } else if (d.type === "overflow") {
            console.warn(`audio: ${d.count} events dropped (buffer full)`);
        } else if (d.type === "heartbeat") {
            Audio.lastHeartbeat = performance.now();
            if (d.outputPeak !== undefined && d.outputPeak < 0) {
                console.error("audio: NaN detected in output");
            }
            if (d.dropped > 0) console.error(`audio: ${d.dropped} blocks dropped`);
        } else if (d.type === "error") {
            console.error(`audio worklet error: ${d.message}`);
        }
    };

    Audio.lastHeartbeat = performance.now();
    Audio.heartbeat = setInterval(() => {
        if (ctx.state !== "running") return;
        if (performance.now() - Audio.lastHeartbeat > 3000) {
            reconnect();
            Audio.lastHeartbeat = performance.now();
        }
    }, 2000);
}

/** tear down the worklet, context, and all host listeners */
export function disposeAudio(): void {
    flush();
    if (Audio.heartbeat) {
        clearInterval(Audio.heartbeat);
        Audio.heartbeat = null;
    }
    if (Audio.resume) {
        document.removeEventListener("pointerdown", Audio.resume);
        document.removeEventListener("keydown", Audio.resume);
        Audio.resume = null;
    }
    if (Audio.onState && Audio.ctx) Audio.ctx.removeEventListener("statechange", Audio.onState);
    if (Audio.onVisibility) document.removeEventListener("visibilitychange", Audio.onVisibility);
    if (Audio.onDevice) navigator.mediaDevices?.removeEventListener("devicechange", Audio.onDevice);
    Audio.onState = Audio.onVisibility = Audio.onDevice = null;
    Audio.node?.disconnect();
    Audio.node = null;
    Audio.ctx?.close();
    Audio.ctx = null;
}

/** flush pending sample uploads + the queued message batch. Once per frame */
export function tickAudio(): void {
    flushSamples((id, channel, channels, data) =>
        enqueue({ type: "set_sample", id, channel, channels, data }),
    );
    flush();
}

function enqueue(msg: object): void {
    Audio.queue.push(msg);
}

function flush(): void {
    if (!Audio.node || Audio.queue.length === 0) return;
    Audio.node.port.postMessage({ type: "batch", commands: Audio.queue });
    Audio.queue.length = 0;
}

// --- voice allocator -------------------------------------------------------

/** raw voice slot of a handle (no validity check) */
export function slotOf(handle: number): number {
    return handle & SLOT_MASK;
}

/** true when the slot still belongs to this handle's generation */
export function valid(handle: number): boolean {
    if (handle < 0) return false;
    const slot = handle & SLOT_MASK;
    return slot < MAX_VOICES && (Audio.gen[slot] & GEN_MASK) === handle >>> 7;
}

/**
 * claim a voice slot, returning a generation-stamped handle (`-1` when the pool
 * is full). The handle invalidates the moment the slot is freed or re-claimed,
 * so a caller holding a stale handle no-ops every op against it
 */
export function alloc(): number {
    const slot = Audio.free.pop();
    if (slot === undefined) return -1;
    const gen = ++Audio.gen[slot] & GEN_MASK;
    enqueue({ type: "voice_active", voiceId: slot, active: true });
    return slot | (gen << 7);
}

/** release a voice slot back to the pool, invalidating its handle */
export function free(handle: number): void {
    if (!valid(handle)) return;
    const slot = handle & SLOT_MASK;
    Audio.gen[slot]++;
    Audio.idle.delete(slot);
    enqueue({ type: "voice_active", voiceId: slot, active: false });
    Audio.free.push(slot);
}

// --- voice ops (gen-validated; send plain objects to the frozen worklet) ----

/** gate a voice on (`value` 1, note-on) or off (0, enters the envelope release) — the musical trigger,
 *  distinct from freeing the slot. No-op on a stale handle. */
export function gate(handle: number, value: number): void {
    if (!valid(handle)) return;
    enqueue({ type: "gate", voiceId: handle & SLOT_MASK, value });
}

/** set one kernel param of a voice by its `offset` in the instrument's compiled param layout — the
 *  per-frame firehose the ECS layer drives volume/pitch through. No-op on a stale handle or negative offset. */
export function setParam(handle: number, offset: number, value: number): void {
    if (!valid(handle) || offset < 0) return;
    enqueue({ type: "params", changes: [[handle & SLOT_MASK, offset, value]] });
}

/** route a voice through the FOA + HRTF spatial path (`true`) or direct stereo (`false`) */
export function spatialize(handle: number, on: boolean): void {
    if (!valid(handle)) return;
    enqueue({ type: "voice_spatial", voiceId: handle & SLOT_MASK, spatial: on });
}

/** mark a voice one-shot: the kernel auto-gates-off + idles it when its envelope completes */
export function oneShot(handle: number): void {
    if (!valid(handle)) return;
    enqueue({ type: "voice_one_shot", voiceId: handle & SLOT_MASK });
}

/** register the slot for idle watching; `cb` fires once when the kernel reports it idle */
export function watchIdle(handle: number, cb: () => void): void {
    if (!valid(handle)) return;
    const slot = handle & SLOT_MASK;
    Audio.idle.set(slot, cb);
    enqueue({ type: "watch_idle", voiceId: slot });
}

function registerInstrument(id: number, inst: Instrument): void {
    if (Audio.sentInstruments.get(id) === inst.version) return;
    enqueue({
        type: "set_instrument",
        id,
        nodeCount: inst.nodes.length,
        outputBuf: inst.outputBuf,
        outputBufR: inst.outputBufR,
        nodes: inst.nodes,
        modulations: inst.modulations,
    });
    Audio.sentInstruments.set(id, inst.version);
}

/** point a voice at an instrument: send its topology (once per version) + static param values */
export function assign(handle: number, id: number): void {
    if (!valid(handle)) return;
    const inst = byId(id);
    if (!inst) return;
    const slot = handle & SLOT_MASK;
    registerInstrument(id, inst);
    enqueue({ type: "set_voice_instrument", voiceId: slot, instrumentId: id });
    const pairs = getParamPairs(id);
    if (pairs.length > 0) {
        enqueue({ type: "params", changes: pairs.map(([off, val]) => [slot, off, val]) });
    }
}

// --- spatial ---------------------------------------------------------------

interface Polar {
    azimuth: number;
    elevation: number;
    distance: number;
}
const _polar: Polar = { azimuth: 0, elevation: 0, distance: 0 };

/**
 * source offset (`d`) and listener basis (`r`/`u`/`f`, the listener world
 * matrix's right/up/forward columns) → listener-relative azimuth, elevation,
 * distance — the polar form the frozen kernel renders FOA + HRTF from
 */
export function polar(
    dx: number,
    dy: number,
    dz: number,
    rx: number,
    ry: number,
    rz: number,
    ux: number,
    uy: number,
    uz: number,
    fx: number,
    fy: number,
    fz: number,
): Polar {
    const localX = dx * rx + dy * ry + dz * rz;
    const localY = dx * ux + dy * uy + dz * uz;
    const localZ = dx * fx + dy * fy + dz * fz;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    _polar.azimuth = Math.atan2(localX, localZ);
    _polar.elevation =
        distance > 0.001 ? Math.asin(Math.max(-1, Math.min(1, localY / distance))) : 0;
    _polar.distance = distance;
    return _polar;
}

/** queue one voice's spatial params (polar) into the per-frame batch */
export function addSpatial(
    handle: number,
    az: number,
    el: number,
    dist: number,
    ref = 3,
    max = 100,
    roll = 1,
): void {
    if (!valid(handle) || Audio.spatialLen + 7 > Audio.spatial.length) return;
    const b = Audio.spatial;
    let i = Audio.spatialLen;
    b[i++] = handle & SLOT_MASK;
    b[i++] = az;
    b[i++] = el;
    b[i++] = dist;
    b[i++] = ref;
    b[i++] = max;
    b[i++] = roll;
    Audio.spatialLen = i;
}

/** flush the accumulated spatial batch as one worklet message */
export function flushSpatial(): void {
    if (Audio.spatialLen === 0) return;
    enqueue({ type: "spatial", data: Audio.spatial.slice(0, Audio.spatialLen) });
    Audio.spatialLen = 0;
}

const C5 = 523.2511;

/** note → frequency in Hz, offsetting `base` by octaves / semitones / cents */
export function noteFreq(base: number, octave = 0, semitone = 0, fine = 0): number {
    const freq = base > 0 ? base : C5;
    return freq * 2 ** (octave + semitone / 12 + fine / 1200);
}

export type { Instrument, InstrumentDef, ModulationDef, NodeDef, NodeType } from "./instrument";
export { byId, getParamPairs, Instruments, instrument, MAX_INSTRUMENTS } from "./instrument";
export { getSample, MAX_SAMPLES, Samples, sample, whenLoaded } from "./sample";
