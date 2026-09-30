import loadAudioWasm from "../../../crates/audio/pkg/shallot_audio.js";
import {
    type AudioDevice,
    audioContextState,
    type AudioContextState as DeviceAudioContextState,
    Devices,
} from "../../core/input";
import type { World } from "../../engine";
import { byId, getParamPairs, type Instrument } from "./instrument";
import { flushSamples } from "./sample";
import { createWorkletURL } from "./worklet";

export const MAX_VOICES = 64;
const SLOT_MASK = 0x7f;
const GEN_MASK = 0xffffff;

/**
 * device-level audio state owned by `AudioPlugin`: the AudioContext + worklet
 * host, the 64-slot voice allocator (a free-list + per-slot generation), and
 * the per-frame message batch. Read through the helper functions, not the
 * fields. The kernel owns all DSP; this owns allocation and the wire. there is
 * no CPU mirror of voice gate/instrument state
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
    /** sample id → version last sent to this worklet; cleared on re-init */
    sentSamples: Map<number, number>;
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

export const Audio: import("../../engine").Resource<Audio> = {
    create: () => ({
        ctx: null,
        node: null,
        free: freeList(),
        gen: new Int32Array(MAX_VOICES),
        queue: [],
        idle: new Map(),
        sentInstruments: new Map(),
        sentSamples: new Map(),
        spatial: new Float32Array(MAX_VOICES * 7),
        spatialLen: 0,
        resume: null,
        onState: null,
        onVisibility: null,
        onDevice: null,
        heartbeat: null,
        lastHeartbeat: 0,
        wasSuspended: false,
    }),
};

function contextState(state: globalThis.AudioContextState): DeviceAudioContextState {
    return state === "running" || state === "closed" ? state : "suspended";
}

function reconnect(world: World): void {
    const _audio = world.resource(Audio);

    if (!_audio.node || !_audio.ctx) return;
    _audio.node.disconnect();
    _audio.node.connect(_audio.ctx.destination);
}

/**
 * stand up the AudioContext + worklet + WASM kernel and reset the allocator.
 * The context may start suspended (no user gesture yet); a one-shot
 * pointer/key listener resumes it; the World-scoped audio record reports the state until then
 */
export async function initAudio(world: World): Promise<void> {
    const _audio = world.resource(Audio);

    disposeAudio(world);
    _audio.free = freeList();
    _audio.gen.fill(0);
    _audio.queue.length = 0;
    _audio.idle.clear();
    _audio.sentInstruments.clear();
    _audio.spatialLen = 0;
    _audio.sentSamples.clear();

    const ctx = new AudioContext();
    _audio.ctx = ctx;
    audioContextState(world, contextState(ctx.state));
    if (ctx.state === "suspended") {
        const resume = () => {
            ctx.resume();
            document.removeEventListener("pointerdown", resume);
            document.removeEventListener("keydown", resume);
            world.resource(Audio).resume = null;
        };
        document.addEventListener("pointerdown", resume);
        document.addEventListener("keydown", resume);
        _audio.resume = resume;
    }

    const wasmBytes = await loadAudioWasm();
    const url = createWorkletURL();
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const node = new AudioWorkletNode(ctx, "synth-processor", { outputChannelCount: [2] });
    node.connect(ctx.destination);
    node.port.postMessage({ type: "init", bytes: wasmBytes });
    _audio.node = node;

    _audio.wasSuspended = ctx.state !== "running";
    _audio.onState = () => {
        const _audio = world.resource(Audio);

        audioContextState(world, contextState(ctx.state));
        if (ctx.state === "running" && _audio.wasSuspended) {
            node.port.postMessage({ type: "reset" });
            reconnect(world);
        }
        _audio.wasSuspended = ctx.state !== "running";
    };
    ctx.addEventListener("statechange", _audio.onState);

    _audio.onVisibility = () => {
        if (document.visibilityState === "visible") {
            ctx.resume();
            reconnect(world);
        }
    };
    document.addEventListener("visibilitychange", _audio.onVisibility);

    _audio.onDevice = () => reconnect(world);
    navigator.mediaDevices?.addEventListener("devicechange", _audio.onDevice);

    node.onprocessorerror = (e) => console.error("audio worklet crashed:", e);
    node.port.onmessage = (e: MessageEvent) => {
        const _audio = world.resource(Audio);

        const d = e.data;
        if (d.type === "voice_idle") {
            const cb = _audio.idle.get(d.voiceId);
            if (cb) {
                _audio.idle.delete(d.voiceId);
                cb();
            }
        } else if (d.type === "overflow") {
            console.warn(`audio: ${d.count} events dropped (buffer full)`);
        } else if (d.type === "heartbeat") {
            _audio.lastHeartbeat = performance.now();
            if (d.outputPeak !== undefined && d.outputPeak < 0) {
                console.error("audio: NaN detected in output");
            }
            if (d.dropped > 0) console.error(`audio: ${d.dropped} blocks dropped`);
        } else if (d.type === "error") {
            console.error(`audio worklet error: ${d.message}`);
        }
    };

    _audio.lastHeartbeat = performance.now();
    _audio.heartbeat = setInterval(() => {
        const _audio = world.resource(Audio);

        if (ctx.state !== "running") return;
        if (performance.now() - _audio.lastHeartbeat > 3000) {
            reconnect(world);
            _audio.lastHeartbeat = performance.now();
        }
    }, 2000);
}

/** tear down the worklet, context, and all host listeners */
export function disposeAudio(
    world: World,
    audio: Audio = world.resource(Audio),
    facts: AudioDevice = world.resource(Devices).audio,
): void {
    const _audio = audio;
    flush(_audio);
    facts.context = _audio.ctx ? "closed" : "none";
    if (_audio.heartbeat) {
        clearInterval(_audio.heartbeat);
        _audio.heartbeat = null;
    }
    if (_audio.resume) {
        document.removeEventListener("pointerdown", _audio.resume);
        document.removeEventListener("keydown", _audio.resume);
        _audio.resume = null;
    }
    if (_audio.onState && _audio.ctx) _audio.ctx.removeEventListener("statechange", _audio.onState);
    if (_audio.onVisibility) document.removeEventListener("visibilitychange", _audio.onVisibility);
    if (_audio.onDevice)
        navigator.mediaDevices?.removeEventListener("devicechange", _audio.onDevice);
    _audio.onState = _audio.onVisibility = _audio.onDevice = null;
    _audio.node?.disconnect();
    _audio.node = null;
    _audio.ctx?.close();
    _audio.ctx = null;
}

/** flush pending sample uploads + the queued message batch. Once per frame */
export function tickAudio(world: World): void {
    flushSamples(world.resource(Audio).sentSamples, (id, channel, channels, data) =>
        enqueue(world, { type: "set_sample", id, channel, channels, data }),
    );
    flush(world.resource(Audio));
}

function enqueue(world: World, msg: object): void {
    world.resource(Audio).queue.push(msg);
}

function flush(_audio: Audio): void {
    if (!_audio.node || _audio.queue.length === 0) return;
    _audio.node.port.postMessage({ type: "batch", commands: _audio.queue });
    _audio.queue.length = 0;
}

// --- voice allocator -------------------------------------------------------

/** raw voice slot of a handle (no validity check) */
export function slotOf(handle: number): number {
    return handle & SLOT_MASK;
}

/** true when the slot still belongs to this handle's generation */
export function valid(world: World, handle: number): boolean {
    if (handle < 0) return false;
    const slot = handle & SLOT_MASK;
    return slot < MAX_VOICES && (world.resource(Audio).gen[slot] & GEN_MASK) === handle >>> 7;
}

/**
 * claim a voice slot, returning a generation-stamped handle (`-1` when the pool
 * is full). The handle invalidates the moment the slot is freed or re-claimed,
 * so a caller holding a stale handle no-ops every op against it
 */
export function alloc(world: World): number {
    const _audio = world.resource(Audio);

    const slot = _audio.free.pop();
    if (slot === undefined) return -1;
    const gen = ++_audio.gen[slot] & GEN_MASK;
    enqueue(world, { type: "voice_active", voiceId: slot, active: true });
    return slot | (gen << 7);
}

/** release a voice slot back to the pool, invalidating its handle */
export function free(world: World, handle: number): void {
    const _audio = world.resource(Audio);

    if (!valid(world, handle)) return;
    const slot = handle & SLOT_MASK;
    _audio.gen[slot]++;
    _audio.idle.delete(slot);
    enqueue(world, { type: "voice_active", voiceId: slot, active: false });
    _audio.free.push(slot);
}

// --- voice ops (gen-validated; send plain objects to the frozen worklet) ----

/** gate a voice on (`value` 1, note-on) or off (0, enters the envelope release): the musical trigger,
 *  distinct from freeing the slot. No-op on a stale handle. */
export function gate(world: World, handle: number, value: number): void {
    if (!valid(world, handle)) return;
    enqueue(world, { type: "gate", voiceId: handle & SLOT_MASK, value });
}

/** set one kernel param of a voice by its `offset` in the instrument's compiled param layout: the
 *  per-frame firehose the ECS layer drives volume/pitch through. No-op on a stale handle or negative offset. */
export function setParam(world: World, handle: number, offset: number, value: number): void {
    if (!valid(world, handle) || offset < 0) return;
    enqueue(world, { type: "params", changes: [[handle & SLOT_MASK, offset, value]] });
}

/** route a voice through the FOA + HRTF spatial path (`true`) or direct stereo (`false`) */
export function spatialize(world: World, handle: number, on: boolean): void {
    if (!valid(world, handle)) return;
    enqueue(world, { type: "voice_spatial", voiceId: handle & SLOT_MASK, spatial: on });
}

/** mark a voice one-shot: the kernel auto-gates-off + idles it when its envelope completes */
export function oneShot(world: World, handle: number): void {
    if (!valid(world, handle)) return;
    enqueue(world, { type: "voice_one_shot", voiceId: handle & SLOT_MASK });
}

/** register the slot for idle watching; `cb` fires once when the kernel reports it idle */
export function watchIdle(world: World, handle: number, cb: () => void): void {
    if (!valid(world, handle)) return;
    const slot = handle & SLOT_MASK;
    world.resource(Audio).idle.set(slot, cb);
    enqueue(world, { type: "watch_idle", voiceId: slot });
}

function registerInstrument(world: World, id: number, inst: Instrument): void {
    const _audio = world.resource(Audio);

    if (_audio.sentInstruments.get(id) === inst.version) return;
    enqueue(world, {
        type: "set_instrument",
        id,
        nodeCount: inst.nodes.length,
        outputBuf: inst.outputBuf,
        outputBufR: inst.outputBufR,
        nodes: inst.nodes,
        modulations: inst.modulations,
    });
    _audio.sentInstruments.set(id, inst.version);
}

/** point a voice at an instrument: send its topology (once per version) + static param values */
export function assign(world: World, handle: number, id: number): void {
    if (!valid(world, handle)) return;
    const inst = byId(id);
    if (!inst) return;
    const slot = handle & SLOT_MASK;
    registerInstrument(world, id, inst);
    enqueue(world, { type: "set_voice_instrument", voiceId: slot, instrumentId: id });
    const pairs = getParamPairs(id);
    if (pairs.length > 0) {
        enqueue(world, { type: "params", changes: pairs.map(([off, val]) => [slot, off, val]) });
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
 * distance: the polar form the frozen kernel renders FOA + HRTF from
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
    world: World,
    handle: number,
    az: number,
    el: number,
    dist: number,
    ref = 3,
    max = 100,
    roll = 1,
): void {
    const _audio = world.resource(Audio);

    if (!valid(world, handle) || _audio.spatialLen + 7 > _audio.spatial.length) return;
    const b = _audio.spatial;
    let i = _audio.spatialLen;
    b[i++] = handle & SLOT_MASK;
    b[i++] = az;
    b[i++] = el;
    b[i++] = dist;
    b[i++] = ref;
    b[i++] = max;
    b[i++] = roll;
    _audio.spatialLen = i;
}

/** flush the accumulated spatial batch as one worklet message */
export function flushSpatial(world: World): void {
    const _audio = world.resource(Audio);

    if (_audio.spatialLen === 0) return;
    enqueue(world, { type: "spatial", data: _audio.spatial.slice(0, _audio.spatialLen) });
    _audio.spatialLen = 0;
}

const C5 = 523.2511;

/** note → frequency in Hz, offsetting `base` by octaves / semitones / cents */
export function noteFreq(base: number, octave = 0, semitone = 0, fine = 0): number {
    const freq = base > 0 ? base : C5;
    return freq * 2 ** (octave + semitone / 12 + fine / 1200);
}
