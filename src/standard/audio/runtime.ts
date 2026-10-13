import loadAudioWasm from "../../../crates/audio/pkg/shallot_audio.js";
import { AudioOutput, AudioPlugin, Clips, Listener, registerClip, Sound } from "../../core/audio";
import { type AudioContextState, Devices } from "../../core/input";
import {
    composeGlobalTransform,
    GlobalTransform,
    PrepareGlobalTransformSystem,
} from "../../core/transform";
import type { EntityRef, Plugin, Resource, System, World } from "../../engine";
import { createWorkletURL } from "./worklet";

const MAX_VOICES = 64;
// Keep a voice-pool-sized tail of completion refs for rollback reconciliation.
const MAX_COMPLETED = MAX_VOICES;
const COMMANDS = 128;
const STRIDE = 9;
const FINISHED_OFFSET = 2 + COMMANDS * STRIDE;
const BUFFER_LENGTH = FINISHED_OFFSET + MAX_VOICES;
const PLAY = 1;
const UPDATE = 2;
const STOP = 3;
const MATRIX = new Float32Array(16);

interface Voice {
    readonly eid: number;
    readonly ref: EntityRef;
    readonly handle: number;
    readonly started: number;
    clip: number;
    loop: boolean;
    volume: number;
    stopping: boolean;
}

interface Runtime {
    context: AudioContext | null;
    node: AudioWorkletNode | null;
    keepalive: ConstantSourceNode | null;
    free: number[];
    generations: Uint32Array;
    owners: Map<number, Voice>;
    slots: (Voice | null)[];
    completed: Set<number>;
    sentClips: Map<number, number>;
    buffer: Float64Array | null;
    resume: (() => void) | null;
    onState: (() => void) | null;
    onVisibility: (() => void) | null;
    onDevice: (() => void) | null;
    heartbeat: ReturnType<typeof setInterval> | null;
    lastHeartbeat: number;
    suspended: boolean;
    serial: number;
}

export const AudioRuntime: Resource<Runtime> = {
    create: () => ({
        context: null,
        node: null,
        keepalive: null,
        free: freeSlots(),
        generations: new Uint32Array(MAX_VOICES),
        owners: new Map(),
        slots: Array.from({ length: MAX_VOICES }, () => null),
        completed: new Set(),
        sentClips: new Map(),
        buffer: new Float64Array(BUFFER_LENGTH),
        resume: null,
        onState: null,
        onVisibility: null,
        onDevice: null,
        heartbeat: null,
        lastHeartbeat: 0,
        suspended: false,
        serial: 0,
    }),
};

function freeSlots(): number[] {
    const free: number[] = [];
    for (let slot = MAX_VOICES - 1; slot >= 0; slot--) free.push(slot);
    return free;
}

function rememberCompleted(runtime: Runtime, ref: EntityRef): void {
    runtime.completed.delete(ref);
    runtime.completed.add(ref);
    if (runtime.completed.size > MAX_COMPLETED) {
        const oldest = runtime.completed.values().next().value;
        if (oldest !== undefined) runtime.completed.delete(oldest);
    }
}

function stateOf(context: globalThis.AudioContextState): AudioContextState {
    return context === "running" || context === "closed" ? context : "suspended";
}

function browserAudioContext(): typeof AudioContext | undefined {
    const globals = globalThis as typeof globalThis & { webkitAudioContext?: typeof AudioContext };
    return globals.AudioContext ?? globals.webkitAudioContext;
}

function connectOutput(world: World, runtime: Runtime): void {
    if (world.disposed || !runtime.context || !runtime.node) return;
    try {
        runtime.node.disconnect(runtime.context.destination);
    } catch {}
    runtime.node.connect(runtime.context.destination);
    const output = world.resource(AudioOutput);
    output.context = runtime.context;
    output.bus = runtime.node;
}

function resetVoices(runtime: Runtime): void {
    runtime.owners.clear();
    runtime.slots.fill(null);
    runtime.free = freeSlots();
}

function installResume(runtime: Runtime): void {
    if (
        runtime.resume ||
        !runtime.context ||
        runtime.context.state !== "suspended" ||
        typeof document === "undefined"
    )
        return;
    const context = runtime.context;
    const resume = () => {
        void context
            .resume()
            .catch((error) => console.error("audio context resume failed:", error));
    };
    document.addEventListener("pointerdown", resume);
    document.addEventListener("keydown", resume);
    document.addEventListener("click", resume);
    runtime.resume = resume;
}

function removeResume(runtime: Runtime): void {
    if (!runtime.resume || typeof document === "undefined") return;
    document.removeEventListener("pointerdown", runtime.resume);
    document.removeEventListener("keydown", runtime.resume);
    document.removeEventListener("click", runtime.resume);
    runtime.resume = null;
}

/** Start the per-World context and compiled-kernel worklet. No browser host is required in Node. */
export async function initializeAudio(world: World): Promise<void> {
    const runtime = world.resource(AudioRuntime);
    const Ctor = browserAudioContext();
    if (!Ctor || typeof AudioWorkletNode === "undefined") return;
    await disposeAudio(world, runtime);
    resetVoices(runtime);
    runtime.sentClips.clear();

    const context = new Ctor();
    runtime.context = context;
    const devices = world.resource(Devices).audio;
    devices.context = stateOf(context.state);
    const bytes = await loadAudioWasm();
    const module = await WebAssembly.compile(bytes);
    const url = createWorkletURL();
    try {
        await context.audioWorklet.addModule(url);
    } finally {
        URL.revokeObjectURL(url);
    }

    const node = new AudioWorkletNode(context, "shallot-audio", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2],
        channelCount: 2,
        channelCountMode: "explicit",
        processorOptions: { module },
    });
    runtime.node = node;
    const keepalive = context.createConstantSource();
    keepalive.offset.value = 0;
    keepalive.connect(node);
    keepalive.start();
    runtime.keepalive = keepalive;
    connectOutput(world, runtime);
    await new Promise<void>((resolve, reject) => {
        node.onprocessorerror = () => reject(new Error("audio worklet processor failed"));
        node.port.onmessageerror = () =>
            reject(new Error("audio worklet message failed to deserialize"));
        node.port.onmessage = (event: MessageEvent) => {
            if (event.data?.type === "ready") {
                node.port.onmessage = (message: MessageEvent) =>
                    receiveAudioMessage(world, message.data, runtime);
                resolve();
                return;
            }
            if (event.data?.type === "init-error") {
                reject(new Error(`audio worklet initialization failed: ${event.data.message}`));
                return;
            }
            receiveAudioMessage(world, event.data, runtime);
        };
    });

    runtime.suspended = context.state !== "running";
    devices.context = stateOf(context.state);
    installResume(runtime);
    runtime.onState = () => {
        if (world.disposed) return;
        const next = stateOf(context.state);
        world.resource(Devices).audio.context = next;
        if (next === "suspended") {
            for (const eid of [...world.query([Sound])]) {
                if (world.storage(Sound).loop.get(eid) === 0) world.destroy(eid);
            }
        }
        if (next === "running" && runtime.suspended) {
            node.port.postMessage({ type: "reset" });
            resetVoices(runtime);
            runtime.buffer = new Float64Array(BUFFER_LENGTH);
            connectOutput(world, runtime);
        }
        runtime.suspended = next !== "running";
        if (next === "running") removeResume(runtime);
        if (next === "suspended") installResume(runtime);
    };
    context.addEventListener("statechange", runtime.onState);
    runtime.onVisibility = () => {
        if (world.disposed) return;
        if (document.visibilityState === "visible") {
            void context.resume();
            connectOutput(world, runtime);
        }
    };
    if (typeof document !== "undefined")
        document.addEventListener("visibilitychange", runtime.onVisibility);
    runtime.onDevice = () => connectOutput(world, runtime);
    if (typeof navigator !== "undefined")
        navigator.mediaDevices?.addEventListener("devicechange", runtime.onDevice);
    runtime.lastHeartbeat = performance.now();
    runtime.heartbeat = setInterval(() => {
        if (world.disposed || context.state !== "running") return;
        if (performance.now() - runtime.lastHeartbeat > 3000) {
            connectOutput(world, runtime);
            runtime.lastHeartbeat = performance.now();
        }
    }, 2000);
}

export async function disposeAudio(
    world: World,
    runtime = world.resource(AudioRuntime),
    output = world.resource(AudioOutput),
    devices = world.resource(Devices),
): Promise<void> {
    const context = runtime.context;
    const node = runtime.node;
    if (runtime.heartbeat !== null) clearInterval(runtime.heartbeat);
    runtime.heartbeat = null;
    removeResume(runtime);
    if (runtime.onState && context) context.removeEventListener("statechange", runtime.onState);
    if (runtime.onVisibility && typeof document !== "undefined")
        document.removeEventListener("visibilitychange", runtime.onVisibility);
    if (runtime.onDevice && typeof navigator !== "undefined")
        navigator.mediaDevices?.removeEventListener("devicechange", runtime.onDevice);
    runtime.resume = runtime.onState = runtime.onVisibility = runtime.onDevice = null;
    runtime.keepalive?.disconnect();
    try {
        runtime.keepalive?.stop();
    } catch {}
    runtime.keepalive = null;
    if (node) {
        node.port.onmessage = null;
        node.port.onmessageerror = null;
        node.onprocessorerror = null;
        node.disconnect();
    }
    if (context) await context.close();
    runtime.context = null;
    runtime.node = null;
    runtime.buffer = new Float64Array(BUFFER_LENGTH);
    resetVoices(runtime);
    runtime.sentClips.clear();
    output.context = null;
    output.bus = null;
    if (context) devices.audio.context = "closed";
}

function uploadClips(world: World, runtime: Runtime): void {
    const context = runtime.context;
    const node = runtime.node;
    if (!context || !node) return;
    world.resource(Clips).registry.forEach((clip) => {
        const id = world.resource(Clips).registry.id(clip.name)!;
        if (runtime.sentClips.get(id) === clip.version) return;
        const channels = clip.channels.map((channel) =>
            resample(channel, clip.sampleRate, context.sampleRate),
        );
        const transfer = channels.map((channel) => channel.buffer as ArrayBuffer);
        node.port.postMessage({ type: "clip", id, channels }, transfer);
        runtime.sentClips.set(id, clip.version);
    });
}

function resample(pcm: Float32Array, sourceRate: number, targetRate: number): Float32Array {
    if (sourceRate === targetRate || pcm.length === 0) return pcm.slice();
    const length = Math.max(1, Math.round((pcm.length * targetRate) / sourceRate));
    const output = new Float32Array(length);
    const step = sourceRate / targetRate;
    for (let i = 0; i < length; i++) {
        const position = i * step;
        const left = Math.min(Math.floor(position), pcm.length - 1);
        const right = Math.min(left + 1, pcm.length - 1);
        const mix = position - left;
        output[i] = pcm[left] * (1 - mix) + pcm[right] * mix;
    }
    return output;
}

function polar(world: World, eid: number, listener: number): [number, number, number] {
    const listenerMatrix = composeGlobalTransform(world, listener, MATRIX);
    const transform = world.storage(GlobalTransform);
    const dx = transform.translation.x.get(eid) - listenerMatrix[12];
    const dy = transform.translation.y.get(eid) - listenerMatrix[13];
    const dz = transform.translation.z.get(eid) - listenerMatrix[14];
    const x = dx * listenerMatrix[0] + dy * listenerMatrix[1] + dz * listenerMatrix[2];
    const y = dx * listenerMatrix[4] + dy * listenerMatrix[5] + dz * listenerMatrix[6];
    const z = dx * listenerMatrix[8] + dy * listenerMatrix[9] + dz * listenerMatrix[10];
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return [
        Math.atan2(x, z),
        distance > 0.001 ? Math.asin(Math.max(-1, Math.min(1, y / distance))) : 0,
        distance,
    ];
}

function push(
    buffer: Float64Array,
    count: number,
    op: number,
    voice: Voice,
    spatial: boolean,
    az = 0,
    el = 0,
    distance = 0,
): number {
    const offset = 2 + count * STRIDE;
    buffer[offset] = op;
    buffer[offset + 1] = voice.handle;
    buffer[offset + 2] = voice.clip;
    buffer[offset + 3] = voice.loop ? 1 : 0;
    buffer[offset + 4] = voice.volume;
    buffer[offset + 5] = spatial ? 1 : 0;
    buffer[offset + 6] = az;
    buffer[offset + 7] = el;
    buffer[offset + 8] = distance;
    return count + 1;
}

function soundVolume(world: World, eid: number): number {
    const value = world.storage(Sound).volume.get(eid);
    return Number.isFinite(value) ? Math.max(0, value) : 0;
}

function allocate(
    runtime: Runtime,
    eid: number,
    ref: EntityRef,
    clip: number,
    loop: boolean,
    volume: number,
): Voice | undefined {
    const slot = runtime.free.pop();
    if (slot === undefined) return undefined;
    const generation = (runtime.generations[slot] + 1) & 0x1ffffff;
    runtime.generations[slot] = generation;
    const voice: Voice = {
        eid,
        ref,
        handle: generation * MAX_VOICES + slot,
        started: ++runtime.serial,
        clip,
        loop,
        volume,
        stopping: false,
    };
    runtime.owners.set(eid, voice);
    runtime.slots[slot] = voice;
    return voice;
}

function stopVoice(voice: Voice, buffer: Float64Array, count: number): number {
    if (voice.stopping) return count;
    voice.stopping = true;
    return push(buffer, count, STOP, voice, false);
}

function stealOldest(world: World, runtime: Runtime, buffer: Float64Array, count: number): number {
    let victim: Voice | undefined;
    for (const voice of runtime.owners.values()) {
        if (voice.stopping || voice.loop) continue;
        if (!victim || voice.started < victim.started) victim = voice;
    }
    if (!victim) return count;
    count = stopVoice(victim, buffer, count);
    if (world.ref(victim.eid) === victim.ref) {
        rememberCompleted(runtime, victim.ref);
        world.destroy(victim.eid);
    }
    return count;
}

/** Reconcile presentation voices against current ECS data; run after derived transforms each draw. */
export function reconcileAudio(world: World): void {
    const runtime = world.resource(AudioRuntime);
    const state = world.resource(Devices).audio.context;
    if (state !== "running") {
        if (state === "suspended") {
            for (const eid of [...world.query([Sound])]) {
                if (world.storage(Sound).loop.get(eid) === 0) world.destroy(eid);
            }
        }
        return;
    }
    if (!runtime.node || !runtime.buffer) return;
    uploadClips(world, runtime);

    const buffer = runtime.buffer;
    buffer[0] = 0;
    buffer[1] = 0;
    let count = 0;
    const listener = world.only([Listener, GlobalTransform]);

    for (const voice of runtime.owners.values()) {
        const eid = voice.eid;
        const ref = world.ref(eid);
        if (ref !== voice.ref || !world.has(eid, Sound)) {
            count = stopVoice(voice, buffer, count);
            continue;
        }
        if (runtime.completed.has(ref)) {
            world.destroy(eid);
            count = stopVoice(voice, buffer, count);
            continue;
        }
        const sounds = world.storage(Sound);
        const clip = sounds.clip.get(eid);
        const loop = sounds.loop.get(eid) !== 0;
        const paused = sounds.paused.get(eid) !== 0;
        if (paused || clip !== voice.clip) {
            count = stopVoice(voice, buffer, count);
            continue;
        }
        if (voice.stopping) continue;
        voice.loop = loop;
        const volume = soundVolume(world, eid);
        voice.volume = volume;
        const spatial = listener >= 0 && world.has(eid, GlobalTransform);
        const [az, el, distance] = spatial ? polar(world, eid, listener) : [0, 0, 0];
        count = push(buffer, count, UPDATE, voice, spatial, az, el, distance);
    }

    const sounds = world.storage(Sound);
    let stoleOneShot = false;
    for (const eid of [...world.query([Sound])]) {
        const ref = world.ref(eid);
        if (runtime.completed.has(ref)) {
            world.destroy(eid);
            continue;
        }
        if (runtime.owners.has(eid)) continue;
        const clip = sounds.clip.get(eid);
        const entry = world.resource(Clips).registry.name(clip);
        if (entry === undefined || !world.resource(Clips).registry.get(entry)) continue;
        const loop = sounds.loop.get(eid) !== 0;
        if (sounds.paused.get(eid) !== 0) continue;
        const voice = allocate(runtime, eid, ref, clip, loop, soundVolume(world, eid));
        if (!voice) {
            if (!stoleOneShot) {
                const next = stealOldest(world, runtime, buffer, count);
                stoleOneShot = next !== count;
                count = next;
            }
            continue;
        }
        const spatial = listener >= 0 && world.has(eid, GlobalTransform);
        const [az, el, distance] = spatial ? polar(world, eid, listener) : [0, 0, 0];
        count = push(buffer, count, PLAY, voice, spatial, az, el, distance);
    }

    buffer[0] = count;
    runtime.buffer = null;
    runtime.node.port.postMessage({ type: "batch", buffer: buffer.buffer }, [buffer.buffer]);
}

function releaseFinished(world: World, runtime: Runtime, handle: number): void {
    const slot = handle % MAX_VOICES;
    const voice = runtime.slots[slot];
    if (!voice || voice.handle !== handle) return;
    runtime.slots[slot] = null;
    runtime.free.push(slot);
    if (runtime.owners.get(voice.eid) === voice) runtime.owners.delete(voice.eid);
    const eid = world.resolve(voice.ref);
    if (
        eid > 0 &&
        !voice.stopping &&
        !voice.loop &&
        world.has(eid, Sound) &&
        world.storage(Sound).clip.get(eid) === voice.clip &&
        world.storage(Sound).loop.get(eid) === 0 &&
        world.storage(Sound).paused.get(eid) === 0
    ) {
        rememberCompleted(runtime, voice.ref);
        world.destroy(eid);
    }
}

/** Accept the worklet's returned command buffer and completed generation handles. */
export function receiveAudioMessage(world: World, data: any, ownedRuntime?: Runtime): void {
    if (world.disposed) return;
    const runtime = ownedRuntime ?? world.resource(AudioRuntime);
    if (data?.type === "recycle") {
        const buffer = new Float64Array(data.buffer as ArrayBuffer);
        const finished = buffer[1];
        for (let i = 0; i < finished; i++)
            releaseFinished(world, runtime, buffer[FINISHED_OFFSET + i]);
        runtime.buffer = buffer;
    } else if (data?.type === "heartbeat") {
        runtime.lastHeartbeat = performance.now();
    } else if (data?.type === "error" || data?.type === "init-error") {
        console.error(`audio worklet error: ${data.message}`);
    }
}

/** Build a returned-buffer event for a host test without a browser processor. */
export function returnAudioBuffer(
    world: World,
    buffer: Float64Array,
    finished: readonly number[] = [],
): void {
    buffer[1] = Math.min(finished.length, MAX_VOICES);
    for (let i = 0; i < buffer[1]; i++) buffer[FINISHED_OFFSET + i] = finished[i];
    receiveAudioMessage(world, { type: "recycle", buffer: buffer.buffer });
}

const AudioFrameSystem: System = {
    name: "audio-frame",
    group: "draw",
    after: [PrepareGlobalTransformSystem],
    update: reconcileAudio,
};

export const StandardAudioPlugin: Plugin = {
    name: "StandardAudio",
    dependencies: [AudioPlugin],
    systems: [AudioFrameSystem],
    async initialize(world) {
        const runtime = world.resource(AudioRuntime);
        const output = world.resource(AudioOutput);
        const devices = world.resource(Devices);
        world.onDispose(() => {
            void disposeAudio(world, runtime, output, devices);
        });
        try {
            await initializeAudio(world);
        } catch (error) {
            console.error("audio initialization failed:", error);
            await disposeAudio(world, runtime, output, devices);
        }
    },
};

/** Decode a URL or Blob and register its PCM in this World's clip registry. */
export async function loadClip(world: World, name: string, source: string | Blob): Promise<number> {
    const output = world.resource(AudioOutput);
    const Ctor = browserAudioContext();
    if (!output.context && !Ctor) throw new Error("AudioContext is not available to decode clips");
    const context = output.context ?? new Ctor!();
    const owned = context !== output.context;
    try {
        const bytes =
            typeof source === "string"
                ? await fetch(source).then((response) => {
                      if (!response.ok)
                          throw new Error(
                              `failed to fetch clip: ${response.status} ${response.statusText}`,
                          );
                      return response.arrayBuffer();
                  })
                : await source.arrayBuffer();
        const decoded = await context.decodeAudioData(bytes);
        const channels: Float32Array[] = [];
        for (let channel = 0; channel < decoded.numberOfChannels; channel++)
            channels.push(decoded.getChannelData(channel).slice());
        return registerClip(world, name, channels, decoded.sampleRate);
    } finally {
        if (owned) await context.close();
    }
}
