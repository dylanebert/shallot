import { resource } from "../../engine";
import type { AudioBackend, Readback } from "./backend";
import type { CompiledInstrument } from "./instrument";
import { instrumentRegistry, getParamPairs } from "./instrument";
import { flushSamples } from "./sample";

const MAX_VOICES = 64;
const MAX_TRANSPORTS = 8;

export interface PitchEntry {
    offset: number;
    baseFreq: number;
    octave: number;
    semitone: number;
    fine: number;
}

export interface ActiveVoice {
    slot: number;
    volumeOffset: number;
    baseVolume: number;
    pitchEntries: PitchEntry[];
}

export interface SoundState {
    voices: Map<number, ActiveVoice>;
    pending: Set<number>;
    systemRemovals: Set<number>;
}

interface VoiceState {
    instrumentId: number;
    gate: number;
}

export interface SpatialResult {
    azimuth: number;
    elevation: number;
    distance: number;
}

export interface AudioState {
    backend: AudioBackend | null;
    voiceFree: number[];
    voices: (VoiceState | null)[];
    registeredVersions: Map<number, number>;
    registeredSampleVersions: Map<number, number>;
    spatialBatch: Float32Array;
    spatialLen: number;
    acousticBatch: Float32Array;
    acousticLen: number;
    transportFree: number[];
    transportBeats: Map<number, number>;
    seekTimes: Map<number, number>;
    idleCallbacks: Map<number, () => void>;
    beatDecoder: DataView;
    warnedVoiceFull: boolean;
    warnedSpatialFull: boolean;
    voiceGen: number[];
    idleWatchGen: Map<number, number>;
}

const _spatialResult: SpatialResult = { azimuth: 0, elevation: 0, distance: 0 };

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
): SpatialResult {
    const localX = dx * rx + dy * ry + dz * rz;
    const localY = dx * ux + dy * uy + dz * uz;
    const localZ = dx * fx + dy * fy + dz * fz;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const azimuth = Math.atan2(localX, localZ);
    const elevation =
        distance > 0.001 ? Math.asin(Math.max(-1, Math.min(1, localY / distance))) : 0;
    _spatialResult.azimuth = azimuth;
    _spatialResult.elevation = elevation;
    _spatialResult.distance = distance;
    return _spatialResult;
}

export const Audio = resource<AudioState>("audio");
export const SoundVoices = resource<SoundState>("sound-voices");

export function createAudioState(): AudioState {
    const voiceFree: number[] = [];
    for (let i = MAX_VOICES - 1; i >= 0; i--) voiceFree.push(i);
    const transportFree: number[] = [];
    for (let i = MAX_TRANSPORTS - 1; i >= 0; i--) transportFree.push(i);
    return {
        backend: null,
        voiceFree,
        voices: new Array(MAX_VOICES).fill(null),
        registeredVersions: new Map(),
        registeredSampleVersions: new Map(),
        spatialBatch: new Float32Array(MAX_VOICES * 7),
        spatialLen: 0,
        acousticBatch: new Float32Array(MAX_VOICES * 5),
        acousticLen: 0,
        transportFree,
        transportBeats: new Map(),
        seekTimes: new Map(),
        idleCallbacks: new Map(),
        beatDecoder: new DataView(new ArrayBuffer(8)),
        warnedVoiceFull: false,
        warnedSpatialFull: false,
        voiceGen: new Array(MAX_VOICES).fill(0),
        idleWatchGen: new Map(),
    };
}

export async function initAudio(audio: AudioState): Promise<void> {
    if (audio.backend) return;

    audio.voiceFree.length = 0;
    for (let i = MAX_VOICES - 1; i >= 0; i--) audio.voiceFree.push(i);
    audio.voices.fill(null);
    audio.registeredVersions.clear();
    audio.registeredSampleVersions.clear();
    audio.idleCallbacks.clear();
    audio.warnedVoiceFull = false;
    audio.warnedSpatialFull = false;
    audio.voiceGen.fill(0);
    audio.idleWatchGen.clear();
    audio.transportFree.length = 0;
    for (let i = MAX_TRANSPORTS - 1; i >= 0; i--) audio.transportFree.push(i);
    audio.transportBeats.clear();

    const { WebBackend } = await import("./web-backend");
    audio.backend = new WebBackend();

    const handler: Readback = {
        onVoiceIdle(voiceId) {
            handleVoiceIdle(audio, voiceId);
        },
        onTransportBeat(tid, beatLo, beatHi) {
            handleTransportBeat(audio, tid, beatLo, beatHi);
        },
    };
    await audio.backend.init(handler);
}

export function started(audio: AudioState): boolean {
    return audio.backend !== null;
}

export function running(audio: AudioState): boolean {
    return audio.backend?.running ?? false;
}

export function tickAudio(audio: AudioState): void {
    audio.backend?.pollReadback();
    flushSamples(audio);
    audio.backend?.flush();
}

export function disposeAudio(audio: AudioState): void {
    audio.backend?.dispose();
    audio.backend = null;
    audio.idleCallbacks.clear();
    audio.transportBeats.clear();
    audio.seekTimes.clear();
}

export function handleVoiceIdle(audio: AudioState, voiceId: number): void {
    const watchGen = audio.idleWatchGen.get(voiceId);
    if (watchGen !== undefined && watchGen !== audio.voiceGen[voiceId]) {
        audio.idleCallbacks.delete(voiceId);
        audio.idleWatchGen.delete(voiceId);
        return;
    }
    audio.idleWatchGen.delete(voiceId);

    const cb = audio.idleCallbacks.get(voiceId);
    if (cb) {
        audio.idleCallbacks.delete(voiceId);
        cb();
        return;
    }

    if (!audio.voices[voiceId]) return;

    deactivateVoice(audio, voiceId);
    audio.voiceFree.push(voiceId);
}

export function handleTransportBeat(
    audio: AudioState,
    tid: number,
    beatLo: number,
    beatHi: number,
): void {
    const now = performance.now();
    const seekTime = audio.seekTimes.get(tid);
    if (seekTime !== undefined) {
        if (now - seekTime < 10) return;
        audio.seekTimes.delete(tid);
    }
    audio.beatDecoder.setUint32(0, beatLo, true);
    audio.beatDecoder.setUint32(4, beatHi, true);
    audio.transportBeats.set(tid, audio.beatDecoder.getFloat64(0, true));
}

export function gate(audio: AudioState, voiceId: number, value: number): void {
    if (!audio.backend) return;
    const voice = audio.voices[voiceId];
    if (voice && voice.gate === value) return;
    audio.backend.send({ type: "gate", voiceId, value });
    if (voice) voice.gate = value;
}

export function setParamDirect(
    audio: AudioState,
    voiceId: number,
    offset: number,
    value: number,
): void {
    if (!audio.backend) return;
    audio.backend.send({ type: "params", changes: [[voiceId, offset, value]] });
}

function sendParamChanges(audio: AudioState, changes: [number, number, number][]): void {
    if (!audio.backend || changes.length === 0) return;
    audio.backend.send({ type: "params", changes });
}

function registerInstrument(audio: AudioState, instId: number, data: CompiledInstrument): void {
    if (!audio.backend) return;
    if (audio.registeredVersions.get(instId) === data.version) return;
    audio.backend.send({
        type: "set_instrument",
        id: instId,
        nodeCount: data.nodes.length,
        outputBuf: data.outputBuf,
        nodes: data.nodes,
        modulations: data.modulations,
    });
    audio.registeredVersions.set(instId, data.version);
    const values = getParamPairs(instId);
    for (let i = 0; i < audio.voices.length; i++) {
        const v = audio.voices[i];
        if (v && v.instrumentId === instId) {
            audio.backend.send({
                type: "set_voice_instrument",
                voiceId: i,
                instrumentId: instId,
            });
            v.gate = -1;
            sendValues(audio, i, values);
        }
    }
}

function setVoiceInstrument(audio: AudioState, voiceId: number, instId: number): void {
    if (!audio.backend) return;
    audio.backend.send({ type: "set_voice_instrument", voiceId, instrumentId: instId });
    const voice = audio.voices[voiceId];
    if (voice) {
        voice.instrumentId = instId;
        voice.gate = -1;
    }
}

function sendValues(audio: AudioState, voiceId: number, pairs: [number, number][]): void {
    if (!audio.backend || pairs.length === 0) return;
    const changes: [number, number, number][] = pairs.map(([offset, value]) => [
        voiceId,
        offset,
        value,
    ]);
    audio.backend.send({ type: "params", changes });
}

function activateVoice(audio: AudioState, voiceId: number): void {
    audio.backend?.send({ type: "voice_active", voiceId, active: true });
}

function deactivateVoice(audio: AudioState, voiceId: number): void {
    audio.backend?.send({ type: "voice_active", voiceId, active: false });
    audio.voices[voiceId] = null;
}

export function alloc(audio: AudioState): number {
    if (audio.voiceFree.length === 0) {
        if (!audio.warnedVoiceFull) {
            console.warn("audio: voice pool full (64)");
            audio.warnedVoiceFull = true;
        }
        return -1;
    }
    const slot = audio.voiceFree.pop()!;
    audio.voiceGen[slot]++;
    audio.voices[slot] = {
        instrumentId: -1,
        gate: -1,
    };
    activateVoice(audio, slot);
    return slot;
}

export function free(audio: AudioState, slot: number): void {
    if (!audio.voices[slot]) return;
    deactivateVoice(audio, slot);
    audio.idleCallbacks.delete(slot);
    audio.voiceFree.push(slot);
    audio.warnedVoiceFull = false;
}

export function voiceGen(audio: AudioState, slot: number): number {
    return audio.voiceGen[slot];
}

export function onIdle(audio: AudioState, slot: number, cb: () => void): void {
    audio.idleCallbacks.set(slot, cb);
    audio.idleWatchGen.set(slot, audio.voiceGen[slot]);
    audio.backend?.send({ type: "watch_idle", voiceId: slot });
}

export function allocTransport(audio: AudioState): number {
    if (audio.transportFree.length === 0) return -1;
    return audio.transportFree.pop()!;
}

export function freeTransport(audio: AudioState, tid: number): void {
    stop(audio, tid);
    clearEvents(audio, tid);
    audio.transportBeats.delete(tid);
    audio.seekTimes.delete(tid);
    audio.transportFree.push(tid);
}

export function beat(audio: AudioState, tid: number): number {
    return audio.transportBeats.get(tid) ?? 0;
}

export function addSpatial(
    audio: AudioState,
    slot: number,
    az: number,
    el: number,
    dist: number,
    ref: number,
    max: number,
    roll: number,
): void {
    if (audio.spatialLen + 7 > audio.spatialBatch.length) {
        if (!audio.warnedSpatialFull) {
            console.warn("audio: spatial batch full");
            audio.warnedSpatialFull = true;
        }
        return;
    }
    audio.spatialBatch[audio.spatialLen++] = slot;
    audio.spatialBatch[audio.spatialLen++] = az;
    audio.spatialBatch[audio.spatialLen++] = el;
    audio.spatialBatch[audio.spatialLen++] = dist;
    audio.spatialBatch[audio.spatialLen++] = ref;
    audio.spatialBatch[audio.spatialLen++] = max;
    audio.spatialBatch[audio.spatialLen++] = roll;
}

export function flushSpatial(audio: AudioState): void {
    if (!audio.backend || audio.spatialLen === 0) return;
    audio.backend.send({ type: "spatial", data: audio.spatialBatch, len: audio.spatialLen });
    audio.spatialLen = 0;
}

export function addAcousticSeparate(
    audio: AudioState,
    slot: number,
    occlusion: number,
    transLow: number,
    transMid: number,
    transHigh: number,
): void {
    if (audio.acousticLen + 5 > audio.acousticBatch.length) return;
    audio.acousticBatch[audio.acousticLen++] = slot;
    audio.acousticBatch[audio.acousticLen++] = occlusion;
    audio.acousticBatch[audio.acousticLen++] = transLow;
    audio.acousticBatch[audio.acousticLen++] = transMid;
    audio.acousticBatch[audio.acousticLen++] = transHigh;
}

export function flushAcoustic(audio: AudioState): void {
    if (!audio.backend || audio.acousticLen === 0) return;
    audio.backend.send({ type: "acoustic", data: audio.acousticBatch, len: audio.acousticLen });
    audio.acousticLen = 0;
}

export function setReflectionIR(audio: AudioState, slot: number, ir: Float32Array): void {
    audio.backend?.send({ type: "reflectionIR", voiceId: slot, ir, irLen: ir.length });
}

export function setReflectionGain(audio: AudioState, slot: number, gain: number): void {
    audio.backend?.send({ type: "reflectionGain", voiceId: slot, gain });
}

export function setRealVoiceBudget(audio: AudioState, budget: number): void {
    audio.backend?.send({ type: "set_budget", budget });
}

export function setReverb(
    audio: AudioState,
    rt60: [number, number, number],
    wetGain: number,
    eq: [number, number, number],
): void {
    audio.backend?.send({
        type: "reverb",
        rt60Low: rt60[0],
        rt60Mid: rt60[1],
        rt60High: rt60[2],
        wetGain,
        eqLow: eq[0],
        eqMid: eq[1],
        eqHigh: eq[2],
    });
}

export function play(audio: AudioState, tid: number): void {
    audio.backend?.send({ type: "transport_play", tid });
}

export function stop(audio: AudioState, tid: number): void {
    audio.backend?.send({ type: "transport_stop", tid });
}

export function pause(audio: AudioState, tid: number): void {
    audio.backend?.send({ type: "transport_pause", tid });
}

export function setBPM(audio: AudioState, tid: number, bpm: number): void {
    audio.backend?.send({ type: "transport_set_bpm", tid, bpm });
}

export function schedule(
    audio: AudioState,
    tid: number,
    beat: number,
    voiceId: number,
    durationBeats: number,
    params?: [number, number][],
): void {
    audio.backend?.send({
        type: "transport_queue_event",
        tid,
        beat,
        voiceId,
        durationBeats,
        p0Off: params?.[0]?.[0] ?? 0,
        p0Val: params?.[0]?.[1] ?? 0,
        p1Off: params?.[1]?.[0] ?? 0,
        p1Val: params?.[1]?.[1] ?? 0,
        p2Off: params?.[2]?.[0] ?? 0,
        p2Val: params?.[2]?.[1] ?? 0,
        p3Off: params?.[3]?.[0] ?? 0,
        p3Val: params?.[3]?.[1] ?? 0,
        paramCount: params?.length ?? 0,
    });
}

export function clearEvents(audio: AudioState, tid: number): void {
    audio.backend?.send({ type: "transport_clear_events", tid });
}

export function setLoop(audio: AudioState, tid: number, length: number): void {
    audio.backend?.send({ type: "transport_set_loop", tid, length });
}

export function setVoiceSpatial(audio: AudioState, slot: number, spatial: boolean): void {
    audio.backend?.send({ type: "voice_spatial", voiceId: slot, spatial });
}

export function setVoiceOneShot(audio: AudioState, slot: number): void {
    audio.backend?.send({ type: "voice_one_shot", voiceId: slot });
}

export function seek(audio: AudioState, tid: number, beat: number): void {
    audio.transportBeats.set(tid, beat);
    audio.seekTimes.set(tid, performance.now());
    audio.backend?.send({ type: "transport_seek", tid, beat });
}

export function upload(audio: AudioState, instId: number, values: Record<string, number>): void {
    const compiled = instrumentRegistry.get(instId);
    if (!compiled || !audio.backend) return;
    const changes: [number, number, number][] = [];
    for (const [key, value] of Object.entries(values)) {
        const entry = compiled.paramLayout.get(key);
        if (entry === undefined) continue;
        for (let i = 0; i < audio.voices.length; i++) {
            if (audio.voices[i]?.instrumentId === instId) changes.push([i, entry, value]);
        }
    }
    sendParamChanges(audio, changes);
}

export function setParam(
    audio: AudioState,
    voiceId: number,
    paramKey: string,
    value: number,
    instId: number,
): void {
    const compiled = instrumentRegistry.get(instId);
    if (!compiled) return;
    const entry = compiled.paramLayout.get(paramKey);
    if (entry === undefined) return;
    setParamDirect(audio, voiceId, entry, value);
}

export function refresh(audio: AudioState, instId: number): void {
    const compiled = instrumentRegistry.get(instId);
    if (!compiled) return;
    registerInstrument(audio, instId, compiled);
}

export function assign(audio: AudioState, voiceId: number, instId: number): void {
    const compiled = instrumentRegistry.get(instId);
    if (!compiled) return;
    registerInstrument(audio, instId, compiled);
    setVoiceInstrument(audio, voiceId, instId);
    sendValues(audio, voiceId, getParamPairs(instId));
}

export { instrumentRegistry, getValues, getParamPairs, setValues } from "./instrument";
export { patternRegistry, pattern, noteFreq, midiFreq } from "./pattern";
export { evalCurve } from "./curve";
export type { CurveMapping } from "./curve";
