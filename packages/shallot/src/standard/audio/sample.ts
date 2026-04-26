import { registry, type Registry } from "../../engine";
import type { AudioState } from "./engine";

export const MAX_SAMPLES = 256;

interface SampleEntry {
    data: Float32Array | null;
    sampleRate: number;
    version: number;
}

let nextVersion = 1;

export const sampleRegistry: Registry<SampleEntry> = registry(MAX_SAMPLES);

const _pending = new Map<number, Promise<void>>();

let _decodeCtx: AudioContext | null = null;
function getDecodeCtx(): AudioContext {
    if (_decodeCtx) return _decodeCtx;
    const g = globalThis as Record<string, unknown>;
    const Ctor = (g.AudioContext ?? g.webkitAudioContext) as (new () => AudioContext) | undefined;
    if (!Ctor) throw new Error("AudioContext not available in this environment");
    _decodeCtx = new Ctor();
    return _decodeCtx;
}

export function downmixToMono(channels: Float32Array[], length: number): Float32Array {
    if (channels.length === 1) return new Float32Array(channels[0]);
    const mono = new Float32Array(length);
    for (const src of channels) {
        for (let i = 0; i < length; i++) mono[i] += src[i];
    }
    const inv = 1 / channels.length;
    for (let i = 0; i < length; i++) mono[i] *= inv;
    return mono;
}

async function decodeToMono(buf: ArrayBuffer): Promise<{ data: Float32Array; sampleRate: number }> {
    const decoded = await getDecodeCtx().decodeAudioData(buf);
    const channels: Float32Array[] = [];
    for (let c = 0; c < decoded.numberOfChannels; c++) channels.push(decoded.getChannelData(c));
    return { data: downmixToMono(channels, decoded.length), sampleRate: decoded.sampleRate };
}

async function loadInto(entry: SampleEntry, source: string | Blob): Promise<void> {
    try {
        const buf =
            typeof source === "string"
                ? await fetch(source).then((r) => {
                      if (!r.ok) {
                          throw new Error(`failed to fetch sample: ${r.status} ${r.statusText}`);
                      }
                      return r.arrayBuffer();
                  })
                : await source.arrayBuffer();
        const { data, sampleRate } = await decodeToMono(buf);
        entry.data = data;
        entry.sampleRate = sampleRate;
        entry.version = nextVersion++;
    } catch (e) {
        console.warn("[audio] sample load failed:", e);
    }
}

/**
 * register a PCM audio sample for use by Sample nodes
 * @example
 * const id = sample("/boom-hit.mp3", "boom");
 */
export function sample(source: Float32Array | string | Blob, name?: string): number {
    if (source instanceof Float32Array) {
        const entry: SampleEntry = { data: source, sampleRate: 0, version: nextVersion++ };
        return sampleRegistry.add(entry, name);
    }
    const entry: SampleEntry = { data: null, sampleRate: 0, version: 0 };
    const id = sampleRegistry.add(entry, name);
    let wrapped: Promise<void>;
    wrapped = loadInto(entry, source).finally(() => {
        if (_pending.get(id) === wrapped) _pending.delete(id);
    });
    _pending.set(id, wrapped);
    return id;
}

/** look up a sample id by name */
export function getSampleByName(name: string): number | undefined {
    return sampleRegistry.getByName(name);
}

/** look up a sample's decoded buffer and sample rate by id; data is null until decode completes */
export function getSample(
    id: number,
): { data: Float32Array | null; sampleRate: number } | undefined {
    const entry = sampleRegistry.get(id);
    if (!entry) return undefined;
    return { data: entry.data, sampleRate: entry.sampleRate };
}

/** await pending decode for a sample id; resolves immediately if already loaded */
export function whenLoaded(id: number): Promise<void> {
    return _pending.get(id) ?? Promise.resolve();
}

export function flushSamples(audio: AudioState): void {
    if (!audio.backend) return;
    const all = sampleRegistry.all();
    for (let i = 0; i < all.length; i++) {
        const entry = all[i];
        if (!entry.data) continue;
        if (audio.registeredSampleVersions.get(i) === entry.version) continue;
        audio.backend.send({ type: "set_sample", id: i, data: entry.data });
        audio.registeredSampleVersions.set(i, entry.version);
    }
}
