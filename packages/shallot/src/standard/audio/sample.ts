import { Registry } from "../../engine";

/** the sample registry's capacity. registering past it warns and drops the sample. */
export const MAX_SAMPLES = 256;

interface SampleEntry {
    name: string;
    // empty until decoded; length 1 = mono, 2 = stereo (>2-channel files collapse to mono)
    channels: Float32Array[];
    sampleRate: number;
    version: number;
}

let _nextVersion = 1;
let _anon = 0;

/** every registered sample, keyed by name with a stable numeric ID */
export const Samples: Registry<SampleEntry> = new Registry<SampleEntry>();

const _pending = new Map<number, Promise<void>>();
// sample id → version last sent to the worklet; cleared on audio re-init so a
// fresh worklet re-receives every decoded sample
const _sent = new Map<number, number>();

let _decodeCtx: AudioContext | null = null;
function decodeCtx(): AudioContext {
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

// the sampler preserves stereo (a 2D bed / music track keeps its baked width); a
// positional voice downmixes to mono in the kernel. >2 channels have no defined 2D
// placement, so they collapse — multichannel game assets are rare
export function keepChannels(channels: Float32Array[], length: number): Float32Array[] {
    if (channels.length <= 2) return channels.map((c) => new Float32Array(c));
    return [downmixToMono(channels, length)];
}

async function decode(buf: ArrayBuffer): Promise<{ channels: Float32Array[]; sampleRate: number }> {
    const decoded = await decodeCtx().decodeAudioData(buf);
    const channels: Float32Array[] = [];
    for (let c = 0; c < decoded.numberOfChannels; c++) channels.push(decoded.getChannelData(c));
    return { channels: keepChannels(channels, decoded.length), sampleRate: decoded.sampleRate };
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
        const { channels, sampleRate } = await decode(buf);
        entry.channels = channels;
        entry.sampleRate = sampleRate;
        entry.version = _nextVersion++;
    } catch (e) {
        console.warn("[audio] sample load failed:", e);
    }
}

/**
 * register a PCM audio sample for use by sample nodes. A URL string or Blob
 * decodes in the background (mp3 / wav / ogg / flac, stereo preserved); a
 * `Float32Array` registers immediately as raw mono PCM. The id returns
 * synchronously; playing before decode finishes plays silence
 * @example
 * const id = sample("/boom-hit.mp3", "boom");
 */
export function sample(source: Float32Array | string | Blob, name?: string): number {
    const n = name ?? `sample-${_anon++}`;
    if (source instanceof Float32Array) {
        return Samples.register({
            name: n,
            channels: [source],
            sampleRate: 0,
            version: _nextVersion++,
        });
    }
    const entry: SampleEntry = { name: n, channels: [], sampleRate: 0, version: 0 };
    const id = Samples.register(entry);
    const wrapped = loadInto(entry, source).finally(() => {
        if (_pending.get(id) === wrapped) _pending.delete(id);
    });
    _pending.set(id, wrapped);
    return id;
}

/** decoded per-channel buffers + sample rate for a sample id; `channels` is empty until decode completes */
export function getSample(
    id: number,
): { channels: Float32Array[]; sampleRate: number } | undefined {
    const name = Samples.name(id);
    const entry = name === undefined ? undefined : Samples.get(name);
    return entry ? { channels: entry.channels, sampleRate: entry.sampleRate } : undefined;
}

/** await a sample's pending decode; resolves immediately if already loaded */
export function whenLoaded(id: number): Promise<void> {
    return _pending.get(id) ?? Promise.resolve();
}

/** clear the sent-version tracking so a fresh worklet re-receives every sample */
export function resetSampleUploads(): void {
    _sent.clear();
}

/** send each newly-decoded sample to the worklet once, one call per channel, via `emit` */
export function flushSamples(
    emit: (id: number, channel: number, channels: number, data: Float32Array) => void,
): void {
    for (const entry of Samples) {
        if (entry.channels.length === 0) continue;
        const id = Samples.id(entry.name)!;
        if (_sent.get(id) === entry.version) continue;
        const count = entry.channels.length;
        for (let c = 0; c < count; c++) emit(id, c, count, entry.channels[c]);
        _sent.set(id, entry.version);
    }
}
