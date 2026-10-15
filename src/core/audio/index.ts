import {
    component,
    f32,
    i32,
    type Plugin,
    Registry,
    type Resource,
    u8,
    type World,
} from "../../engine";
import { GlobalTransform, Transform, TransformPlugin } from "../transform";

const MAX_CLIPS = 256;

/** Decoded mono or stereo PCM owned by one World's clip registry. */
export interface Clip {
    readonly name: string;
    readonly channels: readonly Float32Array[];
    readonly sampleRate: number;
    readonly version: number;
}

interface ClipStore {
    readonly registry: Registry<Clip>;
    nextId: number;
    nextVersion: number;
}

export const Clips: Resource<ClipStore> = {
    create: () => ({ registry: new Registry<Clip>(), nextId: 0, nextVersion: 1 }),
};

/** Playback data, with no voice or device state.
 * On restore, a once Sound survives only when its same runtime voice is still playing; otherwise
 * it is discarded. A once Sound created after the restore plays normally.
 */
export const Sound = component(
    "Sound",
    {
        /** ID returned by {@link registerClip}. */
        clip: i32,
        /** 0 plays once and despawns; 1 repeats until the component is removed. */
        loop: u8,
        /** Linear playback gain. */
        volume: f32,
        /** A paused sound does not start; pausing a live voice fades it to silence. */
        paused: u8,
    },
    { defaults: () => ({ clip: -1, loop: 0, volume: 1, paused: 0 }) },
);

/** Marks the entity whose world transform defines the listener's position and orientation. */
export const Listener = component("Listener", {}, {});

/** Audio insertion point. The running audio implementation supplies its context and bus. */
export interface AudioOutputValue {
    context: AudioContext | null;
    /** Connect processor outputs here; the running processor receives the mix at its input. */
    bus: AudioNode | null;
}

export const AudioOutput: Resource<AudioOutputValue> = {
    create: () => ({ context: null, bus: null }),
};

/** Register decoded PCM in this World's clip registry, returning its stable clip ID or -1 at capacity.
 * Re-registering a name replaces its PCM and advances its version. Mono and stereo are retained; wider
 * sources are downmixed to mono. PCM is copied so later caller mutation cannot change a registered clip.
 */
export function registerClip(
    world: World,
    name: string,
    pcm: Float32Array | readonly Float32Array[],
    sampleRate = 48000,
): number {
    if (!name) throw new Error("clip name must not be empty");
    if (!Number.isFinite(sampleRate) || sampleRate <= 0)
        throw new Error("sampleRate must be positive");
    const source = pcm instanceof Float32Array ? [pcm] : [...pcm];
    if (source.length === 0 || source.some((channel) => !(channel instanceof Float32Array))) {
        throw new Error("clip PCM must contain at least one Float32Array channel");
    }
    const length = source[0].length;
    if (length === 0) throw new Error("clip PCM must not be empty");
    if (source.some((channel) => channel.length !== length)) {
        throw new Error("clip PCM channels must have equal lengths");
    }
    const channels =
        source.length <= 2 ? source.map((channel) => channel.slice()) : [downmix(source, length)];
    const store = world.resource(Clips);
    const id = store.registry.id(name);
    if (id === undefined && store.nextId >= MAX_CLIPS) return -1;
    const registered = store.registry.register({
        name,
        channels,
        sampleRate,
        version: store.nextVersion++,
    });
    if (id === undefined) store.nextId++;
    return registered;
}

function downmix(channels: readonly Float32Array[], length: number): Float32Array {
    const mono = new Float32Array(length);
    const scale = 1 / channels.length;
    for (const channel of channels) {
        for (let i = 0; i < length; i++) mono[i] += channel[i] * scale;
    }
    return mono;
}

/** Start a registered clip and return its entity ID, or -1 when the clip is unknown. `position`
 * adds a `Transform`; it is spatial when a `Listener` with derived placement exists.
 */
export function play(
    world: World,
    clip: string | number,
    options: { loop?: boolean; volume?: number; position?: readonly [number, number, number] } = {},
): number {
    const clips = world.resource(Clips).registry;
    const id = typeof clip === "string" ? clips.id(clip) : clip;
    const name = id === undefined ? undefined : clips.name(id);
    if (id === undefined || id < 0 || name === undefined || !clips.get(name)) return -1;
    const eid = world.create();
    world.add(eid, Sound);
    const sound = world.storage(Sound);
    sound.clip.set(eid, id);
    sound.loop.set(eid, options.loop ? 1 : 0);
    if (options.volume !== undefined) sound.volume.set(eid, options.volume);
    if (options.position !== undefined) {
        world.add(eid, GlobalTransform);
        world.add(eid, Transform);
        world
            .storage(Transform)
            .translation.set(eid, options.position[0], options.position[1], options.position[2]);
    }
    return eid;
}

/** Owns playback data and its transform dependency; the implementation is optional. */
export const AudioPlugin: Plugin = {
    name: "Audio",
    dependencies: [TransformPlugin],
    components: [Sound, Listener],
    initialize(world) {
        world.resource(Clips);
        world.resource(AudioOutput);
    },
};
