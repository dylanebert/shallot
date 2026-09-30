// Destination: core/audio and standard/audio; owner: audio-boundary.md.

import { Devices } from "../../core/input";
import {
    composeGlobalTransform,
    f32,
    GlobalTransform,
    i32,
    not,
    type Plugin,
    type System,
    Transform,
    u8,
    type World,
} from "../../engine";
import {
    Audio,
    addSpatial,
    alloc,
    assign,
    disposeAudio,
    flushSpatial,
    free,
    gate,
    initAudio,
    noteFreq,
    oneShot,
    polar,
    setParam,
    spatialize,
    tickAudio,
    watchIdle,
} from "./device";
import { byId, Instruments, instrument } from "./instrument";
import { markCooldown, policyFor, type SfxPolicy, withinCooldown } from "./policy";
import { getSample, Samples } from "./sample";

/**
 * a playing sound, mirroring Bevy's `AudioPlayer` + `PlaybackSettings`: add it
 * to start, remove it to stop a loop. `instrument` is a registered instrument
 * id; `loop` 0 = one-shot (frees itself when the envelope idles), 1 = loop;
 * `volume` (0–1, quadratic) and `pitch` (semitones, oscillator instruments
 * only) firehose live to the voice. `voice` is the allocated voice handle,
 * managed by {@link SoundSystem}; don't write it
 */
export const Sound = {
    /** registered instrument or sample name, resolved to an id (a bare sample auto-wraps a sampler) */
    instrument: i32,
    /** 0 = one-shot (frees itself when the envelope idles), 1 = loop until the `Sound` is removed */
    loop: u8,
    /** playback level 0–1, quadratic; firehoses live to the voice */
    volume: f32,
    /** pitch offset in semitones (oscillator instruments only); firehoses live to the voice */
    pitch: f32,
    /** allocated voice handle, managed by the audio system. read-only, don't author */
    voice: i32,
};

/** marks the spatial listener entity. its `Transform` orients the FOA + HRTF render */
export const Listener = {};

// a Sound that holds a live voice. Presence is the liveness signal (Bevy's
// inserted `AudioSink`): `[Sound, not(Voiced)]` needs a voice, `[Voiced,
// not(Sound)]` is a stopped loop to free. Internal — consumers never touch it
const Voiced = {};

const _m = new Float32Array(16);

/**
 * resolve a name to an instrument id. A registered instrument wins; otherwise a
 * registered sample lazily registers a built-in sampler instrument (cached
 * under the same name) so the trivial case authors no DAG
 */
function resolveInstrument(world: World, name: string): number | undefined {
    const inst = Instruments.id(name);
    if (inst !== undefined) return inst;
    const sampleId = Samples.id(name);
    if (sampleId === undefined) {
        console.warn(`audio: no instrument or sample named "${name}"`);
        return undefined;
    }
    return sampler(world, name, sampleId);
}

// sustain=1 + decay≈sample length makes the envelope a duration timer: the
// sample plays in full, then the one-shot gate-off + 50ms release frees the
// voice cleanly (juice's registerOneShot shape). For a loop the same envelope
// holds sustain and the sample wraps until the Sound is removed. decay falls
// back to 2s for a sample still decoding at first play
function sampler(world: World, name: string, sampleId: number): number {
    const s = getSample(sampleId);
    const frames = s?.channels[0]?.length ?? 0;
    const dur = frames
        ? frames / (s!.sampleRate || world.resource(Audio).ctx?.sampleRate || 48000)
        : 2;
    const env = { attack: 0.001, decay: dur, sustain: 1, release: 0.05 };
    if ((s?.channels.length ?? 1) >= 2) {
        // two parallel mono chains, one per channel, kept in lockstep by the shared
        // gate + rate; left → bus-left, right → bus-right (a positional voice
        // downmixes 0.5*(L+R) in the kernel). The baked stereo width is preserved
        return instrument(
            {
                nodes: {
                    srcL: { type: "sample" },
                    envL: { type: "envelope", input: "srcL" },
                    volL: { type: "gain", input: "envL" },
                    srcR: { type: "sample" },
                    envR: { type: "envelope", input: "srcR" },
                    volR: { type: "gain", input: "envR" },
                },
                output: "volL",
                outputR: "volR",
                volumeParam: ["volL.level", "volR.level"],
                loopParam: ["srcL.loop", "srcR.loop"],
                values: {
                    "srcL.bufferId": sampleId,
                    "srcL.rate": 1,
                    "srcL.loop": 0,
                    "srcL.channel": 0,
                    "srcR.bufferId": sampleId,
                    "srcR.rate": 1,
                    "srcR.loop": 0,
                    "srcR.channel": 1,
                    "envL.attack": env.attack,
                    "envL.decay": env.decay,
                    "envL.sustain": env.sustain,
                    "envL.release": env.release,
                    "envR.attack": env.attack,
                    "envR.decay": env.decay,
                    "envR.sustain": env.sustain,
                    "envR.release": env.release,
                    "volL.level": 1,
                    "volR.level": 1,
                },
            },
            name,
        );
    }
    return instrument(
        {
            nodes: {
                src: { type: "sample" },
                env: { type: "envelope", input: "src" },
                vol: { type: "gain", input: "env" },
            },
            output: "vol",
            volumeParam: "vol.level",
            loopParam: "src.loop",
            values: {
                "src.bufferId": sampleId,
                "src.rate": 1,
                "src.loop": 0,
                "src.volume": 1,
                "env.attack": env.attack,
                "env.decay": env.decay,
                "env.sustain": env.sustain,
                "env.release": env.release,
                "vol.level": 1,
            },
        },
        name,
    );
}

// free the least-churned voice to make room when the pool is exhausted. Loops
// (the crowd bed / music) are lifecycle-owned — stopped only by remove(Sound),
// never culled — so a one-shot burst never steals the bed. A graceful default
// (no crash); a Sound.priority bias is the FMOD-style follow-up
function steal(world: World): void {
    let victim = -1;
    let oldest = Number.POSITIVE_INFINITY;
    for (const eid of world.query([Sound, Voiced])) {
        if (world.storage(Sound).loop.get(eid) === 1) continue;
        const gen = world.storage(Sound).voice.get(eid) >>> 7;
        if (gen < oldest) {
            oldest = gen;
            victim = eid;
        }
    }
    if (victim < 0) return;
    free(world, world.storage(Sound).voice.get(victim));
    world.destroy(victim);
}

// apply a name's SFX policy before play() spawns. Drop inside the cooldown
// window; at the instance cap either cull a playing instance (oldest by voice
// generation, quietest by volume) or drop. The count is over pending + voiced
// of the same instrument, so a same-frame burst is bounded; only voiced
// instances are cull victims (a pending one isn't sounding yet). A culled
// victim drops Sound — SoundSystem gates it off + frees it on the next tick,
// the graceful loop-stop path (the cap is well under the pool, so a slot is
// free; no hard cut needed)
function admit(world: World, name: string, policy: Required<SfxPolicy>, id: number): boolean {
    if (withinCooldown(world, name, policy.cooldown, world.time.elapsed)) return false;
    if (policy.max > 0) {
        let count = 0;
        let victim = -1;
        let best = Number.POSITIVE_INFINITY;
        for (const eid of world.query([Sound])) {
            if (world.storage(Sound).instrument.get(eid) !== id) continue;
            count++;
            if (!world.has(eid, Voiced)) continue;
            const key =
                policy.steal === "quietest"
                    ? world.storage(Sound).volume.get(eid)
                    : world.storage(Sound).voice.get(eid) >>> 7;
            if (key < best) {
                best = key;
                victim = eid;
            }
        }
        if (count >= policy.max) {
            if (policy.steal === "none" || victim < 0) return false;
            world.remove(victim, Sound);
        }
    }
    if (policy.cooldown > 0) markCooldown(world, name, world.time.elapsed);
    return true;
}

/**
 * two reactive queries + the firehose + spatial derivation. Voice liveness is
 * the `Voiced` marker; the kernel owns DSP, this owns allocation and the
 * per-frame param feed
 */
const SoundSystem: System = {
    name: "sound",
    group: "simulation",
    update(world) {
        const context = world.resource(Devices).audio.context;
        if (context === "none" || context === "closed") return;

        // a loop stopped (Sound removed, Voiced kept): gate off, free after the
        // release tail. sparse fields survive remove, so Sound.voice still reads
        for (const eid of [...world.query([Voiced, not(Sound)])]) {
            const handle = world.storage(Sound).voice.get(eid);
            gate(world, handle, 0);
            watchIdle(world, handle, () => free(world, handle));
            world.remove(eid, Voiced);
        }

        const ctxRunning = context === "running";
        const listenerEid = world.only([Listener, GlobalTransform]);
        const hasListener = listenerEid >= 0;

        for (const eid of [...world.query([Sound, not(Voiced)])]) {
            const loop = world.storage(Sound).loop.get(eid);
            if (!ctxRunning) {
                // suspended (no user gesture): drop one-shots so they don't burst
                // on resume; loops stay pending until the context runs
                if (loop === 0) world.destroy(eid);
                continue;
            }
            const id = world.storage(Sound).instrument.get(eid);
            if (id < 0) continue;
            let handle = alloc(world);
            if (handle < 0) {
                steal(world);
                handle = alloc(world);
            }
            if (handle < 0) continue;
            assign(world, handle, id);
            const inst = byId(id);
            if (inst) for (const off of inst.loopOffsets) setParam(world, handle, off, loop);
            spatialize(world, handle, hasListener && world.has(eid, GlobalTransform));
            gate(world, handle, 1);
            if (loop === 0) {
                // gate-on must precede the idle watch: the worklet clears a slot
                // from _releasing on any gate(value != 0) (the re-gate-cancels-
                // death-watch guard), so a watch_idle queued before the initial
                // gate-on is wiped in the same batch — the voice then never
                // reports idle and leaks until steal reclaims it
                oneShot(world, handle);
                watchIdle(world, handle, () => {
                    free(world, handle);
                    if (world.exists(eid)) world.destroy(eid);
                });
            }
            world.storage(Sound).voice.set(eid, handle);
            world.add(eid, Voiced);
        }

        for (const eid of world.query([Sound, Voiced])) {
            const inst = byId(world.storage(Sound).instrument.get(eid));
            if (!inst) continue;
            const handle = world.storage(Sound).voice.get(eid);
            if (inst.volumeOffsets.length > 0) {
                const v = world.storage(Sound).volume.get(eid);
                const level = v * v * inst.baseVolume;
                for (const off of inst.volumeOffsets) setParam(world, handle, off, level);
            }
            if (inst.pitchEntries.length > 0) {
                const semis = world.storage(Sound).pitch.get(eid);
                for (const pe of inst.pitchEntries) {
                    const freq = noteFreq(pe.baseFreq, pe.octave, semis + pe.semitone, pe.fine);
                    setParam(world, handle, pe.offset, freq);
                }
            }
        }

        // spatial: a positioned voice + a listener pans + attenuates; the polar
        // derivation reads the listener's world basis (column-major right/up/fwd)
        if (hasListener) {
            const m = composeGlobalTransform(world, listenerEid, _m);
            for (const eid of world.query([Sound, Voiced, GlobalTransform])) {
                const dx = world.storage(GlobalTransform).translation.x.get(eid) - m[12];
                const dy = world.storage(GlobalTransform).translation.y.get(eid) - m[13];
                const dz = world.storage(GlobalTransform).translation.z.get(eid) - m[14];
                const p = polar(dx, dy, dz, m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]);
                addSpatial(
                    world,
                    world.storage(Sound).voice.get(eid),
                    p.azimuth,
                    p.elevation,
                    p.distance,
                );
            }
            flushSpatial(world);
        }

        tickAudio(world);
    },
};

/**
 * play a sound by instrument or sample name, returning the spawned entity id.
 * The event-driven path (a data constructor: `create` + `add(Sound)` + field
 * writes); scenes declare static / looping audio with `<a sound transform>`. A
 * sample name auto-registers a sampler, so the trivial case needs no
 * instrument. `pos` adds a `Transform`, making it spatial when a `Listener`
 * exists. Stop a loop with `world.remove(eid, Sound)`, not a bare destroy;
 * destroying a looping voice directly orphans its slot until it's stolen. A
 * name with a registered {@link sfx} policy is capped / cooled / stolen-from
 * first; a dropped trigger returns -1 like an unresolved name
 * @example
 * play(world, "explosion", { pos: [x, y, z] });
 */
export function play(
    world: World,
    name: string,
    opts?: { loop?: boolean; volume?: number; pos?: readonly [number, number, number] },
): number {
    const id = resolveInstrument(world, name);
    if (id === undefined) return -1;
    const policy = policyFor(name);
    if (policy && !admit(world, name, policy, id)) return -1;
    const eid = world.create();
    world.add(eid, Sound);
    world.storage(Sound).instrument.set(eid, id);
    world.storage(Sound).loop.set(eid, opts?.loop ? 1 : 0);
    if (opts?.volume !== undefined) world.storage(Sound).volume.set(eid, opts.volume);
    if (opts?.pos) {
        world.add(eid, Transform);
        world.storage(Transform).translation.set(eid, opts.pos[0], opts.pos[1], opts.pos[2], 0);
    }
    return eid;
}

export type { InstrumentDef, ModulationDef, NodeDef, NodeType } from "./instrument";
export { type SfxPolicy, sfx } from "./policy";
export { sample } from "./sample";
export { Instruments, instrument };

/**
 * procedural audio: the `Sound` + `Listener` components and the voice allocator over the WASM synth
 * kernel. Declare a sound with `<a sound>` or spawn one with {@link play}; mark the spatial reference with
 * `Listener` on the camera. Opt-in, not in the default plugin set.
 */
export const AudioPlugin: Plugin = {
    name: "Audio",
    components: { Sound, Listener, Voiced },
    systems: [SoundSystem],
    traits: {
        Sound: {
            defaults: () => ({ instrument: -1, loop: 0, volume: 1, pitch: 0, voice: -1 }),
            parse: { instrument: (name, world) => resolveInstrument(world, name) },
        },
        Listener: { requires: [GlobalTransform] },
    },
    async initialize(world) {
        // the whole audio teardown (worklet + context + host listeners + heartbeat) rides the World's
        // lifetime — registered up front so a partial init that then throws still tears down; disposeAudio
        // is idempotent, so the top-of-initAudio reinit and this dispose can't double-free.
        const audio = world.resource(Audio);
        const facts = world.resource(Devices).audio;
        world.onDispose(() => disposeAudio(world, audio, facts));
        try {
            await initAudio(world);
        } catch {
            // no AudioContext (headless / unsupported) — SoundSystem stays inert
        }
    },
};
