import { onAdd, onRemove, traits, buf, type State, type System, type Plugin } from "../../engine";
import { createFieldProxy } from "../../engine/ecs/core";
import {
    type AudioState,
    type SoundState,
    type PitchEntry,
    Audio,
    SoundVoices,
    createAudioState,
    initAudio,
    started,
    running,
    tickAudio,
    disposeAudio,
    alloc,
    free,
    assign,
    gate,
    setParamDirect,
    setVoiceSpatial,
    setVoiceOneShot,
    addSpatial,
    flushSpatial,
    onIdle,
    voiceGen,
    polar,
} from "./engine";
import { instrumentRegistry, getValues } from "./instrument";
import { noteFreq } from "./pattern";
import { Transform, WorldTransform } from "../transforms";

const SoundData = buf(Float32Array, 5, 0);

export const Sound = {
    instrument: createFieldProxy(SoundData, 5, 0),
    loop: createFieldProxy(SoundData, 5, 1),
    volume: createFieldProxy(SoundData, 5, 2),
    pitch: createFieldProxy(SoundData, 5, 3),
    spatial: createFieldProxy(SoundData, 5, 4),
};

traits(Sound, {
    defaults: () => ({ loop: 0, volume: 1, pitch: 0, spatial: 0 }),
    parse: { instrument: (name: string) => instrumentRegistry.getByName(name) },
});

export const Listener = {};
traits(Listener, { requires: [Transform] });

function evictVoice(audio: AudioState, ss: SoundState, state: State): void {
    let oldestEid = -1;
    let oldestGen = Infinity;
    for (const [eid, voice] of ss.voices) {
        const gen = voiceGen(audio, voice.slot);
        if (gen < oldestGen) {
            oldestGen = gen;
            oldestEid = eid;
        }
    }
    if (oldestEid < 0) return;
    console.warn("audio: voice pool full, evicting oldest voice");
    const voice = ss.voices.get(oldestEid)!;
    free(audio, voice.slot);
    ss.voices.delete(oldestEid);
    ss.systemRemovals.add(oldestEid);
    if (state.entityExists(oldestEid)) {
        if (state.hasComponent(oldestEid, Sound)) state.removeComponent(oldestEid, Sound);
        state.removeEntity(oldestEid);
    }
}

function allocateVoice(eid: number, audio: AudioState, ss: SoundState, state: State): boolean {
    let slot = alloc(audio);
    if (slot === -1) {
        evictVoice(audio, ss, state);
        slot = alloc(audio);
        if (slot === -1) return false;
    }

    const instId = Sound.instrument[eid];
    assign(audio, slot, instId);
    setVoiceSpatial(audio, slot, Sound.spatial[eid] === 1);
    gate(audio, slot, 1);

    const inst = instrumentRegistry.get(instId);
    let volumeOffset = -1;
    let baseVolume = 1;
    const pitchEntries: PitchEntry[] = [];
    if (inst) {
        const vals = getValues(instId);
        if (inst.volumeParam) {
            volumeOffset = inst.paramLayout.get(inst.volumeParam) ?? -1;
            if (vals) baseVolume = vals.get(inst.volumeParam) ?? 1;
        }
        if (inst.pitchParams) {
            for (const pp of inst.pitchParams) {
                const offset = inst.paramLayout.get(pp) ?? -1;
                if (offset >= 0) {
                    const baseFreq = vals?.get(pp) ?? 440;
                    const node = pp.slice(0, pp.indexOf("."));
                    const octave = vals?.get(`${node}.octave`) ?? 0;
                    const semitone = vals?.get(`${node}.semitone`) ?? 0;
                    const fine = vals?.get(`${node}.fine`) ?? 0;
                    pitchEntries.push({ offset, baseFreq, octave, semitone, fine });
                }
            }
        }
    }
    ss.voices.set(eid, { slot, volumeOffset, baseVolume, pitchEntries });

    if (Sound.loop[eid] === 0) {
        setVoiceOneShot(audio, slot);
        onIdle(audio, slot, () => {
            free(audio, slot);
            ss.voices.delete(eid);
            ss.systemRemovals.add(eid);
            if (state.entityExists(eid)) {
                if (state.hasComponent(eid, Sound)) {
                    state.removeComponent(eid, Sound);
                }
                state.removeEntity(eid);
            }
        });
    }

    return true;
}

const SoundSystem: System = {
    group: "simulation",
    update(state: State) {
        const audio = Audio.from(state);
        if (!audio || !started(audio)) return;
        tickAudio(audio);

        const ss = SoundVoices.from(state);
        if (!ss) return;

        if (!running(audio)) {
            for (const eid of ss.pending) {
                if (!state.hasComponent(eid, Sound)) {
                    ss.pending.delete(eid);
                    continue;
                }
                if (Sound.loop[eid] === 0) {
                    ss.pending.delete(eid);
                    ss.systemRemovals.add(eid);
                    if (state.entityExists(eid)) {
                        if (state.hasComponent(eid, Sound)) {
                            state.removeComponent(eid, Sound);
                        }
                        state.removeEntity(eid);
                    }
                }
            }
            return;
        }

        for (const eid of ss.pending) {
            if (!state.hasComponent(eid, Sound)) {
                ss.pending.delete(eid);
                continue;
            }
            if (allocateVoice(eid, audio, ss, state)) {
                ss.pending.delete(eid);
            } else {
                break;
            }
        }

        for (const [eid, voice] of ss.voices) {
            if (voice.volumeOffset >= 0) {
                const v = Sound.volume[eid];
                setParamDirect(audio, voice.slot, voice.volumeOffset, v * v * voice.baseVolume);
            }
            for (const pe of voice.pitchEntries) {
                const freq = noteFreq(
                    pe.baseFreq,
                    pe.octave,
                    Sound.pitch[eid] + pe.semitone,
                    pe.fine,
                );
                setParamDirect(audio, voice.slot, pe.offset, freq);
            }
        }

        const listenerEid = state.only([Listener, WorldTransform]);
        if (listenerEid >= 0) {
            const m = WorldTransform.data;
            const lo = listenerEid * 16;
            const lx = m[lo + 12],
                ly = m[lo + 13],
                lz = m[lo + 14];
            const rx = m[lo],
                ry = m[lo + 1],
                rz = m[lo + 2];
            const ux = m[lo + 4],
                uy = m[lo + 5],
                uz = m[lo + 6];
            const fx = m[lo + 8],
                fy = m[lo + 9],
                fz = m[lo + 10];

            for (const [eid, voice] of ss.voices) {
                if (Sound.spatial[eid] !== 1) continue;
                if (!state.hasComponent(eid, WorldTransform)) continue;
                const so = eid * 16;
                const dx = m[so + 12] - lx;
                const dy = m[so + 13] - ly;
                const dz = m[so + 14] - lz;
                const { azimuth, elevation, distance } = polar(
                    dx,
                    dy,
                    dz,
                    rx,
                    ry,
                    rz,
                    ux,
                    uy,
                    uz,
                    fx,
                    fy,
                    fz,
                );
                addSpatial(audio, voice.slot, azimuth, elevation, distance, 3.0, 100.0, 1.0);
            }
            flushSpatial(audio);
        }
    },
    dispose(state: State) {
        const audio = Audio.from(state);
        const ss = SoundVoices.from(state);
        if (audio && ss) {
            for (const [, voice] of ss.voices) {
                gate(audio, voice.slot, 0);
                free(audio, voice.slot);
            }
            ss.voices.clear();
            ss.pending.clear();
            ss.systemRemovals.clear();
        }
        if (audio) disposeAudio(audio);
    },
};

export { instrument } from "./instrument";
export type { Instrument } from "./instrument";
export { noteFreq, midiFreq } from "./pattern";
export { sample, getSampleByName, getSample, whenLoaded } from "./sample";

export const AudioPlugin: Plugin = {
    name: "Audio",
    systems: [SoundSystem],
    components: { Sound, Listener },
    async initialize(state: State) {
        const audio = createAudioState();
        state.setResource(Audio, audio);

        const ss: SoundState = {
            voices: new Map(),
            pending: new Set(),
            systemRemovals: new Set(),
        };
        state.setResource(SoundVoices, ss);

        state.observe(onAdd(Sound), (eid: number) => {
            ss.pending.add(eid);
        });

        state.observe(onRemove(Sound), (eid: number) => {
            if (ss.systemRemovals.has(eid)) {
                ss.systemRemovals.delete(eid);
                return;
            }
            ss.pending.delete(eid);

            const voice = ss.voices.get(eid);
            if (!voice) return;
            ss.voices.delete(eid);

            gate(audio, voice.slot, 0);
            onIdle(audio, voice.slot, () => {
                free(audio, voice.slot);
            });
        });

        try {
            await initAudio(audio);
        } catch {
            audio.backend = null;
        }
    },
};
