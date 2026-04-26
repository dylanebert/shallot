import { resource, traits, type State, type System, type Plugin } from "../../engine";
import { WorldTransform } from "../../standard/transforms";
import { Compute, ComputePlugin } from "../../standard/compute";
import { Physics, PhysicsPlugin } from "../../standard/physics";
import { Sound, Listener } from "../../standard/audio";
import {
    Audio,
    SoundVoices,
    started,
    addAcousticSeparate,
    flushAcoustic,
} from "../../standard/audio/engine";
import { createOcclusionNode, createOcclusionState, type OcclusionState } from "./occlusion";
import { createReflectionNode, createReflectionState, type ReflectionState } from "./reflection";
import { AcousticMaterial } from "./material";
import { processHistogram, MAX_SOURCES } from "./dsp";

export { AcousticMaterial, MaterialPreset } from "./material";
export type { MaterialValues } from "./material";

export const Acoustic = {};
traits(Acoustic, {});

const OcclusionKey = resource<OcclusionState>("acoustics-occlusion");
const ReflectionKey = resource<ReflectionState>("acoustics-reflection");

const AcousticSystem: System = {
    group: "fixed",
    update(state: State) {
        const audio = Audio.from(state);
        if (!audio || !started(audio)) return;

        const occ = OcclusionKey.from(state);
        if (!occ) return;

        const refl = ReflectionKey.from(state);
        const ss = SoundVoices.from(state);

        if (state.time.throttled) {
            if (refl) {
                refl.smoothed.fill(0);
                refl.histogramReady = false;
                refl.generation++;
                refl.needsReset = true;
            }
            occ.readbackReady = false;
            occ.generation++;
            return;
        }

        if (occ.readbackReady) {
            occ.readbackReady = false;
            const buf = occ.readbackBuf;
            const count = occ.readbackCount;
            for (let i = 0; i < count; i++) {
                const slot = occ.slots[i];
                const base = i * 4;
                addAcousticSeparate(
                    audio,
                    slot,
                    buf[base],
                    buf[base + 1],
                    buf[base + 2],
                    buf[base + 3],
                );
            }
            flushAcoustic(audio);
        }

        if (refl) {
            refl.fixedTick = state.time.fixedTick;
            if (refl.histogramReady) {
                refl.histogramReady = false;
                const t0 = performance.now();
                processHistogram(refl, occ.sourceCount, occ.slots, audio);
                state.scheduler.reportCpu("Acoustic/0:histogram", performance.now() - t0);
            }
        }

        const listenerEid = state.only([Listener, WorldTransform]);
        if (listenerEid < 0) {
            occ.sourceCount = 0;
            return;
        }

        const m = WorldTransform.data;
        const lo = listenerEid * 16;
        occ.listener[0] = m[lo + 12];
        occ.listener[1] = m[lo + 13];
        occ.listener[2] = m[lo + 14];

        let srcIdx = 0;
        for (const eid of state.query([Sound, Acoustic, WorldTransform])) {
            if (Sound.spatial[eid] !== 1) continue;
            if (srcIdx >= MAX_SOURCES) break;

            const so = eid * 16;
            const offset = srcIdx * 4;
            occ.sources[offset] = m[so + 12];
            occ.sources[offset + 1] = m[so + 13];
            occ.sources[offset + 2] = m[so + 14];

            const voice = ss?.voices.get(eid);
            if (!voice) continue;

            occ.sourcesU32[offset + 3] = voice.slot;
            occ.slots[srcIdx] = voice.slot;
            srcIdx++;
        }
        occ.sourceCount = srcIdx;
    },
};

export const AcousticPlugin: Plugin = {
    name: "Acoustic",
    components: { Acoustic, AcousticMaterial },
    dependencies: [ComputePlugin, PhysicsPlugin],

    async initialize(state: State) {
        const compute = Compute.from(state);
        if (!compute) return;

        const gpu = Physics.from(state);
        if (!gpu) return;

        const occ = createOcclusionState();
        state.setResource(OcclusionKey, occ);

        const refl = createReflectionState();
        state.setResource(ReflectionKey, refl);

        const occNode = createOcclusionNode(gpu, occ);
        compute.graph.add(occNode);

        const reflNode = createReflectionNode(gpu, occ, refl);
        compute.graph.add(reflNode);
    },

    systems: [AcousticSystem],
};
