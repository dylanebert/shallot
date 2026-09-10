import { Fog, type Plugin, type State } from "@dylanebert/shallot";

// density, color, and height falloff shape the look and live in the scene; the march-quality knobs
// (step count, per-pixel jitter) trade cost for smoothness and set once here on load
const FogTune = {
    name: "FogTune",
    warm(state: State) {
        for (const fog of state.query([Fog])) {
            Fog.steps.set(fog, 48);
            Fog.jitter.set(fog, 1);
        }
    },
} satisfies Plugin;

// the scene's sun carries `volumetric` for its shafts; a point or spot light opts in the same way, and
// `scattering` / `anisotropy` on the fog shape how they glow

export default FogTune;
