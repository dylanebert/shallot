import { Fog, type Plugin, type State } from "@dylanebert/shallot";

// #doc:intro
// Volumetric atmosphere: haze that fades the scene toward a color with distance, plus optional height fog
// that pools low and thins with altitude. A compute pass marches each pixel from the camera to the scene
// depth, so near geometry stays crisp while the distance dissolves into the haze.

// #doc:code source:fog/public/scenes/fog.scene
// Fog is opt-in — `"Fog": true` in the manifest registers the pass. Give the scene one `Fog` singleton for
// the look and give the rendering camera a `depth` lane, which the march reads for scene depth. `density`
// sets how fast the scene fades; `height-falloff` turns the haze into ground fog, densest at `height-base`
// and thinning upward. The editor edits these live.

// #doc:code
// ### Tune the feel
//
// Density, color, and the height falloff shape the look and live in the scene. The march-quality knobs
// (step count and per-pixel jitter, trading cost for smoothness) set once in code on load:
// #region tune
const FogTune = {
    name: "FogTune",
    warm(state: State) {
        for (const fog of state.query([Fog])) {
            Fog.steps.set(fog, 48);
            Fog.jitter.set(fog, 1);
        }
    },
} satisfies Plugin;
// #endregion

// #doc:code
// Add `Volumetric` to a point, spot, or sun light and it casts visible shafts through the haze, shaped by
// `scattering` and `anisotropy`. See the reference for the full field set.

export default FogTune;
