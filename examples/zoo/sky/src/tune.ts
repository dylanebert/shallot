import { type Plugin, Sky, type State } from "@dylanebert/shallot";

// #doc:intro
// A procedural sky drawn behind the scene: a gradient from horizon to zenith, a sun disk, clouds, stars,
// and a haze band, with no geometry.

// #doc:code source:sky/public/scenes/sky.scene
// Sky is an extra, not a default — `"Sky": true` in the manifest registers the `sky` backdrop. Give the
// scene one `Sky` singleton for the look and select that backdrop on the camera (`backdrop="name: sky"`).
// The sun's position follows the scene's directional light; the Sky fields set only its appearance, and
// the editor edits them live.

// #doc:code
// ### Tune the feel
//
// Colors and the sun live in the scene; the denser look knobs — cloud thickness, cloud height, star
// density — set once in code on load:
// #region tune
const SkyTune = {
    name: "SkyTune",
    warm(state: State) {
        for (const sky of state.query([Sky])) {
            Sky.cloudDensity.set(sky, 0.7);
            Sky.cloudHeight.set(sky, 4);
            Sky.starAmount.set(sky, 0.5);
        }
    },
} satisfies Plugin;
// #endregion

export default SkyTune;
