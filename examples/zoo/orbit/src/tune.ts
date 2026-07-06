import { Orbit, type Plugin, type State } from "@dylanebert/shallot";

// #doc:intro
// A third-person camera: drag to rotate around a target, scroll to zoom, right-drag to fly.

// #doc:code source:orbit/public/scenes/orbit.scene
// Add the `Orbit` component to a camera and the controls are live. The pose (`distance`, `yaw`, `pitch`)
// lives in the scene, so the editor edits it directly; `target: @box` orbits the box instead of the
// world origin, tracking it as it moves.

// #doc:code
// ### Tune the feel
//
// Set feel knobs in code with a plugin that runs once on load:
// #region tune
const OrbitTune = {
    name: "OrbitTune",
    warm(state: State) {
        for (const cam of state.query([Orbit])) {
            Orbit.sensitivity.set(cam, 0.005); // orbit drag, radians per pixel
            Orbit.flySpeed.set(cam, 8); // fly speed, units per second
        }
    },
} satisfies Plugin;
// #endregion

// #doc:code
// Hold the fly button to look with the mouse and move with WASD/QE; scroll while flying adjusts
// `flySpeed`, Shift boosts it.

export default OrbitTune;
