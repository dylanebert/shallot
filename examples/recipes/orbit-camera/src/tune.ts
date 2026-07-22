import { Orbit, type Plugin, type State } from "@dylanebert/shallot";

// Add `Orbit` to a camera and the controls are live: drag to rotate, scroll to zoom, right-drag to fly.
// The pose (`distance`, `yaw`, `pitch`) lives in the scene; `target: @box` orbits the box, tracking it as
// it moves — omit it to orbit the world origin. Feel knobs (not pose) set once in code on load.
const OrbitTune = {
    name: "OrbitTune",
    warm(state: State) {
        for (const cam of state.query([Orbit])) {
            Orbit.sensitivity.set(cam, 0.005); // orbit drag, radians per pixel
            Orbit.flySpeed.set(cam, 8); // fly speed, units per second
        }
    },
} satisfies Plugin;

// Hold the fly button to look with the mouse and move with WASD/QE; scroll while flying adjusts
// `flySpeed`, Shift boosts it.
export default OrbitTune;
