import { DirectionalLight, type Plugin, Sky, type State, type System } from "@dylanebert/shallot";

// Sky is an extra: `"Sky": true` in the manifest registers the `sky` backdrop. One `Sky` singleton in the
// scene sets the look; select it on the camera with `backdrop="name: sky"`. The sun follows the scene's
// directional light. Colors and the sun live in the scene; the denser look knobs set once here on load.
function tune(state: State) {
    for (const sky of state.query([Sky])) {
        Sky.cloudDensity.set(sky, 0.7);
        Sky.cloudHeight.set(sky, 4);
        Sky.starAmount.set(sky, 0.5);
    }
}

// the day-night cycle: sweep the sun's direction over time so the sky (which follows the scene's
// directional light) moves with it. The angle derives from `state.time.elapsed`, never a `+= dt`
// accumulator, so it stays correct across a reload.
const cycle: System = {
    name: "cycle",
    update(state: State) {
        const sun = state.only([DirectionalLight]);
        if (sun < 0) return;
        const t = state.time.elapsed * 0.2; // radians/sec — a slow arc across the sky
        DirectionalLight.direction.set(sun, Math.cos(t), -Math.sin(t) - 0.1, -0.35, 0);
    },
};

const SkyTune = {
    name: "SkyTune",
    warm: tune,
    systems: [cycle],
} satisfies Plugin;

export default SkyTune;
