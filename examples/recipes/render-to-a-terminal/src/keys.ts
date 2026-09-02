import { Inputs, Orbit, type Plugin, type System } from "@dylanebert/shallot";

// Arrow-key orbit: nudges yaw/pitch directly by a fixed rate × dt, independent of Orbit's own mouse-drag
// reads. The web target orbits by mouse drag (`Orbit`'s own system); the terminal target has no mouse, so
// this is the keyboard-only control path a raw-mode stdin bridge (S4) drives instead — pointer input and
// keyboard input are deliberately not claimed to feel the same (`specs/shallot-tui.md`'s Out of scope:
// "Parity between the two input paths is explicitly not claimed"). `updateKeyOrbit` takes no engine state
// beyond the entity id, dt, and the four held-key booleans, so it's callable with no device and no live
// `State` — `Orbit.yaw`/`Orbit.pitch` are module-level sparse stores (`engine/ecs/sparse.ts`), not fields
// on a booted world.

const KEY_YAW_RATE = 1.2; // rad/s
const KEY_PITCH_RATE = 0.9; // rad/s

/** nudge one orbiting camera's yaw/pitch by the held-key state, clamped to its own pitch limits.
 *  @example updateKeyOrbit(camEid, dt, { left: true, right: false, up: false, down: false }); */
export function updateKeyOrbit(
    eid: number,
    dt: number,
    keys: { left: boolean; right: boolean; up: boolean; down: boolean },
): void {
    let yaw = Orbit.yaw.get(eid);
    let pitch = Orbit.pitch.get(eid);
    if (keys.left) yaw -= KEY_YAW_RATE * dt;
    if (keys.right) yaw += KEY_YAW_RATE * dt;
    const minPitch = Orbit.minPitch.get(eid);
    const maxPitch = Orbit.maxPitch.get(eid);
    if (keys.up) pitch = Math.max(minPitch, pitch - KEY_PITCH_RATE * dt);
    if (keys.down) pitch = Math.min(maxPitch, pitch + KEY_PITCH_RATE * dt);
    Orbit.yaw.set(eid, yaw);
    Orbit.pitch.set(eid, pitch);
}

const KeyOrbitSystem: System = {
    name: "keyOrbit",
    group: "simulation",
    update(state) {
        const dt = state.time.deltaTime;
        const keys = {
            left: Inputs.isKeyDown("ArrowLeft"),
            right: Inputs.isKeyDown("ArrowRight"),
            up: Inputs.isKeyDown("ArrowUp"),
            down: Inputs.isKeyDown("ArrowDown"),
        };
        for (const eid of state.query([Orbit])) updateKeyOrbit(eid, dt, keys);
    },
};

// arrow-key orbit alongside the web's mouse-drag `Orbit` — the keyboard control path this recipe shares
// with a future terminal-side stdin bridge
const KeyOrbitPlugin: Plugin = {
    name: "KeyOrbit",
    systems: [KeyOrbitSystem],
};

export default KeyOrbitPlugin;
