import { Character, Player, type Plugin, type State, type System } from "@dylanebert/shallot";
import { Physics } from "@dylanebert/shallot/physics/core";

// #doc:intro
// A first-person player: a capsule you walk with WASD, look with the mouse, and jump with Space. Under the
// hood it's a kinematic `Character` (the swept capsule that walks, climbs small steps, and rides platforms)
// with a mouse look and a follow camera on top. It's opt-in: add `PlayerPlugin`.

// #doc:code source:player/public/scenes/player.scene
// Give one entity a `body` (a capsule: `shape: 2`, sized by `half-extents` as half-height + radius in `w`),
// a `character` (walkable slope, jump, gravity), and a `player` (look + camera). Point `player.camera` at a
// separate camera entity, which the controller poses every frame. Click the canvas to capture the mouse, then
// WASD moves, Shift sprints, and Space jumps.

// #doc:code
// ### Tune the feel
//
// Move and look tuning lives on `Player`; the walk physics (jump height, gravity, walkable slope) lives on
// `Character`. Set them in a plugin that runs once on load:
// #region tune
function tune(state: State) {
    for (const eid of state.query([Player])) {
        Player.speed.set(eid, 7); // walk speed, m/s
        Player.sensitivity.set(eid, 1.5); // mouse look, radians per 1080 px of motion
    }
    for (const eid of state.query([Character])) {
        Character.jumpSpeed.set(eid, 7); // jump launch speed
        Character.gravity.set(eid, -30); // per-character gravity, snappier than the world's
    }
}
// #endregion

// #doc:code
// ### Ride a moving platform
//
// A `mass: 0` body is kinematic: the solver never moves it, you do. Drive its pose each fixed tick with
// `setKinematic`, and a character standing on it is carried along. This slides the platform back and forth
// (the scene tags it with a `moving` marker so the system finds it):
// #region platform
const Moving = {};

const slide: System = {
    name: "slide",
    group: "fixed",
    update(state: State) {
        const step = Physics.step;
        if (!step) return;
        const x = -4 + Math.sin(state.time.elapsed) * 3;
        for (const eid of state.query([Moving])) {
            step.setKinematic(eid, [x, 0.75, 0], [0, 0, 0, 1]);
        }
    },
};
// #endregion

export const Demo = {
    name: "Demo",
    components: { Moving },
    warm: tune,
    systems: [slide],
} satisfies Plugin;

export default Demo;
