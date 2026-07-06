import { Body, Player, type Plugin, type State, type System } from "@dylanebert/shallot";
import { pose, teleport } from "@dylanebert/shallot/character/core";
import { Physics } from "@dylanebert/shallot/physics/core";

// #doc:intro
// Let's build an obby — an obstacle course you run and jump across, falling into the void if you miss a
// platform. Everything here is the engine you've already met: a first-person `Player`, some platforms, and a
// few small systems you write yourself. Open it in the editor to walk the course, or read on to see how the
// pieces fit.

// #doc:code source:obby/public/scenes/obby.scene
// ## The course
//
// The scene is a `Player` (a capsule with a follow camera — the [Player](doc:standard/player) page covers it) and
// a row of platforms. Each platform is a `body` with `mass: 0` (solid, but the solver never moves it) and a
// `part` to draw it. There's no ground between them, so a missed jump drops you into the void. Two platforms
// carry a `checkpoint` marker and the last a `goal` — plain tags, no fields, that the systems below query for.

// the markers the scene tags platforms with — an empty object is a component with no data, just a name to
// query by. The scene's `checkpoint` / `goal` / `moving` attributes add them to those entities.
const Checkpoint = {};
const Goal = {};
const Moving = {};

// where the player is sent back to when they fall — the last checkpoint reached, a little above it.
const respawn: [number, number, number] = [0, 2, 0];
let won = false;

// scratch for the player's position, filled by `player` below and read by the systems this frame.
const at: [number, number, number] = [0, 0, 0];

// the one player, with its swept position read into `at`; -1 before the controller has registered it.
function player(state: State): number {
    for (const eid of state.query([Player])) if (pose(eid, at)) return eid;
    return -1;
}

// squared distance from the player (in `at`) to a platform body, on the ground plane — cheaper than a real
// distance, and we only compare it to a fixed radius.
function near(eid: number): number {
    const dx = at[0] - Body.pos.x.get(eid);
    const dz = at[2] - Body.pos.z.get(eid);
    return dx * dx + dz * dz;
}

// #doc:code
// ## Your first system: fall and respawn
//
// A system is a name, a `group` (when it runs — `simulation` is every frame), and an `update`. This one reads
// the player's height each frame and, if they've dropped below the course, teleports them back to the last
// checkpoint. The controller owns the player's position, so we reach it through `pose` (read) and `teleport`
// (write) from `@dylanebert/shallot/character/core`.
// #region fall
const fall: System = {
    name: "fall",
    group: "simulation",
    update(state: State) {
        const p = player(state);
        if (p >= 0 && at[1] < -4) teleport(p, respawn[0], respawn[1], respawn[2]);
    },
};
// #endregion

// #doc:code
// ## Checkpoints, from a distance check
//
// No trigger volumes needed — a checkpoint is "reached" when the player gets close enough. Each frame, check
// the distance to every checkpoint platform; step onto one and it becomes the new respawn point (raised a
// little so you drop back onto it, not into it).
// #region checkpoints
const checkpoints: System = {
    name: "checkpoints",
    group: "simulation",
    update(state: State) {
        if (player(state) < 0) return;
        for (const cp of state.query([Checkpoint, Body])) {
            if (near(cp) < 4) {
                respawn[0] = Body.pos.x.get(cp);
                respawn[1] = Body.pos.y.get(cp) + 2;
                respawn[2] = Body.pos.z.get(cp);
            }
        }
    },
};
// #endregion

// #doc:code
// ## A moving platform
//
// A `mass: 0` body is kinematic — you move it, the solver doesn't, and a character standing on it is carried
// along. Drive its pose each fixed tick with `setKinematic`, sliding it side to side around its authored
// position (read back off the body, so there are no magic numbers to keep in sync with the scene).
// #region lift
const lift: System = {
    name: "lift",
    group: "fixed",
    update(state: State) {
        const step = Physics.step;
        if (!step) return;
        for (const eid of state.query([Moving])) {
            const x = Body.pos.x.get(eid) + Math.sin(state.time.elapsed * 1.2) * 2.5;
            step.setKinematic(eid, [x, Body.pos.y.get(eid), Body.pos.z.get(eid)], [0, 0, 0, 1]);
        }
    },
};
// #endregion

// #doc:code
// ## Reaching the goal
//
// The same distance check, fired once. Touch the goal platform and the run is won — here we just log it, but
// a real game would show a screen or load the next level.
// #region goal
const finish: System = {
    name: "goal",
    group: "simulation",
    update(state: State) {
        if (won || player(state) < 0) return;
        for (const g of state.query([Goal, Body])) {
            if (near(g) < 4) {
                won = true;
                console.log("you win!");
            }
        }
    },
};
// #endregion

export const Game = {
    name: "Game",
    components: { Checkpoint, Goal, Moving },
    systems: [fall, checkpoints, lift, finish],
    // module state is per-load; reset it so a re-run of the scene starts fresh.
    dispose() {
        respawn[0] = 0;
        respawn[1] = 2;
        respawn[2] = 0;
        won = false;
    },
} satisfies Plugin;

export default Game;
