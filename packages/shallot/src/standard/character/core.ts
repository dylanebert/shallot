// Character extension surface — the eid-keyed drive (`move` / `jump`) + readback (`pose` / `grounded`), for
// the gym `character` scenario + custom controllers. The happy path (the `Character` component +
// `CharacterPlugin`, which registers every `[Character, Body]` with the solver) ships on the barrel.

// #doc:dev
// `Player` is one controller over the character. `Character` owns the physics — the swept capsule, the
// walkable slope, the jump, the carry — and exposes a small drive surface keyed by body eid, so a controller
// is just a system that reads intent and calls into it. Give any entity `[Character, Body]`, then each fixed
// tick: `move(eid, vx, vz)` sets the horizontal velocity, `jump(eid)` requests a jump, `grounded(eid)` and
// `pose(eid, out)` read the swept result back, and `teleport(eid, x, y, z)` places it (the respawn primitive,
// since the controller owns the pose). That's the whole seam — build a twin-stick, a click-to-move, a
// vehicle, or an AI agent against it without touching the sweep. `Player` composes exactly this surface; so
// does the gym `character` scenario.
export { grounded, jump, move, pose, teleport } from "./drive";
