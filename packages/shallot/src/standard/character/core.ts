// Character extension surface — the eid-keyed drive (`move` / `jump`) + readback (`pose` / `grounded`), for
// the gym `character` scenario + custom controllers. The happy path (the `Character` component +
// `CharacterPlugin`, which registers every `[Character, Body]` with the solver) ships on the barrel.

export { grounded, jump, move, pose, teleport } from "./drive";
