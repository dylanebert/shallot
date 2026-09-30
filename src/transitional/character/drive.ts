import { GlobalTransform, type World } from "../../engine";
import { setKinematic } from "../physics";
import type { CharState } from "./sweep";

/** Per-State intent and controller state shared by the sweep and its callers. */
export interface CharacterDrive {
    states: Map<number, CharState>;
    moves: Map<number, [number, number]>;
    jumped: Set<number>;
    stamps: Map<number, number>;
    signature: number;
}

const driveKey = { create: createDrive };

function createDrive(): CharacterDrive {
    return {
        states: new Map(),
        moves: new Map(),
        jumped: new Set(),
        stamps: new Map(),
        signature: -1,
    };
}

export function driveFor(state: World): CharacterDrive {
    return state.resource(driveKey);
}

/** push a character's per-frame horizontal move input (world x/z velocity), by body eid. Call each fixed tick
 *  it should move; a character given no input idles (gravity still pulls it down while airborne).
 *
 * @example
 * ```
 * move(state, player, dir[0] * speed, dir[2] * speed);   // each fixed tick
 * ```
 */
export function move(state: World, eid: number, vx: number, vz: number): void {
    const moves = driveFor(state).moves;
    const m = moves.get(eid);
    if (m) {
        m[0] = vx;
        m[1] = vz;
    } else {
        moves.set(eid, [vx, vz]);
    }
}

/** request a jump for a character this tick (by body eid). fires only if grounded or within the coyote
 *  window, consuming both so a held button can't re-fire mid-air (the sweep gates it).
 *
 * @example
 * ```
 * if (pressed) jump(state, player);   // on the press edge, not the held key
 * ```
 */
export function jump(state: World, eid: number): void {
    driveFor(state).jumped.add(eid);
}

/** Read a character's GlobalTransform position into `out`; returns false until the character is
 *  registered. The body producer writes it in the fixed tick, so a follower (a camera) tracks the
 *  player with no GPU readback.
 *
 * @example
 * ```
 * const p: [number, number, number] = [0, 0, 0];
 * if (globalTransform(state, player, p)) placeModelAt(p);
 * ```
 */
export function globalTransform(state: World, eid: number, out: [number, number, number]): boolean {
    if (!driveFor(state).states.has(eid) || !state.has(eid, GlobalTransform)) return false;
    const global = state.of(GlobalTransform);
    const column = global.pos.column;
    const offset = eid * 4;
    out[0] = column[offset];
    out[1] = column[offset + 1];
    out[2] = column[offset + 2];
    return true;
}

/** place a character at a world position (by body eid), clearing its velocity: the respawn primitive a
 *  fall-recovery system calls. Returns false (a no-op) until the character is registered. The controller
 *  owns its motion, so this is the way to move a swept character from the outside; the next sweep
 *  integrates from here, and zeroing the velocity keeps a mid-air respawn from inheriting the old fall speed.
 *
 * @example
 * ```
 * const p: [number, number, number] = [0, 0, 0];
 * if (globalTransform(state, player, p) && p[1] < -20) teleport(state, player, 0, 4, 0);
 * ```
 */
export function teleport(state: World, eid: number, x: number, y: number, z: number): boolean {
    const st = driveFor(state).states.get(eid);
    if (!st) return false;
    st.pos[0] = x;
    st.pos[1] = y;
    st.pos[2] = z;
    st.vel[0] = st.vel[1] = st.vel[2] = 0;
    st.realizedVel[0] = st.realizedVel[1] = st.realizedVel[2] = 0;
    setKinematic(state, eid, st.pos, st.quat, true, st.realizedVel);
    return true;
}

/** whether a character is grounded (by body eid): the swept-step result the controller keys jump + slope
 *  hold on. False until the character is registered.
 *
 * @example
 * ```
 * anim.set(grounded(state, player) ? "idle" : "fall");
 * ```
 */
export function grounded(state: World, eid: number): boolean {
    return driveFor(state).states.get(eid)?.grounded ?? false;
}

/** clear the drive state owned by a disposed or rebuilt State. */
export function resetDrive(state: World): void {
    const drive = driveFor(state);
    drive.states.clear();
    drive.moves.clear();
    drive.jumped.clear();
    drive.stamps.clear();
    drive.signature = -1;
}
