import type { CharState } from "./sweep";

// The eid-keyed drive state + its public surface (`character/core`). The sweep (index.ts) POPULATES `states`
// (the per-character controller pose) and CONSUMES `moves` / `jumped` each fixed tick; a controller — Player,
// or a custom one — writes intent through `move` / `jump` and reads the swept result through `pose` /
// `grounded`. The maps are the shared seam between the drive API and the sweep: module state, cleared by
// `resetDrive` on dispose (reload-safety). Living in a sibling keeps the drive off the main barrel — it's the
// `character/core` extension surface (exports.md barrel rules), not the `Character`-component happy path.

// the CPU controller state per character eid — the sweep owns it across fixed ticks (pose, velocity, jump
// timers, grounded). Built from the Body authored fields when a character is first registered; the swept pose
// then drifts from the Body spawn slab (the controller owns it; the Body slab is only the spawn pose).
export const states = new Map<number, CharState>();
// the per-frame drive: the latest horizontal input (persists until changed — a held direction) + the jump
// requests this tick (a Set so a held/spammed button is one press edge; the buffer/coyote in the sweep gate it).
export const moves = new Map<number, [number, number]>();
export const jumped = new Set<number>();

/** push a character's per-frame horizontal move input (world x/z velocity), by body eid. Call each fixed tick
 *  it should move; a character given no input idles (gravity still pulls it down while airborne).
 *
 * @example
 * ```
 * move(player, dir[0] * speed, dir[2] * speed);   // each fixed tick
 * ```
 */
export function move(eid: number, vx: number, vz: number): void {
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
 * if (pressed) jump(player);   // on the press edge, not the held key
 * ```
 */
export function jump(eid: number): void {
    jumped.add(eid);
}

/** read a character's swept pose into `out` (by body eid); returns false (leaving `out` untouched) until the
 *  character is registered. The CPU sweep owns this pose same-frame, so a follower (a camera) tracks the
 *  player with no GPU readback.
 *
 * @example
 * ```
 * const p: [number, number, number] = [0, 0, 0];
 * if (pose(player, p)) placeModelAt(p);
 * ```
 */
export function pose(eid: number, out: [number, number, number]): boolean {
    const st = states.get(eid);
    if (!st) return false;
    out[0] = st.pos[0];
    out[1] = st.pos[1];
    out[2] = st.pos[2];
    return true;
}

/** place a character at a world position (by body eid), clearing its velocity: the respawn primitive a
 *  fall-recovery system calls. Returns false (a no-op) until the character is registered. The controller
 *  owns the pose, so this is the ONLY way to move a swept character from the outside; the next sweep
 *  integrates from here, and zeroing the velocity keeps a mid-air respawn from inheriting the old fall speed.
 *
 * @example
 * ```
 * const p: [number, number, number] = [0, 0, 0];
 * if (pose(player, p) && p[1] < -20) teleport(player, 0, 4, 0);   // fell off — back to the start
 * ```
 */
export function teleport(eid: number, x: number, y: number, z: number): boolean {
    const st = states.get(eid);
    if (!st) return false;
    st.pos[0] = x;
    st.pos[1] = y;
    st.pos[2] = z;
    st.vel[0] = st.vel[1] = st.vel[2] = 0;
    st.realizedVel[0] = st.realizedVel[1] = st.realizedVel[2] = 0;
    return true;
}

/** whether a character is grounded (by body eid): the swept-step result the controller keys jump + slope
 *  hold on. False until the character is registered.
 *
 * @example
 * ```
 * anim.set(grounded(player) ? "idle" : "fall");
 * ```
 */
export function grounded(eid: number): boolean {
    return states.get(eid)?.grounded ?? false;
}

// clear the drive state (plugin dispose / reload) — the maps are module-level, so a rebuilt State must start clean.
export function resetDrive(): void {
    states.clear();
    moves.clear();
    jumped.clear();
}
