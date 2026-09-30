import type { PointerLockStatus } from "../../core/input";
import {
    devices,
    InputPlugin,
    inputEnabled,
    pointerLockRefusal as readPointerLockRefusal,
    pointerLockStatus as readPointerLockStatus,
    releasePointerLock,
    requirePointerLock,
} from "../../core/input";
import { Camera, RenderPlugin } from "../../core/rendering";
import { entity, f32, not, type Plugin, type State, type System, Transform } from "../../engine";
import { clamp, lerp } from "../../engine/utils";
import {
    Character,
    CharacterPlugin,
    CharacterSweepSystem,
    globalTransform,
    jump,
    move,
} from "../../transitional/character";
import { Body } from "../../transitional/physics";
import { PlayerFollow } from "./follow";

// First-person player controller — composes a kinematic `Character` (the §6.4 controller) with WASD + a
// pointer-lock mouse look + a follow camera. The Player entity IS the character's capsule body (Body +
// Character, mass <= 0); a separate camera entity (Camera + a renderer marker + Transform) is linked via
// `Player.camera`. The controller owns the look (yaw/pitch, instant) + the move/jump intent (driven through
// the `character` module's eid-keyed `move`/`jump`); the CPU sweep produces fixed-tick GlobalTransform.
// The camera follows its position with fixed-timestep interpolation: a
// `fixed`-group system (`after: [CharacterSweepSystem]`) snapshots this tick's swept GlobalTransform (`character.globalTransform`,
// off the CPU controller state) into prev/curr, and the camera renders `lerp(prev, curr, fixedAlpha)` — see
// `PlayerSnapshotSystem`. Input → GlobalTransform → camera position carries no readback lag (it stops scaling with
// GPU frame time, mouse-look already did); the only camera latency is the kept one-tick interpolation +
// the irreducible display fence. Walk/jump/slope tuning lives on `Character`.
//
// The rig is written pivot-first so third-person drops in later: the camera sits at `pivot − forward·distance`
// where `pivot = charPos + eyeHeight` and `distance` defaults to 0 (first-person — the camera is AT the eye).
// A future third-person mode sets `Player.distance > 0`; nothing else here changes.

const MAX_PITCH = Math.PI / 2 - 0.01;
// the look normalizes by this fixed reference height, never the live canvas — the why is in
// PlayerControlSystem.update (resolution-independence).
const LOOK_REFERENCE_HEIGHT = 1080;

/**
 * first-person player: the look + camera layer over a kinematic {@link Character}. Lives on the same capsule
 * {@link Body} as a {@link Character} (`mass <= 0`); the character module registers + drives it, this adds the
 * mouse look + a follow camera. `camera` is the eid of a separate camera entity (Camera + Transform + a
 * renderer marker); the controller writes its authored Transform each frame. `distance` is the camera's pull-back from the eye. 0 is
 * first-person (the default), `> 0` is the third-person scaffolding. Walk/jump/slope tuning lives on `Character`.
 *
 * @example
 * ```
 * const body = state.create();
 * state.add(body, Body); state.add(body, Character); state.add(body, Player);   // a capsule, mass 0
 * Body.shape.set(body, ShapeKind.Capsule);
 * Body.halfExtents.set(body, 0, 0.6, 0, 0.3); Body.mass.set(body, 0);
 * Character.jumpSpeed.set(body, 6); Character.gravity.set(body, -30);           // snappy jump/fall
 * const cam = state.create();
 * state.add(cam, Transform); state.add(cam, Camera); state.add(cam, Sear);
 * Player.camera.set(body, cam);
 * ```
 */
export const Player = {
    /** look yaw in radians (turn around world Y); set it to face a direction at spawn */
    yaw: f32,
    /** look pitch in radians (clamped to ±90°); set it to tilt the view at spawn */
    pitch: f32,
    /** walk speed (m/s) the move input is scaled to */
    speed: f32,
    /** sprint multiplier applied while Shift is held */
    sprint: f32,
    /** mouse-look radians per pixel of pointer-lock movement, at a fixed 1080px reference height (the look
     * speed is resolution-independent, so the same mouse motion turns the same angle at any canvas size) */
    sensitivity: f32,
    /** camera height above the capsule centre (the eye offset) */
    eyeHeight: f32,
    /** camera pull-back from the eye: 0 = first-person, > 0 = third-person (scaffolding) */
    distance: f32,
    /** the linked camera entity (a Camera + Transform); set this or the camera never moves */
    camera: entity,
};

/** Pointer-lock reads are State-scoped. */
export type { PointerLockStatus } from "../../core/input";
export function pointerLockStatus(state: State): PointerLockStatus {
    return readPointerLockStatus(state);
}
export function pointerLockRefusal(state: State): string | null {
    return readPointerLockRefusal(state);
}

function setupPointerLock(state: State): void {
    requirePointerLock(state, true);
    state.onDispose(() => {
        releasePointerLock(state);
        requirePointerLock(state, false);
    });
}

// scratch for the per-tick swept GlobalTransform read (character.globalTransform), reused across players.
const _globalTransform: [number, number, number] = [0, 0, 0];
// query terms held once, so a steady frame mints no array.
const PLAYER_BODIES = [Player, Body];
const ORPHAN_FOLLOWS = [not(Player), PlayerFollow];

// Snapshot the character's fixed-tick GlobalTransform position once per tick, so the camera can
// render-interpolate it by `fixedAlpha`, matching the engine renderer. The character controller writes
// GlobalTransform this tick, so this system runs `after: [CharacterSweepSystem]`; it does not read back
// from the GPU. Capturing on the fixed clock is what keeps
// the camera smooth at ANY render rate; the only camera lag is the kept one-tick interpolation, no readback.
const PlayerSnapshotSystem: System = {
    name: "snapshot",
    group: "fixed",
    after: [CharacterSweepSystem],
    update(state: State) {
        for (const eid of state.query(PLAYER_BODIES)) {
            if (!globalTransform(state, eid, _globalTransform)) continue; // The body producer has not registered yet.
            const x = _globalTransform[0];
            const y = _globalTransform[1];
            const z = _globalTransform[2];
            if (state.has(eid, PlayerFollow)) {
                state
                    .of(PlayerFollow)
                    .prev.set(
                        eid,
                        state.of(PlayerFollow).curr.x.get(eid),
                        state.of(PlayerFollow).curr.y.get(eid),
                        state.of(PlayerFollow).curr.z.get(eid),
                        0,
                    );
            } else {
                // first snapshot: prev == curr, and membership becomes the "initialized" flag
                state.add(eid, PlayerFollow);
                state.of(PlayerFollow).prev.set(eid, x, y, z, 0);
            }
            state.of(PlayerFollow).curr.set(eid, x, y, z, 0);
        }
        // drop the follow state when a player is gone (mirrors the derived-state cleanup in orbit)
        for (const eid of state.query(ORPHAN_FOLLOWS)) state.remove(eid, PlayerFollow);
    },
};

// the player's render position: lerp between the two most recent fixed-tick GlobalTransform positions by `fixedAlpha`.
// Falls back to the authored Body spawn position until the first snapshot lands, so the first frames aren't at the origin.
function followPos(state: State, eid: number, out: [number, number, number]): void {
    if (state.has(eid, PlayerFollow)) {
        const a = state.time.fixedAlpha;
        out[0] = lerp(
            state.of(PlayerFollow).prev.x.get(eid),
            state.of(PlayerFollow).curr.x.get(eid),
            a,
        );
        out[1] = lerp(
            state.of(PlayerFollow).prev.y.get(eid),
            state.of(PlayerFollow).curr.y.get(eid),
            a,
        );
        out[2] = lerp(
            state.of(PlayerFollow).prev.z.get(eid),
            state.of(PlayerFollow).curr.z.get(eid),
            a,
        );
        return;
    }
    out[0] = state.of(Body).pos.x.get(eid);
    out[1] = state.of(Body).pos.y.get(eid);
    out[2] = state.of(Body).pos.z.get(eid);
}

function findCamera(state: State, eid: number): number {
    const cam = state.of(Player).camera.get(eid);
    if (!cam || !state.has(cam, Camera)) {
        // warn once, latched on the derived PlayerFollow (added by the snapshot system); if it isn't up yet
        // (the character hasn't registered), skip — the next frame with GlobalTransform warns.
        if (state.has(eid, PlayerFollow) && !state.of(PlayerFollow).warned.get(eid)) {
            state.of(PlayerFollow).warned.set(eid, 1);
            console.warn(
                `[player] entity ${eid} has Player but Player.camera points at no Camera — set it to a camera eid`,
            );
        }
        return -1;
    }
    return cam;
}

// FPS orientation from yaw (around world Y) then pitch (around the camera's right axis). Matches the
// forward used for the move basis + the third-person offset (forward = q·(0,0,−1)).
function setLook(cam: number, yaw: number, pitch: number): void {
    const hy = yaw * 0.5;
    const hp = pitch * 0.5;
    const sy = Math.sin(hy);
    const cy = Math.cos(hy);
    const sp = Math.sin(hp);
    const cp = Math.cos(hp);
    Transform.rot.set(cam, cy * sp, sy * cp, -sy * sp, cy * cp);
}

const _pos: [number, number, number] = [0, 0, 0];

/**
 * the first-person controller: mouse-look + WASD/jump intent + the follow-camera Transform, run in the
 * `simulation` group. Exported as an ordering anchor: a camera-juice system that perturbs the authored camera
 * placement declares `after: [PlayerControlSystem]`, reading the base
 * `Transform` this writes before `BeginFrameSystem` (draw) consumes it.
 */
export const PlayerControlSystem: System = {
    name: "control",
    group: "simulation",

    setup: setupPointerLock,

    update(state: State) {
        // input suspended (a menu/cutscene): release the lock so the cursor frees + mouse-look stops, and let
        // the loop run with neutral device data — every key reads up, so move resolves to 0 and the player freezes.
        const input = devices(state);
        const active = inputEnabled(state);
        for (const eid of state.query(PLAYER_BODIES)) {
            let yaw = state.of(Player).yaw.get(eid);
            let pitch = state.of(Player).pitch.get(eid);
            if (active && input.pointer.lock.status === "locked") {
                // Resolution-independent mouse-look. Pointer-lock movementX/Y is physical mouse motion in CSS
                // px — independent of canvas size — so the angle per pixel must NOT scale with the canvas.
                const s = state.of(Player).sensitivity.get(eid) / LOOK_REFERENCE_HEIGHT;
                yaw -= input.pointer.deltaX * s;
                pitch = clamp(pitch - input.pointer.deltaY * s, -MAX_PITCH, MAX_PITCH);
                state.of(Player).yaw.set(eid, yaw);
                state.of(Player).pitch.set(eid, pitch);
            }

            const cy = Math.cos(yaw);
            const sy = Math.sin(yaw);
            const sprint =
                input.keys.held.has("ShiftLeft") || input.keys.held.has("ShiftRight")
                    ? state.of(Player).sprint.get(eid)
                    : 1;

            let lx = 0;
            let lz = 0;
            if (input.keys.held.has("KeyW")) lz -= 1;
            if (input.keys.held.has("KeyS")) lz += 1;
            if (input.keys.held.has("KeyA")) lx -= 1;
            if (input.keys.held.has("KeyD")) lx += 1;
            const len = Math.sqrt(lx * lx + lz * lz);
            if (len > 0) {
                const v = (state.of(Player).speed.get(eid) * sprint) / len;
                move(state, eid, (lz * sy + lx * cy) * v, (lz * cy - lx * sy) * v);
            } else {
                move(state, eid, 0, 0);
            }
            // one-shot: the press edge, not the held key. A held key refills the jump buffer every
            // frame, re-firing the instant the char re-grounds (a ledge, a landing); the buffer +
            // coyote forgiveness lives in the character pass.
            if (input.keys.pressed.has("Space")) jump(state, eid);

            const cam = findCamera(state, eid);
            if (cam < 0) continue;

            // pivot = the eye; the camera sits `distance` back along the look forward (0 = first-person).
            followPos(state, eid, _pos);
            const cp = Math.cos(pitch);
            const fx = -cp * sy;
            const fy = Math.sin(pitch);
            const fz = -cp * cy;
            const dist = state.of(Player).distance.get(eid);
            state
                .of(Transform)
                .pos.set(
                    cam,
                    _pos[0] - fx * dist,
                    _pos[1] + state.of(Player).eyeHeight.get(eid) - fy * dist,
                    _pos[2] - fz * dist,
                    1,
                );
            setLook(cam, yaw, pitch);
        }

        // InputPlugin clears the shared pointer delta at the draw boundary.
    },
};

/** first-person player plugin: pointer-lock mouse look, WASD/sprint/jump, and a fixed-timestep follow
 *  camera over a kinematic {@link Character}. Depends on {@link CharacterPlugin} (the controller it composes),
 *  input, and the renderer; add `PhysicsPlugin` to the scene, and
 *  the character sweeps against it. Give an entity {@link Body} + {@link Character} + {@link Player}. */
export const PlayerPlugin: Plugin = {
    name: "Player",
    systems: [PlayerSnapshotSystem, PlayerControlSystem],
    components: { Player },
    dependencies: [CharacterPlugin, InputPlugin, RenderPlugin],
    traits: {
        Player: {
            requires: [Body, Character],
            defaults: () => ({
                yaw: 0,
                pitch: 0,
                speed: 6,
                sprint: 1.8,
                sensitivity: 1.5,
                eyeHeight: 0.7,
                distance: 0,
                camera: 0,
            }),
        },
    },
};
