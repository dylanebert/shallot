import { entity, f32, not, type Plugin, type State, type System, sparse } from "../../engine";
import { clamp, lerp } from "../../engine/utils";
import { Character, CharacterPlugin, CharacterSweepSystem, jump, move, pose } from "../character";
import type { PointerLockStatus } from "../input";
import {
    devices,
    InputPlugin,
    inputEnabled,
    pointerLockRefusal as readPointerLockRefusal,
    pointerLockStatus as readPointerLockStatus,
    requirePointerLock,
} from "../input";
import { Body } from "../physics";
import { Camera, RenderPlugin } from "../render";
import { Transform, TransformsPlugin } from "../transforms";
import { PlayerFollow } from "./follow";

// First-person player controller — composes a kinematic `Character` (the §6.4 controller) with WASD + a
// pointer-lock mouse look + a follow camera. The Player entity IS the character's capsule body (Body +
// Character, mass <= 0); a separate camera entity (Camera + a renderer marker + Transform) is linked via
// `Player.camera`. The controller owns the look (yaw/pitch, instant) + the move/jump intent (driven through
// the `character` module's eid-keyed `move`/`jump`); the CPU sweep owns the swept pose, written same-frame
// on the fixed clock with no GPU readback. The camera follows that pose with fixed-timestep interpolation: a
// `fixed`-group system (`after: [CharacterSweepSystem]`) snapshots this tick's swept pose (`character.pose`,
// off the CPU controller state) into prev/curr, and the camera renders `lerp(prev, curr, fixedAlpha)` — see
// `PlayerSnapshotSystem`. So input → pose → camera position carries no readback lag (it stops scaling with
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
 * renderer marker) the controller poses each frame. `distance` is the camera's pull-back from the eye. 0 is
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
    yaw: sparse(f32),
    /** look pitch in radians (clamped to ±90°); set it to tilt the view at spawn */
    pitch: sparse(f32),
    /** walk speed (m/s) the move input is scaled to */
    speed: sparse(f32),
    /** sprint multiplier applied while Shift is held */
    sprint: sparse(f32),
    /** mouse-look radians per pixel of pointer-lock movement, at a fixed 1080px reference height (the look
     * speed is resolution-independent, so the same mouse motion turns the same angle at any canvas size) */
    sensitivity: sparse(f32),
    /** camera height above the capsule centre (the eye offset) */
    eyeHeight: sparse(f32),
    /** camera pull-back from the eye: 0 = first-person, > 0 = third-person (scaffolding) */
    distance: sparse(f32),
    /** the linked camera entity (a Camera + Transform); set this or the camera never moves */
    camera: sparse(entity),
};

/** Pointer-lock reads are State-scoped. */
export type { PointerLockStatus } from "../input";
export function pointerLockStatus(state: State): PointerLockStatus {
    return readPointerLockStatus(state);
}
export function pointerLockRefusal(state: State): string | null {
    return readPointerLockRefusal(state);
}

function exitLock(): void {
    if (typeof document === "undefined") return;
    if (typeof document.exitPointerLock === "function") document.exitPointerLock();
}
// scratch for the per-tick swept-pose read (character.pose), reused across players.
const _pose: [number, number, number] = [0, 0, 0];
// query terms held once, so a steady frame mints no array.
const PLAYER_BODIES = [Player, Body];
const ORPHAN_FOLLOWS = [not(Player), PlayerFollow];

// Snapshot the player's swept pose on the FIXED clock (once per tick) into prev/curr, so the camera can
// render-interpolate it by `fixedAlpha` — standard fixed-timestep interpolation (Gaffer), the same the
// engine's body renderer uses. The pose is the CPU controller's own `CharState` (read by `character.pose`),
// written THIS tick by the sweep — so this system runs `after: [CharacterSweepSystem]` to capture the fresh
// pose, not last tick's, and there is no GPU readback in the path. Capturing on the fixed clock is what keeps
// the camera smooth at ANY render rate; the only camera lag is the kept one-tick interpolation, no readback.
const PlayerSnapshotSystem: System = {
    name: "snapshot",
    group: "fixed",
    after: [CharacterSweepSystem],
    update(state: State) {
        for (const eid of state.query(PLAYER_BODIES)) {
            if (!pose(eid, _pose)) continue; // unregistered (the sweep hasn't built its CharState) — keep the fallback pose
            const [x, y, z] = _pose;
            if (state.has(eid, PlayerFollow)) {
                PlayerFollow.prev.set(
                    eid,
                    PlayerFollow.curr.x.get(eid),
                    PlayerFollow.curr.y.get(eid),
                    PlayerFollow.curr.z.get(eid),
                    0,
                );
            } else {
                // first snapshot: prev == curr, and membership becomes the "initialized" flag
                state.add(eid, PlayerFollow);
                PlayerFollow.prev.set(eid, x, y, z, 0);
            }
            PlayerFollow.curr.set(eid, x, y, z, 0);
        }
        // drop the follow state when a player is gone (mirrors the derived-state cleanup in orbit)
        for (const eid of state.query(ORPHAN_FOLLOWS)) state.remove(eid, PlayerFollow);
    },
};

// the player's render position: lerp between the two most recent fixed-tick poses by `fixedAlpha`. Falls back
// to the Body spawn pose (the CPU slab) until the first snapshot lands, so the first frames aren't at the origin.
function followPos(state: State, eid: number, out: [number, number, number]): void {
    if (state.has(eid, PlayerFollow)) {
        const a = state.time.fixedAlpha;
        out[0] = lerp(PlayerFollow.prev.x.get(eid), PlayerFollow.curr.x.get(eid), a);
        out[1] = lerp(PlayerFollow.prev.y.get(eid), PlayerFollow.curr.y.get(eid), a);
        out[2] = lerp(PlayerFollow.prev.z.get(eid), PlayerFollow.curr.z.get(eid), a);
        return;
    }
    out[0] = Body.pos.x.get(eid);
    out[1] = Body.pos.y.get(eid);
    out[2] = Body.pos.z.get(eid);
}

function findCamera(state: State, eid: number): number {
    const cam = Player.camera.get(eid);
    if (!cam || !state.has(cam, Camera)) {
        // warn once, latched on the derived PlayerFollow (added by the snapshot system); if it isn't up yet
        // (the character hasn't registered), skip — the next frame with a fresh pose warns.
        if (state.has(eid, PlayerFollow) && !PlayerFollow.warned.get(eid)) {
            PlayerFollow.warned.set(eid, 1);
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
 * the first-person controller: mouse-look + WASD/jump intent + the follow-camera pose, run in the
 * `simulation` group. Exported as an ordering anchor: a camera-juice / additive-pose system that perturbs
 * the camera on top of the controller's base pose declares `after: [PlayerControlSystem]`, reading the base
 * `Transform` this writes before `BeginFrameSystem` (draw) consumes it.
 */
export const PlayerControlSystem: System = {
    name: "control",
    group: "simulation",

    setup(state: State) {
        // the DOM canvas, not Views: setup runs in the simulation group, before BeginFrameSystem (draw) auto-
        // binds the camera View, so Views is still empty here. The canvas is mounted before run(), so the DOM
        // query (the same one run() uses) finds it regardless of View-attach timing.
        const canvas = typeof document === "undefined" ? null : document.querySelector("canvas");
        if (!canvas) return;
        // Pointer lock is requested by Input's canvas click handler, so the browser effect remains behind
        // the explicit public `requestPointerLock(state)` seam and retains the engagement gesture.
        const supported = typeof canvas.requestPointerLock === "function";
        requirePointerLock(state, supported);
        state.onDispose(() => {
            requirePointerLock(state, false);
            if (readPointerLockStatus(state) === "locked") exitLock();
        });
    },

    update(state: State) {
        // input suspended (a menu/cutscene): release the lock so the cursor frees + mouse-look stops, and let
        // the loop run with neutral device data — every key reads up, so move resolves to 0 and the player freezes.
        const input = devices(state);
        const active = inputEnabled(state);
        if (!active && input.pointer.lock.status === "locked") exitLock();
        for (const eid of state.query(PLAYER_BODIES)) {
            let yaw = Player.yaw.get(eid);
            let pitch = Player.pitch.get(eid);
            if (active && input.pointer.lock.status === "locked") {
                // Resolution-independent mouse-look. Pointer-lock movementX/Y is physical mouse motion in CSS
                // px — independent of canvas size — so the angle per pixel must NOT scale with the canvas.
                const s = Player.sensitivity.get(eid) / LOOK_REFERENCE_HEIGHT;
                yaw -= input.pointer.deltaX * s;
                pitch = clamp(pitch - input.pointer.deltaY * s, -MAX_PITCH, MAX_PITCH);
                Player.yaw.set(eid, yaw);
                Player.pitch.set(eid, pitch);
            }

            const cy = Math.cos(yaw);
            const sy = Math.sin(yaw);
            const sprint =
                input.keys.held.has("ShiftLeft") || input.keys.held.has("ShiftRight")
                    ? Player.sprint.get(eid)
                    : 1;

            let lx = 0;
            let lz = 0;
            if (input.keys.held.has("KeyW")) lz -= 1;
            if (input.keys.held.has("KeyS")) lz += 1;
            if (input.keys.held.has("KeyA")) lx -= 1;
            if (input.keys.held.has("KeyD")) lx += 1;
            const len = Math.hypot(lx, lz);
            if (len > 0) {
                const v = (Player.speed.get(eid) * sprint) / len;
                move(eid, (lz * sy + lx * cy) * v, (lz * cy - lx * sy) * v);
            } else {
                move(eid, 0, 0);
            }
            // one-shot: the press edge, not the held key. A held key refills the jump buffer every
            // frame, re-firing the instant the char re-grounds (a ledge, a landing); the buffer +
            // coyote forgiveness lives in the character pass.
            if (input.keys.pressed.has("Space")) jump(eid);

            const cam = findCamera(state, eid);
            if (cam < 0) continue;

            // pivot = the eye; the camera sits `distance` back along the look forward (0 = first-person).
            followPos(state, eid, _pos);
            const cp = Math.cos(pitch);
            const fx = -cp * sy;
            const fy = Math.sin(pitch);
            const fz = -cp * cy;
            const dist = Player.distance.get(eid);
            Transform.pos.set(
                cam,
                _pos[0] - fx * dist,
                _pos[1] + Player.eyeHeight.get(eid) - fy * dist,
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
    dependencies: [CharacterPlugin, InputPlugin, RenderPlugin, TransformsPlugin],
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
