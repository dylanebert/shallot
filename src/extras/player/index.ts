import {
    Devices,
    InputPlugin,
    inputEnabled,
    releasePointerLock,
    requirePointerLock,
} from "../../core/input";
import { Camera, RenderPlugin } from "../../core/rendering";
import {
    entity,
    f32,
    not,
    type Plugin,
    registration,
    type System,
    Transform,
    type World,
} from "../../engine";
import { clamp, lerp } from "../../engine/utils";
import {
    Character,
    CharacterPlugin,
    globalTransform,
    jump,
    move,
    SweepCharactersSystem,
} from "../../transitional/character";
import { Body } from "../../transitional/physics";
import { PlayerFollow } from "./follow";

// First-person player controller — composes a kinematic `Character` (the §6.4 controller) with WASD + a
// pointer-lock mouse look + a follow camera. The Player entity IS the character's capsule body (Body +
// Character, mass <= 0); a separate camera entity (Camera + a renderer marker + Transform) is linked via
// `Player.camera`. The controller owns the look (yaw/pitch, instant) + the move/jump intent (driven through
// the `character` module's eid-keyed `move`/`jump`); the CPU sweep produces fixed-tick GlobalTransform.
// The camera follows its position with fixed-timestep interpolation: a
// `fixed`-group system (`after: [SweepCharactersSystem]`) snapshots this tick's swept GlobalTransform (`character.globalTransform`,
// off the CPU controller state) into previous/current, and the camera renders `lerp(previous, current, fixedAlpha)` — see
// `SnapshotPlayerPositionSystem`. Input → GlobalTransform → camera position carries no readback lag (it stops scaling with
// GPU frame time, mouse-look already did); the only camera latency is the kept one-tick interpolation +
// the irreducible display fence. Walk/jump/slope tuning lives on `Character`.
//
// The rig is written pivot-first so third-person drops in later: the camera sits at `pivot − forward·distance`
// where `pivot = charPos + eyeHeight` and `distance` defaults to 0 (first-person — the camera is AT the eye).
// A future third-person mode sets `Player.distance > 0`; nothing else here changes.

const MAX_PITCH = Math.PI / 2 - 0.01;
// the look normalizes by this fixed reference height, never the live canvas — the why is in
// UpdatePlayerControlSystem.update (resolution-independence).
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
 * const body = world.create();
 * world.add(body, Body); world.add(body, Character); world.add(body, Player);   // a capsule, mass 0
 * const bodies = world.storage(Body);
 * const characters = world.storage(Character);
 * bodies.shape.set(body, ShapeKind.Capsule);
 * bodies.halfExtents.set(body, 0, 0.6, 0, 0.3); bodies.mass.set(body, 0);
 * characters.jumpSpeed.set(body, 6); characters.gravity.set(body, -30);         // snappy jump/fall
 * const cam = world.create();
 * world.add(cam, Transform); world.add(cam, Camera); world.add(cam, StandardRenderer);
 * world.storage(Player).camera.set(body, cam);
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

/** Pointer-lock reads are World-scoped. */
export type { PointerLockStatus } from "../../core/input";

function setupPointerLock(world: World): void {
    requirePointerLock(world, true);
    world.onDispose(() => {
        releasePointerLock(world);
    });
}

// scratch for the per-tick swept GlobalTransform read (character.globalTransform), reused across players.
const _globalTransform: [number, number, number] = [0, 0, 0];
// query terms held once, so a steady frame mints no array.
const PLAYER_BODIES = [Player, Body];
const ORPHAN_FOLLOWS = [not(Player), PlayerFollow];

// Snapshot the character's fixed-tick GlobalTransform position once per tick, so the camera can
// render-interpolate it by `fixedAlpha`, matching the engine renderer. The character controller writes
// GlobalTransform this tick, so this system runs `after: [SweepCharactersSystem]`; it does not read back
// from the GPU. Capturing on the fixed clock is what keeps
// the camera smooth at ANY render rate; the only camera lag is the kept one-tick interpolation, no readback.
const SnapshotPlayerPositionSystem: System = {
    name: "snapshot",
    group: "fixed",
    after: [SweepCharactersSystem],
    update(world: World) {
        for (const eid of world.query(PLAYER_BODIES)) {
            if (!globalTransform(world, eid, _globalTransform)) continue; // The body producer has not registered yet.
            const x = _globalTransform[0];
            const y = _globalTransform[1];
            const z = _globalTransform[2];
            if (world.has(eid, PlayerFollow)) {
                world
                    .storage(PlayerFollow)
                    .previous.set(
                        eid,
                        world.storage(PlayerFollow).current.x.get(eid),
                        world.storage(PlayerFollow).current.y.get(eid),
                        world.storage(PlayerFollow).current.z.get(eid),
                        0,
                    );
            } else {
                // first snapshot: previous == current, and membership becomes the "initialized" flag
                world.add(eid, PlayerFollow);
                world.storage(PlayerFollow).previous.set(eid, x, y, z, 0);
            }
            world.storage(PlayerFollow).current.set(eid, x, y, z, 0);
        }
        // drop the follow state when a player is gone (mirrors the derived-state cleanup in orbit)
        for (const eid of world.query(ORPHAN_FOLLOWS)) world.remove(eid, PlayerFollow);
    },
};

// the player's render position: lerp between the two most recent fixed-tick GlobalTransform positions by `fixedAlpha`.
// Falls back to the authored Body spawn position until the first snapshot lands, so the first frames aren't at the origin.
function followPos(world: World, eid: number, out: [number, number, number]): void {
    if (world.has(eid, PlayerFollow)) {
        const a = world.time.fixedAlpha;
        out[0] = lerp(
            world.storage(PlayerFollow).previous.x.get(eid),
            world.storage(PlayerFollow).current.x.get(eid),
            a,
        );
        out[1] = lerp(
            world.storage(PlayerFollow).previous.y.get(eid),
            world.storage(PlayerFollow).current.y.get(eid),
            a,
        );
        out[2] = lerp(
            world.storage(PlayerFollow).previous.z.get(eid),
            world.storage(PlayerFollow).current.z.get(eid),
            a,
        );
        return;
    }
    out[0] = world.storage(Body).position.x.get(eid);
    out[1] = world.storage(Body).position.y.get(eid);
    out[2] = world.storage(Body).position.z.get(eid);
}

function findCamera(world: World, eid: number): number {
    const cam = world.storage(Player).camera.get(eid);
    if (!cam || !world.has(cam, Camera)) {
        // warn once, latched on the derived PlayerFollow (added by the snapshot system); if it isn't up yet
        // (the character hasn't registered), skip — the next frame with GlobalTransform warns.
        if (
            world.has(eid, PlayerFollow) &&
            !world.storage(PlayerFollow).missingCameraWarned.get(eid)
        ) {
            world.storage(PlayerFollow).missingCameraWarned.set(eid, 1);
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
function setLook(world: World, cam: number, yaw: number, pitch: number): void {
    const hy = yaw * 0.5;
    const hp = pitch * 0.5;
    const sy = Math.sin(hy);
    const cy = Math.cos(hy);
    const sp = Math.sin(hp);
    const cp = Math.cos(hp);
    world.storage(Transform).rotation.set(cam, cy * sp, sy * cp, -sy * sp, cy * cp);
}

const _pos: [number, number, number] = [0, 0, 0];

/**
 * the first-person controller: mouse-look + WASD/jump intent + the follow-camera Transform, run in the
 * `simulation` group. Exported as an ordering anchor: a camera-juice system that perturbs the authored camera
 * placement declares `after: [UpdatePlayerControlSystem]`, reading the base
 * `Transform` this writes before `BeginFrameSystem` (draw) consumes it.
 */
export const UpdatePlayerControlSystem: System = {
    name: "control",
    group: "simulation",

    setup: setupPointerLock,

    update(world: World) {
        // input suspended (a menu/cutscene): release the lock so the cursor frees + mouse-look stops, and let
        // the loop run with neutral device data — every key reads up, so move resolves to 0 and the player freezes.
        const input = world.resource(Devices);
        const active = inputEnabled(world);
        for (const eid of world.query(PLAYER_BODIES)) {
            let yaw = world.storage(Player).yaw.get(eid);
            let pitch = world.storage(Player).pitch.get(eid);
            if (active && input.pointer.lock.status === "locked") {
                // Resolution-independent mouse-look. Pointer-lock movementX/Y is physical mouse motion in CSS
                // px — independent of canvas size — so the angle per pixel must NOT scale with the canvas.
                const s = world.storage(Player).sensitivity.get(eid) / LOOK_REFERENCE_HEIGHT;
                yaw -= input.pointer.deltaX * s;
                pitch = clamp(pitch - input.pointer.deltaY * s, -MAX_PITCH, MAX_PITCH);
                world.storage(Player).yaw.set(eid, yaw);
                world.storage(Player).pitch.set(eid, pitch);
            }

            const cy = Math.cos(yaw);
            const sy = Math.sin(yaw);
            const sprint =
                input.keys.held.has("ShiftLeft") || input.keys.held.has("ShiftRight")
                    ? world.storage(Player).sprint.get(eid)
                    : 1;

            let lx = 0;
            let lz = 0;
            if (input.keys.held.has("KeyW")) lz -= 1;
            if (input.keys.held.has("KeyS")) lz += 1;
            if (input.keys.held.has("KeyA")) lx -= 1;
            if (input.keys.held.has("KeyD")) lx += 1;
            const len = Math.sqrt(lx * lx + lz * lz);
            if (len > 0) {
                const v = (world.storage(Player).speed.get(eid) * sprint) / len;
                move(world, eid, (lz * sy + lx * cy) * v, (lz * cy - lx * sy) * v);
            } else {
                move(world, eid, 0, 0);
            }
            // one-shot: the press edge, not the held key. A held key refills the jump buffer every
            // frame, re-firing the instant the char re-grounds (a ledge, a landing); the buffer +
            // coyote forgiveness lives in the character pass.
            if (input.keys.pressed.has("Space")) jump(world, eid);

            const cam = findCamera(world, eid);
            if (cam < 0) continue;

            // pivot = the eye; the camera sits `distance` back along the look forward (0 = first-person).
            followPos(world, eid, _pos);
            const cp = Math.cos(pitch);
            const fx = -cp * sy;
            const fy = Math.sin(pitch);
            const fz = -cp * cy;
            const dist = world.storage(Player).distance.get(eid);
            world
                .storage(Transform)
                .translation.set(
                    cam,
                    _pos[0] - fx * dist,
                    _pos[1] + world.storage(Player).eyeHeight.get(eid) - fy * dist,
                    _pos[2] - fz * dist,
                    1,
                );
            setLook(world, cam, yaw, pitch);
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
    systems: [SnapshotPlayerPositionSystem, UpdatePlayerControlSystem],
    components: [
        registration("Player", Player, {
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
        }),
    ],
    dependencies: [CharacterPlugin, InputPlugin, RenderPlugin],
};
