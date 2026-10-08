import {
    Devices,
    InputPlugin,
    inputEnabled,
    releasePointerLock,
    requirePointerLock,
} from "../../core/input";
import { Body } from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import {
    component,
    entity,
    f32,
    not,
    type Plugin,
    type System,
    Time,
    vec4,
    type World,
} from "../../engine";
import { clamp, lerp } from "../../engine/utils";
import { Character, CharacterPlugin, GroundState } from "../../standard/physics";
import { PlayerFollow } from "./follow";

const MAX_PITCH = Math.PI / 2 - 0.01;
// the look normalizes by this fixed reference height, never the live canvas — the why is in
// UpdatePlayerControlSystem.update (resolution-independence).
const LOOK_REFERENCE_HEIGHT = 1080;

/** First-person feel on a kinematic capsule Body with Character. Input is consumed on fixed ticks; the separate camera's Transform is authored during simulation using interpolated fixed placement. */
export const Player = component(
    "Player",
    {
        /** look yaw in radians (turn around world Y); set it to face a direction at spawn */
        yaw: f32,
        /** look pitch in radians (clamped to ±90°); set it to tilt the view at spawn */
        pitch: f32,
        /** walk speed (m/s) the move input is scaled to */
        speed: f32,
        /** Acceleration coefficient in inverse seconds, multiplied by maximum speed. */
        acceleration: f32,
        /** Horizontal damping coefficient in inverse seconds. */
        friction: f32,
        /** Downward acceleration in metres per second squared. */
        gravity: f32,
        /** Launch speed relative to the ground in metres per second. */
        jumpSpeed: f32,
        /** Grace period after leaving walkable ground, in seconds. */
        coyoteTime: f32,
        /** Lifetime of a jump press before landing, in seconds. */
        jumpBuffer: f32,
        /** sprint multiplier applied while Shift is held */
        sprint: f32,
        /** mouse-look radians per pixel of pointer-lock movement, at a fixed 1080px reference height (the look
         * speed is resolution-independent, so the same mouse motion turns the same angle at any canvas size) */
        sensitivity: f32,
        /** camera height above the capsule centre (the eye offset) */
        eyeHeight: f32,
        /** camera pull-back from the eye: 0 = first-person, > 0 = third-person (scaffolding) */
        distance: f32,
        /** The linked camera: an entity with a Transform. Without a valid link, Player writes no camera pose. */
        camera: entity,
    },
    {
        defaults: () => ({
            yaw: 0,
            pitch: 0,
            speed: 6,
            sprint: 1.5,
            acceleration: 30,
            friction: 4,
            gravity: 15,
            jumpSpeed: 5,
            coyoteTime: 0.15,
            jumpBuffer: 0.2,
            sensitivity: 1.5,
            eyeHeight: 0.7,
            distance: 0,
            camera: 0,
        }),
    },
);

/** Pointer-lock reads are World-scoped. */
export type { PointerLockStatus } from "../../core/input";

function setupPointerLock(world: World): void {
    requirePointerLock(world, true);
    world.onDispose(() => {
        releasePointerLock(world);
    });
}

const PlayerMotion = component("PlayerMotion", {
    carry: vec4,
    coyote: f32,
    buffer: f32,
});

/** Fixed-tick input and feel, before the published character movement systems. */
export const DrivePlayerSystem: System = {
    name: "drive",
    group: "fixed",
    before: CharacterPlugin.systems,
    update(world) {
        const players = world.storage(Player);
        const characters = world.storage(Character);
        const motion = world.storage(PlayerMotion);
        const input = world.resource(Devices);
        const dt = Time.FIXED_DT;
        for (const eid of world.query(PLAYER_BODIES)) {
            if (!world.has(eid, PlayerMotion)) world.add(eid, PlayerMotion);
            const grounded = characters.groundState.column[eid] === GroundState.OnGround;
            let coyote = grounded
                ? players.coyoteTime.column[eid]
                : Math.max(0, motion.coyote.column[eid] - dt);
            let buffer = Math.max(0, motion.buffer.column[eid] - dt);
            if (input.keys.pressed.has("Space")) buffer = players.jumpBuffer.column[eid];
            let vx = characters.velocity.column[eid * 4] - motion.carry.column[eid * 4];
            let vy = characters.velocity.column[eid * 4 + 1] - motion.carry.column[eid * 4 + 1];
            let vz = characters.velocity.column[eid * 4 + 2] - motion.carry.column[eid * 4 + 2];
            const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
            if (speed < 0.01) {
                vx = 0;
                vz = 0;
            } else {
                const ratio =
                    Math.max(0, speed - Math.max(1, speed) * players.friction.column[eid] * dt) /
                    speed;
                vx *= ratio;
                vz *= ratio;
            }
            if (grounded) vy = 0;
            const yaw = players.yaw.column[eid];
            let lx = 0,
                lz = 0;
            if (inputEnabled(world)) {
                if (input.keys.held.has("KeyW")) lz--;
                if (input.keys.held.has("KeyS")) lz++;
                if (input.keys.held.has("KeyA")) lx--;
                if (input.keys.held.has("KeyD")) lx++;
            }
            const length = Math.sqrt(lx * lx + lz * lz);
            const maxSpeed =
                players.speed.column[eid] *
                (grounded && (input.keys.held.has("ShiftLeft") || input.keys.held.has("ShiftRight"))
                    ? players.sprint.column[eid]
                    : 1);
            if (length > 0) {
                const dx = (lz * Math.sin(yaw) + lx * Math.cos(yaw)) / length;
                const dz = (lz * Math.cos(yaw) - lx * Math.sin(yaw)) / length;
                const acceleration = Math.min(
                    Math.max(0, maxSpeed - vx * dx - vz * dz),
                    players.acceleration.column[eid] * maxSpeed * dt,
                );
                vx += acceleration * dx;
                vz += acceleration * dz;
            }
            if (buffer > 0 && (grounded || coyote > 0)) {
                vy = players.jumpSpeed.column[eid];
                coyote = 0;
                buffer = 0;
            }
            vy -= players.gravity.column[eid] * dt;
            const gx = grounded ? characters.groundVelocity.column[eid * 4] : 0;
            const gy = grounded ? characters.groundVelocity.column[eid * 4 + 1] : 0;
            const gz = grounded ? characters.groundVelocity.column[eid * 4 + 2] : 0;
            const offset = eid * 4;
            const velocity = characters.velocity.column;
            velocity[offset] = vx + gx;
            velocity[offset + 1] = vy + gy;
            velocity[offset + 2] = vz + gz;
            velocity[offset + 3] = 0;
            characters.velocity.markChanged(eid);
            const carry = motion.carry.column;
            carry[offset] = gx;
            carry[offset + 1] = gy;
            carry[offset + 2] = gz;
            motion.carry.markChanged(eid);
            motion.coyote.column[eid] = coyote;
            motion.coyote.markChanged(eid);
            motion.buffer.column[eid] = buffer;
            motion.buffer.markChanged(eid);
        }
    },
};
// query terms held once, so a steady frame mints no array.
const PLAYER_BODIES = [Player, Body];
const ORPHAN_FOLLOWS = [not(Player), PlayerFollow];

// Camera interpolation samples the body's fixed placement rather than a rendered or read-back pose.
const SnapshotPlayerPositionSystem: System = {
    name: "snapshot",
    group: "fixed",
    after: CharacterPlugin.systems,
    update(world: World) {
        for (const eid of world.query(PLAYER_BODIES)) {
            if (!world.has(eid, GlobalTransform)) continue;
            const pose = world.storage(GlobalTransform).translation;
            const offset = eid * 4;
            const x = pose.column[offset];
            const y = pose.column[offset + 1];
            const z = pose.column[offset + 2];
            const initialized = world.has(eid, PlayerFollow);
            if (!initialized) world.add(eid, PlayerFollow);
            const follow = world.storage(PlayerFollow);
            const previous = follow.previous.column;
            const current = follow.current.column;
            previous[offset] = initialized ? current[offset] : x;
            previous[offset + 1] = initialized ? current[offset + 1] : y;
            previous[offset + 2] = initialized ? current[offset + 2] : z;
            follow.previous.markChanged(eid);
            current[offset] = x;
            current[offset + 1] = y;
            current[offset + 2] = z;
            follow.current.markChanged(eid);
        }
        // drop the follow state when a player is gone (mirrors the derived-state cleanup in orbit)
        for (const eid of world.query(ORPHAN_FOLLOWS)) world.remove(eid, PlayerFollow);
    },
};

// the player's render position: lerp between the two most recent fixed-tick GlobalTransform positions by `fixedAlpha`.
// Falls back to the authored Body spawn position until the first snapshot lands, so the first frames aren't at the origin.
function followPos(world: World, eid: number, out: [number, number, number]): void {
    const offset = eid * 4;
    if (world.has(eid, PlayerFollow)) {
        const a = world.time.fixedAlpha;
        const follow = world.storage(PlayerFollow);
        const previous = follow.previous.column;
        const current = follow.current.column;
        out[0] = lerp(previous[offset], current[offset], a);
        out[1] = lerp(previous[offset + 1], current[offset + 1], a);
        out[2] = lerp(previous[offset + 2], current[offset + 2], a);
        return;
    }
    const position = world.storage(Body).position.column;
    out[0] = position[offset];
    out[1] = position[offset + 1];
    out[2] = position[offset + 2];
}

function findCamera(world: World, eid: number): number {
    const cam = world.storage(Player).camera.get(eid);
    if (!cam || !world.has(cam, Transform)) {
        // warn once, latched on the derived PlayerFollow (added by the snapshot system); if it isn't up yet
        // (the character hasn't registered), skip — the next frame with GlobalTransform warns.
        if (
            world.has(eid, PlayerFollow) &&
            !world.storage(PlayerFollow).missingCameraWarned.get(eid)
        ) {
            world.storage(PlayerFollow).missingCameraWarned.set(eid, 1);
            console.warn(
                `[player] entity ${eid}: Player.camera must point to an entity with a Transform`,
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
    const rotation = world.storage(Transform).rotation;
    const offset = cam * 4;
    rotation.column[offset] = cy * sp;
    rotation.column[offset + 1] = sy * cp;
    rotation.column[offset + 2] = -sy * sp;
    rotation.column[offset + 3] = cy * cp;
    rotation.markChanged(cam);
}

const _pos: [number, number, number] = [0, 0, 0];

/**
 * Mouse-look and the follow-camera Transform, run in the
 * `simulation` group. Exported as an ordering anchor: a camera-juice system that perturbs the authored camera
 * placement declares `after: [UpdatePlayerControlSystem]`, reading the base
 * `Transform` this writes before `BeginFrameSystem` (draw) consumes it.
 */
export const UpdatePlayerControlSystem: System = {
    name: "control",
    group: "simulation",

    setup: setupPointerLock,

    update(world: World) {
        // Suspended input leaves presentation following the physics pose without consuming look.
        const input = world.resource(Devices);
        const active = inputEnabled(world);
        for (const eid of world.query(PLAYER_BODIES)) {
            let yaw = world.storage(Player).yaw.column[eid];
            let pitch = world.storage(Player).pitch.column[eid];
            if (active && input.pointer.lock.status === "locked") {
                // Resolution-independent mouse-look. Pointer-lock movementX/Y is physical mouse motion in CSS
                // px — independent of canvas size — so the angle per pixel must NOT scale with the canvas.
                const s = world.storage(Player).sensitivity.column[eid] / LOOK_REFERENCE_HEIGHT;
                yaw -= input.pointer.deltaX * s;
                pitch = clamp(pitch - input.pointer.deltaY * s, -MAX_PITCH, MAX_PITCH);
                world.storage(Player).yaw.set(eid, yaw);
                world.storage(Player).pitch.set(eid, pitch);
            }

            const sy = Math.sin(yaw);
            const cy = Math.cos(yaw);

            const cam = findCamera(world, eid);
            if (cam < 0) continue;

            // pivot = the eye; the camera sits `distance` back along the look forward (0 = first-person).
            followPos(world, eid, _pos);
            const cp = Math.cos(pitch);
            const fx = -cp * sy;
            const fy = Math.sin(pitch);
            const fz = -cp * cy;
            const dist = world.storage(Player).distance.column[eid];
            const translation = world.storage(Transform).translation;
            const offset = cam * 4;
            translation.column[offset] = _pos[0] - fx * dist;
            translation.column[offset + 1] =
                _pos[1] + world.storage(Player).eyeHeight.column[eid] - fy * dist;
            translation.column[offset + 2] = _pos[2] - fz * dist;
            translation.column[offset + 3] = 1;
            translation.markChanged(cam);
            setLook(world, cam, yaw, pitch);
        }

        // InputPlugin clears the shared pointer delta at the draw boundary.
    },
};

/** First-person feel and camera over standard physics's velocity-driven capsule. */
export const PlayerPlugin: Plugin = {
    name: "Player",
    // Fixed movement and follow history live in components; Devices belongs to Input.
    recovery: "stateless",
    systems: [DrivePlayerSystem, SnapshotPlayerPositionSystem, UpdatePlayerControlSystem],
    components: [PlayerMotion, Player],
    dependencies: [CharacterPlugin, InputPlugin],
};
