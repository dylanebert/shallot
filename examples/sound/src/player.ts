import {
    Body,
    Character,
    CharacterPlugin,
    clamp,
    component,
    Devices,
    entity,
    f32,
    GlobalTransform,
    GroundState,
    InputPlugin,
    lerp,
    mountOverlay,
    not,
    type Plugin,
    pressKey,
    type Resource,
    releaseKey,
    releasePointerLock,
    requirePointerLock,
    Shape,
    type System,
    Time,
    type TouchPoint,
    Transform,
    u8,
    u32,
    Viewports,
    vec2,
    type World,
} from "@dylanebert/shallot";
import { PlayerFollow } from "./follow";
import { PlayerMotion } from "./motion";

const MAX_PITCH = Math.PI / 2 - 0.01;
// the look normalizes by this fixed reference height, never the live canvas size.
const LOOK_REFERENCE_HEIGHT = 1080;

/** First-person feel on a kinematic capsule Body with Character. Input is consumed on fixed ticks; the separate camera's Transform is authored during simulation using interpolated fixed placement. */
export const Player = component(
    "Player",
    {
        /**
         * Look yaw in radians around world Y. Set spawn facing here; each fixed tick copies PlayerInput.yaw.
         */
        yaw: f32,
        /**
         * Look pitch in radians. Set spawn tilt here; each fixed tick copies PlayerInput.pitch. Local look clamps it to ±90°.
         */
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
        /** Height the capsule floats above walkable ground, in metres. Player writes it to `Character.pogoRest` and turns on `Character.steepWalls`, so walking climbs a step only about as tall as the capsule's walkable rounding below `maxSlope`. */
        hover: f32,
        /** camera height above the capsule centre (the eye offset) */
        eyeHeight: f32,
        /** camera pull-back from the eye: 0 = first-person, > 0 = third-person */
        distance: f32,
        /** The linked camera: an entity with a Transform. Without a valid link, Player writes no camera pose. */
        camera: entity,
    },
    {
        defaults: () => ({
            yaw: 0,
            pitch: 0,
            speed: 4.317,
            sprint: 1.3,
            acceleration: 20,
            friction: 16,
            gravity: 29.5,
            jumpSpeed: 8.85,
            coyoteTime: 0.1,
            jumpBuffer: 0.1,
            hover: 0.05,
            eyeHeight: 0.7,
            distance: 0,
            camera: 0,
        }),
    },
);

/** Pointer-lock reads are World-scoped. */
export type { PointerLockStatus } from "@dylanebert/shallot";

function setupLocalControls(world: World): void {
    if (typeof document !== "undefined") {
        const overlay = mountOverlay(document.querySelector("canvas"), world);
        mountTouchButton(overlay, world, "Jump", "Space");
        const stick = document.createElement("div");
        stick.dataset.touchStick = "";
        stick.style.cssText =
            "display:none;position:absolute;width:140px;height:140px;border:1px solid #797369;border-radius:50%;background:#282622cc;pointer-events:none;transform:translate(-50%,-50%)";
        const knob = document.createElement("div");
        knob.style.cssText =
            "position:absolute;left:50%;top:50%;width:48px;height:48px;border-radius:50%;background:#eee9df80";
        stick.append(knob);
        overlay.append(stick);
        Object.assign(world.resource(TouchControls), { stick, knob });
        world.onDispose(() => stick.remove());
    }
    requirePointerLock(world, true);
    world.onDispose(() => {
        releasePointerLock(world);
    });
}

/** Per-tick actions for one player. Producers write this record; fixed gameplay reads it without sampling devices. */
export const PlayerInput = component(
    "PlayerInput",
    {
        /** held lateral (positive right) and forward (positive forward) axes in [-1, 1] */
        move: vec2,
        /** whether sprint is held */
        sprint: u8,
        /** absolute look yaw in radians */
        yaw: f32,
        /** absolute look pitch in radians */
        pitch: f32,
        /** cumulative number of jump presses produced for this player */
        jumpPresses: u32,
    },
    { defaults: () => ({ move: [0, 0], sprint: 0, yaw: 0, pitch: 0, jumpPresses: 0 }) },
);

/** Local device ownership and the camera angles accumulated before each fixed loop. */
export const LocalPlayer = component(
    "LocalPlayer",
    {
        /** look radians per mouse CSS pixel at the fixed 1080px reference height */
        sensitivity: f32,
        /** touch look degrees per CSS pixel of drag */
        touchSensitivity: f32,
        /** locally accumulated view yaw in radians */
        viewYaw: f32,
        /** locally accumulated view pitch in radians */
        viewPitch: f32,
        /** whether the view angles have been seeded from Player */
        initialized: u8,
    },
    {
        defaults: () => ({
            sensitivity: 1.5,
            touchSensitivity: 0.5,
            viewYaw: 0,
            viewPitch: 0,
            initialized: 0,
        }),
    },
);

const LOCAL_PLAYERS = [LocalPlayer, Player];

export const TouchControls: Resource<{
    used: boolean;
    buttons: HTMLButtonElement[];
    stick?: HTMLDivElement;
    knob?: HTMLDivElement;
}> = { create: () => ({ used: false, buttons: [] }) };

/** Mount an action on the example overlay; touch reveals it and feeds the keyboard producer. */
export function mountTouchButton(
    parent: HTMLElement,
    world: World,
    label: string,
    key: string,
): void {
    const controls = world.resource(TouchControls);
    const button = document.createElement("button");
    button.textContent = label;
    button.style.cssText = `display:none;position:absolute;right:24px;bottom:${key === "Space" ? 32 : 104}px;width:64px;height:64px;border-radius:50%;border:1px solid #797369;background:#282622cc;color:#eee9df;font:400 14px/1.3 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;pointer-events:auto;touch-action:none`;
    button.addEventListener("pointerdown", (event) => {
        event.stopPropagation();
        event.preventDefault();
        pressKey(world, key);
        try {
            button.setPointerCapture(event.pointerId);
        } catch {
            // An inactive pointer can refuse capture; its press still belongs to this frame.
        }
    });
    const release = () => releaseKey(world, key);
    button.addEventListener("pointerup", release);
    button.addEventListener("pointercancel", release);
    button.addEventListener("lostpointercapture", release);
    controls.buttons.push(button);
    parent.append(button);
    world.onDispose(() => button.remove());
}

/** Sample the local devices once per frame, before any fixed ticks. */
const LocalPlayerInputSystem: System = {
    name: "local-input",
    group: "setup",
    update(world) {
        const devices = world.resource(Devices);
        const active = devices.enabled;
        const controls = world.resource(TouchControls);
        const width = world.resource(Viewports).get(devices.focused)?.cssWidth ?? 0;
        let stick: TouchPoint | undefined;
        let look: TouchPoint | undefined;
        for (const point of devices.touch.held.values()) {
            if (point.startX < width / 2) stick ??= point;
            else look ??= point;
        }
        let stickX = 0;
        let stickY = 0;
        if (stick && active) {
            const x = stick.x - stick.startX;
            const y = stick.y - stick.startY;
            const magnitude = Math.hypot(x, y);
            if (magnitude > 7) {
                const scale = Math.min(1, magnitude / 70) / magnitude;
                stickX = x * scale;
                stickY = -y * scale;
            }
        }
        if (devices.touch.pressed.size) controls.used = true;
        for (const button of controls.buttons) button.style.display = controls.used ? "" : "none";
        if (controls.stick && controls.knob) {
            controls.stick.style.display = stick && active ? "" : "none";
            if (stick) {
                controls.stick.style.left = `${stick.startX}px`;
                controls.stick.style.top = `${stick.startY}px`;
                controls.knob.style.transform = `translate(calc(-50% + ${stickX * 70}px), calc(-50% - ${stickY * 70}px))`;
            }
        }
        const players = world.storage(Player);
        const local = world.storage(LocalPlayer);
        const inputs = world.storage(PlayerInput);
        for (const eid of world.query(LOCAL_PLAYERS)) {
            if (!world.has(eid, PlayerInput)) {
                world.add(eid, PlayerInput, {
                    yaw: players.yaw.get(eid),
                    pitch: players.pitch.get(eid),
                });
            }
            if (!local.initialized.get(eid)) {
                local.viewYaw.set(eid, players.yaw.get(eid));
                local.viewPitch.set(eid, players.pitch.get(eid));
                local.initialized.set(eid, 1);
            }

            let yaw = local.viewYaw.get(eid);
            let pitch = local.viewPitch.get(eid);
            if (
                active &&
                devices.touch.held.size === 0 &&
                devices.pointer.lock.status === "locked"
            ) {
                const sensitivity = local.sensitivity.get(eid) / LOOK_REFERENCE_HEIGHT;
                yaw -= devices.pointer.deltaX * sensitivity;
                pitch = clamp(pitch - devices.pointer.deltaY * sensitivity, -MAX_PITCH, MAX_PITCH);
                local.viewYaw.set(eid, yaw);
                local.viewPitch.set(eid, pitch);
            }

            if (active && look) {
                const sensitivity = (local.touchSensitivity.get(eid) * Math.PI) / 180;
                yaw -= look.deltaX * sensitivity;
                pitch = clamp(pitch - look.deltaY * sensitivity, -MAX_PITCH, MAX_PITCH);
                local.viewYaw.set(eid, yaw);
                local.viewPitch.set(eid, pitch);
            }

            const move = inputs.move.column;
            const offset = eid * 2;
            move[offset] = active
                ? Number(devices.keys.held.has("KeyD")) - Number(devices.keys.held.has("KeyA"))
                : 0;
            move[offset + 1] = active
                ? Number(devices.keys.held.has("KeyW")) - Number(devices.keys.held.has("KeyS"))
                : 0;
            move[offset] = clamp(move[offset] + stickX, -1, 1);
            move[offset + 1] = clamp(move[offset + 1] + stickY, -1, 1);
            inputs.move.markChanged(eid);
            inputs.sprint.set(
                eid,
                Number(
                    active &&
                        (devices.keys.held.has("ShiftLeft") || devices.keys.held.has("ShiftRight")),
                ),
            );
            inputs.yaw.set(eid, yaw);
            inputs.pitch.set(eid, pitch);
            if (active && !world.time.paused && devices.keys.pressed.has("Space")) {
                inputs.jumpPresses.set(eid, inputs.jumpPresses.get(eid) + 1);
            }
        }
    },
};

/** Fixed-tick input and feel, before the published character movement systems. */
export const DrivePlayerSystem: System = {
    name: "drive",
    group: "fixed",
    before: CharacterPlugin.systems,
    update(world) {
        const players = world.storage(Player);
        const characters = world.storage(Character);
        const motion = world.storage(PlayerMotion);
        const inputs = world.storage(PlayerInput);
        const shapes = world.storage(Shape);
        const dt = Time.FIXED_DT;
        for (const eid of world.query(DRIVEN_PLAYERS)) {
            if (!world.has(eid, PlayerMotion)) world.add(eid, PlayerMotion);
            if (!world.has(eid, PlayerInput)) {
                world.add(eid, PlayerInput, {
                    yaw: players.yaw.get(eid),
                    pitch: players.pitch.get(eid),
                });
            }
            const pogoRest = 1 + players.hover.column[eid] / shapes.radius.column[eid];
            if (characters.pogoRest.column[eid] !== Math.fround(pogoRest))
                characters.pogoRest.set(eid, pogoRest);
            if (characters.steepWalls.column[eid] !== 1) characters.steepWalls.set(eid, 1);
            const grounded = characters.groundState.column[eid] === GroundState.OnGround;
            let coyote = grounded
                ? players.coyoteTime.column[eid]
                : Math.max(0, motion.coyote.column[eid] - dt);
            let buffer = Math.max(0, motion.buffer.column[eid] - dt);
            const jumpPresses = inputs.jumpPresses.column[eid];
            if (jumpPresses !== motion.lastJumpPresses.column[eid]) {
                buffer = players.jumpBuffer.column[eid];
            }
            motion.lastJumpPresses.set(eid, jumpPresses);
            const yaw = inputs.yaw.column[eid];
            players.yaw.set(eid, yaw);
            players.pitch.set(eid, inputs.pitch.column[eid]);
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
            const inputOffset = eid * 2;
            const lx = inputs.move.column[inputOffset];
            const forward = inputs.move.column[inputOffset + 1];
            const length = Math.sqrt(lx * lx + forward * forward);
            const maxSpeed =
                players.speed.column[eid] *
                (grounded && inputs.sprint.column[eid] !== 0 ? players.sprint.column[eid] : 1);
            if (length > 0) {
                const dx = (-forward * Math.sin(yaw) + lx * Math.cos(yaw)) / length;
                const dz = (-forward * Math.cos(yaw) - lx * Math.sin(yaw)) / length;
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
const DRIVEN_PLAYERS = [Player, Body, Character];
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
        for (const eid of world.query(ORPHAN_FOLLOWS)) world.remove(eid, PlayerFollow);
    },
};

// the player's render position: lerp between the two most recent fixed-tick GlobalTransform positions by `fixedAlpha`.
// Falls back to Transform until the first snapshot lands, so the first frames aren't at the origin.
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
    const position = world.storage(Transform).translation.column;
    out[0] = position[offset];
    out[1] = position[offset + 1];
    out[2] = position[offset + 2];
}

function findCamera(world: World, eid: number): number {
    const cam = world.storage(Player).camera.get(eid);
    if (!cam || !world.has(cam, Transform)) {
        // Warning requires the snapshot's PlayerFollow latch; earlier calls silently skip the missing camera.
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
 * Pose the follow-camera Transform in the `simulation` group. Exported as an ordering anchor: a camera-juice system that perturbs the authored camera
 * placement declares `after: [UpdatePlayerControlSystem]`, reading the base
 * `Transform` this writes before `BeginFrameSystem` (draw) consumes it.
 */
export const UpdatePlayerControlSystem: System = {
    name: "control",
    group: "simulation",

    setup: setupLocalControls,

    update(world: World) {
        const players = world.storage(Player);
        const local = world.storage(LocalPlayer);
        for (const eid of world.query(PLAYER_BODIES)) {
            const yaw = world.has(eid, LocalPlayer)
                ? local.viewYaw.column[eid]
                : players.yaw.column[eid];
            const pitch = world.has(eid, LocalPlayer)
                ? local.viewPitch.column[eid]
                : players.pitch.column[eid];
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
            const eyeY = _pos[1] + world.storage(Player).eyeHeight.column[eid];
            world
                .storage(Transform)
                .translation.set(cam, _pos[0] - fx * dist, eyeY - fy * dist, _pos[2] - fz * dist);
            setLook(world, cam, yaw, pitch);
        }
    },
};

/** First-person feel and camera over standard physics's velocity-driven capsule. */
export const PlayerPlugin: Plugin = {
    name: "Player",
    // Fixed movement and follow history live in components; Devices belongs to Input.
    recovery: "stateless",
    systems: [
        DrivePlayerSystem,
        SnapshotPlayerPositionSystem,
        UpdatePlayerControlSystem,
        LocalPlayerInputSystem,
    ],
    components: [PlayerMotion, Player, PlayerInput, LocalPlayer],
    dependencies: [CharacterPlugin, InputPlugin],
};
