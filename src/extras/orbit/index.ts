import { Devices, InputPlugin, type Pointer } from "../../core/input";
import { Camera, CameraMode } from "../../core/rendering";
import { cameraView } from "../../core/rendering/view";
import {
    deriveTransforms,
    GlobalTransform,
    Transform,
    TransformPlugin,
} from "../../core/transform";
import {
    component,
    entity,
    f32,
    not,
    type Plugin,
    type System,
    u8,
    Viewports,
    vec3,
    type World,
} from "../../engine";
import { clamp, lookAtRotationInto } from "../../engine/utils";
import { followPosition, OrbitFollowPlugin } from "./follow";
import { OrbitSmooth } from "./smooth";

const Tau = Math.PI * 2;
const Deg2Rad = Math.PI / 180;

// Multiplicative fly-speed changes make wheel input feel equal across speed magnitudes.
const FlyScrollRate = Math.log(1.15) / 100; // ≈ 0.0014

/** `Free` consumes orbit, pan, zoom and fly input; `Locked` disables look; `Manual` consumes no device input and follows authored Orbit fields. */
export const OrbitMode = { Free: 0, Locked: 1, Manual: 2 } as const;

/**
 * Orbit, pan and zoom around a target. Hold the fly button for perspective flythrough
 * (mouse look, WASD/QE movement, Shift boost, scroll speed); in orthographic mode it orbits
 * instead, and scroll zooms size. Locked disables look; Manual leaves all input to the game.
 * On touch, one finger rotates, two-finger pinch zooms, two-finger drag pans.
 * In Free, C glides back to the pose the game last authored.
 */
export const Orbit = component(
    "Orbit",
    {
        /** horizontal orbit angle around the target, radians */
        yaw: f32,
        /** vertical orbit angle, radians; clamped to min/maxPitch */
        pitch: f32,
        /** camera distance from the target, world units (perspective zoom) */
        distance: f32,
        /** orthographic half-height, world units (ortho zoom) */
        size: f32,
        /** lower pitch clamp, radians */
        minPitch: f32,
        /** upper pitch clamp, radians */
        maxPitch: f32,
        /** closest perspective distance */
        minDistance: f32,
        /** farthest perspective distance */
        maxDistance: f32,
        /** smallest orthographic size */
        minSize: f32,
        /** largest orthographic size */
        maxSize: f32,
        /** follow damping, 0–1; higher snaps to the target pose faster */
        smoothness: f32,
        /** fly look damping, 0–1; higher is snappier */
        flySmoothness: f32,
        /** orbit look speed (yaw/pitch), radians per drag across the viewport's height, so a drag
         * turns the view by the same share of the screen at any canvas size. Without a positive
         * focused CSS height, use the camera's texture destination height; without either, drag
         * rotation is disabled. */
        rotateSpeed: f32,
        /** fly look speed (yaw/pitch), radians per pixel */
        flySensitivity: f32,
        /** held-arrow orbit speed, radians per second */
        keyRate: f32,
        /** held-arrow acceleration toward keyRate, radians per second squared */
        keyAcceleration: f32,
        /** released-arrow velocity damping, inverse seconds; higher stops sooner */
        keyDamping: f32,
        /** zoom factor applied to wheel, pinch and vertical mouse drag */
        zoomSpeed: f32,
        /** nonzero ignores wheel produced while any pointer button or touch contact is held */
        ignoreDragWheel: u8,
        /** Fraction of pending input pan applied per 60 Hz frame; authored pan remains immediate. */
        panSmoothness: f32,
        /** Input zoom response per 60 Hz frame: -1 inherits smoothness; [0,1] overrides it.
         * Authored distance/size changes still use smoothness. */
        zoomSmoothness: f32,
        /** Nonzero scales distance/size by inverse contact separation ratio instead of raw pinch pixels. */
        proportionalPinch: u8,
        /** Remaining contacts after any lift/cancel resume (1), or stop until the next press (0). */
        touchResume: u8,
        /** mouse button that orbits: 0 left, 1 middle, 2 right; 255 disables */
        orbitButton: u8,
        /** mouse button that pans: 0 left, 1 middle, 2 right; 255 disables */
        panButton: u8,
        /** mouse button for perspective flythrough (look and WASD/QE movement), or orthographic orbit: 0 left, 1 middle, 2 right; 255 disables */
        flyButton: u8,
        /** mouse button for vertical zoom drag: 0 left, 1 middle, 2 right; 255 disables */
        zoomButton: u8,
        /** nonzero swaps orbit/pan for Shift/Ctrl/Meta mouse presses until release; 0 disables; touch is unaffected */
        swapPanOrbit: u8,
        /** pan offset from the orbit target, world units */
        pan: vec3,
        /** WASD/QE fly speed, world units per second; scroll while flying adjusts it (clamped to flyMin/flyMax) */
        flySpeed: f32,
        /** fly speed multiplier while Shift is held; the boosted speed is never written to flySpeed */
        flyBoost: f32,
        /** lower clamp for scroll-adjusted flySpeed, world units per second */
        flyMin: f32,
        /** upper clamp for scroll-adjusted flySpeed, world units per second */
        flyMax: f32,
        /** nonzero lets C in Free glide back to the last authored yaw, pitch, distance, size and pan */
        recenter: u8,
        /** Free consumes device input; Locked leaves pan and zoom; Manual follows authored fields without device input */
        mode: u8,
        /** Entity to orbit, with pan relative to its rendered placement (0 = world origin).
         * Includes fixed-sample interpolation and VisualCorrection. The camera's Transform is
         * presentation-authored, so fixed readers of an orbit camera also read presentation. */
        target: entity,
    },
    {
        defaults: () => ({
            yaw: Math.PI / 6,
            pitch: Math.PI / 9,
            distance: 10,
            size: 5,
            // shy of ±90° so the look-at pose never degenerates at the pole
            minPitch: -89 * Deg2Rad,
            maxPitch: 89 * Deg2Rad,
            minDistance: 0.1,
            maxDistance: 900,
            minSize: 0.05,
            maxSize: 900,
            smoothness: 0.5,
            flySmoothness: 0.6,
            // TypeGPU Kit's 0.008 rad/px on its 478 px tall canvases
            rotateSpeed: 3.8,
            flySensitivity: 0.003,
            keyRate: 3,
            keyAcceleration: 30,
            keyDamping: 30,
            zoomSpeed: 0.025,
            ignoreDragWheel: 0,
            panSmoothness: 1,
            zoomSmoothness: -1,
            proportionalPinch: 0,
            touchResume: 1,
            orbitButton: 0,
            panButton: 1,
            flyButton: 2,
            zoomButton: 255,
            swapPanOrbit: 0,
            pan: [0, 0, 0],
            flySpeed: 5,
            flyBoost: 3,
            flyMin: 0.5,
            flyMax: 100,
            recenter: 1,
            mode: 0,
            target: 0,
        }),
    },
);

function smoothLerp(smoothness: number, dt: number): number {
    const s = Math.max(0, Math.min(1, smoothness));
    return 1 - (1 - s) ** (dt * 60);
}

function normalizeAngle(a: number): number {
    return ((a % Tau) + Tau) % Tau;
}

function angleDiff(from: number, to: number): number {
    const diff = normalizeAngle(to - from);
    return diff > Math.PI ? diff - Tau : diff;
}

function isButton(pointer: Readonly<Pointer>, button: number): boolean {
    if (button === 0) return pointer.left;
    if (button === 1) return pointer.middle;
    return button === 2 && pointer.right;
}

function heldButton(buttons: number, button: number): boolean {
    return button >= 0 && button <= 2 && (buttons & (1 << button)) !== 0;
}

function modifiedButton(modifiers: number, button: number): boolean {
    return button >= 0 && button <= 2 && ((modifiers >> (button * 4)) & 7) !== 0;
}

const orbitAdd = [Orbit, not(OrbitSmooth)];
const orbitRemove = [not(Orbit), OrbitSmooth];
const orbitPose = [Orbit, OrbitSmooth, Transform];
const orbitState = {
    create: () => ({
        rotation: new Float64Array(4),
        aim: new Float64Array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
    }),
};

function accelerate(
    velocity: number,
    direction: number,
    damping: number,
    rate: number,
    acceleration: number,
    dt: number,
): number {
    if (direction === 0) return velocity * Math.exp(-damping * dt);
    const target = direction * rate;
    const step = acceleration * dt;
    return velocity < target
        ? Math.min(target, velocity + step)
        : Math.max(target, velocity - step);
}

const OrbitSystem: System = {
    group: "simulation",

    update(world: World) {
        deriveTransforms(world);
        const input = world.resource(Devices);
        const orbit = world.storage(Orbit);
        const smooth = world.storage(OrbitSmooth);
        const transform = world.storage(Transform);
        const global = world.storage(GlobalTransform);
        const camera = world.storage(Camera);
        const dt = world.time.deltaTime;

        for (const eid of world.query(orbitAdd)) {
            world.add(eid, OrbitSmooth);
            smooth.yaw.set(eid, orbit.yaw.get(eid));
            smooth.pitch.set(eid, orbit.pitch.get(eid));
            smooth.distance.set(eid, orbit.distance.get(eid));
            smooth.size.set(eid, orbit.size.get(eid));
            smooth.keyYawVelocity.set(eid, 0);
            smooth.keyPitchVelocity.set(eid, 0);
            smooth.flyActive.set(eid, 0);
            smooth.pan.x.set(eid, orbit.pan.x.get(eid));
            smooth.pan.y.set(eid, orbit.pan.y.get(eid));
            smooth.pan.z.set(eid, orbit.pan.z.get(eid));
            smooth.panImpulse.set(eid, 0, 0, 0);
            smooth.panTarget.set(eid, orbit.target.get(eid));
            smooth.touchStopped.set(eid, 0);
            smooth.zoomDistanceImpulse.set(eid, 0);
            smooth.zoomSizeImpulse.set(eid, 0);
            smooth.zoomDistance.set(eid, orbit.distance.get(eid));
            smooth.zoomSize.set(eid, orbit.size.get(eid));
            smooth.publishedYaw.set(eid, orbit.yaw.get(eid));
            smooth.publishedPitch.set(eid, orbit.pitch.get(eid));
            smooth.homeYaw.set(eid, orbit.yaw.get(eid));
            smooth.homePitch.set(eid, orbit.pitch.get(eid));
            smooth.homeDistance.set(eid, orbit.distance.get(eid));
            smooth.homeSize.set(eid, orbit.size.get(eid));
            smooth.homePan.x.set(eid, orbit.pan.x.get(eid));
            smooth.homePan.y.set(eid, orbit.pan.y.get(eid));
            smooth.homePan.z.set(eid, orbit.pan.z.get(eid));
            // Warn on initialization because the pose query silently excludes entities without Transform.
            if (!world.has(eid, Transform)) {
                console.warn(
                    `[orbit] entity ${eid} has Orbit but no Transform — add Transform or it won't move`,
                );
            }
        }

        for (const eid of world.query(orbitRemove)) {
            world.remove(eid, OrbitSmooth);
        }

        for (const eid of world.query(orbitPose)) {
            const cssHeight = world.resource(Viewports).get(input.focused)?.cssHeight ?? 0;
            let height = cssHeight;
            if (height === 0 && world.has(eid, Camera)) {
                const view = cameraView(world, eid);
                if (view?.texture) height = view.viewport?.height ?? view.texture.height;
            }
            const sensitivity = height > 0 ? orbit.rotateSpeed.get(eid) / height : 0;
            const zoomSpeed = orbit.zoomSpeed.get(eid);
            const minPitch = orbit.minPitch.get(eid);
            const maxPitch = orbit.maxPitch.get(eid);
            const smoothness = orbit.smoothness.get(eid);

            let yawO = orbit.yaw.get(eid);
            let pitchO = orbit.pitch.get(eid);
            let distO = orbit.distance.get(eid);
            let sizeO = orbit.size.get(eid);
            let panX = orbit.pan.x.get(eid);
            let panY = orbit.pan.y.get(eid);
            let panZ = orbit.pan.z.get(eid);
            let flyActive = smooth.flyActive.get(eid);
            let flySpd = orbit.flySpeed.get(eid);
            let yawS = smooth.yaw.get(eid);
            let pitchS = smooth.pitch.get(eid);
            let distS = smooth.distance.get(eid);
            let sizeS = smooth.size.get(eid);
            let keyYawVelocity = smooth.keyYawVelocity.get(eid);
            let keyPitchVelocity = smooth.keyPitchVelocity.get(eid);
            // A value that differs from the one this system last published was authored, and becomes home.
            if (yawO !== smooth.publishedYaw.get(eid)) smooth.homeYaw.set(eid, yawO);
            if (pitchO !== smooth.publishedPitch.get(eid)) smooth.homePitch.set(eid, pitchO);
            if (distO !== smooth.zoomDistance.get(eid)) smooth.homeDistance.set(eid, distO);
            if (sizeO !== smooth.zoomSize.get(eid)) smooth.homeSize.set(eid, sizeO);
            if (
                panX !== smooth.pan.x.get(eid) ||
                panY !== smooth.pan.y.get(eid) ||
                panZ !== smooth.pan.z.get(eid)
            )
                smooth.homePan.set(eid, panX, panY, panZ);

            const hasCamera = world.has(eid, Camera);
            const isOrtho = hasCamera && camera.mode.get(eid) === CameraMode.Orthographic;
            const manual = orbit.mode.get(eid) === OrbitMode.Manual;
            const locked = orbit.mode.get(eid) !== OrbitMode.Free;
            const touchCount = manual ? 0 : input.touch.count;
            const authoredZoom =
                distO !== smooth.zoomDistance.get(eid) || sizeO !== smooth.zoomSize.get(eid);
            const zoomReset = manual || !input.enabled || authoredZoom;
            let distanceImpulse = zoomReset ? 0 : smooth.zoomDistanceImpulse.get(eid);
            let sizeImpulse = zoomReset ? 0 : smooth.zoomSizeImpulse.get(eid);
            const distanceBeforeInput = distO;
            const sizeBeforeInput = sizeO;
            // Touch capture also holds pointer.left; reading mouse buttons here would orbit during pinch/pan.
            // Fly has no touch equivalent.
            const orbitButton = orbit.orbitButton.get(eid);
            const panButton = orbit.panButton.get(eid);
            const mouseOrbitOrPan =
                isButton(input.pointer, orbitButton) || isButton(input.pointer, panButton);
            const zoomHeld =
                !manual && touchCount === 0 && isButton(input.pointer, orbit.zoomButton.get(eid));
            const orbitOrPanHeld = !manual && (touchCount > 0 || mouseOrbitOrPan);
            const flyHeld =
                !manual &&
                (touchCount > 0 ? false : isButton(input.pointer, orbit.flyButton.get(eid)));
            // Leave movement keys to gameplay unless the fly button is held; orbit and pan take priority.
            const flying = !isOrtho && !orbitOrPanHeld && !zoomHeld && flyHeld;
            let dragX = 0;
            let dragY = 0;
            let pinch = 0;
            let pinchDistance = distO;
            let pinchSize = sizeO;
            let touchStopped = smooth.touchStopped.get(eid);
            let zoomDrag = 0;
            if (!manual) {
                for (const motion of input.pointer.motion) {
                    const orbitSwap =
                        orbit.swapPanOrbit.get(eid) !== 0 &&
                        modifiedButton(motion.modifiers, orbitButton);
                    const panSwap =
                        orbit.swapPanOrbit.get(eid) !== 0 &&
                        modifiedButton(motion.modifiers, panButton);
                    const rotate =
                        (heldButton(motion.buttons, orbitButton) && !orbitSwap) ||
                        (heldButton(motion.buttons, panButton) && panSwap);
                    const pan =
                        (heldButton(motion.buttons, panButton) && !panSwap) ||
                        (heldButton(motion.buttons, orbitButton) && orbitSwap);
                    const zoom = heldButton(motion.buttons, orbit.zoomButton.get(eid));
                    const fly = heldButton(motion.buttons, orbit.flyButton.get(eid));
                    if (!locked && !zoom && (rotate || (!pan && fly))) {
                        const speed =
                            !isOrtho && !rotate && !pan && fly
                                ? orbit.flySensitivity.get(eid)
                                : sensitivity;
                        yawO -= motion.deltaX * speed;
                        pitchO = clamp(pitchO + motion.deltaY * speed, minPitch, maxPitch);
                    }
                    if (!zoom && pan) {
                        dragX += motion.deltaX;
                        dragY += motion.deltaY;
                    }
                    if (zoom) zoomDrag += motion.deltaY;
                }
                for (const motion of input.touch.events) {
                    if (motion.phase === "start") touchStopped = 0;
                    if (motion.phase === "end" || motion.phase === "cancel") touchStopped = 1;
                    if (
                        motion.phase !== "move" ||
                        (orbit.touchResume.get(eid) === 0 && touchStopped)
                    )
                        continue;
                    if (motion.count === 1 && !locked) {
                        yawO -= motion.deltaX * sensitivity;
                        pitchO = clamp(pitchO + motion.deltaY * sensitivity, minPitch, maxPitch);
                    } else if (motion.count === 2) {
                        dragX += motion.deltaX;
                        dragY += motion.deltaY;
                        if (orbit.proportionalPinch.get(eid) === 0) pinch += motion.pinchDelta;
                        else if (motion.ratio !== 1) {
                            if (isOrtho)
                                pinchSize = clamp(
                                    pinchSize / motion.ratio,
                                    orbit.minSize.get(eid),
                                    orbit.maxSize.get(eid),
                                );
                            else
                                pinchDistance = clamp(
                                    pinchDistance / motion.ratio,
                                    orbit.minDistance.get(eid),
                                    orbit.maxDistance.get(eid),
                                );
                        }
                    }
                }
            }

            const keyRate = orbit.keyRate.get(eid);
            const keyAcceleration = orbit.keyAcceleration.get(eid);
            const keyDamping = orbit.keyDamping.get(eid);
            const keyYaw =
                Number(input.keys.held.has("ArrowRight")) -
                Number(input.keys.held.has("ArrowLeft"));
            const keyPitch =
                Number(input.keys.held.has("ArrowUp")) - Number(input.keys.held.has("ArrowDown"));
            keyYawVelocity = accelerate(
                keyYawVelocity,
                locked ? 0 : keyYaw,
                keyDamping,
                keyRate,
                keyAcceleration,
                dt,
            );
            keyPitchVelocity = accelerate(
                keyPitchVelocity,
                locked ? 0 : keyPitch,
                keyDamping,
                keyRate,
                keyAcceleration,
                dt,
            );
            if (!locked) {
                yawO += keyYawVelocity * dt;
                pitchO = clamp(pitchO + keyPitchVelocity * dt, minPitch, maxPitch);
            }

            if (!manual && height > 0 && (dragX !== 0 || dragY !== 0)) {
                const cy = Math.cos(yawS);
                const sy = Math.sin(yawS);
                const cp = Math.cos(pitchS);
                const sp = Math.sin(pitchS);

                const rightX = cy;
                const rightZ = -sy;
                const upX = -sp * sy;
                const upY = cp;
                const upZ = -sp * cy;

                const worldPerPixel = isOrtho
                    ? (camera.size.get(eid) * 2) / height
                    : (2 *
                          distO *
                          Math.tan((hasCamera ? camera.fov.get(eid) : 60) * Deg2Rad * 0.5)) /
                      height;

                const dx = dragX * worldPerPixel;
                const dy = dragY * worldPerPixel;
                panX += dy * upX - dx * rightX;
                panY += dy * upY;
                panZ += dy * upZ - dx * rightZ;
            }

            const authoredX = orbit.pan.x.get(eid);
            const authoredY = orbit.pan.y.get(eid);
            const authoredZ = orbit.pan.z.get(eid);
            const resetPan =
                manual ||
                !input.enabled ||
                orbit.target.get(eid) !== smooth.panTarget.get(eid) ||
                authoredX !== smooth.pan.x.get(eid) ||
                authoredY !== smooth.pan.y.get(eid) ||
                authoredZ !== smooth.pan.z.get(eid);
            let impulseX = (resetPan ? 0 : smooth.panImpulse.x.get(eid)) + panX - authoredX;
            let impulseY = (resetPan ? 0 : smooth.panImpulse.y.get(eid)) + panY - authoredY;
            let impulseZ = (resetPan ? 0 : smooth.panImpulse.z.get(eid)) + panZ - authoredZ;
            const panResponse = orbit.panSmoothness.get(eid);
            const panT = panResponse >= 1 ? 1 : smoothLerp(panResponse, dt);
            const panDeltaX = impulseX * panT;
            const panDeltaY = impulseY * panT;
            const panDeltaZ = impulseZ * panT;
            panX = authoredX + panDeltaX;
            panY = authoredY + panDeltaY;
            panZ = authoredZ + panDeltaZ;
            impulseX -= panDeltaX;
            impulseY -= panDeltaY;
            impulseZ -= panDeltaZ;

            distO = pinchDistance;
            sizeO = pinchSize;

            // Wheel-away zooms out; a positive pinch spread zooms in. Both can arrive in one frame.
            let scroll = input.pointer.scroll;
            if (orbit.ignoreDragWheel.get(eid) !== 0) {
                scroll = 0;
                for (const fact of input.pointer.wheel) {
                    if (fact.buttons === 0 && fact.contacts === 0) scroll += fact.delta;
                }
            }
            const wheel = manual ? 0 : scroll - pinch;
            if (flying && wheel !== 0) {
                // Sustained fly uses final held levels; earlier zoom drags still change orbit distance.
                flySpd = clamp(
                    flySpd * Math.exp(-wheel * FlyScrollRate),
                    orbit.flyMin.get(eid),
                    orbit.flyMax.get(eid),
                );
            }
            const zoomInput = (flying ? 0 : wheel) + zoomDrag;
            if (zoomInput !== 0) {
                if (isOrtho) {
                    const sizeScale = Math.max(0.1, sizeO * 0.08);
                    sizeO = clamp(
                        sizeO + zoomInput * zoomSpeed * sizeScale,
                        orbit.minSize.get(eid),
                        orbit.maxSize.get(eid),
                    );
                } else {
                    const distanceScale = Math.max(0.3, distO * 0.08);
                    distO = clamp(
                        distO + zoomInput * zoomSpeed * distanceScale,
                        orbit.minDistance.get(eid),
                        orbit.maxDistance.get(eid),
                    );
                }
            }

            if (
                orbit.recenter.get(eid) !== 0 &&
                !locked &&
                !flying &&
                input.keys.pressed.has("KeyC")
            ) {
                yawO = smooth.homeYaw.get(eid);
                pitchO = smooth.homePitch.get(eid);
                distO = smooth.homeDistance.get(eid);
                sizeO = smooth.homeSize.get(eid);
                panX = smooth.homePan.x.get(eid);
                panY = smooth.homePan.y.get(eid);
                panZ = smooth.homePan.z.get(eid);
                impulseX = 0;
                impulseY = 0;
                impulseZ = 0;
                distanceImpulse = 0;
                sizeImpulse = 0;
                keyYawVelocity = 0;
                keyPitchVelocity = 0;
            }

            // Publish through typed columns: uninlined numeric setter calls box double arguments.
            const t = smoothLerp(smoothness, dt);
            const tLook = flying ? smoothLerp(orbit.flySmoothness.get(eid), dt) : t;
            yawS += angleDiff(yawS, yawO) * tLook;
            pitchS += (pitchO - pitchS) * tLook;
            const zoomResponse = orbit.zoomSmoothness.get(eid);
            if (zoomResponse < 0) {
                distanceImpulse = 0;
                sizeImpulse = 0;
                distS += (distO - distS) * t;
                if (isOrtho) sizeS += (sizeO - sizeS) * t;
            } else {
                const tZoom = smoothLerp(zoomResponse, dt);
                distanceImpulse += distO - distanceBeforeInput;
                sizeImpulse += sizeO - sizeBeforeInput;
                const distanceStep = distanceImpulse * tZoom;
                const sizeStep = sizeImpulse * tZoom;
                distS += (distO - distanceImpulse - distS) * t + distanceStep;
                distanceImpulse -= distanceStep;
                if (isOrtho) {
                    sizeS += (sizeO - sizeImpulse - sizeS) * t + sizeStep;
                    sizeImpulse -= sizeStep;
                }
            }

            if (isOrtho) {
                camera.size.column[eid] = sizeS;
                camera.size.markChanged(eid);
            }

            if (flying) {
                flyActive = 1;
                const boost =
                    input.keys.held.has("ShiftLeft") || input.keys.held.has("ShiftRight")
                        ? orbit.flyBoost.get(eid)
                        : 1;
                const speed = flySpd * boost * dt;
                const fp = -pitchS;
                const cy = Math.cos(yawS);
                const sy = Math.sin(yawS);
                const cp = Math.cos(fp);
                const sp = Math.sin(fp);

                let mx = 0;
                let my = 0;
                let mz = 0;
                if (input.keys.held.has("KeyW")) mz -= 1;
                if (input.keys.held.has("KeyS")) mz += 1;
                if (input.keys.held.has("KeyA")) mx -= 1;
                if (input.keys.held.has("KeyD")) mx += 1;
                if (input.keys.held.has("KeyQ")) my -= 1;
                if (input.keys.held.has("KeyE")) my += 1;

                // normalize the world move so a diagonal (e.g. forward + up) travels at `speed`, not faster
                let wx = mz * sy * cp + mx * cy;
                let wy = my - mz * sp;
                let wz = mz * cy * cp - mx * sy;
                const len = Math.hypot(wx, wy, wz);
                if (len > 0) {
                    const k = speed / len;
                    wx *= k;
                    wy *= k;
                    wz *= k;
                }

                transform.translation.column[eid * 4] =
                    global.translation.x.get(eid) + wx + panDeltaX;
                transform.translation.column[eid * 4 + 1] =
                    global.translation.y.get(eid) + wy + panDeltaY;
                transform.translation.column[eid * 4 + 2] =
                    global.translation.z.get(eid) + wz + panDeltaZ;
                transform.translation.markChanged(eid);

                const hy = yawS * 0.5;
                const hp = fp * 0.5;
                const shy = Math.sin(hy);
                const chy = Math.cos(hy);
                const shp = Math.sin(hp);
                const chp = Math.cos(hp);
                transform.rotation.column[eid * 4 + 0] = chy * shp;
                transform.rotation.column[eid * 4 + 1] = shy * chp;
                transform.rotation.column[eid * 4 + 2] = -shy * shp;
                transform.rotation.column[eid * 4 + 3] = chy * chp;
                transform.rotation.markChanged(eid);
            } else {
                if (flyActive) {
                    flyActive = 0;
                    let entityTargetX = 0;
                    let entityTargetY = 0;
                    let entityTargetZ = 0;
                    const targetEid = orbit.target.get(eid);
                    if (targetEid > 0 && world.has(targetEid, GlobalTransform)) {
                        entityTargetX = followPosition(world, eid, targetEid, 0);
                        entityTargetY = followPosition(world, eid, targetEid, 1);
                        entityTargetZ = followPosition(world, eid, targetEid, 2);
                    }
                    panX =
                        global.translation.x.get(eid) -
                        distS * Math.cos(pitchS) * Math.sin(yawS) -
                        entityTargetX +
                        panDeltaX;
                    panY =
                        global.translation.y.get(eid) -
                        distS * Math.sin(pitchS) -
                        entityTargetY +
                        panDeltaY;
                    panZ =
                        global.translation.z.get(eid) -
                        distS * Math.cos(pitchS) * Math.cos(yawS) -
                        entityTargetZ +
                        panDeltaZ;
                }

                let targetX = panX;
                let targetY = panY;
                let targetZ = panZ;
                const targetEid = orbit.target.get(eid);
                if (targetEid > 0 && world.has(targetEid, GlobalTransform)) {
                    targetX += followPosition(world, eid, targetEid, 0);
                    targetY += followPosition(world, eid, targetEid, 1);
                    targetZ += followPosition(world, eid, targetEid, 2);
                }

                const camX = targetX + distS * Math.cos(pitchS) * Math.sin(yawS);
                const camY = targetY + distS * Math.sin(pitchS);
                const camZ = targetZ + distS * Math.cos(pitchS) * Math.cos(yawS);

                transform.translation.column[eid * 4 + 0] = camX;
                transform.translation.column[eid * 4 + 1] = camY;
                transform.translation.column[eid * 4 + 2] = camZ;
                transform.translation.markChanged(eid);
                const pose = world.resource(orbitState);
                const r = pose.rotation;
                pose.aim[0] = camX;
                pose.aim[1] = camY;
                pose.aim[2] = camZ;
                pose.aim[3] = targetX;
                pose.aim[4] = targetY;
                pose.aim[5] = targetZ;
                lookAtRotationInto(pose.aim, r);
                transform.rotation.column[eid * 4 + 0] = r[0]!;
                transform.rotation.column[eid * 4 + 1] = r[1]!;
                transform.rotation.column[eid * 4 + 2] = r[2]!;
                transform.rotation.column[eid * 4 + 3] = r[3]!;
                transform.rotation.markChanged(eid);
            }

            orbit.yaw.column[eid] = yawO;
            orbit.yaw.markChanged(eid);
            orbit.pitch.column[eid] = pitchO;
            orbit.pitch.markChanged(eid);
            smooth.publishedYaw.set(eid, orbit.yaw.get(eid));
            smooth.publishedPitch.set(eid, orbit.pitch.get(eid));
            orbit.distance.column[eid] = distO;
            orbit.distance.markChanged(eid);
            orbit.size.column[eid] = sizeO;
            orbit.size.markChanged(eid);
            orbit.pan.column[eid * 4 + 0] = panX;
            orbit.pan.column[eid * 4 + 1] = panY;
            orbit.pan.column[eid * 4 + 2] = panZ;
            orbit.pan.markChanged(eid);
            smooth.pan.x.set(eid, orbit.pan.x.get(eid));
            smooth.pan.y.set(eid, orbit.pan.y.get(eid));
            smooth.pan.z.set(eid, orbit.pan.z.get(eid));
            smooth.panTarget.set(eid, orbit.target.get(eid));
            smooth.panImpulse.x.set(eid, impulseX);
            smooth.panImpulse.y.set(eid, impulseY);
            smooth.panImpulse.z.set(eid, impulseZ);
            smooth.touchStopped.set(eid, touchStopped);
            smooth.zoomDistanceImpulse.set(eid, distanceImpulse);
            smooth.zoomSizeImpulse.set(eid, sizeImpulse);
            smooth.zoomDistance.set(eid, orbit.distance.get(eid));
            smooth.zoomSize.set(eid, orbit.size.get(eid));
            orbit.flySpeed.column[eid] = flySpd;
            orbit.flySpeed.markChanged(eid);
            smooth.flyActive.column[eid] = flyActive;
            smooth.flyActive.markChanged(eid);
            smooth.yaw.column[eid] = yawS;
            smooth.yaw.markChanged(eid);
            smooth.pitch.column[eid] = pitchS;
            smooth.pitch.markChanged(eid);
            smooth.distance.column[eid] = distS;
            smooth.distance.markChanged(eid);
            smooth.size.column[eid] = sizeS;
            smooth.size.markChanged(eid);
            smooth.keyYawVelocity.column[eid] = keyYawVelocity;
            smooth.keyYawVelocity.markChanged(eid);
            smooth.keyPitchVelocity.column[eid] = keyPitchVelocity;
            smooth.keyPitchVelocity.markChanged(eid);
        }
    },
};

/** orbit / pan / zoom / fly camera controls via the {@link Orbit} component; one of the default plugins */
export const OrbitPlugin: Plugin = {
    name: "Orbit",
    systems: [OrbitSystem],
    components: [Orbit],

    dependencies: [InputPlugin, TransformPlugin, OrbitFollowPlugin],
};
