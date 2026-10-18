import { Devices, InputPlugin, type Pointer } from "../../core/input";
import { Camera, CameraMode } from "../../core/rendering";
import { GlobalTransform, Transform, TransformPlugin } from "../../core/transform";
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
import { OrbitSmooth } from "./smooth";

const Tau = Math.PI * 2;
const Deg2Rad = Math.PI / 180;

// Multiplicative fly-speed changes make wheel input feel equal across speed magnitudes.
const FlyScrollRate = Math.log(1.15) / 100; // ≈ 0.0014

/** `Free` orbits, pans, and zooms; `Locked` disables all look (orbit rotation and fly look), leaving pan and zoom. */
export const OrbitMode = { Free: 0, Locked: 1 } as const;

/**
 * Orbit, pan and zoom around a target. Hold the fly button for perspective flythrough
 * (mouse look, WASD/QE movement, Shift boost, scroll speed); in orthographic mode it orbits
 * instead, and scroll zooms size. Locked disables look.
 * On touch, one finger rotates, two-finger pinch zooms, two-finger drag pans.
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
        /** orbit look speed (yaw/pitch), radians per pixel of mouse drag */
        sensitivity: f32,
        /** fly look speed (yaw/pitch), radians per pixel */
        flySensitivity: f32,
        /** held-arrow orbit speed, radians per second */
        keyRate: f32,
        /** held-arrow acceleration toward keyRate, radians per second squared */
        keyAcceleration: f32,
        /** released-arrow velocity damping, inverse seconds; higher stops sooner */
        keyDamping: f32,
        /** zoom factor applied per scroll-wheel notch */
        zoomSpeed: f32,
        /** mouse button that orbits: 0 left, 1 middle, 2 right */
        orbitButton: u8,
        /** mouse button that pans: 0 left, 1 middle, 2 right */
        panButton: u8,
        /** mouse button for perspective flythrough (look and WASD/QE movement), or orthographic orbit: 0 left, 1 middle, 2 right */
        flyButton: u8,
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
        /** Free orbits, pans, and zooms; Locked disables all look (orbit rotation and fly look), leaving pan and zoom */
        mode: u8,
        /** entity to orbit; pan is relative to its position (0 = world origin) */
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
            smoothness: 0.3,
            flySmoothness: 0.6,
            sensitivity: 0.005,
            flySensitivity: 0.003,
            keyRate: 3,
            keyAcceleration: 30,
            keyDamping: 30,
            zoomSpeed: 0.025,
            orbitButton: 0,
            panButton: 1,
            flyButton: 2,
            pan: [0, 0, 0],
            flySpeed: 5,
            flyBoost: 3,
            flyMin: 0.5,
            flyMax: 100,
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
    return pointer.right;
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
            // sparse storage survives destroy — a recycled eid could inherit a stale latch
            smooth.flyActive.set(eid, 0);
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
            const sensitivity = orbit.sensitivity.get(eid);
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

            const hasCamera = world.has(eid, Camera);
            const isOrtho = hasCamera && camera.mode.get(eid) === CameraMode.Orthographic;
            const locked = orbit.mode.get(eid) === OrbitMode.Locked;
            const touchCount = input.touch.count;
            // Touch capture also holds pointer.left; reading mouse buttons here would orbit during pinch/pan.
            // Fly has no touch equivalent.
            const orbitHeld =
                touchCount > 0
                    ? touchCount === 1
                    : isButton(input.pointer, orbit.orbitButton.get(eid));
            const panHeld =
                touchCount > 0
                    ? touchCount >= 2
                    : isButton(input.pointer, orbit.panButton.get(eid));
            const flyHeld =
                touchCount > 0 ? false : isButton(input.pointer, orbit.flyButton.get(eid));
            // Leave movement keys to gameplay unless the fly button is held; orbit and pan take priority.
            const flying = !isOrtho && !orbitHeld && !panHeld && flyHeld;
            const looking = orbitHeld || (!panHeld && flyHeld);
            const lookSpeed = flying ? orbit.flySensitivity.get(eid) : sensitivity;

            if (!locked && looking) {
                yawO -= input.pointer.deltaX * lookSpeed;
                pitchO = clamp(pitchO + input.pointer.deltaY * lookSpeed, minPitch, maxPitch);
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

            if (!flying && panHeld) {
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
                    ? (camera.size.get(eid) * 2) /
                      (world.resource(Viewports).get(input.focused)?.cssHeight ?? 0)
                    : (2 *
                          distO *
                          Math.tan((hasCamera ? camera.fov.get(eid) : 60) * Deg2Rad * 0.5)) /
                      (world.resource(Viewports).get(input.focused)?.cssHeight ?? 0);

                const dragX = touchCount > 0 ? input.touch.deltaX : input.pointer.deltaX;
                const dragY = touchCount > 0 ? input.touch.deltaY : input.pointer.deltaY;
                const dx = dragX * worldPerPixel;
                const dy = dragY * worldPerPixel;
                panX += dy * upX - dx * rightX;
                panY += dy * upY;
                panZ += dy * upZ - dx * rightZ;
            }

            // Wheel-away zooms out; a positive pinch spread zooms in. Both can arrive in one frame.
            const pinch = touchCount > 0 ? input.touch.pinchDelta : 0;
            const zoomInput = input.pointer.scroll - pinch;
            if (zoomInput !== 0) {
                if (flying) {
                    // flying drives Transform directly, so the orbit distance is invisible — scroll
                    // retargets to fly speed, multiplicative so each notch feels equal across magnitudes.
                    flySpd = clamp(
                        flySpd * Math.exp(-zoomInput * FlyScrollRate),
                        orbit.flyMin.get(eid),
                        orbit.flyMax.get(eid),
                    );
                } else if (isOrtho) {
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

            // Publish through typed columns: uninlined numeric setter calls box double arguments.
            const t = smoothLerp(smoothness, dt);
            const tLook = flying ? smoothLerp(orbit.flySmoothness.get(eid), dt) : t;
            yawS += angleDiff(yawS, yawO) * tLook;
            pitchS += (pitchO - pitchS) * tLook;
            distS += (distO - distS) * t;

            if (isOrtho) {
                sizeS += (sizeO - sizeS) * t;
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

                transform.translation.column[eid * 4] = global.translation.x.get(eid) + wx;
                transform.translation.column[eid * 4 + 1] = global.translation.y.get(eid) + wy;
                transform.translation.column[eid * 4 + 2] = global.translation.z.get(eid) + wz;
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
                        entityTargetX = global.translation.x.get(targetEid);
                        entityTargetY = global.translation.y.get(targetEid);
                        entityTargetZ = global.translation.z.get(targetEid);
                    }
                    panX =
                        global.translation.x.get(eid) -
                        distS * Math.cos(pitchS) * Math.sin(yawS) -
                        entityTargetX;
                    panY = global.translation.y.get(eid) - distS * Math.sin(pitchS) - entityTargetY;
                    panZ =
                        global.translation.z.get(eid) -
                        distS * Math.cos(pitchS) * Math.cos(yawS) -
                        entityTargetZ;
                }

                let targetX = panX;
                let targetY = panY;
                let targetZ = panZ;
                const targetEid = orbit.target.get(eid);
                if (targetEid > 0 && world.has(targetEid, GlobalTransform)) {
                    targetX += global.translation.x.get(targetEid);
                    targetY += global.translation.y.get(targetEid);
                    targetZ += global.translation.z.get(targetEid);
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
            orbit.distance.column[eid] = distO;
            orbit.distance.markChanged(eid);
            orbit.size.column[eid] = sizeO;
            orbit.size.markChanged(eid);
            orbit.pan.column[eid * 4 + 0] = panX;
            orbit.pan.column[eid * 4 + 1] = panY;
            orbit.pan.column[eid * 4 + 2] = panZ;
            orbit.pan.markChanged(eid);
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

    dependencies: [InputPlugin, TransformPlugin],
};
