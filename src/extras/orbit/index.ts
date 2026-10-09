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
    vec4,
    type World,
} from "../../engine";
import { clamp, lookAtRotation } from "../../engine/utils";
import { OrbitSmooth } from "./smooth";

const Tau = Math.PI * 2;
const Deg2Rad = Math.PI / 180;

// scroll-to-flyspeed ramp: geometric, so each notch feels equal across magnitudes (5→10 like 50→100).
// one wheel notch (~100 accumulated deltaY) scales flySpeed ×1.15. scroll up (deltaY < 0) speeds up;
// the negation at the call site makes up = faster, like Unity's / Blender's scene-view accelerator.
const FlyScrollRate = Math.log(1.15) / 100; // ≈ 0.0014

/** `Free` orbits, pans, and zooms; `Locked` disables all look (orbit rotation and fly look), leaving pan and zoom. */
export const OrbitMode = { Free: 0, Locked: 1 } as const;

/**
 * contextual left-click hook (PlayCanvas-style): a picker registers `claim` so that pressing the orbit
 * button over something interactive starts an interaction instead of an orbit. Consulted once, at the orbit
 * button's press edge, with the cursor in canvas-local CSS pixels; returning true suppresses orbit rotation
 * for that whole drag (until the button releases), while pan and fly stay unaffected. Unregistered, every
 * press orbits (the optional slot is a `?.` no-op). Analogous to `world.gpu.span`.
 * @example
 * OrbitPick.claim = (x, y) => bodyUnderCursor(x, y) !== null;
 */
export const OrbitPick: { claim?: (x: number, y: number) => boolean } = {};

/**
 * orbit camera controls: drag to rotate around a target, scroll to zoom;
 * on touch, one finger rotates, two-finger pinch zooms, two-finger drag pans
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
        /** fly look damping, 0–1; higher is snappier; default tighter than orbit so first-person look tracks closely */
        flySmoothness: f32,
        /** orbit look speed (yaw/pitch), radians per pixel of mouse drag */
        sensitivity: f32,
        /** fly look speed (yaw/pitch), radians per pixel; separate so fly look reads calmer than orbit */
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
        /** mouse button that flies (hold to look around, WASD/QE to move): 0 left, 1 middle, 2 right */
        flyButton: u8,
        /** pan offset from the orbit target, world units */
        pan: vec4,
        /** WASD/QE fly speed, world units per second; scroll while flying adjusts it (clamped to flyMin/flyMax) */
        flySpeed: f32,
        /** shift-held fly boost multiplier, transient; scales flySpeed while shift is down, never stored */
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
            // permissive by default so a scene at any reasonable scale isn't clamped: the bounds span
            // the default camera frustum (near 0.1 → far 1000), and the geometric zoom step makes a wide
            // range cost nothing. Tighten per-camera for a game that wants to constrain zoom.
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
            pan: [0, 0, 0, 0],
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

const OrbitSystem: System = {
    group: "simulation",

    update(world: World) {
        const input = world.resource(Devices);
        const dt = world.time.deltaTime;

        for (const eid of world.query([Orbit, not(OrbitSmooth)])) {
            world.add(eid, OrbitSmooth);
            world.storage(OrbitSmooth).yaw.set(eid, world.storage(Orbit).yaw.get(eid));
            world.storage(OrbitSmooth).pitch.set(eid, world.storage(Orbit).pitch.get(eid));
            world.storage(OrbitSmooth).distance.set(eid, world.storage(Orbit).distance.get(eid));
            world.storage(OrbitSmooth).size.set(eid, world.storage(Orbit).size.get(eid));
            world.storage(OrbitSmooth).keyYawVelocity.set(eid, 0);
            world.storage(OrbitSmooth).keyPitchVelocity.set(eid, 0);
            // sparse storage survives destroy — a recycled eid could inherit a stale latch
            world.storage(OrbitSmooth).flyActive.set(eid, 0);
            world.storage(OrbitSmooth).orbitLatch.set(eid, 0);
            // the pose loop below requires Transform — orbit drives pos + rot through it. Without one the
            // camera silently never moves; warn at init (once per Orbit entity) so it's not a blank screen.
            if (!world.has(eid, Transform)) {
                console.warn(
                    `[orbit] entity ${eid} has Orbit but no Transform — add Transform or it won't move`,
                );
            }
        }

        for (const eid of world.query([not(Orbit), OrbitSmooth])) {
            world.remove(eid, OrbitSmooth);
        }

        for (const eid of world.query([Orbit, OrbitSmooth, Transform])) {
            const sensitivity = world.storage(Orbit).sensitivity.get(eid);
            const zoomSpeed = world.storage(Orbit).zoomSpeed.get(eid);
            const minPitch = world.storage(Orbit).minPitch.get(eid);
            const maxPitch = world.storage(Orbit).maxPitch.get(eid);
            const smoothness = world.storage(Orbit).smoothness.get(eid);

            let yawO = world.storage(Orbit).yaw.get(eid);
            let pitchO = world.storage(Orbit).pitch.get(eid);
            let distO = world.storage(Orbit).distance.get(eid);
            let sizeO = world.storage(Orbit).size.get(eid);
            let panX = world.storage(Orbit).pan.x.get(eid);
            let panY = world.storage(Orbit).pan.y.get(eid);
            let panZ = world.storage(Orbit).pan.z.get(eid);
            let flyActive = world.storage(OrbitSmooth).flyActive.get(eid);
            let flySpd = world.storage(Orbit).flySpeed.get(eid);
            let yawS = world.storage(OrbitSmooth).yaw.get(eid);
            let pitchS = world.storage(OrbitSmooth).pitch.get(eid);
            let distS = world.storage(OrbitSmooth).distance.get(eid);
            let sizeS = world.storage(OrbitSmooth).size.get(eid);
            let keyYawVelocity = world.storage(OrbitSmooth).keyYawVelocity.get(eid);
            let keyPitchVelocity = world.storage(OrbitSmooth).keyPitchVelocity.get(eid);

            const hasCamera = world.has(eid, Camera);
            const isOrtho =
                hasCamera && world.storage(Camera).mode.get(eid) === CameraMode.Orthographic;
            const locked = world.storage(Orbit).mode.get(eid) === OrbitMode.Locked;
            const touchCount = input.touch.count;
            // touch overrides the mouse-button read entirely rather than adding to it, while any finger
            // is down: the first finger's continued capture keeps `pointer.left` synthesized true for the
            // whole gesture (the old incidental path), so reading it during a two-finger pinch/pan would
            // orbit alongside the intended gesture. Gesture count alone decides the mode instead — one
            // finger rotates, two-plus pan, matching three.js OrbitControls' / Babylon's touch map. Fly
            // has no touch equivalent (out of scope), so any touch keeps it held-off.
            const orbitHeld =
                touchCount > 0
                    ? touchCount === 1
                    : isButton(input.pointer, world.storage(Orbit).orbitButton.get(eid));
            const panHeld =
                touchCount > 0
                    ? touchCount >= 2
                    : isButton(input.pointer, world.storage(Orbit).panButton.get(eid));
            const flyHeld =
                touchCount > 0
                    ? false
                    : isButton(input.pointer, world.storage(Orbit).flyButton.get(eid));
            // the held button picks the mode; fly engages only while the fly button is held (hold-to-fly, the
            // Unity/UE scene-view idiom), and orbit/pan win over it. bare WASD/QE never fly, so a gameplay
            // scene owns the movement keys by default — the camera only takes them while fly is held.
            const flying = !orbitHeld && !panHeld && flyHeld;
            // look applies the active mode's drag: fly looks in place (fly button), orbit swings the target.
            const looking = flying ? flyHeld : orbitHeld;
            const lookSpeed = flying ? world.storage(Orbit).flySensitivity.get(eid) : sensitivity;

            // consult the picker once, at the orbit button's down-edge (latch idle → a fresh press). a true
            // claim suppresses this drag's orbit rotation so an interaction owns the press; the latch holds
            // until release, so a mid-drag claim change can't flip it. only orbit look is gated — fly look
            // reads flyHeld (which needs the orbit button up, so suppression can't coincide) and pan/zoom
            // read their own buttons.
            let orbitLatch = world.storage(OrbitSmooth).orbitLatch.get(eid);
            if (orbitHeld) {
                if (orbitLatch === 0)
                    orbitLatch = OrbitPick.claim?.(input.pointer.x, input.pointer.y) ? 1 : 2;
            } else {
                orbitLatch = 0;
            }
            const suppressed = orbitLatch === 1;

            if (!locked && looking && !suppressed) {
                yawO -= input.pointer.deltaX * lookSpeed;
                pitchO = clamp(pitchO + input.pointer.deltaY * lookSpeed, minPitch, maxPitch);
            }

            const keyRate = world.storage(Orbit).keyRate.get(eid);
            const keyAcceleration = world.storage(Orbit).keyAcceleration.get(eid);
            const keyDamping = world.storage(Orbit).keyDamping.get(eid);
            const keyYaw =
                Number(input.keys.held.has("ArrowRight")) -
                Number(input.keys.held.has("ArrowLeft"));
            const keyPitch =
                Number(input.keys.held.has("ArrowUp")) - Number(input.keys.held.has("ArrowDown"));
            const accelerate = (velocity: number, direction: number): number => {
                if (direction === 0) return velocity * Math.exp(-keyDamping * dt);
                const target = direction * keyRate;
                const step = keyAcceleration * dt;
                return velocity < target
                    ? Math.min(target, velocity + step)
                    : Math.max(target, velocity - step);
            };
            keyYawVelocity = accelerate(keyYawVelocity, locked ? 0 : keyYaw);
            keyPitchVelocity = accelerate(keyPitchVelocity, locked ? 0 : keyPitch);
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
                    ? (world.storage(Camera).size.get(eid) * 2) /
                      (world.resource(Viewports).get(input.focused)?.cssHeight ?? 0)
                    : (2 *
                          distO *
                          Math.tan(
                              (hasCamera ? world.storage(Camera).fov.get(eid) : 60) * Deg2Rad * 0.5,
                          )) /
                      (world.resource(Viewports).get(input.focused)?.cssHeight ?? 0);

                // two-finger centroid drag while touching; single-pointer capture delta otherwise —
                // `Touch.deltaX/deltaY` is only ever populated at two fingers (input/index.ts), so this
                // never reads a stale value.
                const dragX = touchCount > 0 ? input.touch.deltaX : input.pointer.deltaX;
                const dragY = touchCount > 0 ? input.touch.deltaY : input.pointer.deltaY;
                const dx = dragX * worldPerPixel;
                const dy = dragY * worldPerPixel;
                panX += dy * upX - dx * rightX;
                panY += dy * upY;
                panZ += dy * upZ - dx * rightZ;
            }

            // pinch shares the wheel's geometric zoom step (same distanceScale/sizeScale/zoomSpeed), just
            // negated: `pointer.scroll`'s own JSDoc states positive = zoom out/away, while `touch.pinchDelta`
            // spreading positive means zoom in — so a spread pinch subtracts. Summing them is safe because
            // a two-finger pinch zeroes `flyHeld` (touch overrides the mouse-button read while any finger is
            // down), not because the two inputs can't coexist in the same frame — so combining them into one
            // input is just addition, not a priority choice.
            const pinch = touchCount > 0 ? input.touch.pinchDelta : 0;
            const zoomInput = input.pointer.scroll - pinch;
            if (zoomInput !== 0) {
                if (flying) {
                    // flying drives Transform directly, so the orbit distance is invisible — scroll
                    // retargets to fly speed, multiplicative like Unity's scene-view accelerator.
                    flySpd = clamp(
                        flySpd * Math.exp(-zoomInput * FlyScrollRate),
                        world.storage(Orbit).flyMin.get(eid),
                        world.storage(Orbit).flyMax.get(eid),
                    );
                } else if (isOrtho) {
                    const sizeScale = Math.max(0.1, sizeO * 0.08);
                    sizeO = clamp(
                        sizeO + zoomInput * zoomSpeed * sizeScale,
                        world.storage(Orbit).minSize.get(eid),
                        world.storage(Orbit).maxSize.get(eid),
                    );
                } else {
                    const distanceScale = Math.max(0.3, distO * 0.08);
                    distO = clamp(
                        distO + zoomInput * zoomSpeed * distanceScale,
                        world.storage(Orbit).minDistance.get(eid),
                        world.storage(Orbit).maxDistance.get(eid),
                    );
                }
            }

            const t = smoothLerp(smoothness, dt);
            // fly look uses its own, tighter damping (flySmoothness) so first-person look tracks the mouse
            // closely without orbit's floaty glide. the exit reproject keeps pose continuous either way.
            const tLook = flying ? smoothLerp(world.storage(Orbit).flySmoothness.get(eid), dt) : t;
            yawS += angleDiff(yawS, yawO) * tLook;
            pitchS += (pitchO - pitchS) * tLook;
            distS += (distO - distS) * t;

            if (isOrtho) {
                sizeS += (sizeO - sizeS) * t;
                world.storage(Camera).size.set(eid, sizeS);
            }

            if (flying) {
                flyActive = 1;
                // shift boosts speed transiently — the stored base (flySpd) is unchanged
                const boost =
                    input.keys.held.has("ShiftLeft") || input.keys.held.has("ShiftRight")
                        ? world.storage(Orbit).flyBoost.get(eid)
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

                world
                    .storage(Transform)
                    .translation.set(
                        eid,
                        world.storage(GlobalTransform).translation.x.get(eid) + wx,
                        world.storage(GlobalTransform).translation.y.get(eid) + wy,
                        world.storage(GlobalTransform).translation.z.get(eid) + wz,
                        0,
                    );

                const hy = yawS * 0.5;
                const hp = fp * 0.5;
                const shy = Math.sin(hy);
                const chy = Math.cos(hy);
                const shp = Math.sin(hp);
                const chp = Math.cos(hp);
                world
                    .storage(Transform)
                    .rotation.set(eid, chy * shp, shy * chp, -shy * shp, chy * chp);
            } else {
                if (flyActive) {
                    flyActive = 0;
                    let entityTargetX = 0;
                    let entityTargetY = 0;
                    let entityTargetZ = 0;
                    const targetEid = world.storage(Orbit).target.get(eid);
                    if (targetEid > 0 && world.has(targetEid, GlobalTransform)) {
                        entityTargetX = world.storage(GlobalTransform).translation.x.get(targetEid);
                        entityTargetY = world.storage(GlobalTransform).translation.y.get(targetEid);
                        entityTargetZ = world.storage(GlobalTransform).translation.z.get(targetEid);
                    }
                    panX =
                        world.storage(GlobalTransform).translation.x.get(eid) -
                        distS * Math.cos(pitchS) * Math.sin(yawS) -
                        entityTargetX;
                    panY =
                        world.storage(GlobalTransform).translation.y.get(eid) -
                        distS * Math.sin(pitchS) -
                        entityTargetY;
                    panZ =
                        world.storage(GlobalTransform).translation.z.get(eid) -
                        distS * Math.cos(pitchS) * Math.cos(yawS) -
                        entityTargetZ;
                }

                let targetX = panX;
                let targetY = panY;
                let targetZ = panZ;
                const targetEid = world.storage(Orbit).target.get(eid);
                if (targetEid > 0 && world.has(targetEid, GlobalTransform)) {
                    targetX += world.storage(GlobalTransform).translation.x.get(targetEid);
                    targetY += world.storage(GlobalTransform).translation.y.get(targetEid);
                    targetZ += world.storage(GlobalTransform).translation.z.get(targetEid);
                }

                const camX = targetX + distS * Math.cos(pitchS) * Math.sin(yawS);
                const camY = targetY + distS * Math.sin(pitchS);
                const camZ = targetZ + distS * Math.cos(pitchS) * Math.cos(yawS);

                world.storage(Transform).translation.set(eid, camX, camY, camZ, 0);
                const r = lookAtRotation(camX, camY, camZ, targetX, targetY, targetZ);
                world.storage(Transform).rotation.set(eid, r.x, r.y, r.z, r.w);
            }

            world.storage(Orbit).yaw.set(eid, yawO);
            world.storage(Orbit).pitch.set(eid, pitchO);
            world.storage(Orbit).distance.set(eid, distO);
            world.storage(Orbit).size.set(eid, sizeO);
            world.storage(Orbit).pan.set(eid, panX, panY, panZ, 0);
            world.storage(Orbit).flySpeed.set(eid, flySpd);
            world.storage(OrbitSmooth).flyActive.set(eid, flyActive);
            world.storage(OrbitSmooth).orbitLatch.set(eid, orbitLatch);
            world.storage(OrbitSmooth).yaw.set(eid, yawS);
            world.storage(OrbitSmooth).pitch.set(eid, pitchS);
            world.storage(OrbitSmooth).distance.set(eid, distS);
            world.storage(OrbitSmooth).size.set(eid, sizeS);
            world.storage(OrbitSmooth).keyYawVelocity.set(eid, keyYawVelocity);
            world.storage(OrbitSmooth).keyPitchVelocity.set(eid, keyPitchVelocity);
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
