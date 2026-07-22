import {
    entity,
    f32,
    not,
    type Plugin,
    type State,
    type System,
    sparse,
    u8,
    vec4,
} from "../../engine";
import { aim, angle, clamp } from "../../engine/utils";
import { InputPlugin, Inputs, type Mouse } from "../../standard/input";
import { Camera, CameraMode } from "../../standard/render";
import { Transform, TransformsPlugin } from "../../standard/transforms";
import { OrbitSmooth } from "./smooth";

const Tau = Math.PI * 2;
const Deg2Rad = Math.PI / 180;

// scroll-to-flyspeed ramp: geometric, so each notch feels equal across magnitudes (5→10 like 50→100).
// one wheel notch (~100 accumulated deltaY) scales flySpeed ×1.15. scroll up (deltaY < 0) speeds up;
// the negation at the call site makes up = faster, like Unity's / Blender's scene-view accelerator.
const FlyScrollRate = Math.log(1.15) / 100; // ≈ 0.0014

/** `Free` orbits, pans, and zooms; `Locked` disables orbit rotation, leaving pan and zoom. */
export const OrbitMode = { Free: 0, Locked: 1 } as const;

/**
 * contextual left-click hook (PlayCanvas-style): a picker registers `claim` so that pressing the orbit
 * button over something interactive starts an interaction instead of an orbit. Consulted once, at the orbit
 * button's press edge, with the cursor in canvas-local CSS pixels; returning true suppresses orbit rotation
 * for that whole drag (until the button releases), while pan and fly stay unaffected. Unregistered, every
 * press orbits (the optional slot is a `?.` no-op). Analogous to `Compute.span`.
 * @example
 * OrbitPick.claim = (x, y) => bodyUnderCursor(x, y) !== null;
 */
export const OrbitPick: { claim?: (x: number, y: number) => boolean } = {};

/**
 * orbit camera controls: drag to rotate around a target, scroll to zoom
 */
export const Orbit = {
    /** horizontal orbit angle around the target, radians */
    yaw: sparse(f32),
    /** vertical orbit angle, radians; clamped to min/maxPitch */
    pitch: sparse(f32),
    /** camera distance from the target, world units (perspective zoom) */
    distance: sparse(f32),
    /** orthographic half-height, world units (ortho zoom) */
    size: sparse(f32),
    /** lower pitch clamp, radians */
    minPitch: sparse(f32),
    /** upper pitch clamp, radians */
    maxPitch: sparse(f32),
    /** closest perspective distance */
    minDistance: sparse(f32),
    /** farthest perspective distance */
    maxDistance: sparse(f32),
    /** smallest orthographic size */
    minSize: sparse(f32),
    /** largest orthographic size */
    maxSize: sparse(f32),
    /** follow damping, 0–1; higher snaps to the target pose faster */
    smoothness: sparse(f32),
    /** fly look damping, 0–1; higher is snappier; default tighter than orbit so first-person look tracks closely */
    flySmoothness: sparse(f32),
    /** orbit look speed (yaw/pitch), radians per pixel of mouse drag */
    sensitivity: sparse(f32),
    /** fly look speed (yaw/pitch), radians per pixel; separate so fly look reads calmer than orbit */
    flySensitivity: sparse(f32),
    /** zoom factor applied per scroll-wheel notch */
    zoomSpeed: sparse(f32),
    /** mouse button that orbits: 0 left, 1 middle, 2 right */
    orbitButton: sparse(u8),
    /** mouse button that pans: 0 left, 1 middle, 2 right */
    panButton: sparse(u8),
    /** mouse button that flies (hold to look around, WASD/QE to move): 0 left, 1 middle, 2 right */
    flyButton: sparse(u8),
    /** pan offset from the orbit target, world units */
    pan: sparse(vec4),
    /** WASD/QE fly speed, world units per second; scroll while flying adjusts it (clamped to flyMin/flyMax) */
    flySpeed: sparse(f32),
    /** shift-held fly boost multiplier, transient; scales flySpeed while shift is down, never stored */
    flyBoost: sparse(f32),
    /** lower clamp for scroll-adjusted flySpeed, world units per second */
    flyMin: sparse(f32),
    /** upper clamp for scroll-adjusted flySpeed, world units per second */
    flyMax: sparse(f32),
    /** Free orbits, pans, and zooms; Locked disables orbit rotation, leaving pan and zoom */
    mode: sparse(u8),
    /** entity to orbit; pan is relative to its position (0 = world origin) */
    target: sparse(entity),
};

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

function isButton(mouse: Readonly<Mouse>, button: number): boolean {
    if (button === 0) return mouse.left;
    if (button === 1) return mouse.middle;
    return mouse.right;
}

const OrbitSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },

    update(state: State) {
        const input = Inputs;
        const dt = state.time.deltaTime;

        for (const eid of state.query([Orbit, not(OrbitSmooth)])) {
            state.add(eid, OrbitSmooth);
            OrbitSmooth.yaw.set(eid, Orbit.yaw.get(eid));
            OrbitSmooth.pitch.set(eid, Orbit.pitch.get(eid));
            OrbitSmooth.distance.set(eid, Orbit.distance.get(eid));
            OrbitSmooth.size.set(eid, Orbit.size.get(eid));
            // sparse storage survives destroy — a recycled eid could inherit a stale latch
            OrbitSmooth.flyActive.set(eid, 0);
            OrbitSmooth.orbitLatch.set(eid, 0);
            // the pose loop below requires Transform — orbit drives pos + rot through it. Without one the
            // camera silently never moves; warn at init (once per Orbit entity) so it's not a blank screen.
            if (!state.has(eid, Transform)) {
                console.warn(
                    `[orbit] entity ${eid} has Orbit but no Transform — add Transform or it won't move`,
                );
            }
        }

        for (const eid of state.query([not(Orbit), OrbitSmooth])) {
            state.remove(eid, OrbitSmooth);
        }

        for (const eid of state.query([Orbit, OrbitSmooth, Transform])) {
            const sensitivity = Orbit.sensitivity.get(eid);
            const zoomSpeed = Orbit.zoomSpeed.get(eid);
            const minPitch = Orbit.minPitch.get(eid);
            const maxPitch = Orbit.maxPitch.get(eid);
            const smoothness = Orbit.smoothness.get(eid);

            let yawO = Orbit.yaw.get(eid);
            let pitchO = Orbit.pitch.get(eid);
            let distO = Orbit.distance.get(eid);
            let sizeO = Orbit.size.get(eid);
            let panX = Orbit.pan.x.get(eid);
            let panY = Orbit.pan.y.get(eid);
            let panZ = Orbit.pan.z.get(eid);
            let flyActive = OrbitSmooth.flyActive.get(eid);
            let flySpd = Orbit.flySpeed.get(eid);
            let yawS = OrbitSmooth.yaw.get(eid);
            let pitchS = OrbitSmooth.pitch.get(eid);
            let distS = OrbitSmooth.distance.get(eid);
            let sizeS = OrbitSmooth.size.get(eid);

            const hasCamera = state.has(eid, Camera);
            const isOrtho = hasCamera && Camera.mode.get(eid) === CameraMode.Orthographic;
            const locked = Orbit.mode.get(eid) === OrbitMode.Locked;
            const orbitHeld = isButton(input.mouse, Orbit.orbitButton.get(eid));
            const panHeld = isButton(input.mouse, Orbit.panButton.get(eid));
            const flyHeld = isButton(input.mouse, Orbit.flyButton.get(eid));
            // the held button picks the mode; fly engages only while the fly button is held (hold-to-fly, the
            // Unity/UE scene-view idiom), and orbit/pan win over it. bare WASD/QE never fly, so a gameplay
            // scene owns the movement keys by default — the camera only takes them while fly is held.
            const flying = !orbitHeld && !panHeld && flyHeld;
            // look applies the active mode's drag: fly looks in place (fly button), orbit swings the target.
            const looking = flying ? flyHeld : orbitHeld;
            const lookSpeed = flying ? Orbit.flySensitivity.get(eid) : sensitivity;

            // consult the picker once, at the orbit button's down-edge (latch idle → a fresh press). a true
            // claim suppresses this drag's orbit rotation so an interaction owns the press; the latch holds
            // until release, so a mid-drag claim change can't flip it. only orbit look is gated — fly look
            // reads flyHeld (which needs the orbit button up, so suppression can't coincide) and pan/zoom
            // read their own buttons.
            let orbitLatch = OrbitSmooth.orbitLatch.get(eid);
            if (orbitHeld) {
                if (orbitLatch === 0)
                    orbitLatch = OrbitPick.claim?.(input.mouse.x, input.mouse.y) ? 1 : 2;
            } else {
                orbitLatch = 0;
            }
            const suppressed = orbitLatch === 1;

            if (!locked && looking && !suppressed) {
                yawO -= input.mouse.deltaX * lookSpeed;
                pitchO = clamp(pitchO + input.mouse.deltaY * lookSpeed, minPitch, maxPitch);
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
                    ? (Camera.size.get(eid) * 2) / input.mouse.canvasHeight
                    : (2 *
                          distO *
                          Math.tan((hasCamera ? Camera.fov.get(eid) : 60) * Deg2Rad * 0.5)) /
                      input.mouse.canvasHeight;

                const dx = input.mouse.deltaX * worldPerPixel;
                const dy = input.mouse.deltaY * worldPerPixel;
                panX += dy * upX - dx * rightX;
                panY += dy * upY;
                panZ += dy * upZ - dx * rightZ;
            }

            if (input.mouse.scroll !== 0) {
                if (flying) {
                    // flying drives Transform directly, so the orbit distance is invisible — scroll
                    // retargets to fly speed, multiplicative like Unity's scene-view accelerator.
                    flySpd = clamp(
                        flySpd * Math.exp(-input.mouse.scroll * FlyScrollRate),
                        Orbit.flyMin.get(eid),
                        Orbit.flyMax.get(eid),
                    );
                } else if (isOrtho) {
                    const sizeScale = Math.max(0.1, sizeO * 0.08);
                    sizeO = clamp(
                        sizeO + input.mouse.scroll * zoomSpeed * sizeScale,
                        Orbit.minSize.get(eid),
                        Orbit.maxSize.get(eid),
                    );
                } else {
                    const distanceScale = Math.max(0.3, distO * 0.08);
                    distO = clamp(
                        distO + input.mouse.scroll * zoomSpeed * distanceScale,
                        Orbit.minDistance.get(eid),
                        Orbit.maxDistance.get(eid),
                    );
                }
            }

            const t = smoothLerp(smoothness, dt);
            // fly look uses its own, tighter damping (flySmoothness) so first-person look tracks the mouse
            // closely without orbit's floaty glide. the exit reproject keeps pose continuous either way.
            const tLook = flying ? smoothLerp(Orbit.flySmoothness.get(eid), dt) : t;
            yawS += angleDiff(yawS, yawO) * tLook;
            pitchS += (pitchO - pitchS) * tLook;
            distS += (distO - distS) * t;

            if (isOrtho) {
                sizeS += (sizeO - sizeS) * t;
                Camera.size.set(eid, sizeS);
            }

            if (flying) {
                flyActive = 1;
                // shift boosts speed transiently — the stored base (flySpd) is unchanged
                const boost =
                    input.isKeyDown("ShiftLeft") || input.isKeyDown("ShiftRight")
                        ? Orbit.flyBoost.get(eid)
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
                if (input.isKeyDown("KeyW")) mz -= 1;
                if (input.isKeyDown("KeyS")) mz += 1;
                if (input.isKeyDown("KeyA")) mx -= 1;
                if (input.isKeyDown("KeyD")) mx += 1;
                if (input.isKeyDown("KeyQ")) my -= 1;
                if (input.isKeyDown("KeyE")) my += 1;

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

                Transform.pos.set(
                    eid,
                    Transform.pos.x.get(eid) + wx,
                    Transform.pos.y.get(eid) + wy,
                    Transform.pos.z.get(eid) + wz,
                    0,
                );

                const hy = yawS * 0.5;
                const hp = fp * 0.5;
                const shy = Math.sin(hy);
                const chy = Math.cos(hy);
                const shp = Math.sin(hp);
                const chp = Math.cos(hp);
                Transform.rot.set(eid, chy * shp, shy * chp, -shy * shp, chy * chp);
            } else {
                if (flyActive) {
                    flyActive = 0;
                    let entityTargetX = 0;
                    let entityTargetY = 0;
                    let entityTargetZ = 0;
                    const targetEid = Orbit.target.get(eid);
                    if (targetEid > 0 && state.has(targetEid, Transform)) {
                        entityTargetX = Transform.pos.x.get(targetEid);
                        entityTargetY = Transform.pos.y.get(targetEid);
                        entityTargetZ = Transform.pos.z.get(targetEid);
                    }
                    panX =
                        Transform.pos.x.get(eid) -
                        distS * Math.cos(pitchS) * Math.sin(yawS) -
                        entityTargetX;
                    panY = Transform.pos.y.get(eid) - distS * Math.sin(pitchS) - entityTargetY;
                    panZ =
                        Transform.pos.z.get(eid) -
                        distS * Math.cos(pitchS) * Math.cos(yawS) -
                        entityTargetZ;
                }

                let targetX = panX;
                let targetY = panY;
                let targetZ = panZ;
                const targetEid = Orbit.target.get(eid);
                if (targetEid > 0 && state.has(targetEid, Transform)) {
                    targetX += Transform.pos.x.get(targetEid);
                    targetY += Transform.pos.y.get(targetEid);
                    targetZ += Transform.pos.z.get(targetEid);
                }

                const camX = targetX + distS * Math.cos(pitchS) * Math.sin(yawS);
                const camY = targetY + distS * Math.sin(pitchS);
                const camZ = targetZ + distS * Math.cos(pitchS) * Math.cos(yawS);

                Transform.pos.set(eid, camX, camY, camZ, 0);
                const r = aim(camX, camY, camZ, targetX, targetY, targetZ);
                Transform.rot.set(eid, r.x, r.y, r.z, r.w);
            }

            Orbit.yaw.set(eid, yawO);
            Orbit.pitch.set(eid, pitchO);
            Orbit.distance.set(eid, distO);
            Orbit.size.set(eid, sizeO);
            Orbit.pan.set(eid, panX, panY, panZ, 0);
            Orbit.flySpeed.set(eid, flySpd);
            OrbitSmooth.flyActive.set(eid, flyActive);
            OrbitSmooth.orbitLatch.set(eid, orbitLatch);
            OrbitSmooth.yaw.set(eid, yawS);
            OrbitSmooth.pitch.set(eid, pitchS);
            OrbitSmooth.distance.set(eid, distS);
            OrbitSmooth.size.set(eid, sizeS);
        }
    },
};

/** orbit / pan / zoom / fly camera controls via the {@link Orbit} component; one of the default plugins */
export const OrbitPlugin: Plugin = {
    name: "Orbit",
    systems: [OrbitSystem],
    components: { Orbit },
    traits: {
        Orbit: {
            requires: [Transform],
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
            enums: { mode: OrbitMode },
            inputs: {
                yaw: angle,
                pitch: angle,
                minPitch: angle,
                maxPitch: angle,
            },
        },
    },
    dependencies: [InputPlugin, TransformsPlugin],
};

export { OrbitOverlayPlugin } from "./overlay";
