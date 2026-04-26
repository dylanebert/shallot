import { not, traits, Target, type State, type System, type Plugin } from "../../engine";
import { clamp, lookAt } from "../../engine/utils";
import { Transform } from "../../standard/transforms";
import { Inputs, InputPlugin, type Mouse } from "../../standard/input";
import { Camera, CameraMode, RenderTarget } from "../../standard/render";

const Tau = Math.PI * 2;
const HalfPi = Math.PI / 2;
const Deg2Rad = Math.PI / 180;

export const Orbit = {
    yaw: [] as number[],
    pitch: [] as number[],
    distance: [] as number[],
    size: [] as number[],
    minPitch: [] as number[],
    maxPitch: [] as number[],
    minDistance: [] as number[],
    maxDistance: [] as number[],
    minSize: [] as number[],
    maxSize: [] as number[],
    smoothness: [] as number[],
    sensitivity: [] as number[],
    zoomSpeed: [] as number[],
    orbitButton: [] as number[],
    panButton: [] as number[],
    panX: [] as number[],
    panY: [] as number[],
    panZ: [] as number[],
    flySpeed: [] as number[],
    flyActive: [] as number[],
    suppress: [] as number[],
};

traits(Orbit, {
    requires: [Transform],
    defaults: () => ({
        yaw: Math.PI / 6,
        pitch: Math.PI / 9,
        distance: 10,
        size: 5,
        minPitch: -HalfPi + 0.01,
        maxPitch: HalfPi - 0.01,
        minDistance: 1,
        maxDistance: 30,
        minSize: 0.5,
        maxSize: 50,
        smoothness: 0.3,
        sensitivity: 0.005,
        zoomSpeed: 0.025,
        orbitButton: 0,
        panButton: 2,
        panX: 0,
        panY: 0,
        panZ: 0,
        flySpeed: 5,
        flyActive: 0,
        suppress: 0,
    }),
});

export const OrbitSmooth = {
    yaw: [] as number[],
    pitch: [] as number[],
    distance: [] as number[],
    size: [] as number[],
};

function smoothLerp(smoothness: number, dt: number): number {
    const s = Math.max(0, Math.min(1, smoothness));
    return 1 - Math.pow(1 - s, dt * 60);
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

function hasMovementKey(input: Inputs): boolean {
    return (
        input.isKeyDown("KeyW") ||
        input.isKeyDown("KeyS") ||
        input.isKeyDown("KeyA") ||
        input.isKeyDown("KeyD") ||
        input.isKeyDown("KeyQ") ||
        input.isKeyDown("KeyE")
    );
}

const OrbitSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },

    update(state: State) {
        const input = Inputs.from(state);
        const dt = state.time.deltaTime;

        for (const eid of state.query([Orbit, not(OrbitSmooth)])) {
            state.addComponent(eid, OrbitSmooth);
            OrbitSmooth.yaw[eid] = Orbit.yaw[eid];
            OrbitSmooth.pitch[eid] = Orbit.pitch[eid];
            OrbitSmooth.distance[eid] = Orbit.distance[eid];
            OrbitSmooth.size[eid] = Orbit.size[eid];
        }

        for (const eid of state.query([not(Orbit), OrbitSmooth])) {
            state.removeComponent(eid, OrbitSmooth);
        }

        for (const eid of state.query([Orbit, OrbitSmooth, Transform])) {
            const sensitivity = Orbit.sensitivity[eid];
            const zoomSpeed = Orbit.zoomSpeed[eid];
            const minPitch = Orbit.minPitch[eid];
            const maxPitch = Orbit.maxPitch[eid];
            const smoothness = Orbit.smoothness[eid];

            const hasCamera = state.hasComponent(eid, Camera);
            const isOrtho = hasCamera && Camera.mode[eid] === CameraMode.Orthographic;
            const isActive = state.getFirstRelationTarget(eid, RenderTarget) >= 0;

            const flying = isActive && !!input && hasMovementKey(input);

            const suppressed = !!Orbit.suppress[eid];

            if (!suppressed && isActive && input && isButton(input.mouse, Orbit.orbitButton[eid])) {
                Orbit.yaw[eid] -= input.mouse.deltaX * sensitivity;
                Orbit.pitch[eid] = clamp(
                    Orbit.pitch[eid] + input.mouse.deltaY * sensitivity,
                    minPitch,
                    maxPitch,
                );
            }

            if (!flying && isActive && input && isButton(input.mouse, Orbit.panButton[eid])) {
                const yaw = OrbitSmooth.yaw[eid];
                const pitch = OrbitSmooth.pitch[eid];
                const dist = Orbit.distance[eid];

                const cy = Math.cos(yaw);
                const sy = Math.sin(yaw);
                const cp = Math.cos(pitch);
                const sp = Math.sin(pitch);

                const rightX = cy;
                const rightZ = -sy;

                const upX = -sp * sy;
                const upY = cp;
                const upZ = -sp * cy;

                let worldPerPixel: number;
                if (isOrtho) {
                    worldPerPixel = (Camera.size[eid] * 2) / input.mouse.canvasHeight;
                } else {
                    const fovRad = hasCamera ? Camera.fov[eid] * Deg2Rad : 60 * Deg2Rad;
                    worldPerPixel = (2 * dist * Math.tan(fovRad * 0.5)) / input.mouse.canvasHeight;
                }

                const dx = input.mouse.deltaX * worldPerPixel;
                const dy = input.mouse.deltaY * worldPerPixel;
                Orbit.panX[eid] += dy * upX - dx * rightX;
                Orbit.panY[eid] += dy * upY;
                Orbit.panZ[eid] += dy * upZ - dx * rightZ;
            }

            if (isActive && input && input.mouse.scroll !== 0) {
                if (isOrtho) {
                    const current = Orbit.size[eid];
                    const sizeScale = Math.max(0.1, current * 0.08);
                    const zoomDelta = input.mouse.scroll * zoomSpeed * sizeScale;
                    Orbit.size[eid] = clamp(
                        current + zoomDelta,
                        Orbit.minSize[eid],
                        Orbit.maxSize[eid],
                    );
                } else {
                    const current = Orbit.distance[eid];
                    const distanceScale = Math.max(0.3, current * 0.08);
                    const zoomDelta = input.mouse.scroll * zoomSpeed * distanceScale;
                    Orbit.distance[eid] = clamp(
                        current + zoomDelta,
                        Orbit.minDistance[eid],
                        Orbit.maxDistance[eid],
                    );
                }
            }

            const t = smoothLerp(smoothness, dt);
            OrbitSmooth.yaw[eid] += angleDiff(OrbitSmooth.yaw[eid], Orbit.yaw[eid]) * t;
            OrbitSmooth.pitch[eid] += (Orbit.pitch[eid] - OrbitSmooth.pitch[eid]) * t;
            OrbitSmooth.distance[eid] += (Orbit.distance[eid] - OrbitSmooth.distance[eid]) * t;

            if (isOrtho) {
                OrbitSmooth.size[eid] += (Orbit.size[eid] - OrbitSmooth.size[eid]) * t;
                Camera.size[eid] = OrbitSmooth.size[eid];
            }

            if (flying) {
                Orbit.flyActive[eid] = 1;
                const speed = Orbit.flySpeed[eid] * dt;
                const yaw = OrbitSmooth.yaw[eid];
                const fp = -OrbitSmooth.pitch[eid];
                const cy = Math.cos(yaw);
                const sy = Math.sin(yaw);
                const cp = Math.cos(fp);
                const sp = Math.sin(fp);

                let mx = 0;
                let mz = 0;
                let my = 0;
                if (input!.isKeyDown("KeyW")) mz -= 1;
                if (input!.isKeyDown("KeyS")) mz += 1;
                if (input!.isKeyDown("KeyA")) mx -= 1;
                if (input!.isKeyDown("KeyD")) mx += 1;
                if (input!.isKeyDown("KeyQ")) my -= 1;
                if (input!.isKeyDown("KeyE")) my += 1;

                Transform.posX[eid] += (mz * sy * cp + mx * cy) * speed;
                Transform.posY[eid] += (my - mz * sp) * speed;
                Transform.posZ[eid] += (mz * cy * cp - mx * sy) * speed;

                const hy = yaw * 0.5;
                const hp = fp * 0.5;
                const shy = Math.sin(hy);
                const chy = Math.cos(hy);
                const shp = Math.sin(hp);
                const chp = Math.cos(hp);
                Transform.quatX[eid] = chy * shp;
                Transform.quatY[eid] = shy * chp;
                Transform.quatZ[eid] = -shy * shp;
                Transform.quatW[eid] = chy * chp;
            } else {
                if (Orbit.flyActive[eid]) {
                    Orbit.flyActive[eid] = 0;
                    const yaw = OrbitSmooth.yaw[eid];
                    const pitch = OrbitSmooth.pitch[eid];
                    const dist = OrbitSmooth.distance[eid];

                    let entityTargetX = 0;
                    let entityTargetY = 0;
                    let entityTargetZ = 0;
                    const targetEid = state.getFirstRelationTarget(eid, Target);
                    if (targetEid >= 0 && state.hasComponent(targetEid, Transform)) {
                        entityTargetX = Transform.posX[targetEid];
                        entityTargetY = Transform.posY[targetEid];
                        entityTargetZ = Transform.posZ[targetEid];
                    }

                    Orbit.panX[eid] =
                        Transform.posX[eid] -
                        dist * Math.cos(pitch) * Math.sin(yaw) -
                        entityTargetX;
                    Orbit.panY[eid] = Transform.posY[eid] - dist * Math.sin(pitch) - entityTargetY;
                    Orbit.panZ[eid] =
                        Transform.posZ[eid] -
                        dist * Math.cos(pitch) * Math.cos(yaw) -
                        entityTargetZ;
                }

                let targetX = Orbit.panX[eid];
                let targetY = Orbit.panY[eid];
                let targetZ = Orbit.panZ[eid];
                const targetEid = state.getFirstRelationTarget(eid, Target);
                if (targetEid >= 0 && state.hasComponent(targetEid, Transform)) {
                    targetX += Transform.posX[targetEid];
                    targetY += Transform.posY[targetEid];
                    targetZ += Transform.posZ[targetEid];
                }

                const yaw = OrbitSmooth.yaw[eid];
                const pitch = OrbitSmooth.pitch[eid];
                const distance = OrbitSmooth.distance[eid];

                const camX = targetX + distance * Math.cos(pitch) * Math.sin(yaw);
                const camY = targetY + distance * Math.sin(pitch);
                const camZ = targetZ + distance * Math.cos(pitch) * Math.cos(yaw);

                Transform.posX[eid] = camX;
                Transform.posY[eid] = camY;
                Transform.posZ[eid] = camZ;

                const rotation = lookAt(camX, camY, camZ, targetX, targetY, targetZ);
                Transform.quatX[eid] = rotation.x;
                Transform.quatY[eid] = rotation.y;
                Transform.quatZ[eid] = rotation.z;
                Transform.quatW[eid] = rotation.w;
            }
        }
    },
};

export const OrbitPlugin: Plugin = {
    name: "Orbit",
    systems: [OrbitSystem],
    components: { Orbit },
    relations: [Target],
    dependencies: [InputPlugin],
};
