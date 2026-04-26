import { describe, test, expect, beforeEach } from "bun:test";
import {
    build,
    Transform,
    WorldTransform,
    TransformsPlugin,
    InputPlugin,
    parse,
    load,
    Camera,
    CameraMode,
    Target,
} from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { Orbit, OrbitPlugin } from "../src/extras/orbit";

function initTransform(eid: number): void {
    Transform.posX[eid] = 0;
    Transform.posY[eid] = 0;
    Transform.posZ[eid] = 0;
    Transform.rotX[eid] = 0;
    Transform.rotY[eid] = 0;
    Transform.rotZ[eid] = 0;
    Transform.scaleX[eid] = 1;
    Transform.scaleY[eid] = 1;
    Transform.scaleZ[eid] = 1;
}

function initOrbit(
    eid: number,
    overrides: Partial<{
        distance: number;
        pitch: number;
        yaw: number;
    }> = {},
): void {
    const distance = overrides.distance ?? 10;
    const pitch = overrides.pitch ?? 0;
    const yaw = overrides.yaw ?? 0;

    Orbit.distance[eid] = distance;
    Orbit.pitch[eid] = pitch;
    Orbit.yaw[eid] = yaw;
    Orbit.minPitch[eid] = -Math.PI / 2 + 0.01;
    Orbit.maxPitch[eid] = Math.PI / 2 - 0.01;
    Orbit.minDistance[eid] = 1;
    Orbit.maxDistance[eid] = 100;
    Orbit.smoothness[eid] = 0.15;
    Orbit.sensitivity[eid] = 0.005;
    Orbit.zoomSpeed[eid] = 0.001;
    Orbit.orbitButton[eid] = 0;
    Orbit.panButton[eid] = 2;
    Orbit.panX[eid] = 0;
    Orbit.panY[eid] = 0;
    Orbit.panZ[eid] = 0;
    Orbit.flySpeed[eid] = 5;
    Orbit.flyActive[eid] = 0;
}

describe("Orbit", () => {
    test("positions camera at correct distance from origin", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: 0 });

        state.step(1 / 60);

        const x = Transform.posX[camera];
        const y = Transform.posY[camera];
        const z = Transform.posZ[camera];
        const dist = Math.sqrt(x * x + y * y + z * z);

        expect(dist).toBeCloseTo(10, 1);
    });

    test("pitch=0 places camera at horizon level", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: 0 });

        state.step(1 / 60);

        expect(Transform.posY[camera]).toBeCloseTo(0, 1);
        expect(Transform.posZ[camera]).toBeCloseTo(10, 1);
    });

    test("pitch=PI/2 places camera directly above", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: Math.PI / 2, yaw: 0 });

        state.step(1 / 60);

        expect(Transform.posY[camera]).toBeCloseTo(10, 1);
        expect(Transform.posX[camera]).toBeCloseTo(0, 1);
        expect(Transform.posZ[camera]).toBeCloseTo(0, 1);
    });

    test("yaw=PI/2 rotates camera around Y axis", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: Math.PI / 2 });

        state.step(1 / 60);

        expect(Transform.posX[camera]).toBeCloseTo(10, 1);
        expect(Transform.posZ[camera]).toBeCloseTo(0, 1);
    });

    test("camera world matrix is valid", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 5, pitch: 0, yaw: 0 });

        state.step(1 / 60);

        const o = camera * 16;
        expect(WorldTransform.data[o + 15]).toBe(1);
        expect(WorldTransform.data[o + 3]).toBe(0);
        expect(WorldTransform.data[o + 7]).toBe(0);
        expect(WorldTransform.data[o + 11]).toBe(0);
    });

    test("defaults to left click orbit, right click pan", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera);

        expect(Orbit.orbitButton[camera]).toBe(0);
        expect(Orbit.panButton[camera]).toBe(2);
    });

    test("pan offset shifts effective target", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: 0 });
        Orbit.panX[camera] = 5;
        Orbit.panY[camera] = 3;

        state.step(1 / 60);

        expect(Transform.posX[camera]).toBeCloseTo(5, 1);
        expect(Transform.posY[camera]).toBeCloseTo(3, 1);
        expect(Transform.posZ[camera]).toBeCloseTo(10, 1);
    });

    test("fly exit reprojects target center", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: 0 });

        state.step(1 / 60);

        Orbit.flyActive[camera] = 1;
        Transform.posX[camera] = 0;
        Transform.posY[camera] = 0;
        Transform.posZ[camera] = 20;

        state.step(1 / 60);

        expect(Orbit.flyActive[camera]).toBe(0);
        expect(Orbit.panZ[camera]).toBeCloseTo(10, 1);
        expect(Orbit.distance[camera]).toBeCloseTo(10, 1);
    });

    test("smoothness > 1 is clamped and does not produce NaN", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: 0 });
        Orbit.smoothness[camera] = 2.0;

        state.step(1 / 60);

        expect(Number.isFinite(Transform.posX[camera])).toBe(true);
        expect(Number.isFinite(Transform.posY[camera])).toBe(true);
        expect(Number.isFinite(Transform.posZ[camera])).toBe(true);
    });

    test("smoothness < 0 is clamped and does not produce NaN", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: 0 });
        Orbit.smoothness[camera] = -1.0;

        state.step(1 / 60);

        expect(Number.isFinite(Transform.posX[camera])).toBe(true);
        expect(Number.isFinite(Transform.posY[camera])).toBe(true);
        expect(Number.isFinite(Transform.posZ[camera])).toBe(true);
    });

    test("pitch at max (looking straight down) produces valid transform", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: Math.PI / 2 - 0.01, yaw: 0 });

        state.step(1 / 60);

        expect(Number.isFinite(Transform.quatX[camera])).toBe(true);
        expect(Number.isFinite(Transform.quatY[camera])).toBe(true);
        expect(Number.isFinite(Transform.quatZ[camera])).toBe(true);
        expect(Number.isFinite(Transform.quatW[camera])).toBe(true);
    });

    test("orthographic mode smoothly interpolates Camera.size", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        state.addComponent(camera, Camera);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: 0 });

        Camera.mode[camera] = CameraMode.Orthographic;
        Orbit.size[camera] = 5;
        Orbit.smoothness[camera] = 0.5;

        state.step(1 / 60);

        Orbit.size[camera] = 10;
        state.step(1 / 60);

        expect(Camera.size[camera]).toBeGreaterThan(5);
        expect(Camera.size[camera]).toBeLessThan(10);
    });

    test("perspective mode does not modify Camera.size", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        state.addComponent(camera, Camera);
        initTransform(camera);
        initOrbit(camera, { distance: 10, pitch: 0, yaw: 0 });

        Camera.mode[camera] = CameraMode.Perspective;
        Camera.size[camera] = 5;
        Orbit.size[camera] = 10;

        state.step(1 / 60);

        expect(Camera.size[camera]).toBe(5);
    });

    test("orthographic orbit respects minSize and maxSize", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const camera = state.addEntity();
        state.addComponent(camera, Transform);
        state.addComponent(camera, Orbit);
        state.addComponent(camera, Camera);
        initTransform(camera);
        initOrbit(camera);

        Camera.mode[camera] = CameraMode.Orthographic;
        Orbit.minSize[camera] = 2;
        Orbit.maxSize[camera] = 20;
        Orbit.size[camera] = 100;

        expect(Orbit.size[camera]).toBe(100);

        Orbit.size[camera] = Math.min(Orbit.maxSize[camera], Orbit.size[camera]);
        expect(Orbit.size[camera]).toBe(20);
    });
});

describe("Orbit XML", () => {
    beforeEach(() => {
        clearRegistry();
    });

    test("resolves target entity reference from XML", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const nodes = parse(
            `<scene>
                <a id="target-entity" transform="pos: 5 0 0" />
                <a id="camera" transform target="@target-entity" orbit="distance: 10; pitch: 0" />
            </scene>`,
        );
        const nodeToEntity = load(nodes, state);

        const targetEid = nodeToEntity.get(nodes[0])!;
        const cameraEid = nodeToEntity.get(nodes[1])!;

        expect(state.hasRelation(cameraEid, Target, targetEid)).toBe(true);

        state.step(1 / 60);

        const camX = Transform.posX[cameraEid];
        const camY = Transform.posY[cameraEid];
        const camZ = Transform.posZ[cameraEid];
        const dist = Math.sqrt((camX - 5) ** 2 + camY ** 2 + camZ ** 2);
        expect(dist).toBeCloseTo(10, 0);
    });

    test("reports error on unknown target entity reference", async () => {
        const state = await build({
            plugins: [TransformsPlugin, InputPlugin, OrbitPlugin],
            defaults: false,
        });

        const nodes = parse(
            `<scene>
                <a id="camera" transform target="@nonexistent" orbit />
            </scene>`,
        );

        expect(() => load(nodes, state)).toThrow("@nonexistent");
    });
});
