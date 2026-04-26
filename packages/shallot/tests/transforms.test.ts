import { test, expect, describe, beforeEach, afterAll } from "bun:test";
import { build, type State, ChildOf, parse, load } from "../src/engine";
import { Transform, WorldTransform, TransformsPlugin } from "../src/standard/transforms";
import { rotate } from "../src/engine/utils";
import { getWorldPosition } from "./helpers/state";

afterAll(() => {});

describe("TransformSyncSystem", () => {
    let state: State;

    beforeEach(async () => {
        state = await build({ plugins: [TransformsPlugin], defaults: false });
    });

    test("euler angles sync to world matrix", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        Transform.rotY[eid] = 90;

        state.step();

        const o = eid * 16;
        expect(WorldTransform.data[o]).toBeCloseTo(0, 5);
        expect(WorldTransform.data[o + 8]).toBeCloseTo(1, 5);
        expect(WorldTransform.data[o + 2]).toBeCloseTo(-1, 5);
        expect(WorldTransform.data[o + 10]).toBeCloseTo(0, 5);
    });

    test("default transform values", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);

        state.step();

        expect(Transform.posX[eid]).toBe(0);
        expect(Transform.posY[eid]).toBe(0);
        expect(Transform.posZ[eid]).toBe(0);
        expect(Transform.rotX[eid]).toBeCloseTo(0, 5);
        expect(Transform.rotY[eid]).toBeCloseTo(0, 5);
        expect(Transform.rotZ[eid]).toBeCloseTo(0, 5);
        expect(Transform.scaleX[eid]).toBe(1);
        expect(Transform.scaleY[eid]).toBe(1);
        expect(Transform.scaleZ[eid]).toBe(1);
        const o = eid * 16;
        expect(WorldTransform.data[o]).toBe(1);
        expect(WorldTransform.data[o + 5]).toBe(1);
        expect(WorldTransform.data[o + 10]).toBe(1);
        expect(WorldTransform.data[o + 15]).toBe(1);
    });
});

describe("Automatic change detection", () => {
    let state: State;

    beforeEach(async () => {
        state = await build({ plugins: [TransformsPlugin], defaults: false });
    });

    test("new transform is processed on first step", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        Transform.posX[eid] = 10;

        state.step();

        expect(getWorldPosition(eid).x).toBe(10);
    });

    test("changed transform is reprocessed", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        state.step();

        Transform.posX[eid] = 50;
        state.step();

        expect(getWorldPosition(eid).x).toBe(50);
    });

    test("parent change cascades to children", () => {
        const parent = state.addEntity();
        const child = state.addEntity();
        state.addComponent(parent, Transform);
        state.addComponent(child, Transform);
        state.addRelation(child, ChildOf, parent);
        state.step();

        Transform.posX[parent] = 10;
        state.step();

        expect(getWorldPosition(child).x).toBeCloseTo(10, 5);
    });
});

describe("Transform from scene", () => {
    let state: State;

    beforeEach(async () => {
        state = await build({ plugins: [TransformsPlugin], defaults: false });
    });

    test("parses transform from scene", () => {
        const nodes = parse(
            `
            <scene>
                <a id="player" transform="pos: 1 2 3; scale: 2" />
            </scene>
        `,
        );
        const nodeToEntity = load(nodes, state);

        const eid = nodeToEntity.get(nodes[0])!;
        expect(state.hasComponent(eid, Transform)).toBe(true);
        expect(Transform.posX[eid]).toBe(1);
        expect(Transform.posY[eid]).toBe(2);
        expect(Transform.posZ[eid]).toBe(3);
        expect(Transform.scaleX[eid]).toBe(2);
        expect(Transform.scaleY[eid]).toBe(2);
        expect(Transform.scaleZ[eid]).toBe(2);
    });
});

describe("Euler wrapping", () => {
    let state: State;

    beforeEach(async () => {
        state = await build({ plugins: [TransformsPlugin], defaults: false });
    });

    test("euler wraps at 360 boundary", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        Transform.rotY[eid] = 355;

        Transform.rotY[eid] = 365;
        const wrapped = Transform.rotY[eid];

        expect(wrapped).toBeCloseTo(5, 0);
    });

    test("euler wraps negative values", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        Transform.rotY[eid] = -45;

        const wrapped = Transform.rotY[eid];
        expect(Math.abs(wrapped - -45) < 1 || Math.abs(wrapped - 315) < 1).toBe(true);
    });
});

describe("Continuous rotation with rotate()", () => {
    let state: State;

    beforeEach(async () => {
        state = await build({ plugins: [TransformsPlugin], defaults: false });
    });

    function applyRotate(eid: number, dx: number, dy: number, dz: number): void {
        const q = rotate(
            Transform.quatX[eid],
            Transform.quatY[eid],
            Transform.quatZ[eid],
            Transform.quatW[eid],
            dx,
            dy,
            dz,
        );
        Transform.quatX[eid] = q.x;
        Transform.quatY[eid] = q.y;
        Transform.quatZ[eid] = q.z;
        Transform.quatW[eid] = q.w;
    }

    test("rotate accumulates past 360 degrees", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);

        for (let i = 0; i < 40; i++) {
            applyRotate(eid, 0, 10, 0);
        }

        state.step();

        expect(WorldTransform.data[eid * 16]).toBeCloseTo(Math.cos((400 * Math.PI) / 180), 3);
    });

    test("continuous rotation simulation", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);

        const speed = 45;
        const dt = 1 / 60;

        for (let i = 0; i < 600; i++) {
            applyRotate(eid, 0, speed * dt, 0);
        }

        state.step();

        const totalDegrees = speed * dt * 600;
        const expectedCos = Math.cos((totalDegrees * Math.PI) / 180);
        expect(WorldTransform.data[eid * 16]).toBeCloseTo(expectedCos, 2);
    });
});
