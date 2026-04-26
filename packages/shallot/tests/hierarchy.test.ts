import { test, expect, describe, beforeEach } from "bun:test";
import { build, type State, ChildOf } from "../src/engine";
import { Transform, WorldTransform, TransformsPlugin } from "../src/standard/transforms";
import { getWorldPosition, getWorldScale } from "./helpers/state";

describe("Transform Hierarchy", () => {
    let state: State;

    beforeEach(async () => {
        state = await build({ plugins: [TransformsPlugin], defaults: false });
    });

    test("root entity: world equals local", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        Transform.posX[eid] = 10;
        Transform.posY[eid] = 5;
        Transform.posZ[eid] = -3;
        Transform.rotY[eid] = 90;
        Transform.scaleX[eid] = 2;
        Transform.scaleY[eid] = 1.5;
        Transform.scaleZ[eid] = 0.8;

        state.step();

        const pos = getWorldPosition(eid);
        const scale = getWorldScale(eid);
        expect(pos.x).toBe(10);
        expect(pos.y).toBe(5);
        expect(pos.z).toBe(-3);
        expect(scale.x).toBeCloseTo(2, 5);
        expect(scale.y).toBeCloseTo(1.5, 5);
        expect(scale.z).toBeCloseTo(0.8, 5);
    });

    test("child inherits parent position", () => {
        const parent = state.addEntity();
        const child = state.addEntity();

        state.addComponent(parent, Transform);
        state.addComponent(child, Transform);
        state.addRelation(child, ChildOf, parent);

        Transform.posX[parent] = 10;
        Transform.posY[parent] = 0;
        Transform.posZ[parent] = 0;

        Transform.posX[child] = 5;
        Transform.posY[child] = 3;
        Transform.posZ[child] = 0;

        state.step();

        const pos = getWorldPosition(child);
        expect(pos.x).toBeCloseTo(15, 5);
        expect(pos.y).toBeCloseTo(3, 5);
        expect(pos.z).toBeCloseTo(0, 5);
    });

    test("child inherits parent rotation", () => {
        const parent = state.addEntity();
        const child = state.addEntity();

        state.addComponent(parent, Transform);
        state.addComponent(child, Transform);
        state.addRelation(child, ChildOf, parent);

        Transform.posX[parent] = 0;
        Transform.rotY[parent] = 90;

        Transform.posX[child] = 10;
        Transform.posY[child] = 0;
        Transform.posZ[child] = 0;

        state.step();

        const pos = getWorldPosition(child);
        expect(pos.x).toBeCloseTo(0, 5);
        expect(pos.z).toBeCloseTo(-10, 5);
    });

    test("child inherits parent scale", () => {
        const parent = state.addEntity();
        const child = state.addEntity();

        state.addComponent(parent, Transform);
        state.addComponent(child, Transform);
        state.addRelation(child, ChildOf, parent);

        Transform.scaleX[parent] = 2;
        Transform.scaleY[parent] = 2;
        Transform.scaleZ[parent] = 2;

        Transform.posX[child] = 5;
        Transform.scaleX[child] = 0.5;
        Transform.scaleY[child] = 0.5;
        Transform.scaleZ[child] = 0.5;

        state.step();

        const pos = getWorldPosition(child);
        const scale = getWorldScale(child);
        expect(pos.x).toBeCloseTo(10, 5);
        expect(scale.x).toBeCloseTo(1, 5);
        expect(scale.y).toBeCloseTo(1, 5);
        expect(scale.z).toBeCloseTo(1, 5);
    });

    test("three-level hierarchy", () => {
        const grandparent = state.addEntity();
        const parent = state.addEntity();
        const child = state.addEntity();

        state.addComponent(grandparent, Transform);
        state.addComponent(parent, Transform);
        state.addComponent(child, Transform);
        state.addRelation(parent, ChildOf, grandparent);
        state.addRelation(child, ChildOf, parent);

        Transform.posX[grandparent] = 10;
        Transform.scaleX[grandparent] = 2;
        Transform.scaleY[grandparent] = 2;
        Transform.scaleZ[grandparent] = 2;

        Transform.posX[parent] = 5;
        Transform.scaleX[parent] = 0.5;
        Transform.scaleY[parent] = 0.5;
        Transform.scaleZ[parent] = 0.5;

        Transform.posX[child] = 2;
        Transform.posY[child] = 1;

        state.step();

        expect(getWorldPosition(grandparent).x).toBeCloseTo(10, 5);
        expect(getWorldPosition(parent).x).toBeCloseTo(20, 5);
        expect(getWorldPosition(child).x).toBeCloseTo(22, 5);
        expect(getWorldPosition(child).y).toBeCloseTo(1, 5);
    });

    test("rotation in hierarchy affects child position", () => {
        const parent = state.addEntity();
        const child = state.addEntity();

        state.addComponent(parent, Transform);
        state.addComponent(child, Transform);
        state.addRelation(child, ChildOf, parent);

        Transform.posY[parent] = 3;
        Transform.rotY[parent] = 90;

        Transform.posX[child] = 10;

        state.step();

        const pos = getWorldPosition(child);
        expect(pos.y).toBeCloseTo(3, 5);
        expect(pos.z).toBeCloseTo(-10, 5);
    });

    test("WorldTransform is automatically added", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);

        state.step();
        expect(state.hasComponent(eid, WorldTransform)).toBe(true);
    });

    test("matrix is computed from world transform", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        Transform.posX[eid] = 5;
        Transform.posY[eid] = 10;
        Transform.posZ[eid] = 15;

        state.step();

        const o = eid * 16;
        expect(WorldTransform.data[o + 12]).toBeCloseTo(5, 5);
        expect(WorldTransform.data[o + 13]).toBeCloseTo(10, 5);
        expect(WorldTransform.data[o + 14]).toBeCloseTo(15, 5);
        expect(WorldTransform.data[o + 15]).toBe(1);
    });
});
