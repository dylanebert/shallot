import { test, expect, describe, beforeAll, afterAll } from "bun:test";
import * as wasm from "../src/standard/transforms/wasm";
import { build } from "../src/engine";
import { eulerToQuaternion } from "../src/engine/utils";
import { Transform, WorldTransform, TransformsPlugin } from "../src/standard/transforms";

describe("WASM Module", () => {
    beforeAll(async () => {
        await wasm.init();
    });

    test("initial capacity is non-zero", () => {
        expect(wasm.posX.length).toBeGreaterThan(0);
    });

    test("arrays are accessible", () => {
        expect(wasm.posX.length).toBeGreaterThan(0);
        expect(wasm.matrices.length).toBe(wasm.posX.length * 16);
    });
});

describe("WASM Memory Access", () => {
    beforeAll(async () => {
        await wasm.init();
    });

    test("JS can write to WASM memory", () => {
        wasm.posX[0] = 123.456;
        wasm.quatW[0] = 1;
        wasm.scaleX[0] = 2;

        expect(wasm.posX[0]).toBeCloseTo(123.456, 3);
        expect(wasm.quatW[0]).toBeCloseTo(1, 5);
        expect(wasm.scaleX[0]).toBeCloseTo(2, 5);
    });

    test("WASM compute reads JS writes", () => {
        wasm.posX[0] = 10;
        wasm.posY[0] = 20;
        wasm.posZ[0] = 30;
        wasm.quatX[0] = 0;
        wasm.quatY[0] = 0;
        wasm.quatZ[0] = 0;
        wasm.quatW[0] = 1;
        wasm.scaleX[0] = 1;
        wasm.scaleY[0] = 1;
        wasm.scaleZ[0] = 1;

        wasm.indices[0] = 0;
        wasm.parents[0] = wasm.NoParent;
        wasm.compute(1);

        expect(wasm.matrices[12]).toBeCloseTo(10, 5);
        expect(wasm.matrices[13]).toBeCloseTo(20, 5);
        expect(wasm.matrices[14]).toBeCloseTo(30, 5);

        expect(wasm.matrices[0]).toBeCloseTo(1, 5);
        expect(wasm.matrices[5]).toBeCloseTo(1, 5);
        expect(wasm.matrices[10]).toBeCloseTo(1, 5);
        expect(wasm.matrices[15]).toBeCloseTo(1, 5);
    });
});

describe("WASM Transform Computation", () => {
    beforeAll(async () => {
        await wasm.init();
    });

    afterAll(() => {});

    test("computes identity transform correctly", () => {
        const eid = 5;
        const o = eid * 16;

        wasm.posX[eid] = 0;
        wasm.posY[eid] = 0;
        wasm.posZ[eid] = 0;
        wasm.quatX[eid] = 0;
        wasm.quatY[eid] = 0;
        wasm.quatZ[eid] = 0;
        wasm.quatW[eid] = 1;
        wasm.scaleX[eid] = 1;
        wasm.scaleY[eid] = 1;
        wasm.scaleZ[eid] = 1;

        wasm.indices[0] = eid;
        wasm.parents[0] = wasm.NoParent;
        wasm.compute(1);

        expect(wasm.matrices[o + 0]).toBeCloseTo(1, 5);
        expect(wasm.matrices[o + 5]).toBeCloseTo(1, 5);
        expect(wasm.matrices[o + 10]).toBeCloseTo(1, 5);
        expect(wasm.matrices[o + 15]).toBeCloseTo(1, 5);
        expect(wasm.matrices[o + 1]).toBeCloseTo(0, 5);
        expect(wasm.matrices[o + 4]).toBeCloseTo(0, 5);
    });

    test("matches JS TransformSystem output for 90deg Y rotation", async () => {
        const state = await build({ plugins: [TransformsPlugin], defaults: false });
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        Transform.rotY[eid] = 90;
        state.step();

        const o = eid * 16;
        const jsM00 = WorldTransform.data[o];
        const jsM20 = WorldTransform.data[o + 8];
        const jsM02 = WorldTransform.data[o + 2];
        const jsM22 = WorldTransform.data[o + 10];

        expect(jsM00).toBeCloseTo(0, 5);
        expect(jsM20).toBeCloseTo(1, 5);
        expect(jsM02).toBeCloseTo(-1, 5);
        expect(jsM22).toBeCloseTo(0, 5);
    });

    test("computes batch correctly", () => {
        const start = 100;
        const end = 200;

        for (let i = start; i < end; i++) {
            const q = eulerToQuaternion(0, i % 360, 0);
            wasm.posX[i] = i;
            wasm.posY[i] = i * 2;
            wasm.posZ[i] = i * 3;
            wasm.quatX[i] = q.x;
            wasm.quatY[i] = q.y;
            wasm.quatZ[i] = q.z;
            wasm.quatW[i] = q.w;
            wasm.scaleX[i] = 1;
            wasm.scaleY[i] = 1;
            wasm.scaleZ[i] = 1;
            wasm.indices[i - start] = i;
            wasm.parents[i - start] = wasm.NoParent;
        }

        wasm.compute(end - start);

        for (let i = start; i < end; i++) {
            const o = i * 16;
            expect(wasm.matrices[o + 12]).toBeCloseTo(i, 5);
            expect(wasm.matrices[o + 13]).toBeCloseTo(i * 2, 5);
            expect(wasm.matrices[o + 14]).toBeCloseTo(i * 3, 5);
        }
    });
});
