import { expect, setDefaultTimeout, test } from "bun:test";
import { RenderPlugin } from "../../core/rendering";
import { Body, forwardRay, GlobalTransform, PhysicsPlugin } from "../../transitional/physics";
import { build } from "../app";
import * as engine from "../index";
import { globalTransformTable, probeBuffer, Transform } from "../index";
import { Time } from "./scheduler";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

function bounded<T>(promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(new Error("GlobalTransform readback exceeded 750 ms")),
            750,
        );
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}
test("GlobalTransform is an engine-owned public schema, independent of Physics", () => {
    expect(Reflect.get(engine, "GlobalTransform")).toBe(GlobalTransform);
});

test("Transform placement lands in the fixed-tick GlobalTransform column and the renderer table", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const state = app.state;
        const eid = state.create();
        state.add(eid, Transform);
        const placement = state.of(Transform);
        placement.pos.set(eid, 12, 7, -3, 0);
        placement.scale.set(eid, 2, 3, 4, 0);
        state.step(Time.FIXED_DT);
        expect(state.has(eid, GlobalTransform)).toBe(true);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(12);
        expect(state.of(GlobalTransform).pos.y.get(eid)).toBe(7);
        expect(state.of(GlobalTransform).pos.z.get(eid)).toBe(-3);
        const table = globalTransformTable(state);
        const row = table.rowIndex(eid);
        expect(row).toBeGreaterThanOrEqual(0);
        const words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([2, 3, 4]);
    } finally {
        app.dispose();
    }
});

test("a Body writes scale as part of fixed-tick GlobalTransform instead of deriving it only in renderer rows", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const state = app.state;
        const eid = state.create();
        state.add(eid, Body);
        state.of(Body).pos.set(eid, 12, 7, -3, 0);
        state.of(Body).halfExtents.set(eid, 1, 2, 3, 0);
        state.of(Body).mass.set(eid, 0);
        state.step(Time.FIXED_DT);
        expect(state.of(GlobalTransform).pos.x.get(eid)).toBe(12);
        expect(Reflect.get(GlobalTransform, "scale")).toBeDefined();
        const scale = Reflect.get(state.of(GlobalTransform), "scale");
        expect([scale.x.get(eid), scale.y.get(eid), scale.z.get(eid)]).toEqual([2, 4, 6]);
        const table = globalTransformTable(state);
        const row = table.rowIndex(eid);
        const words = new Float32Array(
            (await bounded(probeBuffer(state, table.buffer, { size: table.buffer.size }))).bytes,
        );
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([2, 4, 6]);
    } finally {
        app.dispose();
    }
});

test("a physics camera query reads fixed-tick GlobalTransform without requiring Transform", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const state = app.state;
        const camera = state.registry.getComponent("camera");
        if (!camera) throw new Error("RenderPlugin must register Camera");
        const eid = state.create();
        state.add(eid, Body);
        state.add(eid, camera);
        state.of(Body).pos.set(eid, 12, 7, -3, 0);
        state.of(Body).mass.set(eid, 0);
        state.step(Time.FIXED_DT);
        expect(state.has(eid, Transform)).toBe(false);
        expect(forwardRay(state, eid)).toEqual({ origin: [12, 7, -3], dir: [0, 0, -1] });
    } finally {
        app.dispose();
    }
});
