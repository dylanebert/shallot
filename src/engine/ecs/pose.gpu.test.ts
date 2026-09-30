import { expect, setDefaultTimeout, test } from "bun:test";
import { RenderPlugin } from "../../core/rendering";
import { Body, forwardRay, PhysicsPlugin, Pose } from "../../transitional/physics";
import { Transform, transformTable } from "../../transitional/transforms";
import * as engine from "../index";
import { build } from "../app";
import { Time } from "./scheduler";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

// Stage 6 red checkpoint: until the engine owns Pose, use the existing Physics schema to expose
// missing production behavior rather than failing module resolution on a nonexistent export.
test("Pose is an engine-owned public schema, independent of Physics", () => {
    expect(Reflect.get(engine, "Pose")).toBe(Pose);
});

test("Transform placement lands in the fixed-tick Pose column and the renderer table", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin, RenderPlugin] });
    try {
        const state = app.state;
        const eid = state.create();
        state.add(eid, Transform);
        const placement = state.of(Transform);
        placement.pos.set(eid, 12, 7, -3, 0);
        placement.scale.set(eid, 2, 3, 4, 0);
        state.step(Time.FIXED_DT);
        expect(state.has(eid, Pose)).toBe(true);
        expect(state.of(Pose).pos.x.get(eid)).toBe(12);
        expect(state.of(Pose).pos.y.get(eid)).toBe(7);
        expect(state.of(Pose).pos.z.get(eid)).toBe(-3);
        const table = transformTable(state);
        const row = table.rowIndex(eid);
        expect(row).toBeGreaterThanOrEqual(0);
        const words = new Float32Array(table.bytes.buffer);
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([2, 3, 4]);
    } finally {
        app.dispose();
    }
});

test("a Body writes scale as part of fixed-tick Pose instead of deriving it only in renderer rows", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin] });
    try {
        const state = app.state;
        const eid = state.create();
        state.add(eid, Body);
        state.of(Body).pos.set(eid, 12, 7, -3, 0);
        state.of(Body).halfExtents.set(eid, 1, 2, 3, 0);
        state.of(Body).mass.set(eid, 0);
        state.step(Time.FIXED_DT);
        expect(state.of(Pose).pos.x.get(eid)).toBe(12);
        expect(Reflect.get(Pose, "scale")).toBeDefined();
        const scale = Reflect.get(state.of(Pose), "scale");
        expect([scale.x.get(eid), scale.y.get(eid), scale.z.get(eid)]).toEqual([2, 4, 6]);
        const table = transformTable(state);
        const row = table.rowIndex(eid);
        const words = new Float32Array(table.bytes.buffer);
        expect(Array.from(words.subarray(row * 12, row * 12 + 3))).toEqual([12, 7, -3]);
        expect(Array.from(words.subarray(row * 12 + 8, row * 12 + 11))).toEqual([2, 4, 6]);
    } finally {
        app.dispose();
    }
});

test("a physics camera query reads fixed-tick Pose without requiring Transform", async () => {
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
