import { expect, setDefaultTimeout, test } from "bun:test";
import { build } from "../../engine";
import { Transform } from "../transforms";
import { Body, PhysicsPlugin } from "./index";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("a second pose producer refuses in either insertion order without changing membership", async () => {
    const app = await build({ defaults: false, plugins: [PhysicsPlugin] });
    try {
        const state = app.state;
        const placed = state.create();
        state.add(placed, Transform);
        expect(() => state.add(placed, Body)).toThrow();
        expect(state.has(placed, Transform)).toBe(true);
        expect(state.has(placed, Body)).toBe(false);

        const simulated = state.create();
        state.add(simulated, Body);
        expect(() => state.add(simulated, Transform)).toThrow();
        expect(state.has(simulated, Body)).toBe(true);
        expect(state.has(simulated, Transform)).toBe(false);
    } finally {
        app.dispose();
    }
});
