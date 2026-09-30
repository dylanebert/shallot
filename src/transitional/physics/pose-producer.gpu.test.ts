import { expect, setDefaultTimeout, test } from "bun:test";
import { build, Transform } from "../../engine";
import { Body, PhysicsPlugin } from "./index";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

for (const [first, second, firstName, secondName] of [
    [Transform, Body, "transform", "body"],
    [Body, Transform, "body", "transform"],
] as const) {
    test(`a second pose producer ${secondName} refuses after ${firstName} without changing membership`, async () => {
        const app = await build({ defaults: false, plugins: [PhysicsPlugin] });
        try {
            const state = app.state;
            const eid = state.create();
            state.add(eid, first);
            expect(() => state.add(eid, second)).toThrow(
                new RegExp(`cannot attach "${secondName}".*excluded by "${firstName}"`),
            );
            expect(state.has(eid, first)).toBe(true);
            expect(state.has(eid, second)).toBe(false);
        } finally {
            app.dispose();
        }
    });
}
