import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp, Time, Transform } from "../../engine";
import { Body, PhysicsPlugin } from "./index";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("physics warns once per entity carrying Body and Transform, not for either alone", async () => {
    const app = await createApp({ defaults: false, plugins: [PhysicsPlugin] });
    const warning = spyOn(console, "warn").mockImplementation(() => {});
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body);
        const transform = world.create();
        world.add(transform, Transform);
        world.step(Time.FIXED_DT);
        expect(warning).not.toHaveBeenCalled();
        const both = world.create();
        world.add(both, Body);
        world.add(both, Transform);
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(warning).toHaveBeenCalledTimes(1);
        expect(warning.mock.calls[0]?.[0]).toBe(
            `physics-sync: entity ${both} carries both Body and Transform; both write GlobalTransform`,
        );
    } finally {
        warning.mockRestore();
        app.dispose();
    }
});
