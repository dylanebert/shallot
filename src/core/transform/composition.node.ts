import { expect, setDefaultTimeout, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../../engine";
import { PhysicsPlugin } from "../physics";
import { RenderingPlugin } from "../rendering";
import { GlobalTransform, Transform, TransformPlugin } from "./index";

setDefaultTimeout(CEILING.node);
await setupGlobals();

for (const plugin of [PhysicsPlugin, RenderingPlugin]) {
    test(`${plugin.name} gets placement through its Transform dependency`, async () => {
        expect(plugin.dependencies?.some((dependency) => dependency === TransformPlugin)).toBe(
            true,
        );
        const app = await createApp({ defaults: false, plugins: [plugin] });
        try {
            expect([...app.world.registry.entries()].map((entry) => entry.key)).toContain(
                "Transform",
            );
            const eid = app.world.create();
            app.world.add(eid, Transform, { translation: [3, 2, 1, 0] });
            app.world.step(1 / 60);
            expect(app.world.storage(GlobalTransform).translation.x.get(eid)).toBe(3);
        } finally {
            app.dispose();
        }
    });
}

test("a composition with no placement reader registers no placement and steps", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    try {
        const names = [...app.world.registry.entries()].map((entry) => entry.key);
        expect(names).not.toContain("Transform");
        expect(names).not.toContain("GlobalTransform");
        app.world.step(1 / 60);
        expect(app.world.time.fixedTick).toBeGreaterThan(0);
    } finally {
        app.dispose();
    }
});
