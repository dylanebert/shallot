import { expect, setDefaultTimeout, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../../engine";
import { PhysicsPlugin } from "../physics";
import { globalTransformTable, RenderingPlugin } from "../rendering";
import { GlobalTransform, Transform, TransformPlugin } from "./index";

setDefaultTimeout(CEILING.node);
await setupGlobals();

test("placement without rendering refuses placement GPU residency", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [{ name: "PlacementResidencyTestDevice", gpu: {} }, TransformPlugin],
    });
    const device = app.world.gpu.device;
    const methods = ["createBuffer", "createComputePipeline", "createRenderPipeline"] as const;
    const descriptors = methods.map((name) => Object.getOwnPropertyDescriptor(device, name));
    let allocations = 0;
    for (const name of methods) {
        const original = device[name].bind(device);
        Object.defineProperty(device, name, {
            configurable: true,
            value: (...args: unknown[]) => {
                allocations++;
                return Reflect.apply(original, device, args);
            },
        });
    }
    try {
        expect(() => globalTransformTable(app.world)).toThrow("RenderingPlugin");
        const eid = app.world.create();
        app.world.add(eid, Transform);
        app.world.tick();
        app.world.step(0);
        expect(allocations).toBe(0);
        expect(app.world.storage(GlobalTransform).scale.x.get(eid)).toBe(1);
    } finally {
        methods.forEach((name, i) => {
            const descriptor = descriptors[i];
            if (descriptor) Object.defineProperty(device, name, descriptor);
            else Reflect.deleteProperty(device, name);
        });
        app.dispose();
    }
});

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
