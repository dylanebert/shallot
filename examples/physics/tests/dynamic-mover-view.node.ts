import { expect, setDefaultTimeout, test } from "bun:test";
import { BaseColor, Devices, Materials, MeshMaterial, Shape } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin, Viewer } from "../src/app";
import { DynamicMoverState } from "../src/dynamic-mover-scene";
import { ShapeMaterial } from "../src/material";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues } from "../src/scenes";
import { ShapeMaterials } from "../src/shape-color";

setDefaultTimeout(TIMEOUT.node);
test("Dynamic Mover writes opaque depth when not aiming and restores it after the fade", async () => {
    const app = await openGpuApp({ plugins: [AppPlugin] });
    try {
        const world = app.world;
        sceneSwitcher(world, world.resource(Viewer))("dynamicMover");
        const state = world.resource(DynamicMoverState);
        function moverMaterial() {
            const eid = [...world.query([Shape, MeshMaterial])].find(
                (eid) => (world.storage(Shape).body.get(eid) || eid) === state.entity,
            )!;
            const handle = world.get(eid, MeshMaterial)!;
            const opaque = world.resource(ShapeMaterials).get("motorized")!;
            const source =
                handle.type === opaque.type
                    ? world.resource(ShapeMaterial)
                    : world.resource(Materials);
            const alpha = world.storage(BaseColor).color.w.get(eid);
            expect(alpha).toBeGreaterThanOrEqual(0.3);
            expect(alpha).toBeLessThanOrEqual(1);
            return source.alphaMode(handle.material);
        }
        world.step(1 / 60);
        await world.frameFence;
        expect(moverMaterial()).toBe(0);
        world.resource(Devices).keys.held.add("KeyQ");
        for (let i = 0; i < 15; i++) {
            world.step(1 / 60);
            await world.frameFence;
        }
        expect(moverMaterial()).not.toBe(0);
        const values = controlValues("dynamicMover");
        changeControl(world, "dynamicMover", [], values, "Box Mover", 1);
        for (let i = 0; i < 3; i++) {
            world.step(1 / 60);
            await world.frameFence;
        }
        expect(moverMaterial()).not.toBe(0);
        changeControl(world, "dynamicMover", [], values, "Draw Probes", 0);
        world.step(1 / 60);
        await world.frameFence;
        expect(moverMaterial()).not.toBe(0);
        world.resource(Devices).keys.held.delete("KeyQ");
        for (let i = 0; i < 15; i++) {
            world.step(1 / 60);
            await world.frameFence;
        }
        expect(moverMaterial()).toBe(0);
    } finally {
        app.dispose();
    }
});
