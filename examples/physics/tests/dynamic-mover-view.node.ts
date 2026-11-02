import { expect, setDefaultTimeout, test } from "bun:test";
import { Devices, Materials, MeshMaterial, Shape } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { AppPlugin, Viewer } from "../src/app";
import { DynamicMoverState } from "../src/dynamic-mover-scene";
import { sceneSwitcher } from "../src/sample";

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
            return world.resource(Materials).alphaMode(handle.material);
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
