import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, captureTexture, RenderLayers } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { Outline, OutlinePlugin } from ".";

setDefaultTimeout(TIMEOUT.node);
const subject = gpuScene(
    import.meta.path,
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
    (world) => {
        const created: number[] = [];
        const entity = () => {
            const eid = world.create();
            created.push(eid);
            return eid;
        };
        const camera = entity();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera, { antialias: 0 });
        world.add(camera, StandardRenderer);
        const target = attachTexture(world, camera, { width: 48, height: 48 });
        const held = entity();
        world.add(held, Transform, { translation: [0, 0, 5] });
        world.add(held, Camera, { order: 1, antialias: 0 });
        world.add(held, StandardRenderer);
        world.add(held, RenderLayers, { mask: 2 });
        attachTexture(world, held, target);
        const mesh = entity();
        world.add(mesh, Transform);
        world.add(mesh, MeshInstance);
        world.add(mesh, Outline, { color: 0xff00ff, width: 4, occlude: 1 });
        const wall = entity();
        world.add(wall, Transform, { translation: [0, 0, 2], scale: [4, 4, 1] });
        world.add(wall, MeshInstance);
        return { world, created, camera, held, target, mesh, wall };
    },
);
test("a held-only camera cannot reveal a world outline behind a world wall; layer edits take effect next frame", async () => {
    const {
        scene: { world, created, camera, held, target, mesh, wall },
    } = subject();
    const band = async () => {
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        let count = 0;
        for (let i = 0; i < rgba.length; i += 4)
            if (rgba[i]! - rgba[i + 1]! > 100 && rgba[i + 2]! - rgba[i + 1]! > 100) count++;
        return count;
    };
    try {
        // Missing layers on the mesh and world camera both mean mask 1.
        world.remove(held, Camera);
        expect(await band()).toBe(0);
        world.add(held, Camera, { order: 1, antialias: 0 });
        attachTexture(world, held, target);
        expect(await band()).toBe(0);
        world.destroy(wall);
        expect(await band()).toBeGreaterThan(0);
        world.add(mesh, RenderLayers, { mask: 4 });
        expect(await band()).toBe(0);
        world.storage(RenderLayers).mask.set(held, 4);
        expect(await band()).toBeGreaterThan(0);
        world.storage(RenderLayers).mask.set(mesh, 6);
        expect(await band()).toBeGreaterThan(0);
        world.storage(RenderLayers).mask.set(held, 2);
        expect(await band()).toBeGreaterThan(0);
        world.storage(RenderLayers).mask.set(mesh, 4);
        expect(await band()).toBe(0);
        world.remove(mesh, RenderLayers);
        expect(await band()).toBeGreaterThan(0);
        world.add(camera, RenderLayers, { mask: 8 });
        expect(await band()).toBe(0);
        world.remove(camera, RenderLayers);
        expect(await band()).toBeGreaterThan(0);
    } finally {
        for (const eid of created.reverse()) if (world.has(eid, Transform)) world.destroy(eid);
    }
});
