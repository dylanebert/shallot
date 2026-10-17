import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, TransparentPhases } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] },
]);

test("snapshot restore reconciles transparent candidates both ways and after destruction and eid reuse", () => {
    const { world } = subjects()[0]!;
    const materials = world.resource(Materials);
    const opaque = materials.add(StandardMaterial());
    const glass = materials.add(StandardMaterial(), { alphaMode: AlphaMode.Blend });
    const mesh = world.resource(Meshes).entries().next().value![0];
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh });
    world.add(eid, MeshMaterial, opaque);
    const cameras: number[] = [];
    for (let i = 0; i < 2; i++) {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 16, height: 16 });
        cameras.push(camera);
    }
    world.step(0);
    const reads = spyOn(materials, "alphaMode");
    const frame = (transparent: boolean) => {
        reads.mockClear();
        world.step(0);
        expect(reads.mock.calls.map(([row]) => row)).toEqual(
            transparent ? cameras.map(() => glass.material) : [],
        );
        for (const camera of cameras) {
            const phase = world.resource(TransparentPhases).get(camera)!;
            expect(phase.metrics.items).toBe(transparent ? 1 : 0);
            if (transparent) expect(phase.item(0).stableId).toBe(eid);
        }
    };
    const storage = world.storage(MeshMaterial);
    try {
        frame(false);
        const opaqueSnapshot = world.snapshot();
        storage.material.set(eid, glass.material);
        frame(true);
        world.restore(opaqueSnapshot);
        frame(false);

        storage.material.set(eid, glass.material);
        frame(true);
        const transparentSnapshot = world.snapshot();
        storage.material.set(eid, opaque.material);
        frame(false);
        world.restore(transparentSnapshot);
        frame(true);

        world.destroy(eid);
        frame(false);
        const reused = world.create();
        expect(reused).toBe(eid);
        world.add(reused, Transform);
        world.add(reused, MeshInstance, { mesh });
        world.add(reused, MeshMaterial, opaque);
        frame(false);
        world.restore(transparentSnapshot);
        expect(world.has(eid, MeshInstance)).toBe(true);
        expect(storage.material.get(eid)).toBe(glass.material);
        frame(true);
    } finally {
        reads.mockRestore();
    }
});
