import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, TransparentPhases } from "../../core/rendering";
import { GlobalTransform, Transform } from "../../core/transform";
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

test("each view visits only transparent candidates through reassignment, row changes, removal and reuse", () => {
    const { world } = subjects()[0]!;
    const materials = world.resource(Materials);
    const opaque = materials.add(StandardMaterial());
    const glass = materials.add(StandardMaterial(), { alphaMode: AlphaMode.Blend });
    const mesh = world.resource(Meshes).entries().next().value![0];
    const eids: number[] = [];
    for (let i = 0; i < 128; i++) {
        const eid = world.create();
        eids.push(eid);
        world.add(eid, Transform);
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, MeshMaterial, i < 2 ? glass : opaque);
    }
    const cameras: number[] = [];
    for (let i = 0; i < 2; i++) {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        cameras.push(camera);
        attachTexture(world, camera, { width: 16, height: 16 });
    }
    world.step(0);
    const storage = world.storage(MeshMaterial);
    const reads = spyOn(materials, "alphaMode");
    const frame = (expected: number[]) => {
        reads.mockClear();
        world.step(0);
        expect(reads.mock.calls.map(([row]) => row)).toEqual(
            Array(expected.length * cameras.length).fill(glass.material),
        );
        for (const camera of cameras)
            expect(world.resource(TransparentPhases).get(camera)!.metrics.items).toBe(
                expected.length,
            );
    };
    try {
        frame(eids.slice(0, 2));
        storage.material.set(eids[2], glass.material);
        frame(eids.slice(0, 3));
        storage.material.set(eids[0], opaque.material);
        frame(eids.slice(1, 3));
        world.remove(eids[1], MeshInstance);
        frame([eids[2]]);
        world.add(eids[1], MeshInstance, { mesh });
        frame(eids.slice(1, 3));
        world.remove(eids[1], GlobalTransform);
        frame([eids[2]]);
        world.add(eids[1], GlobalTransform);
        frame(eids.slice(1, 3));
        world.remove(eids[1], MeshMaterial);
        frame([eids[2]]);
        world.add(eids[1], MeshMaterial, glass);
        frame(eids.slice(1, 3));
        world.destroy(eids[2]);
        const reused = world.create();
        expect(reused).toBe(eids[2]);
        world.add(reused, Transform);
        world.add(reused, MeshInstance, { mesh });
        world.add(reused, MeshMaterial, opaque);
        frame([eids[1]]);
        materials.update(glass, {}, { alphaMode: AlphaMode.Opaque });
        frame([]);
        materials.update(glass, {}, { alphaMode: AlphaMode.Add });
        frame([eids[1]]);
    } finally {
        reads.mockRestore();
    }
});
