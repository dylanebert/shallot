import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance, registerMesh } from "../../core/mesh";
import { attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(CEILING.node);

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] },
]);

function quad(halfExtent: number): { vertices: Float32Array; indices: Uint32Array } {
    const vertices = new Float32Array([
        -halfExtent,
        -halfExtent,
        0,
        0,
        0,
        0,
        1,
        0,
        halfExtent,
        -halfExtent,
        0,
        1,
        0,
        0,
        1,
        0,
        halfExtent,
        halfExtent,
        0,
        1,
        0,
        0,
        1,
        1,
        -halfExtent,
        halfExtent,
        0,
        0,
        0,
        0,
        1,
        1,
    ]);
    return { vertices, indices: new Uint32Array([0, 1, 2, 0, 2, 3]) };
}

test("two meshes with the same label keep distinct geometry when rendered", async () => {
    const { world } = subjects()[0];
    const large = registerMesh(world, { name: "shared-label", ...quad(0.9) });
    const small = registerMesh(world, { name: "shared-label", ...quad(0.2) });
    expect(small).not.toBe(large);
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 64, height: 64 });
    const material = world.resource(Materials).add(StandardMaterial({ emissive: [1, 1, 1] }));
    for (const [x, mesh] of [
        [-1, large],
        [1, small],
    ] as const) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [x, 0, 0, 0] });
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, MeshMaterial, material);
    }
    world.step(0);
    world.step(0);
    const { rgba } = await captureTexture(world, camera);
    let covered = 0;
    for (let i = 0; i < rgba.length; i += 4) {
        if (rgba[i]! > 64 || rgba[i + 1]! > 64 || rgba[i + 2]! > 64) covered++;
    }
    expect(covered).toBeGreaterThan(100);
});
