import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance, registerMesh } from "../../core/mesh";
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

const rendering = { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] };
// the shared labels and handles world, then one for the draw-args growth row
const subjects = gpuApps(import.meta.path, [rendering, rendering]);

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
    expect(await coveredPixels(world, camera)).toBeGreaterThan(100);
});

test("a MeshInstance with an unregistered mesh handle draws nothing", async () => {
    const { world } = subjects()[0];
    const mesh = registerMesh(world, { name: "registered", ...quad(0.9) });
    const camera = world.create();
    // the subject world is shared; this scene sits clear of the previous test's view
    world.add(camera, Transform, { translation: [0, 100, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 64, height: 64 });
    const material = world.resource(Materials).add(StandardMaterial({ emissive: [1, 1, 1] }));
    world.step(0);
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 100, 0, 0] });
    world.add(eid, MeshInstance, { mesh: world.resource(Meshes).size + mesh });
    world.add(eid, MeshMaterial, material);
    world.step(0);
    world.step(0);
    expect(await coveredPixels(world, camera)).toBe(0);
    world.storage(MeshInstance).mesh.set(eid, mesh);
    world.step(0);
    world.step(0);
    expect(await coveredPixels(world, camera)).toBeGreaterThan(100);
});

test("registering a mesh after warm writes the draw-args records once per view slot, and growing only the instance rows rewrites none", async () => {
    const { world } = subjects()[1];
    const cameras = [-50, 50].map((x) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [x, 0, 5, 0] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 64, height: 64 });
        return camera;
    });
    const material = world.resource(Materials).add(StandardMaterial({ emissive: [1, 1, 1] }));
    for (let i = 0; i < 20; i++) registerMesh(world, { name: `warm-${i}`, ...quad(0.1) });
    world.step(0);
    world.step(0);

    const queue = world.gpu.device.queue;
    const writeBuffer = queue.writeBuffer.bind(queue);
    let writes = 0;
    queue.writeBuffer = ((buffer: GPUBuffer, ...rest: unknown[]) => {
        if (buffer.label === "shallot-draw-args") writes++;
        return (writeBuffer as (...args: unknown[]) => void)(buffer, ...rest);
    }) as GPUQueue["writeBuffer"];
    try {
        // a mesh joins the pair dimension: every record moves, but one write per view slot carries them
        const mesh = registerMesh(world, { name: "late", ...quad(0.9) });
        const eid = world.create();
        world.add(eid, Transform, { translation: [-50, 0, 0, 0] });
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, MeshMaterial, material);
        world.step(0);
        world.step(0);
        expect(writes).toBeGreaterThan(0);
        expect(writes).toBeLessThanOrEqual(cameras.length);
        expect(await coveredPixels(world, cameras[0]!)).toBeGreaterThan(100);

        // spawning past the instance table's capacity replaces the instance lists, not the draw args
        writes = 0;
        for (let i = 0; i < 300; i++) {
            const extra = world.create();
            world.add(extra, Transform, { translation: [50, 0, -100, 0] });
            world.add(extra, MeshInstance, { mesh });
            world.add(extra, MeshMaterial, material);
        }
        world.step(0);
        world.step(0);
        expect(writes).toBe(0);
        expect(await coveredPixels(world, cameras[0]!)).toBeGreaterThan(100);
    } finally {
        queue.writeBuffer = writeBuffer;
    }
});

async function coveredPixels(world: ReturnType<typeof subjects>[0]["world"], camera: number) {
    const { rgba } = await captureTexture(world, camera);
    let covered = 0;
    for (let i = 0; i < rgba.length; i += 4) {
        if (rgba[i]! > 64 || rgba[i + 1]! > 64 || rgba[i + 2]! > 64) covered++;
    }
    return covered;
}
