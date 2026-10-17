import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { cube, Meshes, MeshInstance } from "../../core/mesh";
import { clearMeshes } from "../../core/mesh/mesh";
import { attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    AlphaMode,
    Draws,
    Materials,
    MaterialTypes,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";
import { MeshDrawBuffers } from "./preprocess";
import { DrawIndexedIndirect } from "./registry";

setDefaultTimeout(TIMEOUT.node);
const config = { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] };
const subjects = gpuApps(import.meta.path, [config, config, config, config, config]);
function scene(index: number) {
    const { world } = subjects()[index]!;
    const meshes = world.resource(Meshes);
    const handles = [0, 1, 2].map((i) =>
        meshes.add({ name: `cube-${i}`, ...cube([0.8, 0.8, 0.8]) }),
    );
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 96, height: 96 });
    const material = world.resource(Materials).add(StandardMaterial({ emissive: [1, 1, 1] }));
    const entities = handles.map((mesh, i) => {
        const eid = world.create();
        world.add(eid, Transform, { translation: [(i - 1) * 1.5, 0, 0, 0] });
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, MeshMaterial, material);
        return eid;
    });
    world.step(0);
    world.step(0);
    return { world, meshes, handles, camera, entities };
}
test("clearing and repopulating meshes preserves alpha-key pair addressing in every view", async () => {
    const s = scene(3);
    const second = s.world.create();
    s.world.add(second, Transform, { translation: [0, 0, 5, 0] });
    s.world.add(second, Camera);
    s.world.add(second, StandardRenderer);
    attachTexture(s.world, second, { width: 96, height: 96 });
    const material = s.world
        .resource(Materials)
        .add(StandardMaterial({ emissive: [1, 1, 1] }), { alphaMode: AlphaMode.Mask(0.5) });
    expect(s.world.resource(MaterialTypes).types.filter(Boolean).length).toBeGreaterThan(1);
    s.world.step(0);
    clearMeshes(s.world);
    const mesh = s.meshes.add({ name: "after-clear", ...cube([0.8, 0.8, 0.8]) });
    for (const eid of s.entities) {
        s.world.storage(MeshInstance).mesh.set(eid, mesh);
        s.world.storage(MeshMaterial).material.set(eid, material.material);
    }
    s.world.step(0);
    s.world.step(0);
    expect((await coverage(s)).every((n) => n > 0)).toBe(true);
    expect((await coverage({ ...s, camera: second })).every((n) => n > 0)).toBe(true);
});
test("an independently registered draw survives mesh republication", () => {
    const s = scene(4);
    const draws = s.world.resource(Draws);
    const independent = {
        ...[...draws][0]!,
        name: "independent",
        args: { indirect: s.world.gpu.root.createBuffer(DrawIndexedIndirect).$usage("indirect") },
    };
    draws.register(independent);
    s.meshes.add({ name: "republish", ...cube() });
    s.world.step(0);
    expect([...draws].find((draw) => draw.name === independent.name)).toBe(independent);
});
async function coverage(s: ReturnType<typeof scene>) {
    const { rgba } = await captureTexture(s.world, s.camera);
    const counts = [0, 0, 0];
    for (let y = 0; y < 96; y++)
        for (let x = 0; x < 96; x++)
            if (rgba[(y * 96 + x) * 4]! > 64) counts[Math.floor(x / 32)]!++;
    return counts;
}
test("removing the middle of three meshes stops only its draw", async () => {
    const s = scene(0);
    expect((await coverage(s)).every((n) => n > 0)).toBe(true);
    expect(s.meshes.remove(s.handles[1]!)).toBeDefined();
    s.world.step(0);
    s.world.step(0);
    const counts = await coverage(s);
    expect(counts[0]).toBeGreaterThan(0);
    expect(counts[1]).toBe(0);
    expect(counts[2]).toBeGreaterThan(0);
});
test("a reissued mesh handle draws the new geometry without a membership-count change", async () => {
    const s = scene(1);
    const before = (await coverage(s))[1]!;
    s.meshes.remove(s.handles[1]!);
    const geometry = cube([0.2, 0.2, 0.2]);
    const replacement = s.meshes.add({
        name: "small",
        ...geometry,
        indices: geometry.indices.slice(0, 6),
    });
    expect(replacement).toBe(s.handles[1]!);
    s.world.step(0);
    s.world.step(0);
    const after = (await coverage(s))[1]!;
    expect(after).toBeGreaterThan(0);
    expect(after).toBeLessThan(before);
    const draw = [...s.world.resource(Draws)].find((draw) => draw.mesh === replacement)!;
    const args = await draw.args.indirect.read();
    expect((args as { indexCount: number }[])[draw.args.offset! / 20]!.indexCount).toBe(6);
});
test("a hundred fixed-live-count mesh turnovers keep draw and bounds buffer sizes flat", () => {
    const s = scene(2);
    const sizes = new Map<string, number>();
    const device = s.world.gpu.device;
    const create = device.createBuffer.bind(device);
    let creations = 0;
    device.createBuffer = (descriptor) => {
        if (descriptor.label === "shallot-mesh-bounds" || descriptor.label === "shallot-draw-args")
            sizes.set(descriptor.label, descriptor.size);
        creations++;
        return create(descriptor);
    };
    try {
        let handle = s.handles[1]!;
        let first: number[] = [];
        for (let i = 0; i <= 100; i++) {
            s.meshes.remove(handle);
            handle = s.meshes.add({ ...s.meshes.get(s.handles[0]!)!, name: `turnover-${i}` });
            s.world.step(0);
            const current = [
                s.world.gpu.root.unwrap(s.world.resource(MeshDrawBuffers).drawArgs!).size,
                sizes.get("shallot-mesh-bounds")!,
            ];
            if (i === 0) {
                first = current;
                creations = 0;
            } else {
                expect(current).toEqual(first);
                expect(creations).toBe(0);
            }
        }
    } finally {
        device.createBuffer = create;
    }
});
