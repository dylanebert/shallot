import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { type Mesh, Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import { meshResourcesKey } from "../../core/mesh/mesh";
import { attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { Outline, OutlinePlugin } from ".";

setDefaultTimeout(TIMEOUT.node);

const SIZE = 32;
const config = { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] };
const subjects = gpuApps(import.meta.path, [config, config]);

function scene(world: World) {
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: SIZE, height: SIZE });
    const outlined = world.create();
    world.add(outlined, Transform);
    world.add(outlined, MeshInstance);
    world.add(outlined, Outline, { color: 0xff00ff, width: 4 });
    return { camera, outlined };
}

test("warm mesh turnover under an outline creates no bind groups", () => {
    const { world } = subjects()[0]!;
    const { outlined } = scene(world);
    const meshes = world.resource(Meshes);
    const spec = { ...meshes.values().next().value!, name: "turnover" };
    let mesh = meshes.add(spec);
    world.storage(MeshInstance).mesh.set(outlined, mesh);
    const turnover = () => {
        meshes.remove(mesh);
        mesh = meshes.add(spec);
        world.storage(MeshInstance).mesh.set(outlined, mesh);
        world.step(0);
    };
    for (let i = 0; i < 5; i++) turnover();
    const device = world.gpu.device;
    const createBindGroup = device.createBindGroup.bind(device);
    let groups = 0;
    device.createBindGroup = (descriptor) => {
        groups++;
        return createBindGroup(descriptor);
    };
    try {
        for (let i = 0; i < 100; i++) turnover();
    } finally {
        device.createBindGroup = createBindGroup;
    }
    expect(groups).toBe(0);
});

test("an outlined mesh reusing a removed mesh's row from another buffer family draws its own band", async () => {
    const { world } = subjects()[1]!;
    const { camera, outlined } = scene(world);
    const meshes = world.resource(Meshes);
    const rows = world.resource(meshResourcesKey).meshes;
    const frame = async () => {
        world.step(0);
        world.step(0);
        return (await captureTexture(world, camera)).rgba;
    };
    const outline = (handle: MeshHandle) => world.storage(MeshInstance).mesh.set(outlined, handle);

    // A triangle up and to the right, packed into its own family.
    const triangle = meshes.add({
        name: "triangle",
        vertices: new Float32Array([
            0.2, 0.2, 0, 0, 0, 0, 1, 0, 1.6, 0.2, 0, 0, 0, 0, 1, 0, 0.2, 1.6, 0, 0, 0, 0, 1, 0,
        ]),
        indices: new Uint32Array([0, 1, 2]),
    });
    world.step(0);
    const cube: Mesh = { ...meshes.get(rows.handleAt(0)!)!, name: "cube" };
    const other: Mesh = { ...meshes.get(triangle)!, name: "other" };
    expect(other.position).not.toBe(cube.position);

    outline(triangle);
    const expected = await frame();

    const before = meshes.add(cube);
    outline(before);
    const cubeFrame = await frame();
    expect(cubeFrame).not.toEqual(expected);
    const row = rows.index(before);
    meshes.remove(before);
    const after = meshes.add(other);
    expect(rows.index(after)).toBe(row);
    outline(after);
    const actual = await frame();

    let difference = 0;
    for (let i = 0; i < expected.length; i++)
        difference = Math.max(difference, Math.abs(expected[i]! - actual[i]!));
    expect(difference).toBeLessThanOrEqual(1);
});
