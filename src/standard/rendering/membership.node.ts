import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { cube, Meshes, MeshInstance } from "../../core/mesh";
import { Transform } from "../../core/transform";
import { probeBuffer } from "../../engine/runtime";
import {
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    registerProducerTable,
    StandardMaterial,
    StandardRenderingPlugin,
} from "./index";
import { MeshDrawBuffers, meshPreprocessKey } from "./preprocess";
import { Draws } from "./registry";

setDefaultTimeout(TIMEOUT.node);
const apps = gpuApps(import.meta.path, [
    { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] },
]);

test("freed mesh handles clear bounds and entity/producer records, then publish the successor without stale state", async () => {
    const { world } = apps()[0]!;
    const meshes = world.resource(Meshes);
    const handle = meshes.add({ name: "membership", ...cube([1, 1, 1]) });
    const material = world.resource(Materials).add(StandardMaterial({}));
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh: handle });
    world.add(eid, MeshMaterial, material);
    registerProducerTable(world, {
        name: "membership",
        capacity: 1,
        meshes: [handle],
        materials: [material],
    });
    world.step(0);
    const original = meshes.get(handle)!;
    const draws = [...world.resource(Draws)].filter((draw) => draw.mesh === handle);
    expect(draws.some((draw) => draw.name.startsWith("producer:"))).toBe(true);
    expect(draws.some((draw) => draw.name.startsWith("mesh:"))).toBe(true);
    const bounds = async () => {
        const state = world.resource(meshPreprocessKey);
        return new Float32Array(
            (await probeBuffer(world, world.gpu.root.unwrap(state.meshBounds!))).bytes,
        ).slice(handle * 4, handle * 4 + 4);
    };
    const records = async () =>
        new Uint32Array(
            (
                await probeBuffer(
                    world,
                    world.gpu.root.unwrap(world.resource(MeshDrawBuffers).drawArgs!),
                )
            ).bytes,
        );
    expect(await bounds()).toEqual(new Float32Array(original.bounds!));
    meshes.remove(handle);
    world.remove(eid, MeshInstance);
    world.step(0);
    expect(await bounds()).toEqual(new Float32Array(4));
    for (const draw of draws) expect(world.resource(Draws).has(draw.name)).toBe(false);
    const empty = await records();
    for (const draw of draws) {
        const offset = draw.args.offset! / 4;
        expect(empty.slice(offset, offset + 5)).toEqual(new Uint32Array(5));
    }
    const successor = {
        ...original,
        name: "successor",
        bounds: [7, 8, 9, 10] as [number, number, number, number],
        indexCount: 3,
        indexBase: original.indexBase + 3,
    };
    expect(meshes.add(successor)).toBe(handle);
    world.add(eid, MeshInstance, { mesh: handle });
    world.step(0);
    expect(await bounds()).toEqual(new Float32Array(successor.bounds));
    const current = await records();
    for (const draw of draws) {
        expect(world.resource(Draws).get(draw.name)).toBe(draw);
        const offset = draw.args.offset! / 4;
        expect(current[offset]).toBe(3);
        expect(current[offset + 2]).toBe(successor.indexBase);
        expect(current[offset + 3]).toBe(0);
    }
});
