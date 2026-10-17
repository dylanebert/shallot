import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import type { System } from "../../engine";
import { ClearChangeMarksSystem } from "../../engine/app";
import {
    AlphaMode,
    MaterialPlugin,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    StandardMaterial,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
const OtherParameters = d.struct({ color: d.vec4f });
const OtherMaterial = materialType({
    name: "AlphaPreparationOther",
    parameters: OtherParameters,
    defaults: { color: d.vec4f(1) },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
});
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(OtherMaterial)],
    },
]);

for (const [schedule, order] of [
    ["simulation-last", { group: "simulation", last: true }],
    ["simulation-terminal", { group: "simulation", terminal: true }],
    ["draw-before-upload", { group: "draw", first: true, before: [ClearChangeMarksSystem] }],
] as const) {
    for (const field of ["material", "alphaMode"] as const) {
        test(`${schedule} ${field} writes reconcile by the next frame`, () => {
            const { world } = subjects()[0];
            const materials = world.resource(Materials);
            const opaque = materials.add(StandardMaterial());
            const glass = materials.add(StandardMaterial(), { alphaMode: AlphaMode.Blend });
            const storage = world.storage(MeshMaterial);
            const eid = world.create();
            world.add(eid, MeshMaterial, field === "material" ? opaque : glass);
            world.step(0);
            let pending = true;
            const writer: System = {
                ...order,
                update() {
                    if (!pending) return;
                    pending = false;
                    if (field === "material") storage.material.set(eid, glass.material);
                    else storage.alphaMode.set(eid, opaque.alphaMode);
                },
            };
            world.addSystem(writer);
            try {
                world.step(0);
                world.step(0);
                expect(storage.alphaMode.get(eid)).toBe(glass.alphaMode);
            } finally {
                world.removeSystem(writer);
                world.destroy(eid);
            }
        });
    }
}

test("alpha preparation reads only changed members, including joins, late writes and reused eids", () => {
    const { world } = subjects()[0];
    const materials = world.resource(Materials);
    const opaque = materials.add(StandardMaterial());
    const glass = materials.add(StandardMaterial(), { alphaMode: AlphaMode.Blend });
    const storage = world.storage(MeshMaterial);
    const entities = Array.from({ length: 256 }, () => {
        const eid = world.create();
        world.add(eid, MeshMaterial, { type: opaque.type, material: opaque.material });
        return eid;
    });
    world.step(0);
    const reads = spyOn(storage.material, "get");
    try {
        world.step(0);
        expect(reads).toHaveBeenCalledTimes(0);
        storage.material.set(entities[0], glass.material);
        world.step(0);
        expect(reads).toHaveBeenCalledTimes(1);
        expect(storage.alphaMode.get(entities[0])).toBe(glass.alphaMode);
        reads.mockClear();
        storage.alphaMode.set(entities[0], opaque.alphaMode);
        storage.alphaCutoff.set(entities[0], 0.1);
        world.step(0);
        expect(reads).toHaveBeenCalledTimes(1);
        expect(storage.alphaMode.get(entities[0])).toBe(glass.alphaMode);
        expect(storage.alphaCutoff.get(entities[0])).toBe(glass.alphaCutoff);
        reads.mockClear();
        world.step(0);
        expect(reads).toHaveBeenCalledTimes(0);
    } finally {
        reads.mockRestore();
    }

    // Draw follows the head-of-draw upload: these marks survive to the next frame.
    let write = true;
    const late = {
        group: "draw" as const,
        update() {
            if (!write) return;
            storage.material.set(entities[1], glass.material);
            storage.alphaMode.set(entities[0], opaque.alphaMode);
            write = false;
        },
    };
    world.addSystem(late);
    world.step(0);
    expect(storage.alphaMode.get(entities[1])).toBe(opaque.alphaMode);
    world.step(0);
    expect(storage.alphaMode.get(entities[1])).toBe(glass.alphaMode);
    expect(storage.alphaMode.get(entities[0])).toBe(glass.alphaMode);
    world.removeSystem(late);

    materials.update(glass, {}, { alphaMode: AlphaMode.Mask(0.3) });
    expect(storage.alphaMode.get(entities[0])).toBe(glass.alphaMode);
    expect(storage.alphaCutoff.get(entities[0])).toBeCloseTo(0.3);
    world.step(0);
    world.destroy(entities[0]);
    const reused = world.create();
    expect(reused).toBe(entities[0]);
    world.add(reused, MeshMaterial, { type: opaque.type, material: opaque.material });
    world.step(0);
    expect(storage.alphaMode.get(reused)).toBe(opaque.alphaMode);
    const other = world
        .resource(OtherMaterial)
        .add({ color: d.vec4f(1) }, { alphaMode: AlphaMode.Add });
    storage.type.set(reused, other.type);
    storage.material.set(reused, other.material);
    world.step(0);
    expect(storage.alphaMode.get(reused)).toBe(other.alphaMode);

    const joined = world.create();
    const stop = world.observeMembership(MeshMaterial, (eid, present) => {
        if (present && eid === joined) storage.material.set(entities[2], glass.material);
    });
    world.add(joined, MeshMaterial, { type: other.type, material: other.material });
    stop();
    world.step(0);
    expect(storage.alphaMode.get(joined)).toBe(other.alphaMode);
    expect(storage.alphaMode.get(entities[2])).toBe(glass.alphaMode);
});
