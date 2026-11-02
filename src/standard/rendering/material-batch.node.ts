import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { precompile, probeBuffer, type World } from "../../engine";
import {
    batchResolver,
    MaterialBatches,
    materialBatch,
    syncMaterialBatches,
} from "./material-batch";
import { MaterialAssets, materialLayout, materialType } from "./material-type";

setDefaultTimeout(TIMEOUT.node);
const layout = tgpu.bindGroupLayout({
    materialBatches: { storage: d.arrayOf(d.u32), access: "readonly" },
    result: { storage: d.arrayOf(d.u32), access: "mutable" },
});
const resolve = batchResolver(layout);
const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    ({ gid }) => {
        "use gpu";
        layout.$.result[gid.x] = resolve(d.u32(gid.x / 3), gid.x % 3);
    },
);
const scenes = new WeakMap<World, ReturnType<typeof initialize>>();
function initialize(world: World) {
    const state = world.resource(MaterialBatches);
    state.types = [0, 0, 1];
    state.rows = new Uint32Array([2, 4, 6, 8, 0, 1, 2, 2]);
    const root = world.gpu.root;
    const map = root.createBuffer(d.arrayOf(d.u32, 8), state.rows).$usage("storage");
    const result = root.createBuffer(d.arrayOf(d.u32, 9)).$usage("storage");
    const pipeline = root.createComputePipeline({ compute: kernel });
    const group = root.createBindGroup(layout, { materialBatches: map, result });
    precompile(world, "batch-rows", () => [root.unwrap(pipeline)]);
    return { result, pipeline, group };
}
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            {
                name: "BatchRows",
                gpu: {},
                initialize(world) {
                    scenes.set(world, initialize(world));
                },
            },
        ],
    },
    { defaults: false, plugins: [{ name: "ImageSets", gpu: {} }] },
]);

test("image sets compare only uploaded rows, leave idle maps untouched and rewrite changed rows in place", async () => {
    const world = subjects()[1].world;
    const parameters = d.struct({ color: d.u32, data: d.i32 });
    const type = materialType({
        name: "TupleProof",
        layout: materialLayout(parameters, {
            color: { type: "image", field: "color" },
            data: { type: "image", field: "data", isSrgb: false },
        }),
        defaults: { color: 0, data: -1 },
        fragment: () => {
            "use gpu";
            return d.vec4f(1);
        },
    });
    const assets = new MaterialAssets(world, type);
    const a = assets.add({ color: 1, data: -1 });
    const b = assets.add({ color: 1, data: 2 });
    const c = assets.add({ color: 1, data: -1 });
    expect(syncMaterialBatches(world)).toBe(true);
    expect(materialBatch(world, a.type, a.material)).toBe(materialBatch(world, c.type, c.material));
    expect(materialBatch(world, a.type, a.material)).not.toBe(
        materialBatch(world, b.type, b.material),
    );
    const state = world.resource(MaterialBatches);
    expect(state.entries[a.type]!.fields.map(({ isSrgb }) => isSrgb)).toEqual([true, false]);
    const buffer = state.buffer;
    const table = assets.table;
    world.step(0);
    expect(syncMaterialBatches(world)).toBe(false);
    const bytes = table.bytes;
    let reads = 0;
    Object.defineProperty(table, "bytes", {
        configurable: true,
        get() {
            reads++;
            return bytes;
        },
    });
    try {
        world.step(0);
        expect(table.pendingUploadSize).toBe(0);
        for (let i = 0; i < 3; i++) expect(syncMaterialBatches(world)).toBe(false);
        expect(reads).toBe(0);
        expect(state.buffer).toBe(buffer);
        const before = state.rows.slice();
        assets.setAt(c.material, { color: 1, data: 2 });
        expect(table.pendingUploadSize).toBe(table.rowBytes);
        world.step(0);
        reads = 0;
        expect(syncMaterialBatches(world)).toBe(true);
        expect(reads).toBe(1);
        expect(syncMaterialBatches(world)).toBe(false);
        expect(reads).toBe(1);
        expect(state.buffer).toBe(buffer);
        expect(materialBatch(world, c.type, c.material)).toBe(
            materialBatch(world, b.type, b.material),
        );
        const changed = Array.from(before, (value, index) =>
            value !== state.rows[index] ? index : -1,
        ).filter((index) => index >= 0);
        expect([...changed]).toEqual([state.rows[c.type + 1]! + c.material]);
        const uploaded = await probeBuffer(world, world.gpu.root.unwrap(buffer!));
        expect([...new Uint32Array(uploaded.bytes)]).toEqual([...state.rows]);

        assets.setAt(a.material, { color: 1, data: 2 });
        world.step(0);
        assets.setAt(b.material, { color: 1, data: -1 });
        world.step(0);
        expect(syncMaterialBatches(world)).toBe(true);
        expect(materialBatch(world, a.type, a.material)).toBe(
            materialBatch(world, c.type, c.material),
        );
        expect(materialBatch(world, a.type, a.material)).not.toBe(
            materialBatch(world, b.type, b.material),
        );
        expect(state.buffer).toBe(buffer);
        const twiceUploaded = await probeBuffer(world, world.gpu.root.unwrap(buffer!));
        expect([...new Uint32Array(twiceUploaded.bytes)]).toEqual([...state.rows]);

        assets.setAt(a.material, { color: 1, data: -1 });
        world.step(0);
        world.step(0);
        expect(syncMaterialBatches(world), "record upload followed by a no-op upload").toBe(true);
        expect(materialBatch(world, a.type, a.material)).toBe(
            materialBatch(world, b.type, b.material),
        );
    } finally {
        Reflect.deleteProperty(table, "bytes");
    }

    table.reserveSlots(table.capacity + 8);
    const laterAssets = new MaterialAssets(
        world,
        materialType({
            name: "LaterTupleProof",
            layout: type.layout,
            defaults: { color: 0, data: 0 },
            fragment: () => {
                "use gpu";
                return d.vec4f(1);
            },
        }),
    );
    const later = laterAssets.add({ color: 0, data: 0 });
    syncMaterialBatches(world);
    const laterBatch = materialBatch(world, later.type, later.material);
    assets.add({ color: 123, data: 456 });
    syncMaterialBatches(world);
    expect(materialBatch(world, later.type, later.material)).toBe(laterBatch);
    table.reserveSlots(table.capacity + 1);
    const moved: number[] = [];
    expect(
        syncMaterialBatches(world, (_, type, row) => {
            if (type === later.type) moved.push(row);
        }),
    ).toBe(true);
    expect(materialBatch(world, later.type, later.material)).not.toBe(laterBatch);
    expect(moved).toContain(later.material);
});

test("CPU and GPU resolve material rows to batches independently of their type", async () => {
    const world = subjects()[0].world;
    const scene = scenes.get(world)!;
    const expected = [0, 1, 0xffffffff, 2, 2, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff];
    for (let index = 0; index < expected.length; index++) {
        const batch = materialBatch(world, Math.floor(index / 3), index % 3);
        expect(batch < 3 ? batch : 0xffffffff).toBe(expected[index]!);
    }
    const encoder = world.gpu.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(world.gpu.root.unwrap(scene.pipeline));
    pass.setBindGroup(0, world.gpu.root.unwrap(scene.group));
    pass.dispatchWorkgroups(9);
    pass.end();
    world.gpu.device.queue.submit([encoder.finish()]);
    const snapshot = await probeBuffer(world, world.gpu.root.unwrap(scene.result));
    expect([...new Uint32Array(snapshot.bytes)]).toEqual(expected);
});
