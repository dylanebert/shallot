import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { precompile, probeBuffer, type World } from "../../engine";
import { batchResolver, MaterialBatches, materialBatch } from "./material-batch";

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
]);

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
