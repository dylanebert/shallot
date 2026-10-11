import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, createApp, StandardRenderingPlugin, Transform } from "@dylanebert/shallot";
import { CEILING } from "../../../scripts/test-tiers";
import { attachTexture, MAX_SLOTS, MAX_VIEWS, RenderContext, Views } from "../../core/rendering";
import { tonemappingStateKey } from "../../core/rendering/tonemapping-state";
import { CLUSTER_COUNT, CLUSTER_Z, Clusters, clusterGpuKey, LightCull } from "./cluster";
import { MAX_POINT_LIGHTS } from "./lighting";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

async function withStandard() {
    return createApp({ plugins: [StandardRenderingPlugin] });
}

function camera(world: Awaited<ReturnType<typeof withStandard>>["world"]): number {
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera);
    attachTexture(world, eid, { width: 64, height: 64 });
    return eid;
}

test("the fixed cull uniform holds the presenting prefix and all 52 shadow slots", () => {
    expect(MAX_SLOTS).toBe(585);
    expect(MAX_VIEWS).toBe(533);
});

test("a ninth presenting camera draws instead of being skipped", async () => {
    const app = await withStandard();
    try {
        const cameras = Array.from({ length: 9 }, () => camera(app.world));
        app.world.step(0);
        const render = app.world.resource(RenderContext);
        expect(render.shadeCount).toBe(9);
        expect(render.viewCount).toBe(9);
        for (const eid of cameras) expect(app.world.resource(Views).get(eid)?.presented).toBe(true);
    } finally {
        app.dispose();
    }
});

test("one presenting camera allocates per-view state for one slot", async () => {
    const app = await withStandard();
    try {
        camera(app.world);
        app.world.step(0);
        const render = app.world.resource(RenderContext);
        const clusters = app.world.resource(Clusters);
        const lightCull = app.world.resource(LightCull);
        const tonemapping = app.world.resource(tonemappingStateKey);
        expect(render.viewBuffers).toHaveLength(1);
        expect(clusters.aabbs?.size).toBe(CLUSTER_COUNT * 2 * 16);
        expect(clusters.views?.size).toBe(8 * 4);
        expect(lightCull.viewMats?.size).toBe(64);
        const clusterGpu = app.world.resource(clusterGpuKey);
        expect(clusterGpu.clusterCounts?.size).toBe(3584 * 8);
        expect(clusterGpu.zSlices?.size).toBe(MAX_POINT_LIGHTS * CLUSTER_Z * 8);
        expect(clusterGpu.rasterArgs?.size).toBe(20);
        expect(tonemapping.buffers).toHaveLength(1);
    } finally {
        app.dispose();
    }
});

test("adding a camera grows once; steady frames allocate and compile nothing", async () => {
    const app = await withStandard();
    const methods = [
        "createBuffer",
        "createBindGroup",
        "createComputePipeline",
        "createComputePipelineAsync",
        "createRenderPipeline",
        "createRenderPipelineAsync",
    ] as const;
    const calls = { buffers: 0, bindGroups: 0, pipelines: 0 };
    const originals: Array<unknown> = [];
    try {
        camera(app.world);
        app.world.step(0);
        const device = app.world.gpu.device!;
        for (const name of methods) originals.push(device[name]);
        for (const [index, name] of methods.entries()) {
            Reflect.set(device, name, (...args: unknown[]) => {
                if (name === "createBuffer") calls.buffers++;
                else if (name === "createBindGroup") calls.bindGroups++;
                else calls.pipelines++;
                return Reflect.apply(
                    originals[index] as (...args: unknown[]) => unknown,
                    device,
                    args,
                );
            });
        }
        const beforeGrowth = { ...calls };
        camera(app.world);
        app.world.step(0);
        await app.world.gpu.device!.queue.onSubmittedWorkDone();
        expect(app.world.resource(RenderContext).shadeCount).toBe(2);
        expect(app.world.resource(RenderContext).viewBuffers).toHaveLength(2);
        const grown = { ...calls };
        expect(grown.buffers).toBeGreaterThan(beforeGrowth.buffers);
        expect(grown.bindGroups).toBeGreaterThan(beforeGrowth.bindGroups);
        expect(grown.pipelines).toBe(beforeGrowth.pipelines);
        app.world.step(0);
        app.world.step(0);
        await app.world.gpu.device!.queue.onSubmittedWorkDone();
        expect(calls).toEqual(grown);
    } finally {
        if (app.world.gpu.device) {
            const device = app.world.gpu.device;
            for (const [index, name] of methods.entries()) {
                if (originals[index]) Reflect.set(device, name, originals[index]);
            }
        }
        app.dispose();
    }
});
