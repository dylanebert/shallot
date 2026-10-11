import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, PointLight, StandardRenderingPlugin, Transform } from "@dylanebert/shallot";
import { createTestGpuApp as createApp } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    attachView,
    captureTexture,
    detachCanvas,
    MAX_SLOTS,
    MAX_VIEWS,
    RenderContext,
    Tonemapping,
    TonemappingMethod,
    Views,
    VolumetricLight,
} from "../../core/rendering";
import { tonemappingStateKey } from "../../core/rendering/tonemapping-state";
import {
    presentingViewRefusal,
    presentingViewSlot,
    viewResourcesKey,
} from "../../core/rendering/view";
import { Fog, FogPlugin } from "../../extras/fog";
import { Outline, OutlinePlugin } from "../../extras/outline";
import { CLUSTER_COUNT, CLUSTER_Z, Clusters, clusterGpuKey, LightCull } from "./cluster";
import { StandardRenderer } from "./forward";
import { MAX_POINT_LIGHTS } from "./lighting";
import { Materials, MeshMaterial, StandardMaterial } from "./material";
import { getGroup } from "./pipelines";
import { Draws } from "./registry";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

async function withStandard() {
    return createApp({ plugins: [StandardRenderingPlugin] });
}

async function withVisualRendering() {
    return createApp({ plugins: [StandardRenderingPlugin, FogPlugin, OutlinePlugin] });
}

function renderedCamera(
    world: Awaited<ReturnType<typeof withStandard>>["world"],
    antialias: number,
    presenting = true,
    x = 0,
) {
    const eid = world.create();
    world.add(eid, Transform, { translation: [x, 0, 5, 0] });
    world.add(eid, Camera);
    world.add(eid, StandardRenderer);
    world.add(eid, Tonemapping, { method: TonemappingMethod.KhronosPbrNeutral });
    world.storage(Camera).antialias.set(eid, antialias);
    if (presenting) attachTexture(world, eid, { width: 64, height: 64 });
    else attachView(world, eid);
    return eid;
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

test("CPU slot assignment accepts presenting camera 533 and refuses 534 without corruption", () => {
    const slots: number[] = [];
    let refusal: string | undefined;
    for (let eid = 1; eid <= MAX_VIEWS + 1; eid++) {
        const slot = presentingViewSlot(slots.length);
        if (slot === undefined) refusal = presentingViewRefusal(eid);
        else slots[slot] = eid;
    }
    expect(slots).toHaveLength(MAX_VIEWS);
    expect(slots).toEqual(Array.from({ length: MAX_VIEWS }, (_, slot) => slot + 1));
    expect(refusal).toBe(
        `shallot: ${MAX_VIEWS} presenting-view cull limit reached; entity ${MAX_VIEWS + 1} skipped`,
    );
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

test("growth keeps lit fogged outline pixels coherent with MSAA on and off", async () => {
    for (const antialias of [0, 1]) {
        const app = await withVisualRendering();
        try {
            const world = app.world;
            const first = renderedCamera(world, antialias, true, -3);
            const fog = world.create();
            world.add(fog, Fog, { density: 0.3, steps: 16, jitter: 0 });
            for (const [x, color] of [
                [-3, [0.85, 0.08, 0.02, 1]],
                [3, [0.02, 0.08, 0.85, 1]],
            ] as const) {
                const mesh = world.create();
                world.add(mesh, Transform, { translation: [x, 0, 0, 0] });
                world.add(mesh, MeshInstance);
                world.add(
                    mesh,
                    MeshMaterial,
                    world
                        .resource(Materials)
                        .add(StandardMaterial({ baseColor: color, perceptualRoughness: 0.8 })),
                );
                world.add(mesh, Outline, { color: [0.1, 1, 0.2, 1], width: 3 });
            }
            const pointSpecs = [
                { x: -3, light: { color: 0xff8844, intensity: 315225.92, range: 3.5 } },
                { x: 3, light: { color: 0x4488ff, intensity: 315225.92, range: 3.5 } },
            ] as const;
            const points = pointSpecs.map(({ x, light }) => {
                const point = world.create();
                world.add(point, Transform, { translation: [x, 0, 2, 0] });
                world.add(point, PointLight, light);
                world.add(point, VolumetricLight);
                return point;
            });
            const second = renderedCamera(world, antialias, false, 3);

            world.step(0);
            world.step(0);
            const beforeGrowth = await captureTexture(world, first);
            const draw = [...world.resource(Draws).values()][0]!;
            const entry = getGroup(world, draw.name)!;
            const oldGroup = entry.engineCache.get(0);
            expect(oldGroup).toBeDefined();
            let greenOutlinePixels = 0;
            for (let i = 0; i < beforeGrowth.rgba.length; i += 4) {
                if (
                    beforeGrowth.rgba[i + 1]! > beforeGrowth.rgba[i]! &&
                    beforeGrowth.rgba[i + 1]! > beforeGrowth.rgba[i + 2]!
                )
                    greenOutlinePixels++;
            }
            expect(greenOutlinePixels).toBeGreaterThan(10);

            detachCanvas(world, second);
            attachTexture(world, second, { width: 64, height: 64 });
            world.gpu.device.pushErrorScope("validation");
            world.step(0);
            world.step(0);
            await world.gpu.device.queue.onSubmittedWorkDone();
            const grownFirst = await captureTexture(world, first);
            const grownSecond = await captureTexture(world, second);
            expect(Buffer.from(grownFirst.rgba).equals(Buffer.from(grownSecond.rgba))).toBe(false);
            expect(Buffer.from(beforeGrowth.rgba).equals(Buffer.from(grownFirst.rgba))).toBe(true);
            const grownEntry = getGroup(world, draw.name)!;
            expect(grownEntry).toBe(entry);
            expect(grownEntry.engineCache.get(0)).not.toBe(oldGroup);
            expect(await world.gpu.device.popErrorScope()).toBeNull();

            for (const eid of [first, second]) {
                const pixels = (await captureTexture(world, eid)).rgba;
                let green = 0;
                for (let i = 0; i < pixels.length; i += 4) {
                    if (pixels[i + 1]! > pixels[i]! && pixels[i + 1]! > pixels[i + 2]!) green++;
                }
                expect(green).toBeGreaterThan(10);
            }

            const litCenter = (rgba: Uint8ClampedArray) => {
                const at = (32 * 64 + 32) * 4;
                return [...rgba.subarray(at, at + 3)];
            };
            const lit = [litCenter(grownFirst.rgba), litCenter(grownSecond.rgba)];
            expect(lit[0]).not.toEqual(lit[1]);
            expect(lit[0][0]!).toBeGreaterThan(lit[0][2]!);
            expect(lit[1][2]!).toBeGreaterThan(lit[1][0]!);

            world.remove(fog, Fog);
            world.step(0);
            world.step(0);
            const withoutFog = [
                await captureTexture(world, first),
                await captureTexture(world, second),
            ];
            for (let slot = 0; slot < 2; slot++) {
                expect(lit[slot]).not.toEqual(litCenter(withoutFog[slot]!.rgba));
            }

            world.add(fog, Fog, { density: 0.3, steps: 16, jitter: 0 });
            world.remove(points[0]!, PointLight);
            world.step(0);
            world.step(0);
            const withoutFirstLight = [
                await captureTexture(world, first),
                await captureTexture(world, second),
            ];
            expect(litCenter(withoutFirstLight[0]!.rgba)).not.toEqual(lit[0]);
            expect(litCenter(withoutFirstLight[1]!.rgba)).toEqual(lit[1]);

            world.add(points[0]!, PointLight, pointSpecs[0].light);
            world.remove(points[1]!, PointLight);
            world.step(0);
            world.step(0);
            const withoutSecondLight = [
                await captureTexture(world, first),
                await captureTexture(world, second),
            ];
            expect(litCenter(withoutSecondLight[0]!.rgba)).toEqual(lit[0]);
            expect(litCenter(withoutSecondLight[1]!.rgba)).not.toEqual(lit[1]);

            world.add(points[1]!, PointLight, pointSpecs[1].light);
            world.step(0);
            world.step(0);
            const restored = [
                await captureTexture(world, first),
                await captureTexture(world, second),
            ];
            for (let slot = 0; slot < 2; slot++)
                expect(
                    Buffer.from(restored[slot]!.rgba).equals(
                        Buffer.from([grownFirst, grownSecond][slot]!.rgba),
                    ),
                ).toBe(true);

            for (const eid of [first, second])
                world.storage(Tonemapping).method.set(eid, TonemappingMethod.None);
            world.step(0);
            world.step(0);
            const withoutOperator = [
                await captureTexture(world, first),
                await captureTexture(world, second),
            ];
            for (let slot = 0; slot < 2; slot++) {
                expect(
                    Buffer.from(restored[slot]!.rgba).equals(
                        Buffer.from(withoutOperator[slot]!.rgba),
                    ),
                ).toBe(false);
            }

            for (const eid of [first, second])
                world.storage(Tonemapping).method.set(eid, TonemappingMethod.KhronosPbrNeutral);
            world.step(0);
            world.step(0);
            const beforeSlotShift = await captureTexture(world, second);
            world.remove(first, Camera);
            world.step(0);
            world.step(0);
            expect(world.resource(Views).get(second)?.slot).toBe(0);
            const afterSlotShift = await captureTexture(world, second);
            expect(Buffer.from(afterSlotShift.rgba).equals(Buffer.from(beforeSlotShift.rgba))).toBe(
                true,
            );
        } finally {
            app.dispose();
        }
    }
});

test("removing a camera releases its targets and shifts the remaining tone group with its slot", async () => {
    const app = await withStandard();
    try {
        const world = app.world;
        const first = renderedCamera(world, 0);
        const second = renderedCamera(world, 0);
        world.storage(Tonemapping).method.set(second, TonemappingMethod.None);
        const mesh = world.create();
        world.add(mesh, Transform);
        world.add(mesh, MeshInstance);
        world.add(
            mesh,
            MeshMaterial,
            world.resource(Materials).add(
                StandardMaterial({
                    baseColor: [0.85, 0.08, 0.02, 1],
                    perceptualRoughness: 0.8,
                }),
            ),
        );
        const point = world.create();
        world.add(point, Transform, { translation: [1, 1, 2, 0] });
        world.add(point, PointLight, { color: 0xff8844, intensity: 315225.92, range: 10 });

        world.step(0);
        world.step(0);
        const beforeRemoval = await captureTexture(world, second);
        const removedView = world.resource(Views).get(first)!;
        const viewResources = world.resource(viewResourcesKey);
        const tonemapping = world.resource(tonemappingStateKey);
        expect(viewResources.offscreen.has(first)).toBe(true);
        expect(tonemapping.targets.has(removedView)).toBe(true);

        world.remove(first, Camera);
        world.step(0);
        world.step(0);
        expect(world.resource(Views).has(first)).toBe(false);
        expect(world.resource(Views).get(second)?.slot).toBe(0);
        expect(viewResources.offscreen.has(first)).toBe(false);
        expect(tonemapping.targets.has(removedView)).toBe(false);
        const afterRemoval = await captureTexture(world, second);
        expect(Buffer.from(afterRemoval.rgba).equals(Buffer.from(beforeRemoval.rgba))).toBe(true);

        world.storage(Tonemapping).method.set(second, TonemappingMethod.KhronosPbrNeutral);
        world.step(0);
        world.step(0);
        const afterToneChange = await captureTexture(world, second);
        expect(Buffer.from(afterToneChange.rgba).equals(Buffer.from(afterRemoval.rgba))).toBe(
            false,
        );
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
