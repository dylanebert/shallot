import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { Transform } from "../../engine";
import { createApp } from "../../engine/app";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { probeTexture } from "../../engine/runtime";
import { BASE_FEATURES } from "../../engine/runtime/gpu";
import { Vignette, VignettePlugin } from "../../extras/vignette";
import { StandardRenderer } from "../../standard/rendering";
import "../../standard";
import {
    attachCanvas,
    attachTexture,
    Camera,
    captureTexture,
    EffectPasses,
    RenderContext,
    Views,
} from "./index";
import { PointsPlugin } from "./points.fixture";

setDefaultTimeout(CEILING.node);
const peerModule = "bun-webgpu";
await (await import(peerModule)).setupGlobals();
if (typeof ResizeObserver === "undefined") {
    globalThis.ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
    };
}

test("standard composition, vignette, after-tonemap and points present and capture without BGRA storage", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Dawn adapter unavailable");
    const requiredLimits: Record<string, number> = { maxStorageBuffersPerShaderStage: 10 };
    for (const limit of [
        "maxStorageBuffersInVertexStage",
        "maxStorageBuffersInFragmentStage",
        "maxStorageTexturesInVertexStage",
        "maxStorageTexturesInFragmentStage",
    ] as const) {
        if (adapter.limits[limit] === 0) requiredLimits[limit] = 0;
    }
    const device = await adapter.requestDevice({
        requiredFeatures: [...BASE_FEATURES],
        requiredLimits,
    });
    console.log("presentation adapter:", adapter.info, "device features:", [...device.features]);
    expect(device.features.has("bgra8unorm-storage")).toBe(false);
    device.pushErrorScope("validation");
    let app: Awaited<ReturnType<typeof createApp>> | undefined;
    try {
        app = await createApp({ device, plugins: [VignettePlugin, PointsPlugin] });
        const { world } = app;
        const canvas = {
            width: 16,
            height: 16,
            style: { imageRendering: "auto" },
            getBoundingClientRect: () => ({ width: 16, height: 16 }),
        } as unknown as HTMLCanvasElement;
        const context = new CanvasContext(canvas, 16, 16);
        canvas.getContext = ((kind: string) =>
            kind === "webgpu" ? context : null) as typeof canvas.getContext;
        const cameras = [world.create(), world.create()];
        for (const eid of cameras) {
            world.add(eid, Transform, { translation: [0, 0, 5, 0] });
            world.add(eid, Camera, { clearColor: 0x4080c0 });
            world.add(eid, StandardRenderer);
        }
        attachCanvas(cameras[0], canvas, world);
        attachTexture(world, cameras[1], { width: 16, height: 16 });
        expect(
            world.resource(Views).get(cameras[1])!.texture!.usage & GPUTextureUsage.STORAGE_BINDING,
        ).toBe(0);
        const module = device.createShaderModule({
            code: `
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
@group(0) @binding(0) var input: texture_2d<f32>;
@fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return vec4f(1.0 - textureLoad(input, vec2u(p.xy), 0).rgb, 1.0);
}`,
        });
        const pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vertex" },
            fragment: {
                module,
                entryPoint: "fragment",
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
        });
        const captures: Uint8ClampedArray[] = [];
        for (const mode of ["standard", "vignette", "after-tonemap"] as const) {
            for (const eid of cameras) {
                if (mode === "vignette") world.add(eid, Vignette, { intensity: 0.5 });
                if (mode === "after-tonemap")
                    world.resource(EffectPasses).set(eid, {
                        before: [],
                        after: [
                            (world, _eid, _view, input, output) => {
                                const group = device.createBindGroup({
                                    layout: pipeline.getBindGroupLayout(0),
                                    entries: [{ binding: 0, resource: input }],
                                });
                                const pass = world
                                    .resource(RenderContext)
                                    .encoder!.beginRenderPass({
                                        colorAttachments: [
                                            { view: output, loadOp: "clear", storeOp: "store" },
                                        ],
                                    });
                                pass.setPipeline(pipeline);
                                pass.setBindGroup(0, group);
                                pass.draw(3);
                                pass.end();
                            },
                        ],
                    });
            }
            world.step(0);
            const shot = await captureTexture(world, cameras[1]);
            expect(shot.rgba.some((byte, i) => i % 4 !== 3 && byte > 0)).toBe(true);
            expect(shot.rgba.filter((_byte, i) => i % 4 === 3).every((byte) => byte === 255)).toBe(
                true,
            );
            if (mode === "standard") {
                let greenPoints = 0;
                for (let i = 0; i < shot.rgba.length; i += 4)
                    if (shot.rgba[i + 1] > shot.rgba[i] && shot.rgba[i + 1] > shot.rgba[i + 2])
                        greenPoints++;
                expect(greenPoints).toBeGreaterThan(0);
            }
            captures.push(shot.rgba);
            const texture = context.getCurrentTexture();
            expect(texture.usage & GPUTextureUsage.STORAGE_BINDING).toBe(0);
            // Adopt this fixture's persistent texture for the world-owned readback seam.
            // A real swapchain texture remains context-owned and is captured in the page.
            world.own(texture);
            const canvasShot = await probeTexture(world, texture);
            expect(canvasShot.bytes).toHaveLength(16 * 16 * 4);
        }
        expect(captures[1]).not.toEqual(captures[0]);
        expect(captures[2]).not.toEqual(captures[1]);
    } finally {
        app?.dispose();
        expect(await device.popErrorScope()).toBeNull();
        device.destroy();
    }
});
