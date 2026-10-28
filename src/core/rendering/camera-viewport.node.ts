import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { type Plugin, precompile, type World } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { Transform } from "../transform";
import { renderFrameKey } from "./frame-state";
import {
    attachCanvas,
    attachTexture,
    Camera,
    CameraMode,
    CameraViewport,
    CorePipelinePlugin,
    captureFrame,
    captureTexture,
    detachView,
    RenderContext,
    RenderingPlugin,
    RenderPhases,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();
const fills = { create: () => new Set<number>() };
const FillPlugin: Plugin = {
    name: "ViewportFillFixture",
    dependencies: [CorePipelinePlugin],
    initialize(world) {
        const module = world.gpu.device.createShaderModule({
            code: `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
 let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
 return vec4f(p[i], 0.5, 1);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
`,
        });
        const pipeline = world.gpu.device.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: world.resource(RenderContext).format }],
            },
            depthStencil: {
                format: "depth32float",
                depthWriteEnabled: true,
                depthCompare: "greater",
            },
        });
        precompile(world, "viewport-fill", () => [pipeline]);
        world.resource(RenderPhases).push({
            opaque(world, eid, _view, pass) {
                if (!world.resource(fills).has(eid)) return;
                pass.setPipeline(pipeline);
                pass.draw(3);
            },
        });
    },
};
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [RenderingPlugin, FillPlugin] },
]);
function camera(world: World, order: number) {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0, 5, 0] });
    world.add(eid, Camera, {
        order,
        antialias: 0,
        mode: CameraMode.Orthographic,
        size: 4,
        clearColor: 0x204060,
    });
    world.add(eid, Tonemapping, { method: TonemappingMethod.None });
    return eid;
}
function pixel(frame: Uint8ClampedArray, width: number, x: number, y: number) {
    return [...frame.subarray((y * width + x) * 4, (y * width + x + 1) * 4)];
}
class TestCanvas {}

for (const canvasTarget of [false, true]) {
    for (const claim of [
        "corner preserves outside bytes",
        "projection aspect",
        "disjoint halves",
        "partial clipping after shrink",
        "outside origin after shrink",
        "bottom clears whole target",
        "zero target",
        "CSS pixels ignore pixelRatio",
    ] as const) {
        if (!canvasTarget && (claim === "zero target" || claim === "CSS pixels ignore pixelRatio"))
            continue; // attachTexture refuses zero dimensions.
        test(`${canvasTarget ? "canvas" : "texture"}: ${claim}`, async () => {
            const { world } = subjects()[0];
            const lower = camera(world, 0);
            const upper = camera(world, 1);
            const display = { width: 64, height: 48 };
            let context: CanvasContext;
            const canvas = Object.assign(Object.create(TestCanvas.prototype), {
                width: 64,
                height: 48,
                clientWidth: 64,
                clientHeight: 48,
                style: { imageRendering: "auto" },
                getContext: () => context,
                getBoundingClientRect: () => display,
            }) as HTMLCanvasElement & { clientWidth: number; clientHeight: number };
            context = new CanvasContext(canvas, 64, 48);
            const previousWindow = Object.getOwnPropertyDescriptor(globalThis, "window");
            if (claim === "CSS pixels ignore pixelRatio")
                Object.defineProperty(globalThis, "window", {
                    configurable: true,
                    value: { devicePixelRatio: 2 },
                });
            const previous = Object.getOwnPropertyDescriptor(globalThis, "HTMLCanvasElement");
            Object.defineProperty(globalThis, "HTMLCanvasElement", {
                configurable: true,
                value: TestCanvas,
            });
            const bind = () => {
                if (canvasTarget) {
                    attachCanvas(world, lower, canvas);
                    attachCanvas(world, upper, canvas);
                } else {
                    const texture = attachTexture(world, lower, display);
                    attachTexture(world, upper, texture);
                }
            };
            const frame = async () => {
                world.gpu.device.pushErrorScope("validation");
                let rgba: Uint8ClampedArray;
                if (canvasTarget) {
                    context.setSize(canvas.width, canvas.height + 1);
                    context.setSize(canvas.width, canvas.height);
                    const captured = captureFrame(canvas);
                    world.step(0);
                    rgba = (await captured).rgba;
                } else {
                    world.step(0);
                    rgba = (await captureTexture(world, lower)).rgba;
                }
                expect(await world.gpu.device.popErrorScope()).toBeNull();
                return rgba;
            };
            const rect = (eid: number, x: number, y: number, width: number, height: number) =>
                world.add(eid, CameraViewport, { x, y, width, height });
            try {
                bind();
                detachView(world, upper);
                world.step(0);
                const baseline = await frame();
                if (canvasTarget) attachCanvas(world, upper, canvas);
                else attachTexture(world, upper, world.resource(Views).get(lower)!.texture!);
                world.resource(fills).add(upper);
                rect(upper, 40, 4, 16, 32);
                if (claim === "CSS pixels ignore pixelRatio") {
                    const corner = await frame();
                    expect([canvas.width, canvas.height]).toEqual([128, 96]);
                    for (let y = 0; y < 96; y++)
                        for (let x = 0; x < 128; x++)
                            expect(pixel(corner, 128, x, y)).toEqual(
                                x >= 80 && x < 112 && y >= 8 && y < 72
                                    ? [0, 255, 0, 255]
                                    : pixel(baseline, 128, x, y),
                            );
                    const view = world.resource(Views).get(upper)!;
                    expect([view.width, view.height]).toEqual([32, 64]);
                } else if (claim === "corner preserves outside bytes") {
                    const corner = await frame();
                    for (let y = 0; y < 48; y++)
                        for (let x = 0; x < 64; x++) {
                            if (x >= 40 && x < 56 && y >= 4 && y < 36)
                                expect(pixel(corner, 64, x, y)).toEqual([0, 255, 0, 255]);
                            else expect(pixel(corner, 64, x, y)).toEqual(pixel(baseline, 64, x, y));
                        }
                } else if (claim === "projection aspect") {
                    await frame();
                    const view = world.resource(Views).get(upper)!;
                    const matrix = world.resource(renderFrameKey).clipFromViews[view.slot]!;
                    expect(matrix[0]! / matrix[5]!).toBeCloseTo(2, 6);
                    expect([view.width, view.height]).toEqual([16, 32]);
                } else if (claim === "disjoint halves") {
                    rect(lower, 0, 0, 32, 48);
                    const storage = world.storage(CameraViewport);
                    storage.x.set(upper, 32);
                    storage.y.set(upper, 0);
                    storage.width.set(upper, 32);
                    storage.height.set(upper, 48);
                    // A red clear fills the bottom half, green geometry the top camera's half.
                    world.storage(Camera).clearColor.set(lower, 0xff0000);
                    const halves = await frame();
                    for (let y = 0; y < 48; y++)
                        for (let x = 0; x < 64; x++)
                            expect(pixel(halves, 64, x, y)).toEqual(
                                x < 32 ? [255, 0, 0, 255] : [0, 255, 0, 255],
                            );
                } else if (claim === "bottom clears whole target") {
                    detachView(world, upper);
                    rect(lower, 8, 8, 16, 16);
                    world.resource(fills).add(lower);
                    const bottom = await frame();
                    expect(pixel(bottom, 64, 0, 0)).toEqual([32, 64, 96, 255]);
                    expect(pixel(bottom, 64, 8, 8)).toEqual([0, 255, 0, 255]);
                    expect(pixel(bottom, 64, 24, 24)).toEqual([32, 64, 96, 255]);
                } else if (claim === "zero target") {
                    display.width = canvas.clientWidth = 0;
                    world.step(0);
                    expect(world.resource(Views).get(upper)!.width).toBe(0);
                    expect(world.resource(Views).get(upper)!.viewport?.width).toBe(0);
                } else {
                    await frame();
                    detachView(world, lower);
                    detachView(world, upper);
                    display.width = canvas.clientWidth =
                        claim === "partial clipping after shrink" ? 48 : 24;
                    display.height = canvas.clientHeight = 20;
                    context.setSize(display.width, display.height);
                    bind();
                    const clipped = await frame();
                    const view = world.resource(Views).get(upper)!;
                    const start = claim === "partial clipping after shrink" ? 40 : 23;
                    expect([view.width, view.height]).toEqual([display.width - start, 16]);
                    const authored = world.storage(CameraViewport);
                    expect([
                        authored.x.get(upper),
                        authored.y.get(upper),
                        authored.width.get(upper),
                        authored.height.get(upper),
                    ]).toEqual([start, 4, display.width - start, 16]);
                    for (let y = 0; y < 20; y++)
                        for (let x = 0; x < display.width; x++)
                            expect(pixel(clipped, display.width, x, y)).toEqual(
                                x >= start && y >= 4 ? [0, 255, 0, 255] : [32, 64, 96, 255],
                            );
                }
            } finally {
                world.resource(fills).clear();
                world.destroy(lower);
                world.destroy(upper);
                world.step(0);
                if (previousWindow) Object.defineProperty(globalThis, "window", previousWindow);
                else Reflect.deleteProperty(globalThis, "window");
                if (previous) Object.defineProperty(globalThis, "HTMLCanvasElement", previous);
                else Reflect.deleteProperty(globalThis, "HTMLCanvasElement");
            }
        });
    }
}
