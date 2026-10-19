import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createTestGpuApp, disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { expectError, expectWarning } from "../../../scripts/test-console";
import type { World } from "../../engine";
import { CanvasContext } from "../../engine/app/canvas.fixture";
import { Transform } from "../transform";
import {
    attachCanvas,
    attachTexture,
    Camera,
    CameraMode,
    captureFrame,
    captureTexture,
    detachView,
    RenderingPlugin,
    Tonemapping,
    TonemappingMethod,
} from "./index";
import { PointsPlugin } from "./points.fixture";

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

const plugins = [RenderingPlugin, PointsPlugin];
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins }]);

// Both cameras look down -Z at the fixture's points with one orthographic scale, offset half a pixel so
// each point covers one pixel's every sample, and a point lands on the same pixel row for both. The upper camera sits one unit right, so two of its points share pixels
// with the lower's, and twice as far back, so its depth there is less than the lower's: a depth buffer
// shared across the stack would reject them.
// World units per pixel of a 64-pixel-tall view at orthographic size 4.
const PIXEL = 1 / 8;

function lowerCamera(world: World, antialias: number): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: [-PIXEL / 2, -PIXEL / 2, 5, 0] });
    world.add(eid, Camera, {
        mode: CameraMode.Orthographic,
        size: 4,
        clearColor: 0x204060,
        antialias,
    });
    world.add(eid, Tonemapping, { method: TonemappingMethod.None });
    return eid;
}

// Default presentation and no effects; its clearColor is unused above the bottom of a target.
function upperCamera(world: World, antialias: number): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: [1 - PIXEL / 2, -PIXEL / 2, 10, 0] });
    world.add(eid, Camera, {
        mode: CameraMode.Orthographic,
        size: 4,
        clearColor: 0xff0000,
        antialias,
        order: 1,
    });
    return eid;
}

/** Pixel indices a lone camera drew on: farther from its corner clear than deband dither reaches. */
function drawn(frame: Uint8ClampedArray): Set<number> {
    const pixels = new Set<number>();
    for (let i = 0; i < frame.length; i += 4) {
        for (let channel = 0; channel < 3; channel++) {
            if (Math.abs(frame[i + channel] - frame[channel]) > 8) {
                pixels.add(i / 4);
                break;
            }
        }
    }
    return pixels;
}

/** The composite of `above` over `below`: above's bytes where it drew, below's elsewhere. */
function over(below: Uint8ClampedArray, above: Uint8ClampedArray): Uint8ClampedArray {
    const out = new Uint8ClampedArray(below);
    for (const pixel of drawn(above)) out.set(above.subarray(pixel * 4, pixel * 4 + 4), pixel * 4);
    return out;
}

function mismatches(actual: Uint8ClampedArray, expected: Uint8ClampedArray, width: number) {
    const out: string[] = [];
    for (let i = 0; i < expected.length && out.length < 8; i += 4) {
        if (
            actual[i] !== expected[i] ||
            actual[i + 1] !== expected[i + 1] ||
            actual[i + 2] !== expected[i + 2] ||
            actual[i + 3] !== expected[i + 3]
        )
            out.push(
                `${(i / 4) % width},${Math.floor(i / 4 / width)}: ${actual.subarray(i, i + 4)} != ${expected.subarray(i, i + 4)}`,
            );
    }
    return out;
}

async function frame(world: World, eid: number): Promise<Uint8ClampedArray> {
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    const error = await world.gpu.device.popErrorScope();
    expect(error?.message ?? null).toBeNull();
    return (await captureTexture(world, eid)).rgba;
}

/** Each camera alone on its own texture, as the references a stack composes. */
async function alone(world: World, eid: number, size: number): Promise<Uint8ClampedArray> {
    attachTexture(world, eid, { width: size, height: size });
    const rgba = await frame(world, eid);
    detachView(world, eid);
    return rgba;
}

test("cameras sharing a texture compose bottom to top by order, each with its own clear, depth and sample count", async () => {
    const { world } = subjects()[0];
    const size = 64;
    for (const [lowerAa, upperAa] of [
        [1, 1],
        [0, 1],
        [1, 0],
    ] as const) {
        const lower = lowerCamera(world, lowerAa);
        const upper = upperCamera(world, upperAa);
        const lowerAlone = await alone(world, lower, size);
        const upperAlone = await alone(world, upper, size);
        const lowerDrawn = drawn(lowerAlone);
        const upperDrawn = drawn(upperAlone);
        const shared = [...upperDrawn].filter((pixel) => lowerDrawn.has(pixel));
        expect(shared.length).toBeGreaterThan(0);
        expect(lowerDrawn.size).toBeGreaterThan(shared.length);

        const target = attachTexture(world, lower, { width: size, height: size });
        expect(attachTexture(world, upper, target)).toBe(target);
        const stacked = await frame(world, lower);
        expect(mismatches(stacked, over(lowerAlone, upperAlone), size)).toEqual([]);
        expect((await captureTexture(world, upper)).rgba).toEqual(stacked);

        world.storage(Camera).order.set(lower, 2);
        expect(mismatches(await frame(world, lower), over(upperAlone, lowerAlone), size)).toEqual(
            [],
        );

        world.storage(Camera).order.set(lower, 1);
        const [below, above] = lower < upper ? [lower, upper] : [upper, lower];
        expectWarning(
            new RegExp(
                `^shallot: cameras ${below} and ${above} share Camera.order 1 on one target`,
            ),
        );
        const tie = lower < upper ? stacked : over(upperAlone, lowerAlone);
        expect(mismatches(await frame(world, lower), tie, size)).toEqual([]);
        world.storage(Camera).order.set(lower, 0);

        detachView(world, upper);
        expect(mismatches(await frame(world, lower), lowerAlone, size)).toEqual([]);
        attachTexture(world, upper, target);
        expect(mismatches(await frame(world, lower), stacked, size)).toEqual([]);
        world.destroy(upper);
        expect(mismatches(await frame(world, lower), lowerAlone, size)).toEqual([]);

        const replacement = upperCamera(world, upperAa);
        attachTexture(world, replacement, target);
        expect(mismatches(await frame(world, lower), stacked, size)).toEqual([]);
        detachView(world, lower);
        expect(mismatches(await frame(world, replacement), upperAlone, size)).toEqual([]);
        world.destroy(lower);
        world.destroy(replacement);
    }
});

class TestCanvas {}

test("a camera without a target joins the first canvas's stack, and the stack follows a canvas resize", async () => {
    const { world } = subjects()[0];
    let context: CanvasContext;
    const display = { width: 640, height: 360 };
    const canvas = Object.assign(Object.create(TestCanvas.prototype), {
        width: display.width,
        height: display.height,
        clientWidth: display.width,
        clientHeight: display.height,
        style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => display,
    }) as HTMLCanvasElement & { clientWidth: number; clientHeight: number };
    context = new CanvasContext(canvas, display.width, display.height);
    const globals = { HTMLCanvasElement: TestCanvas, document: undefined as unknown };
    globals.document = { querySelector: () => canvas, querySelectorAll: () => [canvas] };
    const previous = new Map<string, PropertyDescriptor | undefined>();
    for (const [name, value] of Object.entries(globals)) {
        previous.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
        Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
    }
    try {
        const lower = lowerCamera(world, 0);
        const upper = upperCamera(world, 0);
        attachCanvas(world, lower, canvas);
        world.step(0);
        world.step(0);

        // References at the size the canvas takes next, single-sample: at this scale a point covers
        // part of a pixel, which a lone camera blends with its clear and a stacked one with the target.
        const width = 1280;
        const height = 720;
        const lowerReference = lowerCamera(world, 0);
        const upperReference = upperCamera(world, 0);
        attachTexture(world, lowerReference, { width, height });
        attachTexture(world, upperReference, { width, height });
        display.width = canvas.clientWidth = width;
        display.height = canvas.clientHeight = height;
        context.setSize(width, height);
        world.step(0);
        world.step(0);
        const lowerAlone = (await captureTexture(world, lowerReference)).rgba;
        const upperAlone = (await captureTexture(world, upperReference)).rgba;
        expect(drawn(upperAlone).size).toBeGreaterThan(0);

        // A fresh canvas texture for the captured frame, as a browser hands each frame.
        context.setSize(width, height + 1);
        context.setSize(width, height);
        const capture = captureFrame(canvas);
        world.step(0);
        const { rgba } = await capture;
        expect(mismatches(rgba, over(lowerAlone, upperAlone), width)).toEqual([]);
        for (const eid of [lower, upper, lowerReference, upperReference]) world.destroy(eid);
        world.step(0);
    } finally {
        for (const [name, descriptor] of previous) {
            if (descriptor) Object.defineProperty(globalThis, name, descriptor);
            else Reflect.deleteProperty(globalThis, name);
        }
    }
});

test("a disposed stack releases every texture its world created", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Dawn adapter unavailable");
    const device = await adapter.requestDevice();
    const live = new Set<GPUTexture>();
    const counted = new Proxy(device, {
        get(target, key) {
            if (key === "createTexture") {
                return (descriptor: GPUTextureDescriptor) => {
                    const texture = target.createTexture(descriptor);
                    live.add(texture);
                    const destroy = texture.destroy.bind(texture);
                    texture.destroy = () => {
                        if (live.delete(texture)) destroy();
                    };
                    return texture;
                };
            }
            const value = Reflect.get(target, key, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    const app = await createTestGpuApp({ defaults: false, plugins, device: counted });
    try {
        const { world } = app;
        const lower = lowerCamera(world, 1);
        const upper = upperCamera(world, 0);
        const target = attachTexture(world, lower, { width: 16, height: 16 });
        attachTexture(world, upper, target);
        await frame(world, lower);
        const labels = [...live].map((texture) => texture.label);
        expect(labels).toContain(`shallot-stacked-${upper}`);
    } finally {
        await disposeGpuApps([app]);
    }
    expect([...live].map((texture) => texture.label)).toEqual([]);
    expectError(/^GPU device lost destroyed: device was destroyed$/);
    device.destroy();
});
