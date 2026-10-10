import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { World } from "../../engine";
import type { WorldGpu } from "../../engine/runtime";
import { captureFrame } from "./capture";
import { RenderContext } from "./render";
import { attachCanvas, detachCanvas } from "./view";

setDefaultTimeout(CEILING.node);

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((yes) => (resolve = yes));
    return { promise, resolve };
}

function testCanvas(): HTMLCanvasElement {
    const context = {
        configure() {},
        unconfigure() {},
    } as unknown as GPUCanvasContext;
    return {
        width: 1280,
        height: 720,
        style: { imageRendering: "auto" },
        getContext: () => context,
        getBoundingClientRect: () => ({ width: 1280, height: 720 }),
        toDataURL: () => "",
    } as unknown as HTMLCanvasElement;
}

function installCanvasGlobals(canvas: HTMLCanvasElement): () => void {
    const previous = new Map<string, PropertyDescriptor | undefined>();
    const replace = (name: string, value: unknown) => {
        previous.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
        Object.defineProperty(globalThis, name, {
            configurable: true,
            writable: true,
            value,
        });
    };
    class TestResizeObserver {
        observe() {}
        disconnect() {}
    }
    replace("navigator", { gpu: { getPreferredCanvasFormat: () => "rgba8unorm" } });
    replace("document", { querySelectorAll: () => [canvas] });
    replace("window", { devicePixelRatio: 1 });
    replace("ResizeObserver", TestResizeObserver);
    replace("GPUTextureUsage", { RENDER_ATTACHMENT: 1, COPY_SRC: 2 });
    replace("GPUBufferUsage", { COPY_DST: 1, MAP_READ: 2 });
    replace("GPUMapMode", { READ: 1 });
    return () => {
        for (const [name, descriptor] of previous) {
            if (descriptor) Object.defineProperty(globalThis, name, descriptor);
            else Reflect.deleteProperty(globalThis, name);
        }
    };
}

function testWorld() {
    const lost = deferred<GPUDeviceLostInfo>();
    const buffers: { destroyed: boolean }[] = [];
    const device = {
        lost: lost.promise,
        createBuffer: (_descriptor: GPUBufferDescriptor) => {
            const buffer = {
                destroyed: false,
                mapAsync: async () => {},
                getMappedRange: () => new ArrayBuffer(0),
                unmap() {},
                destroy() {
                    buffer.destroyed = true;
                },
            };
            buffers.push(buffer);
            return buffer;
        },
    } as unknown as GPUDevice;
    const world = new World();
    world.attachGpu({
        device,
        adapter: { class: "unidentified", identity: "capture lifecycle mock", reason: "test" },
        root: {} as WorldGpu["root"],
        frame: 0,
        pending: () => 0,
        sync: async () => {},
        fences: { issued: 0, completed: 0 },
        buffers: new Map(),
        textures: new Map(),
        samplers: new Map(),
        typed: new Map(),
    } as WorldGpu);
    world.resource(RenderContext).format = "rgba8unorm";
    return { world, device, lost, buffers };
}

function bindCanvas(world: World, canvas: HTMLCanvasElement): number {
    const eid = world.create();
    attachCanvas(eid, canvas, world);
    return eid;
}

function bindCanvasRef(
    world: World,
    registry: FinalizationRegistry<number>,
    tag: number,
): { eid: number; ref: WeakRef<HTMLCanvasElement> } {
    const canvas = testCanvas();
    registry.register(canvas, tag);
    const ref = new WeakRef(canvas);
    const eid = bindCanvas(world, canvas);
    return { eid, ref };
}

function bindAndDetachCanvas(
    world: World,
    registry: FinalizationRegistry<number>,
    tag: number,
): WeakRef<HTMLCanvasElement> {
    const { eid, ref } = bindCanvasRef(world, registry, tag);
    detachCanvas(world, eid);
    return ref;
}

async function collectUntil(done: () => boolean): Promise<void> {
    const deadline = Date.now() + 1_000;
    while (Date.now() < deadline && !done()) {
        // A deref keeps its target alive for this job; yield before the next collection.
        await new Promise((resolve) => setTimeout(resolve, 0));
        Bun.gc(true);
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
}

test("a pending canvas frame capture rejects on GPU device loss with its label", async () => {
    const canvas = testCanvas();
    const restore = installCanvasGlobals(canvas);
    const fixture = testWorld();
    try {
        bindCanvas(fixture.world, canvas);
        const pending = captureFrame(canvas).then(
            () => new Error("capture unexpectedly resolved"),
            (error: unknown) => error,
        );
        fixture.lost.resolve({
            message: "mock device loss",
            reason: "unknown",
        } as unknown as GPUDeviceLostInfo);

        const failure = await pending;
        expect(failure).toBeInstanceOf(Error);
        expect((failure as Error).message).toBe(
            "canvas frame capture: GPU device lost: mock device loss",
        );
        expect(fixture.buffers).toHaveLength(1);
        expect(fixture.buffers[0]!.destroyed).toBe(true);
    } finally {
        fixture.world.dispose();
        restore();
    }
});

test("a pending canvas frame capture rejects on world disposal with its label", async () => {
    const canvas = testCanvas();
    const restore = installCanvasGlobals(canvas);
    const fixture = testWorld();
    try {
        bindCanvas(fixture.world, canvas);
        const pending = captureFrame(canvas).then(
            () => new Error("capture unexpectedly resolved"),
            (error: unknown) => error,
        );
        fixture.world.dispose();

        const failure = await pending;
        expect(failure).toBeInstanceOf(Error);
        expect((failure as Error).message).toBe(
            "canvas frame capture: captureFrame refused: world disposed before presentation",
        );
        expect(fixture.buffers[0]!.destroyed).toBe(true);
    } finally {
        fixture.world.dispose();
        restore();
    }
});

test("detaching repeatedly does not retain old canvas bindings", async () => {
    const canvas = testCanvas();
    const restore = installCanvasGlobals(canvas);
    const { world } = testWorld();
    try {
        const finalized = new Set<number>();
        const registry = new FinalizationRegistry<number>((tag) => finalized.add(tag));
        const { eid: retainedEid, ref: retainedRef } = bindCanvasRef(world, registry, -1);
        const detached = Array.from({ length: 8 }, (_, tag) =>
            bindAndDetachCanvas(world, registry, tag),
        );
        await collectUntil(
            () => detached.every((ref) => ref.deref() === undefined) && finalized.size === 8,
        );

        expect(detached.every((ref) => ref.deref() === undefined)).toBe(true);
        expect(finalized).toEqual(new Set([0, 1, 2, 3, 4, 5, 6, 7]));
        expect(retainedRef.deref()).toBeDefined();

        detachCanvas(world, retainedEid);
        await collectUntil(() => retainedRef.deref() === undefined && finalized.has(-1));
        expect(retainedRef.deref()).toBeUndefined();
        expect(finalized.has(-1)).toBe(true);
    } finally {
        world.dispose();
        restore();
    }
});
