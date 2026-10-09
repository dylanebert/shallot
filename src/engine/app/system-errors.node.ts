import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { RenderingPlugin } from "../../core/rendering";
import { createApp, type Plugin, runApp } from "./index";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("runApp logs a system error once, keeps healthy frames running and resumes after swapSystem", async () => {
    const cause = new Error("broken");
    let attempts = 0;
    let frames = 0;
    let resumed = 0;
    const broken = {
        name: "spawn",
        update() {
            attempts++;
            throw cause;
        },
    };
    const healthy = {
        after: [broken],
        update() {
            frames++;
        },
    };
    const logged = spyOn(console, "error").mockImplementation(() => {});
    const app = await runApp({
        defaults: false,
        plugins: [{ name: "Game", systems: [broken, healthy] }],
    });
    try {
        const waitFor = async (ready: () => boolean) => {
            const deadline = performance.now() + CEILING.node / 2;
            while (!ready()) {
                if (performance.now() > deadline) throw new Error("frame loop did not progress");
                await Bun.sleep(1);
            }
        };
        await waitFor(() => frames >= 3);
        expect(attempts).toBe(1);
        expect(logged).toHaveBeenCalledTimes(1);
        expect(logged.mock.calls[0]).toEqual([
            'System "Game/spawn" threw and is paused until its next reload:',
            cause,
        ]);
        const before = frames;
        app.world.swapSystem(broken, {
            update() {
                resumed++;
            },
        });
        await waitFor(() => frames >= before + 3);
        expect(resumed).toBeGreaterThanOrEqual(3);
        expect(logged).toHaveBeenCalledTimes(1);
    } finally {
        app.dispose();
        logged.mockRestore();
    }
});

test("log-and-pause skips a failed draw system but still submits later draw work", async () => {
    const cause = new Error("broken draw");
    let attempts = 0;
    let healthyFrames = 0;
    let submissions = 0;
    let descriptor: PropertyDescriptor | undefined;
    let queue: GPUQueue | undefined;
    let buffer: GPUBuffer | undefined;
    const broken = {
        name: "broken-draw",
        group: "draw" as const,
        update(world: import("../ecs/world").World) {
            attempts++;
            world.frameEncoder()!.clearBuffer(buffer!);
            throw cause;
        },
    };
    const healthy = {
        name: "healthy-draw",
        group: "draw" as const,
        after: [broken],
        update(world: import("../ecs/world").World) {
            healthyFrames++;
            world.frameEncoder()!.clearBuffer(buffer!);
        },
    };
    const logged = spyOn(console, "error").mockImplementation(() => {});
    const plugin: Plugin = {
        name: "PausedDraw",
        gpu: {},
        initialize(world) {
            queue = world.gpu.device.queue;
            descriptor = Object.getOwnPropertyDescriptor(queue, "submit");
            const submit = queue.submit.bind(queue);
            Object.defineProperty(queue, "submit", {
                configurable: true,
                value: (...args: Parameters<GPUQueue["submit"]>) => {
                    submissions++;
                    return submit(...args);
                },
            });
            buffer = world.gpu.device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST,
            });
            world.own(buffer);
        },
        systems: [broken, healthy],
    };
    const app = await createApp({ defaults: false, plugins: [plugin] });
    app.world.logAndPauseSystemErrors();
    try {
        expect(() => app.world.step(0)).not.toThrow();
        expect(attempts).toBe(1);
        expect(healthyFrames).toBe(1);
        expect(submissions).toBe(1);
        expect(logged).toHaveBeenCalledTimes(1);
        await app.world.frameFence;
    } finally {
        app.dispose();
        if (queue) {
            if (descriptor) Object.defineProperty(queue, "submit", descriptor);
            else Reflect.deleteProperty(queue, "submit");
        }
        logged.mockRestore();
    }
});

test("runApp waits on the engine frame fence instead of issuing a second completion fence", async () => {
    let syncs = 0;
    let reads = 0;
    let sameFence = true;
    let latest: Promise<void> | undefined;
    const observed: Plugin = {
        name: "FrameFenceObserver",
        initialize(world) {
            const sync = world.gpu.sync;
            const buffer = world.gpu.device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST,
            });
            world.own(buffer);
            world.addSystem({
                group: "draw",
                update(world) {
                    world.frameEncoder()!.clearBuffer(buffer);
                },
            });
            world.gpu.sync = () => {
                latest = sync();
                syncs++;
                return latest;
            };
            const descriptor = Object.getOwnPropertyDescriptor(
                Object.getPrototypeOf(world),
                "frameFence",
            );
            if (!descriptor?.get) throw new Error("World.frameFence getter is missing");
            Object.defineProperty(world, "frameFence", {
                configurable: true,
                get() {
                    reads++;
                    const fence = descriptor.get!.call(world) as Promise<void> | undefined;
                    if (fence !== latest) sameFence = false;
                    return fence;
                },
            });
        },
    };
    const app = await runApp({ defaults: false, plugins: [RenderingPlugin, observed] });
    try {
        const deadline = performance.now() + CEILING.node / 2;
        while (app.world.gpu.frame < 3) {
            if (performance.now() > deadline)
                throw new Error("render loop did not submit three frames");
            await Bun.sleep(1);
        }
        expect(syncs).toBeGreaterThanOrEqual(3);
        expect(reads).toBeGreaterThanOrEqual(3);
        expect(sameFence).toBe(true);
    } finally {
        app.dispose();
    }
});
