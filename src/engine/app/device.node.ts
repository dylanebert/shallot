import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { expectError, expectWarning } from "../../../scripts/test-console";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { Body, BodyType, Shape } from "../../core/physics";
import { GlobalTransform } from "../../core/transform";
import { hashPhysics, StandardPhysicsPlugin } from "../../standard/physics";
import "../../standard";

import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { globalTransformTable } from "../../core/rendering";
import { type Plugin, Time, type World } from "../index";
import { createApp, runApp } from "./index";

await setupGlobals();

let live: Awaited<ReturnType<typeof createApp>> | null = null;

afterEach(() => {
    live?.dispose();
    live = null;
});

function replaceGpu(gpu: GPU | undefined): () => void {
    const previous = Object.getOwnPropertyDescriptor(navigator, "gpu");
    Object.defineProperty(navigator, "gpu", { configurable: true, value: gpu });
    return () => {
        if (previous) Object.defineProperty(navigator, "gpu", previous);
        else Reflect.deleteProperty(navigator, "gpu");
    };
}

const AcquisitionPlugin: Plugin = { name: "AcquisitionProbe", gpu: {} };

async function refusalMessage(): Promise<string> {
    return createApp({ defaults: false, plugins: [AcquisitionPlugin] }).then(
        (app) => {
            app.dispose();
            return "build unexpectedly succeeded";
        },
        (error: unknown) => (error instanceof Error ? error.message : String(error)),
    );
}

test("missing navigator.gpu names the Bun runtime and optional webgpu peer fix", async () => {
    const restore = replaceGpu(undefined);
    try {
        const message = await refusalMessage();
        expect(message).toContain("navigator.gpu is missing in Bun");
        expect(message).toContain("optional webgpu peer");
    } finally {
        restore();
    }
});

test("no adapter has its own acquisition refusal", async () => {
    const restore = replaceGpu({ requestAdapter: async () => null } as unknown as GPU);
    try {
        expect(await refusalMessage()).toBe(
            "No WebGPU adapter is available in Bun. Use a supported Dawn backend; on Linux without a GPU, install a software Vulkan driver such as mesa-vulkan-drivers.",
        );
    } finally {
        restore();
    }
});

test("a destroyed device before the first frame does not reach the loading error", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("runApp device-loss test has no adapter");
    const device = await adapter.requestDevice();
    expectWarning(/adapter with no identity/);
    expectError(/GPU device lost destroyed: device was destroyed/);
    let failure: unknown;
    live = await runApp({
        defaults: false,
        device,
        plugins: [],
        loading: {
            show() {},
            update() {},
            error(error) {
                failure = error;
            },
        },
    });

    device.destroy();
    const info = await device.lost;
    expect(info.reason).toBe("destroyed");
    expect(failure).toBeUndefined();
});

test("a supplied device lost after completed frames reaches the loading error and stops the loop", async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("runApp device-loss test has no adapter");
    const nativeDevice = await adapter.requestDevice();
    let resolveLost!: (info: GPUDeviceLostInfo) => void;
    const lost = new Promise<GPUDeviceLostInfo>((resolve) => {
        resolveLost = resolve;
    });
    const device = new Proxy(nativeDevice, {
        get(target, key) {
            if (key === "lost") return lost;
            const value = Reflect.get(target, key, target);
            return typeof value === "function" ? value.bind(target) : value;
        },
    }) as GPUDevice;
    expectWarning(/adapter with no identity/);
    expectError(/GPU device lost unknown: controlled device loss/);
    let failure: unknown;
    let frames = 0;
    let running: Awaited<ReturnType<typeof runApp>> | undefined;
    try {
        running = live = await runApp({
            defaults: false,
            device,
            plugins: [
                {
                    name: "DeviceLossFrame",
                    systems: [
                        {
                            name: "submit-frame",
                            group: "draw",
                            update(world) {
                                frames++;
                                world.frameEncoder();
                            },
                        },
                    ],
                },
            ],
            loading: {
                show() {},
                update() {},
                error(error) {
                    failure = error;
                },
            },
        });

        await waitFor(() => frames >= 2 && running!.world.frameFence !== undefined);
        await running.world.frameFence;
        await Promise.resolve();
        resolveLost({ reason: "unknown", message: "controlled device loss" } as GPUDeviceLostInfo);
        await new Promise<void>((resolve) => setTimeout(resolve, 25));
        expect(failure).toBeInstanceOf(Error);
        expect((failure as Error).message).toContain(
            "GPU device lost unknown: controlled device loss",
        );
        expect((failure as Error).message).toContain("Reloading the page may resolve this issue.");
        const lostFrames = frames;
        await new Promise<void>((resolve) => setTimeout(resolve, 25));
        expect(frames).toBe(lostFrames);

        running.dispose();
        live = null;
        const buffer = nativeDevice.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
        buffer.destroy();
    } finally {
        running?.dispose();
        live = null;
        nativeDevice.destroy();
    }
});

async function waitFor(predicate: () => boolean): Promise<void> {
    const deadline = Date.now() + 5000;
    while (!predicate()) {
        if (Date.now() >= deadline) throw new Error("runApp did not submit completed frames");
        await new Promise<void>((resolve) => setTimeout(resolve, 5));
    }
}

test("device creation failure names both the stage and its cause", async () => {
    const cause = "fixture device creation failed";
    const adapter = {
        features: new Set<GPUFeatureName>(["indirect-first-instance", "rg11b10ufloat-renderable"]),
        limits: { maxStorageBuffersPerShaderStage: 8 },
        requestDevice: async () => {
            throw new Error(cause);
        },
    } as unknown as GPUAdapter;
    const restore = replaceGpu({ requestAdapter: async () => adapter } as unknown as GPU);
    try {
        expect(await refusalMessage()).toBe(`WebGPU device creation failed in Bun: ${cause}`);
    } finally {
        restore();
    }
});

test("adapter request failure keeps its cause distinct from no adapter", async () => {
    const restore = replaceGpu({
        requestAdapter: async () => {
            throw new Error("fixture adapter request failed");
        },
    } as unknown as GPU);
    try {
        expect(await refusalMessage()).toBe(
            "WebGPU adapter request failed in Bun: fixture adapter request failed",
        );
    } finally {
        restore();
    }
});

test("overlapping public builds serialize their setup and then coexist as independent worlds", async () => {
    let release!: () => void;
    let entered!: () => void;
    const gate = new Promise<void>((resolve) => {
        release = resolve;
    });
    const started = new Promise<void>((resolve) => {
        entered = resolve;
    });
    const held = {
        name: "held build",
        initialize: async () => {
            entered();
            await gate;
        },
    };

    const firstPromise = createApp({ defaults: false, plugins: [held] });
    await started;
    const secondPromise = createApp({ defaults: false, plugins: [] });
    release();
    const first = await firstPromise;
    const second = await secondPromise;
    first.world.step(Time.FIXED_DT);
    second.world.step(Time.FIXED_DT);
    first.dispose();
    second.world.step(Time.FIXED_DT);
    second.dispose();
});

test("StandardPhysicsPlugin builds and steps without navigator.gpu", async () => {
    const restore = replaceGpu(undefined);
    try {
        live = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        live.world.step(Time.FIXED_DT);
        expect(live.world.time.fixedTick).toBe(1);
    } finally {
        restore();
    }
});

test("live Physics apps keep their authored component values and solver worlds isolated", async () => {
    const author = (world: World, y: number) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        const body = world.storage(Body);
        body.position.set(eid, 0, y, 0, 0);
        return eid;
    };
    const first = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const firstEid = author(first.world, 2);
    for (let i = 0; i < 8; i++) first.world.step(Time.FIXED_DT);
    if (!first.world.has(firstEid, GlobalTransform))
        throw new Error("first Physics App did not publish its body placement");

    const second = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const secondEid = author(second.world, 20);
    expect(second.world.storage(Body).position.y.get(secondEid)).toBe(20);
    expect(first.world.storage(Body).position.y.get(firstEid)).toBe(2);
    expect(first.world.storage(Body).position.column).not.toBe(
        second.world.storage(Body).position.column,
    );
    expect(() => globalTransformTable(first.world)).toThrow("RenderingPlugin");
    expect(() => globalTransformTable(second.world)).toThrow("RenderingPlugin");

    first.dispose();
    for (let i = 0; i < 8; i++) second.world.step(Time.FIXED_DT);
    expect(second.world.storage(GlobalTransform).translation.y.get(secondEid)).toBeLessThan(20);
    second.dispose();
});

test("two live Physics apps keep sibling bodies and hash unchanged when only one steps", async () => {
    const author = (world: World, y: number) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape);
        const body = world.storage(Body);
        body.position.set(eid, 0, y, 0, 0);
        return eid;
    };
    let first: Awaited<ReturnType<typeof createApp>> | undefined;
    let second: Awaited<ReturnType<typeof createApp>> | undefined;
    try {
        first = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        author(first.world, 2);
        for (let i = 0; i < 8; i++) first.world.step(Time.FIXED_DT);

        second = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
        const secondEid = author(second.world, 20);
        for (let i = 0; i < 8; i++) second.world.step(Time.FIXED_DT);
        const placement = second.world.storage(GlobalTransform);
        const offset = secondEid * 4;
        const bodyBefore = [
            placement.translation.column.slice(offset, offset + 4),
            placement.rotation.column.slice(offset, offset + 4),
            placement.linearVelocity.column.slice(offset, offset + 4),
        ];
        const hashBefore = hashPhysics(second.world);

        for (let i = 0; i < 8; i++) first.world.step(Time.FIXED_DT);
        expect({
            body: [
                placement.translation.column.slice(offset, offset + 4),
                placement.rotation.column.slice(offset, offset + 4),
                placement.linearVelocity.column.slice(offset, offset + 4),
            ],
            hash: hashPhysics(second.world),
        }).toEqual({ body: bodyBefore, hash: hashBefore });
    } finally {
        second?.dispose();
        first?.dispose();
    }
});

test("a failed plugin initialize releases its world and permits a later build", async () => {
    const broken = {
        name: "broken build",
        initialize: () => {
            throw new Error("intentional initialize failure");
        },
    };
    await expect(createApp({ defaults: false, plugins: [broken] })).rejects.toThrow(
        "intentional initialize failure",
    );
    const recovered = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    recovered.world.step(Time.FIXED_DT);
    recovered.dispose();
});

test("disposing a Physics build leaves slab or solver state behind, so a sequential re-entry produces a different fixed-step world", async () => {
    const author = (world: World) => {
        const eid = world.create();
        world.add(eid, Body);
        world.add(eid, Shape);
        world.storage(Body).position.set(eid, 0, 2, 0, 0);
        world.storage(Body).type.set(eid, BodyType.Dynamic);
        return eid;
    };
    const stepAndHash = (world: World): bigint => {
        for (let i = 0; i < 8; i++) world.step(Time.FIXED_DT);
        return hashPhysics(world);
    };

    const first = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    author(first.world);
    const firstHash = stepAndHash(first.world);
    expect(() => first.world.gpu).toThrow("no enabled plugin declares a GPU requirement");
    first.dispose();

    live = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    author(live.world);
    expect(stepAndHash(live.world)).toBe(firstHash);
});
