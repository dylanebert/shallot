import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { Body, BodyType, ShapeKind } from "../../core/physics";
import { hashPhysics, readBody, StandardPhysicsPlugin } from "../../standard/physics";
import "../../standard";

import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { globalTransformTable } from "../../core/transform";
import { Time, type World } from "../index";
import { createApp } from "./index";

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

async function refusalMessage(): Promise<string> {
    return createApp({ defaults: false, plugins: [] }).then(
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
        expect(await refusalMessage()).toBe("No WebGPU adapter is available in Bun.");
    } finally {
        restore();
    }
});

test("device creation failure names both the stage and its cause", async () => {
    const cause = "fixture device creation failed";
    const adapter = {
        features: new Set<GPUFeatureName>(["indirect-first-instance", "rg11b10ufloat-renderable"]),
        limits: { maxStorageBuffersPerShaderStage: 10 },
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

test("live Physics apps keep their authored component values and solver worlds isolated", async () => {
    const author = (world: World, y: number) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        const body = world.storage(Body);
        body.shape.set(eid, ShapeKind.Box);
        body.position.set(eid, 0, y, 0, 0);
        body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
        body.mass.set(eid, 1);
        return eid;
    };
    const first = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const firstEid = author(first.world, 2);
    for (let i = 0; i < 8; i++) first.world.step(Time.FIXED_DT);
    const firstBefore = readBody(first.world, firstEid);
    if (!firstBefore) throw new Error("first Physics App did not produce a live body");

    const second = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const secondEid = author(second.world, 20);
    expect(second.world.storage(Body).position.y.get(secondEid)).toBe(20);
    expect(first.world.storage(Body).position.y.get(firstEid)).toBe(2);
    expect(first.world.storage(Body).position.column).not.toBe(
        second.world.storage(Body).position.column,
    );
    expect(globalTransformTable(first.world).buffer).not.toBe(
        globalTransformTable(second.world).buffer,
    );
    expect(globalTransformTable(first.world).eidToRowBuffer).toBeDefined();
    expect(globalTransformTable(first.world).eidToRowBuffer).not.toBe(
        globalTransformTable(second.world).eidToRowBuffer,
    );

    first.dispose();
    for (let i = 0; i < 8; i++) second.world.step(Time.FIXED_DT);
    const secondAfter = readBody(second.world, secondEid);
    expect(secondAfter?.position[1]).toBeLessThan(20);
    second.dispose();
});

test("two live Physics apps keep sibling bodies and hash unchanged when only one steps", async () => {
    const author = (world: World, y: number) => {
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic });
        const body = world.storage(Body);
        body.shape.set(eid, ShapeKind.Box);
        body.position.set(eid, 0, y, 0, 0);
        body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
        body.mass.set(eid, 1);
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
        const bodyBefore = readBody(second.world, secondEid);
        if (!bodyBefore) throw new Error("second Physics App did not produce a live body");
        const hashBefore = hashPhysics(second.world);

        for (let i = 0; i < 8; i++) first.world.step(Time.FIXED_DT);
        expect({
            body: readBody(second.world, secondEid),
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
        world.storage(Body).shape.set(eid, ShapeKind.Box);
        world.storage(Body).position.set(eid, 0, 2, 0, 0);
        world.storage(Body).halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
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
    first.dispose();
    expect(first.world.gpu.buffers.size).toBe(0);
    expect(first.world.gpu.typed.size).toBe(0);

    live = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    author(live.world);
    expect(stepAndHash(live.world)).toBe(firstHash);
});
