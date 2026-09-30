import { afterEach, expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import {
    Body,
    hash as hashPhysics,
    PhysicsPlugin,
    readBody,
    ShapeKind,
} from "../../transitional/physics";
import "../../standard";
import { globalTransformTable, type State, Time } from "../index";
import { build } from "./index";

const peerModule = "bun-webgpu";
const peer = (await import(peerModule)) as Record<string, unknown> & {
    setupGlobals(): Promise<void>;
};
const { setupGlobals } = peer;
await setupGlobals();

let live: Awaited<ReturnType<typeof build>> | null = null;

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
    return build({ defaults: false, plugins: [] }).then(
        (app) => {
            app.dispose();
            return "build unexpectedly succeeded";
        },
        (error: unknown) => (error instanceof Error ? error.message : String(error)),
    );
}

test("missing navigator.gpu names the Bun runtime and optional bun-webgpu peer fix", async () => {
    const restore = replaceGpu(undefined);
    try {
        const message = await refusalMessage();
        expect(message).toContain("navigator.gpu is missing in Bun");
        expect(message).toContain("optional bun-webgpu peer");
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
        features: new Set<GPUFeatureName>([
            "indirect-first-instance",
            "bgra8unorm-storage",
            "rg11b10ufloat-renderable",
        ]),
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

    const firstPromise = build({ defaults: false, plugins: [held] });
    await started;
    const secondPromise = build({ defaults: false, plugins: [] });
    release();
    const first = await firstPromise;
    const second = await secondPromise;
    first.state.step(Time.FIXED_DT);
    second.state.step(Time.FIXED_DT);
    first.dispose();
    second.state.step(Time.FIXED_DT);
    second.dispose();
});

test("live Physics apps keep their authored component values and solver worlds isolated", async () => {
    const author = (state: State, y: number) => {
        const eid = state.create();
        state.add(eid, Body);
        const body = state.of(Body);
        body.shape.set(eid, ShapeKind.Box);
        body.pos.set(eid, 0, y, 0, 0);
        body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
        body.mass.set(eid, 1);
        return eid;
    };
    const first = await build({ defaults: false, plugins: [PhysicsPlugin] });
    const firstEid = author(first.state, 2);
    for (let i = 0; i < 8; i++) first.state.step(Time.FIXED_DT);
    const firstBefore = readBody(first.state, firstEid);
    if (!firstBefore) throw new Error("first Physics App did not produce a live body");

    const second = await build({ defaults: false, plugins: [PhysicsPlugin] });
    const secondEid = author(second.state, 20);
    expect(second.state.of(Body).pos.y.get(secondEid)).toBe(20);
    expect(first.state.of(Body).pos.y.get(firstEid)).toBe(2);
    expect(first.state.of(Body).pos.column).not.toBe(second.state.of(Body).pos.column);
    expect(globalTransformTable(first.state).buffer).not.toBe(
        globalTransformTable(second.state).buffer,
    );
    expect(globalTransformTable(first.state).eidToRowBuffer).toBeDefined();
    expect(globalTransformTable(first.state).eidToRowBuffer).not.toBe(
        globalTransformTable(second.state).eidToRowBuffer,
    );

    first.dispose();
    for (let i = 0; i < 8; i++) second.state.step(Time.FIXED_DT);
    const secondAfter = readBody(second.state, secondEid);
    expect(secondAfter?.pos[1]).toBeLessThan(20);
    second.dispose();
});

test("two live Physics apps keep sibling bodies and hash unchanged when only one steps", async () => {
    const author = (state: State, y: number) => {
        const eid = state.create();
        state.add(eid, Body);
        const body = state.of(Body);
        body.shape.set(eid, ShapeKind.Box);
        body.pos.set(eid, 0, y, 0, 0);
        body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
        body.mass.set(eid, 1);
        return eid;
    };
    let first: Awaited<ReturnType<typeof build>> | undefined;
    let second: Awaited<ReturnType<typeof build>> | undefined;
    try {
        first = await build({ defaults: false, plugins: [PhysicsPlugin] });
        author(first.state, 2);
        for (let i = 0; i < 8; i++) first.state.step(Time.FIXED_DT);

        second = await build({ defaults: false, plugins: [PhysicsPlugin] });
        const secondEid = author(second.state, 20);
        for (let i = 0; i < 8; i++) second.state.step(Time.FIXED_DT);
        const bodyBefore = readBody(second.state, secondEid);
        if (!bodyBefore) throw new Error("second Physics App did not produce a live body");
        const hashBefore = hashPhysics(second.state);

        for (let i = 0; i < 8; i++) first.state.step(Time.FIXED_DT);
        expect({
            body: readBody(second.state, secondEid),
            hash: hashPhysics(second.state),
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
    await expect(build({ defaults: false, plugins: [broken] })).rejects.toThrow(
        "intentional initialize failure",
    );
    const recovered = await build({ defaults: false, plugins: [PhysicsPlugin] });
    recovered.state.step(Time.FIXED_DT);
    recovered.dispose();
});

test("disposing a Physics build leaves slab or solver state behind, so a sequential re-entry produces a different fixed-step world", async () => {
    const author = (state: State) => {
        const eid = state.create();
        state.add(eid, Body);
        Body.shape.set(eid, ShapeKind.Box);
        Body.pos.set(eid, 0, 2, 0, 0);
        Body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
        Body.mass.set(eid, 1);
        return eid;
    };
    const stepAndHash = (state: State): bigint => {
        for (let i = 0; i < 8; i++) state.step(Time.FIXED_DT);
        return hashPhysics(state);
    };

    const first = await build({ defaults: false, plugins: [PhysicsPlugin] });
    author(first.state);
    const firstHash = stepAndHash(first.state);
    first.dispose();
    expect(first.state.gpu.buffers.size).toBe(0);
    expect(first.state.gpu.typed.size).toBe(0);

    live = await build({ defaults: false, plugins: [PhysicsPlugin] });
    author(live.state);
    expect(stepAndHash(live.state)).toBe(firstHash);
});
