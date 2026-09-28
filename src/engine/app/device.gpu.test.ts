import { afterEach, expect, test } from "bun:test";
import {
    Body,
    hash as hashPhysics,
    PhysicsPlugin,
    readBody,
    ShapeKind,
} from "../../transitional/physics";
import { Slab } from "../../transitional/slab";
import { type State, Time } from "../index";
import { build } from "./index";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

let live: Awaited<ReturnType<typeof build>> | null = null;

afterEach(() => {
    live?.dispose();
    live = null;
});

const BUILD_REFUSAL =
    "build refused: another App is building or live in this process; call app.dispose() before building another";

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

test("overlapping public builds can mutate process-global registries beneath one another instead of refusing before lifecycle work", async () => {
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
    await expect(build({ defaults: false, plugins: [] })).rejects.toThrow(BUILD_REFUSAL);
    await expect(build({ defaults: false, plugins: [] })).rejects.toThrow(BUILD_REFUSAL);
    release();
    const first = await firstPromise;
    first.dispose();

    const recovered = await build({ defaults: false, plugins: [] });
    recovered.state.step(Time.FIXED_DT);
    recovered.dispose();
}, 20_000);

test("a second public Physics build can reset the first App's slabs and world instead of refusing while the first remains live", async () => {
    const author = (state: State) => {
        const eid = state.create();
        state.add(eid, Body);
        Body.shape.set(eid, ShapeKind.Box);
        Body.pos.set(eid, 0, 2, 0, 0);
        Body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0);
        Body.mass.set(eid, 1);
        return eid;
    };
    const first = await build({ defaults: false, plugins: [PhysicsPlugin] });
    const eid = author(first.state);
    for (let i = 0; i < 8; i++) first.state.step(Time.FIXED_DT);
    const before = readBody(first.state, eid);
    if (!before) throw new Error("first Physics App did not produce a live body");
    await expect(build({ defaults: false, plugins: [PhysicsPlugin] })).rejects.toThrow(
        BUILD_REFUSAL,
    );
    first.state.step(Time.FIXED_DT);
    const after = readBody(first.state, eid);
    expect(after).not.toBeNull();
    expect(after?.pos[1]).toBeLessThan(before.pos[1]);
    first.dispose();

    const recovered = await build({ defaults: false, plugins: [PhysicsPlugin] });
    author(recovered.state);
    for (let i = 0; i < 8; i++) recovered.state.step(Time.FIXED_DT);
    expect(hashPhysics(recovered.state)).toBeDefined();
    recovered.dispose();
}, 20_000);

test("a plugin initialize failure can strand the public build lifecycle lease and refuse every later recovery build", async () => {
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
}, 20_000);

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
    expect((Slab as unknown as { _all: unknown[] })._all).toHaveLength(0);

    live = await build({ defaults: false, plugins: [PhysicsPlugin] });
    author(live.state);
    expect(stepAndHash(live.state)).toBe(firstHash);
}, 20_000);
