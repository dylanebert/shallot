import { afterEach, expect } from "bun:test";
import { check } from "../../harness/check";
// Registers DEFAULT_PLUGINS as the build default set that the default-build refusal row reads.
import "../../standard/defaults";
import {
    Body,
    hash as hashPhysics,
    PhysicsPlugin,
    readBody,
    ShapeKind,
} from "../../standard/physics";
import { Slab } from "../../standard/slab";
import { Compute, State, stampAdapter, Time } from "../index";
import { diagnose, load, parse } from "../scene";
import { build } from "./index";

const fallbackAdapter = {
    info: {
        vendor: "google",
        architecture: "swiftshader",
        device: "fallback",
        description: "SwiftShader",
        isFallbackAdapter: true,
    },
} as unknown as GPUAdapter;

let live: Awaited<ReturnType<typeof build>> | null = null;

afterEach(() => {
    live?.dispose();
    live = null;
});

const BUILD_REFUSAL =
    "build refused: another App is building or live in this process; call app.dispose() before building another";

check(
    "public build refuses overlapping in-flight and live Apps, then recovers sequentially",
    {
        claim: "overlapping public builds can mutate process-global registries beneath one another instead of refusing before lifecycle work",
        subject: "src/engine/app/index.ts",
    },
    async () => {
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
    },
);

check(
    "a live CPU Physics App refuses a second build without losing its stepped state",
    {
        claim: "a second public Physics build can reset the first App's slabs and world instead of refusing while the first remains live",
        subject: "src/engine/app/index.ts",
    },
    async () => {
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
    },
);

check(
    "a failed public build releases only its owned lease",
    {
        claim: "a plugin initialize failure can strand the public build lifecycle lease and refuse every later recovery build",
        subject: "src/engine/app/index.ts",
    },
    async () => {
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
    },
);

check(
    "a sequential Physics build re-enters with the same world hash",
    {
        claim: "disposing a CPU Physics build leaves slab or solver state behind, so a sequential re-entry produces a different fixed-step world",
    },
    async () => {
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
    },
);

check(
    "a default build refuses with its device-bound plugins",
    {
        claim: "the default composition hides its GPU requirement behind a generic device error instead of naming the requiring plugins and CPU forms",
    },
    async () => {
        await expect(build({ plugins: [] })).rejects.toThrow(
            /required plugins: Render, Sear, Glaze.*defaults: false.*plugins.*plugins:.*exclude.*Render, Sear, Glaze/s,
        );
    },
);

check(
    "GPU acquisition stamps a fallback adapter",
    {
        claim: "GPU acquisition accepts a fallback adapter without stamping its verdict, so an app can look like it has real hardware",
        subject: ["src/engine/runtime/gpu.ts", "src/engine/app/index.ts"],
    },
    () => {
        stampAdapter(fallbackAdapter);
        expect(Compute.adapter.class).toBe("fallback");
        expect(Compute.adapter.identity).toContain("SwiftShader");
    },
);

check(
    "an external device without its adapter is stamped unidentified",
    {
        claim: "an externally supplied GPU device without its adapter can be mistaken for a real adapter",
        subject: "src/engine/runtime/gpu.ts",
    },
    () => {
        stampAdapter();
        expect(Compute.adapter.class).toBe("unidentified");
        expect(Compute.adapter.identity).toBe("unidentified");
    },
);

check(
    "scene load reports attrs missing from the active registration",
    {
        claim: "a CPU scene silently loses render-only attrs when those plugins are absent, so authors cannot see what the composition dropped",
    },
    () => {
        const state = new State();
        const nodes = parse('<scene><a mesh="name: cube" material="name: default" /></scene>');
        const messages = diagnose(nodes);
        expect(messages.map((diagnostic) => diagnostic.message)).toEqual([
            '"mesh" has no active plugin registration; dropped',
            '"material" has no active plugin registration; dropped',
        ]);
        const result = load(nodes, state);
        expect(result.dropped).toEqual(["mesh", "material"]);
        state.dispose();
    },
);
