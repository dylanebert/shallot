import { afterEach, expect } from "bun:test";
import { check } from "../../harness/check";
import { DEFAULT_PLUGINS } from "../../standard/defaults";
import { PhysicsPlugin, physicsWorld } from "../../standard/physics";
import { Compute, State, stampAdapter, Time } from "../index";
import { diagnose, load, parse } from "../scene";
import { build, deviceTier } from "./index";

const GPU_PLUGIN = { name: "GPU test", device: "required" as const };
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

check(
    "a CPU Physics build steps without navigator.gpu",
    {
        claim: "the build unconditionally requests WebGPU, so CPU physics cannot build and step in Bun",
    },
    async () => {
        expect(globalThis.navigator?.gpu).toBeUndefined();
        live = await build({ defaults: false, plugins: [PhysicsPlugin] });
        expect(Compute.device).toBeUndefined();
        expect(physicsWorld(live.state)).not.toBeNull();
        live.state.step(Time.FIXED_DT);
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
    "deviceTier classifies the standard default set",
    {
        claim: "the standard plugin set has no pure composition data for deciding whether build should acquire a device",
    },
    () => {
        expect(deviceTier(DEFAULT_PLUGINS)).toEqual({
            tier: "gpu",
            required: ["Render", "Sear", "Glaze"],
            optional: ["Slab", "Transforms", "Part"],
        });
    },
);

check(
    "GPU acquisition stamps and surfaces a fallback adapter once",
    {
        claim: "GPU acquisition accepts a fallback adapter without stamping or surfacing its verdict, so an app can look like it has real hardware",
        subject: ["src/engine/runtime/gpu.ts", "src/engine/app/index.ts"],
    },
    () => {
        const warnings: unknown[][] = [];
        const previousWarn = console.warn;
        const notices: import("../runtime/adapter").AdapterVerdict[] = [];
        console.warn = (...args: unknown[]) => warnings.push(args);
        try {
            stampAdapter(fallbackAdapter, (verdict) => notices.push(verdict));
            expect(Compute.adapter.class).toBe("fallback");
            expect(Compute.adapter.identity).toContain("SwiftShader");
            expect(warnings).toHaveLength(1);
            expect(warnings[0]?.[0]).toContain("fallback adapter");
            expect(notices).toEqual([Compute.adapter]);
        } finally {
            console.warn = previousWarn;
        }
    },
);

check(
    "an external device without its adapter is stamped unidentified",
    {
        claim: "an externally supplied GPU device without its adapter can be mistaken for a real adapter",
        subject: "src/engine/runtime/gpu.ts",
    },
    () => {
        const warnings: unknown[][] = [];
        const previousWarn = console.warn;
        console.warn = (...args: unknown[]) => warnings.push(args);
        try {
            stampAdapter();
            expect(Compute.adapter.class).toBe("unidentified");
            expect(Compute.adapter.identity).toBe("unidentified");
            expect(warnings).toHaveLength(1);
        } finally {
            console.warn = previousWarn;
        }
    },
);

check(
    "a real GPU seat stamps its real adapter on a GPU-tier build",
    {
        claim: "a GPU-tier build can pass on a fallback or unidentified adapter while claiming the gpu seat",
        size: "integration",
        requires: ["gpu"],
        host: "mac",
        subject: ["src/engine/runtime/gpu.ts", "src/engine/app/index.ts"],
    },
    async () => {
        const peer = (await new Function("return import('bun-webgpu')")()) as {
            setupGlobals(): Promise<void>;
        };
        await peer.setupGlobals();
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("S2 GPU seat refused: no adapter");
        const device = await adapter.requestDevice();
        live = await build({ defaults: false, plugins: [GPU_PLUGIN], device, adapter });
        expect(Compute.adapter.class).toBe("real");
        expect(Compute.adapter.identity.length).toBeGreaterThan(0);
        return { ok: true, hardware: Compute.adapter.identity };
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
