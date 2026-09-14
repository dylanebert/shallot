import { afterEach, expect } from "bun:test";
import { check } from "../../harness/check";
import { DEFAULT_PLUGINS } from "../../standard/defaults";
import { PhysicsPlugin, physicsWorld } from "../../standard/physics";
import { Compute, State, Time } from "../index";
import { diagnose, load, parse } from "../scene";
import { build, deviceTier } from "./index";

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
