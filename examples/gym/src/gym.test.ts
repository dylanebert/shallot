import { expect, test } from "bun:test";
import type { State } from "@dylanebert/shallot";
import { Profile } from "@dylanebert/shallot/extras";
import type { Verdict as WireVerdict } from "@dylanebert/shallot/harness";
import { getScenario, installHarness, resolveNoRender, type Scenario } from "./gym";
import "./scenarios/render";
import { SCENARIO_BUDGETS } from "./scenarios/budgets";

const installedNoRender = (mode: string): boolean => {
    const scenario = getScenario("render");
    if (!scenario) throw new Error("render scenario was not registered");
    const savedWindow = globalThis.window;
    const fakeWindow = {} as typeof window;
    try {
        globalThis.window = fakeWindow;
        installHarness(scenario, {} as State, () => true, { mode });
        return fakeWindow.__harness?.noRender === true;
    } finally {
        globalThis.window = savedWindow;
    }
};

test("the registered render scenario forwards only its two reference-probe rows", () => {
    expect(installedNoRender("spec")).toBe(true);
    expect(installedNoRender("cascade-boundary")).toBe(true);
    expect(installedNoRender("cull")).toBe(false);
    expect(installedNoRender("cascade")).toBe(false);
});

test("static noRender declarations keep their diagnostic-scenario behavior", () => {
    expect(resolveNoRender(true, {})).toBe(true);
    expect(resolveNoRender(undefined, {})).toBe(false);
});

// `run()` inlines the pipeline-count extraction (`shallot-perf-gates` stage 3b review finding 1):
// `[...Profile.compile.keys()].filter(k => Profile.compiledPipelines.has(k)).length`, never
// `Profile.compile.size` — a `precompile` scope-only span (the `sear-typed-variants` shape) lands in
// `compile` without ever registering in `compiledPipelines`, so `.size` overcounts. Seeds `Profile`'s
// module-singleton Maps directly (no live GPU device needed — they're plain data) with two real
// pipeline-produced entries and one scope-only entry, then reads the `budget:pipelines` check `run()`
// actually reports back — not a re-derivation of the same filter, a hand-counted expected value against
// a fixture scenario carrying the real `accel` budget entry (`SCENARIO_BUDGETS.accel.pipelines === 44`).
test("run() reports the filtered pipeline count, not Profile.compile.size", async () => {
    const compile = Profile.compile as Map<string, number>;
    const pipelines = Profile.compiledPipelines as Set<string>;
    const savedCompile = new Map(compile);
    const savedPipelines = new Set(pipelines);
    compile.clear();
    pipelines.clear();
    try {
        // two real pipeline-produced compiles...
        compile.set("sear:color", 1.2);
        pipelines.add("sear:color");
        compile.set("sear:pointshadow", 0.8);
        pipelines.add("sear:pointshadow");
        // ...and one scope-only span (the `sear-typed-variants` shape): present in `compile`, absent
        // from `compiledPipelines` — the exact case that would inflate `Profile.compile.size` to 3.
        compile.set("sear-typed-variants", 0.16);

        // a fixture scenario, not the real "accel" scenario object — same name as a real
        // `SCENARIO_BUDGETS` key so `assertBudget` emits a check, but `params: []` so
        // `isDefaultParams` is vacuously true and no live GPU/Mirror assert path ever runs.
        const scenario: Scenario = {
            name: "accel",
            params: [],
            build: () => Promise.reject(new Error("not exercised — run() never calls build")),
        };
        const savedWindow = globalThis.window;
        const fakeWindow = {} as unknown as typeof window;
        (fakeWindow as unknown as { __benchmark: unknown }).__benchmark = {
            ready: true,
            measure: async () => ({}),
        };
        globalThis.window = fakeWindow;
        try {
            installHarness(scenario, {} as State, () => true, {});
            const verdict = (await fakeWindow.__harness?.run?.()) as WireVerdict;
            const check = verdict.checks?.find((c) => c.name === "budget:pipelines");
            expect(check?.detail).toBe("measured 2, budget 44");
        } finally {
            globalThis.window = savedWindow;
        }
    } finally {
        compile.clear();
        for (const [k, v] of savedCompile) compile.set(k, v);
        pipelines.clear();
        for (const k of savedPipelines) pipelines.add(k);
    }
});

// `run()` computes `budget:bytes`'s measured quantity as `Profile.bufferBytes + Profile.textureBytes -
// Profile.lazyBytes`, never the raw sum (`shallot-perf-gates` stage 4e): a lazily-grown pool's live bytes
// are timing-dependent, so a lazy pool growing (bufferBytes AND lazyBytes moving together) must NOT red
// the gate, while a bogus allocation landing in a GATED label (bufferBytes moving alone) must still red
// it. Seeds `Profile`'s module-singleton scalar fields directly (no live GPU device needed) against the
// real `accel` budget entry.
test("run()'s budget:bytes excludes Profile.lazyBytes from the measured total", async () => {
    const golden = SCENARIO_BUDGETS.accel.gpuBytes as number;
    const mutable = Profile as unknown as {
        bufferBytes: number;
        textureBytes: number;
        lazyBytes: number;
    };
    const saved = {
        bufferBytes: mutable.bufferBytes,
        textureBytes: mutable.textureBytes,
        lazyBytes: mutable.lazyBytes,
    };
    const scenario: Scenario = {
        name: "accel",
        params: [],
        build: () => Promise.reject(new Error("not exercised — run() never calls build")),
    };
    const savedWindow = globalThis.window;
    const fakeWindow = {} as unknown as typeof window;
    (fakeWindow as unknown as { __benchmark: unknown }).__benchmark = {
        ready: true,
        measure: async () => ({}),
    };
    globalThis.window = fakeWindow;

    const run = async () => {
        installHarness(scenario, {} as State, () => true, {});
        const verdict = (await fakeWindow.__harness?.run?.()) as WireVerdict;
        return verdict.checks?.find((c) => c.name === "budget:bytes");
    };

    try {
        // exact match at the gated total, no lazy bytes live right now
        mutable.bufferBytes = golden;
        mutable.textureBytes = 0;
        mutable.lazyBytes = 0;
        let check = await run();
        expect(check?.ok).toBe(true);
        expect(check?.detail).toBe(`measured ${golden}, budget ${golden}`);

        // a lazy pool growing by 4096 B (bufferBytes AND lazyBytes both move by the same amount) — the
        // excluded-bytes case the check must not red on.
        mutable.bufferBytes = golden + 4096;
        mutable.lazyBytes = 4096;
        check = await run();
        expect(check?.ok).toBe(true);
        expect(check?.detail).toBe(`measured ${golden}, budget ${golden}`);

        // a bogus allocation landing in a GATED (non-lazy) label — bufferBytes moves 8 B further with no
        // matching lazyBytes growth — must still red.
        mutable.bufferBytes = golden + 4096 + 8;
        mutable.lazyBytes = 4096;
        check = await run();
        expect(check?.ok).toBe(false);
        expect(check?.detail).toBe(`measured ${golden + 8}, budget ${golden}`);
    } finally {
        mutable.bufferBytes = saved.bufferBytes;
        mutable.textureBytes = saved.textureBytes;
        mutable.lazyBytes = saved.lazyBytes;
        globalThis.window = savedWindow;
    }
});
