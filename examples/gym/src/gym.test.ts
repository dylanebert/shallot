import { expect, test } from "bun:test";
import type { State } from "@dylanebert/shallot";
import { Profile } from "@dylanebert/shallot/extras";
import type { Verdict as WireVerdict } from "@dylanebert/shallot/harness";
import { getScenario, installHarness, resolveNoRender, type Scenario } from "./gym";
import "./scenarios/render";

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
// a fixture scenario carrying the real `render` budget entry (`SCENARIO_BUDGETS.render.pipelines === 29`).
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

        // a fixture scenario, not the real "render" scenario object — same name as a real
        // `SCENARIO_BUDGETS` key so `assertBudget` emits a check, but `params: []` so
        // `isDefaultParams` is vacuously true and no live GPU/Mirror assert path ever runs.
        const scenario: Scenario = {
            name: "render",
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
            expect(check?.detail).toBe("measured 2, budget 29");
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
