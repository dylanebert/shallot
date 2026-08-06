import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { SCENARIO_GATES } from "../../../examples/gym/src/scenarios/timeouts";
import {
    benchTimeout,
    type ForMatch,
    formatForResolution,
    formatRoster,
    forUnmatchedReason,
    groupByTimeout,
    normalizeForPath,
    partitionSweep,
    resolveFor,
} from "../../../scripts/bench";
import {
    batchPass,
    bootArm,
    buildUrl,
    type Checkpoint,
    coerceVerdict,
    type FrameSample,
    failureArtifacts,
    fitMemory,
    formatTimings,
    gpuLogChecks,
    gridDiff,
    harnessPass,
    hasStructure,
    LEAK_BYTES_PER_SEC,
    type MemorySample,
    parseVerifyArgs,
    report,
    reportBatch,
    resolveBatchQueries,
    settlePass,
    spansFromCheckpoints,
    stepWait,
    structured,
    type WaitState,
    withTimeout,
} from "./verify";

const REPO_ROOT = resolve(import.meta.dir, "../../..");

describe("parseVerifyArgs", () => {
    test("defaults: dir '.', dev boot, 60s budget", () => {
        const a = parseVerifyArgs([]);
        expect(a.dir).toBe(".");
        expect(a.dist).toBe(false);
        expect(a.json).toBe(false);
        expect(a.timeoutMs).toBe(60_000);
        expect(a.query).toEqual([]);
    });

    test("positional dir + flags", () => {
        const a = parseVerifyArgs([
            "examples/x",
            "--dist",
            "--json",
            "--screenshot",
            "out.png",
            "--port",
            "5300",
            "--timeout",
            "9000",
        ]);
        expect(a.dir).toBe("examples/x");
        expect(a.dist).toBe(true);
        expect(a.json).toBe(true);
        expect(a.screenshot).toBe("out.png");
        expect(a.port).toBe(5300);
        expect(a.timeoutMs).toBe(9000);
    });

    test("--query repeats; --k=v form accepted", () => {
        const a = parseVerifyArgs(["--query", "scene=a", "--query=mode=fall", "--port=4000"]);
        expect(a.query).toEqual(["scene=a", "mode=fall"]);
        expect(a.port).toBe(4000);
    });

    test("--connect takes a ws endpoint (space and = forms); default undefined", () => {
        expect(parseVerifyArgs([]).connect).toBeUndefined();
        expect(parseVerifyArgs(["--connect", "ws://host:9/abc"]).connect).toBe("ws://host:9/abc");
        expect(parseVerifyArgs(["--connect=ws://host:9/abc"]).connect).toBe("ws://host:9/abc");
    });

    test("unknown option throws", () => {
        expect(() => parseVerifyArgs(["--nope"])).toThrow("unknown option: --nope");
    });

    test("a second positional throws", () => {
        expect(() => parseVerifyArgs(["a", "b"])).toThrow("unexpected argument: b");
    });

    test("non-numeric / non-positive --port and --timeout are rejected, not NaN'd", () => {
        expect(() => parseVerifyArgs(["--port", "abc"])).toThrow('invalid --port value "abc"');
        expect(() => parseVerifyArgs(["--port=0"])).toThrow('invalid --port value "0"');
        expect(() => parseVerifyArgs(["--timeout", "xyz"])).toThrow(
            'invalid --timeout value "xyz"',
        );
        expect(() => parseVerifyArgs(["--timeout=-5"])).toThrow('invalid --timeout value "-5"');
    });

    test("--memory and --alloc are separate flags, default false", () => {
        const a = parseVerifyArgs([]);
        expect(a.memory).toBe(false);
        expect(a.alloc).toBe(false);
        expect(parseVerifyArgs(["--memory"]).memory).toBe(true);
        expect(parseVerifyArgs(["--alloc"]).alloc).toBe(true);
    });

    test("--memory and --alloc together are a parse error (their samplers conflict)", () => {
        expect(() => parseVerifyArgs(["--memory", "--alloc"])).toThrow("mutually exclusive");
    });

    test("--leak defaults to 0 (off) and parses a positive rate (with --memory)", () => {
        expect(parseVerifyArgs([]).leak).toBe(0);
        expect(parseVerifyArgs(["--leak", "122880", "--memory"]).leak).toBe(122880);
        expect(parseVerifyArgs(["--leak=122880", "--memory"]).leak).toBe(122880);
        expect(() => parseVerifyArgs(["--leak", "0"])).toThrow("expected a positive number");
    });

    test("--leak without --memory is a parse error (nothing samples the injected allocation)", () => {
        expect(() => parseVerifyArgs(["--leak", "122880"])).toThrow("--leak requires --memory");
        expect(() => parseVerifyArgs(["--leak=122880"])).toThrow("--leak requires --memory");
    });

    test("--timings defaults off, flips on", () => {
        expect(parseVerifyArgs([]).timings).toBe(false);
        expect(parseVerifyArgs(["--timings"]).timings).toBe(true);
    });

    test("--run defaults to empty (single-run path); repeats, both space and = forms", () => {
        expect(parseVerifyArgs([]).run).toEqual([]);
        const a = parseVerifyArgs(["--run", "scenario=outline", "--run=scenario=sprite"]);
        expect(a.run).toEqual(["scenario=outline", "scenario=sprite"]);
    });
});

describe("resolveBatchQueries — the pure batch arg surface", () => {
    test("each run's spec layers on the shared query, `&`-split into k=v segments", () => {
        expect(resolveBatchQueries(["seed=1"], ["scenario=outline", "scenario=sprite"])).toEqual([
            ["seed=1", "scenario=outline"],
            ["seed=1", "scenario=sprite"],
        ]);
    });

    test("a multi-param run spec splits on &", () => {
        expect(resolveBatchQueries([], ["scenario=outline&count=8"])).toEqual([
            ["scenario=outline", "count=8"],
        ]);
    });

    test("no shared query and an empty run spec yields an empty per-run array", () => {
        expect(resolveBatchQueries([], [""])).toEqual([[]]);
    });

    test("no runs yields no query arrays", () => {
        expect(resolveBatchQueries(["seed=1"], [])).toEqual([]);
    });
});

describe("batchPass — the batch result aggregation", () => {
    test("passes only when every run passed", () => {
        expect(batchPass([{ pass: true }, { pass: true }])).toBe(true);
        expect(batchPass([{ pass: true }, { pass: false }])).toBe(false);
    });

    test("empty input is a fail, not a vacuous pass", () => {
        expect(batchPass([])).toBe(false);
    });
});

describe("spansFromCheckpoints — the pure phase-timing math", () => {
    test("each span is the gap before its checkpoint, attributed to that checkpoint's name", () => {
        const checkpoints: Checkpoint[] = [
            { name: "start", t: 1_000 },
            { name: "server boot", t: 1_090 },
            { name: "first page load", t: 1_860 },
            { name: "harness ready", t: 2_200 },
        ];
        expect(spansFromCheckpoints(checkpoints)).toEqual([
            { name: "server boot", ms: 90 },
            { name: "first page load", ms: 770 },
            { name: "harness ready", ms: 340 },
        ]);
    });

    test("one checkpoint (no elapsed phase) yields no spans", () => {
        expect(spansFromCheckpoints([{ name: "start", t: 1_000 }])).toEqual([]);
    });

    test("empty input yields no spans", () => {
        expect(spansFromCheckpoints([])).toEqual([]);
    });
});

describe("formatTimings — the pure rendering of spans", () => {
    test("pads names to the longest and prints milliseconds", () => {
        const text = formatTimings([
            { name: "server boot", ms: 90 },
            { name: "run", ms: 1_234 },
        ]);
        expect(text).toBe("  server boot  90ms\n  run          1234ms");
    });

    test("empty spans render nothing", () => {
        expect(formatTimings([])).toBe("");
    });
});

describe("bootArm", () => {
    test("a shallot manifest/.scene project wins even when an index.html is also present", () => {
        expect(bootArm(true, false)).toBe("project");
        expect(bootArm(true, true)).toBe("project");
    });

    test("no manifest but an index.html → ejected vite app", () => {
        expect(bootArm(false, true)).toBe("ejected");
    });

    test("neither shape → none (the actionable setup error)", () => {
        expect(bootArm(false, false)).toBe("none");
    });
});

describe("buildUrl", () => {
    test("no query → base unchanged", () => {
        expect(buildUrl("http://localhost:5173/", [])).toBe("http://localhost:5173/");
    });

    test("query params append", () => {
        const u = new URL(buildUrl("http://localhost:5173/", ["scenario=fall", "count=8"]));
        expect(u.searchParams.get("scenario")).toBe("fall");
        expect(u.searchParams.get("count")).toBe("8");
    });

    test("a bare key (no =) becomes an empty-valued param", () => {
        const u = new URL(buildUrl("http://localhost:5173/", ["debug"]));
        expect(u.searchParams.has("debug")).toBe(true);
    });
});

describe("settle-wait primitives", () => {
    test("structured: center far from corner is structure, near is blank", () => {
        expect(structured([200, 200, 200], [10, 10, 10])).toBe(true);
        expect(structured([12, 12, 12], [10, 10, 10])).toBe(false); // spread 6 < 12
    });

    test("gridDiff: identical grids are 0, a shifted grid is the mean channel delta", () => {
        const g = [1, 2, 3, 4, 5, 6];
        expect(gridDiff(g, g)).toBe(0);
        expect(gridDiff([0, 0, 0], [10, 20, 30])).toBeCloseTo((10 + 20 + 30) / 3, 6);
    });

    test("hasStructure: a centrally-framed scene renders; a flat clear or a null (no canvas) does not", () => {
        // centre lifted off the cleared corner — a rendered scene
        expect(hasStructure({ grid: [], center: [200, 200, 200], corner: [10, 10, 10] })).toBe(
            true,
        );
        // centre reads the clear color (a model that never rendered — the gltf symptom) → not rendered
        expect(hasStructure({ grid: [], center: [10, 10, 10], corner: [10, 10, 10] })).toBe(false);
        // no capturable canvas → not rendered
        expect(hasStructure(null)).toBe(false);
    });
});

describe("verdict interpretation", () => {
    test("harnessPass: ok verdict + rendered + no errors passes; a blank canvas or error fails", () => {
        expect(harnessPass({ ok: true }, true, 0)).toBe(true);
        expect(harnessPass({ ok: true }, true, 1)).toBe(false);
        expect(harnessPass({ ok: false }, true, 0)).toBe(false);
        // an ok verdict over a canvas that rendered nothing is a FAIL — the pixel-honest gate.
        expect(harnessPass({ ok: true }, false, 0)).toBe(false);
        // a declared no-render opt-out passes the pixel gate on the verdict alone (renders nothing by
        // design), but the verdict + error checks still hold.
        expect(harnessPass({ ok: true }, "opt-out", 0)).toBe(true);
        expect(harnessPass({ ok: false }, "opt-out", 0)).toBe(false);
        expect(harnessPass({ ok: true }, "opt-out", 1)).toBe(false);
    });

    test("settlePass needs booted + rendered + zero errors", () => {
        expect(settlePass(true, true, 0)).toBe(true);
        expect(settlePass(true, false, 0)).toBe(false);
        expect(settlePass(true, true, 2)).toBe(false);
    });
});

describe("coerceVerdict", () => {
    test("a verdict-shaped object passes through, extra fields intact", () => {
        const v = coerceVerdict({ ok: true, checks: [], fps: 60 });
        expect(v.ok).toBe(true);
        expect(v.fps).toBe(60);
    });

    test("undefined / null / bare values / non-boolean ok are a clean FAIL", () => {
        for (const bad of [undefined, null, 42, "passed", [1], {}, { ok: "yes" }]) {
            const v = coerceVerdict(bad);
            expect(v.ok).toBe(false);
            expect(v.checks?.[0]?.detail).toContain("returned no verdict");
        }
    });
});

describe("withTimeout", () => {
    test("a resolving promise passes its value through", async () => {
        expect(await withTimeout(Promise.resolve(7), 1000, "x")).toBe(7);
    });

    test("a hung promise rejects with the bound named", async () => {
        const hung = new Promise(() => {});
        expect(withTimeout(hung, 5, "run()")).rejects.toThrow("run() did not resolve within 5ms");
    });

    test("a rejecting promise keeps its own error", async () => {
        expect(withTimeout(Promise.reject(new Error("boom")), 1000, "x")).rejects.toThrow("boom");
    });
});

describe("fitMemory — the leak slope", () => {
    const line = (rate: number, n: number): MemorySample[] =>
        Array.from({ length: n }, (_, i) => ({ t: i * 1000, heap: 1_000_000 + rate * i }));

    test("fewer than three samples → null (two can't drop the cold-start reading)", () => {
        expect(fitMemory([], 0, 0)).toBeNull();
        expect(fitMemory([{ t: 0, heap: 1 }], 0, 0)).toBeNull();
        expect(fitMemory(line(100_000, 2), 0, 0)).toBeNull();
    });

    test("a steep upward slope (100 KB/s) is a leak; endpoints are the fitted range", () => {
        // 4 samples, so the cold-start first is dropped before fitting; fit runs over t=1..3s.
        const m = fitMemory(line(100_000, 4), 3, 12.5);
        expect(m).not.toBeNull();
        expect(m?.growthPerSecond).toBeCloseTo(100_000, 3);
        expect(m?.leak).toBe(true);
        expect(m?.start).toBe(1_100_000); // first fitted sample (cold-start dropped)
        expect(m?.end).toBe(1_300_000);
        expect(m?.gcCount).toBe(3);
        expect(m?.gcPauseMs).toBe(12.5);
    });

    test("a gentle slope (10 KB/s) below the threshold is not a leak", () => {
        const m = fitMemory(line(10_000, 4), 0, 0);
        expect(m?.growthPerSecond).toBeCloseTo(10_000, 3);
        expect(m?.leak).toBe(false);
    });

    test("the leak boundary is strict at LEAK_BYTES_PER_SEC", () => {
        // exactly at threshold is not a leak; just over is. Derives the test from the constant, no
        // magic number — pins the re-derived 1024 B/frame @ 60fps = 61_440 B/s boundary.
        const at = fitMemory(line(LEAK_BYTES_PER_SEC, 4), 0, 0);
        expect(at?.growthPerSecond).toBeCloseTo(LEAK_BYTES_PER_SEC, 3);
        expect(at?.leak).toBe(false);
        const over = fitMemory(line(LEAK_BYTES_PER_SEC + 1, 4), 0, 0);
        expect(over?.leak).toBe(true);
    });
});

describe("stepWait — the unified wait decision", () => {
    const structuredSample = (grid: number[]): FrameSample => ({
        grid,
        center: [200, 200, 200],
        corner: [10, 10, 10],
    });

    test("a defined harness always wins — even mid-settle (the no-downgrade rule)", () => {
        const st: WaitState = { booted: false, prev: null };
        expect(stepWait(st, false, structuredSample([9, 9, 9]))).toBe("continue");
        // the next sample would have settled, but the harness appeared: harness path, not settle
        expect(stepWait(st, true, structuredSample([9, 9, 9]))).toBe("harness");
        expect(stepWait(st, true, null)).toBe("harness");
    });

    test("two consecutive matching structured frames settle", () => {
        const st: WaitState = { booted: false, prev: null };
        expect(stepWait(st, false, structuredSample([9, 9, 9]))).toBe("continue");
        expect(stepWait(st, false, structuredSample([9, 9, 9]))).toBe("settled");
        expect(st.booted).toBe(true);
    });

    test("null samples (no canvas yet) keep polling without claiming boot", () => {
        const st: WaitState = { booted: false, prev: null };
        expect(stepWait(st, false, null)).toBe("continue");
        expect(st.booted).toBe(false);
    });

    test("unstructured (blank) frames boot but never settle", () => {
        const st: WaitState = { booted: false, prev: null };
        const blank: FrameSample = {
            grid: [10, 10, 10],
            center: [10, 10, 10],
            corner: [10, 10, 10],
        };
        expect(stepWait(st, false, blank)).toBe("continue");
        expect(stepWait(st, false, blank)).toBe("continue");
        expect(st.booted).toBe(true);
        expect(st.prev).toBeNull();
    });
});

// the scenario-declared bench timeout: `bun bench` drives a scenario that declared a budget under it, keeps
// the tight 60s default for every scenario that didn't, and lets an explicit --timeout override either.
describe("benchTimeout", () => {
    test("a declared scenario drives under its budget, above the 60s default", () => {
        expect(benchTimeout("stress")).toBe(SCENARIO_GATES.stress.timeoutMs);
        // the whole point: stress legitimately needs more than the default hang detector.
        expect(benchTimeout("stress")).toBeGreaterThan(60_000);
    });

    test("an undeclared scenario stays undefined so verify's 60s default holds", () => {
        expect(benchTimeout("render")).toBeUndefined();
        expect(benchTimeout("pile")).toBeUndefined();
    });

    test("an explicit --timeout overrides a declared budget (operator override)", () => {
        expect(benchTimeout("stress", 90_000)).toBe(90_000);
    });

    test("an explicit --timeout is honored on an undeclared scenario too", () => {
        expect(benchTimeout("render", 5_000)).toBe(5_000);
    });
});

// stage 4's driver surface (`bun bench --list` / `--for` / `--sweep`) — pure logic over plain data, no
// page boot required. `--list`'s real-registration seam (`registeredScenarios`) is exercised by hand
// (`bun bench --list`), not here — it needs the WebGPU polyfill up before the scenario barrel imports.
describe("formatRoster", () => {
    test("sorts and one-per-line", () => {
        expect(formatRoster(["outline", "accel", "backend"])).toBe("accel\nbackend\noutline");
    });

    test("empty roster is an empty string, not a crash", () => {
        expect(formatRoster([])).toBe("");
    });
});

describe("resolveFor", () => {
    const table = {
        outline: { covers: ["packages/shallot/src/extras/outline/**/*.ts"] },
        sprite: { covers: ["packages/shallot/src/extras/sprite/**/*.ts"] },
        backend: { covers: ["packages/shallot/src/standard/avbd/**/*.ts"] },
        "stacking-arch": {},
    };

    test("a path matches every scenario whose covers glob resolves it", () => {
        const [m] = resolveFor(["packages/shallot/src/extras/outline/pass.ts"], table);
        expect(m.scenarios).toEqual(["outline"]);
    });

    test("a path under two scenarios' globs matches both, sorted", () => {
        const twoWay = {
            ...table,
            accel: { covers: ["packages/shallot/src/standard/avbd/**/*.ts"] },
        };
        const [m] = resolveFor(["packages/shallot/src/standard/avbd/collide.ts"], twoWay);
        expect(m.scenarios).toEqual(["accel", "backend"]);
    });

    // red-proven: before the `covers` filter existed this returned every table key regardless of the
    // path, which would have silently unioned a tumble path (below) into the whole roster.
    test("a path no glob covers resolves to an empty scenario list, not the whole table", () => {
        const [m] = resolveFor(["packages/shallot/src/engine/ecs/state.ts"], table);
        expect(m.scenarios).toEqual([]);
    });

    test("resolves one entry per input path, in order", () => {
        const matches = resolveFor(
            [
                "packages/shallot/src/extras/outline/pass.ts",
                "packages/shallot/src/extras/sprite/x.ts",
            ],
            table,
        );
        expect(matches.map((m: ForMatch) => m.path)).toEqual([
            "packages/shallot/src/extras/outline/pass.ts",
            "packages/shallot/src/extras/sprite/x.ts",
        ]);
    });
});

describe("forUnmatchedReason", () => {
    test("a tumble path points at tumble.md's own standing gates, not this table", () => {
        expect(forUnmatchedReason("packages/shallot/src/standard/tumble/body.ts")).toContain(
            "tumble.md",
        );
    });

    test("any other unmatched path names SCENARIO_GATES as what it's outside of", () => {
        expect(forUnmatchedReason("packages/shallot/src/engine/ecs/state.ts")).toContain(
            "SCENARIO_GATES",
        );
    });
});

describe("formatForResolution", () => {
    test("a matched path prints its scenarios; an unmatched one prints why, not an empty roster", () => {
        const out = formatForResolution([
            { path: "a.ts", scenarios: ["outline", "sprite"] },
            { path: "packages/shallot/src/standard/tumble/b.ts", scenarios: [] },
        ]);
        const lines = out.split("\n");
        expect(lines[0]).toBe("a.ts → outline, sprite");
        expect(lines[1]).toContain("tumble.md");
        expect(lines[1]).not.toBe("packages/shallot/src/standard/tumble/b.ts → ");
    });
});

describe("partitionSweep", () => {
    test("a declared-isolate scenario runs alone, never folded into the shared batch (fixture)", () => {
        const table = { a: {}, b: { isolate: true }, c: {} };
        const { batch, isolate } = partitionSweep(["a", "b", "c"], table);
        expect(batch).toEqual(["a", "c"]);
        expect(isolate).toEqual(["b"]);
    });

    // real data, as a declared registry asserted BOTH directions (`coding.md`): the literal below is the
    // independent half. Deriving the expectation with the same `g.isolate` filter the production code
    // implements would re-derive the rule under test and discriminate almost nothing — it would pass for
    // any table, including one where somebody dropped `isolate` off `stress`. Naming the three scenarios
    // means adding or removing an `isolate` declaration reds here and forces the decision to be made
    // deliberately: each one costs a whole process per sweep, forever, and is justified only by a
    // perf-threshold check that sweep contention would make untrustworthy (the `stress` finding).
    const IsolateScenarios = ["character", "render", "stress"];

    test("exactly the declared isolate scenarios carry isolate in the real table", () => {
        const inTable = Object.entries(SCENARIO_GATES)
            .filter(([, g]) => g.isolate)
            .map(([name]) => name)
            .sort();
        expect(inTable).toEqual(IsolateScenarios);
    });

    test("every real isolate-declared scenario partitions out of the batch", () => {
        const { batch, isolate } = partitionSweep(Object.keys(SCENARIO_GATES), SCENARIO_GATES);
        expect(isolate.sort()).toEqual(IsolateScenarios);
        for (const name of IsolateScenarios) expect(batch).not.toContain(name);
    });

    test("empty input partitions to two empty arrays", () => {
        expect(partitionSweep([], {})).toEqual({ batch: [], isolate: [] });
    });
});

describe("groupByTimeout", () => {
    test("scenarios sharing a resolved budget stay in one batch", () => {
        expect(groupByTimeout(["outline", "sprite", "text"])).toEqual([
            { timeoutMs: undefined, names: ["outline", "sprite", "text"] },
        ]);
    });

    test("an operator --timeout reaches the batch instead of being dropped", () => {
        expect(groupByTimeout(["outline", "sprite"], 120_000)).toEqual([
            { timeoutMs: 120_000, names: ["outline", "sprite"] },
        ]);
    });

    // the latent half: `--timeout` is process-global, so a batch carrying two distinct budgets has to
    // split. Today only `stress` declares one and it isolates, so this never fires on the real table —
    // it is what stops a future non-isolate `timeoutMs` entry from silently running under the 60s default.
    test("distinct budgets split into separate batches, in first-appearance order", () => {
        expect(groupByTimeout(["outline", "stress", "sprite"])).toEqual([
            { timeoutMs: undefined, names: ["outline", "sprite"] },
            { timeoutMs: 180_000, names: ["stress"] },
        ]);
    });
});

describe("normalizeForPath", () => {
    const root = "/repo";

    test("an absolute path inside the repo becomes the repo-relative spelling the globs use", () => {
        expect(
            normalizeForPath("/repo/packages/shallot/src/standard/sear/pipelines.ts", root, root),
        ).toBe("packages/shallot/src/standard/sear/pipelines.ts");
    });

    test("a path relative to a cwd deeper than the root resolves against that cwd", () => {
        expect(
            normalizeForPath("src/standard/sear/pipelines.ts", "/repo/packages/shallot", root),
        ).toBe("packages/shallot/src/standard/sear/pipelines.ts");
    });

    test("an already-normalized path is unchanged", () => {
        const p = "packages/shallot/src/extras/outline/pass.ts";
        expect(normalizeForPath(p, root, root)).toBe(p);
    });

    test("a path outside the repo passes through, so the unmatched reason stays honest", () => {
        expect(normalizeForPath("/elsewhere/foo.ts", root, root)).toBe("/elsewhere/foo.ts");
    });
});

describe("gpuLogChecks", () => {
    test("nothing logged means no checks — verdicts stay unchanged", () => {
        expect(gpuLogChecks(null)).toEqual([]);
        expect(gpuLogChecks({ lines: [], errors: [] })).toEqual([]);
    });

    test("lines surface as one informational check, truncated past 8", () => {
        const lines = Array.from({ length: 11 }, (_, i) => `line ${i}`);
        const [check, ...rest] = gpuLogChecks({ lines, errors: [] });
        expect(rest).toEqual([]);
        expect(check.name).toBe("gpu.log");
        expect(check.ok).toBe(true);
        expect(check.detail).toContain("line 0");
        expect(check.detail).toContain("(+3 more)");
        expect(check.detail).not.toContain("line 8");
    });

    test("a GPU console.error is a failing check — the shader-side assert channel", () => {
        const checks = gpuLogChecks({ lines: ["ok", "bad normal"], errors: ["bad normal"] });
        expect(checks.map((c) => [c.name, c.ok])).toEqual([
            ["gpu.log", true],
            ["gpu.error", false],
        ]);
        expect(checks[1].detail).toBe("bad normal");
    });
});

describe("shader artifacts", () => {
    const capture = {
        artifacts: [
            {
                label: "forward",
                stage: "vertex+fragment",
                source: "@vertex fn vs() {}",
                hash: "0123456789abcdef",
                messages: [
                    {
                        type: "error",
                        message: "invalid shader",
                        lineNum: 1,
                        linePos: 2,
                        offset: 3,
                        length: 4,
                    },
                ],
            },
        ],
    };

    test("verify retains exact shader records only for failures", () => {
        expect(failureArtifacts(true, capture)).toBeUndefined();
        expect(failureArtifacts(false, capture)).toEqual(capture.artifacts);
        expect(failureArtifacts(false, { artifacts: [] })).toBeUndefined();
    });

    test("human and JSON failure output carry the exact artifact", () => {
        const result = {
            project: "/tmp/gpu-diagnostic",
            mode: "dev",
            url: "http://localhost/gpu-diagnostic",
            hardware: "test-gpu",
            harness: true,
            booted: true,
            rendered: "opt-out",
            errors: [],
            pass: false,
            artifacts: capture.artifacts,
        };
        const lines: string[] = [];
        const original = console.log;
        console.log = (...args: unknown[]) => lines.push(args.join(" "));
        try {
            report(result as never, false);
            const human = lines.join("\n");
            expect(human).toContain("forward [vertex+fragment] 0123456789abcdef");
            expect(human).toContain("error 1:2 invalid shader");
            expect(human).toContain("@vertex fn vs() {}");

            lines.length = 0;
            report(result as never, true);
            expect(JSON.parse(lines[0]).artifacts).toEqual(capture.artifacts);
        } finally {
            console.log = original;
        }
    });
});

describe("reportBatch — the batch human/JSON rendering", () => {
    const mk = (pass: boolean) => ({
        project: "/tmp/gym",
        mode: "dev",
        url: "http://localhost/gym",
        hardware: "test-gpu",
        harness: true,
        booted: true,
        rendered: true,
        errors: [],
        pass,
    });

    test("JSON mode is one array, in run order, no per-run header", () => {
        const lines: string[] = [];
        const original = console.log;
        console.log = (...args: unknown[]) => lines.push(args.join(" "));
        try {
            reportBatch(
                [mk(true), mk(false)] as never,
                ["scenario=outline", "scenario=sprite"],
                true,
            );
            expect(lines).toHaveLength(1);
            expect(JSON.parse(lines[0]).map((r: { pass: boolean }) => r.pass)).toEqual([
                true,
                false,
            ]);
        } finally {
            console.log = original;
        }
    });

    test("human mode labels each run with its --run spec and index", () => {
        const lines: string[] = [];
        const original = console.log;
        console.log = (...args: unknown[]) => lines.push(args.join(" "));
        try {
            reportBatch(
                [mk(true), mk(false)] as never,
                ["scenario=outline", "scenario=sprite"],
                false,
            );
            const human = lines.join("\n");
            expect(human).toContain("[run 1/2] scenario=outline");
            expect(human).toContain("[run 2/2] scenario=sprite");
            expect(human).toContain("PASS");
            expect(human).toContain("FAIL");
        } finally {
            console.log = original;
        }
    });
});

describe("stdout survives process.exit — the 64 KiB pipe truncation (shallot-perf-gates stage 7)", () => {
    // The truncation is a Node behavior, not a Bun one (measured: `bun -e "console.log(huge); process.exit(0)"`
    // piped never truncates; `node` does, exactly at 65,536 B). It's Node that hits it in production — the
    // WSL bridge runs a `bun build --target node` bundle of this CLI (`scripts/wsl-bridge.ts` buildBundle),
    // because that's how the batch sweep drives a real GPU on this platform. So the regression test bundles
    // `verify.ts` the same way and runs it under `node`, reproducing the CLI's real report boundary —
    // `reportBatch` under --json is the exact call a sweep's batch report makes — as two runs sharing one
    // >64 KiB payload: one exits the old way (no flush, the bug), one exits through `flushStdout` (the fix).
    // Both read through a real OS pipe, the same channel `scripts/verify.ts`'s `spawnVerify` reads.
    const hasNode = Bun.which("node") != null;

    test.skipIf(!hasNode)(
        "a >64 KiB batch report exits truncated without flushStdout, whole with it",
        async () => {
            const dir = mkdtempSync(join(REPO_ROOT, "node_modules/.cache/shallot-flush-test-"));
            try {
                const bundle = join(dir, "verify.bundle.mjs");
                const build = Bun.spawnSync(
                    [
                        "bun",
                        "build",
                        join(import.meta.dir, "verify.ts"),
                        "--target",
                        "node",
                        "--format",
                        "esm",
                        "--outfile",
                        bundle,
                        "--define",
                        `import.meta.dir="${import.meta.dir}"`,
                        "--external",
                        "vite",
                        "--external",
                        "playwright",
                        "--external",
                        "lightningcss",
                        "--external",
                        "@swc/*",
                        "--external",
                        "esbuild",
                        "--external",
                        "rollup",
                        "--external",
                        "fsevents",
                    ],
                    { cwd: REPO_ROOT, stdout: "pipe", stderr: "pipe" },
                );
                expect(build.exitCode).toBe(0);

                const makeResults =
                    "Array.from({ length: 4000 }, (_, i) => ({ project: '/tmp/gym', mode: 'dev', " +
                    "url: 'http://localhost/gym', hardware: 'test-gpu', harness: true, booted: true, " +
                    "rendered: true, errors: [], pass: true, " +
                    "verdict: { ok: true, checks: [], metrics: { scenario: i, blob: 'x'.repeat(300) } } }))";
                const labels = "results.map((_, i) => 'scenario=s' + i)";

                const unflushed = join(dir, "unflushed.mjs");
                writeFileSync(
                    unflushed,
                    `import { reportBatch } from "./verify.bundle.mjs";\n` +
                        `const results = ${makeResults};\n` +
                        `reportBatch(results, ${labels}, true);\n` +
                        `process.exit(0);\n`,
                );
                const flushed = join(dir, "flushed.mjs");
                writeFileSync(
                    flushed,
                    `import { flushStdout, reportBatch } from "./verify.bundle.mjs";\n` +
                        `const results = ${makeResults};\n` +
                        `reportBatch(results, ${labels}, true);\n` +
                        `await flushStdout();\n` +
                        `process.exit(0);\n`,
                );

                const run = async (script: string) => {
                    const proc = Bun.spawn(["node", script], {
                        cwd: dir,
                        stdout: "pipe",
                        stderr: "inherit",
                    });
                    const stdout = await new Response(proc.stdout).text();
                    await proc.exited;
                    return stdout;
                };

                const [before, after] = await Promise.all([run(unflushed), run(flushed)]);
                const fullLength = Buffer.byteLength(after, "utf8");

                // the payload is genuinely big enough to hit the wall...
                expect(fullLength).toBeGreaterThan(65_536);
                // ...the unflushed exit cuts it off before the pipe drains (the bug, still reproducible
                // today — the exact byte count where it cuts off is reader-timing-dependent, so only the
                // fact of truncation is asserted, not a specific count)...
                expect(Buffer.byteLength(before, "utf8")).toBeLessThan(fullLength);
                // ...and the flushed exit delivers every byte, parseable back into all 4000 verdicts (the fix).
                expect(JSON.parse(after.trim())).toHaveLength(4000);
            } finally {
                rmSync(dir, { recursive: true, force: true });
            }
        },
    );

    // The test above drives the primitives (`flushStdout` + `reportBatch`) hand-sequenced, which proves
    // the mechanism but not the wiring: reverting `runVerify`'s own `await flushStdout()` left it green
    // (found by the stage 7 review pass). This one drives `runVerify` itself — production's exact
    // function, no browser needed: `parseVerifyArgs` throws on an unrecognized flag (verify.ts:115)
    // before `importPlaywright` is ever reached, so a >64 KiB flag name reaches `reportError`'s JSON
    // payload through the same path a real setup failure takes. One arm is bare (replicates the pre-fix
    // shape — parse, print, exit, no flush) to prove the payload genuinely risks truncation on this pipe;
    // the other calls `runVerify` and exits on its returned code. Reverting `runVerify`'s flush makes the
    // production arm truncate exactly like the bare one, since both arms run through the current bundle.
    test.skipIf(!hasNode)(
        "a >64 KiB setup-error payload through runVerify survives process.exit; a bare exit does not",
        async () => {
            const dir = mkdtempSync(
                join(REPO_ROOT, "node_modules/.cache/shallot-flush-runverify-test-"),
            );
            try {
                const bundle = join(dir, "verify.bundle.mjs");
                const build = Bun.spawnSync(
                    [
                        "bun",
                        "build",
                        join(import.meta.dir, "verify.ts"),
                        "--target",
                        "node",
                        "--format",
                        "esm",
                        "--outfile",
                        bundle,
                        "--define",
                        `import.meta.dir="${import.meta.dir}"`,
                        "--external",
                        "vite",
                        "--external",
                        "playwright",
                        "--external",
                        "lightningcss",
                        "--external",
                        "@swc/*",
                        "--external",
                        "esbuild",
                        "--external",
                        "rollup",
                        "--external",
                        "fsevents",
                    ],
                    { cwd: REPO_ROOT, stdout: "pipe", stderr: "pipe" },
                );
                expect(build.exitCode).toBe(0);

                // large enough that the bare arm reliably hits the pipe wall (200 KB flirted with it and
                // flaked — a fast reader sometimes drains a single ~200 KB write before exit runs).
                const badArg = "'--' + 'x'.repeat(1_000_000)";

                const bare = join(dir, "bare.mjs");
                writeFileSync(
                    bare,
                    `import { parseVerifyArgs } from "./verify.bundle.mjs";\n` +
                        `try {\n` +
                        `  parseVerifyArgs([${badArg}, "--json"]);\n` +
                        `} catch (err) {\n` +
                        `  console.log(JSON.stringify({ pass: false, error: err.message }));\n` +
                        `}\n` +
                        `process.exit(2);\n`,
                );
                const viaRunVerify = join(dir, "via-runverify.mjs");
                writeFileSync(
                    viaRunVerify,
                    `import { runVerify } from "./verify.bundle.mjs";\n` +
                        `const code = await runVerify([${badArg}, "--json"]);\n` +
                        `process.exit(code);\n`,
                );

                const run = async (script: string) => {
                    const proc = Bun.spawn(["node", script], {
                        cwd: dir,
                        stdout: "pipe",
                        stderr: "inherit",
                    });
                    const stdout = await new Response(proc.stdout).text();
                    await proc.exited;
                    return stdout;
                };

                const [bareOut, runVerifyOut] = await Promise.all([run(bare), run(viaRunVerify)]);

                // the expected payload, computed independently of either subprocess's output — pins
                // against a known size rather than comparing two runs that can each truncate at a
                // reader-timing-dependent point (measured: comparing them against each other flaked when
                // both truncated, since the loser of that race is not always the same arm).
                const expectedMessage = `unknown option: --${"x".repeat(1_000_000)}`;
                const expectedJson = `${JSON.stringify({ pass: false, error: expectedMessage })}\n`;
                const expectedLength = Buffer.byteLength(expectedJson, "utf8");

                // the payload is genuinely big enough to hit the wall...
                expect(expectedLength).toBeGreaterThan(65_536);
                // ...the bare exit (parse, print, exit — no flush) cuts it off before the pipe drains...
                expect(Buffer.byteLength(bareOut, "utf8")).toBeLessThan(expectedLength);
                // ...and `runVerify` itself delivers every byte, byte-identical to the un-truncated JSON.
                expect(runVerifyOut).toBe(expectedJson);
                expect(JSON.parse(runVerifyOut.trim())).toEqual({
                    pass: false,
                    error: expectedMessage,
                });
            } finally {
                rmSync(dir, { recursive: true, force: true });
            }
        },
    );
});
