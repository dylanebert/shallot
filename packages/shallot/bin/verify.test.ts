import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { SCENARIO_GATES } from "../../../examples/gym/src/scenarios/timeouts";
import {
    benchTimeout,
    type ForMatch,
    formatForResolution,
    formatRoster,
    forUnmatchedReason,
    groupByTimeout,
    missingAssets,
    normalizeForPath,
    partitionSweep,
    resolveFor,
} from "../../../scripts/bench";
import { parsePhases, parseResources, parseTransformLine } from "../../../scripts/boot-cost";
import { verifyDiagnostic } from "../../../scripts/install-test";
import type { ShaderArtifactSummary, VerifyResult } from "../../../scripts/verify";
import { initialFrameSamplerState, sampleFrame } from "../../../site/rum-sampler";
import {
    ATTRIBUTION_INIT_SCRIPT,
    batchPass,
    bootArm,
    buildUrl,
    type Checkpoint,
    type CpuProfileNode,
    classifyCpuFrame,
    coerceVerdict,
    decodeSampleNode,
    displayGateExit,
    displayGateMessage,
    driveHarness,
    EXIT_NO_DISPLAY,
    type FrameSample,
    failureArtifacts,
    fitMemory,
    formatExtraTimings,
    formatTimings,
    gpuLogChecks,
    gridDiff,
    HARNESS_INPAGE_FUNCTION_NAMES,
    HARNESS_PROBE_SCRIPT,
    harnessBucketNames,
    harnessInstallMs,
    harnessPass,
    hasStructure,
    installHarnessProbe,
    isSoftwareAdapter,
    LEAK_BYTES_PER_SEC,
    type LoAFEntry,
    type LoAFScriptEntry,
    loafByCompilePhase,
    type MemorySample,
    parseVerifyArgs,
    pollFrameSample,
    type RawCpuProfile,
    RESOURCE_BUFFER,
    type RenderProbe,
    type ResourceEntry,
    type Result,
    report,
    reportBatch,
    resolveBatchQueries,
    SetupError,
    STRUCTURE_THRESHOLD,
    sampleFrameNode,
    selfTimeMsByNodeId,
    serveDev,
    serveDist,
    settlePass,
    spansFromCheckpoints,
    stepWait,
    structured,
    summarizeCpuProfile,
    summarizeResourceTiming,
    TIMINGS_INIT_SCRIPT,
    type WaitState,
    waitDiagnosis,
    withGpuLog,
    withTimeout,
} from "./verify";

const REPO_ROOT = resolve(import.meta.dir, "../../..");

/**
 * a scratch dir under the repo's own `node_modules/.cache`, not `tmpdir()` — the bundles and fixture
 * projects below are loaded by a Node subprocess that has to resolve the repo's dependencies, which only
 * works from inside the tree. `mkdtempSync` does not create parents, and `.cache` is a build artifact
 * absent from a fresh clone or a fresh worktree, so create it first: without this the suite reds with a
 * bare `ENOENT ... mkdtemp` that reads as a real failure.
 */
const cacheDir = (prefix: string): string => {
    const base = join(REPO_ROOT, "node_modules/.cache");
    mkdirSync(base, { recursive: true });
    return mkdtempSync(join(base, prefix));
};

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

    test("an empty --port value is rejected as empty, not silently coerced to 0", () => {
        expect(() => parseVerifyArgs(["--port="])).toThrow(
            'invalid --port value "" — must not be empty',
        );
    });

    test("a whitespace-only --port value is rejected, not silently coerced to 0", () => {
        expect(() => parseVerifyArgs(["--port", " "])).toThrow(
            'invalid --port value " " — must not be empty',
        );
    });

    test("--port 8080abc is rejected as non-numeric, not truncated to 8080", () => {
        expect(() => parseVerifyArgs(["--port", "8080abc"])).toThrow(
            'invalid --port value "8080abc"',
        );
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
    });

    test("--leak without --memory is a parse error (nothing samples the injected allocation)", () => {
        expect(() => parseVerifyArgs(["--leak", "122880"])).toThrow("--leak requires --memory");
        expect(() => parseVerifyArgs(["--leak=122880"])).toThrow("--leak requires --memory");
    });

    // `--leak 0` is accepted as off: `numNonNeg` allows zero (the `--leak` JSDoc: "0 = off"), and the
    // `--leak requires --memory` guard branches on `args.leak > 0`, so 0 never trips it. The sibling arm
    // above pins `parseVerifyArgs([]).leak === 0`; this one pins `parseVerifyArgs(["--leak", "0"]).leak
    // === 0` so a parser that substitutes a nonzero value for 0 can't pass both.
    test("--leak 0 is accepted as off (leak === 0, not rejected)", () => {
        expect(() => parseVerifyArgs(["--leak", "0"])).not.toThrow();
        expect(parseVerifyArgs(["--leak", "0"]).leak).toBe(0);
    });

    test("an empty --leak value is rejected as empty, not silently coerced to 0", () => {
        expect(() => parseVerifyArgs(["--leak="])).toThrow(
            'invalid --leak value "" — must not be empty',
        );
    });

    test("a whitespace-only --leak value is rejected, not silently coerced to 0", () => {
        expect(() => parseVerifyArgs(["--leak", " "])).toThrow(
            'invalid --leak value " " — must not be empty',
        );
    });

    test("--timings defaults off, flips on", () => {
        expect(parseVerifyArgs([]).timings).toBe(false);
        expect(parseVerifyArgs(["--timings"]).timings).toBe(true);
    });

    test("--attribution defaults off, flips on", () => {
        expect(parseVerifyArgs([]).attribution).toBe(false);
        expect(parseVerifyArgs(["--attribution"]).attribution).toBe(true);
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

// The CLI's own display gate: on a software adapter the run clears every feature/limit check and then
// dies mid-execution (`GPU device lost`, oversized `mappedAtCreation`) — measured 2026-08-18 running this
// exact CLI against `examples/showcase/voxel` and `roads` under WSL's `dzn`/SwiftShader fallback, hardware
// read as "google / swiftshader". `isSoftwareAdapter` is the pure classification that refuses it before
// any check runs; `displayGateExit` is the refusal-path seam reduced to its exit code, with no device
// execution (`testing.md`: a default-suite verdict must not depend on device execution).
describe("isSoftwareAdapter / displayGateExit — the CLI's own display gate", () => {
    test("real-hardware identity strings pass", () => {
        expect(isSoftwareAdapter("nvidia / ... / geforce rtx 4090 / ...")).toBe(false);
        expect(isSoftwareAdapter("apple / metal / apple m2 / apple m2")).toBe(false);
        expect(isSoftwareAdapter("amd / vulkan / radeon rx 7900 xtx")).toBe(false);
        expect(isSoftwareAdapter("intel / vulkan / intel(r) uhd graphics")).toBe(false);
    });

    test("software rasterizer strings refuse — the showcase drivers' name list, plus the measured WSL string", () => {
        expect(isSoftwareAdapter("google / swiftshader")).toBe(true); // measured 2026-08-18, this sandbox
        expect(isSoftwareAdapter("mesa / llvmpipe")).toBe(true);
        expect(isSoftwareAdapter("mesa / lavapipe")).toBe(true);
        expect(isSoftwareAdapter("microsoft basic render driver")).toBe(true);
        expect(isSoftwareAdapter("d3d12 / warp")).toBe(true);
    });

    test('no adapter offered at all (readHardware\'s "unknown" fallback) refuses too', () => {
        expect(isSoftwareAdapter("unknown")).toBe(true);
    });

    test("an unlisted adapter still passes — bias narrow, never over-block real hardware", () => {
        expect(isSoftwareAdapter("some future vendor / new arch")).toBe(false);
    });

    test("displayGateExit maps the classification to the distinct refusal exit code, or null to proceed", () => {
        expect(displayGateExit("google / swiftshader")).toBe(EXIT_NO_DISPLAY);
        expect(displayGateExit("unknown")).toBe(EXIT_NO_DISPLAY);
        expect(displayGateExit("nvidia / ... / geforce rtx 4090 / ...")).toBeNull();
        // distinct from every other exit code this CLI already uses
        expect(EXIT_NO_DISPLAY).toBe(4);
    });

    test("displayGateMessage names the adapter and the reason, never the caller or its environment", () => {
        const msg = displayGateMessage("google / swiftshader");
        expect(msg).toContain("google / swiftshader");
        expect(msg).toContain("software rasterizer");
        expect(msg.toLowerCase()).not.toContain("wsl");
        expect(msg.toLowerCase()).not.toContain("kex");
        expect(msg.toLowerCase()).not.toContain("bridge");

        const noAdapter = displayGateMessage("unknown");
        expect(noAdapter).toContain("no GPU adapter was offered");
    });
});

describe("runtime-agnostic boot path", () => {
    // `scripts/wsl-bridge.ts` bundles this CLI and drives it with node, where `Bun` is undefined, so any
    // `Bun.` in it is a boot-path defect whatever function holds it — file granularity is the property's
    // own granularity. Twice measured: `serveDist`'s `Bun.serve`/`Bun.file`, then `pickPort`'s port probe,
    // each surfacing only as `boot failed: Bun is not defined` on a display-gated bridge run.
    test("verify.ts reaches for no bun global", () => {
        const src = readFileSync(join(import.meta.dir, "verify.ts"), "utf8");
        const hits = src
            .split("\n")
            .map((line, i) => [i + 1, line] as const)
            .filter(([, line]) => /(?<![\w.])Bun\s*\./.test(line));
        expect(hits).toEqual([]);
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

// shallot-boot-stall-repair S1d. The three states a wait can time out in are the three the loop's own
// counters distinguish, and the branch that motivated the function is the first: `shallot verify
// examples/gym` with no `?scenario=` lands on gym's index page (a list of links, no canvas), and read
// `booted: false, samples: 0, errors: []` after a 60s clock — indistinguishable from a broken boot.
describe("waitDiagnosis — what a timed-out wait names", () => {
    test("only a timeout carries one", () => {
        expect(waitDiagnosis("harness", 0, false)).toBeNull();
        expect(waitDiagnosis("settled", 4, true)).toBeNull();
    });

    test("zero samples names the missing canvas and the URL-selection escape", () => {
        const d = waitDiagnosis("timeout", 0, false);
        expect(d).toContain("no capturable <canvas>");
        expect(d).toContain("--query scenario=");
    });

    test("samples without structure names a blank canvas, carrying the threshold it was judged against", () => {
        const d = waitDiagnosis("timeout", 7, false);
        expect(d).toContain("blank");
        expect(d).toContain("7 sample(s)");
        expect(d).toContain(String(STRUCTURE_THRESHOLD));
        expect(d).not.toContain("<canvas>");
    });

    test("structure that never settles names the animated scene, not a failure", () => {
        const d = waitDiagnosis("timeout", 12, true);
        expect(d).toContain("never went still");
        expect(d).toContain("12 sample(s)");
    });
});

describe("installHarnessProbe + harnessInstallMs — the pure setter mechanism", () => {
    test("writing __harness stamps __harnessInstallAt at that exact moment", () => {
        const win: Record<string, unknown> = {};
        installHarnessProbe(win);
        const before = Date.now();
        win.__harness = { ready: true };
        const after = Date.now();
        expect(win.__harness).toEqual({ ready: true });
        const installedAt = win.__harnessInstallAt as number;
        expect(installedAt).toBeGreaterThanOrEqual(before);
        expect(installedAt).toBeLessThanOrEqual(after);
        expect(harnessInstallMs(before, installedAt)).toBeGreaterThanOrEqual(0);
    });

    // red-proof, the "seen reporting the wrong thing when the setter is removed" half: a page that
    // never had the probe installed still accepts the same __harness write, but leaves no install-time
    // signal at all — the missing reading a probeless reader would silently treat as "never installed"
    // even when the harness installed just fine.
    test("without the probe, the same write leaves no install-time signal", () => {
        const win: Record<string, unknown> = {};
        win.__harness = { ready: true };
        expect(win.__harness).toEqual({ ready: true });
        expect(win.__harnessInstallAt).toBeUndefined();
        expect(
            harnessInstallMs(Date.now(), win.__harnessInstallAt as number | undefined),
        ).toBeNull();
    });

    test("harnessInstallMs: null when nothing installed, the elapsed gap otherwise", () => {
        expect(harnessInstallMs(1_000, undefined)).toBeNull();
        expect(harnessInstallMs(1_000, null)).toBeNull();
        expect(harnessInstallMs(1_000, 1_340)).toBe(340);
    });
});

// the cheap sentinel for `verify.probes.ts`, the browser-launching gate these two constants are proven
// in (a gate that leaves the default suite leaves a sentinel behind). Both concerns must reach
// the page before the first request — a raised resource-timing buffer and the harness-install setter —
// and dropping either from the init script is the silent failure the probes exist to prevent.
describe("TIMINGS_INIT_SCRIPT — what --timings installs pre-navigation", () => {
    test("carries the buffer raise and the harness probe, in that order", () => {
        expect(TIMINGS_INIT_SCRIPT).toContain(`setResourceTimingBufferSize(${RESOURCE_BUFFER})`);
        expect(TIMINGS_INIT_SCRIPT).toContain(HARNESS_PROBE_SCRIPT);
        expect(TIMINGS_INIT_SCRIPT.indexOf("setResourceTimingBufferSize")).toBeLessThan(
            TIMINGS_INIT_SCRIPT.indexOf(HARNESS_PROBE_SCRIPT),
        );
    });

    test("the probe script is the tested function's own source, not a hand-written copy", () => {
        expect(HARNESS_PROBE_SCRIPT).toContain(installHarnessProbe.toString());
    });
});

describe("ATTRIBUTION_INIT_SCRIPT — what --attribution installs pre-navigation", () => {
    test("is syntactically valid, self-contained script source", () => {
        expect(() => new Function(ATTRIBUTION_INIT_SCRIPT)).not.toThrow();
    });

    test("initializes the longtask sink before observing, and never throws on a missing API", () => {
        expect(ATTRIBUTION_INIT_SCRIPT.indexOf("__shallotLongTasks = []")).toBeLessThan(
            ATTRIBUTION_INIT_SCRIPT.indexOf("PerformanceObserver"),
        );
        expect(ATTRIBUTION_INIT_SCRIPT).toContain("try {");
    });

    test("initializes the sink to an empty array regardless of PerformanceObserver support", () => {
        const win: Record<string, unknown> = {};
        win.requestAnimationFrame = (() => 0) as unknown as typeof requestAnimationFrame;
        new Function("window", ATTRIBUTION_INIT_SCRIPT)(win);
        expect(win.__shallotLongTasks).toEqual([]);
    });

    // S1e: the LoAF observer is installed beside the longtask one, same guard shape — an unsupported
    // entryTypes throws synchronously in the observer constructor, caught so a missing LoAF degrades
    // to an empty array rather than killing the init script.
    test("initializes the LoAF sink before observing, and never throws on a missing API", () => {
        expect(ATTRIBUTION_INIT_SCRIPT.indexOf("__shallotLoAF = []")).toBeLessThan(
            ATTRIBUTION_INIT_SCRIPT.indexOf("long-animation-frame"),
        );
    });

    test("initializes the LoAF sink to an empty array regardless of PerformanceObserver support", () => {
        const win: Record<string, unknown> = {};
        win.requestAnimationFrame = (() => 0) as unknown as typeof requestAnimationFrame;
        new Function("window", ATTRIBUTION_INIT_SCRIPT)(win);
        expect(win.__shallotLoAF).toEqual([]);
    });

    // the unsupported-engine path exercised: an engine that supports longtask but NOT long-animation-
    // frame (the current Playwright chromium's LoAF support is the open question this test arms over)
    // must degrade the LoAF sink to an empty array while keeping the longtask sink working.
    test("degrades the LoAF sink to [] when the observer constructor throws on long-animation-frame, while keeping longtask working", () => {
        const win: Record<string, unknown> = {};
        win.requestAnimationFrame = (() => 0) as unknown as typeof requestAnimationFrame;
        win.PerformanceObserver = ((_cb: (list: { getEntries(): unknown[] }) => void) => ({
            observe(opts: { entryTypes: string[] }) {
                if (opts.entryTypes.includes("long-animation-frame")) {
                    throw new TypeError(
                        "The provided value 'long-animation-frame' is not a valid enum value",
                    );
                }
            },
        })) as unknown as PerformanceObserver;
        new Function("window", ATTRIBUTION_INIT_SCRIPT)(win);
        expect(win.__shallotLoAF).toEqual([]);
        expect(win.__shallotLongTasks).toEqual([]);
    });

    // S1f: the rAF-delta sampler is installed beside the longtask and LoAF observers on
    // `window.__shallotRafDeltas`. It mirrors `site/rum-sampler.ts`'s pure `sampleFrame` — same
    // threshold (50ms), same `delta = timestamp - lastTimestamp`, same first-frame rule (never
    // reports). An `addInitScript` string cannot import the module, so the mirror is a copy and
    // drift is the exposure: the arm below feeds the same fixture timestamps to both the mirror
    // (extracted from the shipped init-script string) and the real exported `sampleFrame`, asserting
    // they agree on which frames are slow and what their deltas/timestamps are.
    test("initializes the rAF-delta sink to an empty array", () => {
        const win: { __shallotRafDeltas?: unknown; requestAnimationFrame?: unknown } = {};
        win.requestAnimationFrame = (() => 0) as unknown as typeof requestAnimationFrame;
        new Function("window", ATTRIBUTION_INIT_SCRIPT)(win);
        expect(win.__shallotRafDeltas).toEqual([]);
    });

    test("rAF-delta mirror agrees with sampleFrame on a fixture timestamp sequence", () => {
        // Evaluate the shipped init script in a mock environment to extract the rAF callback.
        // The mock `requestAnimationFrame` captures the first callback (the one the IIFE registers)
        // and ignores subsequent registrations (the callback's own recursive rAF call), so the test
        // can drive the callback manually with fixture timestamps.
        const win: Record<string, unknown> = {};
        let rafCallback: ((timestamp: number) => void) | null = null;
        win.requestAnimationFrame = ((cb: (timestamp: number) => void) => {
            if (rafCallback === null) rafCallback = cb;
            return 0;
        }) as unknown as typeof requestAnimationFrame;
        win.PerformanceObserver = class {
            observe() {}
        } as unknown as typeof PerformanceObserver;
        new Function("window", ATTRIBUTION_INIT_SCRIPT)(win);
        expect(rafCallback).not.toBeNull();
        expect(win.__shallotRafDeltas).toEqual([]);

        // Fixture: a mix of normal frames (16ms), slow frames (>=50ms), a near-threshold frame
        // (45ms — below the 50ms threshold but above a perturbed 40ms threshold, so a threshold
        // perturbation is detectable), an exact-threshold frame (50ms — the boundary case, reported
        // by both sides because the comparison is >=), and a first frame.
        // deltas: first(0→0, no report), 16, 80(slow), 16, 60(slow), 16, 45(near), 16, 200(slow), 16, 1000(slow), 50(boundary)
        const timestamps = [0, 16, 96, 112, 172, 188, 233, 249, 449, 465, 1465, 1515];

        // Drive the mirror (extracted from the shipped init-script string)
        for (const t of timestamps) {
            rafCallback!(t);
        }
        const mirrorDeltas = win.__shallotRafDeltas as { delta: number; timestamp: number }[];

        // Drive the real exported `sampleFrame` with the same timestamps
        let state = initialFrameSamplerState;
        const moduleReports: { duration: number; msSinceLoad: number }[] = [];
        for (const t of timestamps) {
            const result = sampleFrame(state, t);
            state = result.state;
            if (result.report) {
                moduleReports.push({
                    duration: result.report.duration,
                    msSinceLoad: result.report.context.msSinceLoad,
                });
            }
        }

        // Assert they agree: same count, same deltas, same timestamps (msSinceLoad)
        expect(mirrorDeltas).toHaveLength(moduleReports.length);
        for (let i = 0; i < mirrorDeltas.length; i++) {
            expect(mirrorDeltas[i].delta).toBe(moduleReports[i].duration);
            expect(mirrorDeltas[i].timestamp).toBe(moduleReports[i].msSinceLoad);
        }
    });
});

describe("selfTimeMsByNodeId — S1b's CPU-profile self-time accounting", () => {
    // S1c: settled against two independent reference implementations (Chrome DevTools'
    // CPUProfileDataModel.ts and speedscope's Chrome importer — cited in full on the function's
    // docblock), not by reasoning — CDP's own spec prose ("the first delta is relative to the profile
    // startTime") is silent on direction. Witnessed red before this fix: the pre-S1c implementation
    // paired timeDeltas[i] with samples[i] over this same fixture (samples.length-bounded loop, so it
    // also double-counted the repeated node id 1 at i=0 and i=2) and read out.get(1)===0.45,
    // out.get(2)===0.3 — exit 1 on `expect(out.get(1)).toBeCloseTo(0.3, 6)`: "Received: 0.45".
    test("attributes timeDeltas[i+1] to samples[i] — the interval FOLLOWING a sample belongs to it, not the interval ending AT it, and the LAST sample gets nothing (no following delta to pair with)", () => {
        const profile: RawCpuProfile = {
            nodes: [],
            startTime: 0,
            endTime: 900,
            samples: [1, 2, 1],
            timeDeltas: [50, 300, 400, 250],
            // [0]: startTime→sample0 (unused). [1]: sample0→sample1 (300us) → charged to samples[0]=1.
            // [2]: sample1→sample2 (400us) → charged to samples[1]=2. [3] would be sample2→next, but
            // sample2 is the LAST sample, so it's dropped rather than double-counted onto node 1.
        };
        const out = selfTimeMsByNodeId(profile);
        expect(out.get(1)).toBeCloseTo(0.3, 6); // timeDeltas[1] only — NOT +timeDeltas[3]
        expect(out.get(2)).toBeCloseTo(0.4, 6); // timeDeltas[2]
        expect(out.size).toBe(2);
    });

    test("a non-positive delta is dropped rather than recorded as a zero-cost node — a real node with no observed cost is absent, not present at 0", () => {
        const profile: RawCpuProfile = {
            nodes: [],
            startTime: 0,
            endTime: 100,
            samples: [1, 2],
            timeDeltas: [0, -5],
        };
        expect(selfTimeMsByNodeId(profile).size).toBe(0);
    });

    test("a profile with no sample stream (samples/timeDeltas absent) reads as an empty map, not a crash", () => {
        const profile: RawCpuProfile = { nodes: [], startTime: 0, endTime: 0 };
        expect(selfTimeMsByNodeId(profile).size).toBe(0);
    });
});

describe("classifyCpuFrame — S1b's named-candidate bucket table", () => {
    test("routes each of the spec's named mechanisms to a distinct bucket", () => {
        const pipelines = classifyCpuFrame(
            "file:///repo/packages/shallot/src/standard/sear/pipelines.ts",
            "preparePipelines",
        );
        const forward = classifyCpuFrame(
            "file:///repo/packages/shallot/src/standard/sear/forward.ts",
            "unwrapVariant",
        );
        const gpu = classifyCpuFrame(
            "file:///repo/packages/shallot/src/engine/runtime/gpu.ts",
            "precompileAll",
        );
        const regather = classifyCpuFrame(
            "file:///repo/packages/shallot/src/standard/sear/regather.ts",
            "prepareRegather",
        );
        const ecs = classifyCpuFrame(
            "file:///repo/packages/shallot/src/engine/ecs/scheduler.ts",
            "tick",
        );
        const scene = classifyCpuFrame(
            "file:///repo/packages/shallot/src/engine/scene/preload.ts",
            "preload",
        );
        const decode = classifyCpuFrame(
            "file:///repo/packages/shallot/src/extras/gltf/pool.ts",
            "decodeInWorker",
        );
        const buckets = new Set([pipelines, forward, gpu, regather, ecs, scene, decode]);
        expect(buckets.size).toBe(7); // every named mechanism gets its own, distinguishable bucket
        expect(pipelines).toContain("pipelines.ts");
        expect(forward).toContain("forward.ts");
        expect(gpu).toContain("gpu.ts");
    });

    test("a native/no-url frame is named by its V8 convention, not folded into a file bucket", () => {
        expect(classifyCpuFrame("", "(program)")).toContain("(program)");
        expect(classifyCpuFrame("", "")).toContain("no source url");
    });

    // S1b's adversarial pass: the prior docblock claimed "no url implies V8-internal", refuted by
    // `decodeSample` (verify.ts's own in-page frame-sampling helper) landing here with no url and its
    // own real name — page.evaluate serializes a function with no backing file, so CDP reports an empty
    // url under the function's OWN name, not a synthetic one. Witnessed red against the pre-fix
    // implementation (`if (functionName) return \`${functionName} — V8-internal/native frame...\`` with
    // no synthetic-name check): `classifyCpuFrame("", "decodeSample")` returned
    // "decodeSample — V8-internal/native frame (no source url)", so this arm's
    // `expect(...).not.toContain("V8-internal")` exited 1 there —
    // "Received: decodeSample — V8-internal/native frame (no source url)".
    test("a NAMED no-url frame that isn't one of V8's synthetic names is NOT V8-internal — it's an evaluated/injected script", () => {
        const decodeSample = classifyCpuFrame("", "decodeSample");
        expect(decodeSample).not.toContain("V8-internal");
        expect(decodeSample).toContain("decodeSample");
        expect(decodeSample).toContain("evaluated script");
        // the four real synthetic names still classify as V8-internal — the fix narrows the no-url
        // branch, it doesn't widen it into "any name is native".
        for (const synthetic of ["(program)", "(idle)", "(garbage collector)", "(root)"]) {
            expect(classifyCpuFrame("", synthetic)).toContain("V8-internal");
        }
    });

    test("an unmatched frame falls through to a per-file bucket rather than disappearing", () => {
        const out = classifyCpuFrame(
            "file:///repo/packages/shallot/src/some/new/module.ts",
            "boot",
        );
        expect(out).toContain("new/module.ts");
    });
});

describe("summarizeCpuProfile — the pure profile-to-attribution reduction", () => {
    test("merges self time across nodes sharing one call-frame identity, and rolls it up by named bucket", () => {
        const pipelinesFrame = {
            functionName: "preparePipelines",
            scriptId: "1",
            url: "file:///repo/packages/shallot/src/standard/sear/pipelines.ts",
            lineNumber: 41,
            columnNumber: 0,
        };
        const nodes: CpuProfileNode[] = [
            { id: 1, callFrame: pipelinesFrame }, // called from call path A
            { id: 2, callFrame: pipelinesFrame }, // same function, called from call path B
            {
                id: 3,
                callFrame: {
                    functionName: "(program)",
                    scriptId: "0",
                    url: "",
                    lineNumber: 0,
                    columnNumber: 0,
                },
            },
        ];
        const profile: RawCpuProfile = {
            nodes,
            startTime: 0,
            endTime: 900,
            // S1c pairing: samples[i]'s self time is timeDeltas[i+1] (the interval FOLLOWING it), and
            // the last sample is dropped — so the stream carries one trailing sample to close node 3.
            samples: [1, 2, 3, 3],
            timeDeltas: [50_000, 400_000, 100_000, 200_000], // us: node1 400ms, node2 100ms, node3 200ms
        };
        const summary = summarizeCpuProfile(profile);

        expect(summary.totalMs).toBeCloseTo(700, 6);
        // node 1 and node 2 share one identity (same function+url+line) — merged into one entry at 500ms.
        expect(summary.entries).toHaveLength(2);
        const merged = summary.entries.find((e) => e.functionName === "preparePipelines");
        expect(merged?.selfMs).toBeCloseTo(500, 6);

        const pipelinesBucket = summary.buckets.find((b) => b.name.includes("pipelines.ts"));
        expect(pipelinesBucket?.selfMs).toBeCloseTo(500, 6);
        const programBucket = summary.buckets.find((b) => b.name.includes("(program)"));
        expect(programBucket?.selfMs).toBeCloseTo(200, 6);
    });

    test("a node with zero sampled self time is excluded from entries entirely — not a false zero", () => {
        const nodes: CpuProfileNode[] = [
            {
                id: 1,
                callFrame: {
                    functionName: "neverSampled",
                    scriptId: "1",
                    url: "file:///x.ts",
                    lineNumber: 0,
                    columnNumber: 0,
                },
            },
        ];
        const profile: RawCpuProfile = { nodes, startTime: 0, endTime: 0 };
        expect(summarizeCpuProfile(profile).entries).toHaveLength(0);
    });
});

describe("summarizeResourceTiming — the pure resource-timing reduction", () => {
    test("counts, sums duration, and ranks the top-N by duration descending", () => {
        const entries: ResourceEntry[] = [
            { name: "/a.js", duration: 10 },
            { name: "/b.js", duration: 50 },
            { name: "/c.js", duration: 30 },
        ];
        const t = summarizeResourceTiming(entries, 2);
        expect(t.count).toBe(3);
        expect(t.totalMs).toBe(90);
        expect(t.top).toEqual([
            { name: "/b.js", duration: 50 },
            { name: "/c.js", duration: 30 },
        ]);
    });

    test("empty entries summarize to a zeroed, empty readout", () => {
        expect(summarizeResourceTiming([])).toEqual({
            count: 0,
            totalMs: 0,
            top: [],
            saturated: false,
        });
    });
});

describe("formatExtraTimings — the pure --timings extra-lines rendering", () => {
    test("reports n/a when the harness-install probe never fired, and lists resources when present", () => {
        const out = formatExtraTimings(null, {
            count: 2,
            totalMs: 45,
            top: [{ name: "/a.js", duration: 40 }],
            saturated: false,
        });
        expect(out).toContain("harness install (probe)  n/a");
        expect(out).toContain("resources  2 requests, 45ms total");
        expect(out).toContain("40ms  /a.js");
        expect(out).not.toContain("floors");
    });

    test("a full buffer says so, so the count is never read as the real total", () => {
        const out = formatExtraTimings(null, {
            count: 250,
            totalMs: 45,
            top: [],
            saturated: true,
        });
        expect(out).toContain(
            "resources  250 requests, 45ms total (buffer full — both are floors)",
        );
    });

    test("reports the measured ms when the probe fired, and omits resources when there's no readout", () => {
        const out = formatExtraTimings(340, null);
        expect(out).toBe("  harness install (probe)  340ms");
    });
});

describe("parsePhases/parseResources — the boot-cost round trip over verify.ts's own formatters", () => {
    // scripts/boot-cost.ts scrapes formatTimings/formatExtraTimings's printed text back into structured
    // values by string format — a data boundary a format drift between the printer and the scraper can
    // silently break. This drives the real formatters' output into the real parser and asserts the
    // values survive the round trip.
    test("phases round-trip through formatTimings", () => {
        const spans = [
            { name: "server boot", ms: 517 },
            { name: "first page load", ms: 3583 },
            { name: "harness ready", ms: 1279 },
        ];
        const parsed = parsePhases(formatTimings(spans));
        expect(parsed).toEqual(spans);
    });

    test("a non-saturated resources line round-trips through formatExtraTimings", () => {
        const out = formatExtraTimings(2603, {
            count: 402,
            totalMs: 205600,
            top: [],
            saturated: false,
        });
        expect(parseResources(out)).toEqual({ count: 402, totalMs: 205600, saturated: false });
    });

    test("a saturated resources line — '(buffer full — both are floors)' — parses saturated: true", () => {
        const out = formatExtraTimings(null, {
            count: 20_000,
            totalMs: 12345,
            top: [],
            saturated: true,
        });
        expect(out).toContain("(buffer full — both are floors)");
        expect(parseResources(out)).toEqual({ count: 20_000, totalMs: 12345, saturated: true });
    });

    test("parsePhases also picks up formatExtraTimings' harness-install line, by design", () => {
        // "harness install (probe)  340ms" matches the same "name  Nms" shape as a checkpoint span, and
        // boot-cost.ts's runTimings relies on that: it folds both blocks' output through one parsePhases
        // call into one "phase medians" table, matching real --timings stdout, which prints both blocks
        // back to back.
        const combined = `${formatTimings([{ name: "run", ms: 1553 }])}\n${formatExtraTimings(340, null)}`;
        expect(parsePhases(combined)).toEqual([
            { name: "run", ms: 1553 },
            { name: "harness install (probe)", ms: 340 },
        ]);
    });
});

describe("parseTransformLine — reading Vite's own DEBUG=vite:plugin-transform format", () => {
    test("a real debugPluginTransform line yields the plugin and its duration", () => {
        // Vite's own shape (`timeFrom` + `createDebugger`, node_modules/vite/dist/node/chunks/node.js):
        // "<namespace> <duration>ms <plugin.name> <prettified module path> [+Nms]". NO_COLOR=1 (set by
        // boot-cost.ts's spawn) keeps the duration plain digits instead of a picocolors escape.
        const line = "vite:plugin-transform 12.34ms unplugin-typegpu src/index.ts +0ms";
        expect(parseTransformLine(line)).toEqual({ plugin: "unplugin-typegpu", ms: 12.34 });
    });

    test("a line outside the namespace, or one missing a field, parses to null", () => {
        expect(parseTransformLine("vite:resolve 1.00ms some-plugin src/index.ts")).toBeNull();
        expect(parseTransformLine("vite:plugin-transform 12.34ms")).toBeNull();
        expect(parseTransformLine("")).toBeNull();
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

// `missingAssets` drives both the sweep's and the single-scenario run's skip path: it reads the
// filesystem under examples/gym/public/ before booting a page, so a missing mount skips instantly with
// a named announcement instead of a 60s ready timeout. The `assets` declarations in timeouts.ts are a
// side table nothing else polices — a declared path drifting from the path the scenario actually fetches
// would surface only at runtime, so these tests pin the declared paths to what the loader expects.
//
// The mounts (sponza, gltf-samples) are gitignored local-only symlinks into reference/ (testing.md:176),
// absent in a clean checkout — so `missingAssets` reports them as missing here. A consumer who mounted
// them locally would see `null` instead; that's the non-standard state, not this one.
describe("missingAssets", () => {
    test("a scenario with no assets declaration returns null", () => {
        expect(missingAssets("stress", [])).toBeNull();
        expect(missingAssets("outline", [])).toBeNull();
    });

    test("gltf with default params reports the sponza path", () => {
        expect(missingAssets("gltf", [])).toEqual(["sponza/Sponza-KTX-Draco.glb"]);
    });

    test("gltf with source=fox reports the Fox path", () => {
        expect(missingAssets("gltf", ["source=fox"])).toEqual(["gltf-samples/Fox/glTF/Fox.gltf"]);
    });

    test("render with mode=gltf-animated reports the Fox path", () => {
        expect(missingAssets("render", ["mode=gltf-animated"])).toEqual([
            "gltf-samples/Fox/glTF/Fox.gltf",
        ]);
    });

    test("render with a non-gltf mode returns null (no mount needed)", () => {
        expect(missingAssets("render", [])).toBeNull();
        expect(missingAssets("render", ["mode=cull"])).toBeNull();
        expect(missingAssets("render", ["mode=fog"])).toBeNull();
    });

    // the drift rung that matters: the asset check resolves against REPO_ROOT, not process.cwd(), so a
    // bench run from a non-root cwd still finds (or misses) the same mounts. Before the fix, resolving
    // the relative GYM against a foreign cwd made the check cwd-dependent — a false skip from /tmp, or a
    // false pass from a subdir that happened to shadow examples/gym/public/. This holds the fix: a temp
    // cwd that shadows the mount path with a real file must NOT make `missingAssets` report the asset as
    // present, because the check never reads that cwd.
    test("the result is independent of process.cwd() (a shadowing cwd does not false-pass)", () => {
        const original = process.cwd();
        const shadow = mkdtempSync(join(import.meta.dir, "shadow-cwd-"));
        try {
            // plant a file at <shadow>/examples/gym/public/sponza/Sponza-KTX-Draco.glb — the relative
            // path the pre-fix resolve(GYM, "public", p) would have read from process.cwd().
            mkdirSync(join(shadow, "examples/gym/public/sponza"), { recursive: true });
            writeFileSync(
                join(shadow, "examples/gym/public/sponza/Sponza-KTX-Draco.glb"),
                "shadow",
            );

            const fromRoot = missingAssets("gltf", []);
            process.chdir(shadow);
            const fromShadow = missingAssets("gltf", []);
            // the shadow file exists at the cwd-relative path, but the check resolves against REPO_ROOT
            // where the mount is absent — so both report the asset as missing, never a false pass.
            expect(fromShadow).toEqual(fromRoot);
            expect(fromShadow).toEqual(["sponza/Sponza-KTX-Draco.glb"]);
        } finally {
            process.chdir(original);
            rmSync(shadow, { recursive: true, force: true });
        }
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

    // real data, as a declared registry asserted BOTH directions: the literal below is the
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

describe("stdout survives process.exit — the 64 KiB pipe truncation", () => {
    // The truncation is a Node behavior, not a Bun one (measured: `bun -e "console.log(huge); process.exit(0)"`
    // piped never truncates; `node` does, exactly at 65,536 B). It's Node that hits it in production — the
    // WSL bridge runs a `bun build --target node` bundle of this CLI (`scripts/wsl-bridge.ts` buildBundle),
    // because that's how the batch sweep drives a real GPU on this platform. So the regression test bundles
    // `verify.ts` the same way and runs it under `node`, reproducing the CLI's real report boundary —
    // `reportBatch` under --json is the exact call a sweep's batch report makes — as two runs sharing one
    // >64 KiB payload: one exits the old way (no flush, the bug), one exits through `flushStdout` (the fix).
    // Both read through a real OS pipe, the same channel `scripts/verify.ts`'s `spawnVerify` reads.
    const hasNode = Bun.which("node") != null;

    // both regression tests below bundle `verify.ts` the same way (a real `node`-targeted build, the
    // externals matching `scripts/wsl-bridge.ts`'s `buildBundle`) and run a script against it the same
    // way — shared here so the bundling command and the subprocess-reading logic live once, not twice.
    const buildVerifyBundle = (dir: string): string => {
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
        return bundle;
    };

    const runNode = async (dir: string, script: string): Promise<string> => {
        const proc = Bun.spawn(["node", script], {
            cwd: dir,
            stdout: "pipe",
            stderr: "inherit",
        });
        const stdout = await new Response(proc.stdout).text();
        await proc.exited;
        return stdout;
    };

    test.skipIf(!hasNode)(
        "a >64 KiB batch report exits truncated without flushStdout, whole with it",
        async () => {
            const dir = cacheDir("shallot-flush-test-");
            try {
                buildVerifyBundle(dir);

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

                const [before, after] = await Promise.all([
                    runNode(dir, unflushed),
                    runNode(dir, flushed),
                ]);
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
            const dir = cacheDir("shallot-flush-runverify-test-");
            try {
                buildVerifyBundle(dir);

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

                const [bareOut, runVerifyOut] = await Promise.all([
                    runNode(dir, bare),
                    runNode(dir, viaRunVerify),
                ]);

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

// Every red verdict this repo's gates ever report routes through driveHarness's failure arms and
// withGpuLog's merge — bench/flows/recipes only ever drive the green path. A duck-typed page stub is
// not module mocking: Page is already typed `any`, and the stub
// supplies only the boundary methods (`waitForFunction`, `evaluate`, `locator`) driveHarness itself
// calls — the pre-existing seam the Locked decision names.
describe("driveHarness — the red arms every gate's red routes through", () => {
    const baseResult = (): Omit<Result, "pass"> => ({
        project: "/tmp/proj",
        timestamp: "2026-01-01T00:00:00.000Z",
        mode: "dev",
        url: "http://localhost:1234/",
        hardware: "unknown",
        harness: false,
        booted: false,
        rendered: false,
        errors: [],
    });

    /** one of driveHarness's evaluate calls: the substring identifying its source, and what the page
     *  answers with (a function is called, so an arm can throw). */
    type Probe = { name: string; match: string; value: unknown | (() => unknown) };

    // a page that never renders anything real: the compositor screenshot always fails closed, so
    // `sampleFrame` reads null rather than fabricating a frame. `evaluate` dispatches on the probed
    // function's own source — the only stable way to tell driveHarness's several distinct evaluate calls
    // apart without threading a call counter through every test. A bare first-match `includes` chain is
    // silent when two probe sources share a matched substring (every one of them contains
    // `window.__harness`): the second probe takes the first arm's value, its own arm never runs, and the
    // test still passes. So match *every* arm and reject an ambiguous source, then record the dispatch
    // sequence — which the caller asserts, pinning that each probe fired exactly once and in order.
    const stubPage = (probes: readonly Probe[]) => {
        const fired: string[] = [];
        const page = {
            waitForFunction: async () => {},
            evaluate: async (fn: (...a: unknown[]) => unknown) => {
                const src = fn.toString();
                const hit = probes.filter((p) => src.includes(p.match));
                if (hit.length === 0) throw new Error(`unexpected evaluate: ${src}`);
                if (hit.length > 1) {
                    const names = hit.map((p) => p.name).join(", ");
                    throw new Error(`ambiguous probe dispatch (${names}) for: ${src}`);
                }
                fired.push(hit[0].name);
                const { value } = hit[0];
                return typeof value === "function" ? (value as () => unknown)() : value;
            },
            locator: () => ({
                first: () => ({
                    screenshot: async () => {
                        throw new Error("no canvas in this stub");
                    },
                }),
            }),
        };
        return { page, fired };
    };

    test("ready-timeout: window.__harness.ready never resolves is a hard FAIL, never a settle downgrade", async () => {
        const page = {
            waitForFunction: async () => {
                throw new Error("timed out");
            },
            evaluate: async () => {
                throw new Error("driveHarness must not evaluate before ready resolves");
            },
            locator: () => ({
                first: () => ({
                    screenshot: async () => {
                        throw new Error("no canvas");
                    },
                }),
            }),
        };
        const args = parseVerifyArgs(["--timeout", "5000"]);
        const result = await driveHarness(page, baseResult(), args, [], []);
        expect(result.booted).toBe(true);
        expect(result.rendered).toBe(false);
        expect(result.pass).toBe(false);
        expect(result.verdict).toEqual({
            ok: false,
            checks: [
                {
                    name: "ready",
                    ok: false,
                    detail: "window.__harness.ready never became true within 5000ms",
                },
            ],
        });
    });

    test('noRender opt-out: rendered reports "opt-out", never a fake true, and the pixel gate is skipped', async () => {
        const { page, fired } = stubPage([
            { name: "noRender", match: "noRender", value: true },
            { name: "hasRun", match: "typeof window.__harness", value: false }, // → fallback verdict
        ]);
        const args = parseVerifyArgs([]);
        const result = await driveHarness(page, baseResult(), args, [], []);
        expect(result.rendered).toBe("opt-out");
        expect(result.verdict).toEqual({ ok: true, checks: [{ name: "ready", ok: true }] });
        expect(result.pass).toBe(true);
        // the opt-out's whole point: the pixel path is never reached, so neither frame probe fires
        expect(fired).toEqual(["noRender", "hasRun"]);
    });

    test("probe-error verdict: a throwing __harness.run probe is a clean FAIL naming the error, never an unhandled rejection", async () => {
        const { page, fired } = stubPage([
            { name: "noRender", match: "noRender", value: false },
            {
                name: "hasRun",
                match: "typeof window.__harness",
                value: () => {
                    throw new Error("page closed");
                },
            },
            { name: "nextFrame", match: "requestAnimationFrame", value: undefined },
            { name: "pixelProbe", match: "pixelProbe", value: undefined },
        ]);
        const args = parseVerifyArgs([]);
        const result = await driveHarness(page, baseResult(), args, [], []);
        expect(result.verdict).toEqual({
            ok: false,
            checks: [
                { name: "run", ok: false, detail: "could not reach __harness.run: page closed" },
            ],
        });
        expect(result.pass).toBe(false);
        // a probe error short-circuits run(), never the pixel path — the capture still happens
        expect(fired).toEqual(["noRender", "hasRun", "nextFrame", "pixelProbe"]);
    });

    test("hasRun-fallback verdict: no run() on the harness reads readiness alone, gated on captured page errors", async () => {
        // a fresh stub per drive: `fired` is per-run, so one shared page would read two runs' probes
        const noRunHarness = () =>
            stubPage([
                { name: "noRender", match: "noRender", value: false },
                { name: "hasRun", match: "typeof window.__harness", value: false },
                { name: "nextFrame", match: "requestAnimationFrame", value: undefined },
                { name: "pixelProbe", match: "pixelProbe", value: undefined },
            ]);
        const args = parseVerifyArgs([]);
        const sequence = ["noRender", "hasRun", "nextFrame", "pixelProbe"];

        const cleanStub = noRunHarness();
        const clean = await driveHarness(cleanStub.page, baseResult(), args, [], []);
        expect(clean.verdict).toEqual({ ok: true, checks: [{ name: "ready", ok: true }] });
        expect(cleanStub.fired).toEqual(sequence);

        const errStub = noRunHarness();
        const withPageError = await driveHarness(
            errStub.page,
            baseResult(),
            args,
            ["a page error"],
            [],
        );
        expect(withPageError.verdict).toEqual({ ok: false, checks: [{ name: "ready", ok: true }] });
        // a captured page error changes the verdict's ok, never which probes run
        expect(errStub.fired).toEqual(sequence);
    });
});

describe("withGpuLog — the verdict merge arithmetic as a branch surface", () => {
    const mkResult = (overrides: Partial<Result>): Result => ({
        project: "/tmp/proj",
        timestamp: "t",
        mode: "dev",
        url: "http://x/",
        hardware: "unknown",
        harness: true,
        booted: true,
        rendered: true,
        errors: [],
        pass: true,
        ...overrides,
    });

    const gpuPage = (gpuLog: unknown, gpuDiagnostics: unknown) => ({
        evaluate: async (fn: (...a: unknown[]) => unknown) => {
            const src = fn.toString();
            if (src.includes("__gpuLog")) return gpuLog;
            if (src.includes("__gpuDiagnostics")) return gpuDiagnostics;
            throw new Error(`unexpected evaluate: ${src}`);
        },
    });

    test("no GPU log leaves the result unchanged — nothing to merge", async () => {
        const page = gpuPage(null, null);
        const result = mkResult({ pass: true });
        const merged = await withGpuLog(page, result);
        expect(merged).toEqual(result);
    });

    test("the ?? fallback reads result.pass only when no verdict exists — it never overrides an existing false", async () => {
        const page = gpuPage({ lines: ["ok"], errors: [] }, null);

        // no verdict at all: ok falls back to result.pass, in both directions.
        const noVerdictPass = await withGpuLog(page, mkResult({ pass: true }));
        expect(noVerdictPass.verdict?.ok).toBe(true);
        const noVerdictFail = await withGpuLog(page, mkResult({ pass: false }));
        expect(noVerdictFail.verdict?.ok).toBe(false);

        // an existing false ok is authoritative: `??` (not `||`) must not fall through to a true pass.
        const existingFalse = await withGpuLog(
            page,
            mkResult({ pass: true, verdict: { ok: false, checks: [{ name: "run", ok: false }] } }),
        );
        expect(existingFalse.verdict?.ok).toBe(false);
    });

    test("!failed flips a previously-ok verdict false on a GPU console.error, and appends the checks", async () => {
        const page = gpuPage({ lines: ["bad thing"], errors: ["bad thing"] }, null);
        const result = mkResult({
            pass: true,
            verdict: { ok: true, checks: [{ name: "ready", ok: true }] },
        });
        const merged = await withGpuLog(page, result);
        expect(merged.verdict?.ok).toBe(false);
        expect(merged.pass).toBe(false);
        expect(merged.verdict?.checks).toEqual([
            { name: "ready", ok: true },
            { name: "gpu.log", ok: true, detail: "bad thing" },
            { name: "gpu.error", ok: false, detail: "bad thing" },
        ]);
    });
});

describe("serve* SetupError guards — by temp dir, asserting the throw class, not message prose", () => {
    const tempProjectDir = (): string => cacheDir("shallot-verify-guard-");

    test("serveDist: no dist/ at all is a SetupError, not a raw ENOENT", async () => {
        const dir = tempProjectDir();
        try {
            await expect(serveDist(dir, 0)).rejects.toBeInstanceOf(SetupError);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    test("serveDist: dist/ exists but carries no index.html is the same SetupError", async () => {
        const dir = tempProjectDir();
        try {
            mkdirSync(join(dir, "dist"));
            await expect(serveDist(dir, 0)).rejects.toBeInstanceOf(SetupError);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });

    test("serveDev: neither a shallot manifest/.scene nor an index.html is a SetupError", async () => {
        const dir = tempProjectDir();
        try {
            await expect(serveDev(dir, 0)).rejects.toBeInstanceOf(SetupError);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    });
});

describe("report — the unrendered arms (⚠ LEAK, compilationError)", () => {
    const base = {
        project: "/tmp/gym",
        mode: "dev",
        url: "http://localhost/gym",
        hardware: "test-gpu",
        harness: false,
        booted: true,
        rendered: true,
        errors: [],
        pass: true,
    };

    test("a timeout's diagnosis prints as its own line; a result without one prints nothing", () => {
        const lines: string[] = [];
        const original = console.log;
        console.log = (...args: unknown[]) => lines.push(args.join(" "));
        try {
            report(
                {
                    ...base,
                    booted: false,
                    rendered: false,
                    pass: false,
                    diagnosis: "no capturable <canvas> ever appeared",
                } as never,
                false,
            );
            expect(lines.some((l) => l.includes("! no capturable <canvas> ever appeared"))).toBe(
                true,
            );
            lines.length = 0;
            report(base as never, false);
            expect(lines.some((l) => l.startsWith("  !"))).toBe(false);
        } finally {
            console.log = original;
        }
    });

    test("memory.leak marks the line with ⚠ LEAK; a clean slope doesn't", () => {
        const lines: string[] = [];
        const original = console.log;
        console.log = (...args: unknown[]) => lines.push(args.join(" "));
        try {
            report(
                {
                    ...base,
                    memory: {
                        start: 1e6,
                        end: 1.1e6,
                        growthPerSecond: 90_000,
                        leak: true,
                        gcCount: 2,
                        gcPauseMs: 3,
                    },
                } as never,
                false,
            );
            expect(lines.join("\n")).toContain("⚠ LEAK");

            lines.length = 0;
            report(
                {
                    ...base,
                    memory: {
                        start: 1e6,
                        end: 1.01e6,
                        growthPerSecond: 500,
                        leak: false,
                        gcCount: 2,
                        gcPauseMs: 3,
                    },
                } as never,
                false,
            );
            expect(lines.join("\n")).not.toContain("⚠ LEAK");
        } finally {
            console.log = original;
        }
    });

    test("a compilationError renders its class and message beneath the artifact", () => {
        const lines: string[] = [];
        const original = console.log;
        console.log = (...args: unknown[]) => lines.push(args.join(" "));
        try {
            report(
                {
                    ...base,
                    pass: false,
                    artifacts: [
                        {
                            label: "forward",
                            stage: "vertex+fragment",
                            source: "@vertex fn vs() {}",
                            hash: "0123456789abcdef",
                            messages: [],
                            compilationError: { errorClass: "validation", message: "bad binding" },
                        },
                    ],
                } as never,
                false,
            );
            expect(lines.join("\n")).toContain("compilation validation: bad binding");
        } finally {
            console.log = original;
        }
    });
});

// S3: the ejected boot arm's diagnostic reporter. `verifyDiagnostic` is the function the ejected boot
// check calls instead of printing `JSON.stringify(green.errors ?? green.error ?? green)` — the old form
// that printed `[]` for an empty errors array and was once dismissed as a flake. The arm feeds
// synthetic `VerifyResult` values through every diagnostic bin the function reads (page errors, setup
// error, verdict checks, shader artifacts) and the empty-diagnostic fallthrough, pinning that each
// names its diagnostic and that the empty case reports a named instrument fault rather than an empty
// container. A pass never reports an instrument fault. The red-when-broken witness (revert-the-behavior)
// is run inline: a mutated copy of the function with the empty-diagnostic branch removed returns the
// wrong thing, proving the arm discriminates.
describe("verifyDiagnostic — the ejected boot arm's diagnostic reporter", () => {
    /** construct a minimal VerifyResult with the given overrides. */
    const mk = (overrides: Partial<VerifyResult>): VerifyResult => ({
        pass: false,
        ...overrides,
    });

    test("a red carrying page errors names them (joined with |)", () => {
        const result = mk({ pass: false, errors: ["shader compile failed", "binding mismatch"] });
        const diag = verifyDiagnostic(result);
        expect(diag).toBe("shader compile failed | binding mismatch");
        expect(diag).not.toContain("instrument fault");
    });

    test("a red carrying only a setup error names it", () => {
        const result = mk({ pass: false, error: "port 5191 already in use" });
        const diag = verifyDiagnostic(result);
        expect(diag).toBe("port 5191 already in use");
        expect(diag).not.toContain("instrument fault");
    });

    test("a red carrying only failed verdict checks names them", () => {
        const result = mk({
            pass: false,
            verdict: {
                ok: false,
                checks: [
                    { name: "ready", ok: true },
                    { name: "render", ok: false, detail: "blank canvas" },
                ],
            },
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("render");
        expect(diag).toContain("blank canvas");
        expect(diag).not.toContain("instrument fault");
    });

    test("a red carrying only a shader artifacts compilation error names it", () => {
        const artifacts: ShaderArtifactSummary[] = [
            {
                label: "forward",
                stage: "vertex+fragment",
                compilationError: { errorClass: "validation", message: "bad binding" },
                messages: [],
            },
        ];
        const result = mk({ pass: false, artifacts });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("forward");
        expect(diag).toContain("bad binding");
        expect(diag).not.toContain("instrument fault");
    });

    test("a red carrying only shader artifact messages (no compilationError) names them", () => {
        const artifacts: ShaderArtifactSummary[] = [
            {
                label: "post",
                stage: "fragment",
                messages: [
                    { type: "error", message: "undeclared variable x", lineNum: 3, linePos: 8 },
                ],
            },
        ];
        const result = mk({ pass: false, artifacts });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("post");
        expect(diag).toContain("undeclared variable x");
        expect(diag).not.toContain("instrument fault");
    });

    test("a red carrying none of those returns the named instrument fault with field state", () => {
        const result = mk({ pass: false });
        const diag = verifyDiagnostic(result);
        // the instrument-fault wording is kept
        expect(diag).toContain("instrument fault");
        expect(diag).toContain("no diagnostic");
        // the failing predicates are named — all three failed since none are true
        expect(diag).toContain("failed predicates: pass===true, booted===true, rendered===true");
        // the field state is reported
        expect(diag).toContain("pass=false");
        expect(diag).toContain("booted=undefined");
        expect(diag).toContain("rendered=undefined");
        expect(diag).toContain("hardware=undefined");
        expect(diag).toContain("verdict=absent");
        expect(diag).toContain("memory=absent");
    });

    // the exact case the coordinator hit: red, no errors, no error, no verdict, no artifacts —
    // a boot that loaded (booted) but never rendered (rendered=false), carrying no diagnostic at
    // all. This is the string a future reader will actually see on a genuine red of this shape, so
    // its content is pinned, not merely that it is non-empty.
    test("the coordinator's case: booted=true, rendered=false, no diagnostic — pins the exact string", () => {
        const result = mk({
            pass: false,
            booted: true,
            rendered: false,
            hardware: "apple / metal / apple m2 / apple m2",
            errors: [],
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toBe(
            "instrument fault: verify red with no diagnostic (no page errors, no verdict checks, no shader artifacts); " +
                "failed predicates: pass===true, rendered===true; " +
                "pass=false, booted=true, rendered=false, hardware=apple / metal / apple m2 / apple m2, verdict=absent, memory=absent, renderProbe=absent",
        );
    });

    test("a red with booted=false names all three predicates as failed", () => {
        const result = mk({
            pass: false,
            booted: false,
            rendered: false,
            hardware: "unknown",
            errors: [],
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("failed predicates: pass===true, booted===true, rendered===true");
        expect(diag).toContain("pass=false, booted=false, rendered=false, hardware=unknown");
    });

    test("a red carrying a verdict with no failed checks still reports the verdict state", () => {
        const result = mk({
            pass: false,
            booted: true,
            rendered: true,
            hardware: "test-gpu",
            verdict: { ok: true, checks: [{ name: "ready", ok: true }] },
            errors: [],
        });
        const diag = verifyDiagnostic(result);
        // pass is false but verdict.ok is true and no checks failed — still an instrument fault
        expect(diag).toContain("instrument fault");
        expect(diag).toContain("failed predicates: pass===true");
        expect(diag).toContain("verdict.ok=true");
        expect(diag).toContain("verdict.checks=1");
    });

    test("a red carrying memory=present reports memory=present, not memory=absent", () => {
        const result = mk({
            pass: false,
            booted: true,
            rendered: false,
            hardware: "test-gpu",
            memory: null,
            errors: [],
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("memory=present");
        expect(diag).not.toContain("memory=absent");
    });

    test("a pass never reports an instrument fault", () => {
        const result = mk({ pass: true, errors: [] });
        const diag = verifyDiagnostic(result);
        expect(diag).toBe("pass");
        expect(diag).not.toContain("instrument fault");
    });

    test("a pass with errors still names the errors, never an instrument fault", () => {
        // a pass carrying errors is unusual but the function must not hide them behind "pass"
        const result = mk({ pass: true, errors: ["a warning"] });
        const diag = verifyDiagnostic(result);
        expect(diag).toBe("a warning");
        expect(diag).not.toContain("instrument fault");
    });

    test("null result returns a named absence, not an instrument fault", () => {
        const diag = verifyDiagnostic(null);
        expect(diag).toBe("no verify result");
        expect(diag).not.toContain("instrument fault");
    });

    // RED-WHEN-BROKEN WITNESS: a mutated copy of verifyDiagnostic with the empty-diagnostic
    // fallthrough removed (the last `return` line deleted, so the function returns an empty string
    // instead of the named instrument fault) must produce a value the instrument-fault arm above does
    // NOT accept — proving the arm discriminates the empty-diagnostic branch. If the arm still passed
    // with the branch broken, it would pin nothing about the instrument-fault path.
    test("red-when-broken: removing the empty-diagnostic fallthrough breaks the instrument-fault arm", () => {
        // the production function, with the final return replaced — the defect this arm exists to catch
        function brokenVerifyDiagnostic(result: VerifyResult | null): string {
            if (!result) return "no verify result";
            const errors = result.errors ?? [];
            if (errors.length > 0) return errors.join(" | ");
            if (result.error) return result.error;
            if (result.pass) return "pass";
            const failedChecks = (result.verdict?.checks ?? []).filter((c) => !c.ok);
            if (failedChecks.length > 0) return JSON.stringify(failedChecks);
            const artifacts = result.artifacts ?? [];
            const diagArtifacts = artifacts.filter(
                (a: ShaderArtifactSummary) =>
                    a.compilationError || (a.messages && a.messages.length > 0),
            );
            if (diagArtifacts.length > 0)
                return JSON.stringify(
                    diagArtifacts.map((a) => ({
                        label: a.label,
                        compilationError: a.compilationError,
                        messages: a.messages,
                    })),
                );
            // THE BROKEN BRANCH: the instrument-fault return with field state is replaced with "".
            return "";
        }

        const emptyResult = mk({ pass: false });
        const broken = brokenVerifyDiagnostic(emptyResult);
        const correct = verifyDiagnostic(emptyResult);

        // the broken version does NOT report the instrument fault — it returns an empty string
        expect(broken).toBe("");
        expect(broken).not.toContain("instrument fault");
        // the correct version DOES report it with field state
        expect(correct).toContain("instrument fault");
        expect(correct).toContain("failed predicates");
        // the arm discriminates: the broken and correct outputs differ
        expect(broken).not.toBe(correct);
    });
});

// S4: the render probe on verify's Result — the pixel evidence behind the `rendered` verdict. The
// settle path records samples taken, the last centre/corner RGB, the spread against `structured`'s
// threshold, the elapsed wait, and how the wait concluded; `verifyDiagnostic`'s instrument-fault branch
// prints it so a blank-render red carries its measurement rather than printing `[]`. These arms feed
// synthetic `VerifyResult` values carrying a `renderProbe` through the instrument-fault branch, pinning
// that each settle state names its measurement. The two-sided vacuity reading — a red carrying no probe
// AND no diagnostic still reports as a named instrument fault under its own name — pins that the
// probe's absence is named, never an empty container. The red-when-broken witness (revert-the-behavior)
// is run inline: a mutated copy of `verifyDiagnostic` with the probe-printing branch removed drops the
// measurement, proving the arms discriminate.
describe("verifyDiagnostic — the render probe in the instrument-fault branch", () => {
    /** construct a minimal VerifyResult with the given overrides. */
    const mk = (overrides: Partial<VerifyResult>): VerifyResult => ({
        pass: false,
        ...overrides,
    });

    /** a probe with spread just under `structured`'s threshold — the blank-render red's own reading. */
    const underThreshold: RenderProbe = {
        samples: 42,
        center: [10, 10, 10],
        corner: [14, 12, 11],
        spread: 7, // |10-14| + |10-12| + |10-11| = 4+2+1 = 7, under 12
        threshold: 12,
        elapsed: 30000,
        outcome: "timeout",
    };

    /** a probe with spread over threshold — the frame showed structure but never settled. */
    const overThreshold: RenderProbe = {
        samples: 60,
        center: [200, 100, 50],
        corner: [10, 10, 10],
        spread: 320, // |200-10| + |100-10| + |50-10| = 190+90+40 = 320, over 12
        threshold: 12,
        elapsed: 30000,
        outcome: "timeout",
    };

    /** a probe with zero samples — no canvas was ever capturable. */
    const zeroSamples: RenderProbe = {
        samples: 0,
        center: null,
        corner: null,
        spread: null,
        threshold: 12,
        elapsed: 30000,
        outcome: "timeout",
    };

    test("a blank-render red (spread under threshold) carries its probe measurement", () => {
        const result = mk({
            pass: false,
            booted: true,
            rendered: false,
            hardware: "apple / metal / apple m2 / apple m2",
            errors: [],
            renderProbe: underThreshold,
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("instrument fault");
        expect(diag).toContain("failed predicates: pass===true, rendered===true");
        // the probe's measurement is printed
        expect(diag).toContain("renderProbe={samples=42");
        expect(diag).toContain("spread=7");
        expect(diag).toContain("threshold=12");
        expect(diag).toContain("elapsed=30000ms");
        expect(diag).toContain("outcome=timeout");
        // the last centre/corner RGB is printed
        expect(diag).toContain("center=[10,10,10]");
        expect(diag).toContain("corner=[14,12,11]");
    });

    test("a red with spread over threshold carries its probe measurement", () => {
        const result = mk({
            pass: false,
            booted: true,
            rendered: true, // structure was seen but never settled
            hardware: "apple / metal / apple m2 / apple m2",
            errors: [],
            renderProbe: overThreshold,
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("instrument fault");
        expect(diag).toContain("renderProbe={samples=60");
        expect(diag).toContain("spread=320");
        expect(diag).toContain("outcome=timeout");
    });

    test("a red with zero samples carries the null probe measurement", () => {
        const result = mk({
            pass: false,
            booted: true,
            rendered: false,
            hardware: "unknown",
            errors: [],
            renderProbe: zeroSamples,
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("instrument fault");
        expect(diag).toContain("renderProbe={samples=0");
        expect(diag).toContain("center=[null]");
        expect(diag).toContain("corner=[null]");
        expect(diag).toContain("spread=null");
        expect(diag).toContain("outcome=timeout");
    });

    test("a red carrying no probe and no diagnostic still reports as a named instrument fault", () => {
        // the two-sided vacuity reading: no probe, no diagnostic — the fault is named under its own
        // shape, never an empty container a future reader can dismiss
        const result = mk({
            pass: false,
            booted: true,
            rendered: false,
            hardware: "apple / metal / apple m2 / apple m2",
            errors: [],
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("instrument fault");
        expect(diag).toContain("renderProbe=absent");
        expect(diag).toContain("failed predicates: pass===true, rendered===true");
        // the probe's absence is named, not silently dropped
        expect(diag).not.toContain("renderProbe={");
    });

    test("a red with a harness-path probe (outcome=harness) carries its measurement", () => {
        const probe: RenderProbe = {
            samples: 3, // wait-loop + ready + post
            center: [0, 0, 0],
            corner: [0, 0, 0],
            spread: 0,
            threshold: 12,
            elapsed: 5000,
            outcome: "harness",
        };
        const result = mk({
            pass: false,
            booted: true,
            rendered: false,
            hardware: "apple / metal / apple m2 / apple m2",
            errors: [],
            renderProbe: probe,
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("renderProbe={samples=3");
        expect(diag).toContain("spread=0");
        expect(diag).toContain("outcome=harness");
    });

    test("a red with a settled probe (outcome=settled) carries its measurement", () => {
        const probe: RenderProbe = {
            samples: 5,
            center: [100, 80, 60],
            corner: [20, 20, 20],
            spread: 180,
            threshold: 12,
            elapsed: 3000,
            outcome: "settled",
        };
        const result = mk({
            pass: false,
            booted: true,
            rendered: true, // settled but pass is still false (e.g. page errors)
            hardware: "test-gpu",
            errors: [],
            renderProbe: probe,
        });
        const diag = verifyDiagnostic(result);
        expect(diag).toContain("renderProbe={samples=5");
        expect(diag).toContain("spread=180");
        expect(diag).toContain("outcome=settled");
    });

    // RED-WHEN-BROKEN WITNESS: a mutated copy of verifyDiagnostic with the probe-printing branch
    // removed (the `renderProbe` field state replaced with a bare `renderProbe=absent` regardless of
    // whether a probe was carried) must drop the measurement, proving the arms discriminate the
    // probe-printing branch. If the arms still passed with the branch broken, they would pin nothing
    // about the probe's content.
    test("red-when-broken: removing the probe-printing branch drops the measurement", () => {
        // the production function, with the probe-printing branch replaced — the defect these arms
        // exist to catch
        function brokenVerifyDiagnostic(result: VerifyResult | null): string {
            if (!result) return "no verify result";
            const errors = result.errors ?? [];
            if (errors.length > 0) return errors.join(" | ");
            if (result.error) return result.error;
            if (result.pass) return "pass";
            const failedChecks = (result.verdict?.checks ?? []).filter((c) => !c.ok);
            if (failedChecks.length > 0) return JSON.stringify(failedChecks);
            const artifacts = result.artifacts ?? [];
            const diagArtifacts = artifacts.filter(
                (a: ShaderArtifactSummary) =>
                    a.compilationError || (a.messages && a.messages.length > 0),
            );
            if (diagArtifacts.length > 0)
                return JSON.stringify(
                    diagArtifacts.map((a) => ({
                        label: a.label,
                        compilationError: a.compilationError,
                        messages: a.messages,
                    })),
                );
            const failed: string[] = [];
            failed.push("pass===true");
            if (result.booted !== true) failed.push("booted===true");
            if (result.rendered !== true) failed.push("rendered===true");
            const fields: string[] = [
                `pass=${result.pass}`,
                `booted=${result.booted ?? "undefined"}`,
                `rendered=${result.rendered ?? "undefined"}`,
                `hardware=${result.hardware ?? "undefined"}`,
            ];
            if (result.verdict) {
                fields.push(`verdict.ok=${result.verdict.ok ?? "undefined"}`);
                fields.push(`verdict.checks=${result.verdict.checks?.length ?? 0}`);
            } else {
                fields.push("verdict=absent");
            }
            fields.push(result.memory !== undefined ? "memory=present" : "memory=absent");
            // THE BROKEN BRANCH: the probe-printing block is replaced with a bare absent — the
            // measurement is silently dropped regardless of whether a probe was carried.
            fields.push("renderProbe=absent");
            return `instrument fault: verify red with no diagnostic (no page errors, no verdict checks, no shader artifacts); failed predicates: ${failed.join(", ")}; ${fields.join(", ")}`;
        }

        const withProbe = mk({
            pass: false,
            booted: true,
            rendered: false,
            hardware: "test-gpu",
            errors: [],
            renderProbe: underThreshold,
        });
        const broken = brokenVerifyDiagnostic(withProbe);
        const correct = verifyDiagnostic(withProbe);

        // the broken version drops the measurement — it reports absent even though a probe was carried
        expect(broken).toContain("renderProbe=absent");
        expect(broken).not.toContain("renderProbe={");
        expect(broken).not.toContain("spread=7");
        // the correct version carries the measurement
        expect(correct).toContain("renderProbe={");
        expect(correct).toContain("spread=7");
        // the arm discriminates: the broken and correct outputs differ
        expect(broken).not.toBe(correct);
    });
});

// S1c's owed pieces: (1) an arm asserting no verify-harness call frame appears in an attribution run's
// bucket table, red on the pre-fix contaminated shape and green on the decontaminated shape; (2) coverage
// for `decodeSampleNode` and `sampleFrameNode`, which had none — including the pngjs-unavailable fallback.

describe("harnessBucketNames — S1c's absence arm over a captured CPU profile", () => {
    // a call frame identity shared by the contaminated and clean fixtures so the only variable is
    // whether the harness frame is present.
    const pipelinesFrame = {
        functionName: "preparePipelines",
        scriptId: "1",
        url: "file:///repo/packages/shallot/src/standard/sear/pipelines.ts",
        lineNumber: 41,
        columnNumber: 0,
    };
    // the harness's own in-page decode helper — `sampleFrame`'s `page.evaluate(decodeSample, ...)`
    // serialized this function into the profiled tab. CDP reports it with an empty url under the
    // function's own name (not a V8 synthetic name), so `classifyCpuFrame` routes it to a bucket
    // containing "decodeSample" and "evaluated script".
    const decodeSampleFrame = {
        functionName: "decodeSample",
        scriptId: "0",
        url: "",
        lineNumber: 0,
        columnNumber: 0,
    };

    // the pre-S1c contaminated shape: `sampleFrame` ran `page.evaluate(decodeSample, ...)` inside the
    // profiled tab on every 500ms poll, so `decodeSample` appears as a real sampled node with self
    // time — exactly the contamination S1b's adversarial pass found as the single largest bucket.
    const contaminatedProfile: RawCpuProfile = {
        nodes: [
            { id: 1, callFrame: decodeSampleFrame },
            { id: 2, callFrame: pipelinesFrame },
        ],
        startTime: 0,
        endTime: 600,
        // S1c pairing: samples[i]'s self time is timeDeltas[i+1]; last sample dropped.
        // decodeSample (node 1) gets 400ms, preparePipelines (node 2) gets 100ms.
        samples: [1, 2, 2],
        timeDeltas: [50_000, 400_000, 100_000, 50_000],
    };

    // the S1c decontaminated shape: `sampleFrameNode` decodes in Node with pngjs, so no harness
    // function appears in the page's CPU profile — only real app frames.
    const cleanProfile: RawCpuProfile = {
        nodes: [{ id: 2, callFrame: pipelinesFrame }],
        startTime: 0,
        endTime: 200,
        samples: [2, 2],
        timeDeltas: [50_000, 100_000, 50_000],
    };

    test("reds on the pre-fix contaminated shape — decodeSample is classified into a real bucket", () => {
        const summary = summarizeCpuProfile(contaminatedProfile);
        const harness = harnessBucketNames(summary);
        // the arm FAILS here: harness buckets are present, so the absence assertion would red.
        // This is the red witness — the arm detects the contamination it exists to catch.
        expect(harness.length).toBeGreaterThan(0);
        expect(harness[0]).toContain("decodeSample");
        expect(harness[0]).toContain("evaluated script");
        // the contaminated profile's bucket table actually carries the harness bucket with real self time
        const decodeBucket = summary.buckets.find((b) => b.name.includes("decodeSample"));
        expect(decodeBucket).toBeDefined();
        expect(decodeBucket?.selfMs).toBeCloseTo(400, 6);
    });

    test("greens on the decontaminated shape — no harness bucket in the table", () => {
        const summary = summarizeCpuProfile(cleanProfile);
        const harness = harnessBucketNames(summary);
        // the arm PASSES here: no harness buckets — the decontaminated profile is clean.
        expect(harness).toHaveLength(0);
        // and the real app frame is still present and attributed — the arm doesn't strip real work
        expect(summary.buckets.length).toBeGreaterThan(0);
        expect(summary.buckets[0].name).toContain("pipelines.ts");
    });

    // RED-WHEN-BROKEN WITNESS: a `harnessBucketNames` that ignores the harness function names (returns
    // [] unconditionally) must green on the contaminated fixture, proving the arm discriminates.
    test("red-when-broken: a no-op harnessBucketNames greens on the contaminated shape (the arm would not red)", () => {
        function brokenHarnessBucketNames(): string[] {
            return []; // the defect: never checks for harness frames
        }
        const summary = summarizeCpuProfile(contaminatedProfile);
        const broken = brokenHarnessBucketNames();
        const correct = harnessBucketNames(summary);
        // the broken version misses the contamination — it returns empty on a contaminated profile
        expect(broken).toHaveLength(0);
        // the correct version catches it
        expect(correct.length).toBeGreaterThan(0);
        // the arm discriminates: the broken and correct outputs differ
        expect(broken).not.toEqual(correct);
    });
});

describe("decodeSampleNode — the Node-side PNG-to-FrameSample decoder", () => {
    // a 4×4 RGBA image: top-left corner is black (0,0,0), center is red (255,0,0). Box-averaging
    // the source pixels covering each 64×64 cell still replicates each source pixel across a 16×16
    // block here (4 divides 64 evenly, so every cell's source rect is exactly one source pixel), so
    // the region averages preserve the source contrast — the same signal `decodeSample`'s canvas
    // resample produces, just via a different downsample path.
    const mk4x4 = (
        tl: number[],
        center: number[],
    ): { width: number; height: number; data: Uint8Array } => {
        const data = new Uint8Array(4 * 4 * 4);
        for (let y = 0; y < 4; y++)
            for (let x = 0; x < 4; x++) {
                const isCenter = x >= 1 && x <= 2 && y >= 1 && y <= 2;
                const [r, g, b] = isCenter ? center : tl;
                const i = (y * 4 + x) * 4;
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                data[i + 3] = 255;
            }
        return { width: 4, height: 4, data };
    };

    test("a structured image (center ≠ corner) produces a FrameSample with visible structure", () => {
        const png = mk4x4([0, 0, 0], [255, 0, 0]);
        const sample = decodeSampleNode(png);
        expect(sample).not.toBeNull();
        // center is red, corner is black — `hasStructure` should pass
        expect(hasStructure(sample)).toBe(true);
        expect(sample!.center[0]).toBeGreaterThan(200); // red channel lifted
        expect(sample!.corner[0]).toBeLessThan(50); // corner stays dark
    });

    test("a flat image (center = corner) produces a FrameSample with no structure", () => {
        const png = mk4x4([50, 50, 50], [50, 50, 50]);
        const sample = decodeSampleNode(png);
        expect(sample).not.toBeNull();
        expect(hasStructure(sample)).toBe(false);
    });

    test("a zero-dimension image returns null, not a crash", () => {
        expect(decodeSampleNode({ width: 0, height: 4, data: new Uint8Array(0) })).toBeNull();
        expect(decodeSampleNode({ width: 4, height: 0, data: new Uint8Array(0) })).toBeNull();
    });

    test("the grid is 64×64×3 channels (12288 values), matching decodeSample's shape", () => {
        const png = mk4x4([0, 0, 0], [255, 0, 0]);
        const sample = decodeSampleNode(png);
        expect(sample).not.toBeNull();
        expect(sample!.grid.length).toBe(64 * 64 * 3);
    });
});

// shallot-boot-stall-repair S1 follow-up (blocker repair): `decodeSample` (the browser path) downsamples
// through `ctx.drawImage(img, 0, 0, 64, 64)` — an area-average resample — while `decodeSampleNode` (the
// Node path every non-`--attribution` boot now takes) has to reproduce that same area-average by
// construction, since it can never run through a real `<canvas>`. Display-gated real-hardware evidence
// (S1's own recorded verdict-parity differential) cannot witness this: `pollFrameSample` returns null
// whenever `harnessDefined`, so every harness project in this repo takes a display-gated path that never
// reaches the Node decoder's non-attribution branch at all, and a plain `shallot verify` at this seat
// doesn't boot ({samples: 0, outcome: "timeout"} identically on both the pre-fix and post-fix branch).
// This arm is the display-independent substitute: a real pngjs-round-tripped PNG, decoded through
// `decodeSampleNode` and through an independently-written area-average reference (fractional pixel-
// coverage weighting — NOT decodeSampleNode's own non-overlapping-cell box grouping, so the two
// implementations can actually disagree), asserting the two 64×64 grids agree within a tight per-channel
// tolerance and that `structured()`/`gridDiff` read the same verdict off both. The high-frequency
// checkerboard region is the arm's whole point: a point-sample (nearest-neighbor) decoder passes per-
// pixel dither straight through at full amplitude where an area-average one suppresses it, so this arm
// reds against the pre-fix nearest-neighbor `decodeSampleNode` and is green only against the box-averaged
// one — witnessed by hand: reverting `decodeSampleNode` to point-sampling (`git show 6ddb68d1:…`) and
// re-running this file reds exactly this describe block, on the checkerboard region's per-channel
// tolerance assertions, with the smooth-gradient region staying green throughout (a point sample of a
// smooth ramp is already close to its local area average — the checkerboard is the discriminating input).
describe("decodeSampleNode area-average equivalence — the blocker repair's own oracle", () => {
    const canvasW = 1280;
    const canvasH = 720; // 720 / 64 = 11.25 — deliberately not integer-divisible, so 64×64 grid cells
    // land on fractional source-pixel boundaries and the two implementations' rounding can diverge.

    // left half: a smooth horizontal luminance ramp (low frequency — a point sample and an area
    // average of a smooth gradient are already close, by construction). Right half: a 1px checkerboard
    // (the highest spatial frequency a raster can carry) — the region a point sample cannot suppress
    // and an area average must.
    function buildSyntheticFrame(): { width: number; height: number; data: Buffer } {
        const data = Buffer.alloc(canvasW * canvasH * 4);
        for (let y = 0; y < canvasH; y++) {
            for (let x = 0; x < canvasW; x++) {
                const i = (y * canvasW + x) * 4;
                let r: number;
                let g: number;
                let b: number;
                if (x < canvasW / 2) {
                    const v = Math.round((x / (canvasW / 2)) * 255);
                    r = v;
                    g = v;
                    b = v;
                } else {
                    const on = (x + y) % 2 === 0;
                    r = on ? 255 : 0;
                    g = on ? 255 : 0;
                    b = on ? 255 : 0;
                }
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                data[i + 3] = 255;
            }
        }
        return { width: canvasW, height: canvasH, data };
    }

    // independently-derived area-average reference: for each 64×64 destination cell, weight every
    // overlapping source pixel by its fractional coverage of the cell's continuous source-space rect —
    // the textbook box-filter resize, not decodeSampleNode's own integer-cell grouping. Two different
    // implementations of "area average", so agreement is evidence rather than a restated identity
    // (checks.md: "an equality between two quantities the same expression produces is only a check when
    // one side comes from outside the artifact").
    function areaAverageReference(png: { width: number; height: number; data: Buffer }): number[] {
        const { width: W0, height: H0, data } = png;
        const W = 64;
        const H = 64;
        const grid: number[] = [];
        for (let y = 0; y < H; y++) {
            const fy0 = (y / H) * H0;
            const fy1 = ((y + 1) / H) * H0;
            const sy0 = Math.floor(fy0);
            const sy1 = Math.min(H0, Math.ceil(fy1));
            for (let x = 0; x < W; x++) {
                const fx0 = (x / W) * W0;
                const fx1 = ((x + 1) / W) * W0;
                const sx0 = Math.floor(fx0);
                const sx1 = Math.min(W0, Math.ceil(fx1));
                let r = 0;
                let g = 0;
                let b = 0;
                let wsum = 0;
                for (let sy = sy0; sy < sy1; sy++) {
                    const yOverlap = Math.min(sy + 1, fy1) - Math.max(sy, fy0);
                    if (yOverlap <= 0) continue;
                    for (let sx = sx0; sx < sx1; sx++) {
                        const xOverlap = Math.min(sx + 1, fx1) - Math.max(sx, fx0);
                        if (xOverlap <= 0) continue;
                        const w = xOverlap * yOverlap;
                        const i = (sy * W0 + sx) * 4;
                        r += data[i] * w;
                        g += data[i + 1] * w;
                        b += data[i + 2] * w;
                        wsum += w;
                    }
                }
                grid.push(r / wsum, g / wsum, b / wsum);
            }
        }
        return grid;
    }

    const region = (
        grid: number[],
        fx0: number,
        fy0: number,
        fx1: number,
        fy1: number,
    ): number[] => {
        const W = 64;
        const H = 64;
        const x0 = Math.floor(fx0 * W);
        const x1 = Math.max(x0 + 1, Math.floor(fx1 * W));
        const y0 = Math.floor(fy0 * H);
        const y1 = Math.max(y0 + 1, Math.floor(fy1 * H));
        let r = 0;
        let g = 0;
        let b = 0;
        let n = 0;
        for (let y = y0; y < y1; y++)
            for (let x = x0; x < x1; x++) {
                const i = (y * W + x) * 3;
                r += grid[i];
                g += grid[i + 1];
                b += grid[i + 2];
                n++;
            }
        return [r / n, g / n, b / n];
    };

    test("decodeSampleNode's box-average grid agrees with an independent area-average reference, tightly, on both the smooth and the high-frequency regions", async () => {
        const { PNG } = await import("pngjs");
        const frame = buildSyntheticFrame();
        const png = new PNG({ width: canvasW, height: canvasH });
        frame.data.copy(png.data);
        // round-trip through a real PNG encode/decode, the same as a Playwright screenshot buffer —
        // not just a raw object literal — so this exercises the real pngjs codepath sampleFrameNode uses.
        const roundTripped = PNG.sync.read(PNG.sync.write(png));

        const nodeSample = decodeSampleNode(roundTripped);
        expect(nodeSample).not.toBeNull();
        const referenceGrid = areaAverageReference(roundTripped);

        // per-channel tolerance: loose enough to absorb the two implementations' different rounding
        // (floor-based integer cells vs. fractional-coverage weighting) and tight enough that a
        // nearest-neighbor decoder — which disagrees with the area-average reference by up to the full
        // 0..255 range on the checkerboard — cannot pass it.
        const tolerance = 8;
        let maxDiff = 0;
        for (let i = 0; i < nodeSample!.grid.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(nodeSample!.grid[i] - referenceGrid[i]));
        }
        expect(maxDiff).toBeLessThanOrEqual(tolerance);

        // the high-frequency checkerboard region specifically — the discriminating input a smooth
        // gradient can't exercise, since a point sample of a smooth ramp already sits close to its
        // local area average.
        const nodeChecker = region(nodeSample!.grid, 0.5, 0, 1, 1);
        const refChecker = region(referenceGrid, 0.5, 0, 1, 1);
        for (let c = 0; c < 3; c++) {
            expect(Math.abs(nodeChecker[c] - refChecker[c])).toBeLessThanOrEqual(tolerance);
        }

        // structured()/gridDiff agree on both grids — the two thresholds the boot-settle heuristic
        // actually gates on read the same verdict off either implementation.
        const nodeCorner = region(nodeSample!.grid, 0, 0, 0.15, 0.15);
        const refCorner = region(referenceGrid, 0, 0, 0.15, 0.15);
        expect(structured(nodeChecker, nodeCorner)).toBe(structured(refChecker, refCorner));
        expect(gridDiff(nodeSample!.grid, referenceGrid)).toBeLessThan(tolerance);
    });
});

describe("sampleFrameNode — the S1c decontaminated frame sampler", () => {
    // a duck-typed page stub matching the shape `sampleFrameNode` calls: `page.locator("canvas").first()
    // .screenshot(...)` and (fallback only) `page.evaluate(decodeSample, b64)`.
    type ScreenshotResult = Buffer | { toString(encoding: string): string };

    const mkPage = (
        screenshotFn: () => Promise<ScreenshotResult>,
        evaluateFn?: (fn: (...a: unknown[]) => unknown, ...args: unknown[]) => Promise<unknown>,
    ) => ({
        locator: () => ({
            first: () => ({
                screenshot: async () => screenshotFn(),
            }),
        }),
        evaluate: evaluateFn ?? (async () => null),
    });

    test("screenshot failure returns null (no canvas, no crash)", async () => {
        const page = mkPage(async () => {
            throw new Error("no canvas");
        });
        expect(await sampleFrameNode(page as never)).toBeNull();
    });

    test("pngjs path: a valid PNG screenshot decodes to a FrameSample in Node (no page.evaluate)", async () => {
        // Build a minimal 2×2 red PNG with pngjs so the screenshot returns real PNG bytes.
        const { PNG } = await import("pngjs");
        const png = new PNG({ width: 2, height: 2 });
        // top-left = black, bottom-right = red — center vs corner contrast
        for (let y = 0; y < 2; y++)
            for (let x = 0; x < 2; x++) {
                const i = (y * 2 + x) * 4;
                const isCenter = x === 1 && y === 1;
                png.data[i] = isCenter ? 255 : 0;
                png.data[i + 1] = 0;
                png.data[i + 2] = 0;
                png.data[i + 3] = 255;
            }
        const shot = PNG.sync.write(png) as Buffer;

        let evaluateCalled = false;
        const page = mkPage(
            async () => shot,
            async () => {
                evaluateCalled = true;
                return null;
            },
        );
        const sample = await sampleFrameNode(page as never);
        expect(sample).not.toBeNull();
        // pngjs decoded in Node — page.evaluate was never reached
        expect(evaluateCalled).toBe(false);
    });

    test("fallback path: when pngjs decode fails, falls back to page.evaluate(decodeSample) (non-attribution)", async () => {
        // Simulate the pngjs-unavailable path by making the screenshot succeed but the pngjs decode
        // fail: `sampleFrameNode` catches the pngjs error and falls back to `page.evaluate(decodeSample, ...)`.
        // We feed a non-PNG buffer that `PNG.sync.read` will reject, forcing the catch → fallback path.
        // Under non-attribution (the default), the fallback is silent and fine — the plain verify path
        // uses `sampleFrame` directly, but a non-attribution caller of `sampleFrameNode` still gets a
        // working boot-settle signal.
        const badShot = Buffer.from("not-a-png");
        let evaluateCalled = false;
        let evaluateArg = "";
        const page = mkPage(
            async () => badShot,
            async (_fn: (...a: unknown[]) => unknown, b64: unknown) => {
                evaluateCalled = true;
                evaluateArg = b64 as string;
                // simulate decodeSample's return shape — a structured frame
                return {
                    grid: [255, 0, 0],
                    center: [255, 0, 0],
                    corner: [0, 0, 0],
                } as FrameSample;
            },
        );
        const sample = await sampleFrameNode(page as never, false);
        // the fallback fired — page.evaluate was called with the base64-encoded screenshot
        expect(evaluateCalled).toBe(true);
        expect(evaluateArg).toBe(badShot.toString("base64"));
        // and it returned the page-side decode result
        expect(sample).not.toBeNull();
        expect(sample?.center).toEqual([255, 0, 0]);
    });

    test("fallback path: when both pngjs and page.evaluate fail, returns null (non-attribution)", async () => {
        const badShot = Buffer.from("not-a-png");
        const page = mkPage(
            async () => badShot,
            async () => {
                throw new Error("page closed");
            },
        );
        expect(await sampleFrameNode(page as never, false)).toBeNull();
    });

    test("attribution path: pngjs decode failure is FATAL (throws, no silent fallback)", async () => {
        // Under --attribution the silent fallback to page.evaluate(decodeSample) is forbidden — it IS
        // the contamination this stage removes. A pngjs failure must throw, not silently fall back.
        const badShot = Buffer.from("not-a-png");
        let evaluateCalled = false;
        const page = mkPage(
            async () => badShot,
            async () => {
                evaluateCalled = true;
                return null;
            },
        );
        await expect(sampleFrameNode(page as never, true)).rejects.toThrow(
            /pngjs decode failed under --attribution/,
        );
        // the forbidden fallback was NOT taken — page.evaluate was never called
        expect(evaluateCalled).toBe(false);
    });
});

// shallot-boot-stall-repair S1: the boot wait poll's routing seam. `drive` itself (the loop's real
// caller, ~line 2161) drives a live `page.goto` + CDP session no unit test mocks, so `pollFrameSample`
// is the extracted pure function the loop now calls (checks.md's "extract it into a pure sibling
// module" escape) — these arms are the oracle over THAT seam, not a re-test of `sampleFrameNode` in
// isolation (already covered above): they prove the poll's non-attribution consumer takes the Node
// decode, not just `--attribution`'s.
describe("pollFrameSample — S1's boot wait poll routing seam", () => {
    type ScreenshotResult = Buffer | { toString(encoding: string): string };
    const mkPage = (
        screenshotFn: () => Promise<ScreenshotResult>,
        evaluateFn?: (fn: (...a: unknown[]) => unknown, ...args: unknown[]) => Promise<unknown>,
    ) => ({
        locator: () => ({
            first: () => ({
                screenshot: async () => screenshotFn(),
            }),
        }),
        evaluate: evaluateFn ?? (async () => null),
    });

    test("harnessDefined short-circuits to null without touching the page at all", async () => {
        let screenshotCalled = false;
        const page = mkPage(async () => {
            screenshotCalled = true;
            return Buffer.from("");
        });
        const sample = await pollFrameSample(page as never, true, false);
        expect(sample).toBeNull();
        expect(screenshotCalled).toBe(false);
    });

    test("non-attribution poll takes the Node decode — pngjs succeeds, page.evaluate never runs", async () => {
        const { PNG } = await import("pngjs");
        const png = new PNG({ width: 2, height: 2 });
        for (let y = 0; y < 2; y++)
            for (let x = 0; x < 2; x++) {
                const i = (y * 2 + x) * 4;
                const isCenter = x === 1 && y === 1;
                png.data[i] = isCenter ? 255 : 0;
                png.data[i + 1] = 0;
                png.data[i + 2] = 0;
                png.data[i + 3] = 255;
            }
        const shot = PNG.sync.write(png) as Buffer;
        let evaluateCalled = false;
        const page = mkPage(
            async () => shot,
            async () => {
                evaluateCalled = true;
                return null;
            },
        );
        // attribution=false — the plain `verify` path's own poll call shape.
        const sample = await pollFrameSample(page as never, false, false);
        expect(sample).not.toBeNull();
        expect(evaluateCalled).toBe(false);
    });

    test("non-attribution poll still falls back silently when the pngjs decode fails (malformed PNG buffer)", async () => {
        const badShot = Buffer.from("not-a-png");
        let evaluateCalled = false;
        const page = mkPage(
            async () => badShot,
            async (_fn: (...a: unknown[]) => unknown, b64: unknown) => {
                evaluateCalled = true;
                expect(b64).toBe(badShot.toString("base64"));
                return { grid: [1], center: [255, 0, 0], corner: [0, 0, 0] } as FrameSample;
            },
        );
        const sample = await pollFrameSample(page as never, false, false);
        expect(evaluateCalled).toBe(true);
        expect(sample).not.toBeNull();
        expect(sample?.center).toEqual([255, 0, 0]);
    });

    test("attribution poll still throws on a pngjs failure (no silent fallback)", async () => {
        const badShot = Buffer.from("not-a-png");
        let evaluateCalled = false;
        const page = mkPage(
            async () => badShot,
            async () => {
                evaluateCalled = true;
                return null;
            },
        );
        await expect(pollFrameSample(page as never, false, true)).rejects.toThrow(
            /pngjs decode failed under --attribution/,
        );
        expect(evaluateCalled).toBe(false);
    });
});

// S1c Finding 2: HARNESS_INPAGE_FUNCTION_NAMES is derived from the function objects' own .name
// properties, not hardcoded strings — so a rename updates the set automatically. This test makes the
// omission of a new in-page helper LOUD: it scans verify.ts for every NAMED function passed to
// page.evaluate (not anonymous arrows — those are the known residual) and asserts each is registered
// in HARNESS_INPAGE_FUNCTION_NAMES. A future in-page page.evaluate helper that isn't added to the set
// fails this test, preventing the arm from being bypassed.
describe("HARNESS_INPAGE_FUNCTION_NAMES — registration coverage for every named page.evaluate function", () => {
    const verifySrc = readFileSync(resolve(import.meta.dir, "verify.ts"), "utf8");
    // find every `page.evaluate(identifierName,` or `page.evaluate(identifierName)` where identifierName
    // is a bare identifier (not an arrow `() =>` or `async () =>` — those are anonymous, the known residual)
    const namedEvaluateRe = /\.evaluate\(\s*(?!async\s*\(|\(?\s*\)\s*=>)([A-Za-z_$][\w$]*)/g;
    const found = new Set<string>();
    let m: RegExpExecArray | null;
    while ((m = namedEvaluateRe.exec(verifySrc)) !== null) {
        found.add(m[1]);
    }

    test("every named function passed to page.evaluate in verify.ts is registered in HARNESS_INPAGE_FUNCTION_NAMES", () => {
        // HARNESS_INPAGE_FUNCTION_NAMES is the closed set the arm checks against. Every named function
        // verify.ts serializes into the page must be here, or the arm is bypassable by omission.
        const unregistered = [...found].filter((name) => !HARNESS_INPAGE_FUNCTION_NAMES.has(name));
        expect(unregistered).toEqual([]);
    });

    test("HARNESS_INPAGE_FUNCTION_NAMES is non-empty and contains decodeSample and decodeRgba", () => {
        expect(HARNESS_INPAGE_FUNCTION_NAMES.size).toBeGreaterThan(0);
        expect(HARNESS_INPAGE_FUNCTION_NAMES.has("decodeSample")).toBe(true);
        expect(HARNESS_INPAGE_FUNCTION_NAMES.has("decodeRgba")).toBe(true);
    });
});

describe("loafByCompilePhase — pre-compile LoAF attribution", () => {
    const script = (over: Partial<LoAFScriptEntry> = {}): LoAFScriptEntry => ({
        startTime: 0,
        duration: 0,
        executionStart: undefined,
        invoker: "",
        invokerType: "classic-script",
        sourceURL: "https://host/src/a.ts",
        sourceFunctionName: "f",
        sourceCharPosition: undefined,
        forcedStyleAndLayoutDuration: undefined,
        pauseDuration: undefined,
        ...over,
    });
    const frame = (
        start: number,
        duration: number,
        scripts: LoAFScriptEntry[] = [],
    ): LoAFEntry => ({
        start,
        duration,
        blockingDuration: Math.max(0, duration - 50),
        renderStart: undefined,
        styleAndLayoutStart: undefined,
        scripts,
    });
    const measure = (name: string, startTime: number, duration = 10) => ({
        name,
        startTime,
        duration,
    });

    test("splits entries around the FIRST prefix-matching measure, not the earliest measure overall", () => {
        const entries = [frame(0, 100), frame(400, 100), frame(1000, 100)];
        const measures = [
            measure("nav", 10),
            measure("pipeline-compile:b", 900),
            measure("pipeline-compile:a", 500),
        ];
        const report = loafByCompilePhase(entries, measures, "pipeline-compile:");
        expect(report.boundaryMs).toBe(500);
        expect(report.before.map((e) => e.start)).toEqual([0, 400]);
        expect(report.after.map((e) => e.start)).toEqual([1000]);
        expect(report.straddling).toEqual([]);
        expect(report.unclassified).toEqual([]);
    });

    test("a frame spanning the boundary is straddling, claimed by neither side", () => {
        const report = loafByCompilePhase([frame(450, 100)], [measure("pc:x", 500)], "pc:");
        expect(report.before).toEqual([]);
        expect(report.after).toEqual([]);
        expect(report.straddling.map((e) => e.start)).toEqual([450]);
    });

    test("boundary edges are inclusive on both sides: end == boundary is before, start == boundary is after", () => {
        const report = loafByCompilePhase(
            [frame(400, 100), frame(500, 100)],
            [measure("pc:x", 500)],
            "pc:",
        );
        expect(report.before.map((e) => e.start)).toEqual([400]);
        expect(report.after.map((e) => e.start)).toEqual([500]);
        expect(report.straddling).toEqual([]);
    });

    test("no prefix-matching measure leaves everything unclassified rather than claiming a pre-compile side", () => {
        const report = loafByCompilePhase([frame(0, 100)], [measure("nav", 10)], "pc:");
        expect(report.boundaryMs).toBeNull();
        expect(report.unclassified.map((e) => e.start)).toEqual([0]);
        expect(report.before).toEqual([]);
        expect(report.after).toEqual([]);
        expect(report.beforeScripts).toEqual([]);
        expect(report.beforeAttributedMs).toBe(0);
        expect(report.beforeUnattributedMs).toBe(0);
    });

    test("rolls up pre-compile scripts by source identity, descending, merging across frames", () => {
        const entries = [
            frame(0, 300, [
                script({ duration: 200, sourceFunctionName: "codegen" }),
                script({
                    duration: 40,
                    sourceFunctionName: "boot",
                    sourceURL: "https://host/src/b.ts",
                }),
            ]),
            frame(310, 120, [script({ duration: 90, sourceFunctionName: "codegen" })]),
            // after the boundary — must not appear in the rollup
            frame(900, 200, [script({ duration: 180, sourceFunctionName: "drain" })]),
        ];
        const report = loafByCompilePhase(entries, [measure("pc:x", 800)], "pc:");
        expect(report.beforeScripts.map((s) => [s.sourceFunctionName, s.totalMs, s.count])).toEqual(
            [
                ["codegen", 290, 2],
                ["boot", 40, 1],
            ],
        );
        expect(report.beforeScripts[0].key).toBe("codegen@https://host/src/a.ts");
    });

    test("the same function name in two files stays two rows", () => {
        const report = loafByCompilePhase(
            [
                frame(0, 200, [
                    script({ duration: 100, sourceURL: "https://host/src/a.ts" }),
                    script({ duration: 60, sourceURL: "https://host/src/b.ts" }),
                ]),
            ],
            [measure("pc:x", 800)],
            "pc:",
        );
        expect(report.beforeScripts).toHaveLength(2);
    });

    test("empty source fields fall back to readable placeholders rather than an empty key", () => {
        const report = loafByCompilePhase(
            [frame(0, 200, [script({ duration: 100, sourceFunctionName: "", sourceURL: "" })])],
            [measure("pc:x", 800)],
            "pc:",
        );
        expect(report.beforeScripts[0].key).toBe("(anonymous)@(no source url)");
    });

    test("unattributed pre-compile delay is reported, never folded into the attributed total", () => {
        const report = loafByCompilePhase(
            [frame(0, 500, [script({ duration: 120 })]), frame(600, 100)],
            [measure("pc:x", 800)],
            "pc:",
        );
        expect(report.beforeAttributedMs).toBe(120);
        expect(report.beforeUnattributedMs).toBe(480);
    });

    test("script durations exceeding the frame duration floor the unattributed remainder at zero", () => {
        const report = loafByCompilePhase(
            [frame(0, 100, [script({ duration: 140 })])],
            [measure("pc:x", 800)],
            "pc:",
        );
        expect(report.beforeUnattributedMs).toBe(0);
    });
});
