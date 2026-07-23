import { describe, expect, test } from "bun:test";
import { SCENARIO_TIMEOUTS } from "../../../examples/gym/src/scenarios/timeouts";
import { benchTimeout } from "../../../scripts/bench";
import {
    bootArm,
    buildUrl,
    coerceVerdict,
    type FrameSample,
    fitMemory,
    gridDiff,
    harnessPass,
    hasStructure,
    LEAK_BYTES_PER_SEC,
    type MemorySample,
    parseVerifyArgs,
    settlePass,
    stepWait,
    structured,
    type WaitState,
    withTimeout,
} from "./verify";

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
        expect(benchTimeout("stress")).toBe(SCENARIO_TIMEOUTS.stress);
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
