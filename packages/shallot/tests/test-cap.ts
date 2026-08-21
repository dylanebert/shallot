/**
 * The per-file test-duration cap, enforced inside bun's own test runner via `[test] preload` in the
 * repo-root `bunfig.toml`. A default-tier `.test.ts` file whose tests run longer than the cap reds,
 * on `bun test <anything>` from the shallot root, with no wrapper to forget.
 *
 * Sibling to `tests/setup.ts` (WebGPU global setup) and `tests/tgsl.ts` (the TypeGPU transform
 * plugin, exactly one instance), appended to `bunfig.toml`'s preload chain — never inlined into
 * either, so each keeps its single responsibility.
 *
 * The cap is a tripwire, not a budget, and the no-gradient claim is about the **verdict**, not the
 * message: a 9 s file and a 5.1 s file are equally red, so the cap cannot compress a suite toward a
 * number. The message deliberately does carry a gradient — it prints the elapsed reading to 0.1 ms,
 * because that reading is the localization value. The verdict forces one of two moves, both named in
 * the throw's own message: derive the scan's size, or promote the file to a by-path tier. Raising the
 * cap and exempting a file in place are not responses, which is why nothing here reads a per-file
 * list and why `TEST_FILE_CAP_MS` exists only so `test-cap.test.ts` can drive the cap from a child
 * process.
 *
 * Exemption is derived from the path, never listed: `*.oracle.ts`, `*.probes.ts`, `*.tier.ts`, and
 * `*.lab.ts` are shallot's four by-path tiers, so they skip the cap by matching their own suffix.
 *
 * Mechanism, measured against bun 1.3.14 — and it is NOT the `beforeAll`/`afterAll` pair:
 *
 *   - A preload module is evaluated ONCE per `bun test` process, not once per test file, so
 *     root-level `beforeAll`/`afterAll` registered there fire once for the whole run. A cap built
 *     on them would measure the run, not the file.
 *   - `beforeEach`/`afterEach` registered in the preload DO fire per test, and `Bun.main` is
 *     the path of the test file currently executing, so a change in `Bun.main` is the file
 *     boundary. `expect.getState()` does not exist on bun 1.3.14.
 *   - A throw from `afterEach` reds that test and the run (exit 1) while sibling files stay green.
 *
 * Reach limit: the window starts at the first `beforeEach` of a file, which bun runs after that
 * file's module evaluation and its own `beforeAll`. So the cap measures a file's TESTS, not its
 * import cost or its fixture prologue. A file slow only in module evaluation or in
 * `beforeAll`/`afterAll` is invisible to it.
 *
 * Shallot's `bunfig.toml` carries `root = "packages/shallot"`, which scopes bun's test discovery
 * to that directory — a `.test.ts` file outside it is never collected by a bare `bun test`, only
 * by an explicit path. The cap still fires on any `bun test` whose cwd is the shallot root,
 * regardless of what paths are passed.
 */
import { afterEach, beforeEach } from "bun:test";

/** Milliseconds a single default-tier test file's tests may run. User-locked, 2026-08-21. */
export const DEFAULT_CAP_MS = 5000;

/** The by-path tier suffixes — a file naming its own tier, not an exemption list. */
const TIER_SUFFIXES = [".oracle.ts", ".probes.ts", ".tier.ts", ".lab.ts"];

/** true when `testPath` names a by-path tier and so is outside the default tier's cap. */
export function isCapExempt(testPath: string): boolean {
    return TIER_SUFFIXES.some((suffix) => testPath.endsWith(suffix));
}

/**
 * The cap in force. `TEST_FILE_CAP_MS` exists so the arms can drive the cap from a child
 * process; it is never a per-file escape, and nothing in the tree sets it. A malformed or
 * negative value resolves to the default rather than throwing — the conservative direction,
 * since the alternative is a typo in one arm killing every run.
 */
export function resolveCapMs(env: Record<string, string | undefined>): number {
    const raw = env.TEST_FILE_CAP_MS;
    if (raw === undefined || raw.trim() === "") return DEFAULT_CAP_MS;
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) return DEFAULT_CAP_MS;
    return parsed;
}

/**
 * The red's text: the file, its reading, the cap, and the only two responses to it.
 *
 * The actionable half prescribes acts a fenced seat can run: writing the tier file and taking the
 * arms out of this one. A by-path tier file's own header is its registry — read that header rather
 * than a registry.
 */
export function capMessage(testPath: string, elapsedMs: number, capMs: number): string {
    return [
        `per-file test cap exceeded: ${testPath} ran ${elapsedMs.toFixed(1)}ms of tests against a ${capMs}ms cap.`,
        "Two responses, and only these two:",
        "(1) derive the scan's size from the property it states — an arm whose own comment claims invariance in N has no reason to run at N;",
        "(2) promote the slow arms to a by-path tier — a sibling file named *.oracle.ts, *.probes.ts, *.tier.ts, or *.lab.ts, holding the arms and the reason in its header, run by path when the paths that header names change.",
    ].join("\n");
}

let currentFile = "";
let startedAt = 0;
let reported = false;

beforeEach(() => {
    if (Bun.main === currentFile) return;
    currentFile = Bun.main;
    startedAt = performance.now();
    reported = false;
});

afterEach(() => {
    if (reported || isCapExempt(currentFile)) return;
    const capMs = resolveCapMs(Bun.env);
    const elapsedMs = performance.now() - startedAt;
    if (elapsedMs <= capMs) return;
    // Once per file: N identical reds would say nothing the first one does not.
    reported = true;
    throw new Error(capMessage(relativeToCwd(currentFile), elapsedMs, capMs));
});

function relativeToCwd(absPath: string): string {
    const prefix = `${process.cwd()}/`;
    return absPath.startsWith(prefix) ? absPath.slice(prefix.length) : absPath;
}
