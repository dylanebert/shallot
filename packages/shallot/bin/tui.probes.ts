import { describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { EXIT_PASS } from "./tui";

// Criterion 6's real determinism red-proof ("the terminal command is deterministic into a pipe" —
// specs/shallot-tui.md), split out of `tui.test.ts` as its own by-path gate — a genuine subprocess boot
// under bun-webgpu costs ~2.5s fixed (measured: GPU device acquisition + pipeline compile, essentially
// independent of --frames), which blows the shared per-file test-duration cap (`tests/test-cap.ts`,
// 5000ms) at just two runs. Mirrors `bin/verify.probes.ts`'s own precedent exactly ("browser probes stay
// out of the default suite for speed... Run when you touch the probes"):
//
//     bun test ./packages/shallot/bin/tui.probes.ts
//
// The sentinel left behind in `tui.test.ts` is the pure/DI half — parseTuiArgs, decodeStdinChunk,
// cellsBytesToGrid, and runTui's own bun-webgpu-absence wiring via a rejecting DI'd loader (criterion 7,
// which needs no real device at all) — which pins the mechanism without a subprocess.

const REPO_ROOT = resolve(import.meta.dir, "../../..");
const CLI = resolve(import.meta.dir, "cli.ts");
const RECIPE_DIR = resolve(REPO_ROOT, "examples/recipes/render-to-a-terminal");

let hasBunWebgpu = false;
try {
    require.resolve("bun-webgpu");
    hasBunWebgpu = true;
} catch {
    hasBunWebgpu = false;
}

function runPiped(extra: string[] = []) {
    return spawnSync("bun", [CLI, "tui", RECIPE_DIR, "--frames", "4", "--fps", "30", ...extra], {
        cwd: REPO_ROOT,
        stdio: ["ignore", "pipe", "pipe"],
        timeout: 60_000,
    });
}

// a software-adapter seat (llvmpipe, dzn) still renders correctly (measured in this unit's own two-seat
// table, specs/shallot-tui.md) — only a seat offering no adapter at all should skip.
function skipReason(result: ReturnType<typeof runPiped>): string | null {
    if (result.status === EXIT_PASS) return null;
    const stderr = result.stderr?.toString() ?? "";
    if (/no.*adapter|NoBackendFound|device could not be created/i.test(stderr)) {
        return "no WebGPU adapter available on this seat";
    }
    return null;
}

describe("shallot tui — deterministic into a pipe (criterion 6, real subprocess)", () => {
    // Two-sided by construction: the hash is pinned against a second live run of the same command, not
    // against a hand-written constant — a run that changed its own output (a selection/encoder
    // regression) reds this the moment the two hashes diverge, and an accidentally-nondeterministic
    // command (real-time pacing, a Math.random seed, an object-iteration-order dependency) reds it
    // exactly as reliably. A hand-pinned golden hash was rejected: the engine's glyph selection/ramp is
    // still evolving (specs/shallot-tui.md's own Live log), so a hard-pinned byte constant would red on
    // every unrelated tuning pass rather than on the property this criterion actually names.
    test.skipIf(!hasBunWebgpu)(
        "two independent runs against the same recipe, piped non-tty, hash identically",
        () => {
            const a = runPiped();
            const skip = skipReason(a);
            if (skip) {
                console.log(`skipping: ${skip}`);
                return;
            }
            const b = runPiped();
            expect(a.status).toBe(EXIT_PASS);
            expect(b.status).toBe(EXIT_PASS);
            expect(a.stdout.length).toBeGreaterThan(0);
            const hash = (buf: Buffer) => createHash("sha256").update(buf).digest("hex");
            expect(hash(a.stdout)).toBe(hash(b.stdout));
            // non-tty stdin+stdout with no --tier override auto-detects `plain` — no cursor-addressing
            // escapes, no CLEAR_SCREEN/alt-screen bytes, exactly the content a pipe consumer (an agent
            // reading stdout as text) should see.
            expect(a.stdout.toString()).not.toContain("\x1b[");
        },
        30_000,
    );

    // The two-sided half required of every criterion-6-adjacent oracle here: an arm that would pass
    // vacuously on a broken pipeline (e.g. a command that always emits the same fixed bytes regardless of
    // tier) is not proving determinism, it's proving silence. Forcing a different --tier and asserting
    // the byte stream actually changes is what makes the "identical" reading above mean something.
    // **This arm alone does not exclude an all-blank grid.** A selector emitting glyph index 0 (space) for
    // every cell still passes both assertions below: plain renders as space/newline padding, byte-
    // identical across runs (arm 1 above), while ansi256 still wraps that same blank content in SGR runs
    // for its bg colour, so the two byte streams differ and `b` still contains `\x1b[` — determinism and
    // a tier-dependent byte stream, with zero glyph ink either way. The arm below closes that gap by
    // reading the plain-tier stdout directly for actual glyph content.
    test.skipIf(!hasBunWebgpu)(
        "a different --tier changes the byte stream (the pipe isn't emitting a fixed constant)",
        () => {
            const a = runPiped(["--tier", "plain"]);
            const skip = skipReason(a);
            if (skip) {
                console.log(`skipping: ${skip}`);
                return;
            }
            const b = runPiped(["--tier", "ansi256"]);
            expect(a.status).toBe(EXIT_PASS);
            expect(b.status).toBe(EXIT_PASS);
            expect(a.stdout.equals(b.stdout)).toBe(false);
            expect(b.stdout.toString()).toContain("\x1b["); // ansi256 carries SGR color escapes
        },
        30_000,
    );

    // Closes the vacuity gap the two arms above admit (see the comment on arm 2): neither excludes a
    // selector that emits glyph index 0 for every cell, since an all-blank grid still hashes identically
    // across runs and still differs byte-for-byte between plain and ansi256 (the bg colour escapes alone
    // do that). This arm reads the plain-tier stdout directly and requires actual glyph content — at
    // least one character outside {space, newline} (`\r` excluded too, though the encoder never emits
    // CRLF) — so an all-blank grid reds here even though it satisfies both arms above.
    test.skipIf(!hasBunWebgpu)(
        "the plain-tier stdout carries real glyph content, not an all-blank grid",
        () => {
            const a = runPiped(["--tier", "plain"]);
            const skip = skipReason(a);
            if (skip) {
                console.log(`skipping: ${skip}`);
                return;
            }
            expect(a.status).toBe(EXIT_PASS);
            const text = a.stdout.toString();
            const nonBlank = [...text].filter((c) => c !== " " && c !== "\n" && c !== "\r");
            expect(nonBlank.length).toBeGreaterThan(0);
        },
        30_000,
    );
});
