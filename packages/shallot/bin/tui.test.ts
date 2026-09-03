import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import {
    buildDisposeAll,
    cellsBytesToGrid,
    createQuitGuard,
    decodeStdinChunk,
    EXIT_NO_BUN_WEBGPU,
    EXIT_SETUP,
    importBunWebgpu,
    noBunWebgpuMessage,
    parseTuiArgs,
    runLoopWithTeardown,
    runTui,
    type TuiArgs,
    terminalGridSize,
    usage,
} from "./tui";
import { makeGrid } from "./tui/index";

// This file stays in the default (fast, no-GPU) `.test.ts` tier — every arm here is pure or DI-driven,
// no subprocess, no real bun-webgpu device. Criterion 6's real determinism oracle needs a genuine
// subprocess boot (~2.5s fixed cost per run, independent of frame count — measured, dominated by GPU
// pipeline compilation, not something a smaller `--frames` shrinks), which blows the shared per-file
// test-duration cap (`tests/test-cap.ts`, 5000ms) at just two runs. Promoted to `bin/tui.probes.ts`,
// mirroring `bin/verify.probes.ts`'s own precedent for exactly this class of check ("browser probes stay
// out of the default suite for speed... Run when you touch the probes") — run it directly:
//
//     bun test ./packages/shallot/bin/tui.probes.ts
const REPO_ROOT = resolve(import.meta.dir, "../../..");
const RECIPE_DIR = resolve(REPO_ROOT, "examples/recipes/render-to-a-terminal");

describe("terminalGridSize", () => {
    test("uses the terminal dimensions until the measured bandwidth ceiling", () => {
        expect(terminalGridSize({ width: 120, height: 40 })).toEqual({ width: 120, height: 40 });
        expect(terminalGridSize({ width: 320, height: 90 })).toEqual({ width: 200, height: 50 });
    });
});

describe("parseTuiArgs", () => {
    test("bare invocation defaults dir to '.', fps to 30, no frames bound", () => {
        expect(parseTuiArgs([])).toEqual<TuiArgs>({ dir: ".", fps: 30, help: false });
    });

    test("a bare dir, --frames, --fps, and --tier all parse, long and = forms alike", () => {
        expect(parseTuiArgs(["proj", "--frames", "10", "--fps", "24", "--tier", "plain"])).toEqual({
            dir: "proj",
            frames: 10,
            fps: 24,
            tier: "plain",
            help: false,
        });
        expect(parseTuiArgs(["--frames=5", "--fps=60", "--tier=truecolor"])).toEqual({
            dir: ".",
            frames: 5,
            fps: 60,
            tier: "truecolor",
            help: false,
        });
    });

    test("--help / -h short-circuits, even with other flags present", () => {
        expect(parseTuiArgs(["--help"]).help).toBe(true);
        expect(parseTuiArgs(["proj", "-h", "--frames", "1"]).help).toBe(true);
    });

    test("an unrecognized -flag throws rather than silently falling through", () => {
        expect(() => parseTuiArgs(["--nope"])).toThrow("unknown option: --nope");
    });

    test("--frames/--fps reject non-positive, non-integer, and empty values", () => {
        expect(() => parseTuiArgs(["--frames", "0"])).toThrow('invalid --frames value "0"');
        expect(() => parseTuiArgs(["--fps", "-1"])).toThrow('invalid --fps value "-1"');
        expect(() => parseTuiArgs(["--frames", "1.5"])).toThrow("expected an integer");
        // the `=` form, not a space-separated empty string — `raw[i + 1]` truthy-checks its value (the
        // same shape `parseCliArgs`'s own `--port`/`--target` use), so a space-separated empty string
        // reads as "no value given" and falls through to "unknown option" rather than this branch; every
        // sibling parser's own empty-value test uses the `=` form for the same reason (cli.test.ts's
        // `--port=`).
        expect(() => parseTuiArgs(["--frames="])).toThrow("must not be empty");
    });

    test("--tier rejects a value outside the four portability tiers", () => {
        expect(() => parseTuiArgs(["--tier", "rainbow"])).toThrow('invalid --tier value "rainbow"');
    });

    test("a second positional argument throws rather than being silently dropped", () => {
        expect(() => parseTuiArgs(["a", "b"])).toThrow("unexpected argument: b");
    });
});

describe("usage", () => {
    test("names the bun-webgpu install remedy, so --help itself documents the optional dependency", () => {
        expect(usage).toContain("bun add -d bun-webgpu");
    });
});

describe("decodeStdinChunk", () => {
    test("decodes the four CSI arrow-key sequences to their KeyboardEvent.code values", () => {
        expect(decodeStdinChunk("\x1b[A").codes).toEqual(["ArrowUp"]);
        expect(decodeStdinChunk("\x1b[B").codes).toEqual(["ArrowDown"]);
        expect(decodeStdinChunk("\x1b[C").codes).toEqual(["ArrowRight"]);
        expect(decodeStdinChunk("\x1b[D").codes).toEqual(["ArrowLeft"]);
    });

    test("Ctrl-C (\\x03) and 'q'/'Q' both signal quit, without also reading as a stray code", () => {
        expect(decodeStdinChunk("\x03")).toEqual({ codes: [], quit: true });
        expect(decodeStdinChunk("q")).toEqual({ codes: [], quit: true });
        expect(decodeStdinChunk("Q")).toEqual({ codes: [], quit: true });
    });

    test("a chunk carrying more than one sequence decodes every one, in order", () => {
        expect(decodeStdinChunk("\x1b[A\x1b[A\x1b[D")).toEqual({
            codes: ["ArrowUp", "ArrowUp", "ArrowLeft"],
            quit: false,
        });
    });

    test("an unrecognized escape sequence and plain text produce no codes and no quit", () => {
        expect(decodeStdinChunk("\x1b[Z")).toEqual({ codes: [], quit: false });
        expect(decodeStdinChunk("hello")).toEqual({ codes: [], quit: false });
    });
});

// Regression for the blocker diagnosed against runTui's own q/Ctrl-C quit path: `stopped` used to be one
// boolean serving two meanings ("the frame loop should stop" and "disposeAll already ran"), so a quit
// request set it directly and disposeAll's own idempotency guard read it as already-disposed before
// disposeAll ever ran once — stdinBridge.stop() (raw mode off), held-key timer clearing, and
// app.dispose() all silently no-op'd. Before this type existed, the reproduction of that exact shared-
// flag shape failed identically to this: requesting a stop first, then calling the dispose sequence,
// left it un-run. `createQuitGuard` is the fix — `requestStop()` and `disposeOnce()` are independent —
// and this test exercises the real production type, not a copy.
//
// This block alone only proves `QuitGuard`'s own contract in isolation, though: it doesn't reach the
// *composition* the original bug actually lived in (one flag serving both the loop's stop condition and
// disposeAll's guard), so a future edit that rewires `runTui`'s own `disposeAll` to read
// `quitGuard.stopped` directly — bypassing `disposeOnce` entirely, re-collapsing the two facts one layer
// up from `QuitGuard` itself — would stay green here. The `buildDisposeAll` block right below reaches
// that composition: it's the exact function `runTui` calls to build its own `disposeAll`, not a copy.
describe("createQuitGuard — requestStop and disposeOnce are independent (regression: q/Ctrl-C quit skipped teardown)", () => {
    test("requesting a stop first still lets disposeOnce run its callback", () => {
        const guard = createQuitGuard();
        guard.requestStop(); // simulates the stdin bridge's onQuit ('q'/Ctrl-C)
        let disposed = false;
        guard.disposeOnce(() => {
            disposed = true;
        });
        expect(disposed).toBe(true);
        expect(guard.stopped).toBe(true);
    });

    test("disposeOnce runs its callback exactly once across repeated calls", () => {
        const guard = createQuitGuard();
        let calls = 0;
        guard.disposeOnce(() => {
            calls++;
        });
        guard.disposeOnce(() => {
            calls++;
        });
        expect(calls).toBe(1);
    });

    test("stopped stays false until requestStop is called, and requestStop is itself idempotent", () => {
        const guard = createQuitGuard();
        expect(guard.stopped).toBe(false);
        guard.requestStop();
        guard.requestStop();
        expect(guard.stopped).toBe(true);
    });
});

// `buildDisposeAll` is the actual production wiring — `runTui` calls it to build its own `disposeAll`,
// not a hand-copied stand-in — so this reaches the composition the block above can't: a future edit
// reading `quitGuard.stopped` here instead of calling `quitGuard.disposeOnce` would re-collapse the two
// independent facts one layer above `QuitGuard` itself, and the test below reproduces the exact original
// symptom (a quit request, then the dispose sequence) through this exact function.
describe("buildDisposeAll — the real runTui composition, not just QuitGuard in isolation (regression: q/Ctrl-C quit skipped teardown)", () => {
    test("requesting a stop first still lets the built disposeAll run its dispose callback", () => {
        const guard = createQuitGuard();
        guard.requestStop(); // simulates the stdin bridge's onQuit ('q'/Ctrl-C)
        let disposed = false;
        const disposeAll = buildDisposeAll(guard, () => {
            disposed = true;
        });
        disposeAll();
        expect(disposed).toBe(true);
    });

    test("the built disposeAll runs its dispose callback exactly once across repeated calls", () => {
        const guard = createQuitGuard();
        let calls = 0;
        const disposeAll = buildDisposeAll(guard, () => {
            calls++;
        });
        disposeAll();
        disposeAll();
        expect(calls).toBe(1);
    });
});

// Regression for the throw-mid-loop finding (S3/S4 batch review round 2): before `runLoopWithTeardown`
// existed, `teardown()`/`disposeAll()` sat inline after the frame loop as a normal-completion-only step
// — a throw from inside the loop (app.state.step, cellsGridFor, staging.mapAsync, encoder.encode)
// propagated straight out of `runTui`, skipping disposeAll entirely: stdinBridge.stop() never ran, so
// setRawMode(false) was never called and held-key timers leaked, the same user-visible symptom the
// q/Ctrl-C createQuitGuard fix removed through a different door. `installTeardown`'s own `"exit"`
// listener doesn't cover this either — it only restores the alt screen on process exit, never calling
// the disposal work (screen.ts's own docblock: `opts.exit` is never invoked for `"exit"`). This is the
// real function `runTui` calls to run its own frame loop, not a copy.
describe("runLoopWithTeardown — teardown/disposeAll run even when the loop throws (regression: a throw mid-loop skipped disposeAll)", () => {
    test("teardown and disposeAll both run when the loop throws, and the throw still propagates", async () => {
        const calls: string[] = [];
        await expect(
            runLoopWithTeardown(
                async () => {
                    calls.push("loop");
                    throw new Error("boom");
                },
                () => calls.push("teardown"),
                () => calls.push("disposeAll"),
            ),
        ).rejects.toThrow("boom");
        expect(calls).toEqual(["loop", "teardown", "disposeAll"]);
    });

    test("teardown and disposeAll both run on normal completion too", async () => {
        const calls: string[] = [];
        await runLoopWithTeardown(
            async () => {
                calls.push("loop");
            },
            () => calls.push("teardown"),
            () => calls.push("disposeAll"),
        );
        expect(calls).toEqual(["loop", "teardown", "disposeAll"]);
    });
});

describe("cellsBytesToGrid", () => {
    // a hand-built 2x1 grid over a trivial fake unpack/glyphChar pair — no GPU, no engine import,
    // proving the pure decode shape independent of the real cell packing.
    function fakeUnpack(_bytes: ArrayBuffer, index: number) {
        return index === 0
            ? { glyph: 1, fg: [255, 0, 0, 255], bg: [0, 0, 0, 255] }
            : { glyph: 2, fg: [0, 255, 0, 255], bg: [10, 20, 30, 255] };
    }
    const fakeGlyphChar = (glyph: number) => (glyph === 1 ? "#" : ".");

    test("decodes glyph + fg/bg (dropping alpha) into the private encoder Grid shape", () => {
        const grid = cellsBytesToGrid(
            new ArrayBuffer(0),
            2,
            1,
            fakeUnpack,
            fakeGlyphChar,
            makeGrid,
        );
        expect(grid).toEqual({
            width: 2,
            height: 1,
            cells: [
                [
                    { glyph: "#", fg: { r: 255, g: 0, b: 0 }, bg: { r: 0, g: 0, b: 0 } },
                    { glyph: ".", fg: { r: 0, g: 255, b: 0 }, bg: { r: 10, g: 20, b: 30 } },
                ],
            ],
        });
    });
});

describe("noBunWebgpuMessage", () => {
    test("names the install command and carries no stack-trace shape", () => {
        const msg = noBunWebgpuMessage();
        expect(msg).toContain("bun add -d bun-webgpu");
        expect(msg).not.toMatch(/\bat \S+\(.*:\d+:\d+\)/); // a "    at fn (file:line:col)" stack frame
        expect(msg).not.toContain("Cannot find module");
    });
});

// Criterion 7: "A missing bun-webgpu produces a named remedy" — two-sided by construction. A rejecting
// loader proves the exit code; a resolving loader (importBunWebgpu's own default, exercised for real by
// every tui.probes.ts subprocess run) proves the success path reaches the engine at all, so the catch
// isn't vacuously "always fails." Removing importBunWebgpu's try/catch (or letting the rejection
// propagate unhandled) reds this suite immediately — runTui would throw instead of returning
// EXIT_NO_BUN_WEBGPU. This layer's own standing check forbids mocking a module ("extract, never mock",
// cli-coverage.test.ts's "no test in this layer mocks a module"), so the printed-message half of
// criterion 7 is proven two other ways instead of a console spy here: noBunWebgpuMessage's own text is
// asserted directly above (pure, no I/O), and scripts/install-test.ts's real bun-webgpu-absence check —
// mirroring its existing playwright-absence rung — captures the genuine printed stderr from a real
// subprocess, never a mock.
describe("importBunWebgpu / runTui — missing bun-webgpu", () => {
    test("importBunWebgpu resolves via its DI'd loader, and returns null (never throws) on rejection", async () => {
        const ok = await importBunWebgpu(() =>
            Promise.resolve({ setupGlobals: async () => {} } as never),
        );
        expect(ok).not.toBeNull();

        const failing = await importBunWebgpu(() =>
            Promise.reject(new Error("Cannot find module 'bun-webgpu'")),
        );
        expect(failing).toBeNull();
    });

    test("runTui exits EXIT_NO_BUN_WEBGPU (never throws) when its bun-webgpu loader rejects", async () => {
        const code = await runTui([RECIPE_DIR], () =>
            Promise.reject(new Error("Cannot find module 'bun-webgpu'")),
        );
        expect(code).toBe(EXIT_NO_BUN_WEBGPU);
    });

    test("runTui rejects bad flags before ever reaching bun-webgpu (EXIT_SETUP, distinct code)", async () => {
        const code = await runTui(["--nope"]);
        expect(code).toBe(EXIT_SETUP);
    });
});
