import { describe, expect, test } from "bun:test";
import { createServer as createTcpServer } from "node:net";
import { resolve } from "node:path";
import { createServer } from "vite";
import { REAL_GPU_LAUNCH } from "../src/harness/browser";
import { CROSS_ORIGIN_ISOLATION } from "../src/project/vite";

// Criterion 2 arm: asserts Dawn's zero-count dispatch warning never appears in a real boot's captured
// console. The warning is `DispatchWorkgroups with a workgroup count of 0 is unusual`, emitted by
// Dawn's `EmitWarningOnce` inside `ComputePassEncoder::APIDispatchWorkgroups` when a compute pass
// dispatches zero workgroups — the engine's old warm idiom (a zero-workgroup dispatch to force pipeline
// compilation). S1 replaced that idiom with `initAsync()`, so the warning should be absent at S1 and
// present at the parent commit `8e5a092` where the dispatch idiom still exists.
//
// The verify protocol's `attachErrorCapture` (bin/verify.ts:1144) already wires `page.on("console")`
// reading `msg.type()`, but a `warning`-type message does NOT survive into the result's `errors` array
// (the filter pushes only `err`, `page-error`, and `ERR_HINT` matches). So this arm listens on the page
// console directly rather than growing the protocol — the sanctioned approach per the spec.
//
// RED-FIRST WITNESS (at 8e5a092, the parent commit where the dispatch idiom still exists):
//   Run in a worktree at 8e5a092 after `bun install && bun run build`:
//     bun test ./packages/shallot/bin/console-warning.probes.ts
//   Result: FAIL (exit 1) — 12 `warning`-type console messages carrying the Dawn string appeared
//   during a real boot of the `render` scenario, one per zero-workgroup dispatch site:
//     slab-scatter-vec4<f32>, slab-scatter-f32, slab-scatter-u32, slab-scatter-vec2<u32>,
//     shallot-transforms-compose, shallot-cluster-aabbs, shallot-light-compact,
//     shallot-light-cull, shallot-part-count, shallot-part-scan, shallot-part-scatter, glaze.
//   At 4bd63df (S1, where the dispatch idiom was replaced by initAsync): PASS (exit 0, 0 warnings).
//   Witnessed 2026-08-24 on darwin/arm64 (M-series Metal), headless Chromium with REAL_GPU_LAUNCH.
//
// This is a `.probes.ts` file (browser-launching gate, too slow for the default suite). Run by path:
//     bun test ./packages/shallot/bin/console-warning.probes.ts

const REPO_ROOT = resolve(import.meta.dir, "..", "..", "..");
const GYM_DIR = resolve(REPO_ROOT, "examples", "gym");

/** the Dawn warning string the old zero-workgroup dispatch idiom trips. */
const DAWN_ZERO_WARNING = "DispatchWorkgroups with a workgroup count of 0 is unusual";

let hasChromium = false;
try {
    require.resolve("playwright");
    hasChromium = true;
} catch {
    hasChromium = false;
}

/** pick a free port the same way verify.ts does. */
function pickPort(): Promise<number> {
    return new Promise((res, rej) => {
        const s = createTcpServer();
        s.on("error", rej);
        s.listen(0, "127.0.0.1", () => {
            const p = (s.address() as { port: number }).port;
            s.close(() => res(p));
        });
    });
}

describe("criterion 2 — Dawn zero-count warning absent from a real boot", () => {
    test.skipIf(!hasChromium)(
        "the `render` scenario boots without DispatchWorkgroups(0) warning in the console",
        async () => {
            const { chromium } = await import("playwright");

            // boot the gym's ejected vite dev server — the same mechanism verify.ts's `serveEjected` uses
            const port = await pickPort();
            const server = await createServer({
                root: GYM_DIR,
                server: { port, strictPort: true, open: false, headers: CROSS_ORIGIN_ISOLATION },
            });
            await server.listen();
            const url = server.resolvedUrls?.local?.[0] ?? `http://localhost:${port}/`;

            try {
                const browser = await chromium.launch({ headless: true, ...REAL_GPU_LAUNCH });
                try {
                    const context = await browser.newContext();
                    const page = await context.newPage();

                    // capture ALL console messages — not just errors. The verify protocol's
                    // attachErrorCapture filters to errors + ERR_HINT matches, so a `warning`-type
                    // message does not survive into the result. This arm listens directly.
                    const consoleMessages: { type: string; text: string }[] = [];
                    page.on("console", (msg) => {
                        consoleMessages.push({ type: msg.type(), text: msg.text() });
                    });

                    // boot the render scenario — it exercises the full warm path (every pipeline
                    // registered via precompile) and is the scenario the boot-cost instrument uses
                    await page.goto(`${url}?scenario=render`, { timeout: 30_000 });

                    // wait for the harness to signal the engine has booted and the scenario's build
                    // completed — the same signal verify.ts waits for (window.__harness with ready: true)
                    await page.waitForFunction(
                        () =>
                            typeof window.__harness !== "undefined" &&
                            window.__harness !== null &&
                            (window.__harness as { ready?: boolean }).ready === true,
                        null,
                        { timeout: 30_000 },
                    );

                    // give the engine a few more frames to settle — the warm drain runs during build,
                    // but any late-registered pipelines drain on arrival and may still be compiling
                    await page.waitForTimeout(2000);

                    // assert the Dawn zero-count warning never appeared in any console message
                    const warnings = consoleMessages.filter((m) =>
                        m.text.includes(DAWN_ZERO_WARNING),
                    );
                    expect(warnings).toBeEmpty();
                } finally {
                    await browser.close();
                }
            } finally {
                await server.close();
            }
        },
        60_000,
    );
});
