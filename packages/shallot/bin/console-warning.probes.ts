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
// LIVENESS WITNESS: the absence assertion (`warnings` is empty) is vacuous when `consoleMessages` is
// empty — a page that never booted, a renamed `ready` signal, or a console listener that attached after
// the messages fired all produce an empty capture that passes having tested nothing. So a non-zero
// console-message floor is asserted before the Dawn-string check: if the capture saw no traffic at all,
// the arm fails rather than letting the absence assertion ride on a broken precondition. A real boot of
// the `render` scenario emits at least Vite's HMR connection messages (`[vite] connecting…`,
// `[vite] connected.`) in the console, so a zero count is never a legitimate pass state.
//
// ADAPTER GATE: Dawn emits the zero-count warning only on a real Metal path; on a software rasterizer
// (SwiftShader, llvmpipe, lavapipe, warp) the engine boots, sees no warning, and passes having tested
// nothing. So `adapter.info` is probed after `page.goto` and BEFORE any boot wait — a software adapter
// throws immediately rather than burning the wait's full timeout. The adapter is logged on both paths
// (proceed and software-skip), and the match is deliberately narrow (a name list, not a capability
// probe) so an unlisted software adapter re-crashes loudly while an over-broad pattern would skip on
// real hardware and report green having tested nothing. This reuses the shape the showcase gates
// already ship (`examples/showcase/{voxel,roads}/test/*.spec.ts`); bun:test has no runtime
// `test.skip()`, so the software-skip path throws — stricter than a skip, but the adapter is logged
// either way and the test never passes green on a software rasterizer.
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

// Software rasterizers by the name they report in `GPUAdapterInfo`: Chromium's SwiftShader, Mesa's two,
// and D3D's WARP. Deliberately a name list rather than a capability probe — SwiftShader clears shallot's
// whole base floor (all three `BASE_FEATURES` plus 10 storage buffers per stage, measured under WSL
// 2026-08-10), so nothing about the floor distinguishes it, and it is only the *execution* that dies.
// Bias narrow: an unlisted software adapter re-crashes loudly, while an over-broad pattern would skip
// this gate on real hardware and report green having tested nothing. Reuses the same regex the
// showcase gates ship (`examples/showcase/{voxel,roads}/test/*.spec.ts`).
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

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

/** the adapter's self-reported identity, or `""` when the browser hands out none. */
function adapterName(page: import("playwright").Page): Promise<string> {
    return page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "";
        const { vendor, architecture, device, description } = adapter.info;
        return [vendor, architecture, device, description].filter(Boolean).join(" ");
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

                    // ADAPTER GATE — probe before any boot wait, or a software rasterizer burns the
                    // wait's full timeout before skipping. Dawn emits the zero-count warning only on a
                    // real Metal path; on SwiftShader the engine boots, sees no warning, and passes
                    // having tested nothing. Log the adapter on both paths so the skip is never silent.
                    const adapter = await adapterName(page);
                    console.log(`console-warning gate adapter: ${adapter || "none offered"}`);
                    if (adapter === "" || SOFTWARE.test(adapter)) {
                        throw new Error(
                            `console-warning gate: no real-GPU adapter (${adapter || "none offered"}) — ` +
                                "Dawn's zero-count warning is Metal-only, so a software rasterizer " +
                                "would pass green having tested nothing",
                        );
                    }

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

                    // LIVENESS WITNESS — a non-zero console-message floor proves the capture saw
                    // traffic from a real boot. Without it, an empty `consoleMessages` (the page never
                    // booted, the listener attached late) would make the absence assertion vacuous.
                    expect(consoleMessages.length).toBeGreaterThan(0);

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
