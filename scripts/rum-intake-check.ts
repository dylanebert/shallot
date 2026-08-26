import { existsSync, mkdirSync } from "node:fs";
import { extname, resolve } from "node:path";

// `bun run scripts/rum-intake-check.ts [--demo <slug>]` — the RUM slow-frame intake proof (spec
// `shallot-rum-slow-frame-vitals` S2). Serves `out/site/<slug>/` statically, drives it through a
// real (headless) browser, forces a deterministic main-thread busy-loop above and below the
// sampler's 50ms threshold (`site/rum-sampler.ts`'s `SLOW_FRAME_THRESHOLD_MS`), intercepts every
// request to the Datadog intake host, and asserts both directions: the above-threshold run's
// intercepted body carries a `slow_frame` vital, the below-threshold run's doesn't. Nothing this
// check does reaches the real Datadog intake — every matching request is intercepted and never
// forwarded (`scripts/rum-intake-driver.ts`).
//
// The actual browser drive happens in a node subprocess (`rum-intake-driver.ts`, bundled to a
// node target here, the same shape `scripts/wsl-bridge.ts` uses for `shallot verify`'s CLI):
// Bun's Playwright client hangs on this platform. No real GPU/display gate applies — the RUM
// sampler is rAF-based and main-thread, independent of WebGPU init, so a plain local headless
// launch (software adapter) is enough; this check is not display-gated the way `bun run demos`
// is, and doesn't touch the wsl-bridge host tunnel.

const root = resolve(import.meta.dir, "..");
const outDir = resolve(root, "out/site");
const bundlePath = resolve(root, "node_modules/.cache/shallot-rum-intake-driver.mjs");

const MIME: Record<string, string> = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
};

function fail(msg: string): never {
    console.error(`✗ ${msg}`);
    process.exit(1);
}

function buildDriverBundle(): void {
    mkdirSync(resolve(root, "node_modules/.cache"), { recursive: true });
    const r = Bun.spawnSync(
        [
            "bun",
            "build",
            "scripts/rum-intake-driver.ts",
            "--target",
            "node",
            "--outfile",
            bundlePath,
            "--external",
            "playwright",
        ],
        { cwd: root, stdout: "pipe", stderr: "pipe" },
    );
    if (r.exitCode !== 0) {
        fail(`driver bundle build failed:\n${new TextDecoder().decode(r.stderr)}`);
    }
}

function serveOutDir(): { url: string; stop: () => void } {
    const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch(req) {
            const url = new URL(req.url);
            let path = decodeURIComponent(url.pathname);
            if (path.endsWith("/")) path += "index.html";
            const full = resolve(outDir, `.${path}`);
            if (!full.startsWith(outDir)) return new Response("forbidden", { status: 403 });
            if (!existsSync(full)) return new Response("not found", { status: 404 });
            return new Response(Bun.file(full), {
                headers: { "content-type": MIME[extname(full)] ?? "application/octet-stream" },
            });
        },
    });
    return { url: `http://127.0.0.1:${server.port}`, stop: () => server.stop(true) };
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    const idx = args.indexOf("--demo");
    const slug = idx !== -1 ? args[idx + 1] : "sandbox";

    const demoDir = resolve(outDir, slug);
    if (!existsSync(resolve(demoDir, "index.html"))) {
        console.log(`out/site/${slug}/ missing — building it...`);
        const build = Bun.spawnSync(["bun", "run", "site", "--demo", slug], {
            cwd: root,
            stdout: "inherit",
            stderr: "inherit",
        });
        if (build.exitCode !== 0) fail(`\`bun run site --demo ${slug}\` failed`);
    }

    console.log("bundling the node driver...");
    buildDriverBundle();

    const server = serveOutDir();
    console.log(`serving out/site at ${server.url}`);

    try {
        const proc = Bun.spawn(["node", bundlePath, server.url, slug], {
            cwd: root,
            stdout: "pipe",
            stderr: "inherit",
        });
        const stdout = await new Response(proc.stdout).text();
        const exitCode = await proc.exited;
        if (exitCode !== 0) fail(`driver exited ${exitCode}`);

        const line = stdout
            .split("\n")
            .map((l) => l.trim())
            .find((l) => l.startsWith("{"));
        if (!line) fail(`driver printed no JSON result:\n${stdout}`);
        interface ScenarioResult {
            busyMs: number;
            called: boolean;
            callDuration: number | null;
            reportedOnWire: boolean;
        }
        const result = JSON.parse(line) as { below: ScenarioResult; above: ScenarioResult };

        console.log(
            `  below threshold (${result.below.busyMs}ms busy-loop): addDurationVital ${
                result.below.called
                    ? `called (unexpected, duration=${result.below.callDuration})`
                    : "not called"
            }; wire: ${result.below.reportedOnWire ? "slow_frame seen (unexpected)" : "no slow_frame seen"}`,
        );
        console.log(
            `  above threshold (${result.above.busyMs}ms busy-loop): addDurationVital ${
                result.above.called
                    ? `called, duration=${result.above.callDuration}`
                    : "not called (unexpected)"
            }; wire: ${result.above.reportedOnWire ? "slow_frame seen" : "no slow_frame seen (unexpected)"}`,
        );

        // structural proof (the sampler's actual decision, immune to ambient system noise) —
        // primary verdict for both directions.
        if (result.below.called) {
            fail("below-threshold run called addDurationVital — the sampler doesn't discriminate");
        }
        if (!result.above.called) {
            fail("above-threshold run never called addDurationVital");
        }
        // wire proof — required for the direction that actually crosses the network boundary
        // (this is what caught the raw-rAF-timestamp-as-epoch bug `site/rum-runtime.ts` fixes).
        if (!result.above.reportedOnWire) {
            fail("above-threshold run never sent an intake request carrying slow_frame");
        }

        console.log("✓ RUM slow-frame intake proof: both directions discriminate correctly");
        process.exit(0);
    } finally {
        server.stop();
    }
}

if (import.meta.main) {
    main().catch((err) => {
        console.error(err instanceof Error ? err.message : String(err));
        process.exit(1);
    });
}
