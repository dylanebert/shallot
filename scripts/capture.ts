import {
    cpSync,
    existsSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    unlinkSync,
    writeFileSync,
} from "fs";
import { basename, join, resolve } from "path";
import { runPlaywright } from "../harness/core/playwright";
import { startServer } from "../harness/core/server";
import { detectDisplay } from "../harness/core/wsl";

const projectDir = resolve(import.meta.dir, "..");
const captureDir = resolve(import.meta.dir, "capture");
const fixtureDir = resolve(captureDir, "fixture");
const PORT = 3004;
// the standalone run({survive}) fixture for the survive-reload flow, on its own dev server (no editor)
const APP_PORT = 3005;
// the glTF fixture for the asset-swap flow — an editor server on its own dir so enabling GltfPlugin + a
// model doesn't perturb the 16 flows that run against the main fixture
const GLTF_PORT = 3006;
// the framework fixtures for the framework-{svelte,react} flows — real vite+svelte / vite+react projects
// (own vite.config) the editor opens through its merged toolchain, each on its own dir + server
const SVELTE_PORT = 3007;
const REACT_PORT = 3008;
// the standalone run() ui-containment fixture (its own dev server, no editor) — proves config.ui's overlay
// stays clipped to the canvas region
const UI_PORT = 3009;
// the zoo specimens each get their own editor server from here up — see the zoo-sweep flow
const ZOO_PORT_BASE = 3010;

const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run capture --out <dir> [options]

Boots the shallot editor against the in-repo fixture (scripts/capture/fixture), runs the Playwright
capture flows, and outputs screenshots + metadata. Display-gated; on WSL it runs via Windows Chrome.

Options:
  --out <dir>     Output directory (required)
  --flow <name>   Run a single flow (default: all)`);
    process.exit(0);
}

const flowIdx = args.indexOf("--flow");
const flowFilter = flowIdx !== -1 ? args[flowIdx + 1] : undefined;
const outIdx = args.indexOf("--out");
const capturesDir = outIdx !== -1 ? resolve(args[outIdx + 1]) : undefined;
if (!capturesDir) {
    console.error("--out <dir> is required");
    process.exit(1);
}

if (!detectDisplay()) {
    console.log("No display available. Skipping capture.");
    process.exit(0);
}

// Boot only the servers the selected flows need. Each flow needs the main editor server (the fixture) by
// default; the few that drive a dedicated fixture map to their own group. A single `--flow` (the common
// targeted run) then boots one server, not ten — most of capture's wall time was idle fixtures, and the
// boot was where a wedged server hung the whole run (server.ts probes are now bounded, so this also
// shrinks the surface that can fail). No `--flow` boots everything (the full suite).
const FLOW_GROUP: Record<string, string> = {
    "survive-reload": "app",
    "ui-containment": "ui",
    "asset-swap": "gltf",
    "gltf-import": "gltf",
    "framework-svelte": "svelte",
    "framework-react": "react",
    "zoo-sweep": "zoo",
};
const flowNames = [...new Bun.Glob("flows/*.pw.ts").scanSync(captureDir)].map((f) =>
    basename(f, ".pw.ts"),
);
const selected = flowFilter ? flowNames.filter((n) => n.includes(flowFilter)) : flowNames;
const needed = new Set(selected.map((n) => FLOW_GROUP[n] ?? "editor"));
console.log(
    `flows: ${selected.length ? selected.join(", ") : "(none matched)"} — servers: ${[...needed].join(", ") || "none"}`,
);

// The CLI's deduplicate plugin resolves `@dylanebert/shallot` from a fixture to engine source, so no pack
// / install is needed.
const cliPath = resolve(projectDir, "packages/shallot/bin/cli.ts");
const edit = (dir: string, port: number, label: string) =>
    startServer(projectDir, port, label, ["bun", cliPath, "edit", dir, "--port", String(port)]);

const server = needed.has("editor") ? await edit(fixtureDir, PORT, "editor") : null;
// the standalone run({survive}) app the survive-reload flow drives — its own dev server, no editor
const appServer = needed.has("app")
    ? await startServer(projectDir, APP_PORT, "survive-app", [
          "bun",
          resolve(captureDir, "serve-app.ts"),
          String(APP_PORT),
      ])
    : null;
// dedicated-fixture editors: glTF (asset-swap), and the real vite+svelte / vite+react projects the
// framework flows open through the editor's merged toolchain — each on its own dir so it can't perturb the
// flows that run against the main fixture
const gltfServer = needed.has("gltf")
    ? await edit(resolve(captureDir, "fixture-gltf"), GLTF_PORT, "gltf-editor")
    : null;
const svelteServer = needed.has("svelte")
    ? await edit(resolve(captureDir, "fixture-svelte"), SVELTE_PORT, "svelte-editor")
    : null;
const reactServer = needed.has("react")
    ? await edit(resolve(captureDir, "fixture-react"), REACT_PORT, "react-editor")
    : null;
// the standalone run() ui-containment fixture — served like the survive app, on the "ui" subdir
const uiServer = needed.has("ui")
    ? await startServer(projectDir, UI_PORT, "ui-app", [
          "bun",
          resolve(captureDir, "serve-app.ts"),
          String(UI_PORT),
          "ui",
      ])
    : null;

// Each zoo specimen is a real, untouched create-shallot project (`examples/zoo/<module>/shallot.json`); the
// zoo-sweep flow drives them all in one browser session. The list is needed for the env below either way;
// the servers boot only for that flow.
const zooDir = resolve(projectDir, "examples/zoo");
const specimens = readdirSync(zooDir)
    .filter((name) => name !== "node_modules" && name !== "public")
    .filter((name) => existsSync(join(zooDir, name, "shallot.json")))
    .sort()
    .map((name, i) => ({ name, dir: join(zooDir, name), port: ZOO_PORT_BASE + i }));

// zoo servers boot sequentially, never in parallel: every editor instance shares the engine package's
// vite dep cache, and concurrent cold optimizes race its rename (deps → deps_temp, ENOENT) — the first
// boot warms the cache, the rest reuse it.
const zooServers: Awaited<ReturnType<typeof edit>>[] = [];
if (needed.has("zoo"))
    for (const s of specimens) zooServers.push(await edit(s.dir, s.port, `zoo:${s.name}`));

const cleanup = () => {
    server?.kill();
    appServer?.kill();
    gltfServer?.kill();
    svelteServer?.kill();
    reactServer?.kill();
    uiServer?.kill();
    for (const s of zooServers) s.kill();
};
process.on("exit", cleanup);
process.on("SIGINT", () => {
    cleanup();
    process.exit(1);
});
process.on("SIGTERM", () => {
    cleanup();
    process.exit(1);
});

console.log("Running capture flows...");
const flows = [...new Bun.Glob("flows/*.pw.ts").scanSync(captureDir)];

// Same staging + spawn path the gym uses (harness/core/playwright.ts). The flow timeouts live in the
// config; this ceiling sits above the config's globalTimeout (900s) as a backstop for a wedged process.
const run = runPlaywright({
    dir: captureDir,
    config: "capture.pw.config.ts",
    args: flowFilter ? ["--grep", flowFilter] : [],
    stage: {
        name: "shallot-capture",
        files: [
            "package.json",
            "capture.pw.config.ts",
            "core.ts",
            "runner.ts",
            "selectors.ts",
            ...flows,
        ],
    },
    env: (staged) => ({
        CAPTURE_PORT: String(PORT),
        CAPTURE_APP_PORT: String(APP_PORT),
        CAPTURE_GLTF_PORT: String(GLTF_PORT),
        CAPTURE_SVELTE_PORT: String(SVELTE_PORT),
        CAPTURE_REACT_PORT: String(REACT_PORT),
        CAPTURE_UI_PORT: String(UI_PORT),
        CAPTURE_ZOO: JSON.stringify(specimens.map((s) => ({ name: s.name, port: s.port }))),
        CAPTURE_OUT: staged ? `${staged.win}\\captures` : capturesDir,
    }),
    inherit: true,
    timeoutMs: 960_000,
});

if (run.staged) {
    const wslCaptures = join(run.staged.wsl, "captures");
    if (existsSync(wslCaptures)) {
        mkdirSync(capturesDir, { recursive: true });
        cpSync(wslCaptures, capturesDir, { recursive: true });
    }
}

if (run.exitCode !== 0) {
    console.error(
        `FAIL: capture flows failed${run.timedOut ? " (spawn ceiling — Playwright did not exit)" : ""}`,
    );
    cleanup();
    process.exit(1);
}

console.log("Converting to WebP...");
const sharp = (await import("sharp")).default;

for (const flow of readdirSync(capturesDir, { withFileTypes: true })) {
    if (!flow.isDirectory()) continue;
    const flowDir = join(capturesDir, flow.name);
    const manifestPath = join(flowDir, "manifest.json");

    for (const file of readdirSync(flowDir)) {
        if (!file.endsWith(".png")) continue;
        const pngPath = join(flowDir, file);
        const webpPath = pngPath.replace(/\.png$/, ".webp");
        await sharp(pngPath).webp({ lossless: true }).toFile(webpPath);
        unlinkSync(pngPath);
    }

    if (existsSync(manifestPath)) {
        const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
        for (const step of manifest.steps) {
            step.screenshot = step.screenshot.replace(/\.png$/, ".webp");
        }
        writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
    }
}

console.log(`PASS: captures written to ${capturesDir}`);
cleanup();
process.exit(0);
