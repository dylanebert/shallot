import { resolve, join } from "path";
import {
    existsSync,
    rmSync,
    mkdirSync,
    copyFileSync,
    writeFileSync,
    readdirSync,
    unlinkSync,
    readFileSync,
    cpSync,
} from "fs";
import { scaffold, startServer, isWSL, detectDisplay } from "./scaffold";

const projectDir = resolve(import.meta.dir, "..");
const PORT = 3004;

const args = process.argv.slice(2);
if (args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run capture --out <dir> [options]

Scaffolds a project from create-shallot, opens the editor,
runs Playwright capture flows, and outputs screenshots + metadata.

Options:
  --out <dir>     Output directory (required)
  --flow <name>   Run a single flow (default: all)
  --keep          Don't delete temp dir after capture`);
    process.exit(0);
}

const keepTmp = args.includes("--keep");
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

const { dir: scaffoldDir, cleanup: cleanupScaffold } = scaffold("capture-test", PORT);
const cliPath = resolve(projectDir, "packages/shallot/bin/cli.ts");

console.log("Starting editor dev server...");
const { cleanup } = await startServer(
    ["bun", cliPath, "dev", scaffoldDir, "--port", String(PORT)],
    projectDir,
    PORT,
    keepTmp ? undefined : cleanupScaffold,
);

console.log("Running capture flows...");

const configFile = resolve(projectDir, "scripts/capture/capture.pw.config.ts");
const flowsDir = resolve(projectDir, "scripts/capture/flows");

let exitCode: number | null;

if (isWSL) {
    const winTempProc = Bun.spawnSync(
        ["powershell.exe", "-Command", "Write-Host -NoNewline $env:TEMP"],
        { stdout: "pipe" },
    );
    const winTempPath = new TextDecoder().decode(winTempProc.stdout).trim().replace(/\r/g, "");
    const wslTempProc = Bun.spawnSync(["wslpath", winTempPath], { stdout: "pipe" });
    const wslTemp = new TextDecoder().decode(wslTempProc.stdout).trim();
    const pwDir = join(wslTemp, "shallot-capture-pw");
    const winPwDir = winTempPath + "\\shallot-capture-pw";

    rmSync(pwDir, { recursive: true, force: true });
    mkdirSync(pwDir, { recursive: true });
    mkdirSync(join(pwDir, "flows"), { recursive: true });

    copyFileSync(resolve(projectDir, "scripts/capture/core.ts"), join(pwDir, "core.ts"));
    copyFileSync(resolve(projectDir, "scripts/capture/runner.ts"), join(pwDir, "runner.ts"));
    copyFileSync(resolve(projectDir, "scripts/capture/selectors.ts"), join(pwDir, "selectors.ts"));
    copyFileSync(configFile, join(pwDir, "capture.pw.config.ts"));

    const flowFiles = new Bun.Glob("*.pw.ts").scanSync(flowsDir);
    for (const f of flowFiles) {
        copyFileSync(join(flowsDir, f), join(pwDir, "flows", f));
    }

    const winCapturesDir = winPwDir + "\\captures";

    writeFileSync(
        join(pwDir, "package.json"),
        JSON.stringify({
            private: true,
            type: "module",
            devDependencies: { "@playwright/test": "latest" },
        }) + "\n",
    );

    console.log("Installing Playwright on Windows...");
    Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `cd '${winPwDir}'; bun install --silent; bunx playwright install chromium`,
        ],
        { stdout: "inherit", stderr: "inherit" },
    );

    const grepArg = flowFilter ? `--grep "${flowFilter}"` : "";
    console.log("Running capture via Windows...");
    const result = Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `$env:CAPTURE_PORT = '${PORT}'; $env:CAPTURE_OUT = '${winCapturesDir}'; $env:PLAYWRIGHT_BROWSERS_PATH = "$env:LOCALAPPDATA\\ms-playwright"; cd '${winPwDir}'; bunx playwright test --config capture.pw.config.ts ${grepArg}`,
        ],
        { stdout: "inherit", stderr: "inherit", timeout: 120000 },
    );
    exitCode = result.exitCode;

    const winCaptures = join(pwDir, "captures");
    if (existsSync(winCaptures)) {
        mkdirSync(capturesDir, { recursive: true });
        cpSync(winCaptures, capturesDir, { recursive: true });
    }

    rmSync(pwDir, { recursive: true, force: true });
} else {
    const pwArgs = ["bunx", "playwright", "test", "--config", configFile];
    if (flowFilter) pwArgs.push("--grep", flowFilter);

    const result = Bun.spawnSync(pwArgs, {
        stdout: "inherit",
        stderr: "inherit",
        cwd: resolve(projectDir, "scripts/capture"),
        env: { ...process.env, CAPTURE_PORT: String(PORT), CAPTURE_OUT: capturesDir },
        timeout: 120000,
    });
    exitCode = result.exitCode;
}

if (exitCode !== 0) {
    console.error("FAIL: capture flows failed");
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
