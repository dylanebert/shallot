import { resolve, join } from "path";
import { existsSync, mkdirSync, rmSync, writeFileSync, cpSync } from "fs";
import { compileShadeShader } from "../packages/shallot/src/extras/raytracing/shaders";
import { surfaceRegistry, surface } from "../packages/shallot/src/standard/render/surface";
import { solverWGSL } from "../packages/shallot/src/standard/physics/solver.wgsl";
import {
    prepareIndirectWGSL,
    packWGSL,
    rebuildWGSL,
    syncTransformsWGSL,
} from "../packages/shallot/src/standard/physics/utility.wgsl";
import { interpolateWGSL } from "../packages/shallot/src/standard/physics/interpolate.wgsl";
import {
    compileRasterShader,
    compileSkyShader,
} from "../packages/shallot/src/standard/raster/forward";
import { compilePresentShader } from "../packages/shallot/src/standard/render/present";

const projectDir = resolve(import.meta.dir, "..");
const isWSL = process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

const surfaces = surfaceRegistry.all();
surface({ fragment: "surface.baseColor = vec3(1.0, 0.0, 0.0);" });
const surfaces5 = surfaceRegistry.all();

interface Variant {
    system: string;
    name: string;
    code: string;
    entryPoint: string;
    render?: boolean;
}

const variants: Variant[] = [
    // Physics
    {
        system: "physics",
        name: "solver (24 entries)",
        code: solverWGSL,
        entryPoint: "warmstartBodies",
    },
    { system: "physics", name: "pack", code: packWGSL, entryPoint: "packBodies" },
    {
        system: "physics",
        name: "rebuild warmstarts",
        code: rebuildWGSL,
        entryPoint: "rebuildWarmstarts",
    },
    { system: "physics", name: "prepare indirect", code: prepareIndirectWGSL, entryPoint: "main" },
    {
        system: "physics",
        name: "sync transforms",
        code: syncTransformsWGSL,
        entryPoint: "syncTransforms",
    },
    { system: "physics", name: "interpolation", code: interpolateWGSL, entryPoint: "interpolate" },

    // RT
    { system: "rt", name: "shade", code: compileShadeShader(surfaces), entryPoint: "main" },
    {
        system: "rt",
        name: `shade (${surfaces5.length} surfaces)`,
        code: compileShadeShader(surfaces5),
        entryPoint: "main",
    },

    // Raster
    {
        system: "raster",
        name: "forward (shadows)",
        code: compileRasterShader(surfaces, true),
        entryPoint: "vs",
        render: true,
    },
    {
        system: "raster",
        name: "forward (no shadows)",
        code: compileRasterShader(surfaces, false),
        entryPoint: "vs",
        render: true,
    },
    { system: "raster", name: "sky", code: compileSkyShader(), entryPoint: "vs", render: true },
    {
        system: "raster",
        name: "present",
        code: compilePresentShader(),
        entryPoint: "vs",
        render: true,
    },
];

const variantData = variants.map((v) => ({
    system: v.system,
    name: v.name,
    code: v.code,
    size: v.code.length,
    entryPoint: v.entryPoint,
    render: v.render || false,
}));

console.log(
    `Generated ${variants.length} variants across ${new Set(variants.map((v) => v.system)).size} systems`,
);

const html = `<!DOCTYPE html>
<html>
<head><title>Shader Compile</title></head>
<body>
<pre id="output">Running...</pre>
<script type="module">
const variants = ${JSON.stringify(variantData)};

async function run() {
    const out = document.getElementById("output");
    const log = (s) => { out.textContent += s + "\\n"; console.log(s); };

    if (!navigator.gpu) { log("WebGPU not available"); return; }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { log("No adapter"); return; }
    log("Adapter: " + (adapter.info?.vendor || "?") + " / " + (adapter.info?.architecture || "?"));

    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        },
    });

    out.textContent = "";

    const results = [];
    for (const v of variants) {
        const t0 = performance.now();
        device.pushErrorScope("validation");
        const module = device.createShaderModule({ code: v.code });
        const err = await device.popErrorScope();
        const moduleMs = performance.now() - t0;

        if (err) {
            log("[" + v.system + "] " + v.name.padEnd(28) + " MODULE ERROR: " + err.message.slice(0, 80));
            results.push({ ...v, moduleMs, pipelineMs: -1, totalMs: -1 });
            continue;
        }

        const t1 = performance.now();
        try {
            if (v.render) {
                const fsEntry = v.entryPoint === "vertexMain" ? "fragmentMain" : "fs";
                await device.createRenderPipelineAsync({
                    layout: "auto",
                    vertex: { module, entryPoint: v.entryPoint },
                    fragment: { module, entryPoint: fsEntry, targets: [{ format: "bgra8unorm" }] },
                    primitive: { topology: "triangle-list" },
                });
            } else {
                await device.createComputePipelineAsync({
                    layout: "auto",
                    compute: { module, entryPoint: v.entryPoint },
                });
            }
        } catch (e) {
            log("[" + v.system + "] " + v.name.padEnd(28) + " PIPELINE ERROR: " + e.message?.slice(0, 80));
            results.push({ ...v, moduleMs, pipelineMs: -1, totalMs: -1 });
            continue;
        }
        const pipelineMs = performance.now() - t1;
        results.push({ ...v, moduleMs, pipelineMs, totalMs: moduleMs + pipelineMs });
    }

    const bar = "=".repeat(90);
    const systems = ["physics", "rt", "raster"];
    const arch = adapter.info?.architecture || "unknown";

    log(bar);
    log("  Shader Compilation — Chrome / " + arch);
    log(bar);

    for (const sys of systems) {
        const sysResults = results.filter(r => r.system === sys);
        if (!sysResults.length) continue;
        const sysTotal = sysResults.reduce((s, r) => s + (r.totalMs > 0 ? r.totalMs : 0), 0);
        log("");
        log("  " + sys.toUpperCase() + " (" + sysTotal.toFixed(0) + "ms total)");
        log("  " + "shader".padEnd(30) + "total".padStart(8) + "module".padStart(8) + "pipeline".padStart(9) + "size".padStart(8));
        log("  " + "-".repeat(30) + "-".repeat(8) + "-".repeat(8) + "-".repeat(9) + "-".repeat(8));
        for (const r of sysResults) {
            const sizeStr = (r.size / 1024).toFixed(1) + "KB";
            if (r.totalMs < 0) { log("  " + r.name.padEnd(30) + "ERR".padStart(8)); continue; }
            log("  " + r.name.padEnd(30)
                + (r.totalMs.toFixed(0) + "ms").padStart(8)
                + (r.moduleMs.toFixed(0) + "ms").padStart(8)
                + (r.pipelineMs.toFixed(0) + "ms").padStart(9)
                + sizeStr.padStart(8));
        }
    }

    const grandTotal = results.reduce((s, r) => s + (r.totalMs > 0 ? r.totalMs : 0), 0);
    log("");
    log(bar);
    log("  TOTAL: " + grandTotal.toFixed(0) + "ms");
    log(bar);

    // === Cache warming experiment ===
    // Test whether compiling a similar shader first speeds up subsequent variants.
    // Uses cache-bust constants to defeat identical-code caching.
    // Each trial gets a fresh device to isolate per-device pipeline cache effects.

    log("");
    log("=".repeat(90));
    log("  Cache Warming Experiment");
    log("=".repeat(90));

    const uberFull = variants.find(v => v.name === "uber (full)");
    const uberShadows = variants.find(v => v.name === "uber (shadows)");
    const uberBase = variants.find(v => v.name === "uber (base)");

    if (uberFull && uberShadows && uberBase) {
        function bustCache(code, id) {
            return code.replace("const EPSILON: f32 =", "const CACHE_BUST_" + id + ": f32 = 0.0;\\nconst EPSILON: f32 =");
        }

        async function timeCompile(dev, code, entry) {
            const t0 = performance.now();
            dev.pushErrorScope("validation");
            const mod = dev.createShaderModule({ code });
            const err = await dev.popErrorScope();
            if (err) return { total: -1, module: -1, pipeline: -1, error: err.message };
            const modMs = performance.now() - t0;
            const t1 = performance.now();
            await dev.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: entry } });
            const pipMs = performance.now() - t1;
            return { total: modMs + pipMs, module: modMs, pipeline: pipMs };
        }

        async function freshDevice() {
            const a = await navigator.gpu.requestAdapter();
            return a.requestDevice({
                requiredLimits: { maxStorageBuffersPerShaderStage: a.limits.maxStorageBuffersPerShaderStage },
            });
        }

        // Trial 1: uber (shadows) cold — fresh device, no prior compilation
        log("");
        log("  Trial 1: uber (shadows) COLD — fresh device");
        const dev1 = await freshDevice();
        const cold = await timeCompile(dev1, bustCache(uberShadows.code, "COLD1"), "main");
        log("    " + (cold.total < 0 ? "ERROR: " + cold.error : cold.total.toFixed(0) + "ms (module " + cold.module.toFixed(0) + "ms, pipeline " + cold.pipeline.toFixed(0) + "ms)"));
        dev1.destroy();

        // Trial 2: uber (full) then uber (shadows) — same device
        log("");
        log("  Trial 2: uber (full) → uber (shadows) — same device");
        const dev2 = await freshDevice();
        const warm2a = await timeCompile(dev2, bustCache(uberFull.code, "WARM2A"), "main");
        log("    full:    " + (warm2a.total < 0 ? "ERROR" : warm2a.total.toFixed(0) + "ms"));
        const warm2b = await timeCompile(dev2, bustCache(uberShadows.code, "WARM2B"), "main");
        log("    shadows: " + (warm2b.total < 0 ? "ERROR" : warm2b.total.toFixed(0) + "ms"));
        log("    speedup: " + (cold.total > 0 && warm2b.total > 0 ? ((1 - warm2b.total / cold.total) * 100).toFixed(1) + "%" : "N/A"));
        dev2.destroy();

        // Trial 3: uber (base) then uber (shadows) — same device
        log("");
        log("  Trial 3: uber (base) → uber (shadows) — same device");
        const dev3 = await freshDevice();
        const warm3a = await timeCompile(dev3, bustCache(uberBase.code, "WARM3A"), "main");
        log("    base:    " + (warm3a.total < 0 ? "ERROR" : warm3a.total.toFixed(0) + "ms"));
        const warm3b = await timeCompile(dev3, bustCache(uberShadows.code, "WARM3B"), "main");
        log("    shadows: " + (warm3b.total < 0 ? "ERROR" : warm3b.total.toFixed(0) + "ms"));
        log("    speedup: " + (cold.total > 0 && warm3b.total > 0 ? ((1 - warm3b.total / cold.total) * 100).toFixed(1) + "%" : "N/A"));
        dev3.destroy();

        // Trial 4: uber (full) then uber (shadows) — different devices (same adapter)
        log("");
        log("  Trial 4: uber (full) → uber (shadows) — different devices");
        const dev4a = await freshDevice();
        const warm4a = await timeCompile(dev4a, bustCache(uberFull.code, "WARM4A"), "main");
        log("    full:    " + (warm4a.total < 0 ? "ERROR" : warm4a.total.toFixed(0) + "ms"));
        dev4a.destroy();
        const dev4b = await freshDevice();
        const warm4b = await timeCompile(dev4b, bustCache(uberShadows.code, "WARM4B"), "main");
        log("    shadows: " + (warm4b.total < 0 ? "ERROR" : warm4b.total.toFixed(0) + "ms"));
        log("    speedup: " + (cold.total > 0 && warm4b.total > 0 ? ((1 - warm4b.total / cold.total) * 100).toFixed(1) + "%" : "N/A"));
        dev4b.destroy();

        // Trial 5: uber (shadows) cold again — confirms trial 1 wasn't an outlier
        log("");
        log("  Trial 5: uber (shadows) COLD again — fresh device, verification");
        const dev5 = await freshDevice();
        const cold2 = await timeCompile(dev5, bustCache(uberShadows.code, "COLD5"), "main");
        log("    " + (cold2.total < 0 ? "ERROR: " + cold2.error : cold2.total.toFixed(0) + "ms (module " + cold2.module.toFixed(0) + "ms, pipeline " + cold2.pipeline.toFixed(0) + "ms)"));
        dev5.destroy();

        log("");
        log("  Summary:");
        log("    Cold baseline (avg): " + ((cold.total + cold2.total) / 2).toFixed(0) + "ms");
        log("    After full (same device): " + (warm2b.total > 0 ? warm2b.total.toFixed(0) + "ms" : "ERR"));
        log("    After base (same device): " + (warm3b.total > 0 ? warm3b.total.toFixed(0) + "ms" : "ERR"));
        log("    After full (diff device): " + (warm4b.total > 0 ? warm4b.total.toFixed(0) + "ms" : "ERR"));
    }

    window.__compileLabResults = results;
    window.__compileLabDone = true;
}

run().catch(e => {
    document.getElementById("output").textContent = "Error: " + e.message;
    window.__compileLabDone = true;
});
</script>
</body>
</html>`;

const testDir = resolve(projectDir, "packages/shallot/tests/gpu");
const labHtmlPath = join(testDir, "shader-compile.html");
writeFileSync(labHtmlPath, html);

const labTestCode = `
import { test, expect } from "@playwright/test";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

test("shader-compile", async ({ page }) => {
    test.setTimeout(300000);
    page.on("console", (msg) => { if (msg.type() === "log") console.log(msg.text()); });
    const dir = dirname(fileURLToPath(import.meta.url));
    const htmlPath = join(dir, "shader-compile.html").replace(/\\\\\\\\/g, "/");
    await page.goto("file:///" + htmlPath);
    await page.waitForFunction(() => (window as any).__compileLabDone === true, null, { timeout: 280000 });
    const results = await page.evaluate(() => (window as any).__compileLabResults);
    expect(results.length).toBeGreaterThan(0);
});
`;

const labTestPath = join(testDir, "shader-compile.pw.ts");
writeFileSync(labTestPath, labTestCode);

if (isWSL) {
    const winTempProc = Bun.spawnSync(
        ["powershell.exe", "-Command", "Write-Host -NoNewline $env:TEMP"],
        { stdout: "pipe" },
    );
    const winTempPath = new TextDecoder().decode(winTempProc.stdout).trim().replace(/\r/g, "");
    const wslTempProc = Bun.spawnSync(["wslpath", winTempPath], { stdout: "pipe" });
    const wslTemp = new TextDecoder().decode(wslTempProc.stdout).trim();
    const testTemp = join(wslTemp, "shallot-compile-lab");
    const winTestTemp = winTempPath + "\\shallot-compile-lab";

    rmSync(testTemp, { recursive: true, force: true });
    mkdirSync(testTemp, { recursive: true });

    const labConfig = `
import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
    testDir: ".",
    testMatch: "shader-compile.pw.ts",
    fullyParallel: false,
    retries: 0,
    workers: 1,
    reporter: [["list"]],
    timeout: 300000,
    use: { trace: "off", video: "off", headless: false },
    projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], channel: "chrome" } }],
});
`;

    cpSync(labHtmlPath, join(testTemp, "shader-compile.html"));
    cpSync(labTestPath, join(testTemp, "shader-compile.pw.ts"));
    cpSync(join(testDir, "package.json"), join(testTemp, "package.json"));
    writeFileSync(join(testTemp, "lab.config.ts"), labConfig);

    console.log("Installing dependencies...");
    Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `cd '${winTestTemp}'; bun install --silent; bunx playwright install chromium`,
        ],
        { stdout: "inherit", stderr: "inherit" },
    );

    console.log("Running shader compile profiler via Windows Chrome...");
    const result = Bun.spawnSync(
        [
            "powershell.exe",
            "-Command",
            `$env:PLAYWRIGHT_BROWSERS_PATH = "$env:LOCALAPPDATA\\ms-playwright"; cd '${winTestTemp}'; bunx playwright test --config lab.config.ts shader-compile.pw.ts`,
        ],
        { stdout: "pipe", stderr: "inherit", timeout: 300000 },
    );

    process.stdout.write(new TextDecoder().decode(result.stdout));

    rmSync(labHtmlPath, { force: true });
    rmSync(labTestPath, { force: true });

    process.exit(result.exitCode);
} else {
    console.log(`Files written to: ${testDir}`);
}
