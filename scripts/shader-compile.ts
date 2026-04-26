import { setupGlobals } from "bun-webgpu";
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

await setupGlobals();

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No GPU adapter");
const device = await adapter.requestDevice({
    requiredLimits: {
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
});

const surfaces = surfaceRegistry.all();

interface Result {
    system: string;
    name: string;
    moduleMs: number;
    pipelineMs: number;
    totalMs: number;
    size: number;
    entryPoint: string;
}

async function measure(
    code: string,
    entryPoint: string,
    isRender = false,
): Promise<{ moduleMs: number; pipelineMs: number; error?: string }> {
    const t0 = performance.now();
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code });
    const err = await device.popErrorScope();
    const moduleMs = performance.now() - t0;

    if (err) return { moduleMs, pipelineMs: -1, error: err.message.slice(0, 120) };

    const t1 = performance.now();
    try {
        if (isRender) {
            const fsEntry = entryPoint === "vertexMain" ? "fragmentMain" : "fs";
            await device.createRenderPipelineAsync({
                layout: "auto",
                vertex: { module, entryPoint },
                fragment: {
                    module,
                    entryPoint: fsEntry,
                    targets: [{ format: "bgra8unorm" }],
                },
                primitive: { topology: "triangle-list" },
            });
        } else {
            await device.createComputePipelineAsync({
                layout: "auto",
                compute: { module, entryPoint },
            });
        }
    } catch (e: any) {
        return { moduleMs, pipelineMs: -1, error: e.message?.slice(0, 120) };
    }
    return { moduleMs, pipelineMs: performance.now() - t1 };
}

// Shader variants to measure, grouped by system
interface Variant {
    system: string;
    name: string;
    code: string | (() => string);
    entryPoint: string;
    render?: boolean;
}

const variants: Variant[] = [
    // === Physics ===
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

    // === Raytracing ===
    { system: "rt", name: "shade", code: () => compileShadeShader(surfaces), entryPoint: "main" },

    // === Raster ===
    {
        system: "raster",
        name: "forward (shadows)",
        code: () => compileRasterShader(surfaces, true),
        entryPoint: "vs",
        render: true,
    },
    {
        system: "raster",
        name: "forward (no shadows)",
        code: () => compileRasterShader(surfaces, false),
        entryPoint: "vs",
        render: true,
    },
    { system: "raster", name: "sky", code: compileSkyShader, entryPoint: "vs", render: true },
    {
        system: "raster",
        name: "present",
        code: compilePresentShader,
        entryPoint: "vs",
        render: true,
    },
];

// Add surface scaling variants
surface({ fragment: "surface.baseColor = vec3(1.0, 0.0, 0.0);" });
const surfaces5 = surfaceRegistry.all();
variants.push({
    system: "rt",
    name: `shade (${surfaces5.length} surfaces)`,
    code: () => compileShadeShader(surfaces5),
    entryPoint: "main",
});

console.log(`Surfaces: ${surfaces.length} default, ${surfaces5.length} extended\n`);

const results: Result[] = [];

for (const v of variants) {
    let code: string;
    try {
        code = typeof v.code === "function" ? v.code() : v.code;
    } catch (e: any) {
        console.log(`  [${v.system}] ${v.name.padEnd(28)} SETUP FAILED: ${e.message}`);
        continue;
    }

    const { moduleMs, pipelineMs, error } = await measure(code, v.entryPoint, v.render);

    if (error) {
        console.log(`  [${v.system}] ${v.name.padEnd(28)} FAILED: ${error}`);
        results.push({
            system: v.system,
            name: v.name,
            moduleMs,
            pipelineMs: -1,
            totalMs: -1,
            size: code.length,
            entryPoint: v.entryPoint,
        });
        continue;
    }

    results.push({
        system: v.system,
        name: v.name,
        moduleMs,
        pipelineMs,
        totalMs: moduleMs + pipelineMs,
        size: code.length,
        entryPoint: v.entryPoint,
    });
}

// Print grouped table
const bar = "=".repeat(90);
const systems = ["physics", "rt", "raster"];

console.log(bar);
console.log("  Shader Compilation — bun-webgpu / Vulkan");
console.log(bar);

for (const sys of systems) {
    const sysResults = results.filter((r) => r.system === sys);
    if (sysResults.length === 0) continue;

    const sysTotal = sysResults.reduce((sum, r) => sum + (r.totalMs > 0 ? r.totalMs : 0), 0);
    console.log(`\n  ${sys.toUpperCase()} (${sysTotal.toFixed(0)}ms total)`);
    console.log(
        `  ${"shader".padEnd(30)} ${"total".padStart(8)} ${"module".padStart(8)} ${"pipeline".padStart(9)} ${"size".padStart(8)}`,
    );
    console.log(
        `  ${"-".repeat(30)} ${"-".repeat(8)} ${"-".repeat(8)} ${"-".repeat(9)} ${"-".repeat(8)}`,
    );

    for (const r of sysResults) {
        const sizeStr = `${(r.size / 1024).toFixed(1)}KB`;
        if (r.totalMs < 0) {
            console.log(`  ${r.name.padEnd(30)} ${"ERR".padStart(8)}`);
            continue;
        }
        console.log(
            `  ${r.name.padEnd(30)} ${(r.totalMs.toFixed(0) + "ms").padStart(8)} ${(r.moduleMs.toFixed(0) + "ms").padStart(8)} ${(r.pipelineMs.toFixed(0) + "ms").padStart(9)} ${sizeStr.padStart(8)}`,
        );
    }
}

const grandTotal = results.reduce((sum, r) => sum + (r.totalMs > 0 ? r.totalMs : 0), 0);
console.log(`\n${bar}`);
console.log(`  TOTAL: ${grandTotal.toFixed(0)}ms`);
console.log(bar);
