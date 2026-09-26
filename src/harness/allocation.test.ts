import { expect } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
    type AllocationSample,
    allocatesNothing,
    allocationFailure,
} from "@dylanebert/shallot/harness/allocation";
import { check } from "@dylanebert/shallot/harness/check";

const ENTRY = resolve(import.meta.dir, "../../examples/first-person/src/allocation.entry.ts");
const ROOT = resolve(import.meta.dir, "../..");

check(
    "Node allocation import leaves Vite unloaded",
    {
        claim: "importing the allocation instrument for Node-only rows does not resolve Vite before the page-build path is requested",
        size: "integration",
        requires: ["node"],
    },
    async () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-allocation-import-"));
        try {
            const loader = resolve(dir, "reject-vite.mjs");
            writeFileSync(
                loader,
                `import { existsSync, statSync } from "node:fs";\nimport { fileURLToPath, pathToFileURL } from "node:url";\nexport async function resolve(specifier, context, nextResolve) {\n    if (specifier === "vite") throw new Error("Node allocation import resolved vite");\n    if (specifier.startsWith(".")) {\n        const path = fileURLToPath(new URL(specifier, context.parentURL));\n        for (const candidate of [path + ".ts", path + "/index.ts", path]) if (existsSync(candidate) && statSync(candidate).isFile()) return { url: pathToFileURL(candidate).href, shortCircuit: true };\n    }\n    return nextResolve(specifier, context);\n}\nexport async function load(url, context, nextLoad) {\n    const loaded = await nextLoad(url, context);\n    if (url === ${JSON.stringify(new URL("./allocation.ts", import.meta.url).href)}) return { ...loaded, source: loaded.source.toString().replaceAll("import.meta.dir", ${JSON.stringify(JSON.stringify(resolve(import.meta.dir)))}) };\n    return loaded;\n}\n`,
            );
            const preload = resolve(dir, "reject-vite-require.mjs");
            writeFileSync(
                preload,
                `import { createRequire } from "node:module";\nconst Module = createRequire(import.meta.url)("node:module");\nconst require = Module.prototype.require;\nModule.prototype.require = function (specifier, ...args) {\n    if (specifier === "vite") throw new Error("Node allocation import required vite");\n    return require.call(this, specifier, ...args);\n};\n`,
            );
            const probe = resolve(dir, "probe.mjs");
            writeFileSync(
                probe,
                `import { allocationFailure } from ${JSON.stringify(new URL("./allocation.ts", import.meta.url).href)};\nif (allocationFailure({ warm: 1, windows: [] }) === undefined) throw new Error("probe did not evaluate allocation exports");\n`,
            );
            const proc = Bun.spawnSync(
                [
                    "node",
                    "--no-warnings",
                    "--import",
                    preload,
                    "--experimental-loader",
                    loader,
                    probe,
                ],
                {
                    stdout: "pipe",
                    stderr: "pipe",
                },
            );
            if (proc.exitCode !== 0)
                throw new Error(`Node allocation import failed: ${proc.stderr.toString()}`);
        } finally {
            rmSync(dir, { recursive: true, force: true });
        }
    },
);

// Profiler modules: the `profile` extra, which owns the physics step's timing clock.
const PROFILER = [/^src\/extras\/profile\//];

check(
    "the gated first-person bundle imports no profiler module",
    {
        claim: "the allocation-gated first-person composition carries no timing or profiling module, so its default step does no diagnostics work",
    },
    async () => {
        const built = await Bun.build({
            entrypoints: [ENTRY],
            target: "node",
            format: "esm",
            metafile: true,
        });
        if (!built.success || built.metafile === undefined)
            throw new Error(`gated bundle failed: ${built.logs.map(String).join("\n")}`);
        const inputs = built.metafile.inputs;
        const paths = Object.keys(inputs).map((path) => resolve(ROOT, path).slice(ROOT.length + 1));
        // Non-vacuity: the graph reaches the physics step whose timers this row is about.
        if (!paths.includes("src/transitional/physics/solver/step.ts"))
            throw new Error(
                `inconclusive: gated bundle graph lacks the physics step (${paths.length} modules)`,
            );
        const importers = (module: string) =>
            Object.entries(inputs)
                .filter(([, input]) =>
                    input.imports.some(
                        (edge) => resolve(ROOT, edge.path) === resolve(ROOT, module),
                    ),
                )
                .map(([path]) => path);
        const found = Object.keys(inputs).filter((path) =>
            PROFILER.some((pattern) => pattern.test(resolve(ROOT, path).slice(ROOT.length + 1))),
        );
        if (found.length !== 0)
            throw new Error(
                `gated bundle imports profiler modules:\n${found.map((path) => `  ${path} <- ${importers(path).join(", ")}`).join("\n")}`,
            );
    },
);

const steadySample = (sites: AllocationSample["windows"][number]["sites"]) => ({
    warm: 120,
    windows: [
        { label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 },
        { label: "after warm 240", sites: [], frames: 120, framesAtMost: 120 },
        { label: "A/A repeat", sites, frames: 120, framesAtMost: 120 },
    ],
});

check(
    "three zero steady windows pass",
    {
        claim: "the allocation gate passes when all three expected steady windows sample zero bytes at zero sites",
    },
    () => {
        const sample = steadySample([]);
        expect(allocatesNothing(sample)).toBe(true);
        expect(allocationFailure(sample)).toBeUndefined();
    },
);

check(
    "an empty steady sample reds for missing windows",
    {
        claim: "the allocation gate reds with the names of all expected windows when a sampler returns no steady windows",
    },
    () => {
        const sample = { warm: 120, windows: [] as AllocationSample["windows"] };
        expect(allocatesNothing(sample)).toBe(false);
        expect(allocationFailure(sample)).toBe(
            "steady allocation sample is missing expected windows: after warm 120, after warm 240, A/A repeat",
        );
    },
);

check(
    "an incomplete steady sample reds for its missing windows",
    {
        claim: "the allocation gate reds with the names of expected steady windows omitted by an incomplete sampler result",
    },
    () => {
        const sample = {
            warm: 120,
            windows: [{ label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 }],
        };
        expect(allocatesNothing(sample)).toBe(false);
        expect(allocationFailure(sample)).toBe(
            "steady allocation sample is missing expected windows: after warm 240, A/A repeat",
        );
    },
);

check(
    "any steady allocation reds and names its sampled site",
    {
        claim: "the allocation gate reds on any sampled steady allocation and prints its site only for diagnosis",
    },
    () => {
        const sample = steadySample([
            { site: "stepChunk src/transitional/character/sweep.ts:42", bytes: 96, count: 3 },
        ]);
        expect(allocatesNothing(sample)).toBe(false);
        expect(allocationFailure(sample)).toBe(
            "steady play allocated JavaScript heap; sampler sites are diagnosis only:\n  A/A repeat: 96 B at stepChunk src/transitional/character/sweep.ts:42",
        );
    },
);
