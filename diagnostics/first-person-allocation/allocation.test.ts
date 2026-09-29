import { expect, test } from "bun:test";
import { resolve } from "node:path";
import {
    type AllocationSample,
    allocatesNothing,
    allocationFailure,
} from "./allocation";

const ENTRY = resolve(import.meta.dir, "../../examples/first-person/src/allocation.entry.ts");
const ROOT = resolve(import.meta.dir, "../..");

test("a Node allocation import does not load the display-only oracle or Hyprland instrument", async () => {
    const built = await Bun.build({
        entrypoints: [resolve(import.meta.dir, "allocation.ts")],
        metafile: true,
        external: ["playwright", "bun-webgpu", "chromium-bidi"],
        target: "bun",
    });
    if (!built.success || built.metafile === undefined)
        throw new Error(`allocation import graph failed: ${built.logs.map(String).join("\\n")}`);
    const inputs = Object.keys(built.metafile.inputs).map((path) => resolve(path));
    expect(inputs.some((path) => path.endsWith("/diagnostics/first-person-allocation/display.ts"))).toBe(false);
    expect(inputs.some((path) => path.endsWith("/diagnostics/first-person-allocation/display-seat.ts"))).toBe(false);

    const node = await Bun.build({
        entrypoints: [resolve(import.meta.dir, "allocation-sampler.mjs")],
        metafile: true,
        target: "node",
        external: ["webgpu"],
    });
    if (!node.success || node.metafile === undefined)
        throw new Error(`Node sampler graph failed: ${node.logs.map(String).join("\\n")}`);
    const nodeInputs = Object.keys(node.metafile.inputs).map((path) => resolve(path));
    expect(nodeInputs.some((path) => path.endsWith("/allocation-sampler.mjs"))).toBe(true);
    expect(nodeInputs.some((path) => path.endsWith("/display.ts"))).toBe(false);
    const nodeImports = Object.values(node.metafile.inputs).flatMap((input) => input.imports);
    expect(nodeImports.some((edge) => edge.external && edge.path === "webgpu")).toBe(true);
    expect(nodeImports.some((edge) => edge.external && /^(?:bun:|playwright)/.test(edge.path))).toBe(
        false,
    );
}, 1_000);

test("a non-page allocation row can import the allocation instrument without loading Vite before requesting a page build", () => {
    const probe = `
            import { plugin } from "bun";
            globalThis.viteLoads = [];
            plugin({
                name: "allocation-vite-load-observer",
                setup(build) {
                    build.onLoad({ filter: /node_modules[/\\\\]vite[/\\\\]/ }, async (args) => {
                        globalThis.viteLoads.push(args.path);
                        return { contents: await Bun.file(args.path).text(), loader: "js" };
                    });
                },
            });
            const { allocationFailure } = await import(${JSON.stringify(resolve(import.meta.dir, "allocation.ts"))});
            if (allocationFailure({ warm: 1, windows: [] }) === undefined)
                throw new Error("allocation export did not evaluate");
            if (globalThis.viteLoads.length !== 0)
                throw new Error("allocation import loaded Vite: " + globalThis.viteLoads.join(", "));
            await import("vite");
            if (globalThis.viteLoads.length === 0)
                throw new Error("Vite load observer did not detect the explicit import");
        `;
    const proc = Bun.spawnSync([process.execPath, "--eval", probe], {
        stdout: "pipe",
        stderr: "pipe",
    });
    if (proc.exitCode !== 0)
        throw new Error(`Bun allocation import failed: ${proc.stderr.toString()}`);
}, 1_000);

// Profiler modules: the `profile` extra, which owns the physics step's timing clock.
const PROFILER = [/^src\/extras\/profile\//];

test("the allocation-gated first-person composition carries no timing or profiling module, so its default step does no diagnostics work", async () => {
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
                input.imports.some((edge) => resolve(ROOT, edge.path) === resolve(ROOT, module)),
            )
            .map(([path]) => path);
    const found = Object.keys(inputs).filter((path) =>
        PROFILER.some((pattern) => pattern.test(resolve(ROOT, path).slice(ROOT.length + 1))),
    );
    if (found.length !== 0)
        throw new Error(
            `gated bundle imports profiler modules:\n${found.map((path) => `  ${path} <- ${importers(path).join(", ")}`).join("\n")}`,
        );
}, 1_000);

const steadySample = (sites: AllocationSample["windows"][number]["sites"]) => ({
    warm: 120,
    windows: [
        { label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 },
        { label: "after warm 240", sites: [], frames: 120, framesAtMost: 120 },
        { label: "A/A repeat", sites, frames: 120, framesAtMost: 120 },
    ],
});

test("the allocation gate passes when all three expected steady windows sample zero bytes at zero sites", () => {
    const sample = steadySample([]);
    expect(allocatesNothing(sample)).toBe(true);
    expect(allocationFailure(sample)).toBeUndefined();
});

test("the allocation gate reds with the names of all expected windows when a sampler returns no steady windows", () => {
    const sample = { warm: 120, windows: [] as AllocationSample["windows"] };
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady allocation sample is missing expected windows: after warm 120, after warm 240, A/A repeat",
    );
});

test("the allocation gate reds with the names of expected steady windows omitted by an incomplete sampler result", () => {
    const sample = {
        warm: 120,
        windows: [{ label: "after warm 120", sites: [], frames: 120, framesAtMost: 120 }],
    };
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady allocation sample is missing expected windows: after warm 240, A/A repeat",
    );
});

test("the allocation gate reds on any sampled steady allocation and prints its site only for diagnosis", () => {
    const sample = steadySample([
        { site: "stepChunk src/transitional/character/sweep.ts:42", bytes: 96, count: 3 },
    ]);
    expect(allocatesNothing(sample)).toBe(false);
    expect(allocationFailure(sample)).toBe(
        "steady play allocated JavaScript heap; sampler sites are diagnosis only:\n  A/A repeat: 96 B at stepChunk src/transitional/character/sweep.ts:42",
    );
});
