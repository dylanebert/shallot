import { expect } from "bun:test";
import { resolve } from "node:path";
import {
    type AllocationSample,
    allocatesNothing,
    allocationFailure,
} from "@dylanebert/shallot/harness/allocation";
import { check } from "@dylanebert/shallot/harness/check";

const ENTRY = resolve(import.meta.dir, "../../examples/first-person/src/allocation.entry.ts");
const ROOT = resolve(import.meta.dir, "../..");

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
