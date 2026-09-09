import { SCENARIO_GATES } from "../bench/src/scenarios/timeouts";

export type ExampleTier = "recipes" | "showcase" | "bench";

export interface ExampleGate {
    dir: string;
    tier: ExampleTier;
    covers: string[];
    gate: string;
    /** Why a recipe has no runtime observable. Mutually exclusive with src/smoke.ts. */
    static?: string;
    /** Why a recipe that does move has no check asserting its subject yet: the deleted smoke asserted
     *  something other than the concept, so the row's verdict is boot plus a nonblank render until the
     *  reframe queue gives it one. Mutually exclusive with src/smoke.ts, like `static`. */
    bootOnly?: string;
    motion?: boolean;
}

/** `packages/shallot-runtime/src/<module>/**` — the cone a row declares when its check claims about
 *  that module. Rows declare their own cones: a runtime file selects a row only when the row's
 *  surviving check asserts something about the module that file lives in. */
const src = (...modules: string[]): string[] =>
    modules.map((module) => `packages/shallot-runtime/src/${module}/**`);

/** The barrels a bare `@dylanebert/shallot` import resolves through. A showcase that imports the root
 *  surface claims about those files, not about every module reachable behind them. */
const BARRELS = [
    "packages/shallot-runtime/src/index.ts",
    "packages/shallot-runtime/src/engine/index.ts",
    "packages/shallot-runtime/src/extras/index.ts",
    "packages/shallot-runtime/src/standard/index.ts",
    "packages/shallot-runtime/src/standard/defaults.ts",
];

/** The rigid-body solver a physics check reads through: the `Physics` component surface, the AVBD
 *  solver behind it, and the tumble backend the runtime bridges to. */
const PHYSICS = [
    ...src("standard/physics", "standard/avbd", "standard/tumble"),
    "packages/shallot-tumble/src/**",
];

/** Shallot-owned selection roster. Each row's cone is its own directory plus the modules its surviving
 *  check claims about, so one runtime edit selects only the rows that claim about it. Whole-roster
 *  escalation on the lock or any manifest lives in `test-changed.ts`. */
export const EXAMPLE_GATES: ExampleGate[] = [
    {
        dir: "examples/recipes/animate-with-clips",
        tier: "recipes",
        covers: [
            "examples/recipes/animate-with-clips/**",
            ...src("extras/animation", "standard/transforms"),
        ],
        gate: "bun run recipes --recipe animate-with-clips",
    },
    {
        dir: "examples/recipes/annotate-the-world",
        tier: "recipes",
        covers: ["examples/recipes/annotate-the-world/**", ...src("extras/text")],
        gate: "bun run recipes --recipe annotate-the-world",
    },
    {
        dir: "examples/recipes/billboards-and-sprites",
        tier: "recipes",
        covers: ["examples/recipes/billboards-and-sprites/**"],
        gate: "bun run recipes --recipe billboards-and-sprites",
        bootOnly: "the sprite fill moved, but billboarding — the concept — was never asserted",
    },
    {
        dir: "examples/recipes/breakable-joints",
        tier: "recipes",
        covers: ["examples/recipes/breakable-joints/**", ...PHYSICS],
        gate: "bun run recipes --recipe breakable-joints",
    },
    {
        dir: "examples/recipes/build-a-scene",
        tier: "recipes",
        covers: ["examples/recipes/build-a-scene/**"],
        gate: "bun run recipes --recipe build-a-scene",
        static: "authored scene structure has no runtime behavior",
    },
    {
        dir: "examples/recipes/compute-and-readback",
        tier: "recipes",
        covers: [
            "examples/recipes/compute-and-readback/**",
            ...src("engine/runtime", "extras/text"),
        ],
        gate: "bun run recipes --recipe compute-and-readback",
    },
    {
        dir: "examples/recipes/custom-material",
        tier: "recipes",
        covers: ["examples/recipes/custom-material/**"],
        gate: "bun run recipes --recipe custom-material",
        static: "material appearance is static",
    },
    {
        dir: "examples/recipes/day-night-sky",
        tier: "recipes",
        covers: ["examples/recipes/day-night-sky/**", ...src("extras/sky", "standard/render")],
        gate: "bun run recipes --recipe day-night-sky",
    },
    {
        dir: "examples/recipes/drive-a-vehicle",
        tier: "recipes",
        covers: ["examples/recipes/drive-a-vehicle/**", ...PHYSICS, ...src("standard/input")],
        gate: "bun run recipes --recipe drive-a-vehicle",
    },
    {
        dir: "examples/recipes/first-person",
        tier: "recipes",
        covers: ["examples/recipes/first-person/**"],
        gate: "bun run recipes --recipe first-person",
        bootOnly: "the check asserted that some Transform moved, not that walking or looking works",
    },
    {
        dir: "examples/recipes/game-loop",
        tier: "recipes",
        covers: ["examples/recipes/game-loop/**"],
        gate: "bun run recipes --recipe game-loop",
        bootOnly: "the check asserted that some Transform moved, not that per-frame code ran",
    },
    {
        dir: "examples/recipes/gpu-particles",
        tier: "recipes",
        // the producer implementation is owned by the private `shallot-gpu-particles` workspace, so a
        // change there selects this recipe — its only in-repo consumer.
        covers: [
            "examples/recipes/gpu-particles/**",
            "packages/shallot-gpu-particles/**",
            ...src("engine/runtime", "standard/render", "standard/sear"),
        ],
        gate: "bun run recipes --recipe gpu-particles",
    },
    {
        dir: "examples/recipes/import-a-model",
        tier: "recipes",
        covers: ["examples/recipes/import-a-model/**"],
        gate: "bun run recipes --recipe import-a-model",
        static: "model import appearance is static",
    },
    {
        dir: "examples/recipes/joints",
        tier: "recipes",
        covers: ["examples/recipes/joints/**", ...PHYSICS],
        gate: "bun run recipes --recipe joints",
    },
    {
        dir: "examples/recipes/measure-performance",
        tier: "recipes",
        covers: ["examples/recipes/measure-performance/**", ...src("extras/profile")],
        gate: "bun run recipes --recipe measure-performance",
    },
    {
        dir: "examples/recipes/moving-platform",
        tier: "recipes",
        covers: ["examples/recipes/moving-platform/**", ...PHYSICS],
        gate: "bun run recipes --recipe moving-platform",
    },
    {
        dir: "examples/recipes/overlay-ui",
        tier: "recipes",
        covers: ["examples/recipes/overlay-ui/**", ...src("engine/app")],
        gate: "bun run recipes --recipe overlay-ui",
    },
    {
        dir: "examples/recipes/physics-playground",
        tier: "recipes",
        covers: ["examples/recipes/physics-playground/**", ...PHYSICS],
        gate: "bun run recipes --recipe physics-playground",
    },
    {
        dir: "examples/recipes/play-sound",
        tier: "recipes",
        covers: ["examples/recipes/play-sound/**"],
        gate: "bun run recipes --recipe play-sound",
        bootOnly:
            "the check asserted that some Transform moved, not that a positional sound played",
    },
    {
        dir: "examples/recipes/ragdoll",
        tier: "recipes",
        covers: ["examples/recipes/ragdoll/**"],
        gate: "bun run recipes --recipe ragdoll",
        bootOnly: "the check asserted that some Transform moved, not that the body went limp",
    },
    {
        dir: "examples/recipes/render-to-a-terminal",
        tier: "recipes",
        covers: ["examples/recipes/render-to-a-terminal/**"],
        gate: "bun run recipes --recipe render-to-a-terminal",
        bootOnly: "the check asserted orbit yaw changed, not that the terminal rendered",
    },
    {
        dir: "examples/recipes/respond-to-input",
        tier: "recipes",
        covers: [
            "examples/recipes/respond-to-input/**",
            ...src("standard/input", "standard/transforms"),
        ],
        gate: "bun run recipes --recipe respond-to-input",
    },
    {
        dir: "examples/recipes/save-and-restore",
        tier: "recipes",
        covers: ["examples/recipes/save-and-restore/**"],
        gate: "bun run recipes --recipe save-and-restore",
        bootOnly: "the check asserted the save; the reload it claims about is never exercised",
    },
    {
        dir: "examples/recipes/stylize-the-look",
        tier: "recipes",
        covers: ["examples/recipes/stylize-the-look/**", ...src("extras/outline")],
        gate: "bun run recipes --recipe stylize-the-look",
    },
    {
        dir: "examples/recipes/surface-friction",
        tier: "recipes",
        covers: ["examples/recipes/surface-friction/**", ...PHYSICS],
        gate: "bun run recipes --recipe surface-friction",
    },
    {
        dir: "examples/showcase/ascii",
        tier: "showcase",
        covers: ["examples/showcase/ascii/**", ...src("harness")],
        gate: "bun run --cwd examples/showcase/ascii gate",
        motion: true,
    },
    {
        dir: "examples/showcase/collapse",
        tier: "showcase",
        covers: [
            "examples/showcase/collapse/**",
            ...BARRELS,
            ...src("harness", "standard/avbd", "standard/render"),
        ],
        gate: "bun run --cwd examples/showcase/collapse gate",
        motion: true,
    },
    {
        dir: "examples/showcase/ocean",
        tier: "showcase",
        covers: [
            "examples/showcase/ocean/**",
            ...BARRELS,
            ...src("harness", "standard/render", "standard/sear", "engine/utils"),
        ],
        gate: "bun run --cwd examples/showcase/ocean gate",
    },
    {
        dir: "examples/showcase/roads",
        tier: "showcase",
        covers: [
            "examples/showcase/roads/**",
            ...BARRELS,
            ...src(
                "harness",
                "extras",
                "standard/physics",
                "standard/render",
                "standard/sear",
                "engine/runtime",
                "engine/utils",
            ),
        ],
        gate: "bun run --cwd examples/showcase/roads gate",
        motion: true,
    },
    {
        dir: "examples/showcase/sandbox",
        tier: "showcase",
        covers: [
            "examples/showcase/sandbox/**",
            ...BARRELS,
            ...src(
                "harness",
                "extras",
                "standard/audio",
                "standard/avbd",
                "standard/physics",
                "standard/render",
                "standard/sear",
                "engine/utils",
            ),
        ],
        gate: "bun run --cwd examples/showcase/sandbox gate",
    },
    {
        dir: "examples/showcase/visualization",
        tier: "showcase",
        covers: ["examples/showcase/visualization/**", ...BARRELS, ...src("harness", "extras")],
        gate: "bun run --cwd examples/showcase/visualization gate",
        motion: true,
    },
    {
        dir: "examples/showcase/voxel",
        tier: "showcase",
        covers: [
            "examples/showcase/voxel/**",
            ...BARRELS,
            ...src(
                "harness",
                "extras",
                "standard/physics",
                "standard/render",
                "standard/sear",
                "engine/runtime",
                "engine/utils",
            ),
        ],
        gate: "bun run --cwd examples/showcase/voxel gate",
        motion: true,
    },
    {
        dir: "bench",
        tier: "bench",
        // The bench selects by its scenarios' own declared cones, not by a blanket runtime glob: each
        // scenario's `covers` in `timeouts.ts` already names the GPU-side modules it exercises, and
        // `bun bench --for <path>` selects with the same table. Deriving the row's cone from it keeps one
        // source of truth — a scenario that stops covering a module stops selecting the row.
        covers: [
            "bench/**",
            ...new Set(Object.values(SCENARIO_GATES).flatMap((gate) => gate.covers ?? [])),
        ],
        gate: "bun bench --sweep && bun run --cwd bench gate",
    },
];
