import { SCENARIO_GATES } from "../examples/gym/src/scenarios/timeouts";

export type ExampleTier = "recipes" | "flows" | "showcase" | "gym";

export interface ExampleGate {
    dir: string;
    tier: ExampleTier;
    covers: string[];
    gate: string;
    /** Why a recipe has no runtime observable. Mutually exclusive with src/smoke.ts. */
    static?: string;
    motion?: boolean;
}

/** A runtime module cone. A row claims a module only when its surviving assertion observes behavior from
 * that module; the row's own directory is added separately to every cone below. */
const src = (...modules: string[]): string[] =>
    modules.map((module) => `packages/shallot/src/${module}/**`);

/** The files crossed by a broad public `@dylanebert/shallot` import. Feature-specific recipe smokes use
 * narrower cones instead; whole showcases and flows that assert the public app surface retain these
 * actual barrel boundaries without claiming every implementation behind them. */
const BARRELS = [
    "packages/shallot/src/index.ts",
    "packages/shallot/src/engine/index.ts",
    "packages/shallot/src/extras/index.ts",
    "packages/shallot/src/standard/index.ts",
    "packages/shallot/src/standard/defaults.ts",
];

const HARNESS = src("harness");
const INPUT = src("standard/input");
const TRANSFORMS = src("standard/transforms");
const ORBIT = src("extras/orbit");
/** Flow and rendered-page assertions observe the boot path even when their specific claim is elsewhere. */
const BOOT = src("engine/ecs", "standard/glaze", "standard/loading");

/** The rigid-body surface and both backends read by the physics assertions. */
const PHYSICS = [
    ...src("standard/physics", "standard/avbd", "standard/tumble"),
    "packages/shallot-tumble/src/**",
];

/** Shallot-owned selection roster. Each row's cone is its own directory plus the runtime and producer
 * modules its surviving assertion claims about. Whole-roster escalation on the lock or any manifest
 * lives in `test-changed.ts`. */
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
        covers: ["examples/recipes/billboards-and-sprites/**", ...src("extras/sprite")],
        gate: "bun run recipes --recipe billboards-and-sprites",
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
            ...src("engine/runtime", "extras/text", "standard/slab", "standard/mirror"),
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
        covers: ["examples/recipes/drive-a-vehicle/**", ...PHYSICS, ...INPUT],
        gate: "bun run recipes --recipe drive-a-vehicle",
    },
    {
        dir: "examples/recipes/first-person",
        tier: "recipes",
        // The incumbent smoke observes a pose after a keyboard action. It is intentionally a narrow
        // input/transform claim, not a blanket claim over the Player/Character boot graph.
        covers: ["examples/recipes/first-person/**", ...INPUT, ...TRANSFORMS],
        gate: "bun run recipes --recipe first-person",
    },
    {
        dir: "examples/recipes/fog-and-light-shafts",
        tier: "recipes",
        covers: ["examples/recipes/fog-and-light-shafts/**"],
        gate: "bun run recipes --recipe fog-and-light-shafts",
        static: "fog and light-shaft appearance is static",
    },
    {
        dir: "examples/recipes/game-loop",
        tier: "recipes",
        covers: ["examples/recipes/game-loop/**", ...TRANSFORMS],
        gate: "bun run recipes --recipe game-loop",
    },
    {
        dir: "examples/recipes/gpu-particles",
        tier: "recipes",
        // The producer implementation is owned by the private `shallot-gpu-particles` workspace, so a
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
        dir: "examples/recipes/orbit-camera",
        tier: "recipes",
        covers: ["examples/recipes/orbit-camera/**"],
        gate: "bun run recipes --recipe orbit-camera",
        static: "camera movement is user-driven rather than autonomous",
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
        // This weak incumbent check observes a Transform after its keyboard action; it does not claim the
        // positional audio path, so the audio implementation is not smuggled into its cone.
        covers: ["examples/recipes/play-sound/**", ...INPUT, ...TRANSFORMS],
        gate: "bun run recipes --recipe play-sound",
    },
    {
        dir: "examples/recipes/ragdoll",
        tier: "recipes",
        // Keep the still-present input/transform assertion subject. A later recipe reframe may narrow or
        // replace this claim; this selection cut does not delete the current smoke.
        covers: ["examples/recipes/ragdoll/**", ...INPUT, ...TRANSFORMS],
        gate: "bun run recipes --recipe ragdoll",
    },
    {
        dir: "examples/recipes/render-to-a-terminal",
        tier: "recipes",
        covers: ["examples/recipes/render-to-a-terminal/**", ...INPUT, ...ORBIT],
        gate: "bun run recipes --recipe render-to-a-terminal",
    },
    {
        dir: "examples/recipes/respond-to-input",
        tier: "recipes",
        covers: ["examples/recipes/respond-to-input/**", ...INPUT, ...TRANSFORMS],
        gate: "bun run recipes --recipe respond-to-input",
    },
    {
        dir: "examples/recipes/save-and-restore",
        tier: "recipes",
        // The smoke observes the authored XML written by serialize/stringify after the InputPlugin key
        // edge. Restore remains unasserted, but the save subject is live and stays selected here.
        covers: ["examples/recipes/save-and-restore/**", ...INPUT, ...src("engine/scene")],
        gate: "bun run recipes --recipe save-and-restore",
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
        dir: "examples/flows/blank",
        tier: "flows",
        covers: ["examples/flows/blank/**", ...HARNESS, ...BOOT, ...src("engine/app"), ...BARRELS],
        gate: "bun run flows --flow blank",
    },
    {
        dir: "examples/flows/no-walls",
        tier: "flows",
        covers: [
            "examples/flows/no-walls/**",
            ...HARNESS,
            ...BOOT,
            ...BARRELS,
            ...src(
                "engine/app",
                "engine/runtime",
                "standard/mirror",
                "standard/render",
                "standard/sear",
            ),
        ],
        gate: "bun run flows --flow no-walls",
    },
    {
        dir: "examples/flows/survive-reload",
        tier: "flows",
        covers: [
            "examples/flows/survive-reload/**",
            ...HARNESS,
            ...BOOT,
            ...BARRELS,
            ...src("engine/app", "engine/scene"),
        ],
        gate: "bun run flows --flow survive-reload",
    },
    {
        dir: "examples/flows/ui-containment",
        tier: "flows",
        covers: [
            "examples/flows/ui-containment/**",
            ...HARNESS,
            ...BOOT,
            ...BARRELS,
            ...src("engine/app"),
        ],
        gate: "bun run flows --flow ui-containment",
    },
    {
        dir: "examples/showcase/ascii",
        tier: "showcase",
        covers: [
            "examples/showcase/ascii/**",
            ...HARNESS,
            ...BOOT,
            ...src("extras/cells", "extras/orbit", "standard/input"),
        ],
        gate: "bun run --cwd examples/showcase/ascii gate",
        motion: true,
    },
    {
        dir: "examples/showcase/collapse",
        tier: "showcase",
        covers: [
            "examples/showcase/collapse/**",
            ...BARRELS,
            ...HARNESS,
            ...BOOT,
            ...src("extras/orbit", "standard/input", "standard/avbd", "standard/render"),
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
            ...HARNESS,
            ...BOOT,
            ...src("standard/render", "standard/sear", "engine/utils"),
        ],
        gate: "bun run --cwd examples/showcase/ocean gate",
    },
    {
        dir: "examples/showcase/roads",
        tier: "showcase",
        covers: [
            "examples/showcase/roads/**",
            ...BARRELS,
            ...HARNESS,
            ...BOOT,
            ...ORBIT,
            ...INPUT,
            ...src(
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
            ...HARNESS,
            ...BOOT,
            ...INPUT,
            ...src("standard/player", "standard/render"),
        ],
        gate: "bun run --cwd examples/showcase/sandbox gate",
    },
    {
        dir: "examples/showcase/visualization",
        tier: "showcase",
        covers: [
            "examples/showcase/visualization/**",
            ...BARRELS,
            ...HARNESS,
            ...BOOT,
            ...ORBIT,
            ...INPUT,
            ...TRANSFORMS,
            ...src("extras/animation", "extras/text", "standard/render"),
        ],
        gate: "bun run --cwd examples/showcase/visualization gate",
        motion: true,
    },
    {
        dir: "examples/showcase/voxel",
        tier: "showcase",
        covers: [
            "examples/showcase/voxel/**",
            ...BARRELS,
            ...HARNESS,
            ...BOOT,
            ...ORBIT,
            ...INPUT,
            ...src(
                "extras/lines",
                "engine/runtime",
                "engine/utils",
                "standard/physics",
                "standard/render",
                "standard/sear",
            ),
        ],
        gate: "bun run --cwd examples/showcase/voxel gate",
        motion: true,
    },
    {
        dir: "examples/gym",
        tier: "gym",
        // GPU scenario metadata is one part of the claim. The shared gym host and the non-GPU backend,
        // character/player/transforms, mirror readback, fog probe, grab, input, and orbit paths are live
        // gate subjects too, so retain them explicitly instead of deriving this row from SCENARIO_GATES
        // alone.
        covers: [
            "examples/gym/**",
            ...BARRELS,
            ...BOOT,
            ...new Set(Object.values(SCENARIO_GATES).flatMap((gate) => gate.covers ?? [])),
            ...src(
                "extras/orbit",
                "standard/character",
                "standard/fog",
                "standard/input",
                "standard/mirror",
                "standard/physics",
                "standard/player",
                "standard/transforms",
                "standard/tumble",
            ),
            "packages/shallot-tumble/src/**",
        ],
        gate: "bun bench --sweep && bun run --cwd examples/gym gate",
    },
] satisfies ExampleGate[];
