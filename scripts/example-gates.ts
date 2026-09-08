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

/** Shallot-owned selection roster. Engine paths are deliberately included in every cone: an
 * engine change can move every consumer, while each example's own path selects only that row. */
export const EXAMPLE_GATES: ExampleGate[] = (
    [
        {
            dir: "examples/recipes/animate-with-clips",
            tier: "recipes",
            covers: ["examples/recipes/animate-with-clips/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe animate-with-clips",
        },
        {
            dir: "examples/recipes/annotate-the-world",
            tier: "recipes",
            covers: ["examples/recipes/annotate-the-world/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe annotate-the-world",
        },
        {
            dir: "examples/recipes/billboards-and-sprites",
            tier: "recipes",
            covers: [
                "examples/recipes/billboards-and-sprites/**",
                "packages/shallot-runtime/src/**",
            ],
            gate: "bun run recipes --recipe billboards-and-sprites",
        },
        {
            dir: "examples/recipes/breakable-joints",
            tier: "recipes",
            covers: ["examples/recipes/breakable-joints/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe breakable-joints",
        },
        {
            dir: "examples/recipes/build-a-scene",
            tier: "recipes",
            covers: ["examples/recipes/build-a-scene/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe build-a-scene",
            static: "authored scene structure has no runtime behavior",
        },
        {
            dir: "examples/recipes/compute-and-readback",
            tier: "recipes",
            covers: ["examples/recipes/compute-and-readback/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe compute-and-readback",
        },
        {
            dir: "examples/recipes/custom-material",
            tier: "recipes",
            covers: ["examples/recipes/custom-material/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe custom-material",
            static: "material appearance is static",
        },
        {
            dir: "examples/recipes/day-night-sky",
            tier: "recipes",
            covers: ["examples/recipes/day-night-sky/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe day-night-sky",
        },
        {
            dir: "examples/recipes/drive-a-vehicle",
            tier: "recipes",
            covers: ["examples/recipes/drive-a-vehicle/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe drive-a-vehicle",
        },
        {
            dir: "examples/recipes/first-person",
            tier: "recipes",
            covers: ["examples/recipes/first-person/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe first-person",
        },
        {
            dir: "examples/recipes/fog-and-light-shafts",
            tier: "recipes",
            covers: ["examples/recipes/fog-and-light-shafts/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe fog-and-light-shafts",
            static: "fog and light-shaft appearance is static",
        },
        {
            dir: "examples/recipes/game-loop",
            tier: "recipes",
            covers: ["examples/recipes/game-loop/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe game-loop",
        },
        {
            dir: "examples/recipes/gpu-particles",
            tier: "recipes",
            // the producer implementation is owned by the private `shallot-gpu-particles` workspace, so a
            // change there selects this recipe — its only in-repo consumer.
            covers: [
                "examples/recipes/gpu-particles/**",
                "packages/shallot-gpu-particles/**",
                "packages/shallot-runtime/src/**",
            ],
            gate: "bun run recipes --recipe gpu-particles",
        },
        {
            dir: "examples/recipes/import-a-model",
            tier: "recipes",
            covers: ["examples/recipes/import-a-model/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe import-a-model",
            static: "model import appearance is static",
        },
        {
            dir: "examples/recipes/joints",
            tier: "recipes",
            covers: ["examples/recipes/joints/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe joints",
        },
        {
            dir: "examples/recipes/measure-performance",
            tier: "recipes",
            covers: ["examples/recipes/measure-performance/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe measure-performance",
        },
        {
            dir: "examples/recipes/moving-platform",
            tier: "recipes",
            covers: ["examples/recipes/moving-platform/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe moving-platform",
        },
        {
            dir: "examples/recipes/orbit-camera",
            tier: "recipes",
            covers: ["examples/recipes/orbit-camera/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe orbit-camera",
            static: "camera movement is user-driven rather than autonomous",
        },
        {
            dir: "examples/recipes/overlay-ui",
            tier: "recipes",
            covers: ["examples/recipes/overlay-ui/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe overlay-ui",
        },
        {
            dir: "examples/recipes/physics-playground",
            tier: "recipes",
            covers: ["examples/recipes/physics-playground/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe physics-playground",
        },
        {
            dir: "examples/recipes/play-sound",
            tier: "recipes",
            covers: ["examples/recipes/play-sound/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe play-sound",
        },
        {
            dir: "examples/recipes/ragdoll",
            tier: "recipes",
            covers: ["examples/recipes/ragdoll/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe ragdoll",
        },
        {
            dir: "examples/recipes/render-to-a-terminal",
            tier: "recipes",
            covers: ["examples/recipes/render-to-a-terminal/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe render-to-a-terminal",
        },
        {
            dir: "examples/recipes/respond-to-input",
            tier: "recipes",
            covers: ["examples/recipes/respond-to-input/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe respond-to-input",
        },
        {
            dir: "examples/recipes/save-and-restore",
            tier: "recipes",
            covers: ["examples/recipes/save-and-restore/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe save-and-restore",
        },
        {
            dir: "examples/recipes/stylize-the-look",
            tier: "recipes",
            covers: ["examples/recipes/stylize-the-look/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe stylize-the-look",
        },
        {
            dir: "examples/recipes/surface-friction",
            tier: "recipes",
            covers: ["examples/recipes/surface-friction/**", "packages/shallot-runtime/src/**"],
            gate: "bun run recipes --recipe surface-friction",
        },
        ...["blank", "no-walls", "survive-reload", "ui-containment"].map(
            (name): ExampleGate => ({
                dir: `examples/flows/${name}`,
                tier: "flows",
                covers: [`examples/flows/${name}/**`, "packages/shallot-runtime/src/**"],
                gate: `bun run flows --flow ${name}`,
            }),
        ),
        ...["ascii", "collapse", "ocean", "roads", "sandbox", "visualization", "voxel"].map(
            (name): ExampleGate => ({
                dir: `examples/showcase/${name}`,
                tier: "showcase",
                covers: [`examples/showcase/${name}/**`, "packages/shallot-runtime/src/**"],
                gate: `bun run --cwd examples/showcase/${name} gate`,
                motion: ["ascii", "collapse", "roads", "visualization", "voxel"].includes(name),
            }),
        ),
        {
            dir: "examples/gym",
            tier: "gym",
            covers: ["examples/gym/**", "packages/shallot-runtime/src/**"],
            gate: "bun bench --sweep && bun run --cwd examples/gym gate",
        },
    ] satisfies ExampleGate[]
).map((row) => ({ ...row, covers: [...row.covers, "packages/shallot-tumble/src/**"] }));
