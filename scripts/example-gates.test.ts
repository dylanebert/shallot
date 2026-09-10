import { expect, test } from "bun:test";

import { resolve } from "node:path";
import { Glob } from "bun";
import { EXAMPLE_GATES } from "./example-gates";
import { selectExampleGates } from "./test-changed";

const RUNTIME_SRC = "src";
const dirs = (paths: string[]) => selectExampleGates(paths).map((row) => row.dir);

/** Measured against the current 306 tracked runtime files. This is a lower-only ratchet: a larger
 * population is fine, but a higher selected-row ceiling is a re-widened cone; when the cones narrow,
 * the measured maximum may fall without forcing a pointless equality pin. */
const INCUMBENT_MAX_ROWS_PER_RUNTIME_FILE = 14;
const MAX_ROWS_PER_RUNTIME_FILE = 14;

const EXPECTED_DIRS = [
    "examples/recipes/animate-with-clips",
    "examples/recipes/annotate-the-world",
    "examples/recipes/billboards-and-sprites",
    "examples/recipes/breakable-joints",
    "examples/recipes/build-a-scene",
    "examples/recipes/compute-and-readback",
    "examples/recipes/custom-material",
    "examples/recipes/day-night-sky",
    "examples/recipes/drive-a-vehicle",
    "examples/recipes/first-person",
    "examples/recipes/fog-and-light-shafts",
    "examples/recipes/game-loop",
    "examples/recipes/gpu-particles",
    "examples/recipes/import-a-model",
    "examples/recipes/joints",
    "examples/recipes/measure-performance",
    "examples/recipes/moving-platform",
    "examples/recipes/orbit-camera",
    "examples/recipes/overlay-ui",
    "examples/recipes/physics-playground",
    "examples/recipes/play-sound",
    "examples/recipes/ragdoll",
    "examples/recipes/render-to-a-terminal",
    "examples/recipes/respond-to-input",
    "examples/recipes/save-and-restore",
    "examples/recipes/stylize-the-look",
    "examples/recipes/surface-friction",
    "examples/flows/blank",
    "examples/flows/no-walls",
    "examples/flows/survive-reload",
    "examples/flows/ui-containment",
    "examples/showcase/ascii",
    "examples/showcase/collapse",
    "examples/showcase/ocean",
    "examples/showcase/roads",
    "examples/showcase/sandbox",
    "examples/showcase/visualization",
    "examples/showcase/voxel",
    "examples/gym",
];

const STANDARD_RENDER_ROWS = [
    "examples/recipes/day-night-sky",
    "examples/recipes/gpu-particles",
    "examples/flows/no-walls",
    "examples/showcase/collapse",
    "examples/showcase/ocean",
    "examples/showcase/roads",
    "examples/showcase/sandbox",
    "examples/showcase/visualization",
    "examples/showcase/voxel",
    "examples/gym",
];

const INPUT_ROWS = [
    "examples/recipes/drive-a-vehicle",
    "examples/recipes/first-person",
    "examples/recipes/play-sound",
    "examples/recipes/ragdoll",
    "examples/recipes/render-to-a-terminal",
    "examples/recipes/respond-to-input",
    "examples/recipes/save-and-restore",
    "examples/showcase/ascii",
    "examples/showcase/collapse",
    "examples/showcase/roads",
    "examples/showcase/sandbox",
    "examples/showcase/visualization",
    "examples/showcase/voxel",
    "examples/gym",
];

const BOOT_ROWS = [
    "examples/flows/blank",
    "examples/flows/no-walls",
    "examples/flows/survive-reload",
    "examples/flows/ui-containment",
    "examples/showcase/ascii",
    "examples/showcase/collapse",
    "examples/showcase/ocean",
    "examples/showcase/roads",
    "examples/showcase/sandbox",
    "examples/showcase/visualization",
    "examples/showcase/voxel",
    "examples/gym",
];

const EXPECTED_RUNTIME_MODULES = [
    "engine/app",
    "engine/ecs",
    "engine/runtime",
    "engine/scene",
    "engine/utils",
    "extras/animation",
    "extras/cells",
    "extras/gltf",
    "extras/lines",
    "extras/orbit",
    "extras/outline",
    "extras/profile",
    "extras/skin",
    "extras/sky",
    "extras/sprite",
    "extras/text",
    "standard/audio",
    "standard/avbd",
    "standard/bvh",
    "standard/character",
    "standard/fog",
    "standard/glaze",
    "standard/input",
    "standard/loading",
    "standard/mirror",
    "standard/part",
    "standard/physics",
    "standard/player",
    "standard/render",
    "standard/sear",
    "standard/slab",
    "standard/transforms",
    "standard/tumble",
    "types",
];

const RUNTIME_MODULE_EXEMPTIONS: Record<string, string> = {
    // The surviving play-sound smoke asserts a Transform after input; it does not assert audio output.
    "standard/audio": "play-sound has no surviving audio assertion subject",
    types: "env.d.ts is a declaration surface, not a runtime implementation module",
};

/** Exact witnesses for every runtime module appearing in a selection cone. A cone mutation must
 * change this table's measured membership, rather than hiding behind another row's incidental cone. */
const EXPECTED_RUNTIME_MODULE_ROWS: Record<string, string[]> = {
    "engine/app": [
        "examples/recipes/overlay-ui",
        "examples/flows/blank",
        "examples/flows/no-walls",
        "examples/flows/survive-reload",
        "examples/flows/ui-containment",
    ],
    "engine/ecs": BOOT_ROWS,
    "engine/runtime": [
        "examples/recipes/compute-and-readback",
        "examples/recipes/gpu-particles",
        "examples/flows/no-walls",
        "examples/showcase/roads",
        "examples/showcase/voxel",
        "examples/gym",
    ],
    "engine/scene": ["examples/recipes/save-and-restore", "examples/flows/survive-reload"],
    "engine/utils": [
        "examples/showcase/ocean",
        "examples/showcase/roads",
        "examples/showcase/voxel",
        "examples/gym",
    ],
    "extras/animation": ["examples/recipes/animate-with-clips", "examples/showcase/visualization"],
    "extras/cells": ["examples/showcase/ascii", "examples/gym"],
    "extras/gltf": ["examples/gym"],
    "extras/lines": ["examples/showcase/voxel", "examples/gym"],
    "extras/orbit": [
        "examples/recipes/render-to-a-terminal",
        "examples/showcase/ascii",
        "examples/showcase/collapse",
        "examples/showcase/roads",
        "examples/showcase/visualization",
        "examples/showcase/voxel",
        "examples/gym",
    ],
    "extras/outline": ["examples/recipes/stylize-the-look", "examples/gym"],
    "extras/profile": ["examples/recipes/measure-performance", "examples/gym"],
    "extras/skin": ["examples/gym"],
    "extras/sky": ["examples/recipes/day-night-sky", "examples/gym"],
    "extras/sprite": ["examples/recipes/billboards-and-sprites", "examples/gym"],
    "extras/text": [
        "examples/recipes/annotate-the-world",
        "examples/recipes/compute-and-readback",
        "examples/showcase/visualization",
        "examples/gym",
    ],
    "standard/audio": [],
    "standard/avbd": [
        "examples/recipes/breakable-joints",
        "examples/recipes/drive-a-vehicle",
        "examples/recipes/joints",
        "examples/recipes/moving-platform",
        "examples/recipes/physics-playground",
        "examples/recipes/surface-friction",
        "examples/showcase/collapse",
        "examples/gym",
    ],
    "standard/bvh": ["examples/gym"],
    "standard/character": ["examples/gym"],
    "standard/fog": ["examples/gym"],
    "standard/glaze": BOOT_ROWS,
    "standard/input": INPUT_ROWS,
    "standard/loading": BOOT_ROWS,
    "standard/mirror": [
        "examples/recipes/compute-and-readback",
        "examples/flows/no-walls",
        "examples/gym",
    ],
    "standard/part": ["examples/gym"],
    "standard/physics": [
        "examples/recipes/breakable-joints",
        "examples/recipes/drive-a-vehicle",
        "examples/recipes/joints",
        "examples/recipes/moving-platform",
        "examples/recipes/physics-playground",
        "examples/recipes/surface-friction",
        "examples/showcase/roads",
        "examples/showcase/voxel",
        "examples/gym",
    ],
    "standard/player": ["examples/showcase/sandbox", "examples/gym"],
    "standard/render": STANDARD_RENDER_ROWS,
    "standard/sear": [
        "examples/recipes/gpu-particles",
        "examples/flows/no-walls",
        "examples/showcase/ocean",
        "examples/showcase/roads",
        "examples/showcase/voxel",
        "examples/gym",
    ],
    "standard/slab": ["examples/recipes/compute-and-readback", "examples/gym"],
    "standard/transforms": [
        "examples/recipes/animate-with-clips",
        "examples/recipes/first-person",
        "examples/recipes/game-loop",
        "examples/recipes/play-sound",
        "examples/recipes/ragdoll",
        "examples/recipes/respond-to-input",
        "examples/showcase/visualization",
        "examples/gym",
    ],
    "standard/tumble": [
        "examples/recipes/breakable-joints",
        "examples/recipes/drive-a-vehicle",
        "examples/recipes/joints",
        "examples/recipes/moving-platform",
        "examples/recipes/physics-playground",
        "examples/recipes/surface-friction",
        "examples/gym",
    ],
    types: [],
};

function runtimeFiles(): string[] {
    return trackedFiles().filter(
        (file) => file.startsWith(`${RUNTIME_SRC}/`) && file.endsWith(".ts"),
    );
}

function runtimeImplementationFiles(): string[] {
    return runtimeFiles().filter((file) => !file.includes(".test."));
}

function runtimeModules(): string[] {
    return [
        ...new Set(
            runtimeImplementationFiles().flatMap((file) => {
                const parts = file.slice(`${RUNTIME_SRC}/`.length).split("/");
                if (parts[0] === "types") return ["types"];
                return ["engine", "extras", "standard"].includes(parts[0]) &&
                    parts[1] &&
                    !parts[1].endsWith(".ts")
                    ? [`${parts[0]}/${parts[1]}`]
                    : [];
            }),
        ),
    ].sort();
}

function rowsForRuntimeModule(module: string): string[] {
    return [
        ...new Set(
            runtimeImplementationFiles()
                .filter((file) => file.startsWith(`${RUNTIME_SRC}/${module}/`))
                .flatMap((file) => dirs([file])),
        ),
    ];
}

function trackedFiles(): string[] {
    const result = Bun.spawnSync(["git", "ls-files"], { cwd: resolve(import.meta.dir, "..") });
    expect(result.success, result.stderr.toString()).toBe(true);
    return result.stdout.toString().split("\n").filter(Boolean);
}

test("the current 27/4/7/gym roster is pinned by identity", () => {
    expect(EXAMPLE_GATES.map((row) => row.dir)).toEqual(EXPECTED_DIRS);
    expect(EXAMPLE_GATES.filter((row) => row.tier === "recipes")).toHaveLength(27);
    expect(EXAMPLE_GATES.filter((row) => row.tier === "flows")).toHaveLength(4);
    expect(EXAMPLE_GATES.filter((row) => row.tier === "showcase")).toHaveLength(7);
    expect(EXAMPLE_GATES.filter((row) => row.tier === "gym")).toHaveLength(1);
});

test("a runtime file selects the rows with the corresponding assertion subject", () => {
    expect(dirs([`${RUNTIME_SRC}/standard/render/plugin.ts`])).toEqual(STANDARD_RENDER_ROWS);
    expect(dirs([`${RUNTIME_SRC}/standard/input/index.ts`])).toEqual(INPUT_ROWS);
    expect(dirs([`${RUNTIME_SRC}/engine/ecs/state.ts`])).toEqual(BOOT_ROWS);
    expect(dirs([`${RUNTIME_SRC}/standard/glaze/index.ts`])).toEqual(BOOT_ROWS);
    expect(dirs([`${RUNTIME_SRC}/standard/loading/index.ts`])).toEqual(BOOT_ROWS);
    expect(dirs([`${RUNTIME_SRC}/standard/fog/index.ts`])).toEqual(["examples/gym"]);
    expect(dirs([`${RUNTIME_SRC}/extras/outline/index.ts`])).toEqual([
        "examples/recipes/stylize-the-look",
        "examples/gym",
    ]);
    expect(dirs([`${RUNTIME_SRC}/standard/slab/index.ts`])).toEqual([
        "examples/recipes/compute-and-readback",
        "examples/gym",
    ]);
    expect(dirs(["packages/shallot-tumble/src/index.ts"])).toEqual([
        "examples/recipes/breakable-joints",
        "examples/recipes/drive-a-vehicle",
        "examples/recipes/joints",
        "examples/recipes/moving-platform",
        "examples/recipes/physics-playground",
        "examples/recipes/surface-friction",
        "examples/gym",
    ]);
    expect(dirs([`${RUNTIME_SRC}/standard/character/index.ts`])).toEqual(["examples/gym"]);
    expect(dirs([`${RUNTIME_SRC}/standard/mirror/index.ts`])).toEqual([
        "examples/recipes/compute-and-readback",
        "examples/flows/no-walls",
        "examples/gym",
    ]);
    expect(dirs([`${RUNTIME_SRC}/standard/player/index.ts`])).toEqual([
        "examples/showcase/sandbox",
        "examples/gym",
    ]);
    expect(dirs([`${RUNTIME_SRC}/standard/transforms/index.ts`])).toEqual([
        "examples/recipes/animate-with-clips",
        "examples/recipes/first-person",
        "examples/recipes/game-loop",
        "examples/recipes/play-sound",
        "examples/recipes/ragdoll",
        "examples/recipes/respond-to-input",
        "examples/showcase/visualization",
        "examples/gym",
    ]);
});

test("static recipes own only their example directories", () => {
    for (const row of EXAMPLE_GATES.filter((candidate) => candidate.static)) {
        expect(row.covers).toEqual([`${row.dir}/**`]);
        expect(dirs([`${row.dir}/src/scene.scene`])).toEqual([row.dir]);
    }
});

test("every row starts with its own directory and no row carries the incumbent blanket", () => {
    const blanket = [`${RUNTIME_SRC}/**`, "**"];
    for (const row of EXAMPLE_GATES) {
        expect(row.covers[0]).toBe(`${row.dir}/**`);
        for (const cover of row.covers) expect(blanket).not.toContain(cover);
    }
});

test("every declared cover names at least one tracked path", () => {
    const tracked = trackedFiles();
    for (const row of EXAMPLE_GATES)
        for (const cover of row.covers)
            expect(
                tracked.some((file) => new Glob(cover).match(file)),
                `${row.dir}: ${cover}`,
            ).toBe(true);
});

test("live runtime modules select a row or name their explicit exemption", () => {
    const modules = runtimeModules();
    expect(modules).toEqual(EXPECTED_RUNTIME_MODULES);
    expect(modules).toEqual(Object.keys(EXPECTED_RUNTIME_MODULE_ROWS).sort());
    expect(modules.length).toBeGreaterThan(20);
    const empty = modules.filter((module) => rowsForRuntimeModule(module).length === 0);
    expect(empty).toEqual(Object.keys(RUNTIME_MODULE_EXEMPTIONS).sort());
    for (const module of modules) {
        const rows = rowsForRuntimeModule(module);
        expect(rows, `${module} membership`).toEqual(EXPECTED_RUNTIME_MODULE_ROWS[module]);
        if (RUNTIME_MODULE_EXEMPTIONS[module]) {
            expect(rows, RUNTIME_MODULE_EXEMPTIONS[module]).toEqual([]);
        } else {
            expect(rows, `${module} selects no example row`).not.toHaveLength(0);
        }
    }
});

test("the tracked runtime population has a lower-than-incumbent, lower-only selection ceiling", () => {
    const files = runtimeFiles();
    expect(files).toHaveLength(306);
    const worst = files
        .map((file) => ({ file, rows: selectExampleGates([file]).length }))
        .sort((a, b) => b.rows - a.rows || a.file.localeCompare(b.file));
    expect(MAX_ROWS_PER_RUNTIME_FILE).toBeLessThanOrEqual(INCUMBENT_MAX_ROWS_PER_RUNTIME_FILE);
    expect(worst[0].rows).toBeLessThan(39);
    expect(worst[0].rows).toBeLessThanOrEqual(MAX_ROWS_PER_RUNTIME_FILE);
    expect(worst.at(-1)?.rows).toBe(0);
});

test("an unrelated path selects no example row", () => {
    expect(dirs(["docs/selector.md"])).toEqual([]);
});
