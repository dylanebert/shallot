import { expect, test } from "bun:test";

import { resolve } from "node:path";
import { Glob } from "bun";
import { EXAMPLE_GATES } from "./example-gates";
import { selectExampleGates } from "./test-changed";

const RUNTIME_SRC = "packages/shallot-runtime/src";
const dirs = (paths: string[]) => selectExampleGates(paths).map((row) => row.dir);

/** Measured against the current 306 tracked runtime files. This is a lower-only ratchet: a larger
 * population is fine, but a higher selected-row ceiling is a re-widened cone; when the cones narrow,
 * lower this admission number with the new measured maximum. */
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

function runtimeFiles(): string[] {
    return trackedFiles().filter(
        (file) => file.startsWith(`${RUNTIME_SRC}/`) && file.endsWith(".ts"),
    );
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
    const blanket = [`${RUNTIME_SRC}/**`, "packages/shallot-runtime/**"];
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

test("the tracked runtime population has a lower-than-incumbent, lower-only selection ceiling", () => {
    const files = runtimeFiles();
    expect(files).toHaveLength(306);
    const worst = files
        .map((file) => ({ file, rows: selectExampleGates([file]).length }))
        .sort((a, b) => b.rows - a.rows || a.file.localeCompare(b.file));
    expect(worst[0].file).toBe("packages/shallot-runtime/src/standard/input/index.ts");
    expect(worst[0].rows).toBeLessThan(39);
    expect(worst[0].rows).toBe(MAX_ROWS_PER_RUNTIME_FILE);
    expect(worst.at(-1)?.rows).toBe(0);
    expect(worst[0].rows > MAX_ROWS_PER_RUNTIME_FILE).toBe(false);
});

test("an unrelated path selects no example row", () => {
    expect(dirs(["docs/selector.md"])).toEqual([]);
});
