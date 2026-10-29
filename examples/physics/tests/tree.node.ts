import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, createApp } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { INK } from "../src/ink-presentation";
import { scenePresentation } from "../src/presentation";
import { changeControl, controlValues, createScene, scenes } from "../src/scenes";
import { TreeBenchmarkPlugin, TreeBenchmarkState, treeRecords } from "../src/tree";
import { TREE_VISITED, treeBounds, treeQueryDrawing } from "../src/tree-presentation";

setDefaultTimeout(TIMEOUT.node);
test("Tree Benchmark draws query shapes in ink and distinguishes visited leaves without adding physics bodies", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [TreeBenchmarkPlugin],
        setup: (world) => {
            createScene(world, "treeBenchmark");
        },
    });
    try {
        const world = app.world,
            s = world.resource(TreeBenchmarkState),
            values = controlValues("treeBenchmark");
        const initial = treeBounds(world, [0, 0, 0]);
        expect(initial).toHaveLength(s.leaves);
        expect(initial.every((row) => row[6] === INK)).toBe(true);
        expect(treeQueryDrawing(world).segments).toHaveLength(0);
        for (const name of ["Ray Cast", "Overlap", "Closet Point"])
            changeControl(world, "treeBenchmark", [], values, name, 1);
        world.tick();
        const drawing = treeQueryDrawing(world);
        expect(drawing.segments).toHaveLength(73);
        expect(drawing.boxes).toHaveLength(1);
        expect(drawing.boxes[0]![6]).toBe(INK);
        expect(drawing.marks).toHaveLength(Number(s.haveClosest));
        expect(treeBounds(world, [0, 0, 0]).some((row) => row[6] === TREE_VISITED)).toBe(true);
        changeControl(world, "treeBenchmark", [], values, "Level", 0);
        expect(treeBounds(world, [0, 0, 0])).toHaveLength(1);
        changeControl(world, "treeBenchmark", [], values, "Level", -1);
        expect(treeBounds(world, [10000, 10000, 10000])).toHaveLength(0);
        expect(s.leaves).toBe(288);
        expect([...world.query([Body])]).toHaveLength(0);
    } finally {
        app.dispose();
    }
});
test("Tree Benchmark's grid follows the loaded bounds' floor", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [TreeBenchmarkPlugin],
        setup: (world) => {
            createScene(world, "treeBenchmark");
        },
    });
    try {
        const world = app.world,
            values = controlValues("treeBenchmark");
        const height = () => scenePresentation(world, "treeBenchmark").grid.lines[0]!.a[1];
        changeControl(world, "treeBenchmark", [], values, "Save", 1);
        changeControl(world, "treeBenchmark", [], values, "File", 1);
        expect(height()).toBeCloseTo(Math.fround(-12.8301) + 0.005, 10);
        changeControl(world, "treeBenchmark", [], values, "File", 2);
        expect(height()).toBeCloseTo(-74 + 0.005, 10);
        changeControl(world, "treeBenchmark", [], values, "Load Scale", 0.5);
        changeControl(world, "treeBenchmark", [], values, "Load", 1);
        expect(height()).toBeCloseTo(Math.fround(-4.79) * 0.5 + 0.005, 10);
    } finally {
        app.dispose();
    }
});
test("Tree Benchmark owns a page-run save slot across scene switches and restores standalone state with snapshots", async () => {
    const app = await createApp({ defaults: false, plugins: [TreeBenchmarkPlugin] });
    try {
        const world = app.world,
            empty = world.snapshot(),
            values = controlValues("treeBenchmark");
        createScene(world, "treeBenchmark");
        const s = world.resource(TreeBenchmarkState);
        changeControl(world, "treeBenchmark", [], values, "Save", 1);
        changeControl(world, "treeBenchmark", [], values, "Overlap", 1);
        world.tick();
        const records = treeRecords(world),
            image = world.snapshot();
        changeControl(world, "treeBenchmark", [], values, "File", 1);
        expect(s.leaves).toBe(4221);
        world.restore(image);
        expect(treeRecords(world)).toEqual(records);
        world.restore(empty);
        createScene(world, "boxStack");
        expect(s.active).toBe(false);
        createScene(world, "treeBenchmark");
        const prior = structuredClone(s.proxies[0]!.bounds);
        changeControl(world, "treeBenchmark", [], values, "Load Scale", 0.5);
        changeControl(world, "treeBenchmark", [], values, "File", 2);
        expect(s.leaves).toBe(268);
        changeControl(world, "treeBenchmark", [], values, "Load", 1);
        expect(s.leaves).toBe(288);
        expect(s.proxies[0]!.bounds.lowerBound.x).toBe(Math.fround(prior.lowerBound.x * 0.5));
        expect(s.proxies[0]!.bounds.upperBound.z).toBe(Math.fround(prior.upperBound.z * 0.5));
        for (const name of [
            "leaves",
            "height",
            "area ratio",
            "build time",
            "ray time",
            "overlap time",
            "closest time",
        ])
            expect(scenes.treeBenchmark.controls[name]!.read).toBeDefined();
    } finally {
        app.dispose();
    }
});
