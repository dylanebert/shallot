import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BenchmarkSamplePlugin } from "../src/benchmarks";
import { drawingBounds, drawSet } from "../src/draw-set";
import { shapePresentation } from "../src/presentation";
import { createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

test("the draw cube widens to include shadow casters on a wide view", () => {
    const bounds = drawingBounds([0, 0, 0], [0, 0, -1], 40);
    // The final Box3D shadow slice is bounded by a sphere swept 50 m toward the sun.
    const near = 0.5 * 10 * 8 ** 0.75 + 0.5 * (10 + 70 * 0.75);
    const height = 80 * Math.tan(Math.PI / 6);
    const radius = Math.hypot((80 - near) / 2, height, height * 40);
    expect(bounds.lowerBound.x).toBe(-radius);
    expect(bounds.upperBound.x).toBe(radius + 25 / Math.hypot(0.5, 0.8, 0.4));
    expect(bounds.lowerBound.z).toBe(-(near + 80) / 2 - radius);
});

test("Large World's opening draw query visits exactly its overlapping floor boxes, with edges", async () => {
    let bodies: number[] = [];
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, BenchmarkSamplePlugin],
        setup(world) {
            bodies = createScene(world, "largeWorld");
        },
    });
    try {
        app.world.tick();
        const pitch = (10 * Math.PI) / 180;
        const eye = [0, 250 * Math.sin(pitch), 250 * Math.cos(pitch)];
        const bounds = drawingBounds(eye, [0, -Math.sin(pitch), -Math.cos(pitch)], 16 / 9);
        expect(bounds.lowerBound).toEqual({ x: -1000, y: eye[1]! - 1000, z: eye[2]! - 1000 });
        expect(bounds.upperBound).toEqual({ x: 1000, y: eye[1]! + 1000, z: eye[2]! + 1000 });
        const visible = drawSet(app.world, bounds);
        const expected = new Set<number>();
        for (let i = 399; i <= 600; i++)
            for (let j = 424; j <= 624; j++) expected.add(bodies[i * 1000 + j]!);
        expect(visible.size).toBe(40_602);
        expect(visible).toEqual(expected);
        expect(shapePresentation(app.world, visible.values().next().value!).edges).toHaveLength(12);
        // Moving the eye must query the world again, not retain an opening-view slice.
        const moved = drawSet(
            app.world,
            drawingBounds([3000, eye[1]!, eye[2]!], [0, 0, -1], 16 / 9),
        );
        expect(moved.size).toBe(40_602);
        expect([...visible].some((eid) => moved.has(eid))).toBe(false);
    } finally {
        app.dispose();
    }
});
