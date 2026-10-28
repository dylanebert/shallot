import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Geometry, GeometryPlugin, geometryNames } from "../src/geometry";
import { drawGeometry, geometryDrawing } from "../src/geometry-presentation";
import { JOINT_INK, JOINT_LINE } from "../src/joint-presentation";
import { changeControl, controlValues, createScene, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
for (const scene of geometryNames)
    test(`${scene}: scene ink outlines its geometry without world axes or bodies`, async () => {
        const app = await createApp({ defaults: false, plugins: [GeometryPlugin] });
        try {
            expect(createScene(app.world, scene)).toEqual([]);
            expect(app.world.resource(PhysicsWorldDefinition).enableAutomaticStep).toBe(false);
            app.world.tick();
            const s = app.world.resource(Geometry),
                drawing = geometryDrawing(app.world);
            const edgeCount = s.hulls.reduce((n, h) => n + (h?.edgeCount ?? 0) / 2, 0);
            expect(drawing.length).toBeGreaterThanOrEqual(edgeCount);
            if (scene !== "capsuleMass") expect(drawing).toHaveLength(edgeCount);
            else expect(drawing.length).toBeGreaterThan(edgeCount);
            expect(drawing.every(({ color }) => color === JOINT_INK)).toBe(true);
            const calls: unknown[][] = [];
            drawGeometry(app.world, (...args) => {
                calls.push(args);
            });
            expect(calls).toEqual(drawing.map(({ a, b, color }) => [a, b, color, JOINT_LINE]));
            let cursor = 0;
            for (const [i, h] of s.hulls.entries())
                if (h) {
                    const x = scene === "hullTransform" ? (i === 0 ? -2 : 2) : 0;
                    for (const [j, e] of h.edges.entries())
                        if (j < e.twin) {
                            const p = h.points[e.origin]!,
                                q = h.points[h.edges[e.twin]!.origin]!;
                            expect(drawing[cursor++]!).toEqual({
                                a: [p.x + x, p.y, p.z],
                                b: [q.x + x, q.y, q.z],
                                color: JOINT_INK,
                            });
                        }
                }
            if (scene === "capsuleMass") {
                for (const axis of [0, 1, 2]) {
                    const values = drawing.flatMap(({ a, b }) => [a[axis]!, b[axis]!]);
                    expect(Math.min(...values)).toBe(axis === 0 ? -2 : -1);
                    expect(Math.max(...values)).toBe(axis === 0 ? 2 : 1);
                }
                expect(s.masses).toHaveLength(3);
                const rows = Object.values(scenes[scene].controls!).filter((c) => c.read);
                expect(rows).toHaveLength(12);
                expect(rows.every((c) => c.label.endsWith("kg") || c.label.endsWith("kg·m²"))).toBe(
                    true,
                );
            }
            if (scene === "boxHull" || scene === "hullTransform") {
                const name = scene === "boxHull" ? "c X" : "px";
                changeControl(app.world, scene, [], controlValues(scene), name, 0.8);
                expect(geometryDrawing(app.world)).not.toEqual(drawing);
            }
        } finally {
            app.dispose();
        }
    });

test("switching away clears geometry ink and returning resets controls and masses", async () => {
    const app = await createApp({ defaults: false, plugins: [GeometryPlugin] });
    try {
        createScene(app.world, "capsuleMass");
        app.world.tick();
        createScene(app.world, "boxStack");
        expect(geometryDrawing(app.world)).toEqual([]);
        expect(app.world.resource(Geometry).masses).toEqual([]);
        createScene(app.world, "capsuleMass");
        app.world.tick();
        expect(app.world.resource(Geometry).sides).toBe(6);
        expect(app.world.resource(Geometry).masses).toHaveLength(3);
    } finally {
        app.dispose();
    }
});
