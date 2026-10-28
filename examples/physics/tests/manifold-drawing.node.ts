import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { INK } from "../src/ink-presentation";
import {
    ManifoldPlugin,
    ManifoldSample,
    manifoldNames,
    manifoldPointLimits,
} from "../src/manifold";
import { manifoldDrawing, manifoldGeometry } from "../src/manifold-presentation";
import { changeControl, controlValues, createScene, enabledControls, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
for (const [index, scene] of manifoldNames.entries())
    test(`${scene}: two role-colored shapes, contact ink, a cache toggle and fixed result rows`, async () => {
        const app = await createApp({ defaults: false, plugins: [ManifoldPlugin] });
        const w = app.world;
        try {
            expect(createScene(w, scene)).toEqual([]);
            w.tick();
            expect(w.resource(PhysicsWorldDefinition).enableAutomaticStep).toBe(false);
            const s = w.resource(ManifoldSample),
                shapes = manifoldGeometry(w);
            expect(shapes.map((g) => g.role)).toEqual(["ground", "subject"]);
            expect(shapes[0]!.pose).toBe(s.a);
            expect(shapes[1]!.pose).toBe(s.b);
            for (const g of shapes) {
                expect(g.mesh.vertices.length % 8).toBe(0);
                expect(g.mesh.indices.length % 3).toBe(0);
                expect(g.edges.length).toBeGreaterThan(0);
                expect([...g.mesh.vertices].every(Number.isFinite)).toBe(true);
                expect([...g.mesh.indices].every((i) => i < g.mesh.vertices.length / 8)).toBe(true);
            }
            const controls = scenes[scene].controls!,
                rows = Object.values(controls).filter((c) => c.read);
            expect(rows).toHaveLength(1 + manifoldPointLimits[index]!);
            expect(rows[0]!.read!(w)).toBe(s.manifold!.pointCount);
            for (let i = 0; i < manifoldPointLimits[index]!; i++)
                expect(rows[i + 1]!.read!(w)).toBe(
                    i < s.manifold!.pointCount
                        ? scene.startsWith("triangle")
                            ? Math.fround(100 * s.manifold!.points[i]!.separation)
                            : s.manifold!.points[i]!.separation
                        : null,
                );
            const values = controlValues(scene);
            expect(enabledControls(scene, values).feature).toBeUndefined();
            changeControl(w, scene, [], values, "Use cache", 1);
            expect(enabledControls(scene, values).feature).toBe(controls.feature);
            changeControl(w, scene, [], values, "feature", 2);
            expect(s.manualFeature).toBe(2);
            changeControl(w, scene, [], values, "Use cache", 0);
            expect(s.manualFeature).toBe(2);
            expect(Object.keys(scenes[scene].controls!)).toEqual(Object.keys(controls));
            const m = s.manifold!;
            m.pointCount = 1;
            m.points[0]!.point = { x: 0, y: 0, z: 0 };
            m.points[0]!.separation = 0;
            m.normal = { x: 0, y: 1, z: 0 };
            let drawing = manifoldDrawing(w);
            expect(drawing.marks).toHaveLength(1);
            expect(drawing.segments).toHaveLength(1);
            expect(drawing.marks[0]!.color).toBe(INK);
            expect(drawing.segments[0]!.color).toBe(INK);
            m.points[0]!.separation = 0.01;
            drawing = manifoldDrawing(w);
            expect(drawing.marks[0]!.color).toBe(0x797369);
            expect(drawing.segments[0]!.color).toBe(INK);
            m.points[0]!.separation = -0.01;
            expect(manifoldDrawing(w).marks[0]!.color).toBe(INK);
            const t = scene.startsWith("triangle") ? s.b : s.a;
            expect(drawing.marks[0]!.point).toEqual([t.p.x, t.p.y, t.p.z]);
            m.pointCount = 0;
            expect(manifoldDrawing(w)).toEqual({ marks: [], segments: [] });
            createScene(w, "boxStack");
            expect(w.resource(ManifoldSample)).toEqual(ManifoldSample.create());
            expect(manifoldGeometry(w)).toEqual([]);
            expect(manifoldDrawing(w)).toEqual({ marks: [], segments: [] });
        } finally {
            app.dispose();
        }
    });
