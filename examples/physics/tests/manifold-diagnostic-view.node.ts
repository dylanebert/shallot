import { expect, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { ManifoldPlugin, ManifoldSample } from "../src/manifold";
import { manifoldDiagnosticDrawing } from "../src/manifold-diagnostic-presentation";
import { manifoldTransform } from "../src/manifold-math";
import { createScene } from "../src/scenes";

test("Hull Culling draws retained edges, optional bounds and normals, and support probes", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [ManifoldPlugin],
        setup: (w) => {
            createScene(w, "hullCulling");
        },
    });
    try {
        app.world.tick();
        const s = app.world.resource(ManifoldSample),
            c = s.culling!,
            full = manifoldDiagnosticDrawing(app.world);
        expect(full.segments.length).toBeGreaterThan(c.keptA + c.keptB + 1);
        expect(full.labels).toHaveLength(0);
        expect(full.marks).toHaveLength(c.probeApplied ? 2 : 0);
        c.showCulled = false;
        c.showNormals = false;
        c.showSpheres = false;
        const retained = manifoldDiagnosticDrawing(app.world);
        expect(retained.segments).toHaveLength(
            c.edgesA.filter((state) => state !== 1).length +
                c.edgesB.filter((state) => state !== 1).length +
                1,
        );
        for (const [h, t, states] of [
            [s.hullA!, s.a, c.edgesA],
            [s.hullB!, s.b, c.edgesB],
        ] as const)
            for (let i = 0; i < h.edgeCount; i += 2)
                if (states[i / 2] === 2) {
                    const a = manifoldTransform(t, h.points[h.edges[i]!.origin]!),
                        b = manifoldTransform(t, h.points[h.edges[i + 1]!.origin]!);
                    expect(
                        retained.segments.some(
                            (line) =>
                                line.a.every((v, j) => v === [a.x, a.y, a.z][j]) &&
                                line.b.every((v, j) => v === [b.x, b.y, b.z][j]),
                        ),
                    ).toBe(true);
                }
        expect(
            full.segments.every((line) =>
                [...line.a, ...line.b, line.width].every(Number.isFinite),
            ),
        ).toBe(true);
    } finally {
        app.dispose();
    }
});
test("Edge Axis Wedge draws its selected hull edges and labeled Gaussian arcs", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [ManifoldPlugin],
        setup: (w) => {
            createScene(w, "edgeAxisWedge");
        },
    });
    try {
        app.world.tick();
        const s = app.world.resource(ManifoldSample),
            w = s.wedge!,
            drawing = manifoldDiagnosticDrawing(app.world);
        expect(w.found).toBe(true);
        expect(w.edgeWins).toBe(true);
        expect(drawing.marks).toHaveLength(2);
        expect(drawing.labels.map((label) => label.label.trim())).toEqual([
            "uA",
            "vA",
            "uB",
            "vB",
            "peak A",
            "peak B",
            "n",
            "d",
        ]);
        const edges = drawing.segments.filter((line) => line.width === 3);
        expect(edges).toHaveLength(2);
        for (const [h, t, index, line] of [
            [s.hullA!, s.a, w.edge.edgeA, edges[0]!],
            [s.hullB!, s.b, w.edge.edgeB, edges[1]!],
        ] as const) {
            const a = manifoldTransform(t, h.points[h.edges[index]!.origin]!),
                b = manifoldTransform(t, h.points[h.edges[index + 1]!.origin]!);
            expect(line.a).toEqual([a.x, a.y, a.z]);
            expect(line.b).toEqual([b.x, b.y, b.z]);
        }
        expect(
            drawing.segments.every((line) =>
                [...line.a, ...line.b, line.width].every(Number.isFinite),
            ),
        ).toBe(true);
        expect(drawing.labels.every((label) => label.point.every(Number.isFinite))).toBe(true);
    } finally {
        app.dispose();
    }
});
