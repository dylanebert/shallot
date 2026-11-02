import type { World } from "@dylanebert/shallot";
import { computeCosSin, type HullData } from "@dylanebert/shallot/standard/physics";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";
import {
    add,
    cross,
    manifoldRotate,
    manifoldTransform,
    mul,
    normalize,
    sub,
    type Vec,
    vector,
} from "./manifold-math";
import { ManifoldSample } from "./manifold-state";

type Point = [number, number, number];
const f = Math.fround,
    AMBER = 0xff8135,
    CYAN = 0x38a4b5,
    MUTED = 0x797369,
    FAINT = 0x49453f,
    SELECTED = 0xff5600;
const point = (v: Vec): Point => [v.x, v.y, v.z];
export function manifoldDiagnosticDrawing(world: World) {
    const s = world.resource(ManifoldSample);
    const segments: { a: Point; b: Point; color: number; width: number }[] = [],
        marks: { point: Point; color: number; size: number }[] = [],
        labels: { point: Point; label: string }[] = [];
    const line = (a: Vec, b: Vec, color: number, width = 1) =>
        segments.push({ a: point(a), b: point(b), color, width });
    const mark = (v: Vec, color: number, size: number) =>
        marks.push({ point: point(v), color, size });
    function arrow(a: Vec, b: Vec, color: number, width = 2, head = 0.25): void {
        line(a, b, color, width);
        const delta = sub(b, a),
            axis = normalize(delta),
            perp = normalize(cross(axis, Math.abs(axis.x) < 0.8 ? vector(1) : vector(0, 1))),
            back = sub(b, mul(head, delta)),
            side = mul(f(head * 0.4), cross(perp, delta));
        line(b, add(back, side), color, width);
        line(b, sub(back, side), color, width);
    }
    function sphere(
        transform: typeof s.a,
        center: Vec,
        radius: number,
        count: number,
        color: number,
    ): void {
        const [c, d] = computeCosSin(world, f(f(2 * f(Math.PI)) / count));
        for (let plane = 0; plane < 3; plane++) {
            let x = 1,
                y = 0;
            const vertex = (x: number, y: number) =>
                manifoldTransform(
                    transform,
                    add(
                        center,
                        plane === 0
                            ? vector(f(radius * x), f(radius * y), 0)
                            : plane === 1
                              ? vector(0, f(radius * x), f(radius * y))
                              : vector(f(radius * y), 0, f(radius * x)),
                    ),
                );
            let previous = vertex(x, y);
            for (let i = 0; i < count; i++) {
                const nx = f(f(c * x) - f(d * y)),
                    ny = f(f(d * x) + f(c * y)),
                    next = vertex(nx, ny);
                line(previous, next, color);
                x = nx;
                y = ny;
                previous = next;
            }
        }
    }
    if (s.culling) {
        const c = s.culling;
        for (const [h, t, states, faces, seed, color] of [
            [s.hullA!, s.a, c.edgesA, c.facesA, c.seedA, AMBER],
            [s.hullB!, s.b, c.edgesB, c.facesB, c.seedB, CYAN],
        ] as const) {
            for (let i = 0; i < h.edgeCount; i += 2) {
                const state = states[i / 2]!;
                if (state === 1 && !c.showCulled) continue;
                line(
                    manifoldTransform(t, h.points[h.edges[i]!.origin]!),
                    manifoldTransform(t, h.points[h.edges[i + 1]!.origin]!),
                    state === 2 ? color : state === 1 ? MUTED : FAINT,
                    state === 2 ? 2 : 1,
                );
            }
            if (c.showNormals)
                for (let i = 0; i < h.faceCount; i++) {
                    const state = faces[i]!;
                    if (state === 0 || (state === 1 && !c.showCulled)) continue;
                    let centroid = vector(),
                        count = 0,
                        e = h.faces[i]!.edge;
                    do {
                        centroid = add(centroid, h.points[h.edges[e]!.origin]!);
                        count++;
                        e = h.edges[e]!.next;
                    } while (e !== h.faces[i]!.edge);
                    centroid = mul(f(1 / count), centroid);
                    const a = manifoldTransform(t, centroid),
                        b = manifoldTransform(t, add(centroid, mul(f(0.4), h.planes[i]!.normal)));
                    if (i === seed) arrow(a, b, SELECTED, 2);
                    else if (state === 2) arrow(a, b, color);
                    else line(a, b, MUTED);
                }
            if (c.showSpheres) sphere(t, h.center, h.innerRadius, 32, color);
        }
        line(manifoldTransform(s.a, s.hullA!.center), manifoldTransform(s.b, s.hullB!.center), INK);
        if (c.probeApplied) {
            mark(manifoldTransform(s.a, s.hullA!.points[c.probeA]!), SELECTED, 12);
            mark(manifoldTransform(s.b, s.hullB!.points[c.probeB]!), SELECTED, 12);
        }
    } else if (s.wedge) {
        const w = s.wedge;
        function hull(h: HullData, t: typeof s.a): void {
            for (let i = 0; i < h.edgeCount; i += 2)
                line(
                    manifoldTransform(t, h.points[h.edges[i]!.origin]!),
                    manifoldTransform(t, h.points[h.edges[i + 1]!.origin]!),
                    MUTED,
                );
        }
        hull(s.hullA!, s.a);
        hull(s.hullB!, s.b);
        const centerA = manifoldTransform(s.a, s.hullA!.center),
            centerB = manifoldTransform(s.b, s.hullB!.center);
        line(centerA, centerB, INK);
        if (w.edge.edgeA !== -1 && w.arcA && w.arcB) {
            for (const [h, t, i, color] of [
                [s.hullA!, s.a, w.edge.edgeA, AMBER],
                [s.hullB!, s.b, w.edge.edgeB, CYAN],
            ] as const)
                line(
                    manifoldTransform(t, h.points[h.edges[i]!.origin]!),
                    manifoldTransform(t, h.points[h.edges[i + 1]!.origin]!),
                    color,
                    3,
                );
            const h = s.hullA!,
                e = w.edge.edgeA,
                mid = mul(
                    0.5,
                    add(h.points[h.edges[e]!.origin]!, h.points[h.edges[e + 1]!.origin]!),
                ),
                at = manifoldTransform(s.a, mid),
                normal = manifoldRotate(s.a.q, w.edge.normal);
            arrow(at, add(at, normal), SELECTED, 2, 0.2);
            const gauss = (v: Vec) =>
                add(w.gaussCenter, mul(w.gaussRadius, manifoldRotate(s.a.q, v)));
            sphere(
                { p: w.gaussCenter, q: { v: vector(), s: 1 } },
                vector(),
                w.gaussRadius,
                24,
                FAINT,
            );
            for (const [arc, color] of [
                [w.arcA, AMBER],
                [w.arcB, CYAN],
            ] as const) {
                const axis = normalize(cross(arc.u, arc.v)),
                    v = cross(axis, arc.u),
                    [dc, ds] = computeCosSin(world, f(f(2 * f(Math.PI)) / 96));
                let x = 1,
                    y = 0,
                    previous = gauss(arc.u);
                for (let i = 0; i < 96; i++) {
                    const nx = f(f(dc * x) - f(ds * y)),
                        ny = f(f(ds * x) + f(dc * y)),
                        next = gauss(add(mul(nx, arc.u), mul(ny, v)));
                    line(previous, next, FAINT);
                    x = nx;
                    y = ny;
                    previous = next;
                }
                previous = gauss(arc.u);
                for (let i = 1; i <= 32; i++) {
                    const fraction = f(i / 32),
                        next = gauss(
                            normalize(add(mul(f(1 - fraction), arc.u), mul(fraction, arc.v))),
                        );
                    line(previous, next, color, 2);
                    previous = next;
                }
                arrow(w.gaussCenter, gauss(arc.peak), SELECTED, 1, 0.15);
                mark(gauss(arc.peak), SELECTED, 7);
            }
            const gaussNormal = gauss(w.edge.normal),
                direction = gauss(normalize(w.centerOffset));
            arrow(w.gaussCenter, gaussNormal, SELECTED, 2, 0.15);
            arrow(w.gaussCenter, direction, INK, 1, 0.15);
            const annotations = [
                [w.arcA.u, "uA"],
                [w.arcA.v, "vA"],
                [w.arcB.u, "uB"],
                [w.arcB.v, "vB"],
                [w.arcA.peak, "peak A"],
                [w.arcB.peak, "peak B"],
                [w.edge.normal, "n"],
                [normalize(w.centerOffset), "d"],
            ] as const;
            for (const [i, [v, label]] of annotations.entries()) {
                const at = add(w.gaussCenter, vector(-5, 3 - i * 0.85, 0));
                labels.push({ point: point(at), label });
                line(add(at, vector(2.5, 0, 0)), gauss(v), MUTED);
            }
        }
    }
    return { segments, marks, labels };
}
export function drawManifoldDiagnostics(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const drawing = manifoldDiagnosticDrawing(world);
    for (const p of drawing.segments) line(p.a, p.b, p.color, { ...JOINT_LINE, width: p.width });
    for (const p of drawing.marks) drawInkPoint(world, camera, p.point, line, p.color, p.size);
}
