import { expect, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { arcCanReach, makeArc, manifoldDiagnosticScenes } from "../src/manifold-diagnostics";
import { vector } from "../src/manifold-math";

test("diagnostic readouts tolerate the UI mounting before scene construction", () => {
    const world = new World();
    for (const control of Object.values(manifoldDiagnosticScenes.edgeAxisWedge.controls))
        if (control.read) expect(control.read(world)).toBeNull();
});
test("arc reach includes its exterior and parallel slack but rejects the bounded interior", () => {
    expect(arcCanReach(1, 0.5, 0.25, 1)).toBe(true);
    expect(arcCanReach(Math.fround(0.9), Math.fround(0.8), 0.5, 1)).toBe(false);
    expect(arcCanReach(Math.fround(0.9), Math.fround(0.9), Math.fround(0.999999), 1)).toBe(true);
});
test("great-circle peaks distinguish inside and outside the edge wedge", () => {
    const u = vector(1),
        v = vector(0, 1);
    const inside = makeArc(u, v, vector(1, 1), -0.1, -0.2);
    expect(inside.inside).toBe(true);
    expect(inside.outsideDegrees).toBe(-45);
    const outside = makeArc(u, v, vector(-1), -0.1, -0.2);
    expect(outside.inside).toBe(false);
    expect(outside.outsideDegrees).toBe(90);
    expect(outside.peak).toEqual(vector(-1));
});
