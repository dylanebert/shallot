import { expect, test } from "bun:test";
import { lookAtRotation, perspective } from "../../engine/utils";
import { spotOuterAngle, spotParams } from "./lighting";
import { EDGE_TEXELS, type PointShadowFrame, spotBasis } from "./shadows";

const radians = Math.PI / 180;
// f32 spacing in [1, 2) is 2^-23; rounding π/2 goes above the valid range.
const lastValidOuter = Math.fround(Math.PI / 2) - 2 ** -23;
type Output = Pick<
    PointShadowFrame,
    "outer" | "tilePx" | "fwd" | "right" | "up" | "coneTanHalf" | "coneFov"
>;
const output = (): Output => ({
    outer: 0,
    tilePx: 512,
    fwd: [0, 0, 0],
    right: [0, 0, 0],
    up: [0, 0, 0],
    coneTanHalf: 0,
    coneFov: 0,
});
const fit = (rotation: readonly number[], outer: number, out: Output) => {
    const placement = new Float32Array(12);
    placement.set(rotation, 4);
    out.outer = outer;
    spotBasis(placement, out);
};

test("an illuminated 85 degree receiver fits the 89 degree shadow cone", () => {
    const out = output();
    const outer = Math.fround(89 * radians);
    fit([0, 0, 0, 1], outer, out);
    const { scale, offset } = spotParams(Math.fround(88 * radians), outer);
    expect(Math.min(1, Math.max(0, Math.cos(85 * radians) * scale + offset))).toBe(1);
    expect(Math.tan(85 * radians) / out.coneTanHalf).toBeLessThan(1);
});

for (const outer of [
    0,
    0.001,
    30 * radians,
    45 * radians,
    55 * radians,
    80 * radians,
    85 * radians,
    89 * radians,
    lastValidOuter,
].map(Math.fround)) {
    test(`finite shadow projection contains every edge at half-angle ${outer}`, () => {
        for (const rotation of [
            [0, 0, 0, 1],
            lookAtRotation(0, 0, 0, 1, 2, 3),
            lookAtRotation(0, 0, 0, 0, 1, 0),
        ]) {
            const out = output();
            fit(rotation, outer, out);
            expect(spotOuterAngle(outer)).toBe(outer);
            const projection = perspective(out.coneFov, 1, 0.1);
            expect(Array.from(projection).every(Number.isFinite)).toBe(true);
            for (let i = 0; i < 32; i++) {
                const angle = (i * Math.PI) / 16;
                const edge = out.fwd.map(
                    (v, axis) =>
                        v * Math.cos(outer) +
                        (out.right[axis]! * Math.cos(angle) + out.up[axis]! * Math.sin(angle)) *
                            Math.sin(outer),
                );
                const dot = (basis: number[]) =>
                    edge.reduce((sum, v, axis) => sum + v * basis[axis]!, 0);
                const view = [dot(out.right), dot(out.up), -dot(out.fwd), 1];
                const clip = [0, 1, 2, 3].map((row) =>
                    view.reduce((sum, value, col) => sum + projection[col * 4 + row]! * value, 0),
                );
                expect(clip.every(Number.isFinite)).toBe(true);
                expect(clip[3]!).toBeGreaterThan(0);
                // Four f32 epsilons cover projection-coefficient rounding and the
                // FOV round trip; even at the last valid angle, double basis recovery
                // has relative depth error below one f32 epsilon.
                const allowance = 4 * 2 ** -23;
                expect(Math.abs(clip[0]! / clip[3]!)).toBeLessThanOrEqual(1 + allowance);
                expect(Math.abs(clip[1]! / clip[3]!)).toBeLessThanOrEqual(1 + allowance);
            }
        }
    });
}

test("ordinary projections retain their existing additive margin, including default 45 and fixture 55 degrees", () => {
    for (const degrees of [1, 30, 45, 55, 80]) {
        const outer = Math.fround(degrees * radians);
        const out = output();
        fit([0, 0, 0, 1], outer, out);
        expect(out.coneTanHalf).toBe(Math.tan(Math.max(outer, radians)) + (2 * EDGE_TEXELS) / 512);
    }
});

test("valid cones retain authored angular falloff at light-data precision", () => {
    for (const outer of [
        0.001,
        30 * radians,
        45 * radians,
        55 * radians,
        80 * radians,
        85 * radians,
        89 * radians,
        lastValidOuter,
    ].map(Math.fround)) {
        const inner = outer * 0.8;
        const scale = 1 / Math.max(Math.cos(inner) - Math.cos(outer), 1e-4);
        const actual = spotParams(inner, outer);
        expect(actual.scale).toBe(scale);
        expect(actual.offset).toBe(-Math.cos(outer) * scale);
    }
});

test("upper-bound policy clamps inner and outer together without narrowing valid cones", () => {
    for (const outer of [Math.PI / 2, Math.PI, 2].map(Math.fround)) {
        const effective = Math.fround(Math.PI / 2 - 1e-4);
        expect(spotOuterAngle(outer)).toBe(effective);
        expect(spotParams(outer, outer)).toEqual(spotParams(effective, effective));
        const invalid = output();
        const clamped = output();
        fit([0, 0, 0, 1], outer, invalid);
        fit([0, 0, 0, 1], effective, clamped);
        expect({ ...invalid, outer: effective }).toEqual(clamped);
    }
});
