import { expect, test } from "bun:test";
import { lookAtRotation, perspective } from "../../engine/utils";
import { spotOuterAngle, spotParams } from "./lighting";
import { EDGE_TEXELS, type PointShadowFrame, spotBasis } from "./shadows";

const radians = Math.PI / 180;
const output = (): Pick<PointShadowFrame, "fwd" | "right" | "up" | "coneTanHalf" | "coneFov"> => ({
    fwd: [0, 0, 0],
    right: [0, 0, 0],
    up: [0, 0, 0],
    coneTanHalf: 0,
    coneFov: 0,
});

test("an illuminated 85 degree receiver fits the 89 degree shadow cone", () => {
    const out = output();
    spotBasis(0, 0, 0, 1, 89 * radians, 512, out);
    const { scale, offset } = spotParams(88 * radians, 89 * radians);
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
    Math.PI / 2 - 1e-7,
    Math.PI / 2 - 1e-12,
]) {
    test(`finite shadow projection contains every edge at half-angle ${outer}`, () => {
        for (const rotation of [
            [0, 0, 0, 1],
            lookAtRotation(0, 0, 0, 1, 2, 3),
            lookAtRotation(0, 0, 0, 0, 1, 0),
        ]) {
            const out = output();
            spotBasis(rotation[0]!, rotation[1]!, rotation[2]!, rotation[3]!, outer, 512, out);
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
                // Recovering a nearly perpendicular edge's tiny depth by dot product loses
                // relative precision; check that basis separately and project its known local depth.
                expect(dot(out.fwd)).toBeCloseTo(Math.cos(outer), 14);
                const depth = Math.cos(outer);
                expect(Math.abs(dot(out.right) / depth / out.coneTanHalf)).toBeLessThanOrEqual(
                    1 + 1e-12,
                );
                expect(Math.abs(dot(out.up) / depth / out.coneTanHalf)).toBeLessThanOrEqual(
                    1 + 1e-12,
                );
            }
        }
    });
}

test("ordinary projections retain their existing additive margin, including default 45 and fixture 55 degrees", () => {
    for (const degrees of [1, 30, 45, 55, 80]) {
        const outer = Math.fround(degrees * radians);
        const out = output();
        spotBasis(0, 0, 0, 1, outer, 512, out);
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
        Math.PI / 2 - 1e-7,
    ]) {
        const inner = outer * 0.8;
        const scale = 1 / Math.max(Math.cos(inner) - Math.cos(outer), 1e-4);
        const actual = spotParams(inner, outer);
        expect(actual.scale).toBe(Math.fround(scale));
        expect(actual.offset).toBe(Math.fround(-Math.cos(outer) * scale));
    }
});

test("upper-bound policy clamps inner and outer together without narrowing valid cones", () => {
    for (const outer of [Math.PI / 2, Math.PI, 2]) {
        const effective = Math.fround(Math.PI / 2 - 1e-4);
        expect(spotOuterAngle(outer)).toBe(effective);
        expect(spotParams(outer, outer)).toEqual(spotParams(effective, effective));
        const invalid = output();
        const clamped = output();
        spotBasis(0, 0, 0, 1, outer, 512, invalid);
        spotBasis(0, 0, 0, 1, effective, 512, clamped);
        expect(invalid).toEqual(clamped);
    }
});
