import { describe, expect, test } from "bun:test";
import { eulerAlias, laneAlias } from "./alias";

const rot = eulerAlias("rot");
const lanes = (q: { x: number; y: number; z: number; w: number }) => ({
    "rot.x": q.x,
    "rot.y": q.y,
    "rot.z": q.z,
    "rot.w": q.w,
});

describe("eulerAlias", () => {
    test("reads the identity quaternion as zero euler", () => {
        expect(rot.axes).toEqual(["x", "y", "z"]);
        expect(rot.read(lanes({ x: 0, y: 0, z: 0, w: 1 }))).toEqual([0, 0, 0]);
    });

    test("writes an edited axis back as a quaternion at dotted lanes, round-tripping", () => {
        const updates = rot.write(1, 90, lanes({ x: 0, y: 0, z: 0, w: 1 }));
        expect(Object.keys(updates).sort()).toEqual(["rot.w", "rot.x", "rot.y", "rot.z"]);
        const [x, y, z] = rot.read(updates);
        expect(x).toBeCloseTo(0, 4);
        expect(y).toBeCloseTo(90, 4);
        expect(z).toBeCloseTo(0, 4);
    });

    test("a missing w lane defaults to 1 (identity), not a zero quaternion", () => {
        expect(rot.read({})).toEqual([0, 0, 0]);
    });
});

describe("laneAlias", () => {
    const mat = laneAlias("params", ["metallic", "roughness", "emissive", "occlusion"]);

    test("axes are the named lanes; read maps them 1:1 to x/y/z/w", () => {
        expect(mat.axes).toEqual(["metallic", "roughness", "emissive", "occlusion"]);
        expect(
            mat.read({ "params.x": 1, "params.y": 0.2, "params.z": 0.5, "params.w": 0.8 }),
        ).toEqual([1, 0.2, 0.5, 0.8]);
    });

    test("write targets one lane by axis index, no cross-lane coupling", () => {
        expect(mat.write(1, 0.3, {})).toEqual({ "params.y": 0.3 });
        expect(mat.write(2, 4, {})).toEqual({ "params.z": 4 });
    });

    test("identity by construction: axes.length === lane count (the parser/serializer discriminator)", () => {
        // euler is 3 axes over a 4-lane quat — non-identity, so it stays editor-only and positional
        expect(mat.axes.length).toBe(4);
        expect(eulerAlias("rot").axes.length).toBe(3);
    });
});
