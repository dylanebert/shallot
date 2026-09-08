import { describe, expect, spyOn, test } from "bun:test";
import { compose } from "../../engine";
import { bakeVat, type SkinInput } from "./vat";

// Spec tests for the VAT bake (vat.ts) — CPU linear-blend skinning to per-frame vertex data. The cases are
// hand-computed LBS: a known rig + a known vertex have a closed-form deformed position, so a regression in
// the hierarchy walk, the skin-matrix product (jointGlobal · inverseBind), the channel sampler, or the
// weight blend reads as a numeric miss here, not a visual one. The math is glTF 3.7.3.1; the decode
// authority is three.js GLTFLoader skinning.

const IDENT = compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
// 90° about +Z, as a quaternion (rotates +x → +y)
const Z90 = [0, 0, Math.SQRT1_2, Math.SQRT1_2] as const;
const node = (
    t: [number, number, number] = [0, 0, 0],
    r: [number, number, number, number] = [0, 0, 0, 1],
    children: number[] = [],
) => ({ t, r, s: [1, 1, 1] as [number, number, number], children });

describe("VAT bake — linear-blend skinning", () => {
    test("parent rotation propagates through the hierarchy to a child-bound vertex", () => {
        // node 0 (parent) at origin rotates 90°Z over the 1s clip; node 1 (child) is translated +1x from it.
        // A vertex at the child's bind origin (1,0,0) bound 100% to the child: at rest it stays put; at the
        // end the parent rotation carries it to (0,1,0). inverseBind(child) = translate(-1,0,0) (inverse of
        // the child's bind global, translate(+1,0,0)).
        const input: SkinInput = {
            nodes: [node([0, 0, 0], [0, 0, 0, 1], [1]), node([1, 0, 0])],
            roots: [0],
            channels: [
                {
                    node: 0,
                    path: "rotation",
                    times: new Float32Array([0, 1]),
                    values: new Float32Array([0, 0, 0, 1, ...Z90]),
                    step: false,
                },
            ],
            joints: [0, 1],
            inverseBind: new Float32Array([...IDENT, ...compose(-1, 0, 0, 0, 0, 0, 1, 1, 1, 1)]),
            jointIndex: new Uint16Array([1, 0, 0, 0]),
            weights: new Float32Array([1, 0, 0, 0]),
            restPos: new Float32Array([1, 0, 0]),
            restNormal: new Float32Array([1, 0, 0]),
            duration: 1,
        };
        const vat = bakeVat(input, { fps: 1 });

        expect(vat.frameCount).toBe(2); // round(1·1)+1
        expect(vat.fps).toBeCloseTo(1, 6); // (frameCount-1)/duration
        // frame 0 (t=0): rest pose
        expect(Array.from(vat.positions.slice(0, 3))).toEqual([1, 0, 0]);
        // frame 1 (t=1): parent rotated 90°Z carries (1,0,0) → (0,1,0), normal likewise
        expect(vat.positions[3]).toBeCloseTo(0, 5);
        expect(vat.positions[4]).toBeCloseTo(1, 5);
        expect(vat.positions[5]).toBeCloseTo(0, 5);
        expect(vat.normals[3]).toBeCloseTo(0, 5);
        expect(vat.normals[4]).toBeCloseTo(1, 5);
        // all-frames AABB encloses both poses: x∈[0,1], y∈[0,1]
        expect(vat.aabb.min[0]).toBeCloseTo(0, 5);
        expect(vat.aabb.max[0]).toBeCloseTo(1, 5);
        expect(vat.aabb.max[1]).toBeCloseTo(1, 5);
    });

    test("a vertex split 50/50 between two joints lands at the weighted average", () => {
        // joint 1 translates +2x, joint 0 is identity; a vertex at the origin weighted 0.5/0.5 sits at the
        // midpoint (1,0,0). No clip (duration 0 → one frame) — isolates the weight blend.
        const input: SkinInput = {
            nodes: [node(), node([2, 0, 0])],
            roots: [0, 1],
            channels: [],
            joints: [0, 1],
            inverseBind: new Float32Array([...IDENT, ...IDENT]),
            jointIndex: new Uint16Array([0, 1, 0, 0]),
            weights: new Float32Array([0.5, 0.5, 0, 0]),
            restPos: new Float32Array([0, 0, 0]),
            restNormal: new Float32Array([0, 1, 0]),
            duration: 0,
        };
        const vat = bakeVat(input);
        expect(vat.frameCount).toBe(1);
        expect(vat.positions[0]).toBeCloseTo(1, 6);
        expect(vat.positions[1]).toBeCloseTo(0, 6);
        expect(vat.positions[2]).toBeCloseTo(0, 6);
    });

    test("STEP interpolation holds the lower keyframe", () => {
        // a parent rotating 0 → 90°Z with STEP: at the midpoint frame the pose is still the t=0 key (no
        // rotation), so a child-bound vertex hasn't moved yet.
        const input: SkinInput = {
            nodes: [node()],
            roots: [0],
            channels: [
                {
                    node: 0,
                    path: "rotation",
                    times: new Float32Array([0, 1]),
                    values: new Float32Array([0, 0, 0, 1, ...Z90]),
                    step: true,
                },
            ],
            joints: [0],
            inverseBind: new Float32Array([...IDENT]),
            jointIndex: new Uint16Array([0, 0, 0, 0]),
            weights: new Float32Array([1, 0, 0, 0]),
            restPos: new Float32Array([1, 0, 0]),
            restNormal: new Float32Array([1, 0, 0]),
            duration: 1,
        };
        // 3 frames → the middle frame samples t=0.5, which STEP holds at the t=0 key (unrotated)
        const vat = bakeVat(input, { fps: 2 });
        expect(vat.frameCount).toBe(3);
        expect(vat.positions[3]).toBeCloseTo(1, 6); // frame 1 (t=0.5): still (1,0,0)
        expect(vat.positions[4]).toBeCloseTo(0, 6);
    });

    test("a clip past the frame cap subsamples and warns", () => {
        // duration·fps+1 = 301 > 120 → frameCount clamps to maxFrames and the effective rate drops below
        // fps; the warn keeps a silently-decimated animation from shipping unnoticed (streaming audit)
        const input: SkinInput = {
            nodes: [node()],
            roots: [0],
            channels: [],
            joints: [0],
            inverseBind: new Float32Array([...IDENT]),
            jointIndex: new Uint16Array([0, 0, 0, 0]),
            weights: new Float32Array([1, 0, 0, 0]),
            restPos: new Float32Array([0, 0, 0]),
            restNormal: new Float32Array([0, 1, 0]),
            duration: 10,
        };
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        const vat = bakeVat(input, { fps: 30 });

        expect(vat.frameCount).toBe(120); // clamped to maxFrames, not 301
        expect(warn).toHaveBeenCalledTimes(1);
        expect(warn.mock.calls[0][0]).toMatch(/subsampled/);
        warn.mockRestore();
    });
});
