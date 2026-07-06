import { describe, expect, test } from "bun:test";
import { mat4, quat, utils } from "wgpu-matrix";
import * as math from "./math";

// wgpu-matrix returns Float32Array; comparing our f64 result against its f32
// reference, the gap is at most one f32 ULP (2⁻²³) for unit-magnitude components.
const F32Tolerance = 2 ** -23;
// euler↔quat↔euler is exact in reals; in f64 over the tested non-gimbal angles
// the round-trip error stays many orders under this.
const EulerRt = 1e-10;

function quatEqual(
    a: { x: number; y: number; z: number; w: number },
    b: Float32Array | number[],
    epsilon = F32Tolerance,
): boolean {
    const directMatch =
        Math.abs(a.x - b[0]) < epsilon &&
        Math.abs(a.y - b[1]) < epsilon &&
        Math.abs(a.z - b[2]) < epsilon &&
        Math.abs(a.w - b[3]) < epsilon;

    const negatedMatch =
        Math.abs(a.x + b[0]) < epsilon &&
        Math.abs(a.y + b[1]) < epsilon &&
        Math.abs(a.z + b[2]) < epsilon &&
        Math.abs(a.w + b[3]) < epsilon;

    return directMatch || negatedMatch;
}

function eulerEqual(
    a: { x: number; y: number; z: number },
    b: { x: number; y: number; z: number },
    epsilon = F32Tolerance,
): boolean {
    return (
        Math.abs(a.x - b.x) < epsilon &&
        Math.abs(a.y - b.y) < epsilon &&
        Math.abs(a.z - b.z) < epsilon
    );
}

describe("Math functions vs wgpu-matrix", () => {
    describe("quat (from euler)", () => {
        test("should match wgpu-matrix for zero rotation", () => {
            const result = math.quat(0, 0, 0);
            const expected = quat.fromEuler(0, 0, 0, "xyz");
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for X rotation", () => {
            const result = math.quat(45, 0, 0);
            const expected = quat.fromEuler(utils.degToRad(45), 0, 0, "xyz");
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for Y rotation", () => {
            const result = math.quat(0, 90, 0);
            const expected = quat.fromEuler(0, utils.degToRad(90), 0, "xyz");
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for Z rotation", () => {
            const result = math.quat(0, 0, 180);
            const expected = quat.fromEuler(0, 0, utils.degToRad(180), "xyz");
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for combined rotation", () => {
            const result = math.quat(45, 90, 180);
            const expected = quat.fromEuler(
                utils.degToRad(45),
                utils.degToRad(90),
                utils.degToRad(180),
                "xyz",
            );
            expect(quatEqual(result, expected)).toBe(true);
        });
    });

    describe("euler (from quat)", () => {
        test("should match wgpu-matrix for identity quaternion", () => {
            const result = math.euler(0, 0, 0, 1);
            expect(eulerEqual(result, { x: 0, y: 0, z: 0 })).toBe(true);
        });

        test("should round-trip through quat-euler-quat", () => {
            const angles = [
                { x: 0, y: 0, z: 0 },
                { x: 45, y: 0, z: 0 },
                { x: 0, y: 90, z: 0 },
                { x: 0, y: 0, z: 180 },
                { x: 30, y: 45, z: 60 },
            ];

            for (const angle of angles) {
                const q = math.quat(angle.x, angle.y, angle.z);
                const e = math.euler(q.x, q.y, q.z, q.w);
                expect(eulerEqual(e, angle, EulerRt)).toBe(true);
            }
        });
    });

    describe("slerp", () => {
        test("should match wgpu-matrix at t=0", () => {
            const from = quat.fromEuler(0, 0, 0, "xyz");
            const to = quat.fromEuler(utils.degToRad(90), 0, 0, "xyz");
            const result = math.slerp(
                from[0],
                from[1],
                from[2],
                from[3],
                to[0],
                to[1],
                to[2],
                to[3],
                0,
            );
            expect(quatEqual(result, from)).toBe(true);
        });

        test("should match wgpu-matrix at t=1", () => {
            const from = quat.fromEuler(0, 0, 0, "xyz");
            const to = quat.fromEuler(utils.degToRad(90), 0, 0, "xyz");
            const result = math.slerp(
                from[0],
                from[1],
                from[2],
                from[3],
                to[0],
                to[1],
                to[2],
                to[3],
                1,
            );
            expect(quatEqual(result, to)).toBe(true);
        });

        test("should match wgpu-matrix at t=0.5", () => {
            const from = quat.fromEuler(0, 0, 0, "xyz");
            const to = quat.fromEuler(utils.degToRad(90), 0, 0, "xyz");
            const result = math.slerp(
                from[0],
                from[1],
                from[2],
                from[3],
                to[0],
                to[1],
                to[2],
                to[3],
                0.5,
            );
            const expected = quat.slerp(from, to, 0.5);
            expect(quatEqual(result, expected)).toBe(true);
        });
    });

    describe("aim", () => {
        test("should match wgpu-matrix looking forward", () => {
            const result = math.aim(0, 0, 0, 0, 0, -1);
            const m = mat4.cameraAim([0, 0, 0], [0, 0, -1], [0, 1, 0]);
            const expected = quat.fromMat(m);
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix looking at arbitrary target", () => {
            const result = math.aim(1, 2, 3, 4, 5, 6);
            const m = mat4.cameraAim([1, 2, 3], [4, 5, 6], [0, 1, 0]);
            const expected = quat.fromMat(m);
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix with custom up vector", () => {
            const result = math.aim(0, 0, 0, 1, 0, 0, 0, 0, 1);
            const m = mat4.cameraAim([0, 0, 0], [1, 0, 0], [0, 0, 1]);
            const expected = quat.fromMat(m);
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should handle eye == target (zero distance)", () => {
            const result = math.aim(0, 0, 0, 0, 0, 0);
            expect(Number.isFinite(result.x)).toBe(true);
            expect(Number.isFinite(result.y)).toBe(true);
            expect(Number.isFinite(result.z)).toBe(true);
            expect(Number.isFinite(result.w)).toBe(true);
        });

        test("should handle looking straight up (parallel to up vector)", () => {
            const result = math.aim(0, 0, 0, 0, 1, 0);
            expect(Number.isFinite(result.x)).toBe(true);
            expect(Number.isFinite(result.y)).toBe(true);
            expect(Number.isFinite(result.z)).toBe(true);
            expect(Number.isFinite(result.w)).toBe(true);
        });

        test("should handle looking straight down (parallel to up vector)", () => {
            const result = math.aim(0, 0, 0, 0, -1, 0);
            expect(Number.isFinite(result.x)).toBe(true);
            expect(Number.isFinite(result.y)).toBe(true);
            expect(Number.isFinite(result.z)).toBe(true);
            expect(Number.isFinite(result.w)).toBe(true);
        });
    });

    describe("clamp", () => {
        test("should clamp value below min", () => {
            expect(math.clamp(-5, 0, 10)).toBe(0);
        });

        test("should clamp value above max", () => {
            expect(math.clamp(15, 0, 10)).toBe(10);
        });

        test("should return value within range", () => {
            expect(math.clamp(5, 0, 10)).toBe(5);
        });
    });

    describe("lerp", () => {
        test("should return a at t=0", () => {
            expect(math.lerp(0, 10, 0)).toBe(0);
        });

        test("should return b at t=1", () => {
            expect(math.lerp(0, 10, 1)).toBe(10);
        });

        test("should interpolate at t=0.5", () => {
            expect(math.lerp(0, 10, 0.5)).toBe(5);
        });
    });
});

describe("invert", () => {
    test("inverts identity matrix to identity", () => {
        const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        const result = math.invert(identity);
        // identity inverts to integer 1s — no rounding, exact
        expect(result[0]).toBe(1);
        expect(result[5]).toBe(1);
        expect(result[10]).toBe(1);
        expect(result[15]).toBe(1);
    });

    test("inverts translation", () => {
        const translated = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 3, 2, 1]);
        const result = math.invert(translated);
        // a pure translation inverts to its negation — integer arithmetic, exact
        expect(result[12]).toBe(-5);
        expect(result[13]).toBe(-3);
        expect(result[14]).toBe(-2);
    });
});

describe("multiply", () => {
    test("identity × identity = identity", () => {
        const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        const result = math.multiply(identity, identity);
        // products of 0s and 1s only — exact
        expect(result[0]).toBe(1);
        expect(result[5]).toBe(1);
        expect(result[10]).toBe(1);
        expect(result[15]).toBe(1);
    });

    test("identity × A = A", () => {
        const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        const a = new Float32Array([2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 1, 2, 3, 1]);
        const result = math.multiply(identity, a);
        // each entry passes through a single 1× term — exact, no rounding
        for (let i = 0; i < 16; i++) {
            expect(result[i]).toBe(a[i]);
        }
    });
});

describe("orthographic", () => {
    // size is the vertical half-extent, widened horizontally by aspect. Only the depth row is reverse-Z;
    // the x/y scale matches wgpu-matrix's WebGPU-clip ortho, so compare those entries against it.
    test("x/y scale matches the reference (square + wide aspect)", () => {
        for (const aspect of [1, 2]) {
            const size = 5,
                near = 0.1,
                far = 100;
            const result = math.orthographic(size, aspect, near, far);
            const ref = mat4.ortho(-size * aspect, size * aspect, -size, size, near, far);
            expect(result[0]).toBeCloseTo(ref[0], 6); // horizontal scale (aspect-widened)
            expect(result[5]).toBeCloseTo(ref[5], 6); // vertical scale
        }
    });

    // reverse-Z: near→1, far→0, and (ortho being linear in depth) the midpoint is exactly 0.5
    test("reverse-Z depth maps near→1, far→0", () => {
        const near = 0.1,
            far = 100;
        const p = math.orthographic(5, 1, near, far);
        const ndc = (d: number) => p[10] * -d + p[14]; // w = 1, so z_ndc = z_clip
        expect(ndc(near)).toBeCloseTo(1, 6);
        expect(ndc(far)).toBeCloseTo(0, 6);
        expect(ndc((near + far) / 2)).toBeCloseTo(0.5, 6);
    });

    test("w component is 1 (no perspective divide)", () => {
        const result = math.orthographic(10, 1.5, 1, 1000);
        expect(result[3]).toBe(0);
        expect(result[7]).toBe(0);
        expect(result[11]).toBe(0);
        expect(result[15]).toBe(1);
    });

    test("throws on size = 0", () => {
        expect(() => math.orthographic(0, 1, 0.1, 100)).toThrow("Invalid orthographic size");
    });

    test("throws on negative size", () => {
        expect(() => math.orthographic(-5, 1, 0.1, 100)).toThrow("Invalid orthographic size");
    });

    test("throws on aspect = 0", () => {
        expect(() => math.orthographic(5, 0, 0.1, 100)).toThrow("Invalid aspect ratio");
    });

    test("throws on near === far", () => {
        expect(() => math.orthographic(5, 1, 100, 100)).toThrow("Invalid depth planes");
    });
});

describe("perspective", () => {
    // only the depth row is reverse-Z; the x/y scale + the w = z row match wgpu-matrix's WebGPU-clip
    // perspective, so compare those against it (it takes the vertical FOV in radians)
    test("x/y scale + w row match the reference", () => {
        const fov = 60,
            aspect = 16 / 9,
            near = 0.1,
            far = 1000;
        const result = math.perspective(fov, aspect, near, far);
        const ref = mat4.perspective((fov * Math.PI) / 180, aspect, near, far);
        expect(result[0]).toBeCloseTo(ref[0], 6); // f / aspect
        expect(result[5]).toBeCloseTo(ref[5], 6); // f
        expect(result[11]).toBeCloseTo(ref[11], 6); // -1 (w_clip = view-space z)
    });

    // reverse-Z: near→1, far→0; the perspective curve stays monotonic in between
    test("reverse-Z depth maps near→1, far→0", () => {
        const near = 0.1,
            far = 1000;
        const p = math.perspective(60, 16 / 9, near, far);
        const ndc = (d: number) => (p[10] * -d + p[14]) / (p[11] * -d); // w = z, clip from view-space -d
        expect(ndc(near)).toBeCloseTo(1, 5);
        expect(ndc(far)).toBeCloseTo(0, 5);
        const mid = ndc((near + far) / 2);
        expect(mid).toBeGreaterThan(0);
        expect(mid).toBeLessThan(1);
    });

    test("throws on FOV = 0", () => {
        expect(() => math.perspective(0, 1, 0.1, 100)).toThrow("Invalid FOV");
    });

    test("throws on negative FOV", () => {
        expect(() => math.perspective(-60, 1, 0.1, 100)).toThrow("Invalid FOV");
    });

    test("throws on aspect = 0", () => {
        expect(() => math.perspective(60, 0, 0.1, 100)).toThrow("Invalid aspect ratio");
    });

    test("throws on near === far", () => {
        expect(() => math.perspective(60, 1, 100, 100)).toThrow("Invalid depth planes");
    });
});
