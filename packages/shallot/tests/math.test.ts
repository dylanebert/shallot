import { describe, test, expect } from "bun:test";
import { quat, mat4, utils } from "wgpu-matrix";
import * as math from "../src/engine/utils/math";

const EPSILON = 0.00001;

function quatEqual(
    a: { x: number; y: number; z: number; w: number },
    b: Float32Array | number[],
    epsilon = EPSILON,
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
    epsilon = EPSILON,
): boolean {
    return (
        Math.abs(a.x - b.x) < epsilon &&
        Math.abs(a.y - b.y) < epsilon &&
        Math.abs(a.z - b.z) < epsilon
    );
}

describe("Math functions vs wgpu-matrix", () => {
    describe("eulerToQuaternion", () => {
        test("should match wgpu-matrix for zero rotation", () => {
            const result = math.eulerToQuaternion(0, 0, 0);
            const expected = quat.fromEuler(0, 0, 0, "xyz");
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for X rotation", () => {
            const result = math.eulerToQuaternion(45, 0, 0);
            const expected = quat.fromEuler(utils.degToRad(45), 0, 0, "xyz");
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for Y rotation", () => {
            const result = math.eulerToQuaternion(0, 90, 0);
            const expected = quat.fromEuler(0, utils.degToRad(90), 0, "xyz");
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for Z rotation", () => {
            const result = math.eulerToQuaternion(0, 0, 180);
            const expected = quat.fromEuler(0, 0, utils.degToRad(180), "xyz");
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for combined rotation", () => {
            const result = math.eulerToQuaternion(45, 90, 180);
            const expected = quat.fromEuler(
                utils.degToRad(45),
                utils.degToRad(90),
                utils.degToRad(180),
                "xyz",
            );
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should return different objects for consecutive calls", () => {
            const q1 = math.eulerToQuaternion(0, 0, 0);
            const q2 = math.eulerToQuaternion(45, 0, 0);
            expect(q1).not.toBe(q2);
            expect(q1.x).not.toBe(q2.x);
        });
    });

    describe("quaternionToEuler", () => {
        test("should match wgpu-matrix for identity quaternion", () => {
            const result = math.quaternionToEuler(0, 0, 0, 1);
            expect(eulerEqual(result, { x: 0, y: 0, z: 0 })).toBe(true);
        });

        test("should round-trip through euler-quat-euler", () => {
            const angles = [
                { x: 0, y: 0, z: 0 },
                { x: 45, y: 0, z: 0 },
                { x: 0, y: 90, z: 0 },
                { x: 0, y: 0, z: 180 },
                { x: 30, y: 45, z: 60 },
            ];

            for (const angle of angles) {
                const q = math.eulerToQuaternion(angle.x, angle.y, angle.z);
                const e = math.quaternionToEuler(q.x, q.y, q.z, q.w);
                expect(eulerEqual(e, angle, 0.01)).toBe(true);
            }
        });

        test("should return different objects for consecutive calls", () => {
            const e1 = math.quaternionToEuler(0, 0, 0, 1);
            const e2 = math.quaternionToEuler(0.383, 0, 0, 0.924);
            expect(e1).not.toBe(e2);
            expect(e1.x).not.toBe(e2.x);
        });
    });

    describe("rotate", () => {
        test("should match wgpu-matrix for identity rotation", () => {
            const result = math.rotate(0, 0, 0, 1, 0, 0, 0);
            const expected = quat.identity();
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for X rotation", () => {
            const q = quat.identity();
            const result = math.rotate(q[0], q[1], q[2], q[3], 45, 0, 0);
            const delta = quat.fromEuler(utils.degToRad(45), 0, 0, "xyz");
            const expected = quat.multiply(q, delta);
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix for compound rotations", () => {
            let q = quat.identity();
            const result1 = math.rotate(q[0], q[1], q[2], q[3], 30, 0, 0);
            q = quat.fromEuler(utils.degToRad(30), 0, 0, "xyz");
            const result2 = math.rotate(result1.x, result1.y, result1.z, result1.w, 0, 45, 0);

            const delta2 = quat.fromEuler(0, utils.degToRad(45), 0, "xyz");
            const expected = quat.multiply(q, delta2);
            expect(quatEqual(result2, expected)).toBe(true);
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

    describe("lookAt", () => {
        test("should match wgpu-matrix looking forward", () => {
            const result = math.lookAt(0, 0, 0, 0, 0, -1);
            const mat = mat4.cameraAim([0, 0, 0], [0, 0, -1], [0, 1, 0]);
            const expected = quat.fromMat(mat);
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix looking at arbitrary target", () => {
            const result = math.lookAt(1, 2, 3, 4, 5, 6);
            const mat = mat4.cameraAim([1, 2, 3], [4, 5, 6], [0, 1, 0]);
            const expected = quat.fromMat(mat);
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should match wgpu-matrix with custom up vector", () => {
            const result = math.lookAt(0, 0, 0, 1, 0, 0, 0, 0, 1);
            const mat = mat4.cameraAim([0, 0, 0], [1, 0, 0], [0, 0, 1]);
            const expected = quat.fromMat(mat);
            expect(quatEqual(result, expected)).toBe(true);
        });

        test("should handle eye == target (zero distance)", () => {
            const result = math.lookAt(0, 0, 0, 0, 0, 0);
            expect(Number.isFinite(result.x)).toBe(true);
            expect(Number.isFinite(result.y)).toBe(true);
            expect(Number.isFinite(result.z)).toBe(true);
            expect(Number.isFinite(result.w)).toBe(true);
        });

        test("should handle looking straight up (parallel to up vector)", () => {
            const result = math.lookAt(0, 0, 0, 0, 1, 0);
            expect(Number.isFinite(result.x)).toBe(true);
            expect(Number.isFinite(result.y)).toBe(true);
            expect(Number.isFinite(result.z)).toBe(true);
            expect(Number.isFinite(result.w)).toBe(true);
        });

        test("should handle looking straight down (parallel to up vector)", () => {
            const result = math.lookAt(0, 0, 0, 0, -1, 0);
            expect(Number.isFinite(result.x)).toBe(true);
            expect(Number.isFinite(result.y)).toBe(true);
            expect(Number.isFinite(result.z)).toBe(true);
            expect(Number.isFinite(result.w)).toBe(true);
        });
    });

    describe("testAABBFrustum", () => {
        function frustumPlanes(fov: number, aspect: number, near: number, far: number) {
            const proj = math.perspective(fov, aspect, near, far);
            const view = math.lookAtMatrix(0, 0, 0, 0, 0, -1);
            const vp = math.multiply(proj, view);
            return math.extractFrustumPlanes(vp);
        }

        test("box at origin is inside a centered frustum", () => {
            const planes = frustumPlanes(90, 1, 0.1, 100);
            expect(math.testAABBFrustum(-1, -1, -5, 1, 1, -1, planes)).toBe(true);
        });

        test("box behind camera is outside", () => {
            const planes = frustumPlanes(90, 1, 0.1, 100);
            expect(math.testAABBFrustum(-1, -1, 1, 1, 1, 5, planes)).toBe(false);
        });

        test("box far to the left is outside", () => {
            const planes = frustumPlanes(90, 1, 0.1, 100);
            expect(math.testAABBFrustum(-100, -1, -5, -90, 1, -1, planes)).toBe(false);
        });

        test("box beyond far plane is outside", () => {
            const planes = frustumPlanes(90, 1, 0.1, 10);
            expect(math.testAABBFrustum(-1, -1, -50, 1, 1, -40, planes)).toBe(false);
        });

        test("box straddling near plane is inside", () => {
            const planes = frustumPlanes(90, 1, 1, 100);
            expect(math.testAABBFrustum(-0.5, -0.5, -2, 0.5, 0.5, 0, planes)).toBe(true);
        });

        test("cubemap face +X frustum culls box on -X side", () => {
            const view = math.lookAtMatrix(0, 0, 0, 1, 0, 0, 0, -1, 0);
            const proj = math.perspective(90, 1, 0.1, 100);
            const vp = math.multiply(proj, view);
            const planes = math.extractFrustumPlanes(vp);
            expect(math.testAABBFrustum(-10, -1, -1, -5, 1, 1, planes)).toBe(false);
            expect(math.testAABBFrustum(5, -1, -1, 10, 1, 1, planes)).toBe(true);
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
