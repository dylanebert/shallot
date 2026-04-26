import { describe, test, expect } from "bun:test";
import { computeCascadeSplits, computeCascadeMatrix } from "../src/standard/raster/shadow";
import {
    lookAtMatrix,
    orthographicBounds,
    extractFrustumCorners,
    extractFrustumPlanes,
    invertMatrix,
    multiply,
    perspective,
} from "../src/engine/utils/math";

describe("cascade shadow mapping", () => {
    describe("computeCascadeSplits", () => {
        test("returns correct number of splits", () => {
            const splits = computeCascadeSplits(0.1, 100, 4);
            expect(splits.length).toBe(4);
        });

        test("splits are monotonically increasing", () => {
            const splits = computeCascadeSplits(0.1, 100, 4);
            for (let i = 1; i < splits.length; i++) {
                expect(splits[i]).toBeGreaterThan(splits[i - 1]);
            }
        });

        test("last split equals far plane", () => {
            const far = 100;
            const splits = computeCascadeSplits(0.1, far, 4);
            expect(splits[3]).toBeCloseTo(far, 3);
        });

        test("splits are between near and far", () => {
            const near = 0.1;
            const far = 100;
            const splits = computeCascadeSplits(near, far, 4);
            for (const split of splits) {
                expect(split).toBeGreaterThan(near);
                expect(split).toBeLessThanOrEqual(far);
            }
        });

        test("lambda=0 gives uniform distribution", () => {
            const near = 1;
            const far = 100;
            const splits = computeCascadeSplits(near, far, 4, 0);
            const expectedStep = (far - near) / 4;
            expect(splits[0]).toBeCloseTo(near + expectedStep, 3);
            expect(splits[1]).toBeCloseTo(near + expectedStep * 2, 3);
            expect(splits[2]).toBeCloseTo(near + expectedStep * 3, 3);
            expect(splits[3]).toBeCloseTo(far, 3);
        });

        test("lambda=1 gives logarithmic distribution", () => {
            const near = 1;
            const far = 100;
            const splits = computeCascadeSplits(near, far, 4, 1);
            const ratio = far / near;
            expect(splits[0]).toBeCloseTo(near * Math.pow(ratio, 0.25), 3);
            expect(splits[1]).toBeCloseTo(near * Math.pow(ratio, 0.5), 3);
            expect(splits[2]).toBeCloseTo(near * Math.pow(ratio, 0.75), 3);
            expect(splits[3]).toBeCloseTo(far, 3);
        });
    });

    describe("computeCascadeMatrix", () => {
        test("returns valid view projection matrix", () => {
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[12] = 0;
            cameraWorld[13] = 5;
            cameraWorld[14] = 10;
            cameraWorld[15] = 1;

            const result = computeCascadeMatrix(cameraWorld, 60, 16 / 9, 0.1, 10, [0, -1, 0], 1024);

            expect(result.viewProj.length).toBe(16);

            for (const v of result.viewProj) {
                expect(Number.isFinite(v)).toBe(true);
            }
        });

        test("far shadow caster not culled by near plane", () => {
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[13] = 5;
            cameraWorld[15] = 1;

            const lightDir: [number, number, number] = [0, -1, 0];
            const result = computeCascadeMatrix(cameraWorld, 60, 1, 0.1, 10, lightDir, 1024);

            const planes = extractFrustumPlanes(new Float32Array(result.viewProj));
            const farCaster: [number, number, number] = [0, 500, -5];
            const radius = 1;

            const nearDist =
                planes[16] * farCaster[0] +
                planes[17] * farCaster[1] +
                planes[18] * farCaster[2] +
                planes[19];
            expect(nearDist).toBeLessThan(-radius);

            planes[16] = 0;
            planes[17] = 0;
            planes[18] = 0;
            planes[19] = 1e30;

            for (let p = 0; p < 6; p++) {
                const nx = planes[p * 4];
                const ny = planes[p * 4 + 1];
                const nz = planes[p * 4 + 2];
                const d = planes[p * 4 + 3];
                const dist = nx * farCaster[0] + ny * farCaster[1] + nz * farCaster[2] + d;
                expect(dist).toBeGreaterThan(-radius);
            }
        });

        test("includes shadow casters closer to the light than the frustum", () => {
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[13] = 5;
            cameraWorld[15] = 1;

            const lightDir: [number, number, number] = [0, -1, 0];
            const result = computeCascadeMatrix(cameraWorld, 60, 1, 0.1, 10, lightDir, 1024);

            const casterAbove: [number, number, number] = [0, 15, -5];
            const wx = casterAbove[0],
                wy = casterAbove[1],
                wz = casterAbove[2];
            const cz =
                result.viewProj[2] * wx +
                result.viewProj[6] * wy +
                result.viewProj[10] * wz +
                result.viewProj[14];
            const cw =
                result.viewProj[3] * wx +
                result.viewProj[7] * wy +
                result.viewProj[11] * wz +
                result.viewProj[15];
            const ndcZ = cz / cw;
            expect(ndcZ).toBeGreaterThanOrEqual(0);
            expect(ndcZ).toBeLessThanOrEqual(1);
        });
        test("large room: distant casters not clipped by shadow volume", () => {
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[15] = 1;

            const lightDir: [number, number, number] = [-0.5, -1, -0.5];
            const result = computeCascadeMatrix(cameraWorld, 60, 16 / 9, 0.1, 300, lightDir, 1024);

            const caster: [number, number, number] = [0, 250, 0];
            const wx = caster[0],
                wy = caster[1],
                wz = caster[2];
            const cz =
                result.viewProj[2] * wx +
                result.viewProj[6] * wy +
                result.viewProj[10] * wz +
                result.viewProj[14];
            const cw =
                result.viewProj[3] * wx +
                result.viewProj[7] * wy +
                result.viewProj[11] * wz +
                result.viewProj[15];
            const ndcZ = cz / cw;
            expect(ndcZ).toBeGreaterThanOrEqual(0);
            expect(ndcZ).toBeLessThanOrEqual(1);
        });
    });

    describe("lookAtMatrix", () => {
        test("creates valid view matrix", () => {
            const mat = lookAtMatrix(0, 5, 10, 0, 0, 0);
            expect(mat.length).toBe(16);
            for (const v of mat) {
                expect(Number.isFinite(v)).toBe(true);
            }
        });

        test("transforms eye position to origin", () => {
            const mat = lookAtMatrix(5, 5, 5, 0, 0, 0);
            const wx = mat[0] * 5 + mat[4] * 5 + mat[8] * 5 + mat[12];
            const wy = mat[1] * 5 + mat[5] * 5 + mat[9] * 5 + mat[13];
            const wz = mat[2] * 5 + mat[6] * 5 + mat[10] * 5 + mat[14];
            expect(wx).toBeCloseTo(0, 3);
            expect(wy).toBeCloseTo(0, 3);
            expect(wz).toBeCloseTo(0, 3);
        });

        test("handles degenerate case of same eye and target", () => {
            const mat = lookAtMatrix(0, 0, 0, 0, 0, 0);
            for (const v of mat) {
                expect(Number.isFinite(v)).toBe(true);
            }
        });

        test("handles looking straight down", () => {
            const mat = lookAtMatrix(0, 10, 0, 0, 0, 0);
            for (const v of mat) {
                expect(Number.isFinite(v)).toBe(true);
            }
        });
    });

    describe("orthographicBounds", () => {
        test("creates valid projection matrix", () => {
            const mat = orthographicBounds(-10, 10, -10, 10, 0.1, 100);
            expect(mat.length).toBe(16);
            for (const v of mat) {
                expect(Number.isFinite(v)).toBe(true);
            }
        });

        test("maps bounds correctly", () => {
            const mat = orthographicBounds(-5, 5, -5, 5, -100, 100);
            const scale = mat[0];
            expect(scale).toBeCloseTo(0.2, 5);
        });
    });

    describe("extractFrustumCorners", () => {
        test("extracts 8 corners", () => {
            const proj = perspective(60, 1, 0.1, 100);
            const view = new Float32Array(16);
            view[0] = 1;
            view[5] = 1;
            view[10] = 1;
            view[15] = 1;
            const viewProj = multiply(proj, view);
            const invViewProj = invertMatrix(viewProj);

            const corners = extractFrustumCorners(invViewProj, 0, 1);
            expect(corners.length).toBe(24);

            for (const v of corners) {
                expect(Number.isFinite(v)).toBe(true);
            }
        });

        test("near corners are closer than far corners", () => {
            const proj = perspective(60, 1, 0.1, 100);
            const view = new Float32Array(16);
            view[0] = 1;
            view[5] = 1;
            view[10] = 1;
            view[15] = 1;
            const viewProj = multiply(proj, view);
            const invViewProj = invertMatrix(viewProj);

            const corners = extractFrustumCorners(invViewProj, 0, 1);

            const nearCenter = {
                x: (corners[0] + corners[3] + corners[6] + corners[9]) / 4,
                y: (corners[1] + corners[4] + corners[7] + corners[10]) / 4,
                z: (corners[2] + corners[5] + corners[8] + corners[11]) / 4,
            };

            const farCenter = {
                x: (corners[12] + corners[15] + corners[18] + corners[21]) / 4,
                y: (corners[13] + corners[16] + corners[19] + corners[22]) / 4,
                z: (corners[14] + corners[17] + corners[20] + corners[23]) / 4,
            };

            const nearDist = Math.abs(nearCenter.z);
            const farDist = Math.abs(farCenter.z);
            expect(farDist).toBeGreaterThan(nearDist);
        });
    });

    describe("invertMatrix", () => {
        test("inverts identity matrix to identity", () => {
            const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
            const inv = invertMatrix(identity);
            for (let i = 0; i < 16; i++) {
                expect(inv[i]).toBeCloseTo(identity[i], 5);
            }
        });

        test("M * M^-1 = I", () => {
            const mat = lookAtMatrix(5, 5, 5, 0, 0, 0);
            const inv = invertMatrix(mat);
            const result = multiply(mat, inv);

            expect(result[0]).toBeCloseTo(1, 4);
            expect(result[5]).toBeCloseTo(1, 4);
            expect(result[10]).toBeCloseTo(1, 4);
            expect(result[15]).toBeCloseTo(1, 4);

            expect(result[1]).toBeCloseTo(0, 4);
            expect(result[2]).toBeCloseTo(0, 4);
            expect(result[4]).toBeCloseTo(0, 4);
        });
    });
});
