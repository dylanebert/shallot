import { describe, test, expect, beforeEach } from "bun:test";
import {
    createBox,
    createSphere,
    createPlane,
    computeShapeAABB,
    getMesh,
    MeshShape,
    clearMeshes,
    MAX_SHAPES,
} from "../src/standard/render/mesh";
import {
    extractFrustumPlanes,
    perspective,
    multiply,
    invert,
    testAABBSphere,
} from "../src/engine/utils/math";
import { packShapeAABBs, SHAPE_AABB_STRIDE } from "../src/standard/render/cull";

describe("Culling", () => {
    describe("computeShapeAABB", () => {
        beforeEach(() => {
            clearMeshes();
        });

        test("box AABB is (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5)", () => {
            const box = createBox();
            const aabb = computeShapeAABB(box);

            expect(aabb.minX).toBeCloseTo(-0.5, 5);
            expect(aabb.minY).toBeCloseTo(-0.5, 5);
            expect(aabb.minZ).toBeCloseTo(-0.5, 5);
            expect(aabb.maxX).toBeCloseTo(0.5, 5);
            expect(aabb.maxY).toBeCloseTo(0.5, 5);
            expect(aabb.maxZ).toBeCloseTo(0.5, 5);
        });

        test("sphere AABB is (-0.5, -0.5, -0.5) to (0.5, 0.5, 0.5)", () => {
            const sphere = createSphere();
            const aabb = computeShapeAABB(sphere);

            expect(aabb.minX).toBeCloseTo(-0.5, 5);
            expect(aabb.minY).toBeCloseTo(-0.5, 5);
            expect(aabb.minZ).toBeCloseTo(-0.5, 5);
            expect(aabb.maxX).toBeCloseTo(0.5, 5);
            expect(aabb.maxY).toBeCloseTo(0.5, 5);
            expect(aabb.maxZ).toBeCloseTo(0.5, 5);
        });

        test("plane AABB is (-0.5, 0, -0.5) to (0.5, 0, 0.5)", () => {
            const plane = createPlane();
            const aabb = computeShapeAABB(plane);

            expect(aabb.minX).toBeCloseTo(-0.5, 5);
            expect(aabb.minY).toBe(0);
            expect(aabb.minZ).toBeCloseTo(-0.5, 5);
            expect(aabb.maxX).toBeCloseTo(0.5, 5);
            expect(aabb.maxY).toBe(0);
            expect(aabb.maxZ).toBeCloseTo(0.5, 5);
        });

        test("handles empty mesh", () => {
            const empty = {
                vertices: new Float32Array(0),
                indices: new Uint16Array(0),
                vertexCount: 0,
                indexCount: 0,
            };
            const aabb = computeShapeAABB(empty);

            expect(aabb.minX).toBe(0);
            expect(aabb.minY).toBe(0);
            expect(aabb.minZ).toBe(0);
            expect(aabb.maxX).toBe(0);
            expect(aabb.maxY).toBe(0);
            expect(aabb.maxZ).toBe(0);
        });

        test("getMesh returns valid AABB for builtin shapes", () => {
            const boxMesh = getMesh(MeshShape.Box);
            const sphereMesh = getMesh(MeshShape.Sphere);
            const planeMesh = getMesh(MeshShape.Plane);

            expect(boxMesh).toBeDefined();
            expect(sphereMesh).toBeDefined();
            expect(planeMesh).toBeDefined();

            const boxAABB = computeShapeAABB(boxMesh!);
            const sphereAABB = computeShapeAABB(sphereMesh!);
            const planeAABB = computeShapeAABB(planeMesh!);

            expect(boxAABB.maxX - boxAABB.minX).toBeCloseTo(1.0, 5);
            expect(sphereAABB.maxY - sphereAABB.minY).toBeCloseTo(1.0, 5);
            expect(planeAABB.minY).toBe(planeAABB.maxY);
        });
    });

    describe("frustum plane extraction", () => {
        test("extracts 6 planes from view projection", () => {
            const proj = perspective(60, 16 / 9, 0.1, 100);
            const view = new Float32Array(16);
            view[0] = 1;
            view[5] = 1;
            view[10] = 1;
            view[15] = 1;

            const viewProj = multiply(proj, view);
            const planes = extractFrustumPlanes(viewProj);

            expect(planes.length).toBe(24);
        });

        test("planes are normalized", () => {
            const proj = perspective(60, 1, 0.1, 100);
            const view = new Float32Array(16);
            view[0] = 1;
            view[5] = 1;
            view[10] = 1;
            view[15] = 1;

            const viewProj = multiply(proj, view);
            const planes = extractFrustumPlanes(viewProj);

            for (let i = 0; i < 6; i++) {
                const nx = planes[i * 4];
                const ny = planes[i * 4 + 1];
                const nz = planes[i * 4 + 2];
                const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
                expect(len).toBeCloseTo(1.0, 3);
            }
        });
    });

    describe("testAABBSphere", () => {
        test("sphere containing AABB returns true", () => {
            expect(testAABBSphere(-1, -1, -1, 1, 1, 1, 0, 0, 0, 10)).toBe(true);
        });

        test("sphere center inside AABB returns true", () => {
            expect(testAABBSphere(-1, -1, -1, 1, 1, 1, 0, 0, 0, 0.5)).toBe(true);
        });

        test("sphere touching corner returns true", () => {
            const d = Math.sqrt(3);
            expect(testAABBSphere(-1, -1, -1, 1, 1, 1, 1 + d, 1 + d, 1 + d, 3)).toBe(true);
        });

        test("sphere fully outside returns false", () => {
            expect(testAABBSphere(-1, -1, -1, 1, 1, 1, 10, 0, 0, 2)).toBe(false);
        });

        test("sphere just touching face returns true", () => {
            expect(testAABBSphere(-1, -1, -1, 1, 1, 1, 3, 0, 0, 2)).toBe(true);
        });

        test("sphere just outside face returns false", () => {
            expect(testAABBSphere(-1, -1, -1, 1, 1, 1, 3.01, 0, 0, 2)).toBe(false);
        });

        test("degenerate zero-size AABB at origin", () => {
            expect(testAABBSphere(0, 0, 0, 0, 0, 0, 0, 0, 0, 1)).toBe(true);
            expect(testAABBSphere(0, 0, 0, 0, 0, 0, 2, 0, 0, 1)).toBe(false);
        });

        test("degenerate zero-radius sphere inside AABB", () => {
            expect(testAABBSphere(-1, -1, -1, 1, 1, 1, 0, 0, 0, 0)).toBe(true);
        });

        test("degenerate zero-radius sphere outside AABB", () => {
            expect(testAABBSphere(-1, -1, -1, 1, 1, 1, 5, 0, 0, 0)).toBe(false);
        });
    });

    describe("packShapeAABBs", () => {
        test("converts 6-float AABBs to 8-float padded format", () => {
            const src = new Float32Array(MAX_SHAPES * 6);
            src[0] = -1;
            src[1] = -2;
            src[2] = -3;
            src[3] = 1;
            src[4] = 2;
            src[5] = 3;
            const out = new Float32Array(MAX_SHAPES * SHAPE_AABB_STRIDE);

            packShapeAABBs(src, out);

            expect(out[0]).toBe(-1);
            expect(out[1]).toBe(-2);
            expect(out[2]).toBe(-3);
            expect(out[3]).toBe(0);
            expect(out[4]).toBe(1);
            expect(out[5]).toBe(2);
            expect(out[6]).toBe(3);
            expect(out[7]).toBe(0);
        });

        test("packs multiple shapes at correct stride", () => {
            const src = new Float32Array(MAX_SHAPES * 6);
            src[6] = -0.5;
            src[7] = -0.5;
            src[8] = -0.5;
            src[9] = 0.5;
            src[10] = 0.5;
            src[11] = 0.5;
            const out = new Float32Array(MAX_SHAPES * SHAPE_AABB_STRIDE);

            packShapeAABBs(src, out);

            const d = SHAPE_AABB_STRIDE;
            expect(out[d]).toBe(-0.5);
            expect(out[d + 1]).toBe(-0.5);
            expect(out[d + 2]).toBe(-0.5);
            expect(out[d + 3]).toBe(0);
            expect(out[d + 4]).toBe(0.5);
            expect(out[d + 5]).toBe(0.5);
            expect(out[d + 6]).toBe(0.5);
            expect(out[d + 7]).toBe(0);
        });

        test("zero-filled source produces zero-filled output", () => {
            const src = new Float32Array(MAX_SHAPES * 6);
            const out = new Float32Array(MAX_SHAPES * SHAPE_AABB_STRIDE);

            packShapeAABBs(src, out);

            for (let i = 0; i < MAX_SHAPES * SHAPE_AABB_STRIDE; i++) {
                expect(out[i]).toBe(0);
            }
        });
    });

    describe("AABB-Frustum intersection", () => {
        function testFrustum(
            aabbMin: [number, number, number],
            aabbMax: [number, number, number],
            planes: Float32Array,
        ): boolean {
            for (let i = 0; i < 6; i++) {
                const nx = planes[i * 4];
                const ny = planes[i * 4 + 1];
                const nz = planes[i * 4 + 2];
                const d = planes[i * 4 + 3];

                const px = nx >= 0 ? aabbMax[0] : aabbMin[0];
                const py = ny >= 0 ? aabbMax[1] : aabbMin[1];
                const pz = nz >= 0 ? aabbMax[2] : aabbMin[2];

                if (nx * px + ny * py + nz * pz + d < 0) {
                    return false;
                }
            }
            return true;
        }

        test("box at origin is inside frustum looking at origin", () => {
            const proj = perspective(60, 1, 0.1, 100);
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[12] = 0;
            cameraWorld[13] = 0;
            cameraWorld[14] = 5;
            cameraWorld[15] = 1;

            const view = invert(cameraWorld);
            const viewProj = multiply(proj, view);
            const planes = extractFrustumPlanes(viewProj);

            const inside = testFrustum([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], planes);
            expect(inside).toBe(true);
        });

        test("box behind camera is culled", () => {
            const proj = perspective(60, 1, 0.1, 100);
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[12] = 0;
            cameraWorld[13] = 0;
            cameraWorld[14] = 5;
            cameraWorld[15] = 1;

            const view = invert(cameraWorld);
            const viewProj = multiply(proj, view);
            const planes = extractFrustumPlanes(viewProj);

            const inside = testFrustum([9.5, -0.5, -0.5], [10.5, 0.5, 0.5], planes);
            expect(inside).toBe(false);
        });

        test("box far to the left is culled", () => {
            const proj = perspective(60, 1, 0.1, 100);
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[12] = 0;
            cameraWorld[13] = 0;
            cameraWorld[14] = 5;
            cameraWorld[15] = 1;

            const view = invert(cameraWorld);
            const viewProj = multiply(proj, view);
            const planes = extractFrustumPlanes(viewProj);

            const inside = testFrustum([-100, -0.5, -0.5], [-99, 0.5, 0.5], planes);
            expect(inside).toBe(false);
        });

        test("box beyond far plane is culled", () => {
            const proj = perspective(60, 1, 0.1, 10);
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[12] = 0;
            cameraWorld[13] = 0;
            cameraWorld[14] = 5;
            cameraWorld[15] = 1;

            const view = invert(cameraWorld);
            const viewProj = multiply(proj, view);
            const planes = extractFrustumPlanes(viewProj);

            const inside = testFrustum([-0.5, -0.5, -20], [0.5, 0.5, -19], planes);
            expect(inside).toBe(false);
        });

        test("box partially inside is not culled", () => {
            const proj = perspective(60, 1, 0.1, 100);
            const cameraWorld = new Float32Array(16);
            cameraWorld[0] = 1;
            cameraWorld[5] = 1;
            cameraWorld[10] = 1;
            cameraWorld[12] = 0;
            cameraWorld[13] = 0;
            cameraWorld[14] = 5;
            cameraWorld[15] = 1;

            const view = invert(cameraWorld);
            const viewProj = multiply(proj, view);
            const planes = extractFrustumPlanes(viewProj);

            const inside = testFrustum([-5, -5, -1], [5, 5, 1], planes);
            expect(inside).toBe(true);
        });
    });
});
