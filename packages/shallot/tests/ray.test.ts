import { describe, test, expect } from "bun:test";
import { generateRay, rayTriangle } from "../src/standard/physics/raycast";
import { extractTriangles } from "../src/extras/raytracing/triangle";
import { createBox } from "../src/standard/render/mesh";

const EPSILON = 1e-5;

function approxEqual(a: number, b: number, epsilon = EPSILON): boolean {
    return Math.abs(a - b) < epsilon;
}

describe("Ray-Triangle Intersection", () => {
    test("hit triangle at origin facing +Z", () => {
        const result = rayTriangle(0, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1, 0, 0, 1, 0);
        expect(result).not.toBeNull();
        expect(approxEqual(result!.t, 1)).toBe(true);
    });

    test("miss when ray parallel to triangle", () => {
        const result = rayTriangle(0, 0, 1, 1, 0, 0, -1, -1, 0, 1, -1, 0, 0, 1, 0);
        expect(result).toBeNull();
    });

    test("miss when ray points away", () => {
        const result = rayTriangle(0, 0, 1, 0, 0, 1, -1, -1, 0, 1, -1, 0, 0, 1, 0);
        expect(result).toBeNull();
    });

    test("miss when ray passes outside bounds", () => {
        const result = rayTriangle(5, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1, 0, 0, 1, 0);
        expect(result).toBeNull();
    });

    test("degenerate triangle (zero area) returns no hit", () => {
        const result = rayTriangle(0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        expect(result).toBeNull();
    });

    test("degenerate triangle (collinear points) returns no hit", () => {
        const result = rayTriangle(0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 2, 0, 0);
        expect(result).toBeNull();
    });

    test("hit from behind triangle (backface)", () => {
        const result = rayTriangle(0, 0, -1, 0, 0, 1, -1, -1, 0, 1, -1, 0, 0, 1, 0);
        expect(result).not.toBeNull();
        expect(approxEqual(result!.t, 1)).toBe(true);
    });

    test("normal points toward ray origin", () => {
        const result = rayTriangle(0, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1, 0, 0, 1, 0);
        expect(result).not.toBeNull();
        expect(result!.nz).toBeGreaterThan(0);
    });
});

describe("Triangle Extraction", () => {
    test("correct count from box (12 triangles)", () => {
        const box = createBox();
        const triangles = extractTriangles(box, 42);
        expect(triangles.length).toBe(12);
    });

    test("entity ID preserved on all triangles", () => {
        const box = createBox();
        const entityId = 123;
        const triangles = extractTriangles(box, entityId);
        for (const tri of triangles) {
            expect(tri.entityId).toBe(entityId);
        }
    });

    test("vertices match mesh data", () => {
        const box = createBox();
        const triangles = extractTriangles(box, 0);
        const firstTriangle = triangles[0];
        expect(approxEqual(firstTriangle.v0.x, box.vertices[0])).toBe(true);
        expect(approxEqual(firstTriangle.v0.y, box.vertices[1])).toBe(true);
        expect(approxEqual(firstTriangle.v0.z, box.vertices[2])).toBe(true);
    });

    test("normals extracted correctly", () => {
        const box = createBox();
        const triangles = extractTriangles(box, 0);
        const firstTriangle = triangles[0];
        expect(approxEqual(firstTriangle.n0.x, box.vertices[3])).toBe(true);
        expect(approxEqual(firstTriangle.n0.y, box.vertices[4])).toBe(true);
        expect(approxEqual(firstTriangle.n0.z, box.vertices[5])).toBe(true);
    });

    test("transform applied to positions", () => {
        const box = createBox();
        const translation = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1]);
        const triangles = extractTriangles(box, 0, translation);
        const firstTriangle = triangles[0];
        expect(approxEqual(firstTriangle.v0.x, box.vertices[0] + 10)).toBe(true);
        expect(approxEqual(firstTriangle.v0.y, box.vertices[1] + 20)).toBe(true);
        expect(approxEqual(firstTriangle.v0.z, box.vertices[2] + 30)).toBe(true);
    });

    test("transform applies rotation to normals", () => {
        const box = createBox();
        const rotate90Y = new Float32Array([0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]);
        const triangles = extractTriangles(box, 0, rotate90Y);
        const frontFace = triangles[0];
        expect(approxEqual(frontFace.n0.x, 1)).toBe(true);
        expect(approxEqual(frontFace.n0.y, 0)).toBe(true);
        expect(approxEqual(frontFace.n0.z, 0)).toBe(true);
    });

    test("normals remain normalized after transform", () => {
        const box = createBox();
        const scale = new Float32Array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
        const triangles = extractTriangles(box, 0, scale);
        for (const tri of triangles) {
            const len0 = Math.sqrt(tri.n0.x * tri.n0.x + tri.n0.y * tri.n0.y + tri.n0.z * tri.n0.z);
            const len1 = Math.sqrt(tri.n1.x * tri.n1.x + tri.n1.y * tri.n1.y + tri.n1.z * tri.n1.z);
            const len2 = Math.sqrt(tri.n2.x * tri.n2.x + tri.n2.y * tri.n2.y + tri.n2.z * tri.n2.z);
            expect(approxEqual(len0, 1)).toBe(true);
            expect(approxEqual(len1, 1)).toBe(true);
            expect(approxEqual(len2, 1)).toBe(true);
        }
    });
});

describe("Ray Generation", () => {
    const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

    test("ray from camera origin", () => {
        const ray = generateRay(0.5, 0.5, 800, 600, 60, 0.1, identity);
        expect(approxEqual(ray.origin.x, 0, 0.2)).toBe(true);
        expect(approxEqual(ray.origin.y, 0, 0.2)).toBe(true);
        expect(approxEqual(ray.origin.z, -0.1, 0.01)).toBe(true);
    });

    test("center screen points down -Z", () => {
        const ray = generateRay(0.5, 0.5, 800, 600, 60, 0, identity);
        expect(approxEqual(ray.direction.x, 0)).toBe(true);
        expect(approxEqual(ray.direction.y, 0)).toBe(true);
        expect(approxEqual(ray.direction.z, -1)).toBe(true);
    });

    test("rays spread based on FOV", () => {
        const narrow = generateRay(1, 0.5, 800, 800, 30, 0, identity);
        const wide = generateRay(1, 0.5, 800, 800, 90, 0, identity);
        expect(Math.abs(narrow.direction.x)).toBeLessThan(Math.abs(wide.direction.x));
    });

    test("aspect ratio handled correctly", () => {
        const wideRay = generateRay(1, 0.5, 1600, 800, 60, 0, identity);
        const squareRay = generateRay(1, 0.5, 800, 800, 60, 0, identity);
        expect(Math.abs(wideRay.direction.x)).toBeGreaterThan(Math.abs(squareRay.direction.x));
    });

    test("camera rotation transforms ray", () => {
        const rotate90Y = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1]);
        const ray = generateRay(0.5, 0.5, 800, 600, 60, 0, rotate90Y);
        expect(approxEqual(ray.direction.x, 1)).toBe(true);
        expect(approxEqual(ray.direction.y, 0)).toBe(true);
        expect(approxEqual(ray.direction.z, 0)).toBe(true);
    });

    test("direction is normalized", () => {
        const ray = generateRay(0.25, 0.75, 800, 600, 60, 0, identity);
        const len = Math.sqrt(
            ray.direction.x * ray.direction.x +
                ray.direction.y * ray.direction.y +
                ray.direction.z * ray.direction.z,
        );
        expect(approxEqual(len, 1)).toBe(true);
    });

    test("camera translation affects origin", () => {
        const translated = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 10, 15, 1]);
        const ray = generateRay(0.5, 0.5, 800, 600, 60, 0, translated);
        expect(approxEqual(ray.origin.x, 5)).toBe(true);
        expect(approxEqual(ray.origin.y, 10)).toBe(true);
        expect(approxEqual(ray.origin.z, 15)).toBe(true);
    });

    test("top-left corner ray points up and left", () => {
        const ray = generateRay(0, 0, 800, 600, 60, 0, identity);
        expect(ray.direction.x).toBeLessThan(0);
        expect(ray.direction.y).toBeGreaterThan(0);
        expect(ray.direction.z).toBeLessThan(0);
    });

    test("bottom-right corner ray points down and right", () => {
        const ray = generateRay(1, 1, 800, 600, 60, 0, identity);
        expect(ray.direction.x).toBeGreaterThan(0);
        expect(ray.direction.y).toBeLessThan(0);
        expect(ray.direction.z).toBeLessThan(0);
    });
});
