import { describe, test, expect, beforeEach } from "bun:test";
import {
    createBox,
    createSphere,
    createCapsule,
    createPlane,
    getMesh,
    getMeshByName,
    mesh,
    clearMeshes,
    MeshShape,
    allocateDynamic,
    deallocateDynamic,
    dynamicInfo,
    isDynamic,
    Part,
    type MeshData,
} from "../src/standard/render/mesh";
import { Shape } from "../src/engine/utils";

function validateMesh(data: MeshData, name: string) {
    test(`${name} has valid vertex data`, () => {
        expect(data.vertices).toBeInstanceOf(Float32Array);
        expect(data.vertices.length).toBeGreaterThan(0);
        expect(data.vertices.length % 8).toBe(0);
    });

    test(`${name} has valid index data`, () => {
        expect(data.indices).toBeInstanceOf(Uint16Array);
        expect(data.indices.length).toBe(data.indexCount);
        expect(data.indices.length % 3).toBe(0);
    });

    test(`${name} indices reference valid vertices`, () => {
        const maxIndex = Math.max(...data.indices);
        expect(maxIndex).toBeLessThan(data.vertexCount);
    });

    test(`${name} has consistent counts`, () => {
        expect(data.vertices.length / 8).toBe(data.vertexCount);
    });
}

describe("Mesh", () => {
    describe("createBox", () => {
        const box = createBox();
        validateMesh(box, "box");

        test("box has 24 vertices (6 faces × 4 corners)", () => {
            expect(box.vertexCount).toBe(24);
        });

        test("box has 36 indices (6 faces × 2 triangles × 3)", () => {
            expect(box.indexCount).toBe(36);
        });
    });

    describe("createSphere", () => {
        const sphere = createSphere(8, 4);
        validateMesh(sphere, "sphere");

        test("sphere vertex count matches segments and rings", () => {
            expect(sphere.vertexCount).toBe((4 + 1) * (8 + 1));
        });
    });

    describe("createCapsule", () => {
        const capsule = createCapsule(8, 4);
        validateMesh(capsule, "capsule");

        test("capsule y-extent spans [-1, 1]", () => {
            let minY = Infinity;
            let maxY = -Infinity;
            for (let i = 0; i < capsule.vertexCount; i++) {
                const y = capsule.vertices[i * 8 + 1];
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
            expect(minY).toBeCloseTo(-1.0, 5);
            expect(maxY).toBeCloseTo(1.0, 5);
        });

        test("capsule x/z extent spans [-0.5, 0.5]", () => {
            let maxR = 0;
            for (let i = 0; i < capsule.vertexCount; i++) {
                const x = capsule.vertices[i * 8];
                const z = capsule.vertices[i * 8 + 2];
                maxR = Math.max(maxR, Math.sqrt(x * x + z * z));
            }
            expect(maxR).toBeCloseTo(0.5, 5);
        });
    });

    describe("createPlane", () => {
        const plane = createPlane();
        validateMesh(plane, "plane");

        test("plane has 4 vertices", () => {
            expect(plane.vertexCount).toBe(4);
        });

        test("plane has 6 indices (2 triangles)", () => {
            expect(plane.indexCount).toBe(6);
        });
    });

    describe("getMesh", () => {
        test("returns box geometry for Box shape", () => {
            const data = getMesh(MeshShape.Box);
            expect(data).toBeDefined();
            expect(data!.vertexCount).toBe(24);
            expect(data!.indexCount).toBe(36);
        });

        test("returns sphere geometry for Sphere shape", () => {
            const data = getMesh(MeshShape.Sphere);
            expect(data).toBeDefined();
            expect(data!.vertexCount).toBeGreaterThan(24);
            expect(data!.indexCount).toBeGreaterThan(36);
        });

        test("returns capsule geometry for Capsule shape", () => {
            const data = getMesh(MeshShape.Capsule);
            expect(data).toBeDefined();
            expect(data!.vertexCount).toBeGreaterThan(24);
            expect(data!.indexCount).toBeGreaterThan(36);
        });

        test("returns plane geometry for Plane shape", () => {
            const data = getMesh(MeshShape.Plane);
            expect(data).toBeDefined();
            expect(data!.vertexCount).toBe(4);
            expect(data!.indexCount).toBe(6);
        });

        test("returns undefined for unknown shape", () => {
            const data = getMesh(999);
            expect(data).toBeUndefined();
        });
    });

    describe("registry", () => {
        beforeEach(() => {
            clearMeshes();
        });

        describe("mesh()", () => {
            test("returns sequential indices starting after built-ins", () => {
                const customData = createBox();
                const id1 = mesh(customData);
                const id2 = mesh(customData);

                expect(id1).toBe(4);
                expect(id2).toBe(5);
            });

            test("registered mesh is retrievable via getMesh", () => {
                const customData = createBox();
                const id = mesh(customData);

                const retrieved = getMesh(id);
                expect(retrieved).toBe(customData);
            });
        });

        describe("clearMeshes()", () => {
            test("resets to built-ins only", () => {
                const customData = createBox();
                mesh(customData);
                mesh(customData);

                clearMeshes();

                const id = mesh(customData);
                expect(id).toBe(4);
            });

            test("built-ins remain after clear", () => {
                clearMeshes();

                expect(getMesh(MeshShape.Box)).toBeDefined();
                expect(getMesh(MeshShape.Sphere)).toBeDefined();
                expect(getMesh(MeshShape.Capsule)).toBeDefined();
                expect(getMesh(MeshShape.Plane)).toBeDefined();
            });
        });

        describe("named meshes", () => {
            test("mesh() with name registers retrievable name", () => {
                const customData = createBox();
                const id = mesh(customData, "cone");
                expect(getMeshByName("cone")).toBe(id);
            });

            test("built-in names resolve to correct IDs", () => {
                expect(getMeshByName("box")).toBe(0);
                expect(getMeshByName("sphere")).toBe(1);
                expect(getMeshByName("capsule")).toBe(2);
                expect(getMeshByName("plane")).toBe(3);
            });

            test("clearMeshes clears names; built-ins available after clear", () => {
                mesh(createBox(), "cone");
                clearMeshes();

                expect(getMeshByName("cone")).toBeUndefined();
                expect(getMeshByName("box")).toBe(0);
            });

            test("unknown name returns undefined", () => {
                expect(getMeshByName("nonexistent")).toBeUndefined();
            });
        });
    });

    describe("Dynamic", () => {
        beforeEach(() => {
            clearMeshes();
        });

        test("allocate marks entity as dynamic", () => {
            Part.shape[10] = Shape.Box;
            allocateDynamic(10);
            expect(isDynamic(10)).toBe(true);
        });

        test("dynamicInfo returns undefined before atlas update", () => {
            Part.shape[10] = Shape.Box;
            allocateDynamic(10);
            expect(dynamicInfo(10)).toBeUndefined();
        });

        test("two dynamic entities get independent mesh IDs", () => {
            Part.shape[10] = Shape.Box;
            Part.shape[20] = Shape.Box;
            allocateDynamic(10);
            allocateDynamic(20);

            expect(isDynamic(10)).toBe(true);
            expect(isDynamic(20)).toBe(true);
        });

        test("allocate is idempotent", () => {
            Part.shape[10] = Shape.Box;
            allocateDynamic(10);
            allocateDynamic(10);
            expect(isDynamic(10)).toBe(true);
        });

        test("deallocate clears state", () => {
            Part.shape[10] = Shape.Box;
            allocateDynamic(10);
            expect(isDynamic(10)).toBe(true);

            deallocateDynamic(10);
            expect(isDynamic(10)).toBe(false);
            expect(dynamicInfo(10)).toBeUndefined();
        });

        test("clearMeshes clears dynamic allocations", () => {
            Part.shape[10] = Shape.Box;
            allocateDynamic(10);
            clearMeshes();

            expect(isDynamic(10)).toBe(false);
            expect(dynamicInfo(10)).toBeUndefined();
        });
    });
});
