import { expect, test } from "bun:test";
import { BodyType, defaultSurfaceMaterial, ShapeType } from "../common/types";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import { makeBoxHull } from "./build-hull";
import { createCompound } from "./compound";
import { createGrid } from "./heightfield";
import { createGridMesh } from "./mesh";
import { readCompoundChildren, readShapeHeightField, readShapeMesh } from "./readgeometry";
import { createCompoundShape, createHeightFieldShape, createMeshShape } from "./shape";

test("debug-draw geometry outputs are fresh views reconstructed from kernel uploads", () => {
    const world = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Static });
        const meshData = createGridMesh(3, 3, 1, 0, true);
        const meshShape = createMeshShape(world, body, testShapeDef(), meshData, {
            x: 1,
            y: 1,
            z: 1,
        })!;
        const mesh = readShapeMesh(world, meshShape);
        expect(mesh.data).toEqual(meshData);
        expect(mesh.scale).toEqual({ x: 1, y: 1, z: 1 });
        mesh.data.vertices[0]!.x = 500;
        expect(readShapeMesh(world, meshShape).data.vertices[0]!.x).not.toBe(500);

        const fieldData = createGrid(3, 3, { x: 1, y: 1, z: 1 }, false);
        const fieldShape = createHeightFieldShape(world, body, testShapeDef(), fieldData)!;
        expect(readShapeHeightField(world, fieldShape)).toEqual(fieldData);

        const hull = makeBoxHull(1, 1, 1);
        const compoundData = createCompound({
            capsules: [
                {
                    capsule: {
                        center1: { x: 0, y: 0, z: 0 },
                        center2: { x: 0, y: 1, z: 0 },
                        radius: 0.25,
                    },
                    material: defaultSurfaceMaterial(),
                },
            ],
            hulls: [
                {
                    hull,
                    transform: { p: { x: 2, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
                    material: defaultSurfaceMaterial(),
                },
            ],
            meshes: [
                {
                    meshData,
                    transform: { p: { x: 0, y: 0, z: 2 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
                    scale: { x: 1, y: 2, z: 1 },
                    materials: [defaultSurfaceMaterial()],
                    materialCount: 1,
                },
            ],
            spheres: [
                {
                    sphere: { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                    material: defaultSurfaceMaterial(),
                },
            ],
        })!;
        const compoundShape = createCompoundShape(world, body, testShapeDef(), compoundData)!;
        const children = readCompoundChildren(world, compoundShape);
        expect(children.map((child) => child.type)).toEqual([
            ShapeType.Capsule,
            ShapeType.Hull,
            ShapeType.Mesh,
            ShapeType.Sphere,
        ]);
        expect(children[0]!.geometry).toEqual(compoundData.capsules[0]!.capsule);
        expect(children[1]!.geometry).toEqual(hull);
        expect(children[2]!.geometry).toEqual({ data: meshData, scale: { x: 1, y: 2, z: 1 } });
        expect(children[3]!.geometry).toEqual(compoundData.spheres[0]!.sphere);
        expect(children[1]!.transform).toEqual(compoundData.hulls[0]!.transform);
        expect(children[2]!.transform).toEqual(compoundData.meshes[0]!.transform);
    } finally {
        destroyTestWorld(world);
    }
});
