import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType, defaultSurfaceMaterial, ShapeType } from "../common/types";
import { createCompound } from "./compound";
import { createGrid } from "./heightfield";
import { makeBoxHull } from "./hull";
import { createGridMesh } from "./mesh";
import { readCompoundChildren, readShapeHeightField, readShapeMesh } from "./readgeometry";

test("debug-draw geometry outputs are fresh views reconstructed from kernel uploads", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Static });
        const meshData = createGridMesh(3, 3, 1, 0, true);
        const meshShape = body.createMesh({}, meshData)!;
        const mesh = readShapeMesh(world.state, meshShape.id.index1 - 1);
        expect(mesh.data).toEqual(meshData);
        expect(mesh.scale).toEqual({ x: 1, y: 1, z: 1 });
        mesh.data.vertices[0].x = 500;
        expect(readShapeMesh(world.state, meshShape.id.index1 - 1).data.vertices[0].x).not.toBe(
            500,
        );

        const fieldData = createGrid(3, 3, { x: 1, y: 1, z: 1 }, false);
        const fieldShape = body.createHeightField({}, fieldData)!;
        expect(readShapeHeightField(world.state, fieldShape.id.index1 - 1)).toEqual(fieldData);

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
        const compoundShape = body.createCompound({}, compoundData)!;
        const children = readCompoundChildren(world.state, compoundShape.id.index1 - 1);
        expect(children.map((child) => child.type)).toEqual([
            ShapeType.Capsule,
            ShapeType.Hull,
            ShapeType.Mesh,
            ShapeType.Sphere,
        ]);
        expect(children[0].geometry).toEqual(compoundData.capsules[0].capsule);
        expect(children[1].geometry).toEqual(hull);
        expect(children[2].geometry).toEqual({ data: meshData, scale: { x: 1, y: 2, z: 1 } });
        expect(children[3].geometry).toEqual(compoundData.spheres[0].sphere);
        expect(children[1].transform).toEqual(compoundData.hulls[0].transform);
        expect(children[2].transform).toEqual(compoundData.meshes[0].transform);
    } finally {
        world.destroy();
    }
});
