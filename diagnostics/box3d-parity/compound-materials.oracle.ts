import { expect, test } from "bun:test";
import { defaultSurfaceMaterial } from "../../src/standard/physics/common/types";
import { createCompound } from "../../src/standard/physics/shapes/compound";
import { createMesh } from "../../src/standard/physics/shapes/mesh";
import { createCompoundShape } from "../../src/standard/physics/shapes/shape";
import { defaultBodyDef, defaultShapeDef, defaultWorldDef } from "../../src/standard/physics/common/types";
import { createBody } from "../../src/standard/physics/world/body";
import { createWorld, getWorld, destroyWorld } from "../../src/standard/physics/world/world";
import { nativeSseOutput } from "./native-evidence";

test("a compound mesh child retains six distinct materials", () => {
    const vertices = Array.from({ length: 6 }, (_, i) => [
        { x: 3 * i, y: 0, z: 0 },
        { x: 3 * i, y: 0, z: 1 },
        { x: 3 * i + 1, y: 0, z: 0 },
    ]).flat();
    const meshData = createMesh({ vertices, indices: vertices.map((_, i) => i), materialIndices: [0, 1, 2, 3, 4, 5] })!;
    const compound = createCompound({ meshes: [{
        meshData,
        transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        scale: { x: 1, y: 1, z: 1 },
        materials: Array.from({ length: 6 }, (_, i) => ({ ...defaultSurfaceMaterial(), userMaterialId: BigInt(100 + i) })),
        materialCount: 6,
    }] })!;
    expect(Number(nativeSseOutput("compound-materials.c", "").trim())).toBe(6);
    expect(compound.materialCount).toBe(6);
    expect(compound.meshes[0].materialIndices).toEqual([0, 1, 2, 3, 4, 5]);
    const world = getWorld(createWorld(undefined, defaultWorldDef()))!;
    try {
        const body = createBody(world, defaultBodyDef());
        expect(createCompoundShape(world, body, defaultShapeDef(), compound)).not.toBeNull();
    } finally {
        destroyWorld(world);
    }
});
