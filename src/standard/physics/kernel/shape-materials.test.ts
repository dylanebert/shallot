import { expect, test } from "bun:test";
import { defaultSurfaceMaterial, PhysicsWorld } from "../api";
import { getShapeMaterials } from "../shapes/shape";
import { kernel } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";

test("one material is inline and multiple materials are contiguous, owned and restored", () => {
    const world = new PhysicsWorld();
    try {
        const materials = [
            { ...defaultSurfaceMaterial(), friction: 0.5, userMaterialId: 0x123456789abcdef0n },
            { ...defaultSurfaceMaterial(), friction: 0.25, userMaterialId: 0xfedcba9876543210n },
        ];
        const body = world.createBody({});
        const sphere = { center: { x: 0, y: 0, z: 0 }, radius: 1 };
        const one = body.createSphere({ baseMaterial: materials[0] }, sphere);
        const many = body.createSphere({ materials }, sphere);
        const state = world.state;
        const k = kernel(state.ecsState);
        const oneId = one.id.index1 - 1;
        const manyId = many.id.index1 - 1;
        const read = (id: number) => getShapeMaterials(state.ecsState, state.shapes[id]);
        expect(k.shapeMaterialPtr(state.worldId, oneId)).toBe(
            state.shapeStore.shapeU.byteOffset + (oneId * SHAPE_STRIDE + 52) * 4,
        );
        expect(state.shapeStore.shapeU[oneId * SHAPE_STRIDE + 16]).toBe(0);
        expect(read(oneId)).toEqual([materials[0]]);
        expect(read(manyId)).toEqual(materials);
        const ptr = k.shapeMaterialPtr(state.worldId, manyId);
        expect(new Float32Array(k.memory.buffer, ptr, 18)[9]).toBe(materials[1].friction);
        materials[1].friction = 0.75;
        expect(read(manyId)[1].friction).toBe(0.25);
        const saved = world.snapshot();
        many.destroy();
        body.createSphere({ baseMaterial: materials[1] }, sphere);
        world.restore(saved);
        expect(read(manyId)[1].friction).toBe(0.25);
        expect(read(manyId)[1].userMaterialId).toBe(0xfedcba9876543210n);
        expect(read(oneId)).toEqual([materials[0]]);
    } finally {
        world.destroy();
    }
});
