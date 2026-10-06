import { expect, test } from "bun:test";
import { BodyType, createCompound, defaultSurfaceMaterial, PhysicsWorld } from "../api";
import { getShapeMaterial, getShapeMaterials } from "../shapes/shape";
import { kernel } from "./kernel";
import { S_MATERIAL_HEAD, SHAPE_STRIDE } from "./shapecolumns";

test("a single-material compound owns its array through replacement, restore and reset", () => {
    const world = new PhysicsWorld();
    try {
        const material = {
            ...defaultSurfaceMaterial(),
            friction: 0.75,
            userMaterialId: 0x123456789abcdef0n,
        };
        const compound = createCompound({
            spheres: [{ sphere: { center: { x: 0, y: 0, z: 0 }, radius: 1 }, material }],
        })!;
        const shape = world.createBody({ type: BodyType.Static }).createCompound({}, compound)!;
        const id = shape.id.index1 - 1;
        const state = world.state;
        const k = kernel(state.ecsState);
        const owned = () => state.shapeStore.shapeU[id * SHAPE_STRIDE + S_MATERIAL_HEAD];
        const out = defaultSurfaceMaterial();
        expect(owned()).toBeGreaterThan(0);
        expect(k.shapeMaterialPtr(state.worldId, id)).toBe(owned());
        expect(getShapeMaterial(state, id, out)).toEqual(material);
        const saved = world.snapshot();
        for (let i = 0; i < 3; ++i) {
            state.shapeStore.writeMaterials(state, id, { ...material, friction: 0.25 });
            expect(getShapeMaterial(state, id, out).friction).toBe(0.25);
            world.restore(saved);
            expect(owned()).toBeGreaterThan(0);
            expect(k.shapeMaterialPtr(state.worldId, id)).toBe(owned());
            expect(getShapeMaterial(state, id, out)).toEqual(material);
            const fresh = getShapeMaterials(state, id);
            fresh[0].tangentVelocity.x = 1;
            expect(getShapeMaterials(state, id)[0]).toEqual(material);
        }
        state.shapeStore.destroyMaterials(state, id);
        expect(owned()).toBe(0);
        expect(k.shapeMaterialCount(state.worldId, id)).toBe(0);
        world.restore(saved);
        expect(getShapeMaterial(state, id, out)).toEqual(material);
        shape.destroy();
        expect(owned()).toBe(0);
    } finally {
        world.destroy();
    }
});

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
        const read = (id: number) => getShapeMaterials(state, id);
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
