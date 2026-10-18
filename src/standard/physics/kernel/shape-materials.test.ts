import { expect, test } from "bun:test";
import { createCompound, defaultSurfaceMaterial } from "../api";
import { restore, snapshot } from "../api/snapshot";
import { BodyType } from "../common/types";
import {
    createCompoundShape,
    createSphereShape,
    destroyShape,
    getShapeMaterial,
    getShapeMaterials,
} from "../shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import { kernel } from "./kernel";
import { SHAPE_STRIDE } from "./shapecolumns";

const sphere = { center: { x: 0, y: 0, z: 0 }, radius: 1 };

test("a single-material compound owns its array through replacement, restore and reset", () => {
    const state = createTestWorld();
    try {
        const material = {
            ...defaultSurfaceMaterial(),
            friction: 0.75,
            userMaterialId: 0x123456789abcdef0n,
        };
        const compound = createCompound({
            spheres: [{ sphere, material }],
        })!;
        const body = createTestBody(state, { type: BodyType.Static });
        const id = createCompoundShape(state, body, testShapeDef(), compound)!;
        const k = kernel(state.ecsState);
        // An owned material array lives outside the shape's resident record.
        const owned = () => {
            state.shapeStore.refreshViews();
            const record = state.shapeStore.shapeU.byteOffset + id * SHAPE_STRIDE * 4;
            const ptr = k.shapeMaterialPtr(state.worldId, id);
            return ptr < record || ptr >= record + SHAPE_STRIDE * 4;
        };
        const out = defaultSurfaceMaterial();
        expect(owned()).toBe(true);
        expect(getShapeMaterial(state, id, out)).toEqual(material);
        const saved = snapshot(state);
        for (let i = 0; i < 3; ++i) {
            state.shapeStore.writeMaterials(state, id, { ...material, friction: 0.25 });
            expect(getShapeMaterial(state, id, out).friction).toBe(0.25);
            restore(state, saved);
            expect(owned()).toBe(true);
            expect(getShapeMaterial(state, id, out)).toEqual(material);
            const fresh = getShapeMaterials(state, id);
            fresh[0].tangentVelocity[0] = 1;
            expect(getShapeMaterials(state, id)[0]).toEqual(material);
        }
        destroyShape(state, id, true);
        expect(owned()).toBe(false);
        expect(k.shapeMaterialCount(state.worldId, id)).toBe(0);
        restore(state, saved);
        expect(owned()).toBe(true);
        expect(getShapeMaterial(state, id, out)).toEqual(material);
        destroyShape(state, id, true);
        expect(owned()).toBe(false);
    } finally {
        destroyTestWorld(state);
    }
});

test("one material is inline and multiple materials are contiguous, owned and restored", () => {
    const state = createTestWorld();
    try {
        const materials = [
            { ...defaultSurfaceMaterial(), friction: 0.5, userMaterialId: 0x123456789abcdef0n },
            { ...defaultSurfaceMaterial(), friction: 0.25, userMaterialId: 0xfedcba9876543210n },
        ];
        const body = createTestBody(state);
        const one = createSphereShape(
            state,
            body,
            testShapeDef({ baseMaterial: materials[0] }),
            sphere,
        )!;
        const many = createSphereShape(state, body, testShapeDef({ materials }), sphere)!;
        const k = kernel(state.ecsState);
        const read = (id: number) => getShapeMaterials(state, id);
        expect(k.shapeMaterialPtr(state.worldId, one)).toBe(
            state.shapeStore.shapeU.byteOffset + (one * SHAPE_STRIDE + 26) * 4,
        );
        expect(read(one)).toEqual([materials[0]]);
        expect(read(many)).toEqual(materials);
        const ptr = k.shapeMaterialPtr(state.worldId, many);
        expect(new Float32Array(k.memory.buffer, ptr, 20)[10]).toBe(materials[1].friction);
        materials[1].friction = 0.75;
        expect(read(many)[1].friction).toBe(0.25);
        const saved = snapshot(state);
        destroyShape(state, many, true);
        createSphereShape(state, body, testShapeDef({ baseMaterial: materials[1] }), sphere);
        restore(state, saved);
        expect(read(many)[1].friction).toBe(0.25);
        expect(read(many)[1].userMaterialId).toBe(0xfedcba9876543210n);
        expect(read(one)).toEqual([materials[0]]);
    } finally {
        destroyTestWorld(state);
    }
});
