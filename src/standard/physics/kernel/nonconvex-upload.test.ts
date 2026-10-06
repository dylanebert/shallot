import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { xf } from "../common/math";
import { BodyType, defaultSurfaceMaterial, ShapeType } from "../common/types";
import { createCompound } from "../shapes/compound";
import { createGrid } from "../shapes/heightfield";
import { makeBoxHull } from "../shapes/hull";
import { hash64NonZero } from "../shapes/hullbytes";
import { createBoxMesh, createGridMesh } from "../shapes/mesh";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./shapecolumns";

function image(world: PhysicsWorld, shape: number): Uint8Array {
    const store = world.state.shapeStore;
    store.refreshViews();
    const u = store.materialU;
    const record = store.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
    const sizeWord = store.shapeU[shape * SHAPE_STRIDE] === ShapeType.Compound ? 2 : 4;
    return new Uint8Array(u.buffer, record, u[record / 4 + sizeWord]);
}
function checkHash(bytes: Uint8Array): void {
    const copy = bytes.slice();
    const view = new DataView(copy.buffer);
    const hash = view.getBigUint64(8, true);
    view.setBigUint64(8, 0n, true);
    expect(hash64NonZero(copy)).toBe(hash);
    copy[copy.length - 1] ^= 1;
    expect(hash64NonZero(copy)).not.toBe(hash);
}

test("mesh and height-field uploads have native headers, packed nodes, byte flags/materials and short heights", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Static });
        const mesh = createGridMesh(4, 4, 1, 0, true);
        const meshShape = body.createMesh({}, mesh)!;
        const field = createGrid(4, 4, { x: 1, y: 1, z: 1 }, false);
        const fieldShape = body.createHeightField({}, field)!;
        const meshBytes = image(world, meshShape.id.index1 - 1);
        const m = new DataView(meshBytes.buffer, meshBytes.byteOffset, meshBytes.byteLength);
        expect(m.getBigUint64(0, true)).toBe(0xaaab9a00f1a8aaf7n);
        expect(m.getUint32(56, true)).toBe(96);
        expect(m.getUint32(64, true) - 96).toBe(mesh.nodes.length * 32);
        for (let i = 0; i < mesh.nodes.length; ++i) {
            const node = mesh.nodes[i];
            expect(m.getUint32(96 + i * 32 + 12, true)).toBe(
                node.leaf
                    ? ((node.triangleCount << 2) | 3) >>> 0
                    : ((node.childOffset << 2) | node.axis) >>> 0,
            );
        }
        expect(
            Array.from(
                new Uint8Array(
                    meshBytes.buffer,
                    meshBytes.byteOffset + m.getUint32(80, true),
                    mesh.triangles.length,
                ),
            ),
        ).toEqual(mesh.materialIndices);
        expect(
            Array.from(
                new Uint8Array(
                    meshBytes.buffer,
                    meshBytes.byteOffset + m.getUint32(88, true),
                    mesh.triangles.length,
                ),
            ),
        ).toEqual(mesh.flags);
        checkHash(meshBytes);
        const fieldBytes = image(world, fieldShape.id.index1 - 1);
        const h = new DataView(fieldBytes.buffer, fieldBytes.byteOffset, fieldBytes.byteLength);
        expect(h.getBigUint64(0, true)).toBe(0x8e41e5fb084848f8n);
        expect(h.getUint32(76, true)).toBe(96);
        expect(
            Array.from(
                new Uint16Array(
                    fieldBytes.buffer,
                    fieldBytes.byteOffset + h.getUint32(76, true),
                    field.compressedHeights.length,
                ),
            ),
        ).toEqual(field.compressedHeights);
        expect(
            Array.from(
                new Uint8Array(
                    fieldBytes.buffer,
                    fieldBytes.byteOffset + h.getUint32(80, true),
                    field.materialIndices.length,
                ),
            ),
        ).toEqual(field.materialIndices);
        expect(
            Array.from(
                new Uint8Array(
                    fieldBytes.buffer,
                    fieldBytes.byteOffset + h.getUint32(84, true),
                    field.flags.length,
                ),
            ),
        ).toEqual(field.flags);
        checkHash(fieldBytes);
    } finally {
        world.destroy();
    }
});

test("compound uploads use Box3D header, tree, material and child-blob offsets", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({ type: BodyType.Static });
        const hull = makeBoxHull(0.5, 0.5, 0.5);
        const meshData = createBoxMesh({ x: 0, y: 0, z: 0 }, { x: 0.5, y: 0.5, z: 0.5 }, false);
        const compound = createCompound({
            capsules: [
                {
                    capsule: {
                        center1: { x: -1, y: 0, z: 0 },
                        center2: { x: 1, y: 0, z: 0 },
                        radius: 0.25,
                    },
                    material: defaultSurfaceMaterial(),
                },
            ],
            hulls: [{ hull, transform: xf.identity(), material: defaultSurfaceMaterial() }],
            meshes: [
                {
                    meshData,
                    transform: xf.identity(),
                    scale: { x: 1, y: 1, z: 1 },
                    materials: [defaultSurfaceMaterial()],
                    materialCount: 1,
                },
            ],
            spheres: [
                {
                    sphere: { center: { x: 5, y: 0, z: 0 }, radius: 0.5 },
                    material: defaultSurfaceMaterial(),
                },
            ],
        })!;
        const shape = body.createCompound({}, compound)!;
        const bytes = image(world, shape.id.index1 - 1);
        const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const u = (word: number) => v.getUint32(word * 4, true);
        expect(v.getBigUint64(0, true)).toBe(
            0xb11dce70fad5622bn ^ 0x93edaf889fd30b4an ^ 0xaaab9a00f1a8aaf7n ^ 0x4a4c9587de57485cn,
        );
        expect(u(2)).toBe(bytes.length);
        expect(u(3)).toBe(120);
        expect(v.getBigUint64(16, true)).toBe(0x93edaf889fd30b4an);
        expect(u(6)).toBe(0);
        expect(u(7)).toBe(compound.tree.root);
        expect(u(9)).toBe(compound.tree.nodeCapacity);
        expect(u(17)).toBe(120 + compound.tree.nodeCapacity * 48);
        expect(u(18)).toBe(compound.materialCount);
        expect(u(19)).toBe(u(17) + compound.materialCount * 40);
        expect(u(21) % 8).toBe(0);
        expect(u(24) % 8).toBe(0);
        const hullData = u(u(21) / 4 + 7);
        expect(
            new DataView(bytes.buffer, bytes.byteOffset + hullData, 8).getBigUint64(0, true),
        ).toBe(0x4a4c9587de57485cn);
        const meshInstance = u(24) / 4;
        const meshBlobOffset = u(meshInstance + 10);
        expect(
            new DataView(bytes.buffer, bytes.byteOffset + meshBlobOffset, 8).getBigUint64(0, true),
        ).toBe(0xaaab9a00f1a8aaf7n);
    } finally {
        world.destroy();
    }
});
