import { expect, test } from "bun:test";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { createGrid } from "../shapes/heightfield";
import { hash64NonZero } from "../shapes/hullbytes";
import { createGridMesh } from "../shapes/mesh";
import { kernel } from "./kernel";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./shapecolumns";

function image(world: PhysicsWorld, shape: number): Uint8Array {
    const store = world.state.shapeStore;
    store.refreshViews();
    const u = store.materialU;
    const pool = u[(kernel(world.state.ecsState).geoLayoutPtr() >>> 2) + 6] >>> 2;
    const record = pool + store.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE];
    return new Uint8Array(u.buffer, record * 4, u[record + 4]);
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
