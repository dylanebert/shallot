import { test, expect, describe, beforeEach } from "bun:test";
import { buf, traits } from "../src";
import {
    clearRegistry,
    createColorProxy,
    createFieldProxy,
    getFieldLayout,
    registerComponent,
    type FieldLayout,
} from "../src/engine/ecs/component";
import { schema } from "../src/engine/ecs/reflection";
import { CHUNK_MASK, CHUNK_SHIFT, CHUNK_SIZE, grow, type Buf } from "../src/engine/ecs/capacity";
import { formatHex } from "../src/engine/ecs/strings";

const ScalarBuf = buf(Uint32Array, 1, 0);
const Scalar = { count: createFieldProxy(ScalarBuf, 1, 0) };

const PackedBuf = buf(Float32Array, 4, 0);
const Packed = {
    mass: createFieldProxy(PackedBuf, 4, 0),
    friction: createFieldProxy(PackedBuf, 4, 1),
    gravity: createFieldProxy(PackedBuf, 4, 2),
    group: createFieldProxy(PackedBuf, 4, 3),
};

const VecBuf = buf(Float32Array, 3, 0);
const Vec = {
    posX: createFieldProxy(VecBuf, 3, 0),
    posY: createFieldProxy(VecBuf, 3, 1),
    posZ: createFieldProxy(VecBuf, 3, 2),
};

const ColorBuf = buf(Float32Array, 4, 0);
const Tinted = {
    tint: createColorProxy(ColorBuf, 4, 0),
    tintR: createFieldProxy(ColorBuf, 4, 0),
    tintG: createFieldProxy(ColorBuf, 4, 1),
    tintB: createFieldProxy(ColorBuf, 4, 2),
    opacity: createFieldProxy(ColorBuf, 4, 3),
};
traits(Tinted, { format: { tint: formatHex } });

const ShapeEnum = { Box: 0, Sphere: 1, Capsule: 2 };
const EnumBuf = buf(Uint8Array, 1, 0);
const Shaped = { shape: createFieldProxy(EnumBuf, 1, 0) };
traits(Shaped, { enums: { shape: ShapeEnum } });

const Tagged = { name: {} as Record<number, string> };

function readSlot(ref: Buf, layout: FieldLayout, eid: number): number {
    return ref.chunks[eid >>> CHUNK_SHIFT][(eid & CHUNK_MASK) * layout.stride + layout.offset];
}

function writeSlot(ref: Buf, layout: FieldLayout, eid: number, value: number): void {
    ref.chunks[eid >>> CHUNK_SHIFT][(eid & CHUNK_MASK) * layout.stride + layout.offset] = value;
}

describe("reflection layout", () => {
    beforeEach(() => {
        clearRegistry();
        registerComponent("Scalar", Scalar);
        registerComponent("Packed", Packed);
        registerComponent("Vec", Vec);
        registerComponent("Tinted", Tinted);
        registerComponent("Shaped", Shaped);
        registerComponent("Tagged", Tagged);
    });

    describe("layout shape", () => {
        test("scalar field exposes a single layout matching its proxy args", () => {
            const s = schema("Scalar")!;
            const f = s.fields[0];
            expect(f.kind).toBe("float");
            expect(f.layout).toEqual({
                bufId: ScalarBuf.id,
                array: "u32",
                stride: 1,
                offset: 0,
            });
        });

        test("packed component preserves stride and per-field offsets", () => {
            const s = schema("Packed")!;
            const named = Object.fromEntries(s.fields.map((f) => [f.name, f]));
            for (const [name, offset] of [
                ["mass", 0],
                ["friction", 1],
                ["gravity", 2],
                ["group", 3],
            ] as const) {
                const layout = named[name].layout as FieldLayout;
                expect(layout).toEqual({
                    bufId: PackedBuf.id,
                    array: "f32",
                    stride: 4,
                    offset,
                });
            }
        });

        test("vec3 emits one layout per axis, in order", () => {
            const s = schema("Vec")!;
            const f = s.fields.find((x) => x.name === "pos")!;
            expect(f.kind).toBe("vec3");
            expect(f.fields).toEqual(["posX", "posY", "posZ"]);
            const layouts = f.layout as FieldLayout[];
            expect(Array.isArray(layouts)).toBe(true);
            expect(layouts.map((l) => l.offset)).toEqual([0, 1, 2]);
            for (const l of layouts) {
                expect(l.bufId).toBe(VecBuf.id);
                expect(l.array).toBe("f32");
                expect(l.stride).toBe(3);
            }
        });

        test("color field exposes a single layout pointing at the rgb block start", () => {
            const s = schema("Tinted")!;
            const f = s.fields.find((x) => x.name === "tint")!;
            expect(f.kind).toBe("color");
            expect(f.layout).toEqual({
                bufId: ColorBuf.id,
                array: "f32",
                stride: 4,
                offset: 0,
            });
        });

        test("enum field carries layout alongside its option map", () => {
            const s = schema("Shaped")!;
            const f = s.fields[0];
            expect(f.kind).toBe("enum");
            expect(f.options).toEqual(ShapeEnum);
            expect(f.layout).toEqual({
                bufId: EnumBuf.id,
                array: "u8",
                stride: 1,
                offset: 0,
            });
        });

        test("string fields carry no layout", () => {
            const s = schema("Tagged")!;
            const f = s.fields.find((x) => x.name === "name")!;
            expect(f.kind).toBe("string");
            expect(f.layout).toBeUndefined();
        });
    });

    describe("memory identity", () => {
        test("fields backed by the same buf share bufId", () => {
            const s = schema("Packed")!;
            const ids = s.fields.map((f) => (f.layout as FieldLayout).bufId);
            expect(new Set(ids).size).toBe(1);
            expect(ids[0]).toBe(PackedBuf.id);
        });

        test("fields backed by different bufs have distinct bufIds", () => {
            const scalar = schema("Scalar")!.fields[0].layout as FieldLayout;
            const packed = schema("Packed")!.fields[0].layout as FieldLayout;
            expect(scalar.bufId).not.toBe(packed.bufId);
        });

        test("bufId is stable across schema() calls", () => {
            const a = (schema("Packed")!.fields[0].layout as FieldLayout).bufId;
            const b = (schema("Packed")!.fields[0].layout as FieldLayout).bufId;
            expect(a).toBe(b);
        });
    });

    describe("round-trip: layout descriptor reproduces proxy behavior", () => {
        test("write through proxy → read through descriptor matches", () => {
            grow(CHUNK_SIZE * 2);
            const layout = schema("Packed")!.fields.find((f) => f.name === "mass")!
                .layout as FieldLayout;
            const eids = [0, 1, 17, CHUNK_SIZE - 1, CHUNK_SIZE, CHUNK_SIZE + 5, CHUNK_SIZE * 2 - 1];
            for (const eid of eids) {
                const value = Math.fround(Math.sin(eid + 1) * 1000);
                Packed.mass.set(eid, value);
                expect(readSlot(PackedBuf, layout, eid)).toBe(value);
            }
        });

        test("write through descriptor → read through proxy matches", () => {
            grow(CHUNK_SIZE * 2);
            const layout = schema("Packed")!.fields.find((f) => f.name === "friction")!
                .layout as FieldLayout;
            const eids = [3, 42, CHUNK_SIZE - 1, CHUNK_SIZE, CHUNK_SIZE * 2 - 1];
            for (const eid of eids) {
                const value = Math.fround(eid * 0.0123);
                writeSlot(PackedBuf, layout, eid, value);
                expect(Packed.friction.get(eid)).toBe(value);
            }
        });

        test("vec3 layouts address the correct slot per axis", () => {
            grow(CHUNK_SIZE * 2);
            const layouts = schema("Vec")!.fields.find((f) => f.name === "pos")!
                .layout as FieldLayout[];
            const proxies = [Vec.posX, Vec.posY, Vec.posZ];
            const eid = CHUNK_SIZE + 11;
            for (let axis = 0; axis < 3; axis++) {
                const v = (axis + 1) * 7.25;
                proxies[axis].set(eid, v);
                expect(readSlot(VecBuf, layouts[axis], eid)).toBe(Math.fround(v));
            }
        });

        test("packed fields written via descriptor don't disturb sibling offsets", () => {
            const fields = schema("Packed")!.fields;
            const massL = fields.find((f) => f.name === "mass")!.layout as FieldLayout;
            const groupL = fields.find((f) => f.name === "group")!.layout as FieldLayout;
            const eid = 7;
            Packed.mass.set(eid, 0);
            Packed.friction.set(eid, 0);
            Packed.gravity.set(eid, 0);
            Packed.group.set(eid, 0);
            writeSlot(PackedBuf, massL, eid, 9.5);
            writeSlot(PackedBuf, groupL, eid, 3);
            expect(Packed.mass.get(eid)).toBe(9.5);
            expect(Packed.friction.get(eid)).toBe(0);
            expect(Packed.gravity.get(eid)).toBe(0);
            expect(Packed.group.get(eid)).toBe(3);
        });
    });

    describe("getFieldLayout direct lookup", () => {
        test("returns the same descriptor as schema for the matching proxy", () => {
            const fromSchema = schema("Packed")!.fields.find((f) => f.name === "mass")!
                .layout as FieldLayout;
            const direct = getFieldLayout(Packed.mass)!;
            expect(direct).toEqual(fromSchema);
        });

        test("color proxy is registered with a layout", () => {
            expect(getFieldLayout(Tinted.tint)).toBeDefined();
        });
    });
});
