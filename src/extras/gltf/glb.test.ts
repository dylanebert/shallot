import { describe, expect, test } from "bun:test";
import { isGlb, parseGlb } from "./glb";
import { type GltfJson, parse } from "./gltf";

const MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

// pack a JSON document + optional BIN buffer into a .glb byte stream, following the spec's 4-byte chunk
// alignment (JSON padded with spaces, BIN with zeros). The inverse of parseGlb — the test's source of truth.
function buildGlb(json: object, bin?: ArrayBuffer): ArrayBuffer {
    const jsonRaw = new TextEncoder().encode(JSON.stringify(json));
    const jsonLen = jsonRaw.length + ((4 - (jsonRaw.length % 4)) % 4);
    const binLen = bin ? bin.byteLength + ((4 - (bin.byteLength % 4)) % 4) : 0;
    const total = 12 + 8 + jsonLen + (bin ? 8 + binLen : 0);

    const out = new ArrayBuffer(total);
    const dv = new DataView(out);
    const u8 = new Uint8Array(out);
    dv.setUint32(0, MAGIC, true);
    dv.setUint32(4, 2, true);
    dv.setUint32(8, total, true);

    dv.setUint32(12, jsonLen, true);
    dv.setUint32(16, JSON_CHUNK, true);
    u8.set(jsonRaw, 20);
    for (let i = 20 + jsonRaw.length; i < 20 + jsonLen; i++) u8[i] = 0x20; // space pad

    if (bin) {
        const o = 20 + jsonLen;
        dv.setUint32(o, binLen, true);
        dv.setUint32(o + 4, BIN_CHUNK, true);
        u8.set(new Uint8Array(bin), o + 8);
    }
    return out;
}

describe("parseGlb", () => {
    test("isGlb recognizes the magic and rejects a plain-JSON / too-short buffer", () => {
        expect(isGlb(buildGlb({ asset: { version: "2.0" } }))).toBe(true);
        expect(isGlb(new TextEncoder().encode('{"asset":{}}').buffer)).toBe(false);
        expect(isGlb(new ArrayBuffer(4))).toBe(false);
    });

    test("splits the JSON document and the BIN chunk", () => {
        const bin = new Uint8Array([1, 2, 3, 4, 5, 6]).buffer; // length 6 → padded to 8
        const { json, bin: out } = parseGlb(buildGlb({ asset: { version: "2.0" }, scene: 0 }, bin));
        expect(json.scene).toBe(0);
        // the BIN chunk includes the spec's trailing pad; the meaningful bytes round-trip
        expect(out).toBeDefined();
        expect([...new Uint8Array(out!).slice(0, 6)]).toEqual([1, 2, 3, 4, 5, 6]);
    });

    test("a JSON-only .glb (no geometry buffer) returns undefined bin", () => {
        const { json, bin } = parseGlb(buildGlb({ asset: { version: "2.0" }, nodes: [] }));
        expect(json.nodes).toEqual([]);
        expect(bin).toBeUndefined();
    });

    test("feeds parse end-to-end: a buffer-0 BIN backs the geometry", () => {
        // one tight-packed triangle: 3 positions (VEC3 f32) then 3 ushort indices, all in buffer 0 (the BIN)
        const posLen = 9 * 4;
        const bin = new ArrayBuffer(posLen + 3 * 2);
        const dv = new DataView(bin);
        [1, 2, 3, 4, 5, 6, 7, 8, 9].forEach((v, i) => {
            dv.setFloat32(i * 4, v, true);
        });
        [0, 1, 2].forEach((v, i) => {
            dv.setUint16(posLen + i * 2, v, true);
        });
        const doc: GltfJson = {
            buffers: [{ byteLength: bin.byteLength }], // no uri — resolves to the BIN chunk
            bufferViews: [
                { buffer: 0, byteOffset: 0, byteLength: posLen },
                { buffer: 0, byteOffset: posLen, byteLength: 6 },
            ],
            accessors: [
                { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
                { bufferView: 1, componentType: 5123, count: 3, type: "SCALAR" },
            ],
            meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
            nodes: [{ mesh: 0 }],
            scenes: [{ nodes: [0] }],
        };

        const { json, bin: chunk } = parseGlb(buildGlb(doc, bin));
        const { meshes } = parse(json, [chunk!]);
        expect(meshes).toHaveLength(1);
        // first vertex position survives the container round-trip
        expect([meshes[0].vertices[0], meshes[0].vertices[1], meshes[0].vertices[2]]).toEqual([
            1, 2, 3,
        ]);
        expect([...meshes[0].indices]).toEqual([0, 1, 2]);
    });

    test("rejects a bad magic and an unsupported version", () => {
        const bad = new ArrayBuffer(12);
        new DataView(bad).setUint32(0, 0xdeadbeef, true);
        expect(() => parseGlb(bad)).toThrow("bad magic");

        const v1 = buildGlb({ asset: { version: "1.0" } });
        new DataView(v1).setUint32(4, 1, true);
        expect(() => parseGlb(v1)).toThrow("version 1");
    });
});
