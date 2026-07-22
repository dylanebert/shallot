import { describe, expect, test } from "bun:test";
import { compose, quat } from "../../engine";
import {
    computeNormals,
    decompose,
    type GltfJson,
    liveSkinnable,
    parse,
    quantizeLive,
    reachBound,
} from "./gltf";
import type { SkinInput } from "./vat";

// one packed vertex = (px py pz u)(nx ny nz v). Integer lanes are f32-exact (toBe); the fractional uv
// lanes round-trip through Float32Array so they need toBeCloseTo (coding.md float-equality discipline).
function expectVertex(v: Float32Array, base: number, exp: number[]) {
    for (let i = 0; i < 8; i++) {
        if (i === 3 || i === 7) expect(v[base + i]).toBeCloseTo(exp[i], 5);
        else expect(v[base + i]).toBe(exp[i]);
    }
}

// build a glTF buffer + JSON for one triangle, attributes interleaved at `stride` bytes per vertex
// (POSITION at 0, NORMAL at 12, TEXCOORD_0 at 24 — the Sponza layout). Indices are ushort.
function triangleFixture(stride = 32) {
    const verts = [
        { p: [1, 2, 3], n: [0, 0, 1], uv: [0.1, 0.2] },
        { p: [4, 5, 6], n: [0, 1, 0], uv: [0.3, 0.4] },
        { p: [7, 8, 9], n: [1, 0, 0], uv: [0.5, 0.6] },
    ];
    const vbLen = verts.length * stride;
    const ibLen = 3 * 2;
    const buf = new ArrayBuffer(vbLen + ibLen);
    const dv = new DataView(buf);
    verts.forEach((v, i) => {
        const o = i * stride;
        v.p.forEach((x, k) => {
            dv.setFloat32(o + k * 4, x, true);
        });
        v.n.forEach((x, k) => {
            dv.setFloat32(o + 12 + k * 4, x, true);
        });
        v.uv.forEach((x, k) => {
            dv.setFloat32(o + 24 + k * 4, x, true);
        });
    });
    [0, 1, 2].forEach((x, i) => {
        dv.setUint16(vbLen + i * 2, x, true);
    });

    const gltf: GltfJson & { buffers: unknown[] } = {
        buffers: [{ byteLength: buf.byteLength }],
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: vbLen, byteStride: stride },
            { buffer: 0, byteOffset: vbLen, byteLength: ibLen },
        ],
        accessors: [
            { bufferView: 0, byteOffset: 0, componentType: 5126, count: 3, type: "VEC3" },
            { bufferView: 0, byteOffset: 12, componentType: 5126, count: 3, type: "VEC3" },
            { bufferView: 0, byteOffset: 24, componentType: 5126, count: 3, type: "VEC2" },
            { bufferView: 1, byteOffset: 0, componentType: 5123, count: 3, type: "SCALAR" },
        ],
        materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.2, 0.4, 0.6, 1] } }],
        meshes: [
            {
                name: "tri",
                primitives: [
                    {
                        attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
                        indices: 3,
                        material: 0,
                    },
                ],
            },
        ],
        nodes: [{ name: "n", mesh: 0, translation: [10, 20, 30], scale: [2, 2, 2] }],
        scenes: [{ nodes: [0] }],
        scene: 0,
    };
    return { gltf, buffers: [buf] };
}

// a minimal 1-vertex, 1-joint, 1-clip skinned mesh — the VAT-bake input boundary (skinBakeable +
// decodeSkinInput + the joint/weight accessor readers). One buffer, non-interleaved accessors; the joint
// is node 1, rotating identity → 90°Z over a 1s clip. `interpolation` toggles the bakeable gate.
function skinnedFixture(interpolation: "LINEAR" | "STEP" | "CUBICSPLINE" = "LINEAR") {
    const buf = new ArrayBuffer(148);
    const dv = new DataView(buf);
    const f = (o: number, ...xs: number[]) =>
        xs.forEach((x, i) => {
            dv.setFloat32(o + i * 4, x, true);
        });
    f(0, 1, 0, 0); // POSITION
    f(12, 0, 0, 1); // NORMAL
    // JOINTS_0 (VEC4 ubyte) @24 = [0,0,0,0] — already zero
    f(28, 1, 0, 0, 0); // WEIGHTS_0 (VEC4 f32)
    f(44, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1); // inverseBind (MAT4 identity)
    f(108, 0, 1); // anim input times
    f(116, 0, 0, 0, 1, 0, 0, Math.SQRT1_2, Math.SQRT1_2); // anim output: identity → 90°Z

    const gltf: GltfJson & { buffers: unknown[] } = {
        buffers: [{ byteLength: 148 }],
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: 12 },
            { buffer: 0, byteOffset: 12, byteLength: 12 },
            { buffer: 0, byteOffset: 24, byteLength: 4 },
            { buffer: 0, byteOffset: 28, byteLength: 16 },
            { buffer: 0, byteOffset: 44, byteLength: 64 },
            { buffer: 0, byteOffset: 108, byteLength: 8 },
            { buffer: 0, byteOffset: 116, byteLength: 32 },
        ],
        accessors: [
            { bufferView: 0, componentType: 5126, count: 1, type: "VEC3" },
            { bufferView: 1, componentType: 5126, count: 1, type: "VEC3" },
            { bufferView: 2, componentType: 5121, count: 1, type: "VEC4" },
            { bufferView: 3, componentType: 5126, count: 1, type: "VEC4" },
            { bufferView: 4, componentType: 5126, count: 1, type: "MAT4" },
            { bufferView: 5, componentType: 5126, count: 2, type: "SCALAR" },
            { bufferView: 6, componentType: 5126, count: 2, type: "VEC4" },
        ],
        meshes: [
            { primitives: [{ attributes: { POSITION: 0, NORMAL: 1, JOINTS_0: 2, WEIGHTS_0: 3 } }] },
        ],
        skins: [{ joints: [1], inverseBindMatrices: 4 }],
        animations: [
            {
                samplers: [{ input: 5, output: 6, interpolation }],
                channels: [{ sampler: 0, target: { node: 1, path: "rotation" } }],
            },
        ],
        nodes: [{ mesh: 0, skin: 0 }, { name: "joint" }],
        scenes: [{ nodes: [0, 1] }],
        scene: 0,
    };
    return { gltf, buffers: [buf] };
}

// a glTF whose primitive OMITS NORMAL (the Khronos Fox does) — one CCW triangle in the XY plane, so the
// spec-mandated synthesized normal is +Z. Indices are ushort; POSITION is the only vertex attribute.
function normallessTriangle() {
    const buf = new ArrayBuffer(3 * 12 + 3 * 2);
    const dv = new DataView(buf);
    const pos = [0, 0, 0, 1, 0, 0, 0, 1, 0];
    pos.forEach((x, i) => {
        dv.setFloat32(i * 4, x, true);
    });
    [0, 1, 2].forEach((x, i) => {
        dv.setUint16(36 + i * 2, x, true);
    });
    const gltf: GltfJson & { buffers: unknown[] } = {
        buffers: [{ byteLength: buf.byteLength }],
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: 36 },
            { buffer: 0, byteOffset: 36, byteLength: 6 },
        ],
        accessors: [
            { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
            { bufferView: 1, componentType: 5123, count: 3, type: "SCALAR" },
        ],
        meshes: [{ name: "tri", primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
        nodes: [{ mesh: 0 }],
        scenes: [{ nodes: [0] }],
        scene: 0,
    };
    return { gltf, buffers: [buf] };
}

// the glTF-spec normal synthesis (a primitive that omits NORMAL — the Khronos Fox — MUST get computed normals;
// without it the bake/decode leaves zero normals and the mesh renders unlit/black). Regression for the
// gltf-overhaul black-Fox: it failed red here (zero normal lanes) before computeNormals was wired in.
describe("normal synthesis", () => {
    test("computeNormals: a CCW triangle in the XY plane → +Z unit normals", () => {
        const n = computeNormals(
            new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            Uint32Array.from([0, 1, 2]),
        );
        for (let v = 0; v < 3; v++) {
            expect(n[v * 3]).toBeCloseTo(0, 6);
            expect(n[v * 3 + 1]).toBeCloseTo(0, 6);
            expect(n[v * 3 + 2]).toBeCloseTo(1, 6);
        }
    });

    test("computeNormals: every vertex normal is unit length across a shared edge (smooth)", () => {
        // two triangles sharing the 1→2 edge — the shared verts accumulate both faces, then renormalize
        const pos = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
        const n = computeNormals(pos, Uint32Array.from([0, 1, 2, 1, 3, 2]));
        for (let v = 0; v < 4; v++)
            expect(Math.hypot(n[v * 3], n[v * 3 + 1], n[v * 3 + 2])).toBeCloseTo(1, 6);
    });

    test("parse synthesizes normals when a primitive omits NORMAL", () => {
        const { gltf, buffers } = normallessTriangle();
        const { meshes } = parse(gltf, buffers);
        // each packed vertex keeps its position + the synthesized +Z normal (lanes 4,5,6), not the old zero
        const exp = [
            [0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
        ];
        for (let i = 0; i < 3; i++) expectVertex(meshes[0].vertices, i * 8, exp[i]);
    });
});

// a one-triangle glTF with KHR_mesh_quantization attributes: POSITION as normalized SHORT, NORMAL as
// normalized BYTE, TEXCOORD_0 as normalized USHORT — each in its own tight bufferView (stride = element size,
// not 4 × comps). The companion codec for meshopt; tested standalone here (no meshopt) so the dequant is
// isolated. Hand-picked raw values map to clean dequantized targets (±max → ±1, half → ~0.5).
function quantizedTriangle() {
    const buf = new ArrayBuffer(46);
    const dv = new DataView(buf);
    // POSITION (Int16) @0, tight stride 6
    [32767, 0, -32767, 16384, -16384, 0, 0, 32767, 8192].forEach((x, i) => {
        dv.setInt16(i * 2, x, true);
    });
    // NORMAL (Int8) @18, tight stride 3
    [127, 0, 0, 0, 127, 0, 0, 0, -127].forEach((x, i) => {
        dv.setInt8(18 + i, x);
    });
    // TEXCOORD_0 (Uint16) @28 (27→28 padded for 2-byte align), tight stride 4
    [65535, 0, 0, 65535, 32768, 49151].forEach((x, i) => {
        dv.setUint16(28 + i * 2, x, true);
    });
    // indices (Uint16) @40
    [0, 1, 2].forEach((x, i) => {
        dv.setUint16(40 + i * 2, x, true);
    });

    const gltf: GltfJson & { buffers: unknown[] } = {
        buffers: [{ byteLength: 46 }],
        extensionsUsed: ["KHR_mesh_quantization"],
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: 18 },
            { buffer: 0, byteOffset: 18, byteLength: 9 },
            { buffer: 0, byteOffset: 28, byteLength: 12 },
            { buffer: 0, byteOffset: 40, byteLength: 6 },
        ],
        accessors: [
            { bufferView: 0, componentType: 5122, normalized: true, count: 3, type: "VEC3" },
            { bufferView: 1, componentType: 5120, normalized: true, count: 3, type: "VEC3" },
            { bufferView: 2, componentType: 5123, normalized: true, count: 3, type: "VEC2" },
            { bufferView: 3, componentType: 5123, count: 3, type: "SCALAR" },
        ],
        meshes: [
            {
                name: "qtri",
                primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3 }],
            },
        ],
        nodes: [{ mesh: 0 }],
        scenes: [{ nodes: [0] }],
        scene: 0,
    };
    return { gltf, buffers: [buf] };
}

describe("quantized accessors (KHR_mesh_quantization)", () => {
    test("readFloats dequantizes normalized BYTE/SHORT/USHORT attributes, honoring the element-size stride", () => {
        const { gltf, buffers } = quantizedTriangle();
        const { meshes, unsupported } = parse(gltf, buffers);

        // the extension is handled (dequant in readFloats), so it's not flagged
        expect(unsupported).toEqual([]);

        // packed (px py pz u)(nx ny nz v); normalized SHORT pos → [-1,1], BYTE normal → [-1,1], USHORT uv → [0,1]
        const exp = [
            [1, 0, -1, 1, 1, 0, 0, 0],
            [16384 / 32767, -16384 / 32767, 0, 0, 0, 1, 0, 1],
            [0, 1, 8192 / 32767, 32768 / 65535, 0, 0, -1, 49151 / 65535],
        ];
        for (let i = 0; i < 3; i++)
            for (let k = 0; k < 8; k++)
                expect(meshes[0].vertices[i * 8 + k]).toBeCloseTo(exp[i][k], 4);
    });

    test("a non-normalized integer position reads raw (the node scale dequantizes downstream)", () => {
        // gltfpack's position path: unnormalized SHORT integers, the dequant scale baked into the node. So
        // readFloats must return the raw integer (32767, not 1), not normalize it.
        const { gltf, buffers } = quantizedTriangle();
        gltf.accessors![0].normalized = false;
        const { meshes } = parse(gltf, buffers);
        expect(meshes[0].vertices[0]).toBe(32767); // raw, un-normalized
        expect(meshes[0].vertices[16]).toBe(0); // v2.x raw
    });
});

describe("gltf parse", () => {
    test("deinterleaves attributes into the (posU)(normalV) layout, honoring byteStride + byteOffset", () => {
        const { gltf, buffers } = triangleFixture();
        const { meshes } = parse(gltf, buffers);
        expect(meshes).toHaveLength(1);
        // vertex 0: pos (1,2,3) u 0.1 | normal (0,0,1) v 0.2
        expectVertex(meshes[0].vertices, 0, [1, 2, 3, 0.1, 0, 0, 1, 0.2]);
        // vertex 2: pos (7,8,9) u 0.5 | normal (1,0,0) v 0.6 — last record proves the stride walk
        expectVertex(meshes[0].vertices, 16, [7, 8, 9, 0.5, 1, 0, 0, 0.6]);
    });

    test("decodes the same geometry when attributes are tightly packed (no byteStride)", () => {
        // POSITION/NORMAL/TEXCOORD in their own tight bufferViews — the stride defaults to the element size
        const buf = new ArrayBuffer((3 + 3 + 2) * 3 * 4);
        const dv = new DataView(buf);
        const pos = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        const nrm = [0, 0, 1, 0, 1, 0, 1, 0, 0];
        const uv = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        pos.forEach((x, i) => {
            dv.setFloat32(i * 4, x, true);
        });
        nrm.forEach((x, i) => {
            dv.setFloat32(36 + i * 4, x, true);
        });
        uv.forEach((x, i) => {
            dv.setFloat32(72 + i * 4, x, true);
        });
        const gltf: GltfJson = {
            bufferViews: [
                { buffer: 0, byteOffset: 0, byteLength: 36 },
                { buffer: 0, byteOffset: 36, byteLength: 36 },
                { buffer: 0, byteOffset: 72, byteLength: 24 },
            ],
            accessors: [
                { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
                { bufferView: 1, componentType: 5126, count: 3, type: "VEC3" },
                { bufferView: 2, componentType: 5126, count: 3, type: "VEC2" },
            ],
            meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 } }] }],
            nodes: [{ mesh: 0 }],
            scenes: [{ nodes: [0] }],
        };
        const { meshes } = parse(gltf, [buf]);
        expectVertex(meshes[0].vertices, 0, [1, 2, 3, 0.1, 0, 0, 1, 0.2]);
        expectVertex(meshes[0].vertices, 16, [7, 8, 9, 0.5, 1, 0, 0, 0.6]);
    });

    test("widens ushort indices to u32 and reads baseColorFactor as linear rgba", () => {
        const { gltf, buffers } = triangleFixture();
        const { meshes } = parse(gltf, buffers);
        expect(meshes[0].indices).toBeInstanceOf(Uint32Array);
        expect([...meshes[0].indices]).toEqual([0, 1, 2]);
        expect(meshes[0].color).toEqual([0.2, 0.4, 0.6, 1]);
    });

    test("widens ubyte and uint indices to u32", () => {
        // one triangle, POSITION tight in bv0, indices in bv1 — non-sequential values catch a byte-width
        // misread (a ubyte stream read as ushort would shift every index)
        const cases: {
            ct: number;
            bytes: number;
            write: (dv: DataView, o: number, v: number) => void;
        }[] = [
            { ct: 5121, bytes: 1, write: (dv, o, v) => dv.setUint8(o, v) },
            { ct: 5125, bytes: 4, write: (dv, o, v) => dv.setUint32(o, v, true) },
        ];
        for (const { ct, bytes, write } of cases) {
            const idx = [2, 0, 1];
            const posLen = 9 * 4;
            const buf = new ArrayBuffer(posLen + idx.length * bytes);
            const dv = new DataView(buf);
            [1, 2, 3, 4, 5, 6, 7, 8, 9].forEach((x, i) => {
                dv.setFloat32(i * 4, x, true);
            });
            idx.forEach((v, i) => {
                write(dv, posLen + i * bytes, v);
            });
            const gltf: GltfJson = {
                bufferViews: [
                    { buffer: 0, byteOffset: 0, byteLength: posLen },
                    { buffer: 0, byteOffset: posLen, byteLength: idx.length * bytes },
                ],
                accessors: [
                    { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
                    { bufferView: 1, componentType: ct, count: 3, type: "SCALAR" },
                ],
                meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
                nodes: [{ mesh: 0 }],
                scenes: [{ nodes: [0] }],
            };
            const { meshes } = parse(gltf, [buf]);
            expect(meshes[0].indices).toBeInstanceOf(Uint32Array);
            expect([...meshes[0].indices]).toEqual([2, 0, 1]);
        }
    });

    test("defaults color to opaque white when the primitive has no material", () => {
        const { gltf, buffers } = triangleFixture();
        delete gltf.meshes![0].primitives[0].material;
        gltf.materials = undefined;
        const { meshes } = parse(gltf, buffers);
        expect(meshes[0].color).toEqual([1, 1, 1, 1]);
    });

    test("emits one instance per node carrying the node's TRS", () => {
        const { gltf, buffers } = triangleFixture();
        const { instances } = parse(gltf, buffers);
        expect(instances).toHaveLength(1);
        expect(instances[0].mesh).toBe(0);
        expect(instances[0].pos).toEqual([10, 20, 30]);
        expect(instances[0].scale[0]).toBeCloseTo(2, 5);
        expect(instances[0].rot).toEqual([0, 0, 0, 1]);
    });

    test("dedupes a mesh shared by two nodes into one geometry with two instances", () => {
        const { gltf, buffers } = triangleFixture();
        gltf.nodes!.push({ mesh: 0, translation: [-1, -2, -3] });
        gltf.scenes![0].nodes = [0, 1];
        const { meshes, instances } = parse(gltf, buffers);
        expect(meshes).toHaveLength(1);
        expect(instances).toHaveLength(2);
        expect(instances[1].pos).toEqual([-1, -2, -3]);
    });

    test("bakes a parent→child chain to the child's world transform", () => {
        const { gltf, buffers } = triangleFixture();
        // parent at +10x with no mesh; the triangle node becomes its child at +5y
        gltf.nodes = [
            { name: "parent", translation: [10, 0, 0], children: [1] },
            { name: "child", mesh: 0, translation: [0, 5, 0] },
        ];
        gltf.scenes![0].nodes = [0];
        const { instances } = parse(gltf, buffers);
        expect(instances).toHaveLength(1);
        expect(instances[0].pos[0]).toBeCloseTo(10, 5);
        expect(instances[0].pos[1]).toBeCloseTo(5, 5);
        expect(instances[0].pos[2]).toBeCloseTo(0, 5);
    });

    test("skips non-triangle primitives (mode != 4)", () => {
        const { gltf, buffers } = triangleFixture();
        gltf.meshes![0].primitives[0].mode = 1; // LINES
        const { meshes, instances } = parse(gltf, buffers);
        expect(meshes).toHaveLength(0);
        expect(instances).toHaveLength(0);
    });
});

// a two-primitive mesh whose primitives reference two materials: material 0 OPAQUE + a uri-sourced
// baseColorTexture, material 1 MASK (cutoff 0.25) + a bufferView-sourced texture — exercises per-primitive
// material mapping, alphaMode/cutoff decode, and the source-agnostic image descriptor (uri vs bufferView).
function texturedFixture() {
    const { gltf, buffers } = triangleFixture();
    gltf.images = [{ uri: "a.png" }, { bufferView: 2, mimeType: "image/png" }];
    gltf.textures = [
        { source: 0, sampler: 0 },
        { source: 1, sampler: 0 },
    ];
    gltf.samplers = [{}];
    gltf.materials = [
        {
            pbrMetallicRoughness: {
                baseColorFactor: [0.2, 0.4, 0.6, 1],
                baseColorTexture: { index: 0 },
            },
        },
        {
            alphaMode: "MASK",
            alphaCutoff: 0.25,
            pbrMetallicRoughness: { baseColorTexture: { index: 1 } },
        },
    ];
    // a second primitive on the same mesh, sharing geometry accessors but using material 1
    gltf.meshes![0].primitives.push({
        attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
        indices: 3,
        material: 1,
    });
    return { gltf, buffers };
}

describe("gltf materials", () => {
    test("decodes baseColorFactor, image ref, alphaMode, and cutoff per material", () => {
        const { gltf, buffers } = texturedFixture();
        const { materials } = parse(gltf, buffers);
        expect(materials).toHaveLength(2);
        expect(materials[0]).toEqual({
            color: [0.2, 0.4, 0.6, 1],
            image: 0,
            // no metallicRoughness given → glTF factor defaults (1, 1), no maps, no emissive
            metallic: 1,
            roughness: 1,
            normalScale: 1,
            occStrength: 1,
            emissive: [0, 0, 0],
            alphaMode: "OPAQUE",
            cutoff: 0.5,
        });
        expect(materials[1].image).toBe(1);
        expect(materials[1].alphaMode).toBe("MASK");
        expect(materials[1].cutoff).toBeCloseTo(0.25, 6);
    });

    test("decodes metallic-roughness PBR fields, texture refs, and emissive strength", () => {
        const { gltf, buffers } = texturedFixture();
        // images: 0 baseColor, 1 baseColor(mask); add 2 metallicRoughness, 3 normal, 4 occlusion, 5 emissive
        gltf.images!.push({ uri: "mr.png" }, { uri: "n.png" }, { uri: "ao.png" }, { uri: "e.png" });
        gltf.textures!.push({ source: 2 }, { source: 3 }, { source: 4 }, { source: 5 });
        gltf.materials![0] = {
            pbrMetallicRoughness: {
                baseColorFactor: [1, 1, 1, 1],
                baseColorTexture: { index: 0 },
                metallicFactor: 0,
                roughnessFactor: 0.4,
                metallicRoughnessTexture: { index: 2 },
            },
            normalTexture: { index: 3, scale: 0.8 },
            occlusionTexture: { index: 4, strength: 0.5 },
            emissiveFactor: [1, 0.5, 0],
            emissiveTexture: { index: 5 },
            // biome-ignore lint/style/useNamingConvention: glTF extension names (KHR_*) are the JSON keys
            extensions: { KHR_materials_emissive_strength: { emissiveStrength: 3 } },
        };
        const m = parse(gltf, buffers).materials[0];
        expect(m.metallic).toBe(0);
        expect(m.roughness).toBeCloseTo(0.4, 6);
        expect(m.mrImage).toBe(2);
        expect(m.normalImage).toBe(3);
        expect(m.normalScale).toBeCloseTo(0.8, 6);
        expect(m.occImage).toBe(4);
        expect(m.occStrength).toBeCloseTo(0.5, 6);
        // emissiveFactor × emissive_strength (3)
        expect(m.emissive[0]).toBeCloseTo(3, 6);
        expect(m.emissive[1]).toBeCloseTo(1.5, 6);
        expect(m.emissive[2]).toBe(0);
        expect(m.emissiveImage).toBe(5);
    });

    test("exposes image sources source-agnostically (external uri and embedded bufferView)", () => {
        const { gltf, buffers } = texturedFixture();
        const { images } = parse(gltf, buffers);
        expect(images).toEqual([{ uri: "a.png" }, { bufferView: 2, mimeType: "image/png" }]);
    });

    test("maps each primitive to its material index, one GltfMesh per primitive", () => {
        const { gltf, buffers } = texturedFixture();
        const { meshes } = parse(gltf, buffers);
        expect(meshes).toHaveLength(2);
        expect(meshes[0].material).toBe(0);
        expect(meshes[1].material).toBe(1);
    });

    test("a factor-only material has no image; a primitive with no material maps to -1", () => {
        const { gltf, buffers } = triangleFixture(); // material 0 is factor-only
        delete gltf.meshes![0].primitives[0].material;
        const { meshes, materials } = parse(gltf, buffers);
        expect(materials[0].image).toBeUndefined();
        expect(materials[0].alphaMode).toBe("OPAQUE");
        expect(meshes[0].material).toBe(-1);
    });
});

describe("gltf unsupported features", () => {
    // the set of feature keys parse flagged — what the conformance suite asserts per model
    function features(gltf: GltfJson, buffers: ArrayBuffer[]): string[] {
        return parse(gltf, buffers)
            .unsupported.map((u) => u.feature)
            .sort();
    }

    test("a fully-supported triangle flags nothing", () => {
        const { gltf, buffers } = triangleFixture();
        expect(parse(gltf, buffers).unsupported).toEqual([]);
    });

    test("flags required + used extensions the importer doesn't decode", () => {
        const { gltf, buffers } = triangleFixture();
        gltf.extensionsRequired = ["KHR_draco_mesh_compression"];
        gltf.extensionsUsed = ["KHR_draco_mesh_compression", "KHR_materials_clearcoat"];
        const u = parse(gltf, buffers).unsupported;
        expect(u.find((x) => x.feature === "KHR_draco_mesh_compression")?.detail).toBe(
            "required extension",
        );
        expect(u.find((x) => x.feature === "KHR_materials_clearcoat")?.detail).toBe("extension");
    });

    test("flags animation, skin, morph, vertex-color, second-uv, and non-triangle modes", () => {
        const { gltf, buffers } = triangleFixture();
        gltf.animations = [{ samplers: [], channels: [] }];
        gltf.skins = [{ joints: [] }];
        gltf.meshes![0].primitives[0].targets = [{}, {}];
        gltf.meshes![0].primitives[0].attributes.COLOR_0 = 0;
        gltf.meshes![0].primitives[0].attributes.TEXCOORD_1 = 2;
        gltf.meshes![0].primitives.push({
            attributes: { POSITION: 0 },
            mode: 1, // LINES
        });
        expect(features(gltf, buffers)).toEqual([
            "animation",
            "morph",
            "primitive-mode",
            "skin",
            "texcoord-1",
            "vertex-color",
        ]);
    });

    test("dedupes a feature seen across many primitives to one entry", () => {
        const { gltf, buffers } = triangleFixture();
        gltf.meshes![0].primitives[0].attributes.COLOR_0 = 0;
        gltf.meshes![0].primitives.push({
            attributes: { POSITION: 0, COLOR_0: 0 },
            indices: 3,
        });
        const colors = parse(gltf, buffers).unsupported.filter((u) => u.feature === "vertex-color");
        expect(colors).toHaveLength(1);
    });

    test("skips a Draco-compressed primitive (no fallback bufferView) without crashing", () => {
        const { gltf, buffers } = triangleFixture();
        gltf.extensionsRequired = ["KHR_draco_mesh_compression"];
        // a Draco primitive's POSITION accessor carries no bufferView — the data lives in the extension
        gltf.accessors![0] = { componentType: 5126, count: 3, type: "VEC3" };
        gltf.meshes![0].primitives[0].extensions = {
            // biome-ignore lint/style/useNamingConvention: glTF extension key
            KHR_draco_mesh_compression: { bufferView: 1, attributes: { POSITION: 0 } },
        };
        const { meshes, unsupported } = parse(gltf, buffers);
        expect(meshes).toHaveLength(0); // skipped, not crashed
        expect(unsupported.map((u) => u.feature)).toContain("KHR_draco_mesh_compression");
    });
});

describe("decompose", () => {
    test("inverts compose for a translate·rotate·scale matrix", () => {
        const q = quat(0, 90, 0); // 90° about Y
        const m = compose(10, 20, 30, q.x, q.y, q.z, q.w, 2, 3, 4);
        const d = decompose(m);
        expect(d.pos).toEqual([10, 20, 30]);
        expect(d.scale[0]).toBeCloseTo(2, 5);
        expect(d.scale[1]).toBeCloseTo(3, 5);
        expect(d.scale[2]).toBeCloseTo(4, 5);
        // a quaternion and its negation are the same rotation — compare via |dot| ≈ 1
        const dot = d.rot[0] * q.x + d.rot[1] * q.y + d.rot[2] * q.z + d.rot[3] * q.w;
        expect(Math.abs(dot)).toBeCloseTo(1, 5);
    });

    test("recovers a non-trivial rotation about an off-axis", () => {
        const q = quat(30, 45, 60);
        const m = compose(0, 0, 0, q.x, q.y, q.z, q.w, 1, 1, 1);
        const d = decompose(m);
        const dot = d.rot[0] * q.x + d.rot[1] * q.y + d.rot[2] * q.z + d.rot[3] * q.w;
        expect(Math.abs(dot)).toBeCloseTo(1, 5);
    });
});

describe("gltf skinned + animated (VAT bake input)", () => {
    test("a bakeable skinned mesh drops the skin + animation flags and yields a skin input", () => {
        const { gltf, buffers } = skinnedFixture();
        const s = parse(gltf, buffers);
        expect(s.unsupported).toEqual([]);
        const skin = s.skinInputs.find(Boolean);
        expect(skin).toBeTruthy();
        expect(skin?.joints).toEqual([1]); // JOINTS_0 indexes the skin's joint slots, not node ids
        expect(skin?.duration).toBe(1);
        expect(skin && Array.from(skin.jointIndex)).toEqual([0, 0, 0, 0]);
        expect(skin?.weights[0]).toBeCloseTo(1, 6);
        expect(skin?.channels[0].path).toBe("rotation");
        expect(skin?.channels[0].step).toBe(false);
        // a skinned instance is identity-placed (the bake is in skeleton space; placement rides the matrix)
        expect(s.instances).toHaveLength(1);
        expect(s.instances[0].pos).toEqual([0, 0, 0]);
    });

    test("STEP is bakeable; CUBICSPLINE stays flagged", () => {
        const step = skinnedFixture("STEP");
        expect(parse(step.gltf, step.buffers).unsupported).toEqual([]);
        const cubic = skinnedFixture("CUBICSPLINE");
        expect(
            parse(cubic.gltf, cubic.buffers)
                .unsupported.map((u) => u.feature)
                .sort(),
        ).toEqual(["animation", "skin"]);
    });
});

const IDENT = [...compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1)];

// a minimal SkinInput carrying only the fields the live route reads (joints, inverseBind, jointIndex,
// weights, restPos); the clip / hierarchy fields are unused stubs. `quantizeLive` / `reachBound` need nothing
// else, so a test builds a rig by hand (the vat.ts SkinInput contract, live half).
function liveInput(over: Partial<SkinInput>): SkinInput {
    return {
        nodes: [],
        roots: [],
        channels: [],
        joints: [0],
        inverseBind: new Float32Array(IDENT),
        jointIndex: new Uint16Array([0, 0, 0, 0]),
        weights: new Float32Array([1, 0, 0, 0]),
        restPos: new Float32Array([0, 0, 0]),
        restNormal: new Float32Array([0, 0, 1]),
        duration: 0,
        ...over,
    };
}

// The live joint-palette route (the runtime-posed twin of the VAT bake): the deviceless decode half — the
// eligibility gate that rescues a clip-less rig skinBakeable drops, the JOINTS_0/WEIGHTS_0 quantization, and
// the reach-bound derivation. The GPU wiring + placement land at the gym `render` `skin-live` gate (stage 6d).
describe("live joint-palette skinning", () => {
    test("a clip-less rig imports live where skinBakeable dropped it", () => {
        const { gltf, buffers } = skinnedFixture();
        delete gltf.animations;
        const scene = parse(gltf, buffers);
        expect(scene.live).toBe(true); // auto-rescued to the live route
        expect(scene.unsupported).toEqual([]); // a handled skin isn't flagged
        // the skinned mesh gets its own-stream SkinInput (the axis the JW block keys on)
        expect(scene.skinInputs.filter((si) => si !== null).length).toBe(1);
    });

    test("a bakeable rig stays VAT; the live flag forces it live; CUBICSPLINE isn't auto-rescued", () => {
        const bake = skinnedFixture();
        expect(parse(bake.gltf, bake.buffers).live).toBe(false); // VAT default
        expect(parse(bake.gltf, bake.buffers, undefined, undefined, 0, true).live).toBe(true); // forced
        const cubic = skinnedFixture("CUBICSPLINE");
        expect(parse(cubic.gltf, cubic.buffers).live).toBe(false); // its clip would be lost — stays flagged
    });

    test("liveSkinnable needs a skin + accessor-backed JOINTS_0/WEIGHTS_0", () => {
        expect(liveSkinnable(skinnedFixture().gltf)).toBe(true);
        const noWeights = skinnedFixture().gltf;
        delete noWeights.meshes![0].primitives[0].attributes.WEIGHTS_0;
        expect(liveSkinnable(noWeights)).toBe(false);
        const noSkin = skinnedFixture().gltf;
        noSkin.skins = [];
        expect(liveSkinnable(noSkin)).toBe(false);
    });

    test("quantizeLive packs joint slots (u8×4) + weights (unorm8×4 summing to 1.0)", () => {
        const input = liveInput({
            joints: [0, 1, 2, 3],
            inverseBind: new Float32Array([...IDENT, ...IDENT, ...IDENT, ...IDENT]),
            jointIndex: new Uint16Array([0, 1, 2, 3]),
            weights: new Float32Array([0.5, 0.3, 0.15, 0.05]),
        });
        const live = quantizeLive(input);
        expect(live.jointCount).toBe(4);
        expect(live.joints).toHaveLength(1); // 1 vertex
        // joint slots round-trip exact through the u8 pack
        const j = live.joints[0];
        expect([j & 0xff, (j >> 8) & 0xff, (j >> 16) & 0xff, (j >> 24) & 0xff]).toEqual([
            0, 1, 2, 3,
        ]);
        // weights unpack (unorm8) sum to 255 → 1.0 after `unpack4x8unorm`, so the surface skips a runtime
        // renorm; each lane is within the derived bound: its own rounding (≤ 0.5/255) plus, for the largest
        // lane, the residual it absorbs (≤ 3 × 0.5/255) → ≤ 2/255
        const w = live.weights[0];
        const dw = [w & 0xff, (w >> 8) & 0xff, (w >> 16) & 0xff, (w >> 24) & 0xff];
        expect(dw[0] + dw[1] + dw[2] + dw[3]).toBe(255);
        const exp = [0.5, 0.3, 0.15, 0.05];
        for (let k = 0; k < 4; k++)
            expect(Math.abs(dw[k] / 255 - exp[k])).toBeLessThanOrEqual(2 / 255);
    });

    test("quantizeLive: an unweighted vertex packs all-zero (collapses to origin, the bakeVat rule)", () => {
        const input = liveInput({ weights: new Float32Array([0, 0, 0, 0]) });
        expect(quantizeLive(input).weights[0]).toBe(0);
    });

    test("quantizeLive throws past 256 joints (the u8 joint slot)", () => {
        const input = liveInput({ joints: new Array(257).fill(0) });
        expect(() => quantizeLive(input)).toThrow(/256/);
    });

    test("reachBound: R = maxᵥ (|bindPos| + |restPos − bindPos|), centered at the origin", () => {
        // joint 0 bound at (2,0,0) (inverseBind = translate(−2,0,0), so its inverse translates to (2,0,0)),
        // joint 1 at (10,0,0); vert 0 → joint 0 at rest (3,0,0) reaches 2 + 1 = 3, vert 1 → joint 1 at rest
        // (10,0,0) reaches 10 + 0 = 10, so R = max = 10.
        const input = liveInput({
            joints: [0, 1],
            inverseBind: new Float32Array([
                ...compose(-2, 0, 0, 0, 0, 0, 1, 1, 1, 1),
                ...compose(-10, 0, 0, 0, 0, 0, 1, 1, 1, 1),
            ]),
            jointIndex: new Uint16Array([0, 0, 0, 0, 1, 0, 0, 0]),
            weights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0]),
            restPos: new Float32Array([3, 0, 0, 10, 0, 0]),
        });
        const [cx, cy, cz, r] = reachBound(input);
        expect([cx, cy, cz]).toEqual([0, 0, 0]);
        expect(r).toBeCloseTo(10, 5);
    });
});
