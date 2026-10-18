// Loaded by run.ts after the TypeGPU transform is registered.

import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { deflateSync } from "node:zlib";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createApp } from "../../src";
import { genMipmapsLayer, registerImage } from "../../src/core/rendering/image";
import { probeTexture } from "../../src/engine";
import { MeshRenderPlugin, StandardRenderingPlugin } from "../../src/standard/rendering";
import {
    boxLevel,
    capturePlane,
    detailBoundary,
    HALF,
    IMAGE_SIZE,
    LEVEL_COLOURS,
    NEAR,
    PLANE_INDICES,
    planeLod,
    planeRows,
    planeVertices,
    rowContrast,
    spawnRecedingPlane,
    uploadDiagnosticImages,
} from "../../src/standard/rendering/receding-plane.fixture";

const label = process.argv[2] ?? "current";
const out = resolve(import.meta.dir, "../../test-results/texture-minification", label);
const SIZES = [
    { width: 1280, height: 720 },
    { width: 320, height: 180 },
];
const REPEAT = { addressModeU: 1, addressModeV: 1 } as const;
const NEAREST = { magFilter: 0, minFilter: 0, mipmapFilter: 0 } as const;

function png(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): Uint8Array {
    const table = new Uint32Array(256).map((_, n) => {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        return c;
    });
    const crc = (bytes: Uint8Array) => {
        let c = 0xffffffff;
        for (const b of bytes) c = table[(c ^ b) & 0xff] ^ (c >>> 8);
        return (c ^ 0xffffffff) >>> 0;
    };
    const chunk = (type: string, data: Uint8Array) => {
        const bytes = new Uint8Array(12 + data.length);
        const view = new DataView(bytes.buffer);
        view.setUint32(0, data.length);
        bytes.set(new TextEncoder().encode(type), 4);
        bytes.set(data, 8);
        view.setUint32(8 + data.length, crc(bytes.subarray(4, 8 + data.length)));
        return bytes;
    };
    const header = new Uint8Array(13);
    const view = new DataView(header.buffer);
    view.setUint32(0, width);
    view.setUint32(4, height);
    header.set([8, 6, 0, 0, 0], 8);
    const raw = new Uint8Array((width * 4 + 1) * height);
    for (let y = 0; y < height; y++)
        raw.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), y * (width * 4 + 1) + 1);
    const parts = [
        new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
        chunk("IHDR", header),
        chunk("IDAT", deflateSync(raw)),
        chunk("IEND", new Uint8Array()),
    ];
    return Buffer.concat(parts);
}

/** The generated glTF equivalent of the plane: repeat wrap and glTF's unspecified filters. */
function generatedGltf() {
    const vertices = planeVertices();
    const count = vertices.length / 8;
    const positions = new Float32Array(count * 3);
    const normals = new Float32Array(count * 3);
    const uvs = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
        positions.set([vertices[i * 8], vertices[i * 8 + 1], vertices[i * 8 + 2]], i * 3);
        normals.set([vertices[i * 8 + 4], vertices[i * 8 + 5], vertices[i * 8 + 6]], i * 3);
        uvs.set([vertices[i * 8 + 3], vertices[i * 8 + 7]], i * 2);
    }
    const bytes = Buffer.concat([
        Buffer.from(positions.buffer),
        Buffer.from(normals.buffer),
        Buffer.from(uvs.buffer),
        Buffer.from(PLANE_INDICES.buffer),
    ]);
    const views = [positions, normals, uvs, PLANE_INDICES].reduce<
        { buffer: number; byteOffset: number; byteLength: number }[]
    >((list, array) => {
        const byteOffset = list.reduce((sum, view) => sum + view.byteLength, 0);
        list.push({ buffer: 0, byteOffset, byteLength: array.byteLength });
        return list;
    }, []);
    const json = {
        asset: { version: "2.0" },
        buffers: [{ byteLength: bytes.length }],
        bufferViews: views,
        accessors: [
            { bufferView: 0, componentType: 5126, count, type: "VEC3", min: [-HALF, 0, NEAR - 2 * HALF], max: [HALF, 0, NEAR] },
            { bufferView: 1, componentType: 5126, count, type: "VEC3" },
            { bufferView: 2, componentType: 5126, count, type: "VEC2" },
            { bufferView: 3, componentType: 5125, count: PLANE_INDICES.length, type: "SCALAR" },
        ],
        meshes: [
            {
                primitives: [
                    { attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0 },
                ],
            },
        ],
        materials: [
            {
                pbrMetallicRoughness: { baseColorTexture: { index: 0 } },
                extensions: { KHR_materials_unlit: {} },
            },
        ],
        extensionsUsed: ["KHR_materials_unlit"],
        textures: [{ source: 0, sampler: 0 }],
        samplers: [{}],
        images: [{ uri: "diagnostic.png" }],
        nodes: [{ mesh: 0 }],
        scenes: [{ nodes: [0] }],
    };
    return { json, buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length) };
}

function nearestLevel(rgba: Uint8ClampedArray, i: number): number {
    let best = 0;
    let distance = Number.POSITIVE_INFINITY;
    LEVEL_COLOURS.forEach(([r, g, b], level) => {
        const d = (rgba[i] - r) ** 2 + (rgba[i + 1] - g) ** 2 + (rgba[i + 2] - b) ** 2;
        if (d < distance) [best, distance] = [level, d];
    });
    return best;
}

await setupGlobals();
await mkdir(out, { recursive: true });
// The importer runs in its own checkout, so its pinned Shallot and TypeGPU stay out of this process.
const gltfCheckout = process.env.SHALLOT_GLTF;
let imported:
    | {
          meshes: { vertices: Float32Array }[];
          textures: { sampler: Record<string, number> }[];
          unsupported: string[];
      }
    | undefined;
if (gltfCheckout) {
    const { json, buffer } = generatedGltf();
    await writeFile(resolve(out, "plane.gltf"), JSON.stringify(json));
    await writeFile(resolve(out, "plane.bin"), new Uint8Array(buffer));
    const script = `import { parse } from "./src/gltf.ts";
const json = JSON.parse(await Bun.file(${JSON.stringify(resolve(out, "plane.gltf"))}).text());
const bin = await Bun.file(${JSON.stringify(resolve(out, "plane.bin"))}).arrayBuffer();
const scene = parse(json, [bin]);
console.log(JSON.stringify({ vertices: [...scene.meshes[0].vertices], sampler: scene.textures[0].sampler, unsupported: scene.unsupported }));`;
    const child = Bun.spawnSync(["bun", "-e", script], { cwd: gltfCheckout });
    if (child.exitCode !== 0) throw new Error(child.stderr.toString());
    const result = JSON.parse(child.stdout.toString().trim().split("\n").at(-1)!);
    imported = {
        meshes: [{ vertices: new Float32Array(result.vertices) }],
        textures: [{ sampler: result.sampler }],
        unsupported: result.unsupported,
    };
}
const ids: Record<string, number> = {};
const app = await createApp({
    defaults: false,
    plugins: [
        StandardRenderingPlugin,
        MeshRenderPlugin,
        {
            name: "RecedingPlaneImages",
            initialize(world) {
                ids.hardware = registerImage(world, new Blob());
                ids.manual = registerImage(world, new Blob(), { sampler: REPEAT });
                ids.hardwareLevels = registerImage(world, new Blob());
                ids.manualLevels = registerImage(world, new Blob(), { sampler: REPEAT });
                ids.nearest = registerImage(world, new Blob(), {
                    sampler: { ...REPEAT, ...NEAREST },
                });
                if (imported)
                    ids.imported = registerImage(world, new Blob(), {
                        sampler: imported.textures[0].sampler,
                    });
            },
        },
    ],
});
const world = app.world;
const adapter = world.gpu.adapter;
const layers = Object.keys(ids).length;
// Shallot's own generator fills each chain, as registration does; boxLevel is the reference.
const texture = uploadDiagnosticImages(world, layers, {
    levelColoured: [ids.hardwareLevels, ids.manualLevels],
    generate: (target, layer) =>
        genMipmapsLayer(world, world.gpu.device, target, layer, target.mipLevelCount, target.format),
});

// Mip generation: each generated level against the box filter of the generated level above it.
const generation: { level: number; maxByteError: number }[] = [];
let above: Uint8Array | undefined;
for (let level = 0; level < texture.mipLevelCount; level++) {
    const size = IMAGE_SIZE >> level;
    const view = await probeTexture(world, texture, {
        mipLevel: level,
        origin: [0, 0, ids.hardware],
        size: [size, size, 1],
    });
    const bytes = new Uint8Array(view.bytes);
    if (above) {
        const box = boxLevel(above, size * 2);
        let maxByteError = 0;
        for (let i = 0; i < box.length; i++)
            maxByteError = Math.max(maxByteError, Math.abs(box[i] - bytes[i]));
        generation.push({ level, maxByteError });
    }
    above = bytes;
}

const vertices = imported ? imported.meshes[0].vertices : planeVertices();
const plane = spawnRecedingPlane(world, ids.hardware, SIZES, vertices);
const report: Record<string, unknown> = {
    adapter,
    sizes: SIZES,
    mipGeneration: generation,
};
const captures: Record<string, Awaited<ReturnType<typeof capturePlane>>> = {};
for (const [name, id] of Object.entries(ids)) {
    captures[name] = await capturePlane(world, plane, id);
    for (const [index, capture] of captures[name].entries()) {
        const { width, height } = SIZES[index];
        await writeFile(resolve(out, `${name}-${width}x${height}.png`), png(capture.rgba, width, height));
    }
}
for (const [name, list] of Object.entries(captures)) {
    report[name] = list.map((capture) => {
        const { width, height } = capture;
        if (name.endsWith("Levels")) {
            // Derivative check: the level each pixel shows against the level of detail the geometry gives it.
            const table: Record<string, Record<number, number>> = {};
            for (let y = 0; y < height; y++)
                for (let x = 0; x < width; x++) {
                    const lod = planeLod(width, height, x, y);
                    if (lod === undefined) continue;
                    const bucket = (Math.round(lod * 4) / 4).toFixed(2);
                    const level = nearestLevel(capture.rgba, (y * width + x) * 4);
                    (table[bucket] ??= {})[level] = (table[bucket][level] ?? 0) + 1;
                }
            return { width, height, levelShownByGeometryLod: table };
        }
        const rows = planeRows(capture);
        return {
            width,
            height,
            boundary: detailBoundary(capture, 0, 4.5),
            rows: rows.map((y) => ({
                y,
                lod: Math.round((planeLod(width, height, width / 2, y) ?? Number.NaN) * 100) / 100,
                contrast: Math.round(rowContrast(capture, y) * 10) / 10,
            })),
        };
    });
}
if (imported) {
    report.importer = {
        sampler: imported.textures[0].sampler,
        unsupported: imported.unsupported,
        identicalToDirectRepeat: captures.imported.every((capture, index) =>
            capture.rgba.every((byte, i) => byte === captures.manual[index].rgba[i]),
        ),
    };
}
await writeFile(resolve(out, "report.json"), `${JSON.stringify(report, null, 2)}\n`);
const summary = Object.fromEntries(
    Object.entries(report).map(([key, value]) => [
        key,
        Array.isArray(value) && key !== "sizes" && key !== "mipGeneration"
            ? value.map((entry) => ({ ...entry, rows: undefined, levelShownByGeometryLod: undefined }))
            : value,
    ]),
);
console.log(JSON.stringify(summary, null, 2));
console.log(`captures: ${out}`);
app.dispose();
