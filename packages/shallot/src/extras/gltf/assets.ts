import { Compute, type Plugin, type State, type System } from "../../engine";
import { readBinary } from "../../engine/runtime";
import type { Node } from "../../engine/scene";
import { Preloads } from "../../engine/scene/core";
import { Color, Part } from "../../standard/part";
import { RenderPlugin } from "../../standard/render";
import type { Binding } from "../../standard/render/core";
import {
    BeginFrameSystem,
    type Mesh,
    Meshes,
    meshBounds,
    packMeshes,
    type QuantStreams,
    quantizeMeshes,
    Surfaces,
    VERTEX_FLOATS,
} from "../../standard/render/core";
import { SlabPlugin } from "../../standard/slab";
import { Transform } from "../../standard/transforms";
import { isGlb, parseGlb } from "./glb";
import {
    type GltfImage,
    type GltfJson,
    type GltfMaterial,
    type GltfScene,
    type GltfUnsupported,
    parse,
} from "./gltf";
import { ALBEDO_NAMES } from "./image";
import { abortDecodes, poolDecode } from "./pool";
import { RouteSystem, routes, scanRefs, Textured } from "./routes";
import { mapSet, materialPreamble } from "./shade";
import {
    type AssembledVat,
    assembleVat,
    disposeVatFallback,
    fallbackVat,
    registerSkinSurfaces,
    Skin,
    SkinSystem,
    skinSurface,
} from "./skin";
import { pickTargets, type Targets, type TranscodeTarget } from "./target";
import {
    type AssembledTextures,
    type DecodedImage,
    type DecodedMap,
    type DecodedTextures,
    disposeTextureFallbacks,
    fallbackTextures,
    publishTextures,
    textureResources,
} from "./textures";
import { beginUnion, stepUnion, type UnionAsset, type UnionStaging } from "./union";
import { bakeVat, type GltfVat } from "./vat";

// #doc:dev
// `gltf/core` is the pipeline under `loadGltf`, for tooling and custom async loading. `loadGltf` is
// `ensureDecoded` (a content-keyed cache over the deviceless `decode`) followed by `register` (the GPU
// assembly that hands back the placement descriptor). Call them apart to keep decode off the hot path
// (`decodeInWorker` runs the same `decode` on the worker pool, and its `DecodedGltf` feeds `register`
// directly), or to manage the cache across a reload: `invalidate` evicts one source (the HMR / asset-swap
// seam), `clearGltfCache` drops everything, and `gltfCacheStats` reports the decode + asset counts.
// `unionPending` is true while the shared texture atlas is still uploading. It streams across frames after
// a load, so a gate waits on it before reading published textures.
//
// Textured assets share one accumulating palette: each material lands in a `texture_2d_array` bucketed by
// baseColor size, and every instance carries a palette index resolved per-primitive on the GPU, so several
// assets coexist in one draw without re-binding.

// the bindings every textured glTF surface declares: the instancing convention (eids + transforms) + the
// per-instance baseColorFactor (color) + the per-instance palette index (materialIndex) + the per-material
// palette (materialData) + the baseColor size-bucket arrays + the three data maps + emissive + one shared
// sampler. `materialData[materialIndex[eid]]` resolves the bucket + layer per primitive entirely on the GPU,
// so the pack stays one drawIndirect per (surface, mesh). The arrays cost no storage buffers (textures are a
// separate limit), so the surface stays at its 10-storage ceiling (gpu.md). The bindings are
// variant-invariant — every map-set variant binds the same arrays (an unused one is a 1×1 fallback, never
// skipped); the `specialize` codegen, not a missing binding, is what drops a sparse-map material's samples.
const texturedBindings: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    color: { type: "storage", element: "u32" },
    materialIndex: { type: "storage", element: "u32" },
    materialData: { type: "storage", element: "MaterialData" },
    // baseColor is the bandwidth lever, so it stays block-compressed; a `texture_2d_array` is one size, so
    // varied-size baseColors split across ALBEDO_BUCKETS arrays (`sampleAlbedo` switches per-material)
    ...Object.fromEntries(ALBEDO_NAMES.map((n) => [n, { type: "texture-2d-array" } as Binding])),
    mr: { type: "texture-2d-array" },
    normalTex: { type: "texture-2d-array" },
    occlusion: { type: "texture-2d-array" },
    emissive: { type: "texture-2d-array" },
    albedoSamp: { type: "sampler" },
};

// register the three alpha-mode variants of the textured surface — opaque, MASK (clip cutout → holed
// shadows), BLEND (alpha). They share the bindings + the `shadePbr` metallic-roughness path, and each
// `specialize`s per material map-set (`materialPreamble`): sear compiles one pipeline per distinct map-set a
// scene draws (keyed by the mesh's `variant`, the importer's `mapSet`), so a sparse-map material samples only
// the maps it carries — no throwaway off-L2 fetch. Only the blend mode + cutout discard differ between the
// three. Registered in GltfPlugin.initialize; a draw stays skipped until loadGltf publishes the arrays +
// `materialData` and its map-set variant compiles (both sear-lazy). `mid` is the per-instance `materialIndex[eid]`.
function registerSurfaces(): void {
    Surfaces.register({
        name: "gltf-albedo",
        bindings: texturedBindings,
        specialize: (variant) => ({ preamble: materialPreamble(variant) }),
        fs: /* wgsl */ `
        let mid = materialIndex[eid];
        let base = sampleAlbedo(mid, uv).rgb * unpackLdrColor(color[eid]).rgb;
        col = vec4<f32>(shadePbr(mid, uv, base, normalize(worldNormal), world), 1.0);`,
    });
    Surfaces.register({
        name: "gltf-albedo-clip",
        blend: "clip",
        bindings: texturedBindings,
        specialize: (variant) => ({ preamble: materialPreamble(variant) }),
        fs: /* wgsl */ `
        let mid = materialIndex[eid];
        let tex = sampleAlbedo(mid, uv);
        let c = unpackLdrColor(color[eid]);
        // shade (and its map samples) before the discard, so a killed lane never poisons a derivative
        let rgb = shadePbr(mid, uv, tex.rgb * c.rgb, normalize(worldNormal), world);
        if (tex.a * c.a < materialData[mid].cutoff) { discard; }
        col = vec4<f32>(rgb, 1.0);`,
    });
    Surfaces.register({
        name: "gltf-albedo-blend",
        blend: "alpha",
        bindings: texturedBindings,
        specialize: (variant) => ({ preamble: materialPreamble(variant) }),
        fs: /* wgsl */ `
        let mid = materialIndex[eid];
        let tex = sampleAlbedo(mid, uv) * unpackLdrColor(color[eid]);
        col = vec4<f32>(shadePbr(mid, uv, tex.rgb, normalize(worldNormal), world), tex.a);`,
    });
}

// the stable registry name for one decoded primitive — `url#index`, namespaced by the asset's url (and clip,
// since a different clip is a distinct decoded asset that re-registers the same primitives) so two assets
// never collide and a rebuild re-registers identical names into the wiped registry. Stable + predictable
// replaces the old monotonic counter: the descriptor's numeric `mesh` id is the handle, this is the readable
// name behind it (`Meshes.id(name)` resolves it).
function specName(url: string, clip: number, meshIndex: number): string {
    return clip ? `${url}@clip${clip}#${meshIndex}` : `${url}#${meshIndex}`;
}

// a local-space bounding sphere `[cx, cy, cz, r]` enclosing a VAT's all-frames AABB — the conservative
// cull bound for a skinned mesh (its rest-pose bound would clip a limb mid-swing)
function sphereOf(aabb: GltfVat["aabb"]): [number, number, number, number] {
    const cx = (aabb.min[0] + aabb.max[0]) * 0.5;
    const cy = (aabb.min[1] + aabb.max[1]) * 0.5;
    const cz = (aabb.min[2] + aabb.max[2]) * 0.5;
    const rx = (aabb.max[0] - aabb.min[0]) * 0.5;
    const ry = (aabb.max[1] - aabb.min[1]) * 0.5;
    const rz = (aabb.max[2] - aabb.min[2]) * 0.5;
    return [cx, cy, cz, Math.hypot(rx, ry, rz)];
}

// quantize each scene mesh's geometry into the GPU-ready vertex streams keyed by scene-mesh index — the
// deviceless half of geometry registration. Static meshes pack into one shared family (sear binds geometry
// once); a skinned mesh keeps its own stream so its index range stays local `[0, vertCount)` — the axis the
// VAT row is keyed by via `indices[vidx]`. bounds drive Part's cull (rest-pose AABB for static, the VAT's
// conservative all-frames sphere for skinned). Each mesh carries its material map-set as `variant`, so the
// textured/skin surface specializes its pipeline to that mesh's map-set (a mesh is one glTF primitive).
function quantizeGeometry(scene: GltfScene, vats: (GltfVat | null)[]): DecodedGeometry {
    const variantOf = (i: number) => {
        const matIdx = scene.meshes[i].material;
        return mapSet(matIdx >= 0 ? scene.materials[matIdx] : undefined);
    };
    const staticIdx = scene.meshes.flatMap((_, i) => (vats[i] ? [] : [i]));
    let statics: DecodedGeometry["static"] = null;
    if (staticIdx.length > 0) {
        const packed = packMeshes(staticIdx.map((i) => scene.meshes[i]));
        statics = {
            quant: quantizeMeshes(packed.vertices, packed.slices),
            indices: packed.indices,
            slices: packed.slices.map((s, k) => ({
                meshIndex: staticIdx[k],
                indexBase: s.indexBase,
                indexCount: s.indexCount,
                bounds: meshBounds(scene.meshes[staticIdx[k]].vertices),
                variant: variantOf(staticIdx[k]),
            })),
        };
    }
    // a skinned mesh keeps its own stream (its index range stays local, the axis the VAT row is keyed by);
    // its static stream is quantized for the shared decode preamble but unread — SKIN_VS overwrites the
    // position from the VAT textures, so the decode result is discarded (dead, harmless).
    const skinned = scene.meshes.flatMap((m, i) => {
        const vat = vats[i];
        if (!vat) return [];
        return [
            {
                meshIndex: i,
                quant: quantizeMeshes(m.vertices, [
                    { vertexBase: 0, vertexCount: m.vertices.length / VERTEX_FLOATS },
                ]),
                indices: m.indices,
                bounds: sphereOf(vat.aabb),
                variant: variantOf(i),
            },
        ];
    });
    return { static: statics, skinned };
}

// one assembled mesh: its scene-mesh index + the register-ready Mesh spec (GPU buffers baked in, name
// namespaced by load). The asset cache holds these; {@link registerGeometry} registers them per build.
interface GeometrySpec {
    meshIndex: number;
    spec: Mesh;
}

// the per-mesh binding override sear resolves for a skinned mesh's draws — its own VAT textures + params, so
// N skinned meshes coexist (the textured firehose shares its albedo arrays globally; the VAT can't, so it
// binds per-mesh via Mesh.bindings). The skin surface declares vatPos/vatNorm/vatSamp/vatParams.
function vatBindings(vat: AssembledVat): Record<string, GPUTexture | GPUSampler | GPUBuffer> {
    return { vatPos: vat.pos, vatNorm: vat.norm, vatSamp: vat.sampler, vatParams: vat.params };
}

// upload the quantized streams into GPU buffers + build the register-ready Mesh specs, keyed by scene-mesh
// index — the cache-owned half of geometry, so a rebuild re-registers without re-uploading. {@link specName}
// namespaces the spec names by `(url, clip)` so two distinct assets don't collide (a rebuild re-registers the
// same names into the wiped registry). Static meshes share one buffer family; each skinned mesh owns its own
// buffers + its VAT bound per-draw (`vats[meshIndex]`, attached as the spec's `Mesh.bindings`).
function assembleGeometry(
    device: GPUDevice,
    geometry: DecodedGeometry,
    vats: (AssembledVat | null)[],
    url: string,
    clip: number,
): GeometrySpec[] {
    const specs: GeometrySpec[] = [];
    if (geometry.static) {
        const g = geometry.static;
        const vertices = gpuBuffer(device, "gltf-main", g.quant.main);
        const position = gpuBuffer(device, "gltf-pos", g.quant.position);
        const quant = gpuBuffer(device, "gltf-quant", g.quant.quant);
        const indices = gpuBuffer(device, "gltf-indices", g.indices);
        for (const s of g.slices) {
            specs.push({
                meshIndex: s.meshIndex,
                spec: {
                    name: specName(url, clip, s.meshIndex),
                    vertices,
                    position,
                    quant,
                    indices,
                    indexBase: s.indexBase,
                    indexCount: s.indexCount,
                    bounds: s.bounds,
                    variant: s.variant,
                },
            });
        }
    }
    for (const sk of geometry.skinned) {
        const name = specName(url, clip, sk.meshIndex);
        specs.push({
            meshIndex: sk.meshIndex,
            spec: {
                name,
                vertices: gpuBuffer(device, `gltf-skin-main:${name}`, sk.quant.main),
                position: gpuBuffer(device, `gltf-skin-pos:${name}`, sk.quant.position),
                quant: gpuBuffer(device, `gltf-skin-quant:${name}`, sk.quant.quant),
                indices: gpuBuffer(device, `gltf-skin-idx:${name}`, sk.indices),
                indexBase: 0,
                indexCount: sk.indices.length,
                bounds: sk.bounds,
                variant: sk.variant,
                bindings: vatBindings(vats[sk.meshIndex] as AssembledVat),
            },
        });
    }
    return specs;
}

// register the assembled Mesh specs into the (per-build) Meshes registry, returning the scene-mesh-index →
// mesh-id map describe reads — the cheap, idempotent per-build half. The cached specs point at the cached
// buffers, so re-running on each rebuild re-registers without re-upload; the name-keyed registry reuses an id
// within a build (two instances of one asset share geometry), and the wiped registry assigns fresh ids each
// build (RenderPlugin.initialize cleared it).
function registerGeometry(specs: GeometrySpec[], meshCount: number): number[] {
    const meshIds: number[] = new Array(meshCount);
    for (const { meshIndex, spec } of specs) meshIds[meshIndex] = Meshes.register(spec);
    return meshIds;
}

function gpuBuffer(device: GPUDevice, label: string, data: Float32Array | Uint32Array): GPUBuffer {
    const buf = device.createBuffer({
        label,
        size: data.byteLength,
        // INDEX so the index buffers this helper makes drive hardware vertex reuse; harmless on the
        // vertex/skin buffers it also makes (they're never set as an index buffer)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, data as Float32Array<ArrayBuffer>);
    return buf;
}

/**
 * one registered glTF primitive, ready to point a {@link Part} at: the rich handle {@link loadGltf} hands
 * back. `mesh` is the registered {@link Meshes} id (the real reference; `name` is its readable
 * `url#index` form), `surface` the resolved {@link Surfaces} id for its route (sear's solid `default`, a
 * `gltf-albedo*` textured variant, or a `skin*` VAT variant), and `material` its index into the shared
 * union palette. {@link placeGltf} wires all of this onto a Part for you; this is the seam if you author the
 * Part yourself.
 */
export interface GltfHandle {
    name: string;
    mesh: number;
    surface: number;
    /** index into the shared union palette (`materialData`): `Textured.id` / the skin `material` lane */
    material: number;
    /** `baseColorFactor`, linear rgba */
    color: [number, number, number, number];
    /** a VAT-deforming skinned primitive: {@link placeGltf} adds {@link Skin} */
    skinned: boolean;
    /** a textured primitive: {@link placeGltf} adds {@link Textured} */
    textured: boolean;
    /** the baked clip's loop duration in seconds (skinned only, else 0) */
    duration: number;
}

/** one node placement of a {@link GltfHandle}: `handle` indexes {@link GltfImport.meshes}, the rest is the
 *  node's baked world TRS. {@link placeScene} replays them; replay them yourself to lay the asset out. */
export interface GltfPlacement {
    handle: number;
    pos: [number, number, number];
    rot: [number, number, number, number];
    scale: [number, number, number];
}

/**
 * the result of importing a glTF: its registered primitives ({@link GltfHandle}s) + their node placements
 * ({@link GltfPlacement}s). Pure data: the import registers the meshes/surfaces/VATs and hands this back; it
 * creates no entities. Point Parts at the handles (a scene `<a part="mesh: …">`, {@link placeGltf}, or
 * {@link placeScene} for the whole asset).
 */
export interface GltfImport {
    meshes: GltfHandle[];
    instances: GltfPlacement[];
}

// build the import descriptor over the already-registered mesh ids — the pure half of placement. Each scene
// primitive resolves to one {@link GltfHandle} with its route: a skinned mesh → a VAT-deforming `skin*`
// surface; a textured material → the matching `gltf-albedo*` surface; everything else → sear's solid
// `default`. `base` is the asset's palette offset in the shared union, so every handle's `material` carries
// `base + localMatId` (the instance reads `materialData[base + localMatId]` in the one accumulated palette).
function describe(
    scene: GltfScene,
    meshIds: number[],
    base: number,
    textured: boolean,
    vats: (GltfVat | null)[],
): GltfImport {
    const solid = Surfaces.id("default") ?? 0;
    const opaque = Surfaces.id("gltf-albedo");
    const clip = Surfaces.id("gltf-albedo-clip");
    const blend = Surfaces.id("gltf-albedo-blend");
    const meshes: GltfHandle[] = scene.meshes.map((m, i) => {
        const matIdx = m.material;
        const mat = matIdx >= 0 ? scene.materials[matIdx] : undefined;
        const alpha = mat?.alphaMode ?? "OPAQUE";
        const vat = vats[i];
        let surface = solid;
        let skinned = false;
        let isTextured = false;
        let duration = 0;
        if (vat) {
            surface = Surfaces.id(skinSurface(alpha)) ?? solid;
            skinned = true;
            duration = vat.duration;
        } else if (textured && mat?.image !== undefined) {
            surface = (alpha === "MASK" ? clip : alpha === "BLEND" ? blend : opaque) ?? solid;
            isTextured = true;
        }
        return {
            name: Meshes.name(meshIds[i]) ?? "",
            mesh: meshIds[i],
            surface,
            material: base + Math.max(matIdx, 0),
            color: m.color,
            skinned,
            textured: isTextured,
            duration,
        };
    });
    const instances: GltfPlacement[] = scene.instances.map((p) => ({
        handle: p.mesh,
        pos: p.pos,
        rot: p.rot,
        scale: p.scale,
    }));
    return { meshes, instances };
}

/**
 * spawn one entity rendering a glTF primitive: the standard part/transform flow for a {@link GltfHandle}
 * from a {@link loadGltf} descriptor. Creates a {@link Transform} + {@link Part} + {@link Color} wired to the
 * handle's mesh, surface, and baseColorFactor, adding {@link Skin} (skinned) or {@link Textured} (textured)
 * as the handle's route dictates. Returns the eid. Pass `pos`/`rot`/`scale` to place it; defaults are the
 * identity pose.
 *
 * @example
 * const { meshes } = await loadGltf(state, "tree.glb");
 * for (const p of grid) placeGltf(state, meshes[0], { pos: p }); // one mesh, instanced N times
 */
export function placeGltf(
    state: State,
    handle: GltfHandle,
    opts: {
        pos?: [number, number, number];
        rot?: [number, number, number, number];
        scale?: [number, number, number];
    } = {},
): number {
    const eid = state.create();
    state.add(eid, Transform);
    state.add(eid, Part);
    state.add(eid, Color);
    const [px, py, pz] = opts.pos ?? [0, 0, 0];
    const [rx, ry, rz, rw] = opts.rot ?? [0, 0, 0, 1];
    const [sx, sy, sz] = opts.scale ?? [1, 1, 1];
    Transform.pos.set(eid, px, py, pz, 0);
    Transform.rot.set(eid, rx, ry, rz, rw);
    Transform.scale.set(eid, sx, sy, sz, 0);
    Part.mesh.set(eid, handle.mesh);
    Part.surface.set(eid, handle.surface);
    Color.rgba.set(eid, handle.color[0], handle.color[1], handle.color[2], handle.color[3]);
    if (handle.skinned) {
        state.add(eid, Skin);
        // lanes: time, palette index (base + local), phase offset, clip duration (SkinSystem loops on it)
        Skin.anim.set(eid, 0, handle.material, 0, handle.duration);
    } else if (handle.textured) {
        state.add(eid, Textured);
        Textured.id.set(eid, handle.material);
    }
    return eid;
}

/**
 * spawn the whole imported asset: one {@link Part} entity per node placement, at its baked TRS, returning
 * the eids. The convenience for a multi-primitive asset or a whole environment (Sponza): `placeScene(state,
 * await loadGltf(state, url))`. The asset lands at its authored origin; reposition the returned entities, or
 * place a single mesh elsewhere with {@link placeGltf}. (No root-transform compose: the substrate is flat TRS
 * by design, a hierarchy/matrix root is the shape it deliberately omits, see the transforms contract.)
 *
 * @example
 * placeScene(state, await loadGltf(state, "sponza/Sponza.gltf"));
 */
export function placeScene(state: State, asset: GltfImport): number[] {
    return asset.instances.map((p) => placeGltf(state, asset.meshes[p.handle], p));
}

// resolve one glTF buffer entry to its bytes — the .glb BIN chunk (no uri), a base64 data-URI, or a relative
// path next to the .gltf. Per spec only buffer 0 of a .glb omits its uri to reference the BIN chunk.
async function resolveBuffer(
    buffer: { uri?: string; byteLength: number; extensions?: Record<string, unknown> },
    dir: string,
    bin?: ArrayBuffer,
): Promise<ArrayBuffer> {
    const uri = buffer.uri;
    if (!uri) {
        if (bin) return bin;
        // a meshopt fallback buffer carries no bytes of its own — every bufferView over it redirects to the
        // compressed source buffer via EXT_meshopt_compression — so a missing uri is expected; zero-fill it
        if (
            buffer.extensions?.EXT_meshopt_compression ||
            buffer.extensions?.KHR_meshopt_compression
        )
            return new ArrayBuffer(buffer.byteLength);
        throw new Error("[gltf] buffer has no uri and no .glb BIN chunk");
    }
    if (uri.startsWith("data:")) return dataUri(uri).bytes;
    return readBinary(dir + uri);
}

// split a `data:` URI into its mime + decoded bytes (shared by buffer + image resolution)
function dataUri(uri: string): { mime: string; bytes: ArrayBuffer } {
    const comma = uri.indexOf(",");
    const mime = uri.slice(5, comma).split(";")[0] || "application/octet-stream";
    const bin = atob(uri.slice(comma + 1));
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return { mime, bytes: bytes.buffer };
}

// the MIME for an image path, so createImageBitmap decodes the right codec (it sniffs too, but be explicit)
function imageMime(uri: string): string {
    const ext = uri.slice(uri.lastIndexOf(".") + 1).toLowerCase();
    return ext === "jpg" || ext === "jpeg"
        ? "image/jpeg"
        : ext === "png"
          ? "image/png"
          : ext === "webp"
            ? "image/webp"
            : "application/octet-stream";
}

// resolve one image source to a Blob — source-agnostic (an external uri, a data-uri, or an embedded
// bufferView), so .glb drops in by feeding bufferView-sourced images without touching this
async function resolveImage(
    image: GltfImage,
    json: GltfJson,
    buffers: ArrayBuffer[],
    dir: string,
): Promise<Blob> {
    if (image.uri !== undefined) {
        if (image.uri.startsWith("data:")) {
            const d = dataUri(image.uri);
            return new Blob([d.bytes], { type: d.mime });
        }
        const bytes = await readBinary(dir + image.uri);
        return new Blob([bytes], { type: imageMime(image.uri) });
    }
    if (image.bufferView !== undefined) {
        const bv = json.bufferViews![image.bufferView];
        const start = bv.byteOffset ?? 0;
        return new Blob([buffers[bv.buffer].slice(start, start + bv.byteLength)], {
            type: image.mimeType ?? "image/png",
        });
    }
    throw new Error("[gltf] image has neither uri nor bufferView");
}

// the 12-byte KTX2 file identifier («KTX 20»\r\n\x1A\n) — lets decodeAlbedo route by content, not mimeType
const KTX2_ID = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];

async function isKtx2(blob: Blob): Promise<boolean> {
    if (blob.size < 12) return false;
    const head = new Uint8Array(await blob.slice(0, 12).arrayBuffer());
    return KTX2_ID.every((b, i) => head[i] === b);
}

// decode the baseColor texture set into deviceless per-image data + the per-material image reference — the
// compression-preserving half of the importer, with bucketing deferred to the union assembly (union.ts), a
// union-level decision across the active set. PNG/JPEG decode to `ImageBitmap`s (the union resizes them into
// one shared bitmap bucket); KTX2/Basis (code-split, pulled in only on a KTX2 scene) transcodes to a device
// block format (BC7 / ETC2 / ASTC) and keeps it + its source bytes (the union re-transcodes to RGBA only on a
// size-bucket spill — the warned last resort). `ref[material]` indexes the returned images (-1 = factor-only).
async function decodeAlbedo(
    scene: GltfScene,
    json: GltfJson,
    buffers: ArrayBuffer[],
    dir: string,
    target: TranscodeTarget | undefined,
): Promise<{ images: DecodedImage[]; ref: Int32Array }> {
    const used = [
        ...new Set(scene.materials.flatMap((m) => (m.image !== undefined ? [m.image] : []))),
    ];
    const slot = new Map(used.map((img, i) => [img, i]));
    const ref = Int32Array.from(scene.materials, (m) =>
        m.image !== undefined ? slot.get(m.image)! : -1,
    );
    if (used.length === 0) return { images: [], ref };

    const blobs = await Promise.all(
        used.map((img) => resolveImage(scene.images[img], json, buffers, dir)),
    );
    // a scene carrying any KTX2 baseColor pulls the transcoder in once (KHR_texture_basisu is uniform per
    // asset in practice, but routing per image keeps a mixed asset correct). The transcode target is resolved
    // main-thread-side from the device features and threaded in — decode stays deviceless (a worker has none).
    const isKtxArr = await Promise.all(blobs.map(isKtx2));
    const ktx = isKtxArr.some(Boolean) ? await import("./basis") : null;
    if (ktx && !target) {
        throw new Error(
            "[gltf] KTX2 baseColor requires a transcode target — pass { targets } from pickTargets(device)",
        );
    }
    if (ktx) await ktx.loadBasis();
    const images = await Promise.all(
        blobs.map(async (b, i): Promise<DecodedImage> => {
            if (ktx && target && isKtxArr[i]) {
                const bytes = new Uint8Array(await b.arrayBuffer());
                return { kind: "compressed", image: ktx.transcodeKtx2(bytes, target), bytes };
            }
            return {
                kind: "bitmap",
                bitmap: await createImageBitmap(b, { premultiplyAlpha: "none" }),
            };
        }),
    );
    return { images, ref };
}

// decode one data-map slot (metallic-roughness / normal / occlusion / emissive) to deviceless per-image data
// + the per-material layer index (`-1` = the material has no image for this slot → the upload binds a 1×1
// fallback the surface discards). Images dedupe to one layer each. A KTX2 source transcodes to the role's
// block format (`target`, from pickTargets — normal → BC5, AO → BC4, mr/emissive → BC7 on a BC device),
// keeping the source bytes for the union's RGBA spill; PNG/JPEG decode to `ImageBitmap`s. A KTX2 source with
// no target throws (the deviceless contract, like decodeAlbedo) — never a silent uncompressed downgrade.
async function decodeMap(
    scene: GltfScene,
    json: GltfJson,
    buffers: ArrayBuffer[],
    dir: string,
    pick: (m: GltfMaterial) => number | undefined,
    target: TranscodeTarget | undefined,
): Promise<DecodedMap> {
    const used = [
        ...new Set(scene.materials.flatMap((m) => (pick(m) !== undefined ? [pick(m)!] : []))),
    ];
    const layerOf = new Map<number, number>();
    used.forEach((img, i) => {
        layerOf.set(img, i);
    });
    const layer = Int32Array.from(scene.materials, (m) => {
        const img = pick(m);
        return img !== undefined ? layerOf.get(img)! : -1;
    });
    if (used.length === 0) return { images: [], layer };
    const blobs = await Promise.all(
        used.map((img) => resolveImage(scene.images[img], json, buffers, dir)),
    );
    const isKtxArr = await Promise.all(blobs.map(isKtx2));
    const ktx = isKtxArr.some(Boolean) ? await import("./basis") : null;
    if (ktx && !target) {
        throw new Error(
            "[gltf] KTX2 data map requires a transcode target — pass { targets } from pickTargets(device)",
        );
    }
    if (ktx) await ktx.loadBasis();
    const images = await Promise.all(
        blobs.map(async (b, i): Promise<DecodedImage> => {
            if (ktx && target && isKtxArr[i]) {
                const bytes = new Uint8Array(await b.arrayBuffer());
                return { kind: "compressed", image: ktx.transcodeKtx2(bytes, target), bytes };
            }
            return {
                kind: "bitmap",
                bitmap: await createImageBitmap(b, { premultiplyAlpha: "none" }),
            };
        }),
    );
    return { images, layer };
}

// decode every texture slot from a decoded scene — the deviceless half of the texture path. Returns the
// per-image data + the per-material references; the bucketing + palette packing are a union-level decision
// the assembly (union.ts) makes across the active set. `textured` (any slot carries a real image) routes the
// static textured surfaces; an untextured scene stays on sear's solid default.
async function decodeTextures(
    scene: GltfScene,
    json: GltfJson,
    buffers: ArrayBuffer[],
    dir: string,
    targets: Targets | undefined,
): Promise<DecodedTextures> {
    const map = (
        pick: (m: GltfMaterial) => number | undefined,
        target: TranscodeTarget | undefined,
    ) => decodeMap(scene, json, buffers, dir, pick, target);
    const albedo = await decodeAlbedo(scene, json, buffers, dir, targets?.albedo);
    const mr = await map((m) => m.mrImage, targets?.mr);
    const normal = await map((m) => m.normalImage, targets?.normalTex);
    const occ = await map((m) => m.occImage, targets?.occlusion);
    const emis = await map((m) => m.emissiveImage, targets?.emissive);
    const textured =
        albedo.images.length > 0 ||
        mr.images.length > 0 ||
        normal.images.length > 0 ||
        occ.images.length > 0 ||
        emis.images.length > 0;
    return {
        albedo: albedo.images,
        albedoRef: albedo.ref,
        maps: { mr, normalTex: normal, occlusion: occ, emissive: emis },
        textured,
    };
}

// log each intentionally-unsupported feature once, pointing at the file — the diagnostic for a scene that
// renders incomplete (a Draco/KTX variant before its codec lands, a skinned/animated/morph asset, etc.)
function warnUnsupported(url: string, unsupported: GltfUnsupported[]): void {
    const name = url.slice(url.lastIndexOf("/") + 1);
    for (const u of unsupported) {
        const detail = u.detail ? ` (${u.detail})` : "";
        console.warn(`[gltf] ${name}: ${u.feature} not implemented${detail}`);
    }
}

// frames/second the importer subsamples a clip at — the VAT memory lever (vat.ts caps the frame count)
const VAT_FPS = 30;

// one decoded mesh slice's register-ready metadata: its cull bound, material map-set variant, and index range.
interface MeshSlice {
    meshIndex: number;
    indexBase: number;
    indexCount: number;
    bounds: [number, number, number, number];
    variant: number;
}

// the decoded geometry payload — quantized vertex streams + index data as typed arrays, no GPU buffers.
// Static meshes pack into one shared family (sear binds geometry once); each skinned mesh owns its stream.
interface DecodedGeometry {
    static: { quant: QuantStreams; indices: Uint32Array; slices: MeshSlice[] } | null;
    skinned: {
        meshIndex: number;
        quant: QuantStreams;
        indices: Uint32Array;
        bounds: [number, number, number, number];
        variant: number;
    }[];
}

/**
 * a fully decoded glTF asset: the deviceless, State-independent output of {@link decode}: quantized
 * geometry, decoded textures + palette, baked VATs, and the source scene describe reads. {@link register}
 * turns it into GPU resources + a {@link GltfImport} descriptor. The payload is the seam the asset cache keys
 * on and the worker transfers.
 */
export interface DecodedGltf {
    url: string;
    /** the animation clip baked into the VAT: part of the `(url, clip)` cache key (a different clip is a
     *  distinct decoded asset). 0 when the asset has no animation. */
    clip: number;
    scene: GltfScene;
    geometry: DecodedGeometry;
    textures: DecodedTextures;
    /** baked VAT per scene mesh (parallel to `scene.meshes`): non-null for every skinned mesh, each binding
     *  its own VAT textures per-draw, so N skinned meshes coexist in one scene. */
    vats: (GltfVat | null)[];
    textured: boolean;
}

/**
 * decode a glTF 2.0 asset to a deviceless {@link DecodedGltf}: `.gltf` (external `.bin` + data-URI buffers)
 * or `.glb` (the binary container, detected by magic). Fetches + parses (Draco), transcodes textures
 * (KTX2/Basis) to GPU-ready bitmaps/blocks, quantizes geometry, and bakes a skinned mesh's clip to a VAT,
 * producing typed-array payloads with no GPU calls and no State. The State-independent half of
 * {@link loadGltf}: the seam the asset cache keys on and the worker transfers. Pair with {@link register}.
 *
 * `targets` are the compressed formats a KTX2 baseColor + data maps transcode to (per slot), resolved from
 * the device's compression support and passed in so decode stays device-free; needed only when the asset
 * carries KTX2 textures (the untextured / PNG path ignores it).
 *
 * @example
 * const decoded = await decode("sponza/Sponza.gltf");
 * const { meshes } = await register(state, decoded);
 */
export async function decode(
    url: string,
    opts: { clip?: number; targets?: Targets } = {},
): Promise<DecodedGltf> {
    const dir = url.slice(0, url.lastIndexOf("/") + 1);
    const bytes = await readBinary(url);
    const { json, bin } = isGlb(bytes)
        ? parseGlb(bytes)
        : { json: JSON.parse(new TextDecoder().decode(bytes)) as GltfJson, bin: undefined };
    const buffers = await Promise.all((json.buffers ?? []).map((b) => resolveBuffer(b, dir, bin)));
    // the Draco codec is code-split — pull it in (and instantiate its wasm) only when a primitive needs it
    const needsDraco = (json.meshes ?? []).some((m) =>
        m.primitives.some((p) => p.extensions?.KHR_draco_mesh_compression),
    );
    let decodeDraco: Parameters<typeof parse>[2];
    if (needsDraco) {
        const draco = await import("./draco");
        await draco.loadDraco();
        decodeDraco = draco.decodeDraco;
    }
    // the meshopt codec is code-split the same way — a bufferView (not a primitive) carries the extension, so
    // an asset without one (and any non-gltf app) never loads the decoder chunk
    const needsMeshopt = (json.bufferViews ?? []).some(
        (bv) => bv.extensions?.EXT_meshopt_compression || bv.extensions?.KHR_meshopt_compression,
    );
    let decodeMeshopt: Parameters<typeof parse>[3];
    if (needsMeshopt) {
        const meshopt = await import("./meshopt");
        await meshopt.loadMeshopt();
        decodeMeshopt = meshopt.decodeMeshopt;
    }
    const scene = parse(json, buffers, decodeDraco, decodeMeshopt, opts.clip ?? 0);
    warnUnsupported(url, scene.unsupported);
    const textures = await decodeTextures(scene, json, buffers, dir, opts.targets);

    // bake every skinned mesh's clip to its own VAT — each skinned mesh binds its own VAT textures per-draw
    // (its geometry already owns its buffers), so N skinned meshes coexist in one scene. The
    // heavy per-frame skinning loop runs once per skinned mesh, off the deviceless conformance walk.
    const vats: (GltfVat | null)[] = scene.skinInputs.map((si) =>
        si ? bakeVat(si, { fps: VAT_FPS }) : null,
    );
    const geometry = quantizeGeometry(scene, vats);
    return {
        url,
        clip: opts.clip ?? 0,
        scene,
        geometry,
        textures,
        vats,
        textured: textures.textured,
    };
}

// the assembled GPU resources behind one decoded asset — the cache-owned, per-asset half: the geometry
// buffers + Mesh specs (each skinned mesh's VAT attached as its spec's `Mesh.bindings`) + the per-skinned-mesh
// VATs. The shared albedo arrays + material palette are NOT here — they accumulate across the active union
// (union.ts) into `_union`, since two assets share one set. `place` re-registers the specs per build; the
// cache frees these on invalidate.
interface AssembledGltf {
    geometry: GeometrySpec[];
    // baked VAT GPU resources per scene mesh (parallel to scene.meshes); non-null for each skinned mesh
    vats: (AssembledVat | null)[];
}

// ───────────────────────────────────────────────────────────────────────────────────────────────────────
// The content-keyed asset cache. A module-level `(src, clip)`-keyed cache
// holding the deviceless {@link DecodedGltf} + its assembled per-asset GPU resources — the audio `Registry`
// shape (module-level, survives every State rebuild, not cleared in a lifecycle hook). So an editor rebuild
// re-registers the cached decode with no re-decode + no re-upload — the spike this kills. Cache-owns /
// registry-borrows: the cache owns the GPU buffers/VATs; each build's {@link place} re-registers the cached
// Mesh specs into the wiped `Meshes` (a pointer copy). The shared textures live in `_union` (below), keyed by
// the active set so a rebuild re-publishes them with no re-upload. All freed only by {@link invalidate} /
// {@link clearGltfCache} — a rebuild never touches files, a deployed game never changes its assets, so the
// key holds forever (no LRU; device-loss recovery is engine-wide, not built here).
interface GltfAsset {
    decoded: DecodedGltf;
    // the GPU resources, assembled lazily on the first register (needs a device); null until then
    assembled: AssembledGltf | null;
    // the in-flight assemble, so concurrent registers of one asset upload once
    assembling: Promise<void> | null;
}

const _cache = new Map<string, GltfAsset>();
// in-flight decodes, so two entities sharing a src decode once (warm loads them concurrently)
const _decoding = new Map<string, Promise<GltfAsset>>();
// total decodes since process start — the decode-count counter the rebuild-reuse gate reads (it must not
// advance across a rebuild). Exposed via {@link gltfCacheStats}.
let _decodes = 0;

function assetKey(src: string, clip: number): string {
    return `${src}|${clip}`;
}

/**
 * decode a glTF asset once, caching the deviceless payload by `(src, clip)`: a repeat `(src, clip)` returns
 * the cached {@link DecodedGltf} with no re-decode (the rebuild win; {@link loadGltf} reuses it across
 * builds). Concurrent calls for one key share a single decode. {@link invalidate} drops it so the next call
 * re-decodes. `targets` (the device's per-slot compressed formats) are passed to the underlying {@link decode}
 * for KTX2 textures; they aren't part of the cache key (one process targets one device, so one set). The seam
 * the worker will populate off-thread.
 */
export async function ensureDecoded(
    src: string,
    clip = 0,
    targets?: Targets,
): Promise<DecodedGltf> {
    const k = assetKey(src, clip);
    const cached = _cache.get(k);
    if (cached) return cached.decoded;
    let pending = _decoding.get(k);
    if (!pending) {
        // the `finally` frees the in-flight slot on settle either way — a success leaves the entry in
        // `_cache`, a failure leaves the slot clear so the next load retries (never a cached rejection)
        pending = (async () => {
            try {
                const decoded = await poolDecode(src, { clip, targets });
                // the worker fetches via an absolutized url, so its decoded.url is absolute — normalize back to
                // the caller's src, the key `register` / `activate` recompute from `decoded.url` (a cache miss
                // there → a duplicate asset). A no-op on the inline path, where url already equals src.
                decoded.url = src;
                _decodes++;
                const entry: GltfAsset = { decoded, assembled: null, assembling: null };
                _cache.set(k, entry);
                return entry;
            } finally {
                _decoding.delete(k);
            }
        })();
        _decoding.set(k, pending);
    }
    return (await pending).decoded;
}

// ───────────────────────────────────────────────────────────────────────────────────────────────────────
// The active set + the union texture assembly. The shared albedo arrays + material palette
// accumulate every ACTIVE asset's layers/materials: a `texture_2d_array` is one set under global binding
// names, so two textured sources can't each publish without clobbering. `_active` is the ordered list of
// active `(src, clip)` keys (the order the palette bases follow), `_paletteBase` each key's material offset,
// `_matCount` the running total. The assembled union is memoized in `_union` keyed by the active-set order and
// survives rebuilds (module-level), so a rebuild re-accumulating the same set re-publishes the same arrays
// with no re-upload — the multi-asset generalization of the sub-stage-2 rebuild win. `_active` /
// `_paletteBase` / `_matCount` reset each build (initialize → {@link clearActive}); `_union` is freed only by
// invalidate / clearGltfCache.
const _active: string[] = [];
const _paletteBase = new Map<string, number>();
let _matCount = 0;
let _union: { key: string; textures: AssembledTextures } | null = null;
// the in-flight staged union upload (the cold path), held across frames as the carry-over and drained one byte
// budget per frame by {@link UnionBuildSystem}; null when no upload is mid-flight. The previous union (or the
// 1×1 fallback) stays bound until it completes — the atomic flip. `_stagingGen` is a monotonic build counter:
// every cold `ensureUnion` bumps it and tags its `beginUnion`, so a later begin (a second concurrent load that
// grows the active set) supersedes an earlier in-flight one — the convergence the removed serialize lock gave.
let _staging: UnionStaging | null = null;
let _stagingGen = 0;
// the in-flight begin (allocate + plan) before it hands off to `_staging` — tracked so `unionPending()` is true
// from the instant a cold `ensureUnion` kicks off, not only once the async begin resolves (else a drain that
// polls right after `loadGltf` resolves could miss the build that hasn't started its first step yet).
let _begin: Promise<unknown> | null = null;

// the per-frame time budget for the staged upload — each frame uploads layers until this many ms of main-thread
// work is spent (always ≥1 layer), so cheap compressed layers batch while one slow mip-blit layer still caps the
// frame near budget. A textured scene pops in over ~N frames; the spend stays single-digit per frame.
const STEP_BUDGET_MS = 4;

const activeKey = (): string => _active.join("\n");

// free the in-flight staging's (partial) textures behind the submit fence + drop it — a discard path (a rebuild
// resets the active set, an invalidate drops the union, or a later begin supersedes this one). The textures were
// never published, so the fence is harmless; it keeps the free uniform with the published-set path.
function freeStaging(): void {
    if (!_staging) return;
    freeBehindFence(textureResources(_staging.textures));
    _staging = null;
}

// install a completed union: memoize it, publish it over the previous binding (the atomic flip), free the
// superseded set behind the fence. Shared by the staged finish + the no-step (untextured / compressed-only)
// inline finish.
function finalizeUnion(key: string, textures: AssembledTextures): void {
    const old = _union;
    _union = { key, textures };
    publishTextures(textures);
    if (old) freeBehindFence(textureResources(old.textures));
}

// reset the per-build active set (GltfPlugin.initialize) + drop any in-flight staging (the rebuild re-begins it
// for the rebuilt set). The `_union` memo survives, so a rebuild re-accumulating the same set hits it; the bases
// re-accumulate as each asset re-places. Routes key on per-build mesh ids, so they reset with the set.
function clearActive(): void {
    _active.length = 0;
    _paletteBase.clear();
    _matCount = 0;
    routes.clear();
    _stagingGen++; // supersede any in-flight begin/staging — the rebuild re-begins for the rebuilt set
    freeStaging();
}

// assign an asset its palette base + active-set slot the first time it places this build (idempotent on the
// rebuild re-place). Append order, so an already-placed asset's base never shifts — a later asset only appends
// its materials to the union palette.
function activate(entry: GltfAsset): number {
    const k = assetKey(entry.decoded.url, entry.decoded.clip);
    let base = _paletteBase.get(k);
    if (base === undefined) {
        base = _matCount;
        _paletteBase.set(k, base);
        _active.push(k);
        _matCount += entry.decoded.scene.materials.length;
    }
    return base;
}

// begin (re)assembling the union texture set for the current active set, memoized by its order. A rebuild
// re-accumulating the same set is a synchronous pointer-republish (no re-upload). A changed/cold set kicks off a
// FRAME-STAGED build (begin allocates the textures + builds the upload step list; {@link UnionBuildSystem}
// drains it, publishing only when complete) — so this returns without waiting for the upload, the previous union
// (or the 1×1 fallback) staying bound until the staged set is ready. Not awaited: a second concurrent load that
// grows the active set re-begins for the fuller set and supersedes the earlier begin via `_stagingGen` (the
// convergence the old serialize lock gave). The superseded set is freed behind the submit fence.
function ensureUnion(): void {
    const device = Compute.device;
    if (!device || _active.length === 0) return;
    const key = activeKey();
    if (_union?.key === key) {
        // the rebuild memo hit: the assembled set is unchanged, but `build()`'s requestGPU wiped
        // `Compute.textures`/`buffers` (clear-then-rebuild), so re-point them at the surviving union — the
        // pointer-republish, no re-upload. Skipping it left the warm-published 1×1 fallback bound (black).
        publishTextures(_union.textures);
        freeStaging(); // a stale in-flight build for this key is superseded by the completed memo
        return;
    }
    const assets: UnionAsset[] = _active.map((k) => {
        const e = _cache.get(k) as GltfAsset;
        return {
            textures: e.decoded.textures,
            materials: e.decoded.scene.materials,
            base: _paletteBase.get(k) as number,
        };
    });
    const matCount = _matCount;
    const gen = ++_stagingGen;
    freeStaging(); // discard a prior in-flight staging — the active set changed, its plan is stale
    const p = beginUnion(device, assets, matCount, key, gen)
        .then((staging) => {
            if (gen !== _stagingGen) {
                // a newer ensureUnion superseded this begin mid-flight — its textures were never published
                freeBehindFence(textureResources(staging.textures));
                return;
            }
            if (staging.steps.length === 0)
                finalizeUnion(key, staging.textures); // untextured / compressed-only — no layers to stage
            else _staging = staging; // hand off to UnionBuildSystem
        })
        .finally(() => {
            if (_begin === p) _begin = null;
        });
    _begin = p;
}

/** true while the frame-staged union upload is mid-flight (the textured set isn't published yet): a load
 *  screen can hold on it, and a test drains on it before asserting the published union. Covers both the
 *  begin (allocate + plan) and the per-frame layer drain. */
export function unionPending(): boolean {
    return _staging !== null || _begin !== null;
}

// drain one byte budget of the in-flight union upload per frame, publishing the whole set the frame it completes
// (the atomic flip). Republishes fragment-stage bindings only (the albedo/data arrays + sampler + palette read
// in the color FS), so `after: [BeginFrameSystem]` suffices — no `before: [PrepassSystem]` geometry edge. The
// busy flag keeps one step batch in flight across frames; the identity + generation guards make a mid-step
// rebuild / supersede / invalidate safe (the resolved batch no-ops if its staging was swapped out).
const UnionBuildSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    annotations: { mode: "always" },
    update() {
        const staging = _staging;
        if (!staging || staging.busy) return;
        staging.busy = true;
        void stepUnion(staging, STEP_BUDGET_MS).then((done) => {
            staging.busy = false;
            if (_staging !== staging) return; // superseded / freed mid-step
            if (done) {
                _staging = null;
                finalizeUnion(staging.key, staging.textures);
            }
        });
    },
};

// the GPU resources behind one cached asset, for the deferred free — geometry buffers + VATs (the shared
// textures are the union's, freed separately). Dedupes the shared static-geometry buffers.
function assetResources(a: AssembledGltf): (GPUTexture | GPUBuffer)[] {
    const bufs = new Set<GPUBuffer>();
    for (const { spec } of a.geometry) {
        bufs.add(spec.vertices);
        if (spec.position) bufs.add(spec.position);
        if (spec.quant) bufs.add(spec.quant);
        bufs.add(spec.indices);
    }
    const res: (GPUTexture | GPUBuffer)[] = [...bufs];
    for (const vat of a.vats) if (vat) res.push(vat.pos, vat.norm, vat.params);
    return res;
}

// destroy GPU resources behind the submit fence — an in-flight frame may still bind them through sear's cached
// group; the caller (a live asset-swap) rebuilds before the next frame so the State re-registers.
function freeBehindFence(res: (GPUTexture | GPUBuffer)[]): void {
    const device = Compute.device;
    if (!device) return;
    device.queue.onSubmittedWorkDone().then(() => {
        for (const r of res) r.destroy();
    });
}

function freeAsset(entry: GltfAsset): void {
    if (entry.assembled) freeBehindFence(assetResources(entry.assembled));
}

// drop the active union + free its shared arrays/palette behind the fence (invalidate / clearGltfCache — a
// changed asset invalidates the accumulated union; the paired rebuild re-accumulates + reassembles).
function dropUnion(): void {
    clearActive();
    if (_union) {
        freeBehindFence(textureResources(_union.textures));
        _union = null;
    }
}

/**
 * drop a glTF source from the asset cache + free its GPU resources (behind the submit fence): every clip
 * variant of `src`, plus the accumulated union (it included this source). The next {@link loadGltf}
 * re-decodes + re-uploads. The push-driven invalidation seam (sub-stage 5 wires it to the editor's
 * file-watch / HMR): pair it with a State rebuild so the active State re-registers before the next frame,
 * since the freed resources may still be bound.
 */
export function invalidate(src: string): void {
    const prefix = `${src}|`;
    for (const [k, entry] of _cache) {
        if (!k.startsWith(prefix)) continue;
        freeAsset(entry);
        _cache.delete(k);
    }
    dropUnion();
}

/** drop every cached glTF asset + the union + free their GPU resources (behind the submit fence). */
export function clearGltfCache(): void {
    for (const entry of _cache.values()) freeAsset(entry);
    _cache.clear();
    dropUnion();
}

/** asset-cache stats. `decodes` is the total successful decodes since process start (must not advance
 *  across a rebuild, the rebuild-reuse gate), `assets` the live cache entries, `inflight` the in-flight
 *  decodes (a failed decode must leave it clear, so the source stays retryable). */
export function gltfCacheStats(): { decodes: number; assets: number; inflight: number } {
    return { decodes: _decodes, assets: _cache.size, inflight: _decoding.size };
}

// assemble a decoded asset's per-asset GPU resources — the VATs + the geometry buffers + Mesh specs (each
// skinned mesh's VAT attached as its spec's `Mesh.bindings`). The cache-owned half of {@link register}: run
// once per `(src, clip)`, reused across rebuilds. The shared textures assemble across the union, not here.
async function assemble(device: GPUDevice, decoded: DecodedGltf): Promise<AssembledGltf> {
    const vats = decoded.vats.map((v) => (v ? assembleVat(device, v) : null));
    const geometry = assembleGeometry(device, decoded.geometry, vats, decoded.url, decoded.clip);
    return { geometry, vats };
}

// the per-asset cache entry for a decoded payload (created on first sight; `ensureDecoded` already created it
// for the cached-decode path, so this resolves it there).
function ensureEntry(decoded: DecodedGltf): GltfAsset {
    const k = assetKey(decoded.url, decoded.clip);
    let entry = _cache.get(k);
    if (!entry) {
        entry = { decoded, assembled: null, assembling: null };
        _cache.set(k, entry);
    }
    return entry;
}

// ensure a cached asset's per-asset GPU resources are assembled (once, concurrency-deduped). A failed assemble
// stays retryable (`assembled` null, `assembling` clear) rather than caching a rejection across rebuilds.
async function ensureAssembled(device: GPUDevice, entry: GltfAsset): Promise<void> {
    if (entry.assembled) return;
    if (!entry.assembling) {
        entry.assembling = (async () => {
            try {
                entry.assembled = await assemble(device, entry.decoded);
            } finally {
                entry.assembling = null;
            }
        })();
    }
    await entry.assembling;
}

// decode through the cache, treating a dispose-time abort (the rejected pool waiter when the State tore down
// mid-decode) as a clean null, not a load failure. A real decode error on a live State still throws (→ the
// caller). The shared guard for {@link loadGltf} / {@link register}.
async function decodeGuarded(
    state: State,
    src: string,
    clip: number,
    targets?: Targets,
): Promise<DecodedGltf | null> {
    try {
        return await ensureDecoded(src, clip, targets);
    } catch (e) {
        if (state.disposed) return null;
        throw e;
    }
}

// the empty descriptor a dead-State / aborted import returns — no meshes registered. A fresh object each call
// so a caller mutating its arrays can't corrupt a shared constant.
function emptyImport(): GltfImport {
    return { meshes: [], instances: [] };
}

/**
 * register a {@link DecodedGltf}'s meshes/surfaces/VATs into the State: the device-bound half of
 * {@link loadGltf}, creating no entities. Cache-aware: the per-asset assembly (geometry buffers, VATs) is
 * memoized by `(url, clip)` and survives State rebuilds, and the shared albedo arrays + palette accumulate
 * across the active union (memoized by the active set), so a rebuild re-registers + re-publishes with no
 * re-upload (the spike this kills); the first load assembles + caches it. Returns the {@link GltfImport}
 * descriptor. Requires a built State (`Compute.device`).
 *
 * @example
 * const decoded = await decode("sponza/Sponza.gltf");
 * const { meshes } = await register(state, decoded);
 */
export async function register(state: State, decoded: DecodedGltf): Promise<GltfImport> {
    if (state.disposed) return emptyImport(); // a late decode onto a torn-down State — no-op, never throw
    const device = Compute.device;
    if (!device) throw new Error("[gltf] no GPU device — call register after build()");
    const entry = ensureEntry(decoded);
    await ensureAssembled(device, entry);
    if (state.disposed) return emptyImport();
    // re-register the cached Mesh specs into the (wiped) `Meshes` + build the descriptor at the asset's palette
    // base, then publish the accumulated union — the cheap, idempotent per-build half (no decode, no re-upload)
    const a = entry.assembled as AssembledGltf;
    const base = activate(entry);
    const meshIds = registerGeometry(a.geometry, entry.decoded.scene.meshes.length);
    const desc = describe(
        entry.decoded.scene,
        meshIds,
        base,
        entry.decoded.textured,
        entry.decoded.vats,
    );
    for (const h of desc.meshes) routes.set(h.mesh, h);
    ensureUnion(); // kicks off the frame-staged union upload; textures pop in over N frames, geometry is ready now
    return desc;
}

/**
 * fetch + decode + import a glTF 2.0 asset in one call: the one-way import utility. `.gltf` (external `.bin`
 * + data-URI buffers) or `.glb` (the binary container, detected by magic). Registers each primitive into
 * {@link Meshes} under a stable `url#index` name, accumulates its baseColor into the shared union arrays (when
 * materials carry textures + {@link GltfPlugin} is loaded), bakes each skinned mesh's animation to a vertex
 * animation texture (the `clip` option picks which), and logs any intentionally-unsupported feature (see
 * {@link GltfUnsupported}). Returns the {@link GltfImport} descriptor: the registered {@link GltfHandle}s +
 * their node placements. Creates no entities: point Parts at the handles yourself via {@link placeGltf} or
 * {@link placeScene} for the whole asset. (A scene referencing a primitive by name, `<a part="mesh: …#0">`,
 * needs no call at all: {@link GltfPlugin}'s preloader imports it before load.) Call it from your
 * plugin's `initialize` or `warm` (both awaited, the device is up, the loading screen covers it) so the
 * registered mesh names resolve when you place them. Cached by `(url, clip)`, so a repeat load (the editor's rebuild)
 * reuses the decode + GPU upload. Multiple distinct textured / skinned sources coexist in one scene. The
 * descriptor + mesh names are ready the moment this resolves, but the union's textures upload across the next
 * frames (the build is frame-budgeted to avoid a freeze), so a first-loaded textured asset renders on the 1×1
 * fallback for a few frames before the textures pop in.
 *
 * @example
 * const { meshes } = await loadGltf(state, "tree.glb"); // register; place via a scene or placeGltf
 * placeScene(state, await loadGltf(state, "sponza/Sponza.gltf")); // whole environment
 */
export async function loadGltf(
    state: State,
    url: string,
    opts: { clip?: number } = {},
): Promise<GltfImport> {
    const device = Compute.device;
    if (!device) throw new Error("[gltf] no GPU device — call loadGltf after build()");
    const decoded = await decodeGuarded(state, url, opts.clip ?? 0, pickTargets(device));
    if (!decoded) return emptyImport(); // aborted mid-decode on a torn-down State (register guards post-decode)
    return register(state, decoded);
}

// the glTF preloader: await every distinct source the scene names, so `load` resolves each mesh name.
// Concurrent — the pool decodes them in parallel; the union accumulates as each registers.
async function resolveRefs(nodes: Node[], state: State): Promise<void> {
    await Promise.all(scanRefs(nodes).map((r) => loadGltf(state, r.src, { clip: r.clip })));
}

/**
 * the glTF importer plugin: registers the textured baseColor + skinned surfaces, the per-instance
 * {@link Textured} / {@link Skin} decorations a loaded asset's primitives ride, and the declarative load:
 * a scene that names a glTF primitive (`<a part="mesh: model.glb#0">`) imports it automatically before
 * load, and the route sync gives the Part its textured/skinned surface + material. Add it alongside the
 * default plugins when importing glTF. Geometry-only imports work without it (they ride sear's solid
 * `default` surface).
 *
 * It is a one-way dependency: it populates the registries and never creates entities. Reference primitives
 * by name in a scene, or import programmatically with {@link loadGltf} and place via {@link placeGltf} /
 * {@link placeScene}.
 *
 * @example
 * ```
 * plugins: [...DEFAULT_PLUGINS, GltfPlugin]
 * // scene: <a part="mesh: sponza/Sponza.gltf#0" transform color />
 * // or programmatically, from your plugin's warm:
 * placeScene(state, await loadGltf(state, "sponza/Sponza.gltf"));
 * ```
 */
export const GltfPlugin: Plugin = {
    name: "Gltf",
    components: { Textured, Skin },
    dependencies: [RenderPlugin, SlabPlugin],
    systems: [RouteSystem, SkinSystem, UnionBuildSystem],
    traits: {
        Textured: { defaults: () => ({ id: 0 }), derived: true },
        Skin: { defaults: () => ({ anim: [0, 0, 0, 0] }), derived: true },
    },
    // surfaces (registries) + the 1×1 texture/VAT fallbacks both belong here, pre-scene: a textured surface
    // binds the fallbacks until an import publishes its real union over them, and the import runs in a project
    // plugin's `initialize` (after this one, via the dependency) so the registered mesh names resolve at scene
    // parse. Publishing the fallbacks at warm would clobber a union an `initialize`-time import already
    // published (both write the same `Compute.textures` names) — so they sit here, before any import.
    initialize() {
        registerSurfaces();
        registerSkinSurfaces();
        // the declarative-load seam: scenes naming glTF meshes import them before load resolves the names.
        // Registered here / deleted in dispose, so a disabled plugin leaves no stale resolver.
        Preloads.register({ name: "gltf", resolve: resolveRefs });
        // reset the per-build active set; the `_union` memo survives so a rebuild re-accumulating the same set
        // re-publishes its arrays with no re-upload
        clearActive();
        if (!Compute.device) return;
        fallbackTextures(Compute.device);
        fallbackVat(Compute.device);
    },
    // free only the build-scoped fallbacks — the cache-owned per-asset resources survive the rebuild (that's
    // the spike this kills); they're freed by invalidate / clearGltfCache, not the plugin lifecycle. Drop the
    // pool's queued decodes too (a scene switch abandons them); an in-flight one finishes into the cache and the
    // load guards no-op it against the dead State. dispose runs before state.dispose (engine/app), so the
    // rejected awaiters see `state.disposed === true` by the time their microtask runs.
    dispose() {
        Preloads.delete("gltf");
        abortDecodes();
        disposeTextureFallbacks();
        disposeVatFallback();
    },
};
