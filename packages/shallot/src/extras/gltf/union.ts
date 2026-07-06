import { allocArray, commonSize, mipLevels, uploadLayer } from "../../standard/render/core";
import type { Ktx2Image } from "./basis";
import type { GltfMaterial } from "./gltf";
import {
    ALBEDO_BUCKETS,
    albedoSampler,
    allocCompressed,
    fallback1x1,
    writeCompressedLayer,
} from "./image";
import { MATERIAL_STRIDE, packMaterials, type SlotLayers } from "./palette";
import {
    type AssembledTextures,
    DATA_FORMAT,
    DATA_NAMES,
    type DataSlots,
    type DecodedImage,
    type DecodedTextures,
} from "./textures";

// Assembling the GPU texture resources from the UNION of active glTF assets. A `texture_2d_array` is one size + format and WebGPU has no bindless, so two distinct
// textured sources can't each publish arrays under the global binding names without clobbering — the firehose
// answer is ONE shared set of size-bucketed albedo arrays + ONE material palette accumulating every active
// asset's layers/materials, with the per-instance index carrying the asset's palette base + local material id
// (flat draws preserved — render.md "glTF texture binding"). This file owns that union: the pure plan (bucket
// assignment + per-material layer rebase, testable without a device) + the device assembly. The caller
// (index.ts) memoizes the result per active-set, so a State rebuild re-publishes the same union with no
// re-upload.

/** one albedo image's kind + size — the bucketing input, decoupled from the decoded payload so the policy
 *  is unit-tested without a device. `size` is the `"WxH"` key compressed images group by. */
export interface AlbedoDesc {
    kind: "bitmap" | "compressed";
    size: string;
}

/** where one albedo image lands: its array bucket + layer within it, and whether it spilled (a compressed
 *  image the cap couldn't keep a dedicated array for → decoded to RGBA into the shared bitmap bucket). */
export interface AlbedoLoc {
    bucket: number;
    layer: number;
    spilled: boolean;
}

export interface AlbedoPlan {
    loc: AlbedoLoc[];
    // per used bucket, its kind — the assembler uploads a "bitmap" bucket via arrayFromBitmaps (rgba8,
    // resized to a common size) and a "compressed" bucket via compressedAlbedoArray (block format, native size)
    buckets: ("bitmap" | "compressed")[];
}

/**
 * bucket the union of active assets' albedo images into ≤ `cap` size-bucketed arrays. PNG/JPEG images all
 * share ONE bitmap bucket (resized to a common size on upload, so any size collapses there); compressed
 * (KTX2) images can't resize, so they group by size, one bucket per distinct size. When the distinct-bucket
 * count exceeds `cap`, the rarest compressed sizes spill — decoded to RGBA into the shared bitmap bucket (the
 * warned last resort). Pure — the layer assignment follows image order, so the assembler uploads in the same
 * order.
 */
export function planAlbedoBuckets(images: AlbedoDesc[], cap = ALBEDO_BUCKETS): AlbedoPlan {
    const hasBitmap = images.some((i) => i.kind === "bitmap");
    // distinct compressed sizes, ranked by population then key (deterministic)
    const pop = new Map<string, number>();
    for (const i of images)
        if (i.kind === "compressed") pop.set(i.size, (pop.get(i.size) ?? 0) + 1);
    const sizes = [...pop.keys()].sort((a, b) => pop.get(b)! - pop.get(a)! || (a < b ? 1 : -1));

    // no-spill budget: compressed buckets get all of cap minus the bitmap bucket (when any bitmaps exist)
    const noSpillBudget = cap - (hasBitmap ? 1 : 0);
    const spill = sizes.length > noSpillBudget;
    // on spill a bitmap bucket is reserved for the spilled RGBA, dropping the compressed budget to cap-1
    const compressedBudget = spill ? cap - 1 : noSpillBudget;
    const bitmapBucket = hasBitmap || spill ? 0 : -1;
    const firstCompressed = bitmapBucket >= 0 ? 1 : 0;
    // the top `compressedBudget` sizes keep a dedicated array; the rest spill
    const sizeBucket = new Map<string, number>();
    sizes.forEach((s, r) => {
        if (r < compressedBudget) sizeBucket.set(s, firstCompressed + r);
    });

    const buckets: ("bitmap" | "compressed")[] = [];
    if (bitmapBucket >= 0) buckets[0] = "bitmap";
    for (const b of sizeBucket.values()) buckets[b] = "compressed";

    // each image's (bucket, layer) — a per-bucket running fill gives the layer, in image order
    const fill: number[] = [];
    const loc: AlbedoLoc[] = images.map((img) => {
        let bucket: number;
        let spilled = false;
        if (img.kind === "bitmap") {
            bucket = bitmapBucket;
        } else {
            const dedicated = sizeBucket.get(img.size);
            if (dedicated !== undefined) {
                bucket = dedicated;
            } else {
                bucket = bitmapBucket; // spilled — decoded to RGBA into the shared bitmap bucket on upload
                spilled = true;
            }
        }
        const layer = fill[bucket] ?? 0;
        fill[bucket] = layer + 1;
        return { bucket, layer, spilled };
    });
    return { loc, buckets };
}

/** an active asset's contribution to the union — its decoded (deviceless) textures, the source materials
 *  the palette's non-layer fields read, and the palette base (material offset) the caller assigned in
 *  active-set order. */
export interface UnionAsset {
    textures: DecodedTextures;
    materials: GltfMaterial[];
    base: number;
}

/** the pure union plan: the albedo bucket assignment over the flattened active albedo, plus the concatenated
 *  material list + the per-material layer indices rebased to the shared arrays. The device assembly uploads
 *  from this; the rebase is the unit-tested logic. */
export interface UnionPlan {
    albedo: AlbedoPlan;
    materials: GltfMaterial[];
    layers: SlotLayers;
}

const sizeOf = (img: DecodedImage): string =>
    img.kind === "bitmap"
        ? `${img.bitmap.width}x${img.bitmap.height}`
        : `${img.image.width}x${img.image.height}`;

const descOf = (img: DecodedImage): AlbedoDesc => ({ kind: img.kind, size: sizeOf(img) });

// a data slot's compressed layers when every layer is a uniform-size block (so they share one compressed
// array — no transcode, no mip blit), else null → the RGBA fallback path. Format is uniform by construction
// (one role target per device), but checking it keeps a cross-device cache honest.
export function uniformBlocks(imgs: DecodedImage[]): Ktx2Image[] | null {
    if (!imgs.every((i) => i.kind === "compressed")) return null;
    const blocks = imgs.map((i) => (i as Extract<DecodedImage, { kind: "compressed" }>).image);
    const ref = blocks[0];
    const uniform = blocks.every(
        (b) => b.width === ref.width && b.height === ref.height && b.format === ref.format,
    );
    return uniform ? blocks : null;
}

/**
 * the pure plan for the union of active assets — buckets the flattened albedo and packs every material's
 * layer indices, rebased into the shared arrays (albedo bucket+layer from {@link planAlbedoBuckets}, data
 * layers concatenated per slot in asset order). Each asset's materials sit at `asset.base`; a factor-only
 * material keeps `-1` layers. Pure, so the rebase is tested without a device.
 */
export function planUnion(assets: UnionAsset[], matCount: number): UnionPlan {
    // flatten active albedo in asset order (the assembler uploads in the same order, so loc indices align)
    const flat: AlbedoDesc[] = [];
    const albedoBase: number[] = [];
    for (const a of assets) {
        albedoBase.push(flat.length);
        for (const img of a.textures.albedo) flat.push(descOf(img));
    }
    const albedo = planAlbedoBuckets(flat);

    // per-slot per-asset concatenation offset — the union layer for asset ai's local layer L is L + mapBase
    const mapBase: Record<string, number[]> = {};
    for (const name of DATA_NAMES) {
        const base: number[] = [];
        let n = 0;
        for (const a of assets) {
            base.push(n);
            n += a.textures.maps[name].images.length;
        }
        mapBase[name] = base;
    }

    const materials: GltfMaterial[] = new Array(matCount);
    const layers: SlotLayers = {
        albedo: new Int32Array(matCount).fill(-1),
        mr: new Int32Array(matCount),
        normal: new Int32Array(matCount),
        occ: new Int32Array(matCount),
        emis: new Int32Array(matCount),
        albedoBucket: new Uint32Array(matCount),
    };
    assets.forEach((a, ai) => {
        a.materials.forEach((m, mi) => {
            const u = a.base + mi;
            materials[u] = m;
            const ar = a.textures.albedoRef[mi];
            if (ar >= 0) {
                const loc = albedo.loc[albedoBase[ai] + ar];
                layers.albedo[u] = loc.layer;
                layers.albedoBucket[u] = loc.bucket;
            }
            const slot = (name: (typeof DATA_NAMES)[number], out: Int32Array) => {
                const l = a.textures.maps[name].layer[mi];
                out[u] = l >= 0 ? l + mapBase[name][ai] : -1;
            };
            slot("mr", layers.mr);
            slot("normalTex", layers.normal);
            slot("occlusion", layers.occ);
            slot("emissive", layers.emis);
        });
    });
    return { albedo, materials, layers };
}

const ALBEDO_SRGB: GPUTextureFormat = "rgba8unorm-srgb";

/** one array layer's upload — a thunk filling exactly one layer (a bitmap resize + copy + mip blit, or a
 *  compressed block write). The whole-layer unit the staged builder runs N of per frame within a time budget;
 *  never split a layer (partial uploads aren't done — Bevy's whole-asset rule). */
type UploadStep = () => Promise<void>;

/** the in-flight union upload — every GPUTexture allocated up front (so the set publishes atomically the
 *  instant the last layer lands), its layers filled across frames by draining `steps`. `key` is the active-set
 *  it was begun for, `gen` the build generation; the caller (index.ts) installs it only if both still match
 *  (a later begin supersedes it). Held in module scope as the carry-over across frames. */
export interface UnionStaging {
    key: string;
    gen: number;
    textures: AssembledTextures;
    steps: UploadStep[];
    cursor: number;
    busy?: boolean;
}

/**
 * allocate the union's GPU textures + palette and build the flat per-layer upload step list — the cheap prep
 * half of the union assembly ({@link planUnion} does the bucketing + rebase). Returns a {@link UnionStaging}
 * the caller drains with {@link stepUnion} across frames; the textures publish only once every step has run
 * (the atomic flip). The only begin-time cost is the plan, the palette pack/write, and the rare spill
 * transcode — every layer copy + mip blit is deferred into a step, so begin doesn't freeze a frame.
 */
export async function beginUnion(
    device: GPUDevice,
    assets: UnionAsset[],
    matCount: number,
    key: string,
    gen: number,
): Promise<UnionStaging> {
    const plan = planUnion(assets, matCount);
    const flatImages = assets.flatMap((a) => a.textures.albedo);

    // a spilled compressed image needs the RGBA transcoder; load it once, only if a spill occurred
    let transcodeRgba:
        | ((bytes: Uint8Array) => { width: number; height: number; rgba: Uint8Array })
        | null = null;
    const transcoder = async () => {
        if (!transcodeRgba) {
            const basis = await import("./basis");
            await basis.loadBasis();
            transcodeRgba = basis.transcodeKtx2Rgba;
        }
        return transcodeRgba;
    };

    // group images per bucket (in plan layer order) → one array per bucket. A bitmap bucket collects
    // ImageBitmaps (real bitmaps + spilled-compressed transcoded to RGBA); a compressed bucket collects blocks
    const bitmapLayers: ImageBitmap[][] = [];
    const blockLayers: Ktx2Image[][] = [];
    for (let i = 0; i < flatImages.length; i++) {
        const img = flatImages[i];
        const { bucket, layer } = plan.albedo.loc[i];
        if (plan.albedo.buckets[bucket] === "bitmap") {
            (bitmapLayers[bucket] ??= [])[layer] =
                img.kind === "bitmap"
                    ? img.bitmap
                    : await spillToBitmap(img.bytes, await transcoder());
        } else if (img.kind === "compressed") {
            (blockLayers[bucket] ??= [])[layer] = img.image;
        }
    }

    const steps: UploadStep[] = [];
    // allocate each used bucket's array + push one upload step per layer; a bitmap bucket resizes + blits per
    // layer (the step's cost), a compressed bucket writes its block mips (cheap, but still a step so no single
    // frame uploads the whole set)
    const real: (GPUTexture | undefined)[] = [];
    for (let b = 0; b < plan.albedo.buckets.length; b++) {
        if (plan.albedo.buckets[b] === "bitmap") {
            const layers = bitmapLayers[b] ?? [];
            const size = commonSize(layers.map((bm) => ({ w: bm.width, h: bm.height })));
            const levels = mipLevels(size);
            const tex = allocArray(device, size, layers.length, levels, ALBEDO_SRGB);
            real[b] = tex;
            layers.forEach((bm, layer) => {
                steps.push(() => uploadLayer(device, tex, bm, layer, size, levels, ALBEDO_SRGB));
            });
        } else if (plan.albedo.buckets[b] === "compressed") {
            const layers = blockLayers[b] ?? [];
            const tex = allocCompressed(device, layers);
            real[b] = tex;
            layers.forEach((img, layer) => {
                steps.push(async () => writeCompressedLayer(device, tex, img, layer));
            });
        }
    }
    const albedo = Array.from(
        { length: ALBEDO_BUCKETS },
        (_, b) => real[b] ?? fallback1x1(device, ALBEDO_SRGB),
    );

    // data maps: concat every asset's slot layers into one array per slot (asset order, matching planUnion). A
    // slot whose layers are all uniform-size blocks uploads as one compressed array (no transcode, no mip blit
    // — the block mips ride from the transcoder); a mixed-source or mixed-size slot falls back to one RGBA array
    // (any compressed layer transcoded to RGBA via the spill path, every layer resized to a common size).
    const data = {} as DataSlots;
    for (const name of DATA_NAMES) {
        const imgs = assets.flatMap((a) => a.textures.maps[name].images);
        if (!imgs.length) {
            data[name] = fallback1x1(device, DATA_FORMAT[name]);
            continue;
        }
        const blocks = uniformBlocks(imgs);
        if (blocks) {
            const tex = allocCompressed(device, blocks, `gltf-${name}`);
            data[name] = tex;
            blocks.forEach((img, layer) => {
                steps.push(async () => writeCompressedLayer(device, tex, img, layer));
            });
        } else {
            const bitmaps = await Promise.all(
                imgs.map(async (i) =>
                    i.kind === "bitmap" ? i.bitmap : spillToBitmap(i.bytes, await transcoder()),
                ),
            );
            const size = commonSize(
                bitmaps.map((bm) => ({ w: bm.width, h: bm.height })),
                2048,
            );
            const levels = mipLevels(size);
            const fmt = DATA_FORMAT[name];
            const tex = allocArray(device, size, bitmaps.length, levels, fmt);
            data[name] = tex;
            bitmaps.forEach((bm, layer) => {
                steps.push(() => uploadLayer(device, tex, bm, layer, size, levels, fmt));
            });
        }
    }

    const paletteBytes = packMaterials(plan.materials, plan.layers);
    const palette = device.createBuffer({
        label: "gltf-material-data",
        size: Math.max(MATERIAL_STRIDE, paletteBytes.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(palette, 0, paletteBytes);

    if (plan.albedo.loc.some((l) => l.spilled)) {
        const sizes = [
            ...new Set(flatImages.filter((i) => i.kind === "compressed").map(sizeOf)),
        ].join(", ");
        console.warn(
            `[gltf] >${ALBEDO_BUCKETS} distinct baseColor sizes across active assets (${sizes}); rarest decoded to RGBA`,
        );
    }

    const textures = { albedo, data, sampler: albedoSampler(device), palette };
    return { key, gen, textures, steps, cursor: 0 };
}

/** drain `staging`'s upload layers until `budgetMs` of main-thread time is spent this call — always at least one
 *  layer (forward progress even if one layer overruns). A TIME budget, not a byte one: the per-layer cost is the
 *  mip-blit encode (bound to layer count, not texels), so budgeting by bytes batched many cheap-byte/expensive-
 *  encode layers into one frame. Returns true when the last layer has uploaded (the caller then publishes the
 *  set). Per-frame from the union build system. */
export async function stepUnion(staging: UnionStaging, budgetMs: number): Promise<boolean> {
    const start = performance.now();
    while (staging.cursor < staging.steps.length) {
        await staging.steps[staging.cursor]();
        staging.cursor++;
        if (performance.now() - start >= budgetMs) break; // after ≥1 layer → a single slow layer still progresses
    }
    return staging.cursor >= staging.steps.length;
}

async function spillToBitmap(
    bytes: Uint8Array,
    transcode: (b: Uint8Array) => { width: number; height: number; rgba: Uint8Array },
): Promise<ImageBitmap> {
    const { width, height, rgba } = transcode(bytes);
    return createImageBitmap(new ImageData(new Uint8ClampedArray(rgba), width, height), {
        premultiplyAlpha: "none",
    });
}
