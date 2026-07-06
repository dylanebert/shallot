import type { State, System } from "../../engine";
import { Compute, checkTextureLimits, vec4 } from "../../engine";
import type { Binding } from "../../standard/render/core";
import { Surfaces } from "../../standard/render/core";
import { slab } from "../../standard/slab";
import { ALBEDO_NAMES } from "./image";
import { materialPreamble } from "./shade";
import type { GltfVat } from "./vat";

// The skinned/animated glTF render path — the GPU half of VAT. A skinned
// mesh is a builtin **skin surface** riding the instancing convention: the base mesh stays standard
// (topology / uv / vidx in the normal buffers), the baked per-frame positions + normals are two filterable
// textures (the VAT), per-instance animation time is the `skin` slab, and the surface's `vs` chunk samples
// the VAT by that time — hardware-lerped across adjacent frame rows — writes the deformed pose, then the
// standard instance transform applies. Because skinning is a VS warp, sear's shadow + prepass passes get
// the deformed pose free (the same compiled vertex math). Decode reference: keijiro HdrpVatExample
// VATHelper.hlsl; the deviceless bake is vat.ts.

/**
 * a skinned glTF instance's animation state — one `slab(vec4)` published as `"skin"`: lane x is the play
 * time in seconds (advanced by {@link SkinSystem}), y the material palette index in the shared union palette
 * (asset base + local id — the `mid` the shade path reads, folded in to keep the surface at the 10-storage
 * ceiling), z a per-instance phase offset (crowd variety; 0 for a single import), w the clip duration
 * {@link SkinSystem} loops the play time on. The duration is per-instance, so N skinned meshes with different
 * clip lengths coexist in one scene. The importer adds it to each skinned instance.
 */
export const Skin = { anim: slab(vec4, "skin") };

// the per-skinned-mesh VAT params the surface decodes against: the object-space AABB the f16 positions remap
// from, the effective sample fps, and the texture's frame/vertex extents. Each skinned mesh binds its own VAT
// textures + this uniform per-draw (via `Mesh.bindings`), so N skinned meshes coexist. Declared
// in the preamble (after the binding decl that references it — module-scope types resolve order-free, like
// MaterialData)
const VAT_PARAMS_WGSL = /* wgsl */ `
struct VatParams {
    aabbMin: vec3<f32>,
    fps: f32,
    aabbSize: vec3<f32>,
    frameCount: f32,
    vertCount: f32,
    frameMax: f32,
    pad0: f32,
    pad1: f32,
}
`;

// the bindings every skin surface declares: the instancing convention (eids + transforms) + baseColorFactor
// (color) + the folded `skin` slab (time + material index) + the per-material palette + the baseColor size
// buckets + four data PBR arrays + sampler (shared with the textured path's published names) + the two VAT
// textures + their sampler + the VAT params. Storage count is 5 (eids/transforms/color/skin/materialData) +
// sear's shared 5 (vertices/pointLights/lightGrid/lightIndices/meshQuant) = 10, the ceiling (gpu.md), zero
// headroom: folding (time, materialIndex) into one `skin` vec4 is what buys the room (replacing the textured
// path's separate `materialIndex` slab) now that the quant table claimed the last shared lane. The texture
// arrays are a separate limit; the baseColor buckets share the textured path's `sampleAlbedo` switch (shade.ts).
const skinBindings: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    color: { type: "storage", element: "u32" },
    skin: { type: "storage", element: "vec4<f32>" },
    materialData: { type: "storage", element: "MaterialData" },
    ...Object.fromEntries(ALBEDO_NAMES.map((n) => [n, { type: "texture-2d-array" } as Binding])),
    mr: { type: "texture-2d-array" },
    normalTex: { type: "texture-2d-array" },
    occlusion: { type: "texture-2d-array" },
    emissive: { type: "texture-2d-array" },
    albedoSamp: { type: "sampler" },
    vatPos: { type: "texture-2d" },
    vatNorm: { type: "texture-2d" },
    vatSamp: { type: "sampler" },
    vatParams: { type: "uniform", struct: "VatParams" },
};

// the VAT_PARAMS struct + the shade helpers specialized to a material map-set — sear compiles one pipeline
// per map-set a scene draws (the mesh's `variant`), so a sparse-map skinned material samples only the maps
// it carries. The VS reads the VAT normal as a plain f16 vec3 (renormalized), so there's no oct decode here.
const skinPreamble = (variant: number) => VAT_PARAMS_WGSL + materialPreamble(variant);

// sample the VAT at this vertex's row for the instance's play time, decode position (AABB-remapped f16)
// + normal (a plain f16 vec3, renormalized — NOT oct: the sampler hardware-lerps adjacent frame rows, and
// oct interpolation is invalid across the octahedral seam, so a vertex whose normal rotates across the seam
// between two frames would lerp to a garbage direction; gpu.md rule 9), then compose the instance transform
// (option a — self-contained, so no sear scaffold change). `vidx` (the hardware vertex index, sear draws
// indexed) is the unique-vertex index the VAT row is keyed by (the skinned mesh owns its buffers, indices
// local [0, vertCount), baseVertex 0); textureSampleLevel because the VS has no implicit LOD, and the
// filtering sampler hardware-lerps the two adjacent frame rows (fractional `fc`).
const SKIN_VS = /* wgsl */ `
    let vid = vidx;
    let fc = clamp(skin[eid].x * vatParams.fps, 0.0, vatParams.frameMax);
    let suv = vec2<f32>((f32(vid) + 0.5) / vatParams.vertCount, (fc + 0.5) / vatParams.frameCount);
    let p = vatParams.aabbMin + textureSampleLevel(vatPos, vatSamp, suv, 0.0).xyz * vatParams.aabbSize;
    let n = normalize(textureSampleLevel(vatNorm, vatSamp, suv, 0.0).xyz);
    let xf = transforms[eid];
    world = vec4<f32>(xformPoint(xf, p), 1.0);
    worldNormal = xformNormal(xf, n);
    localPos = p;`;

// register the three alpha-mode skin surfaces — opaque / MASK (clip cutout → holed shadows) / BLEND. They
// share SKIN_VS (the VAT deform) + the `shadePbr` path; only the blend mode + cutout discard differ, exactly
// like the textured `gltf-albedo*` trio. `mid` (the palette index) is the folded `skin[eid].y` lane.
export function registerSkinSurfaces(): void {
    Surfaces.register({
        name: "skin",
        bindings: skinBindings,
        specialize: (variant) => ({ preamble: skinPreamble(variant) }),
        vs: SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let base = sampleAlbedo(mid, uv).rgb * unpackLdrColor(color[eid]).rgb;
        col = vec4<f32>(shadePbr(mid, uv, base, normalize(worldNormal), world), 1.0);`,
    });
    Surfaces.register({
        name: "skin-clip",
        blend: "clip",
        bindings: skinBindings,
        specialize: (variant) => ({ preamble: skinPreamble(variant) }),
        vs: SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let tex = sampleAlbedo(mid, uv);
        let c = unpackLdrColor(color[eid]);
        let rgb = shadePbr(mid, uv, tex.rgb * c.rgb, normalize(worldNormal), world);
        if (tex.a * c.a < materialData[mid].cutoff) { discard; }
        col = vec4<f32>(rgb, 1.0);`,
    });
    Surfaces.register({
        name: "skin-blend",
        blend: "alpha",
        bindings: skinBindings,
        specialize: (variant) => ({ preamble: skinPreamble(variant) }),
        vs: SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let tex = sampleAlbedo(mid, uv) * unpackLdrColor(color[eid]);
        col = vec4<f32>(shadePbr(mid, uv, tex.rgb, normalize(worldNormal), world), tex.a);`,
    });
}

// the surface name per glTF alphaMode — the importer routes each skinned instance by its material's mode
export function skinSurface(alphaMode: "OPAQUE" | "MASK" | "BLEND"): string {
    return alphaMode === "MASK" ? "skin-clip" : alphaMode === "BLEND" ? "skin-blend" : "skin";
}

/**
 * the assembled VAT GPU resources — the two filterable textures + the params uniform + the clip duration,
 * built once per skinned mesh + held by the asset cache (survives a State rebuild). Bound per-draw via the
 * mesh's `Mesh.bindings` (so N skinned meshes coexist); the cache frees them on invalidate.
 */
export interface AssembledVat {
    pos: GPUTexture;
    norm: GPUTexture;
    params: GPUBuffer;
    sampler: GPUSampler;
    duration: number;
}

// the build-scoped 1×1 VAT set published at warm under the global binding names — so a skin surface's no-op
// draws over non-skinned meshes resolve cleanly before any skinned mesh loads — and freed in dispose. A real
// skinned mesh binds its own VAT per-draw (Mesh.bindings), never this; the per-instance clip duration rides
// the `skin` slab's w lane (SkinSystem), not a module global.
let _fallbackVat: AssembledVat | null = null;

// bind a VAT set to the global `vat*` binding names — the build-scoped fallback ({@link fallbackVat}) uses
// this so a skin surface's no-op draws over non-skinned meshes resolve before any skinned mesh loads. A real
// skinned mesh overrides these per-draw via `Mesh.bindings`, so its VAT is never published globally.
function publishVat(vat: AssembledVat): void {
    Compute.textures.set("vatPos", vat.pos);
    Compute.textures.set("vatNorm", vat.norm);
    Compute.samplers.set("vatSamp", vat.sampler);
    Compute.buffers.set("vatParams", vat.params);
}

/**
 * encode a baked clip into the two filterable VAT textures + the params uniform, returning the assembled set
 * the asset cache holds + {@link publishVat} binds. Positions → `rgba16float` remapped to the per-mesh AABB
 * `[0,1]` (where f16 holds the most precision), normals → `rgba16float` plain xyz (renormalized in the VS —
 * **not** oct, so the hardware frame-lerp can't cross the octahedral seam; gpu.md rule 9). Both are core
 * filterable formats (the 16-bit *norm* formats need the non-floor `texture-formats-tier1` feature), so the
 * linear clamp sampler gives the VS free hardware frame-lerp on every floor device. The remap keeps f16
 * sub-0.1-unit on a model-scale mesh — far inside the rgba8 banding the survey rejected.
 */
export function assembleVat(device: GPUDevice, vat: GltfVat): AssembledVat {
    const { frameCount, vertCount, positions, normals, aabb } = vat;
    // fail loud + clear before the f16 fill + upload: the VAT is a vertCount × frameCount texture, so a mesh
    // with too many unique vertices or baked frames blows the device's 2D-texture limit. Throw a named error
    // here, not at an opaque createTexture validation error (the contact-store guard's pattern, texture form).
    checkTextureLimits(
        "[gltf] a skinned mesh's VAT",
        { width: vertCount, height: frameCount },
        device.limits,
        "Reduce the mesh's unique vertex count or bake fewer animation frames (the VAT is " +
            "vertCount × frameCount texels).",
    );
    const sx = Math.max(aabb.max[0] - aabb.min[0], 1e-6);
    const sy = Math.max(aabb.max[1] - aabb.min[1], 1e-6);
    const sz = Math.max(aabb.max[2] - aabb.min[2], 1e-6);

    const texels = frameCount * vertCount;
    // Float16Array is the platform floor (Chrome 126+ / our bundled Chromium) — this VAT path is GPU-only
    const pos16 = new Float16Array(texels * 4);
    const nrm16 = new Float16Array(texels * 4);
    for (let i = 0; i < texels; i++) {
        pos16[i * 4] = (positions[i * 3] - aabb.min[0]) / sx;
        pos16[i * 4 + 1] = (positions[i * 3 + 1] - aabb.min[1]) / sy;
        pos16[i * 4 + 2] = (positions[i * 3 + 2] - aabb.min[2]) / sz;
        pos16[i * 4 + 3] = 1;
        nrm16[i * 4] = normals[i * 3];
        nrm16[i * 4 + 1] = normals[i * 3 + 1];
        nrm16[i * 4 + 2] = normals[i * 3 + 2];
        nrm16[i * 4 + 3] = 0;
    }

    const dim = { width: vertCount, height: frameCount };
    const posTex = device.createTexture({
        label: "gltf-vat-pos",
        size: dim,
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: posTex }, pos16, { bytesPerRow: vertCount * 8 }, dim);
    const normTex = device.createTexture({
        label: "gltf-vat-norm",
        size: dim,
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture({ texture: normTex }, nrm16, { bytesPerRow: vertCount * 8 }, dim);

    const params = new Float32Array(12);
    params.set([aabb.min[0], aabb.min[1], aabb.min[2], vat.fps], 0);
    params.set([sx, sy, sz, frameCount], 4);
    params.set([vertCount, Math.max(frameCount - 1, 0)], 8);
    const paramsBuf = device.createBuffer({
        label: "gltf-vat-params",
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, params);

    return {
        pos: posTex,
        norm: normTex,
        params: paramsBuf,
        sampler: vatSampler(device),
        duration: vat.duration,
    };
}

// a filterable clamp sampler: the frame axis (v) hardware-lerps adjacent rows; the vertex axis (u) is
// sampled at exact texel centers so it returns the discrete vertex without neighbor bleed
function vatSampler(device: GPUDevice): GPUSampler {
    return device.createSampler({
        label: "gltf-vat-sampler",
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
    });
}

/**
 * publish 1×1 VAT fallbacks + a zero params uniform under the global `vat*` names so a skin surface's no-op
 * draws over non-skinned meshes bind cleanly (the same fallback shape the texture arrays use; an unbound draw
 * is skipped). A real skinned mesh binds its own VAT per-draw (`Mesh.bindings`), so this set is never replaced
 * — only the no-op pairs read it. Build-scoped, freed in {@link disposeVatFallback}.
 */
export function fallbackVat(device: GPUDevice): void {
    const fallback = (format: GPUTextureFormat) =>
        device.createTexture({
            label: "gltf-vat-fallback",
            size: { width: 1, height: 1 },
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
    const params = device.createBuffer({
        label: "gltf-vat-params-fallback",
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _fallbackVat = {
        pos: fallback("rgba16float"),
        norm: fallback("rgba16float"),
        params,
        sampler: vatSampler(device),
        duration: 0,
    };
    publishVat(_fallbackVat);
}

/**
 * free the build-scoped VAT fallback (GltfPlugin.dispose). The cache-owned real VATs survive — freed only by
 * the asset cache's invalidate / clearGltfCache.
 */
export function disposeVatFallback(): void {
    if (!_fallbackVat) return;
    _fallbackVat.pos.destroy();
    _fallbackVat.norm.destroy();
    _fallbackVat.params.destroy();
    _fallbackVat = null;
}

/**
 * advance each skinned instance's play time, looping on its own clip duration (`Skin.anim.w`, so N meshes
 * with different clip lengths coexist). Reload-safe — time is derived from `state.time.elapsed` + the
 * instance's phase lane, never accumulated (ecs.md "no module-level accumulator"). A `simulation`-group
 * system, so SlabSystem (`draw`, first) flushes the write before sear's geometry passes read it. Runs in play
 * only (default mode), so the editor shows the rest/bind pose.
 */
export const SkinSystem: System = {
    name: "Skin",
    group: "simulation",
    update(state: State) {
        const t = state.time.elapsed;
        for (const eid of state.query([Skin])) {
            const duration = Skin.anim.w.get(eid);
            if (duration <= 0) continue;
            const phase = Skin.anim.z.get(eid);
            let time = (t + phase) % duration;
            if (time < 0) time += duration;
            Skin.anim.x.set(eid, time);
        }
    },
};
