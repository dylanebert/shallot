// Sear's WGSL codegen: device-free, pure text generation. Everything here builds a WGSL module string (or
// a fragment of one) from a {@link Surface} / {@link Background} + the shared frame/view/lighting scaffold —
// no `GPUDevice`, no pipeline, no buffer. `pipelines.ts` compiles the strings this file produces; `atlas.ts`
// owns the shadow-atlas GPU state the point/cascade codegen here reads config from (via `./shadows`'
// pure getters, fenced by `checkShadowConfig`).

import tgpu from "typegpu";
import {
    chunk,
    ldrColorUnpackWgsl,
    octEncodeWgsl,
    posQuantWgsl,
    spliceNs,
    xformWgsl,
} from "../../engine/utils/core";
import type { Binding, Surface, View } from "../render/core";
import {
    clusterCell,
    distanceAttenuation,
    frameWgsl,
    lightingWgsl,
    pointLightsWgsl,
    spotFactor,
    VIEW_BYTES,
    viewWgsl,
} from "../render/core";
import type { Background } from "./forward";
import { COMBO_SHIFT, EID_MASK } from "./regather";
import {
    casterWgsl,
    comboMetaSchema,
    faceVPsSchema,
    pbrWgsl,
    pointShadowWgsl,
    sunShadowWgsl,
    sunStructWgsl,
    tileRectsSchema,
} from "./shade";
import {
    cascadeAtlasSize,
    MAX_CASCADES,
    pointAtlasSize,
    pointCasters,
    sunCascades,
    sunResolution,
} from "./shadows";

export const VS_FS = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;

// the depth format shared by the color pass's own 4× MSAA depth, the 1× prepass depth, and the shadow
// map. depth32float is sampleable and the reverse-Z precision win needs a float buffer (an integer depth
// gains nothing from reverse-Z); one source of truth for every depth-stencil state — and the shadow map
// renders through sear's compiled prepass depth pipeline, so one format keeps them sharing it
/** the depth-stencil format for every sear depth target (color pass, prepass, shadow atlases): one format so they share the compiled prepass depth pipeline */
export const DEPTH_FORMAT: GPUTextureFormat = "depth32float";

// the geometric-AA sample count when a camera's `Camera.antialias` is on (the default); off renders
// single-sample straight into the offscreen, no resolve. Per-camera + runtime-toggleable, sample-based
// only (no post-process blur). The prepass + shadow map stay single-sample regardless — an id can't
// MSAA-resolve and a render pass can't mix counts, so they own their own 1× depth, never varying by this
export const SAMPLE_COUNT = 4;

// the id lane's screen-space target, published per camera as `view.tag`. r32uint holds the front-most
// fragment's tag per pixel, filled by the single-sample prepass (not the color MRT — see PrepassSystem)
// for cameras carrying the `Tag` marker. The tag is surface-authored — a mutable fs local defaulting to
// the entity's eid for an instanced surface and TAG_NONE otherwise, which a surface's fs overrides
// (terrain → `capacity + cell`). A consumer reads `view.tag` to know which surface owns each pixel
// (hover, outline, debug)
/** the id-lane texture format (`r32uint`): an integer id can't MSAA-resolve, which forces the single-sample prepass */
export const TAG_FORMAT: GPUTextureFormat = "r32uint";

// the reserved tag sentinel: the default for a non-instanced surface that authors no tag, and the
// value the tag target clears to (the background). eids are bounded by `capacity`, so 0xffffffff
// never collides with a real one — a reader takes any other value as a literal surface tag. Exported
// so a consumer interprets the readback without re-deriving the sentinel
/** the reserved id-lane sentinel: a pixel no surface owns. A consumer decoding `view.tag` reads any other value as a literal surface tag */
export const TAG_NONE = 0xffffffff;

/**
 * opt a Sear camera into the **id lane**: the `view.tag` target {@link PrepassSystem} fills. Unreal's
 * `CustomStencil` generalized from an 8-bit stencil to a u32 lane; Bevy's prepass carries no id (it
 * CPU-raycasts), so the id rides this single-sample pass because it's the same rasterization. A marker
 * in the spirit of Bevy's `DepthPrepass` / `NormalPrepass`: add it to enable one extra camera output;
 * omit it and the lane is absent (no target allocated). Per-camera, so a minimap opts out while the main
 * view opts in. A consumer that reads `view.tag` (hover, outline, picking) wants this on its camera; the
 * engine itself stays tag-agnostic.
 *
 * @example
 * ```
 * <a camera sear tag transform />
 * ```
 */
export const Tag = {};

// ---- prepass lanes: opt-in screen-space outputs, a closed engine-owned union (not a consumer registry).
// Each lane is gated by a camera marker (Bevy's DepthPrepass / NormalPrepass shape). Two ship: the `depth`
// lane (the depth-stencil itself, marker `Depth`, published as `view.depth`) and the id lane (a color
// attachment, marker `Tag`, published as `view.tag`). normal / motion are the future rows — adding one is
// a COLOR_LANES entry + a `View.*` field, and the subset codegen in `surfaceCode` + the prepass already
// iterate it; never a new pass. `depth` isn't a color attachment (it's the depth-stencil), so it's stored
// or discarded by the `Depth` marker, separate from COLOR_LANES (the color attachments the prepass MRTs)
export interface ColorLane {
    // the `view.*` field + lane identity ("tag"); `set` publishes the rendered texture onto that field
    name: string;
    marker: object;
    format: GPUTextureFormat;
    usage: number;
    clear: GPUColor;
    // the mutable fs local a surface authors (symmetric with `col`), its WGSL type, and its per-surface
    // default (the fs chunk may override it)
    local: string;
    type: string;
    init(instanced: boolean): string;
    set(view: View, texture: GPUTexture): void;
}

// the id lane: the front-most opaque / `clip` surface's tag per pixel. r32uint (an integer id can't
// MSAA-resolve — what forces the prepass single-sample), COPY_SRC for a hover readback + TEXTURE_BINDING
// for an outline sample, cleared to TAG_NONE (the "no surface" background). The tag local defaults to the
// instance's eid (instanced) or TAG_NONE (a world-space producer like terrain authors its own)
export const COLOR_LANES: ColorLane[] = [
    {
        name: "tag",
        marker: Tag,
        format: TAG_FORMAT,
        usage:
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_SRC,
        clear: { r: TAG_NONE, g: 0, b: 0, a: 0 },
        local: "tag",
        type: "u32",
        init: (instanced) => (instanced ? "eid" : `${TAG_NONE}u`),
        set: (view, texture) => {
            view.tag = texture;
        },
    },
];

// every subset of the color lanes (the requestable MRT lane-sets — each lane has its own marker, so any
// combination is valid), smallest first: [] then [tag]. The empty set is the position-only depth prepass
// (the shadow map reuses it); [tag] is the id lane. `surfaceCode` emits one prepass fragment per subset;
// `prepareSear` compiles one pipeline per subset — bounded, since the lane set is engine-closed
export function laneSubsets(): ColorLane[][] {
    return COLOR_LANES.reduce<ColorLane[][]>(
        (acc, lane) => acc.concat(acc.map((s) => [...s, lane])),
        [[]],
    );
}

// a lane-set's stable key (the prepass pipeline-map key): "" for the empty depth-only set, "tag" for the
// id lane, "tag-normal" when normal lands
export function laneKey(lanes: ColorLane[]): string {
    return lanes.map((l) => l.name).join("-");
}

// a lane-set's WGSL fragment entry point: fsPrepass (empty depth-only set), fsPrepassTag (id lane)
export function prepassEntry(lanes: ColorLane[]): string {
    return `fsPrepass${lanes.map((l) => l.name[0].toUpperCase() + l.name.slice(1)).join("")}`;
}

// frame/view/lighting + vertex pull + the clustered point-light bindings (0..7), then the
// surface's own bindings
export const FRAME = 0;
export const VIEW = 1;
export const LIGHTING = 2;
// slot 3 is the vertex stream: the color pipelines bind the 16 B main stream (`array<vec4<u32>>`), the
// prepass/shadow pipelines bind the 8 B position-only stream (`array<vec2<u32>>`) — same slot, distinct
// buffers via the two bind groups (the layout entry is `read-only-storage` either way). `meshQuant` is the
// one net-new shared binding: it pushes the heaviest surfaces (skin / glTF-textured) to exactly the
// 10-storage-per-stage ceiling (gpu.md), no headroom — reuse a cols-buffer lane before adding another
export const VERTICES = 3;
export const POINT_LIGHTS = 4;
export const LIGHT_GRID = 5;
export const LIGHT_INDICES = 6;
export const MESH_QUANT = 7;
export const SURFACE_BASE = 8;

/** storage bindings every sear color pass shares (indices `VERTICES..SURFACE_BASE`): vertices, pointLights, lightGrid, lightIndices, meshQuant. A surface's own storage plus this must fit the 10-per-stage ceiling (gpu.md), so a surface has `10 − SHARED_STORAGE_COUNT` for its own. Derived from the binding indices so a sixth shared binding (which bumps `SURFACE_BASE`) updates it automatically — the gltf storage-ceiling audit (`extras/gltf/live.test.ts`) imports it. */
export const SHARED_STORAGE_COUNT = SURFACE_BASE - VERTICES;

// a background's own bindings start here — after frame/view/lighting (0/1/2). A backdrop needs none of
// the surface group's vertex-pull / light-grid bindings, so its group 0 is just the three uniforms + these
export const BG_BASE = 3;

// the clustered point/spot evaluation primitives, relocatable (no view / fragCoord globals): both sear's
// color FS and the fog volumetric march splice them. `distanceAttenuation` + `spotFactor` live with the
// compacted light list they read (render/lighting.ts), `clusterCell` with the froxel grid it indexes
// (render/cluster.ts) — this chunk is the relocatable bundle a consumer splices
/** relocatable clustered-light WGSL (`distanceAttenuation` / `spotFactor` / `clusterCell`) so a screen-space consumer evaluates the same froxel lights sear's color FS does */
export function lightEvalWgsl(): string {
    // force the base chunks first: `PointLightGpu` belongs to the light-list chunk and `octDecodeNormal`
    // to the oct chunk, and every consumer splices both ahead of this one
    pointLightsWgsl();
    octEncodeWgsl();
    return lightEvalChunk();
}

export const lightEvalChunk = chunk(
    "lightEvalWgsl",
    [distanceAttenuation, spotFactor, clusterCell],
    spliceNs,
);

// the lighting helpers sear exposes to surface chunks: `lightFactor(normal)`
// is ambient + sun·ndl·shadow + the point-light sum (callable in the vs for per-vertex
// shading), `lit(base, normal)` applies it to a base color (per-fragment). A surface
// shades by calling them, or writes `col` directly to stay unlit.
//
// `sunVisibility` is the sun-shadow seam: the visibility of the sun at this fragment,
// multiplied into the sun term only (ambient stays unshadowed, matching niagara's
// ndotl·shadow·sun + ambient). The fragment scaffold sets it by projecting the fragment's
// world position into the shadow map inline (`sampleSunShadow`, see SHADOW_WGSL) before
// splicing the surface chunk; it defaults to 1.0, so the vertex stage (no world fragment)
// and shadowless frames are fully lit — the no-op fallback.
//
// `fragWorld` + `fragCoord` + `pointScale` are the point-light seam, the same scaffold-filled
// shape: the color FS sets the fragment's world position + screen coords and enables the
// cluster loop; the defaults (pointScale 0) make the vs and the prepass entries a no-op. So
// per-vertex shading (lightFactor in the vs) receives NEITHER sun shadows NOR point lights;
// per-pixel lit() gets both. The loop reads only the fragment's froxel cluster's lights —
// `clusterOf` maps (fragCoord.xy, view depth) to the grid cell the light cull binned into
// (view.cluster carries near/far/perspective/slot). The falloff is Bevy's
// getDistanceAttenuation — inverse-square windowed smoothly to exactly zero at the range;
// `distanceAttenuation` is the one function both this shader and the CPU oracles call
// (render/lighting.ts, via lightEvalWgsl).
//
// The temporary string-contract adapter retains this raw binding scaffold until its consumers migrate.
// `engineScaffoldWgsl()` cannot replace it directly: the canonical typed group 0 puts `pointLights` at
// binding 3, while this legacy layout still puts the pass-specific vertex stream there. Rewriting emitted
// binding attributes would be a more fragile second codegen path than keeping this local copy for one stage.
const LEGACY_LIGHT_WGSL = /* wgsl */ `
var<private> sunVisibility: f32 = 1.0;
var<private> fragWorld: vec3<f32> = vec3<f32>(0.0);
var<private> fragCoord: vec4<f32> = vec4<f32>(0.0);
var<private> pointScale: f32 = 0.0;

// the fragment's slot-major cluster index (clusterCell, lightEvalWgsl). View depth recovers from the
// position builtin: perspective clip.w is the view depth (fragCoord.w = 1/clip.w); orthographic depth is
// linear in fragCoord.z
fn clusterOf() -> u32 {
    let near = view.cluster.x;
    let far = view.cluster.y;
    var viewZ = 1.0 / fragCoord.w;
    if (view.cluster.z < 0.5) { viewZ = near + fragCoord.z * (far - near); }
    return clusterCell(
        fragCoord.x / view.resolution.x, fragCoord.y / view.resolution.y, viewZ, near, far, u32(view.cluster.w));
}

fn pointFactor(normal: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    if (pointScale == 0.0) { return sum; }
    let entry = lightGrid[clusterOf()];
    for (var i = 0u; i < entry.y; i = i + 1u) {
        let light = pointLights.lights[lightIndices[entry.x + i]];
        let toLight = light.posRange.xyz - fragWorld;
        let distSq = dot(toLight, toLight);
        let radiusSq = light.params.x * light.params.x;
        let L = toLight * inverseSqrt(max(distSq, 0.0001));
        let diff = halfLambert(dot(normal, L));
        sum += light.color.rgb
            * (distanceAttenuation(distSq, light.posRange.w, radiusSq) * diff * spotFactor(light, L) * pointShadowOf(light, normal, fragWorld));
    }
    return sum;
}

fn lightFactor(normal: vec3<f32>) -> vec3<f32> {
    let L = -lighting.sunDirection.xyz;
    let sun = halfLambert(dot(normal, L));
    return lighting.ambientColor.rgb * lighting.ambientColor.a
        + lighting.sunColor.rgb * sun * sunVisibility
        + pointFactor(normal);
}

fn lit(baseColor: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return baseColor * lightFactor(normal);
}

// per-pixel (or per-vertex) metallic-roughness shading. Shares the sun-shadow / point-cluster seam with
// \`lightFactor\`: \`sunVisibility\` and \`pointScale\`/\`fragWorld\` are the same fs-scaffold privates, so a
// vs-side call (pointScale 0, sunVisibility 1) gets neither point lights nor sun shadows, same as the
// diffuse path. At metallic 0 / roughness 1 / dielectric 0 this reduces to \`lit(albedo, normal)\` exactly.
fn litPbr(s: Pbr, normal: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let V = normalize(view.eye.xyz - world);
    var radiance = lighting.ambientColor.rgb * lighting.ambientColor.a * s.albedo * s.occlusion;
    radiance += lighting.sunColor.rgb * sunVisibility * brdf(s, normal, V, -lighting.sunDirection.xyz);
    if (pointScale != 0.0) {
        let entry = lightGrid[clusterOf()];
        for (var i = 0u; i < entry.y; i = i + 1u) {
            let light = pointLights.lights[lightIndices[entry.x + i]];
            let toLight = light.posRange.xyz - fragWorld;
            let distSq = dot(toLight, toLight);
            let radiusSq = light.params.x * light.params.x;
            let dist = sqrt(max(distSq, 1e-8));
            let L = toLight / dist;
            let f = distanceAttenuation(distSq, light.posRange.w, radiusSq) * spotFactor(light, L) * pointShadowOf(light, normal, fragWorld);
            radiance += light.color.rgb * f * brdfSphere(s, normal, V, L, dist, light.params.x);
        }
    }
    return radiance;
}
`;

// the shared group-0 bindings. Slot 3 is pass-specific: the color module reads the 16 B main stream
// (`vertices`, `vec4<u32>` — pos + meshId / oct normal / uv), the prepass + shadow module reads the
// 8 B position-only stream (`position`, `vec2<u32>` — pos + meshId), bound at the same slot by the two
// per-draw bind groups. `meshQuant` (MESH_QUANT_WGSL) is spliced after posQuantWgsl() defines MeshQuant
export const uniformWgsl = (pass: "color" | "prepass") => /* wgsl */ `
@group(0) @binding(${FRAME}) var<uniform> frame: Frame;
@group(0) @binding(${VIEW}) var<uniform> view: View;
@group(0) @binding(${LIGHTING}) var<uniform> lighting: Lighting;
${
    pass === "color"
        ? `@group(0) @binding(${VERTICES}) var<storage, read> vertices: array<vec4<u32>>;`
        : `@group(0) @binding(${VERTICES}) var<storage, read> position: array<vec2<u32>>;`
}
@group(0) @binding(${POINT_LIGHTS}) var<storage, read> pointLights: PointLights;
@group(0) @binding(${LIGHT_GRID}) var<storage, read> lightGrid: array<vec2<u32>>;
@group(0) @binding(${LIGHT_INDICES}) var<storage, read> lightIndices: array<u32>;`;

// the per-mesh dequant table — spliced after posQuantWgsl() (it references the MeshQuant struct it defines)
export const MESH_QUANT_WGSL = /* wgsl */ `@group(0) @binding(${MESH_QUANT}) var<storage, read> meshQuant: array<MeshQuant>;`;

// the relocatable shadow chunks the color FS splices (via `shadowWgsl`) are in ./shade, and a
// screen-space consumer's pass (the fog volumetric march) splices the same ones — one source of truth,
// each consumer declaring the group-1 bindings they reference by name. The color pass's opaque +
// transparent pipelines call `sampleSunShadow` / `pointShadowOf`; the tag + depth pipelines omit group 1,
// so their fragments never reference these and stay valid. `enabled: 0` / an empty caster slot is the
// no-cast fallback → fully lit.
export const shadowWgsl = () => /* wgsl */ `
${sunStructWgsl()}
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;
@group(1) @binding(2) var<uniform> sunShadow: SunShadow;
${sunShadowWgsl()}

${casterWgsl()}
@group(1) @binding(3) var pointAtlas: texture_depth_2d;
@group(1) @binding(4) var<uniform> pointShadows: PointCasters;
@group(1) @binding(5) var<uniform> tileRects: TileRects;
${pointShadowWgsl()}`;

// the prepass module's shadow seam: stubs with the color module's signatures, no group-1
// declarations. The prepass entries splice the same surface chunk as the color fs, and a chunk's
// `lit()` reaches `pointFactor` → `pointShadowOf` — a real atlas read there would statically pull
// group 1 into the prepass pipelines (which bind group 0 alone) and make the atlas render's own
// `clip` discard pass sample the very texture it's writing. The stubs keep the prepass module
// group-1-free; at runtime the chunk's lighting result is discarded anyway (pointScale stays 0)
export const SHADOW_STUB_WGSL = /* wgsl */ `
fn sampleSunShadow(worldPos: vec3<f32>, normal: vec3<f32>) -> f32 { return 1.0; }
fn pointShadowOf(light: PointLightGpu, normal: vec3<f32>, fragWorld: vec3<f32>) -> f32 { return 1.0; }`;

// the standard instance transform. A surface declaring `eids` + `transforms` storage
// bindings is instanced: sear reads this instance's entity id and applies its world
// transform to position + normal, before splicing the surface's own vs chunk. The normal
// uses the inverse-transpose (`xformNormal`), correct under non-uniform scale. A surface
// without those bindings (a producer whose geometry is already world-space) stays identity.
// This is the instancing convention — nothing here is Part-specific; Part is one producer
// that publishes `eids`, and any producer publishing them gets the transform
export const INSTANCE_VS = /* wgsl */ `eid = eids[iid];
    let xf = transforms[eid];
    world = vec4<f32>(xformPoint(xf, world.xyz), world.w);
    worldNormal = xformNormal(xf, worldNormal);`;

// a Binding maps to a WGSL declaration and a matching layout entry; both number
// the slot identically so they stay in lockstep
export function bindingDecl(name: string, b: Binding, i: number): string {
    switch (b.type) {
        case "uniform":
            return `@group(0) @binding(${i}) var<uniform> ${name}: ${b.struct};`;
        case "storage": {
            const access = b.access === "read_write" ? "read_write" : "read";
            return `@group(0) @binding(${i}) var<storage, ${access}> ${name}: array<${b.element}>;`;
        }
        case "texture-2d":
            return `@group(0) @binding(${i}) var ${name}: texture_2d<f32>;`;
        case "texture-2d-array":
            return `@group(0) @binding(${i}) var ${name}: texture_2d_array<f32>;`;
        case "texture-depth-2d":
            return `@group(0) @binding(${i}) var ${name}: texture_depth_2d;`;
        case "sampler":
            return `@group(0) @binding(${i}) var ${name}: sampler;`;
        case "sampler-comparison":
            return `@group(0) @binding(${i}) var ${name}: sampler_comparison;`;
    }
}

export function bindingEntry(b: Binding, i: number): GPUBindGroupLayoutEntry {
    switch (b.type) {
        case "uniform":
            return { binding: i, visibility: VS_FS, buffer: { type: "uniform" } };
        case "storage": {
            const type = b.access === "read_write" ? "storage" : "read-only-storage";
            return {
                binding: i,
                visibility: b.access === "read_write" ? GPUShaderStage.FRAGMENT : VS_FS,
                buffer: { type },
            };
        }
        case "texture-2d":
            return { binding: i, visibility: VS_FS, texture: { sampleType: "float" } };
        case "texture-2d-array":
            return {
                binding: i,
                visibility: VS_FS,
                texture: { sampleType: "float", viewDimension: "2d-array" },
            };
        case "texture-depth-2d":
            return { binding: i, visibility: VS_FS, texture: { sampleType: "depth" } };
        case "sampler":
            return { binding: i, visibility: VS_FS, sampler: { type: "filtering" } };
        case "sampler-comparison":
            return { binding: i, visibility: VS_FS, sampler: { type: "comparison" } };
    }
}

export const UNIFORM_LAYOUT: GPUBindGroupLayoutEntry[] = [
    { binding: FRAME, visibility: VS_FS, buffer: { type: "uniform" } },
    // one whole per-slot View buffer, no dynamic offset — `record` binds `Render.viewBuffers[slot]`
    // directly (the per-slot-buffer design, Approach 4a)
    { binding: VIEW, visibility: VS_FS, buffer: { type: "uniform", minBindingSize: VIEW_BYTES } },
    { binding: LIGHTING, visibility: VS_FS, buffer: { type: "uniform" } },
    // slot 3 (the vertex stream) — same layout entry for both bind groups; the color group binds the
    // 16 B main buffer here, the prepass group the 8 B position buffer (the element type is shader-side)
    { binding: VERTICES, visibility: VS_FS, buffer: { type: "read-only-storage" } },
    { binding: POINT_LIGHTS, visibility: VS_FS, buffer: { type: "read-only-storage" } },
    { binding: LIGHT_GRID, visibility: VS_FS, buffer: { type: "read-only-storage" } },
    { binding: LIGHT_INDICES, visibility: VS_FS, buffer: { type: "read-only-storage" } },
    { binding: MESH_QUANT, visibility: VS_FS, buffer: { type: "read-only-storage" } },
];

// built-in inter-stage varyings lead the struct; custom interpolators follow, capped at 16 total. The
// set is per-surface, not fixed: `worldNormal` (a plain vec3, renormalized in the fs — oct-encoding a
// normal for interpolation breaks across the octahedral seam, and a vec2 fills a full @location slot
// anyway, so oct bought nothing here; see `builtinFields`) + `eid` + `world` always cross (the color
// scaffold's sampleSunShadow / fragWorld / tag default reach them), but `uv` + `localPos` cross only when
// the surface's fs reads them — an unread interpolator is per-fragment varying bandwidth, pruned on desktop
const MAX_INTERSTAGE = 16;

const RESERVED = new Set([
    "clip",
    "clipPos",
    "worldNormal",
    "uv",
    "localPos",
    "localNormal",
    "eid",
    "world",
    "vidx",
    "iid",
    "v",
    "out",
    "fin",
    "col",
    "tag",
    "lit",
    "lightFactor",
    "sunVisibility",
    "fragWorld",
    "fragCoord",
    "pointScale",
    "pointFactor",
    "pointLights",
    "lightGrid",
    "lightIndices",
    "clusterOf",
    "distanceAttenuation",
    "sampleSunShadow",
    "shadowMap",
    "shadowSamp",
    "sunShadow",
    "pointShadowOf",
    "pointFaceOf",
    "pointAtlas",
    "pointShadows",
    "tileRects",
    "position",
    "meshQuant",
    "octEncodeNormal",
    "octDecodeNormal",
    "decodePos",
    "decodeUv",
    "meshIdOf",
]);

// lower a surface's custom interpolators into the WGSL fragments the scaffold splices, so the vs chunk
// and the fs see the same varyings. `base` is the @location the customs start at — the builtin count,
// which varies per surface now that uv/localPos prune (see builtinFields)
export function interp(record: Record<string, string>, name: string, base: number) {
    const fields = Object.entries(record);
    if (base + fields.length > MAX_INTERSTAGE) {
        throw new Error(
            `sear: surface "${name}" declares ${fields.length} interpolators; max ${MAX_INTERSTAGE - base} (inter-stage limit)`,
        );
    }
    for (const [n] of fields) {
        if (RESERVED.has(n)) {
            throw new Error(
                `sear: surface "${name}" interpolator "${n}" collides with a reserved name`,
            );
        }
    }
    return {
        // integer varyings must be flat — the rasterizer can't interpolate them
        out: fields
            .map(
                ([n, t], i) =>
                    `    @location(${base + i}) ${/\b(u32|i32)\b/.test(t) ? "@interpolate(flat) " : ""}${n}: ${t},`,
            )
            .join("\n"),
        decls: fields.map(([n, t]) => `    var ${n}: ${t};`).join("\n"),
        toOut: fields.map(([n]) => `    out.${n} = ${n};`).join("\n"),
        fromOut: fields.map(([n]) => `    let ${n} = fin.${n};`).join("\n"),
    };
}

// the built-in interstage fields a surface carries, in struct order. `worldNormal`, `eid`, and `world`
// always cross — the color scaffold (sampleSunShadow / fragWorld) and the tag default reach them regardless
// of the chunk. `uv` / `localPos` cross only when the fs reads them (the fs rebinds `fin.uv` / `fin.localPos`;
// an unread one is pure per-fragment varying waste). Returns the field list + the rebind fragments the vs
// and fs splice.
//
// The normal crosses as a **plain vec3, renormalized in the fs** — NOT oct-encoded. Oct-encoding a normal
// for *interpolation* is invalid across the octahedral seam: a triangle whose world normals straddle the
// z=0 equator interpolates between the inner diamond (z>0) and the outer corners (z<0) of the oct square,
// passing through coordinates that decode to garbage normals (the symptom: jagged normal zigzag on the faces
// of curved/draped geometry oriented to straddle the seam, e.g. one wall's banners but not the other's).
// A `vec2` oct field fills a full `@location` slot anyway, so the `vec3` costs no extra interpolator — oct
// here only bought the seam bug. (Oct stays correct for per-vertex *storage* `octEncodeWgsl()`, which decodes
// once per vertex without interpolating — the same seam hazard is why the VAT normal texture is a plain vec3.)
export function builtinFields(fs: string) {
    const needsUv = /\buv\b/.test(fs);
    const needsLocal = /\blocalPos\b/.test(fs);
    const out = [`    @location(0) worldNormal: vec3<f32>,`];
    let loc = 1;
    if (needsUv) out.push(`    @location(${loc++}) uv: vec2<f32>,`);
    if (needsLocal) out.push(`    @location(${loc++}) localPos: vec3<f32>,`);
    out.push(`    @location(${loc++}) @interpolate(flat) eid: u32,`);
    out.push(`    @location(${loc++}) world: vec3<f32>,`);
    return {
        count: loc,
        struct: out.join("\n"),
        toOut: [
            `    out.worldNormal = normalize(worldNormal);`,
            ...(needsUv ? [`    out.uv = uv;`] : []),
            ...(needsLocal ? [`    out.localPos = localPos;`] : []),
            `    out.eid = eid;`,
            `    out.world = world.xyz;`,
        ].join("\n"),
        // fs rebinds the present fields as locals (the chunk + scaffold read these names). renormalize the
        // normal — linear interpolation of a unit vector across the triangle denormalizes it.
        fromOut: [
            `    let eid = fin.eid;`,
            `    let world = fin.world;`,
            `    let worldNormal = normalize(fin.worldNormal);`,
            ...(needsUv ? [`    let uv = fin.uv;`] : []),
            ...(needsLocal ? [`    let localPos = fin.localPos;`] : []),
        ].join("\n"),
    };
}

/**
 * the WGSL module for a surface: the shared frame/view/lighting + vertex pull, then the surface's bindings,
 * vs chunk, and fs chunk (which writes `col`). Two `pass` modules: `"color"` carries the real group-1 shadow
 * bindings + the color fs; `"prepass"` carries the prepass entry points with shadow **stubs** (see
 * SHADOW_STUB_WGSL: a surface chunk's `lit()` statically reaches the shadow samplers, and the prepass
 * pipelines bind group 0 alone). `variant` is the material map-set a specializing surface compiles a pipeline
 * per (the glTF importer; `surface.specialize(variant)` overrides the preamble/fs); a non-specializing surface
 * ignores it. Pure: exported for structural tests
 */
export function surfaceCode(
    surface: Surface,
    pass: "color" | "prepass" = "color",
    variant = 0,
): string {
    // a specializing surface splices the variant's preamble / fs (Bevy's on-demand `specialize`); its own
    // `preamble` / `fs` are the fallback. The glTF importer returns only a `preamble` (the map-set helpers
    // for `variant`), so the surface's own `fs` stands; a surface with no `specialize` ignores `variant`.
    const spec = surface.specialize?.(variant);
    const preamble = spec?.preamble ?? surface.preamble ?? "";
    const fsChunk = spec?.fs ?? surface.fs ?? "";
    const built = builtinFields(fsChunk);
    const i = interp(surface.interpolators ?? {}, surface.name, built.count);
    const binds = surface.bindings ?? {};
    // the instancing convention: declaring both bindings opts the surface into sear's
    // standard per-instance transform (see INSTANCE_VS)
    const instanced = !!(binds.eids && binds.transforms);
    // a screen-space surface (lines) projects its own endpoints: the vs chunk writes `clipPos` and
    // sear emits `out.clip = clipPos`. A world-space surface (the default) gets `view.viewProj * world`
    // computed after the chunk, so a chunk that displaces `world` projects correctly
    const screen = surface.screen === true;
    const decls = Object.entries(binds)
        .map(([n, b], k) => bindingDecl(n, b, k + SURFACE_BASE))
        .join("\n");
    // a surface that binds an f16 storage element needs the directive, which must lead the module. No
    // engine surface does — sear's `material` mirror binds `vec2<u32>` and decodes with the core
    // `unpack2x16float` — so this is the consumer opt-in path: a surface declaring an f16 element must
    // also declare `shader-f16` in its plugin's `Plugin.features`, since it is not on the platform floor.
    const enableF16 = Object.values(binds).some(
        (b) => "element" in b && (b.element ?? "").includes("f16"),
    );
    return /* wgsl */ `${enableF16 ? "enable f16;\n" : ""}
${frameWgsl()}
${viewWgsl()}
${lightingWgsl()}
${pointLightsWgsl()}
${lightEvalWgsl()}
${pbrWgsl()}
${LEGACY_LIGHT_WGSL}
${uniformWgsl(pass)}
${octEncodeWgsl()}
${posQuantWgsl()}
${MESH_QUANT_WGSL}
${xformWgsl()}
${ldrColorUnpackWgsl()}
${pass === "color" ? shadowWgsl() : SHADOW_STUB_WGSL}
${decls}
${preamble}
struct VertexOut {
    @builtin(position) clip: vec4<f32>,
${built.struct}
${i.out}
}

@vertex
fn vs(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iid: u32) -> VertexOut {
${
    // color decodes the full quantized vertex (pos + oct normal + uv) from the 16 B main stream; the
    // prepass decodes only position from the 8 B stream (depth needs no normal / uv — defaulted), both
    // dequantizing against the meshId-selected MeshQuant. A surface vs chunk may still override any of them
    pass === "color"
        ? `    let v = vertices[vidx];
    let mq = meshQuant[meshIdOf(v.y)];
    var localPos = decodePos(v.x, v.y, mq);
    var localNormal = octDecodeNormal(v.z);
    var uv = decodeUv(v.w, mq);`
        : `    let v = position[vidx];
    var localPos = decodePos(v.x, v.y, meshQuant[meshIdOf(v.y)]);
    var localNormal = vec3<f32>(0.0, 0.0, 1.0);
    var uv = vec2<f32>(0.0);`
}
    var eid: u32 = 0u;
    var world = vec4<f32>(localPos, 1.0);
    var worldNormal = localNormal;
${screen ? "    var clipPos = vec4<f32>(0.0);" : ""}
${i.decls}
${instanced ? `    { ${INSTANCE_VS} }` : ""}
${surface.vs ? `    { ${surface.vs} }` : ""}
    var out: VertexOut;
    out.clip = ${screen ? "clipPos" : "view.viewProj * world"};
${built.toOut}
${i.toOut}
    return out;
}

${pass === "color" ? colorFragment(fsChunk, built.fromOut, i.fromOut, instanced) : ""}
${
    // alpha writes no prepass lanes (no id, no depth-write, no shadow cast); every other mode emits one
    // prepass fragment per color-lane subset (the empty subset is depth-only — a fragment only for `clip`)
    pass === "color" || surface.blend === "alpha"
        ? ""
        : laneSubsets()
              .map((s) =>
                  prepassFragment(
                      fsChunk,
                      built.fromOut,
                      i.fromOut,
                      instanced,
                      surface.blend === "clip",
                      s,
                  ),
              )
              .join("\n")
}
`;
}

/**
 * the WGSL module for a background: frame/view/lighting uniforms + the background's own bindings, a
 * fullscreen-triangle VS at the reverse-Z far plane, and an fs that reconstructs the world-space view ray
 * `dir` per-pixel then splices the chunk (which writes the HDR `col: vec3<f32>`). `dir` comes from
 * `@builtin(position)` + `view.invViewProj` (the fog reconstruct), **not** an interstage interpolator:
 * the view ray is derivable from the pixel, so crossing it would waste a varying slot (gpu.md rule 9). Pure:
 * exported for structural tests.
 */
export function backgroundCode(bg: Background): string {
    const binds = bg.bindings ?? {};
    const decls = Object.entries(binds)
        .map(([n, b], k) => bindingDecl(n, b, k + BG_BASE))
        .join("\n");
    const enableF16 = Object.values(binds).some(
        (b) => "element" in b && (b.element ?? "").includes("f16"),
    );
    return /* wgsl */ `${enableF16 ? "enable f16;\n" : ""}
${frameWgsl()}
${viewWgsl()}
${lightingWgsl()}
@group(0) @binding(${FRAME}) var<uniform> frame: Frame;
@group(0) @binding(${VIEW}) var<uniform> view: View;
@group(0) @binding(${LIGHTING}) var<uniform> lighting: Lighting;
${decls}
${bg.preamble ?? ""}
struct VertexOut {
    @builtin(position) clip: vec4<f32>,
}

// a fullscreen triangle (the three corners are the vertex index — no vertex pull), emitted at the
// reverse-Z far plane (clip z = 0) so the depth-equal test admits only un-rendered (background) pixels
@vertex
fn vs(@builtin(vertex_index) vidx: u32) -> VertexOut {
    let c = vec2<f32>(f32((vidx << 1u) & 2u), f32(vidx & 2u));
    var out: VertexOut;
    out.clip = vec4<f32>(c * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs(fin: VertexOut) -> @location(0) vec4<f32> {
    // reconstruct the world-space view ray from the pixel + the inverse view-projection at the far plane
    // (reverse-Z far = 0) — projection-agnostic, and derived from @builtin(position) (gpu.md rule 9)
    let uv = fin.clip.xy / view.resolution;
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0);
    let far = view.invViewProj * vec4<f32>(ndc, 1.0);
    let dir = normalize(far.xyz / far.w - view.eye.xyz);
    var col = vec3<f32>(0.0);
    { ${bg.fs} }
    return vec4<f32>(col, 1.0);
}
`;
}

// the shared fragment interior every entry point splices: rebind varyings as locals, declare `col` (the
// shaded color) + every color lane's mutable local (`tag`, …), then splice the surface chunk which writes
// `col` and may override any lane local (symmetric with `col` — `tag` defaults to the instance's `eid`
// for an instanced surface and TAG_NONE otherwise, which terrain overrides to `capacity + cell`).
// `shaded` entries also sample the sun shadow inline (lit()/lightFactor multiply the sun term by it); the
// prepass entries discard without lighting and bind no shadow map, so they omit the sample. `col` is
// unused in the prepass entries, the lane locals in the color entry — the chunk is one source, spliced
// into each entry that returns a slice of its output
export function fragmentBody(
    fs: string,
    builtinFrom: string,
    fromOut: string,
    shaded: boolean,
    instanced: boolean,
): string {
    // project the fragment's world position into the shadow map and PCF-compare (1.0 when no light
    // casts — see sampleSunShadow); the sun term in lit()/lightFactor reads it
    const sun = shaded
        ? `    sunVisibility = sampleSunShadow(world, worldNormal);\n    fragWorld = world;\n    fragCoord = fin.clip;\n    pointScale = 1.0;\n`
        : "";
    const lanes = COLOR_LANES.map(
        (l) => `    var ${l.local}: ${l.type} = ${l.init(instanced)};`,
    ).join("\n");
    return `${builtinFrom}
${fromOut}
${sun}    var col = vec4<f32>(0.0, 0.0, 0.0, 1.0);
${lanes}
${fs ? `    { ${fs} }` : ""}`;
}

// the color fragment — one single-target shape for opaque, `clip`, and `alpha` alike (they differ
// only in pipeline blend + depth policy, never in fragment output). No forced lighting: surfaces
// shade via `lit` / `lightFactor`, or write a flat `col`. The lane locals (tag, …) are NOT returned
// here — they ride the prepass (see prepassFragment / PrepassSystem), so the color pass is the
// engine's one blendable, postfx-fed lane
export function colorFragment(
    fs: string,
    builtinFrom: string,
    fromOut: string,
    instanced: boolean,
): string {
    return /* wgsl */ `@fragment
fn fs(fin: VertexOut) -> @location(0) vec4<f32> {
${fragmentBody(fs, builtinFrom, fromOut, true, instanced)}
    return col;
}`;
}

// the prepass fragment for one color-lane subset (opaque + `clip` surfaces), spliced per subset by
// surfaceCode. Runs the surface chunk for its authored lane locals (and a `clip` chunk's `discard`),
// then returns the subset's lanes. The empty subset is position-only depth: a fragment is emitted only
// for a `clip` surface (to discard) — a plain opaque surface's empty-subset pipeline has no fragment
// stage. One lane returns a scalar at @location(0); two or more an MRT output struct. Binds group 0
// only — no shadow read (the prepass carries no lighting), so the group-1 bindings stay unreferenced
export function prepassFragment(
    fs: string,
    builtinFrom: string,
    fromOut: string,
    instanced: boolean,
    clip: boolean,
    lanes: ColorLane[],
): string {
    const body = fragmentBody(fs, builtinFrom, fromOut, false, instanced);
    if (lanes.length === 0) {
        if (!clip) return ""; // position-only depth — no fragment stage
        return /* wgsl */ `@fragment
fn ${prepassEntry(lanes)}(fin: VertexOut) {
${body}
}`;
    }
    if (lanes.length === 1) {
        const l = lanes[0];
        return /* wgsl */ `@fragment
fn ${prepassEntry(lanes)}(fin: VertexOut) -> @location(0) ${l.type} {
${body}
    return ${l.local};
}`;
    }
    const out = `PrepassOut${lanes.map((l) => l.name[0].toUpperCase() + l.name.slice(1)).join("")}`;
    return /* wgsl */ `struct ${out} {
${lanes.map((l, k) => `    @location(${k}) ${l.local}: ${l.type},`).join("\n")}
}
@fragment
fn ${prepassEntry(lanes)}(fin: VertexOut) -> ${out} {
${body}
    var out: ${out};
${lanes.map((l) => `    out.${l.local} = ${l.local};`).join("\n")}
    return out;
}`;
}

/**
 * the point-shadow atlas module for a surface (depth-only): one indirect draw per casting mesh, the VS
 * reading the **re-gathered** instance list: each combo (cube face / spot cone) culled independently
 * through the Part pack (its own depth-only view slot), then concatenated mesh-major into one contiguous
 * run + a per-instance combo index (`renderPointShadows`). The list is packed `(combo << COMBO_SHIFT) | eid`
 * and bound at the surface's `eids` lane (so no new storage binding past the 10-ceiling, gpu.md). The VS
 * reads each instance's (caster, face) from `comboMeta[combo]`, transforms by that combo's CPU-computed
 * viewProj (`faceVP[combo]`), and remaps clip XY into the face's atlas tile (the tile placement folded into
 * the viewProj, so the hardware does the divide + near-plane clip and the depth matches `pointShadowOf`'s
 * analytic receiver). The FS discards fragments outside the tile rect (seam bleed): depth-only, so a kept
 * fragment writes its face's depth into exactly its tile. Reuses the prepass group-0 (position-only stream +
 * the instance bindings, `eids` → the re-gathered list, SHADOW_STUB so no group-1 sampler), adds a
 * point-only group 1 (face viewProjs + combo meta + the caster rects, all uniforms). `screen` surfaces have
 * no atlas placement and `alpha` casts nothing: the caller compiles neither.
 */
export function pointShadowCode(surface: Surface, variant: number, cascade = false): string {
    const spec = surface.specialize?.(variant);
    const preamble = spec?.preamble ?? surface.preamble ?? "";
    const fsChunk = spec?.fs ?? surface.fs ?? "";
    const clip = surface.blend === "clip";
    // a clip cutout runs its chunk's discard so it casts a holed shadow — cross the varyings the chunk
    // reads (matching the prepass; the position-only decode means uv is 0, the same prepass limitation).
    // An opaque surface is the tile discard alone — no chunk, no builtin varyings
    const built = clip ? builtinFields(fsChunk) : { count: 0, struct: "", toOut: "", fromOut: "" };
    const customs = surface.interpolators ?? {};
    const i = interp(customs, surface.name, built.count);
    const tileLoc = built.count + Object.keys(customs).length;
    const binds = surface.bindings ?? {};
    const instanced = !!(binds.eids && binds.transforms);
    const decls = Object.entries(binds)
        .map(([n, b], k) => bindingDecl(n, b, k + SURFACE_BASE))
        .join("\n");
    const enableF16 = Object.values(binds).some(
        (b) => "element" in b && (b.element ?? "").includes("f16"),
    );
    // point: one tile per (caster, face) — 6·casters slots, rect indexed slot·6 + face. cascade: one tile per
    // cascade — MAX_CASCADES slots, rect indexed by the cascade index (meta.x). Same VS, the count + index differ
    const slots = cascade ? MAX_CASCADES : 6 * pointCasters();
    const atlas = cascade ? cascadeAtlasSize(sunResolution(), sunCascades()) : pointAtlasSize();
    const rectExpr = cascade ? "m.x" : "m.x * 6u + m.y";
    return /* wgsl */ `${enableF16 ? "enable f16;\n" : ""}
${frameWgsl()}
${viewWgsl()}
${lightingWgsl()}
${pointLightsWgsl()}
${lightEvalWgsl()}
${pbrWgsl()}
${LEGACY_LIGHT_WGSL}
${uniformWgsl("prepass")}
${octEncodeWgsl()}
${posQuantWgsl()}
${MESH_QUANT_WGSL}
${xformWgsl()}
${ldrColorUnpackWgsl()}
${SHADOW_STUB_WGSL}
${decls}
${preamble}
// each combo viewProj has its atlas tile placement folded in (tileTransform), so the VS needs no manual
// divide — it reads faceVP for the clip position and tileRects (indexed by comboMeta's caster·6+face) only
// for the tile-discard bounds. The receiver samples the same tileRects on the color pass's group 1.
// FaceVPs / ComboMeta / TileRects come from the schemas (shade.ts), config-folded to this atlas's slot
// count — this module's own independent resolve (not the shared spliceNs), so it stays self-contained: a
// point/cascade pass never shares a shader module with the color pass's own TileRects (casterWgsl's)
${tgpu.resolve([faceVPsSchema(slots), comboMetaSchema(slots), tileRectsSchema(slots)], { names: "strict" })}
@group(1) @binding(0) var<uniform> faceVP: FaceVPs;
@group(1) @binding(1) var<uniform> comboMeta: ComboMeta;
@group(1) @binding(2) var<uniform> tileRects: TileRects;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
${built.struct}
${i.out}
    // the tile's pixel rect (origin.xy, size, _) for the seam discard — the tile sizes vary per caster now
    @location(${tileLoc}) @interpolate(flat) tileBox: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    let v = position[vidx];
    var localPos = decodePos(v.x, v.y, meshQuant[meshIdOf(v.y)]);
    var localNormal = vec3<f32>(0.0, 0.0, 1.0);
    var uv = vec2<f32>(0.0);
    // the re-gathered instance list (bound at the eids slot): one entry per (combo, surviving member),
    // packed eid in the low ${COMBO_SHIFT} bits + the dense combo index above. instance_index starts at the
    // record's firstInstance (indirect-first-instance, base floor), so it indexes the mesh's run directly.
    // comboMeta maps the combo to its (caster slot, face); faceVP[combo] carries the tile-folded projection
    let packed = eids[iid];
    var eid: u32 = packed & ${EID_MASK}u;
    let combo = packed >> ${COMBO_SHIFT}u;
    let xf = transforms[eid];
    var world = vec4<f32>(xformPoint(xf, localPos), 1.0);
    var worldNormal = xformNormal(xf, localNormal);
${i.decls}
${surface.vs ? `    { ${surface.vs} }` : ""}
    let m = comboMeta.m[combo];
    let rect = tileRects.rects[${rectExpr}]; // the allocated atlas-UV rect (point: caster·6+face; cascade: index)
    var out: VertexOut;
    // the combo viewProj has its atlas tile placement folded in (tileTransform), so this projects straight
    // into the tile — the hardware does the perspective divide AND the near-plane clip. A manual fc.xy/fc.w
    // here can't clip: a triangle behind the face near plane (fc.w ≤ 0) divides to garbage and lands in-tile
    out.clip = faceVP.m[combo] * world;
    out.tileBox = vec4<f32>(rect.xy * ${atlas}.0, rect.z * ${atlas}.0, 0.0);
${built.toOut}
${i.toOut}
    return out;
}

@fragment
fn fsPoint(fin: VertexOut) {
    // tile-seam discard: a triangle outside its face frustum remaps outside the tile (but inside the
    // atlas, so the hardware doesn't clip it) — discard it so each tile gets exactly its face's depth
    let p = fin.clip.xy;
    let mn = fin.tileBox.xy;
    let sz = fin.tileBox.z;
    if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) { discard; }
${clip ? fragmentBody(fsChunk, built.fromOut, i.fromOut, false, instanced) : ""}
}
`;
}
