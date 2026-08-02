// Sear — the one kitchen renderer. A GPU-driven raster *forward* pass (Aaltonen-Haar / niagara
// submission spine, primary visibility only) with sun shadows sampled inline in the FS, matching Bevy's
// clustered-forward shape. One renderer, one plugin (`SearPlugin`), no layers behind seams: one color
// pass (opaque draws then `blend` draws composited over them in a single `beginRenderPass`), an
// opt-in single-sample **prepass** emitting per-camera lanes (the `Tag` / `Depth` markers, Bevy's
// `DepthPrepass` / `NormalPrepass` shape), and sun shadows (the `Shadow` component on a directional
// light) are all sear-internal features, gated by data the way Bevy gates a shadow map on light data —
// not composed plugins coordinating through a singleton.
//
// Sun shadows: the CPU/ECS half (the off-screen light camera + placement) lives in ./shadows; the GPU half
// (the shadow map, its render through sear's compiled prepass depth pipelines, and the group-1 binding the
// FS samples) lives in ./atlas. This file owns the WGSL-scaffold-agnostic renderer plumbing: components +
// registries, per-draw bind-group resolution, per-camera targets, the systems, and the plugin — the pure
// codegen lives in ./codegen, pipeline compilation in ./pipelines. Sear renders its own map and reads its
// own shadow state directly — nothing publishes into it. Add a `Shadow` to the sun to cast; omit it for the
// fully-lit bare path (no map allocated), exactly like a camera without a lane marker runs no prepass.

import type { TgpuBindGroupLayout } from "typegpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { Plugin, State, System } from "../../engine";
import {
    Compute,
    capacity,
    f16x4,
    laneAlias,
    Registry,
    sparse,
    u32,
    unpackColor,
} from "../../engine";
import { precompile } from "../../engine/runtime";
import { unpackLdrColor, Xform } from "../../engine/utils/core";
import { GlazeSystem } from "../glaze";
import { Camera, RenderPlugin } from "../render";
import type { Binding, Draw, View } from "../render/core";
import {
    BeginFrameSystem,
    Draws,
    Frame,
    LightCull,
    Lighting,
    Meshes,
    Render,
    Surfaces,
    Views,
} from "../render/core";
import { SlabPlugin, slab } from "../slab";
import {
    cascadeRegather,
    disposeShadowAtlas,
    pointRegather,
    renderCascades,
    renderPointShadows,
    resetShadowAtlas,
    setPointFrames,
    shadowGroup,
    shadowGroupTyped,
    shadowLayoutTyped,
    shadowReady,
} from "./atlas";
import {
    BG_BASE,
    COLOR_LANES,
    type ColorLane,
    DEPTH_FORMAT,
    FRAME,
    LIGHT_GRID,
    LIGHT_INDICES,
    LIGHTING,
    laneKey,
    MESH_QUANT,
    POINT_LIGHTS,
    SAMPLE_COUNT,
    SURFACE_BASE,
    Tag,
    VERTICES,
    VIEW,
} from "./codegen";
import {
    fsCtxSchema,
    type TypedBackground,
    TypedBackgrounds,
    type Binding as TypedContractBinding,
    type TypedSurface,
    TypedSurfaces,
    surfaceLayout as typedLayout,
    registerSurface as typedRegister,
    VsIn,
    vsPatchSchema,
} from "./contract";
import { engineLayout, litPbr } from "./engine";
import {
    type BindResource,
    bgQuant,
    type Compiled,
    type CompiledBg,
    type CompiledTyped,
    type CompiledTypedBg,
    clearBackgrounds,
    clearGroups,
    compileTypedBg,
    compileTypedVariant,
    ensureSingle,
    ensureTypedSingle,
    ensureVariant,
    type GroupEntry,
    getBackground,
    getCompiled,
    getCompiledTyped,
    getGroup,
    getTypedBg,
    getTypedGroup,
    knownTypedVariants,
    preparePipelines,
    resetPipelineCaches,
    setGroup,
    setTypedGroup,
    type TypedGroupEntry,
    typedEngineGroup,
} from "./pipelines";
import { prepareRegather } from "./regather";
import { checkShadowConfig, Pbr } from "./shade";
import {
    cascadeCount,
    destroyCascades,
    destroyPointShadows,
    MAX_CASCADES,
    pointCasters,
    resetCascades,
    resetPointShadows,
    SHADOW_DEFAULTS,
    Shadow,
    updateCascades,
    updatePointShadows,
} from "./shadows";

export { DEPTH_FORMAT, TAG_FORMAT, TAG_NONE, Tag } from "./codegen";

/**
 * marker selecting Sear as the active renderer on a Camera entity. A camera carrying it renders through
 * sear's color pass, plus the opt-in prepass lanes its {@link Tag} / {@link Depth} markers request.
 * Lives in the renderer impl with the systems that query it; the thin `sear` barrel re-exports it to the
 * game author, `sear/core` re-exports the systems to an extender.
 *
 * @example
 * ```
 * <a camera sear transform />
 * ```
 */
export const Sear = {};

/**
 * opt a Sear camera into the **depth lane**: the prepass *stores* its single-sample depth and publishes
 * it as `view.depth` (without this marker the prepass depth is discarded, never reaching main memory).
 * Requestable on its own (a depth-only consumer needs no id) or alongside {@link Tag} (one prepass writes
 * both). Bevy's `DepthPrepass`. A screen-space consumer (AO, fog, volumetrics) adds it to read
 * `view.depth`.
 *
 * @example
 * ```
 * <a camera sear depth transform />
 * ```
 */
export const Depth = {};

/**
 * per-entity PBR material the `default` / `vertex` surfaces read (alongside `Color`, the base albedo).
 * One slab `Quad` published as `"material"`, lanes `(metallic, roughness, emissive, occlusion)`:
 * `metallic` and `roughness` are the metallic-roughness knobs ([0,1]); `emissive` is a glow **strength**
 * tinting the base color (`Color.rgb * emissive`); `occlusion` dims ambient ([0,1]). Defaults are flat
 * (metallic 0, roughness 1, emissive 0, occlusion 1), so a Part without it shades exactly like the
 * pre-PBR diffuse `lit`. Independent (non-tinted) emissive + texture-driven maps are the glTF importer's
 * job (it drives sear's `litPbr` from its own per-material palette).
 *
 * @example
 * ```
 * <a part material="metallic: 1; roughness: 0.2" transform />
 * ```
 */
export const Material = {
    /** the four PBR lanes `(metallic, roughness, emissive, occlusion)`, authored named via the `material` attribute (`material="metallic: 1; roughness: 0.2"`). */
    params: slab(f16x4, "material"),
};

const MATERIAL_FLAT: [number, number, number, number] = [0, 1, 0, 1];

const MaterialTraits = {
    defaults: () => ({ params: MATERIAL_FLAT }),
    aliases: { params: laneAlias("params", ["metallic", "roughness", "emissive", "occlusion"]) },
};

// base every slot to the flat material so a Part lacking the Material component shades like the pre-PBR
// diffuse default (an entity with Material overwrites its slot via the trait default on add). Mirrors
// Part.initPart's magenta Color base; the pack gates each slot on membership, so stale slots never draw.
function initMaterial(): void {
    for (let i = 0; i < capacity; i++) Material.params.set(i, ...MATERIAL_FLAT);
}

/**
 * a registered background: a renderer-agnostic *view-ray → HDR color* recipe sear draws as a fullscreen
 * backdrop on the un-rendered pixels (the standard infinite-skybox technique). `fs` is a WGSL chunk that
 * writes the HDR color into `col: vec3<f32>` from a normalized world-space view ray `dir` (sear
 * reconstructs it per-pixel from `view.invViewProj`), with read access to `view`, `lighting`, `frame`, and
 * any declared `bindings`. `preamble` is an optional module-scope WGSL chunk (helpers / structs /
 * constants the `fs` calls). Modeled on {@link Surface}, but backdrop-only: no mesh, instancing,
 * interpolators, or blend modes; the engine names no sky concept, a plugin owns its own sky math.
 */
export interface Background {
    name: string;
    bindings?: Record<string, Binding>;
    preamble?: string;
    fs: string;
}

/** every registered background, keyed by name with a stable numeric ID; cleared on `SearPlugin.initialize` */
export const Backgrounds: Registry<Background> = new Registry<Background>();

function isTypedBg(spec: Background | TypedBackground): spec is TypedBackground {
    // the same two-part discriminant `contract.ts`'s `isTyped` uses for surfaces: `layout` presence alone
    // would misroute a legacy spec carrying an incidental `layout` key, so `fs`'s shape is the load-bearing
    // second conjunct (legacy `fs` is always a WGSL string, typed `fs` always a TgpuFn).
    return "layout" in spec && typeof spec.fs !== "string";
}

/**
 * the background dual-accept shim (4a-ii-c-3a-4, `contract.ts`'s `register()` precedent for surfaces):
 * accepts either the legacy string-contract {@link Background} or a typed spec, discriminated the same way.
 * Lives here rather than in `contract.ts` because the legacy `Background` type + the real `Backgrounds`
 * registry it delegates to are owned by this module, which already imports `contract.ts` — the reverse
 * import would cycle. A legacy spec delegates straight to `Backgrounds.register` (unchanged behavior); a
 * typed spec lands in `contract.ts`'s `TypedBackgrounds` for a future consumer conversion to compile
 * against. Sanctioned until the string path retires.
 *
 * @example
 * registerBackground({ name: "gradient", fs: "col = vec3<f32>(dir.y);" }); // legacy
 * registerBackground({ name: "typed-gradient", layout, fs }); // typed
 */
export function registerBackground<B extends Record<string, TypedContractBinding>>(
    spec: TypedBackground<B>,
): number;
export function registerBackground(spec: Background): number;
export function registerBackground(spec: Background | TypedBackground): number {
    if (isTypedBg(spec)) return TypedBackgrounds.register(spec);
    return Backgrounds.register(spec);
}

/**
 * select a Sear camera's backdrop: the {@link Backgrounds} recipe drawn behind the scene as a fullscreen
 * view-ray → color fill on the un-rendered pixels. Without it the camera shows the flat `Camera.clearColor`
 * (the opt-in fallback). The recipe is registered in code (`Backgrounds.register`); this picks one per
 * camera by name.
 *
 * @example
 * ```
 * <a camera sear backdrop="name: gradient" transform />
 * ```
 */
export const Backdrop = {
    /** the registered background drawn behind the scene (selected by name) */
    name: sparse(u32),
};

// name ↔ Backgrounds-id at scene parse / format, the PartTraits surface pattern (id stored, name authored).
// The two registries are separate id spaces during dual-accept, so a typed background's id stores offset
// by BACKDROP_TYPED_BASE — an unambiguous discriminant (the legacy registry can never reach 2^31 entries),
// where consulting both registries at the same raw id could misroute
const BACKDROP_TYPED_BASE = 0x80000000;
const BackdropTraits = {
    parse: {
        name: (value: string) => {
            const legacy = Backgrounds.id(value);
            if (legacy !== undefined) return legacy;
            const typed = TypedBackgrounds.id(value);
            return typed !== undefined ? typed + BACKDROP_TYPED_BASE : undefined;
        },
    },
    format: {
        name: (value: number) =>
            value >= BACKDROP_TYPED_BASE
                ? TypedBackgrounds.name(value - BACKDROP_TYPED_BASE)
                : Backgrounds.name(value),
    },
};

// a draw resolving to null is a silent skip — usually a typo'd binding or an
// unpublished resource. Warn once per draw so it's visible without spamming
const _warned = new Set<string>();

function warnSkip(draw: string, cause: string): null {
    if (!_warned.has(draw)) {
        _warned.add(draw);
        console.warn(`sear: draw "${draw}" skipped — ${cause}`);
    }
    return null;
}

const registryFor = (type: Binding["type"]) =>
    type.startsWith("texture")
        ? Compute.textures
        : type.startsWith("sampler")
          ? Compute.samplers
          : Compute.buffers;

// shape a resolved resource into a bind-group entry resource by binding type: a 2d-array texture binds a
// 2d-array view, any other texture its default view, a sampler itself, a buffer wrapped
const bindResource = (type: Binding["type"], res: BindResource): GPUBindingResource =>
    type === "texture-2d-array"
        ? (res as GPUTexture).createView({ dimension: "2d-array" })
        : type.startsWith("texture")
          ? (res as GPUTexture).createView()
          : type.startsWith("sampler")
            ? (res as GPUSampler)
            : { buffer: res as GPUBuffer };

type RecordedLegacy = {
    // the compiled surface — the color pass reads `color`/`transparent` (or the single-sample `single`
    // twin for a no-AA camera); the prepass + shadow passes read `prepass`. Carried whole so the per-camera
    // AA selection happens at draw time without baking each pipeline ref per draw
    c: Compiled;
    // the per-draw group-0 state: the shared bindings (everything but slot 3, the vertex stream, and
    // VIEW) plus the two per-slot caches `colorGroup`/`prepassGroup` build against — one whole
    // `Render.viewBuffers[slot]` buffer per bind, no dynamic offset (Approach 4a's per-slot-buffer design)
    entry: GroupEntry;
    // the mesh's index buffer, bound via setIndexBuffer before each drawIndexedIndirect (geometry pulls
    // vertices from the storage binding, but the hardware index buffer drives vertex reuse)
    index: GPUBuffer;
};

// a draw resolved through the typed contract (4a-ii-c-3b): the compiled typed pipeline set + the
// per-draw group-2 state (engine group 0 resolves per slot at draw time via `typedEngineGroup`;
// group 1 is the pass's — `shadowGroupTyped` for color, the atlas layouts' own for point/cascade)
type RecordedTyped = { t: CompiledTyped; g: TypedGroupEntry; index: GPUBuffer };

// the two shapes discriminate on `"c" in r` — a draw is exactly one of them, decided by `record()`'s
// typed-registry-first lookup (the built-in flip: a name registered in both draws typed)
export type Recorded = RecordedLegacy | RecordedTyped;

/**
 * the color pass's group 0 for a recorded draw at a given view slot (slot 3 = the 16 B main vertex
 * stream, VIEW = `Render.viewBuffers[slot]`) — lazily built and cached per slot on the draw's
 * {@link GroupEntry} (typically one entry: a scene usually has one shading camera).
 */
function colorGroup(r: RecordedLegacy, slot: number): GPUBindGroup {
    return slotGroup(r.entry, r.entry.main, r.entry.colorCache, slot, `sear-color-${slot}`);
}

/**
 * the prepass + shadow-map pass's group 0 for a recorded draw at a given view slot (slot 3 = the 8 B
 * position-only stream). Same per-slot caching as {@link colorGroup}.
 */
function prepassGroupOf(r: RecordedLegacy, slot: number): GPUBindGroup {
    return slotGroup(r.entry, r.entry.position, r.entry.prepassCache, slot, `sear-prepass-${slot}`);
}

function slotGroup(
    entry: GroupEntry,
    vertex: GPUBuffer,
    cache: Map<number, GPUBindGroup>,
    slot: number,
    label: string,
): GPUBindGroup {
    const cached = cache.get(slot);
    if (cached) return cached;
    const group = Compute.device.createBindGroup({
        label,
        layout: entry.layout,
        entries: [
            { binding: VERTICES, resource: { buffer: vertex } },
            { binding: VIEW, resource: { buffer: Render.viewBuffers[slot] } },
            ...entry.shared,
        ],
    });
    cache.set(slot, group);
    return group;
}

/**
 * the color + transparent + per-lane-set prepass pipelines and the bind-group state sear records a draw
 * with, or null to skip it. All pipelines share one bind group (same group-0 layout). A surface with
 * no compiled pipeline isn't sear's (silent skip); a missing mesh or unpublished binding warns once.
 * The per-slot bind groups cache per draw, rebuilt only on a resource identity change; the fixed uniforms
 * are stable, so untracked
 */
function record(draw: Draw): Recorded | null {
    // the typed registry wins (the built-in flip, 4a-ii-c-3b): a surface registered in both — the
    // built-ins during dual-accept — draws through the typed path in EVERY pass; a partial (per-pass)
    // flip would break shadow casting. The string registration stays for Part's automatic draw
    // registration + any legacy consumer reading `Surfaces`
    const typed = TypedSurfaces.get(draw.surface);
    if (typed) return recordTyped(draw, typed);
    const surface = Surfaces.get(draw.surface);
    if (!surface) return null; // not a sear surface — silent skip
    const mesh = Meshes.get(draw.mesh);
    if (!mesh) return warnSkip(draw.name, `mesh "${draw.mesh}" not registered`);
    // a specializing surface (glTF) selects the pipeline for this mesh's material map-set; every other
    // surface is variant 0. The variant is constant per mesh, so the cached bind group (variant-invariant)
    // stays valid across frames
    const variant = surface.specialize ? (mesh.variant ?? 0) : 0;
    const c = getCompiled(draw.surface, variant);
    if (!c) {
        ensureVariant(surface, variant); // kick off the lazy compile; skip the draw until it lands
        return null;
    }

    // a mesh registered before the quantized format (or by an un-migrated producer) has no position /
    // quant stream — skip it loudly rather than bind a garbage decode
    if (!mesh.position || !mesh.quant)
        return warnSkip(draw.name, `mesh "${draw.mesh}" has no quantized position/quant stream`);
    const main = mesh.vertices;
    const position = mesh.position;
    const quant = mesh.quant;

    // resolve the surface bindings to live resources by type (geometry first: main, position, quant, index).
    // A per-mesh override (`mesh.bindings`, e.g. a skinned mesh's own VAT) wins over the published global, so
    // meshes needing distinct resources for the same binding name (one VAT per skinned mesh) each bind their own
    const resources: BindResource[] = [main, position, quant, mesh.indices];
    for (const { name, type } of c.slots) {
        const res = mesh.bindings?.[name] ?? registryFor(type).get(name);
        if (!res) return warnSkip(draw.name, `binding "${name}" (${type}) not published`);
        resources.push(res);
    }

    const prev = getGroup(draw.name);
    if (
        prev &&
        prev.resources.length === resources.length &&
        prev.resources.every((b, k) => b === resources[k])
    ) {
        return { c, entry: prev, index: mesh.indices };
    }

    // the bindings shared by both groups (everything but slot 3, the vertex stream, and VIEW)
    const shared: GPUBindGroupEntry[] = [
        { binding: FRAME, resource: { buffer: Frame.buffer } },
        { binding: LIGHTING, resource: { buffer: Lighting.buffer } },
        { binding: POINT_LIGHTS, resource: { buffer: LightCull.lights! } },
        { binding: LIGHT_GRID, resource: { buffer: LightCull.grid! } },
        { binding: LIGHT_INDICES, resource: { buffer: LightCull.indices! } },
        { binding: MESH_QUANT, resource: { buffer: quant } },
    ];
    // textures bind a default view — cache-miss path only, never per-frame on a hit
    c.slots.forEach(({ type }, k) => {
        shared.push({ binding: k + SURFACE_BASE, resource: bindResource(type, resources[k + 4]) });
    });

    const entry: GroupEntry = {
        main,
        position,
        layout: c.layout,
        shared,
        colorCache: new Map(),
        prepassCache: new Map(),
        pointGroup: null,
        cascadeGroup: null,
        resources,
    };

    // a shadow-atlas group 0: the prepass group with the `eids` lane bound to a re-gathered packed instance
    // list (the point atlas's or the cascade atlas's), and VIEW bound to slot 0's buffer as an unread
    // placeholder (the point/cascade VS projects by its own tile viewProj, never `view`). Reusing the eids
    // lane keeps the atlas pipelines at zero new storage bindings, so the heaviest surfaces stay within the
    // 10-per-stage ceiling (gpu.md). Built only for a casting surface, and only once that atlas's packed
    // list exists (its alloc clears the group cache, rebuilding)
    const eidsK = c.slots.findIndex((s) => s.name === "eids");
    const eidsSwap = (
        pipe: GPURenderPipeline | null,
        listEids: GPUBuffer | null,
    ): GPUBindGroup | null => {
        if (!pipe || !listEids || eidsK < 0) return null;
        const entries: GPUBindGroupEntry[] = [
            { binding: VERTICES, resource: { buffer: position } },
            { binding: VIEW, resource: { buffer: Render.viewBuffers[0] } },
            ...shared,
        ].map((e) =>
            e.binding === SURFACE_BASE + eidsK
                ? { binding: e.binding, resource: { buffer: listEids } }
                : e,
        );
        return Compute.device.createBindGroup({
            label: `sear-shadowcast-${draw.name}`,
            layout: c.layout,
            entries,
        });
    };
    entry.pointGroup = eidsSwap(c.point, pointRegather.eids());
    entry.cascadeGroup = eidsSwap(c.cascade, cascadeRegather.eids());
    setGroup(draw.name, entry);
    return { c, entry, index: mesh.indices };
}

// resolve a typed layout's own bindings (never the sear-injected `vertices`) to live resources by the
// entry's kind — the typed twin of `record`'s `registryFor`/`bindResource` walk, with the same per-mesh
// override + warn-once skip semantics. Returns the createBindGroup value record + the identity list, or
// the missing binding's name
function typedResources(
    entries: Record<string, object>,
    override?: Record<string, BindResource>,
): { values: Record<string, unknown>; resources: BindResource[] } | string {
    const values: Record<string, unknown> = {};
    const resources: BindResource[] = [];
    for (const [name, entry] of Object.entries(entries)) {
        if (name === "vertices") continue;
        const registry =
            "texture" in entry
                ? Compute.textures
                : "sampler" in entry
                  ? Compute.samplers
                  : Compute.buffers;
        const res = override?.[name] ?? registry.get(name);
        if (!res) return name;
        resources.push(res);
        // a texture binds a view of the schema's own dimension (the legacy `bindResource` shape)
        values[name] =
            "texture" in entry
                ? (res as GPUTexture).createView({
                      dimension: (entry as { texture: { dimension: GPUTextureViewDimension } })
                          .texture.dimension,
                  })
                : res;
    }
    return { values, resources };
}

/**
 * the typed twin of {@link record}: compiled typed pipelines + the per-draw group-2 state (the two
 * layout-object caches the c-2 verdict names — `color` against `layout`; opaque depth-side groups against
 * `layout.depthVariant`; clip depth-side groups against the full layout so cutoff sees material UVs —
 * plus the atlas `eids` swaps and the slot-0 engine group the atlas passes bind).
 */
function recordTyped(draw: Draw, surface: TypedSurface): Recorded | null {
    const mesh = Meshes.get(draw.mesh);
    if (!mesh) return warnSkip(draw.name, `mesh "${draw.mesh}" not registered`);
    if (!mesh.position || !mesh.quant)
        return warnSkip(draw.name, `mesh "${draw.mesh}" has no quantized position/quant stream`);
    const variant = surface.specialize ? (mesh.variant ?? 0) : 0;
    let t = getCompiledTyped(surface.name, variant);
    if (!t || t.owner !== surface || t.layout !== surface.layout) {
        // registered after warm (`preparePipelines` compiles the rest) — sync, so no skip frame; a
        // throwing compile (a contract guard, or shader/device validation) must not take down the frame
        // loop, so it degrades to the warn-once skip
        try {
            t = compileTypedVariant(surface, variant);
        } catch (e) {
            return warnSkip(draw.name, `typed surface "${surface.name}" failed to compile: ${e}`);
        }
    }

    const resolved = typedResources(
        surface.layout.entries as Record<string, object>,
        mesh.bindings as Record<string, BindResource> | undefined,
    );
    if (typeof resolved === "string")
        return warnSkip(draw.name, `binding "${resolved}" not published`);
    const pointList = pointRegather.eids();
    const cascadeList = cascadeRegather.eids();
    // geometry + the atlas packed lists join the identity check (a re-gather realloc also clears the
    // whole cache via `clearGroups` — the lists here make the entry self-consistent even without it)
    const resources: BindResource[] = [
        mesh.vertices,
        mesh.position,
        mesh.quant,
        mesh.indices,
        ...resolved.resources,
    ];
    if (pointList) resources.push(pointList);
    if (cascadeList) resources.push(cascadeList);

    const prev = getTypedGroup(draw.name, surface);
    if (
        prev &&
        prev.resources.length === resources.length &&
        prev.resources.every((b, k) => b === resources[k])
    ) {
        return { t, g: prev, index: mesh.indices };
    }

    const root = Compute.root;
    // the two layout objects share one loose signature here — the color/depth `vertices` element split
    // is real at authoring time, but a bind group takes raw buffers either way (the `layout.$` cast class)
    const group = (lay: unknown, vertices: GPUBuffer, override?: Record<string, BindResource>) =>
        root.unwrap(
            root.createBindGroup(
                lay as TgpuBindGroupLayout,
                {
                    ...resolved.values,
                    ...override,
                    vertices,
                } as never,
            ),
        );
    const engineCache = new Map<number, GPUBindGroup>();
    const clip = surface.blend === "clip";
    const depthLayout = clip ? surface.layout : surface.layout.depthVariant;
    const depthVertices = clip ? mesh.vertices : mesh.position;
    const entry: TypedGroupEntry = {
        owner: surface,
        layout: surface.layout,
        quant: mesh.quant,
        color: group(surface.layout, mesh.vertices),
        // `alpha` compiles no depth-side pipelines, so it needs no depth-shape groups
        depth: surface.blend === "alpha" ? null : group(depthLayout, depthVertices),
        point: t.point && pointList ? group(depthLayout, depthVertices, { eids: pointList }) : null,
        cascade:
            t.cascade && cascadeList
                ? group(depthLayout, depthVertices, { eids: cascadeList })
                : null,
        engineCache,
        atlasG0: typedEngineGroup(engineCache, 0, mesh.quant),
        resources,
    };
    setTypedGroup(draw.name, entry);
    return { t, g: entry, index: mesh.indices };
}

/**
 * the frame's draw list: every registered {@link Draw} with a compiled surface + published
 * bindings, paired with its cached group-0 state. Camera-independent (the per-slot bind groups it builds
 * against are cached lazily by slot, not baked per camera), so {@link PrepassSystem}
 * resolves it once per frame into `_frameDraws` and the prepass, shadow map, and color pass all
 * render every camera against that one list
 */
function resolveDraws(): { draw: Draw; r: Recorded }[] {
    const items: { draw: Draw; r: Recorded }[] = [];
    for (const draw of Draws.values()) {
        const r = record(draw);
        if (r) items.push({ draw, r });
    }
    return items;
}

const _depth = new Map<
    number,
    { texture: GPUTexture; view: GPUTextureView; w: number; h: number }
>();

// the per-camera single-sample depth the prepass writes — always the front-most-fragment test the id
// lane needs, but only *stored* + published as `view.depth` when the camera carries `Depth` (else the
// store is discarded). Allocated when the prepass runs (any lane marker). TEXTURE_BINDING so a
// screen-space consumer (AO, volumetrics) can sample it the same frame
function depthView(eid: number, w: number, h: number): GPUTextureView {
    const cached = _depth.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached.view;
    cached?.texture.destroy();
    const texture = Compute.device.createTexture({
        label: `sear-depth-${eid}`,
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const view = texture.createView();
    _depth.set(eid, { texture, view, w, h });
    return view;
}

const _laneTargets = new Map<
    string,
    { texture: GPUTexture; view: GPUTextureView; w: number; h: number }
>();

// the per-(camera, color-lane) screen-space target, sibling to depthView — filled by the prepass.
// Sized to the framebuffer, recreated on resize; the format + usage are the lane's (the id lane is
// r32uint + COPY_SRC for a hover readback + TEXTURE_BINDING for an outline sample). Returns the texture
// (published onto `view.<lane>`) + the color-attachment view — one cache keyed `${eid}:${lane.name}`
// that drives both, so adding a lane needs no new allocator
function laneTarget(
    eid: number,
    lane: ColorLane,
    w: number,
    h: number,
): { texture: GPUTexture; view: GPUTextureView } {
    const key = `${eid}:${lane.name}`;
    const cached = _laneTargets.get(key);
    if (cached && cached.w === w && cached.h === h) return cached;
    cached?.texture.destroy();
    const texture = Compute.device.createTexture({
        label: `sear-${lane.name}-${eid}`,
        size: { width: w, height: h },
        format: lane.format,
        usage: lane.usage,
    });
    const entry = { texture, view: texture.createView(), w, h };
    _laneTargets.set(key, entry);
    return entry;
}

const _colorTargets = new Map<
    number,
    {
        color: GPUTexture | null;
        colorView: GPUTextureView | null;
        depth: GPUTexture;
        depthView: GPUTextureView;
        w: number;
        h: number;
        aa: boolean;
    }
>();

// the per-camera color-pass targets, by AA mode. AA on: a 4× MSAA color (resolved into the offscreen at
// pass end) + a 4× depth. AA off: no MSAA color (the pass renders straight into view.framebuffer) + a 1×
// depth. The color pass owns this depth (`less` + write, cleared each frame); the prepass + shadow map
// keep their own 1× depth (never cross-compared). Sized to the view + keyed on AA, recreated on resize/toggle
function colorTargets(
    eid: number,
    w: number,
    h: number,
    aa: boolean,
): { color: GPUTextureView | null; depth: GPUTextureView } {
    const cached = _colorTargets.get(eid);
    if (cached && cached.w === w && cached.h === h && cached.aa === aa)
        return { color: cached.colorView, depth: cached.depthView };
    cached?.color?.destroy();
    cached?.depth.destroy();
    const samples = aa ? SAMPLE_COUNT : 1;
    const color = aa
        ? Compute.device.createTexture({
              label: `sear-color-msaa-${eid}`,
              size: { width: w, height: h },
              format: Render.format,
              sampleCount: SAMPLE_COUNT,
              usage: GPUTextureUsage.RENDER_ATTACHMENT,
          })
        : null;
    const depth = Compute.device.createTexture({
        label: `sear-color-depth-${eid}`,
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        sampleCount: samples,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const entry = {
        color,
        colorView: color?.createView() ?? null,
        depth,
        depthView: depth.createView(),
        w,
        h,
        aa,
    };
    _colorTargets.set(eid, entry);
    return { color: entry.colorView, depth: entry.depthView };
}

// the geometry pass (one color target — the prepass lanes ride their own pass). AA on: the opaque draws
// clear + write `msaaColor`, the transparent draws blend over, and it resolves into the offscreen once at
// pass end (`discard` — the resolve fires regardless and nothing reads the MSAA target after). AA off:
// `msaaColor` is null — render straight into the offscreen, no resolve, **`store`** the result (`discard`
// would throw away the only copy → a black frame). The depth `discard`s either way (transient)
function beginColor(
    eid: number,
    msaaColor: GPUTextureView | null,
    depth: GPUTextureView,
    framebuffer: GPUTextureView,
    clear: ReturnType<typeof unpackColor>,
) {
    const clearValue = { ...clear, a: 1 };
    return Render.encoder!.beginRenderPass({
        label: `sear-color/${eid}`,
        timestampWrites: Compute.span?.("sear:color"),
        colorAttachments: [
            msaaColor
                ? {
                      view: msaaColor,
                      resolveTarget: framebuffer,
                      loadOp: "clear",
                      storeOp: "discard",
                      clearValue,
                  }
                : { view: framebuffer, loadOp: "clear", storeOp: "store", clearValue },
        ],
        depthStencilAttachment: {
            view: depth,
            depthLoadOp: "clear",
            depthStoreOp: "discard",
            depthClearValue: 0,
        },
    });
}

// set the pipeline + per-draw, per-slot group 0 and issue the indirect draw. Shared by the prepass and
// the color pass's opaque + transparent draws — they differ in pipeline and in which group-0 bind group
// (the color pass binds `colorGroup(r, slot)`, the prepass `prepassGroupOf(r, slot)`); the color pass
// additionally binds group 1, its caller's concern, not here
function bind(
    pass: GPURenderPassEncoder,
    pipeline: GPURenderPipeline,
    draw: Draw,
    r: RecordedLegacy,
    group: GPUBindGroup,
    slot: number,
): void {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.setIndexBuffer(r.index, "uint32");
    // per-view-culled producers lay DrawIndexedIndirect records out slot-major (`viewStride`
    // bytes/camera); a view-independent draw leaves it 0
    pass.drawIndexedIndirect(
        draw.args.indirect,
        (draw.args.offset ?? 0) + slot * (draw.args.viewStride ?? 0),
    );
}

/**
 * one camera's prepass (`sear:prepass`) recorded onto `Render.encoder`: a single single-sample pass
 * emitting the camera's opt-in lanes. It owns its own depth (cleared + `less` + write, so only the
 * front-most opaque / `clip` fragment writes each lane); that depth is *stored* + published as
 * `view.depth` when the camera carries {@link Depth}, otherwise discarded (TBDR: it stays in tile memory,
 * never reaching main RAM). Each requested color lane is one MRT attachment cleared to the lane's clear
 * value and published onto `view.<lane>` (today the id lane → `view.tag`). Binds group 0 only: no shadow
 * map, no lighting. `alpha` surfaces are excluded (a transparent pixel has no single owner). **One
 * prepass regardless of lane count**: the lane set selects the pipeline + the attachment list, not the
 * pass count; an empty draw list still clears every lane
 */
function renderPrepass(
    eid: number,
    view: View,
    items: { draw: Draw; r: Recorded }[],
    lanes: ColorLane[],
    storeDepth: boolean,
): void {
    if (!Render.encoder || !view.framebuffer) return;
    const depth = depthView(eid, view.width, view.height);
    const key = laneKey(lanes);
    const colorAttachments = lanes.map((lane) => {
        const target = laneTarget(eid, lane, view.width, view.height);
        lane.set(view, target.texture); // publish `view.<lane>`
        return {
            view: target.view,
            loadOp: "clear" as const,
            storeOp: "store" as const,
            clearValue: lane.clear,
        };
    });
    const pass = Render.encoder.beginRenderPass({
        label: `sear-prepass/${eid}`,
        timestampWrites: Compute.span?.("sear:prepass"),
        colorAttachments,
        depthStencilAttachment: {
            view: depth,
            depthLoadOp: "clear",
            // store the depth only for a `Depth` consumer; the id lane needs the test, not the stored
            // result, so a tag-only camera discards it (TBDR keeps it in tile memory)
            depthStoreOp: storeDepth ? "store" : "discard",
            depthClearValue: 0,
        },
    });
    let draws = 0;
    for (const { draw, r } of items) {
        if ("c" in r) {
            const pipe = r.c.prepass.get(key);
            if (pipe) {
                bind(pass, pipe, draw, r, prepassGroupOf(r, view.slot), view.slot);
                draws++;
            }
        } else {
            const pipe = r.t.prepass.get(key);
            if (pipe && r.g.depth) {
                // the depth-shape group registers under BOTH group-2 layout objects: a vs-chunk
                // surface's own reads resolve against `layout`, the depth vertex pull against
                // `layout.depthVariant`, and the resolution records only one of the two (structurally
                // identical, WebGPU-group-equivalent) at idx 2
                pipe.with(pass)
                    .with(engineLayout, typedEngineGroup(r.g.engineCache, view.slot, r.g.quant))
                    // inert group-1 fill (the stub receiver samples nothing) — the typed prepass
                    // pipelines declare it only to keep typegpu's group-indexed pipeline layout dense
                    .with(shadowLayoutTyped, shadowGroupTyped())
                    .with(r.g.layout, r.g.depth)
                    .with(r.g.layout.depthVariant, r.g.depth)
                    .withIndexBuffer(r.index, "uint32")
                    .drawIndexedIndirect(
                        draw.args.indirect,
                        (draw.args.offset ?? 0) + view.slot * (draw.args.viewStride ?? 0),
                    );
                draws++;
            }
        }
    }
    pass.end();
    Compute.indirect?.("sear:prepass", draws);
    view.depth = storeDepth ? depth : null;
}

// build (and cache) a background's group 0 for one view slot — frame / view (one whole per-slot buffer,
// no dynamic offset) / lighting + its own bindings resolved by name. Lazy, keyed on (bg × slot): the
// buffers are stable post-warm, so a slot builds once on first use and caches on the CompiledBg (a
// missing binding skips the draw)
function bgGroup(cb: CompiledBg, slot: number): GPUBindGroup | null {
    const cached = cb.groupCache.get(slot);
    if (cached) return cached;
    const entries: GPUBindGroupEntry[] = [
        { binding: FRAME, resource: { buffer: Frame.buffer } },
        { binding: VIEW, resource: { buffer: Render.viewBuffers[slot] } },
        { binding: LIGHTING, resource: { buffer: Lighting.buffer } },
    ];
    for (let k = 0; k < cb.slots.length; k++) {
        const { name, type } = cb.slots[k];
        const res = registryFor(type).get(name);
        if (!res)
            return warnSkip(`background:${cb.name}`, `binding "${name}" (${type}) not published`);
        entries.push({ binding: k + BG_BASE, resource: bindResource(type, res) });
    }
    const group = Compute.device.createBindGroup({
        label: `sear-bg-${cb.name}-${slot}`,
        layout: cb.layout,
        entries,
    });
    cb.groupCache.set(slot, group);
    return group;
}

// the camera's selected backdrop — legacy or typed, discriminated by the stored id's
// BACKDROP_TYPED_BASE offset (BackdropTraits) — or null (no `Backdrop` component, or its name isn't a
// compiled background). Membership-gated — a bare `Backdrop.name.get` reads 0 for a non-member, which
// would alias the first registered background, so the `state.has` check is what keeps the no-backdrop
// path on the clear
type BackdropPick = { cb: CompiledBg } | { bg: TypedBackground; ct: CompiledTypedBg };
function backdrop(state: State, eid: number): BackdropPick | null {
    if (!state.has(eid, Backdrop)) return null;
    const id = Backdrop.name.get(eid);
    if (id >= BACKDROP_TYPED_BASE) {
        const name = TypedBackgrounds.name(id - BACKDROP_TYPED_BASE);
        const bg = name ? TypedBackgrounds.get(name) : undefined;
        if (!bg) return null;
        const ct = getTypedBg(bg.name) ?? compileTypedBg(bg);
        return { bg, ct };
    }
    const name = Backgrounds.name(id);
    const cb = name ? getBackground(name) : undefined;
    return cb ? { cb } : null;
}

// build (and cache on the CompiledTypedBg) a typed background's own group-2 bind group — slot-invariant
// (the per-slot View rides the engine group 0). Returns null while a binding is unpublished (skip, like
// `bgGroup`); a binding-free background carries no group at all (its empty layout never enters the
// pipeline layout)
function typedBgGroup(bg: TypedBackground, ct: CompiledTypedBg): GPUBindGroup | null | "none" {
    const entries = bg.layout.entries as Record<string, object>;
    if (Object.keys(entries).length === 0) return "none";
    const resolved = typedResources(entries);
    if (typeof resolved === "string") {
        return warnSkip(`background:${bg.name}`, `binding "${resolved}" not published`);
    }
    if (
        ct.group2 &&
        ct.group2.resources.length === resolved.resources.length &&
        ct.group2.resources.every((b, k) => b === resolved.resources[k])
    ) {
        return ct.group2.group;
    }
    const group = Compute.root.unwrap(
        Compute.root.createBindGroup(bg.layout, resolved.values as never),
    );
    ct.group2 = { group, resources: resolved.resources };
    return group;
}

/**
 * one camera's geometry pass (`sear:color`) recorded onto `Render.encoder`: shades every opaque draw,
 * then composites every `blend` draw over them (`less-equal` depth-tested against the opaque depth,
 * depth-write off): one color target, no MRT (the screen-space lanes are {@link renderPrepass}'s). With
 * `Camera.antialias` on (the default) it's a 4× MSAA pass resolved into the offscreen; off, it renders
 * single-sample straight into the offscreen (and binds the surfaces' single-sample pipeline twins,
 * compiled lazily by {@link ensureSingle}). Opaque and transparent share one `beginRenderPass` (nothing
 * reads the color between them, so they fuse into one tile round-trip). Group 1 is the sun shadow seam:
 * sear's own shadow map + light params, or its 1×1 fallback (fully lit) when no light casts. An empty
 * draw list still clears the framebuffer. `bg` (the camera's {@link Backdrop} selection) draws a fullscreen
 * backdrop between the opaque and blend draws: masked to far-plane pixels by the depth test, so geometry
 * overdraws it and blended draws composite over it; null leaves the flat clear color as the only backdrop
 */
function renderColor(
    eid: number,
    view: View,
    items: { draw: Draw; r: Recorded }[],
    bg: BackdropPick | null = null,
): void {
    if (!Render.encoder || !view.framebuffer) return;
    // per-camera AA: 4× MSAA when `Camera.antialias` is on (the default the Camera trait seeds), else
    // single-sample. A scene attribute or a runtime `Camera.antialias.set(eid, 0)` flips it live
    const aa = Camera.antialias.get(eid) !== 0;
    const clear = unpackColor(Camera.clearColor.get(eid));
    const { color: msaaColor, depth } = colorTargets(eid, view.width, view.height, aa);
    const pass = beginColor(eid, msaaColor, depth, view.framebuffer, clear);
    pass.setBindGroup(1, shadowGroup());
    // a typed draw sets its own group 1 (`shadowLayoutTyped`'s group — structurally different, so NOT
    // group-equivalent to the raw one), which invalidates the raw group bound at the pass top for any
    // legacy draw after it; track and re-set on the typed→legacy switch
    let rawG1 = true;
    // tally the indirect draws this camera issues (opaque + blend) so the profiler derives the injected
    // validation floor (gpu.md); the honest count is post the `if (pipe)` skip
    let draws = 0;
    const drawTyped = (draw: Draw, r: RecordedTyped, pipe: (typeof r.t)["color"]): void => {
        pipe!
            .with(pass)
            .with(engineLayout, typedEngineGroup(r.g.engineCache, view.slot, r.g.quant))
            .with(shadowLayoutTyped, shadowGroupTyped())
            .with(r.g.layout, r.g.color)
            .withIndexBuffer(r.index, "uint32")
            .drawIndexedIndirect(
                draw.args.indirect,
                (draw.args.offset ?? 0) + view.slot * (draw.args.viewStride ?? 0),
            );
        rawG1 = false;
        draws++;
    };
    for (const { draw, r } of items) {
        if ("c" in r) {
            if (!aa) ensureSingle(r.c); // lazy-compile the single-sample twin for the no-AA camera
            const pipe = aa ? r.c.color : r.c.single?.color;
            if (pipe) {
                if (!rawG1) {
                    pass.setBindGroup(1, shadowGroup());
                    rawG1 = true;
                }
                bind(pass, pipe, draw, r, colorGroup(r, view.slot), view.slot);
                draws++;
            }
        } else {
            if (!aa) ensureTypedSingle(r.t);
            const pipe = aa ? r.t.color : r.t.single?.color;
            if (pipe) drawTyped(draw, r, pipe);
        }
    }
    // the backdrop: a fullscreen triangle at the far plane, after opaque (the depth test masks it to
    // un-rendered pixels) and before blend (so transparent draws composite over it). The bg pipeline
    // carries the shadow group 1 in its layout (unused) like every color pipeline, so the group bound at
    // the pass top survives the switch for the blend draws after it
    if (bg && "cb" in bg) {
        const pipe = aa ? bg.cb.color : bg.cb.single;
        const group = bgGroup(bg.cb, view.slot);
        if (group) {
            if (!rawG1) {
                pass.setBindGroup(1, shadowGroup());
                rawG1 = true;
            }
            pass.setPipeline(pipe);
            pass.setBindGroup(0, group);
            pass.draw(3);
        }
    } else if (bg) {
        // a typed backdrop: the shared engine group 0 (a never-read `bgQuant()` fills the meshQuant
        // slot — a background pulls no mesh), the typed shadow group 1 (declared-but-unused, the
        // group-count-compatibility reason `compileBackground` documents), and its own group 2
        const group = typedBgGroup(bg.bg, bg.ct);
        if (group) {
            const pipe = aa ? bg.ct.color : bg.ct.single;
            let bound = pipe
                .with(pass)
                .with(engineLayout, typedEngineGroup(bg.ct.engineCache, view.slot, bgQuant()))
                .with(shadowLayoutTyped, shadowGroupTyped());
            if (group !== "none") bound = bound.with(bg.bg.layout, group);
            bound.draw(3);
            rawG1 = false;
        }
    }
    for (const { draw, r } of items) {
        if ("c" in r) {
            const pipe = aa ? r.c.transparent : r.c.single?.transparent;
            if (pipe) {
                if (!rawG1) {
                    pass.setBindGroup(1, shadowGroup());
                    rawG1 = true;
                }
                bind(pass, pipe, draw, r, colorGroup(r, view.slot), view.slot);
                draws++;
            }
        } else {
            const pipe = aa ? r.t.transparent : r.t.single?.transparent;
            if (pipe) drawTyped(draw, r, pipe);
        }
    }
    pass.end();
    Compute.indirect?.("sear:color", draws);
}

// the frame's draw list, resolved once by PrepassSystem (the first geometry pass) and shared across the
// prepass, shadow atlases, and color pass — they all draw the same resolved records, so resolving per-pass
// (the old 3×) was wasted work
let _frameDraws: { draw: Draw; r: Recorded }[] = [];

/**
 * compile the forward pipelines for every registered surface, sharing one shader module: a 4× MSAA
 * single-target color pipeline (its own depth, `less` + write) that writes shaded color resolved into
 * the offscreen framebuffer, a 1× tag pipeline (its own single-sample depth, `less` + write, single
 * `r32uint` target) that stamps the front-most fragment's surface tag into `view.tag`, and a 1× depth
 * pipeline (position-only, the shadow map renders through it). Color is one camera-independent shape
 * across opaque / `clip` / `alpha`: no MRT; the tag is its own single-sample lane. Color samples the
 * sun shadow inline (group 1 = the map + comparison sampler + light params); the tag + depth pipelines
 * omit group 1. Sear declares the vertex-pull bindings itself; each draw selects its mesh via
 * `Draw.mesh`. Uniform across surfaces: no "Part-shaped" detection. Also (re)creates the sun-shadow
 * GPU resources sear owns (the comparison sampler, the 1×1 fallback, the group-1 layout, and the real
 * params buffer — `./atlas`), surviving HMR re-warms
 */
async function prepareSear(device: GPUDevice): Promise<void> {
    // the caster count + atlas size fold into the shadow WGSL at its first resolve and the uniforms below
    // size from the same schemas, so a config mutated between builds is a hard error, not a silent mismatch
    checkShadowConfig();
    resetPipelineCaches();
    _warned.clear();
    resetShadowAtlas(device);
    // the lazily-allocated packed list binds at each atlas pipeline's `eids` lane, so allocating it clears
    // the resolved-bind-group cache to rebuild with it
    pointRegather.reset(() => clearGroups());
    cascadeRegather.reset(() => clearGroups());
    // eager-compile non-specializing surfaces plus the shared re-gather A/B pipelines (idempotent — the
    // point + cascade atlases share them). Specializing variants queue after Part publishes its draws;
    // a specializing mesh registered after warm remains lazy.
    await Promise.all([prepareRegather(device), preparePipelines(device, Backgrounds)]);
    await precompileTypedVariants();
}

function unwrapTypedVariant(surface: TypedSurface, variant: number): unknown[] {
    const typed = compileTypedVariant(surface, variant);
    const warmed: unknown[] = [];
    for (const pipeline of [
        typed.color,
        typed.transparent,
        typed.point,
        typed.cascade,
        ...typed.prepass.values(),
    ]) {
        if (pipeline) warmed.push(Compute.root.unwrap(pipeline));
    }
    return warmed;
}

/** queue specializing typed-surface discovery after Part publishes its draw/mesh pairs.
 * `warm` is injectable so the ordering contract stays device-free in unit tests; production unwraps
 * every discovered variant's real pipelines.
 * @internal */
export function precompileTypedVariants(
    warm: (surface: TypedSurface, variant: number) => unknown = unwrapTypedVariant,
    surfaces: Iterable<TypedSurface> = TypedSurfaces,
    variants: (surface: TypedSurface) => number[] = knownTypedVariants,
): Promise<void> {
    return precompile(
        "sear-typed-variants",
        () => {
            const warmed: unknown[] = [];
            for (const surface of surfaces) {
                if (!surface.specialize) continue;
                for (const variant of variants(surface)) {
                    warmed.push(warm(surface, variant));
                }
            }
            return warmed;
        },
        { after: ["kitchen-part-count"] },
    );
}

/**
 * sear's geometry-emit ordering anchor **and** prepass, per camera carrying a lane marker
 * ({@link Tag} / {@link Depth}). It collapses the old empty depth anchor + the tag pass into one
 * single-sample pass that emits the camera's opt-in lanes (the id lane → `view.tag`, the depth lane →
 * `view.depth`), the shape Bevy's prepass takes. It's also the **anchor**: a producer whose per-frame
 * compute writes the geometry sear reads (vertices / indices, or an instanced surface's `transforms` /
 * `eids`) declares `before: [PrepassSystem]` so its emit precedes every geometry-reading pass (the
 * prepass, the shadow map, and the color pass all read it within the frame; an emit landing between them
 * would desync the reads). It runs first among the geometry passes (`after: [BeginFrameSystem]`), so it
 * resolves the frame's draw list **once** into `_frameDraws` for the shadow map + color pass to share. A
 * screen-space effect still slots into the `after: [PrepassSystem], before: [ColorSystem]` seam. A camera
 * carrying no lane marker runs no prepass (the bare path), but the anchor + resolve still run
 */
export const PrepassSystem: System = {
    name: "prepass",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    update(state) {
        if (!Render.encoder) return;
        // resolve once for the prepass + shadow map + color pass (they all run after this)
        _frameDraws = resolveDraws();
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            // markers → the requested lanes. The id lane is a color attachment; depth is the
            // depth-stencil, stored only when the camera carries `Depth`. Reset both each frame so a
            // camera that drops a marker stops publishing its lane
            view.tag = null;
            view.depth = null;
            const lanes = COLOR_LANES.filter((l) => state.has(eid, l.marker));
            const storeDepth = state.has(eid, Depth);
            if (lanes.length === 0 && !storeDepth) continue; // no lane requested — bare path
            renderPrepass(eid, view, _frameDraws, lanes, storeDepth);
        }
    },
};

/**
 * sear's geometry pass, per camera: shades every opaque draw then composites every `blend` draw over
 * them in one 4× MSAA pass (its own 4× color + depth), resolved into the offscreen once. Binds the sun
 * shadow seam (group 1): sear's own shadow map + light params it samples inline, or its fallback (fully
 * lit) when no light casts. Renders the shared `_frameDraws` (resolved once by {@link PrepassSystem}).
 * Runs after every screen-space effect ordered `before: [ColorSystem]`; `before: [GlazeSystem]` makes it
 * sear's terminal offscreen write, so glaze reads `view.framebuffer` only after the resolve lands (glaze
 * never imports sear)
 */
export const ColorSystem: System = {
    name: "color",
    group: "draw",
    annotations: { mode: "always" },
    after: [PrepassSystem],
    before: [GlazeSystem],
    update(state) {
        if (!Render.encoder) return;
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            renderColor(eid, view, _frameDraws, backdrop(state, eid));
        }
    },
};

/**
 * pose the sun's CSM cascade cameras + the point/spot combo cameras from the casting lights + the main Sear
 * camera, so `BeginFrameSystem` packs their viewProjs this frame and the Part pack culls casters into each
 * slot as one more view (the unified culled-combo spine). `simulation` group, before the draw frame opens.
 * No-op for the sun when no directional light carries a {@link Shadow} (the zero-cost off path): the atlas
 * pass is skipped and sear falls back to fully lit
 */
const ShadowCameraSystem: System = {
    name: "shadow-camera",
    group: "simulation",
    annotations: { mode: "always" },
    update(state) {
        let main = -1;
        for (const eid of state.query([Camera, Sear])) {
            main = eid;
            break;
        }
        const frames = updatePointShadows(state, main);
        setPointFrames(frames);
        updateCascades(state, main);
        // allocate each atlas's re-gather list here, before record() (PrepassSystem) builds the cast bind
        // groups that bind it — so the first casting frame's groups include it (the alloc clears the
        // resolved-bind-group cache), no one-frame delay. Idempotent once allocated; the render fns call it
        // again harmlessly
        if (frames.length > 0 && shadowReady()) pointRegather.ensure(pointCasters() * 6);
        if (cascadeCount() > 0 && shadowReady()) cascadeRegather.ensure(MAX_CASCADES);
    },
};

/**
 * render the casters' depth into the shadow atlases (the point/spot tiles + the CSM cascades) and publish the
 * seams for sear's color pass to sample inline. `after: [PrepassSystem]` so every position-writing producer
 * (pinned before the anchor) has emitted and `_frameDraws` is resolved; `before: [ColorSystem]` so the
 * atlases + seams are ready before sear shades. No casting light → no pass, sear falls back to fully lit.
 * Bevy's shape: the shadow maps are light-data-gated, sampled inline, no separate shadow plugin
 */
const ShadowMapSystem: System = {
    name: "shadowmap",
    group: "draw",
    annotations: { mode: "always" },
    after: [PrepassSystem],
    before: [ColorSystem],
    update() {
        renderPointShadows(_frameDraws);
        renderCascades(_frameDraws);
    },
};

// the bindings sear's default materials read: the `eids` + `transforms` instance convention (sear
// applies the standard transform) plus per-entity `color`. Producers publish these buffers by name
// (Part does); a surface declaring them is instanced
const colorBindings: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    // sRGB-packed LDR color (the `srgb8x4` slab) — one u32 per entity, decoded with `unpackLdrColor`
    color: { type: "storage", element: "u32" },
};

// the lit materials add the per-entity `material` slab (the Material component's metallic / roughness /
// emissive / occlusion packed two f16 per u32 word — f16 keeps emissive HDR-capable as a glow strength
// while the bounded lanes stay finer than unorm8; the binding is `vec2<u32>` + `unpack2x16float` rather
// than `vec4<f16>` so no engine shader needs `enable f16`). `unlit` omits it — it never shades, so it
// stays on `colorBindings`. One extra storage binding: the color pass goes 8 → 9 of the 10-per-stage
// ceiling (gpu.md)
const litBindings: Record<string, Binding> = {
    ...colorBindings,
    material: { type: "storage", element: "vec2<u32>" },
};

// the typed twin of `litBindings`, group 2 (`layout()`'s $idx(2) synthesis) — same four bindings, same
// element shapes, feeding the typed `default` surface below (4a-ii-c-2's template port).
const typedDefaultLayout = typedLayout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    color: { type: "storage", element: d.u32 },
    material: { type: "storage", element: d.vec2u },
});

// the typed twin of the raw `default` fs (`matOf`/`emissiveOf`/`litPbr`, string-registered above):
// `Pbr(albedo, metallic, roughness, occlusion, dielectric)` from the packed `material` lanes — word x
// (metallic, roughness), word y (emissive, occlusion) — same f16-via-`unpack2x16float` shape, no
// `shader-f16` needed. `litPbr` (`sear/engine.ts`) reads the fs-scaffold privates the typed pipeline
// builder (`pipelines.ts`) fills before calling this.
const typedDefaultFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const m = typedDefaultLayout.$.material[ctx.eid];
    const mr = std.unpack2x16float(m.x);
    const eo = std.unpack2x16float(m.y);
    const albedo = unpackLdrColor(typedDefaultLayout.$.color[ctx.eid]).xyz;
    const pbr = Pbr({ albedo, metallic: mr.x, roughness: mr.y, occlusion: eo.y, dielectric: 0 });
    const emissive = std.mul(albedo, eo.x);
    return d.vec4f(std.add(litPbr(pbr, ctx.worldNormal, ctx.world), emissive), 1);
});

// the typed twin of `colorBindings` — `unlit`'s three bindings, no `material` (it never shades).
const typedColorLayout = typedLayout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    color: { type: "storage", element: d.u32 },
});

// the typed twin of the raw `unlit` fs above (4a-ii-c-3): `unpackLdrColor(color[eid]).rgb` verbatim, no
// lighting call — the simplest surface the typed template carries.
const typedUnlitFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return d.vec4f(unpackLdrColor(typedColorLayout.$.color[ctx.eid]).xyz, 1);
});

// the typed twin of the raw `vertex` surface (per-vertex Gouraud): `litColor` crosses vs→fs as a custom
// varying through the `typedVaryingVs`/`typedVaryingFs` copier pair (`pipelines.ts`), so this `vs` runs
// `litPbr` once per vertex. `sunVisibility`/`pointScale`/`fragWorld` sit at their defaults here (per-vertex
// shading runs before the fs scaffold fills them) — the same fully-lit sun / zero-point-contribution the
// raw path's per-vertex mode gets (render.md "Surface authoring").
const typedVertexVaryings = { litColor: d.vec3f };
const typedVertexPatch = vsPatchSchema(typedVertexVaryings);
const typedVertexVs = tgpu.fn(
    [VsIn],
    typedVertexPatch,
)((vsIn) => {
    "use gpu";
    const m = typedDefaultLayout.$.material[vsIn.eid];
    const mr = std.unpack2x16float(m.x);
    const eo = std.unpack2x16float(m.y);
    const albedo = unpackLdrColor(typedDefaultLayout.$.color[vsIn.eid]).xyz;
    const pbr = Pbr({ albedo, metallic: mr.x, roughness: mr.y, occlusion: eo.y, dielectric: 0 });
    const emissive = std.mul(albedo, eo.x);
    const litColor = std.add(
        litPbr(pbr, std.normalize(vsIn.worldNormal), vsIn.world.xyz),
        emissive,
    );
    return typedVertexPatch({
        world: vsIn.world,
        worldNormal: vsIn.worldNormal,
        clip: d.vec4f(0),
        litColor,
    });
});
const typedVertexFs = tgpu.fn(
    [fsCtxSchema(typedVertexVaryings)],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return d.vec4f(ctx.litColor, 1);
});

// free every GPU resource sear owns (at plugin dispose): the shadow atlases (point + cascade, ./atlas) +
// their params, and the per-camera prepass depth / lane targets / MSAA color+depth. The cascade Camera
// entities live in a State, so destroyCascades (./shadows) tears those down separately
function disposeSear(): void {
    disposeShadowAtlas();
    for (const c of _depth.values()) c.texture.destroy();
    for (const c of _laneTargets.values()) c.texture.destroy();
    for (const c of _colorTargets.values()) {
        c.color?.destroy();
        c.depth.destroy();
    }
    _depth.clear();
    _laneTargets.clear();
    _colorTargets.clear();
    clearBackgrounds();
}

/**
 * Sear: the one kitchen renderer. A GPU-driven raster forward pass: a 4× MSAA color pass (opaque draws
 * then `blend` draws composited over them, fused into one render pass) and an opt-in single-sample
 * prepass emitting per-camera lanes (the {@link Tag} → `view.tag` id lane, the {@link Depth} →
 * `view.depth` lane), with sun shadows sampled inline in the FS. Add `SearPlugin` and give a Camera the
 * {@link Sear} marker and the happy path renders. Sun shadows are data-gated on the {@link Shadow}
 * component on a `DirectionalLight`: add it to cast (and tune), omit it for the fully-lit bare path (no
 * shadow map allocated), exactly like a camera without a lane marker runs no prepass. No separate shadow
 * plugin, no coordination singleton: sear owns its own shadow map and binds it (Bevy's clustered-forward
 * shape). Sear renders into the offscreen
 * (`view.framebuffer`) and never the swapchain; presenting it is a separate **composite** the consumer
 * picks: {@link GlazePlugin} (the engine default postfx composite) or a custom one (orrstead ships a
 * fused compute composite). So sear depends only on {@link RenderPlugin}; list a composite alongside it
 * or nothing reaches the swapchain. `ColorSystem` still orders `before: [GlazeSystem]` so glaze, *when
 * present*, composites after the resolve (the ordering ref drops harmlessly when glaze isn't registered).
 */
export const SearPlugin: Plugin = {
    name: "Sear",
    components: { Sear, Tag, Depth, Shadow, Material, Backdrop },
    systems: [PrepassSystem, ColorSystem, ShadowCameraSystem, ShadowMapSystem],
    // SlabPlugin: the `Material` slab is collected + published as `"material"`, and `initMaterial`
    // bases every slot through it (Part brings SlabPlugin anyway; declaring it keeps sear self-sufficient)
    dependencies: [RenderPlugin, SlabPlugin],
    traits: {
        Shadow: { defaults: () => ({ ...SHADOW_DEFAULTS }) },
        Material: MaterialTraits,
        Backdrop: BackdropTraits,
    },

    // sear's default materials, shading per-instance `color` + `material` at three lighting modes. They
    // ship with the renderer, not Part: Part publishes the data (`eids` + `color`), sear adds its own
    // `Material` slab and shades with its metallic-roughness `litPbr`. `Part.surface` defaults to
    // "default" (per-pixel); "vertex" (per-vertex Gouraud) and "unlit" are picked per-Part. The instance
    // transform is sear's convention — these declare the bindings and omit a transform vs chunk. `pbr()`
    // builds the Pbr struct from the packed `material` lanes; the engine default has no specular until a
    // Material sets metallic > 0 (dielectric 0), so a bare Part shades exactly like the pre-PBR diffuse.
    initialize(state) {
        // a fresh State recreates its own off-screen shadow cameras lazily — drop any eids cached by
        // a prior build so this re-run never aliases recycled entities (ecs.md module-scope contract)
        resetPointShadows();
        resetCascades();
        initMaterial();
        // clear the backdrop registry so a rebuild re-registers identically and a plugin toggled off leaves
        // no stale entry (the Surfaces/Draws/Meshes reload-safety shape — RenderPlugin.initialize clears those)
        Backgrounds.clear();
        // build the Pbr struct + the emissive tint (Color.rgb * the emissive strength lane) from the
        // f16 material lanes: word x holds (metallic, roughness), word y (emissive, occlusion), each
        // unpacked to f32 for the shading math. emissive is an unbounded HDR glow strength;
        // dielectric 0 → metallic 0 is specular-free (the flat shallot default)
        const PbrPreamble = /* wgsl */ `
        fn matOf(eid: u32) -> Pbr {
            let m = material[eid];
            let mr = unpack2x16float(m.x);
            let eo = unpack2x16float(m.y);
            return Pbr(unpackLdrColor(color[eid]).rgb, mr.x, mr.y, eo.y, 0.0);
        }
        fn emissiveOf(eid: u32) -> vec3<f32> {
            return unpackLdrColor(color[eid]).rgb * unpack2x16float(material[eid].y).x;
        }`;
        Surfaces.register({
            name: "default",
            bindings: litBindings,
            preamble: PbrPreamble,
            fs: /* wgsl */ `col = vec4<f32>(litPbr(matOf(eid), worldNormal, world) + emissiveOf(eid), 1.0);`,
        });
        // the typed twin of the string `default` above, registered ADDITIONALLY into `TypedSurfaces`
        // under the same name — a separate registry (`contract.ts`), so it doesn't collide with the
        // string one. `record()` consults the typed registry first, so `default` DRAWS through the typed
        // path in every pass (the built-in flip); the string registration stays for Part's automatic
        // draw registration + legacy consumers until 4a-ii-d. `matOf`/`emissiveOf` above stay
        // string-only helpers; this is their TGSL equivalent, statement-for-statement
        // (`engine.test.ts`-style differential in `pipelines.test.ts`).
        typedRegister(state, {
            name: "default",
            layout: typedDefaultLayout,
            fs: typedDefaultFs,
        });
        Surfaces.register({
            name: "vertex",
            bindings: litBindings,
            preamble: PbrPreamble,
            interpolators: { litColor: "vec3<f32>" },
            vs: /* wgsl */ `litColor = litPbr(matOf(eid), normalize(worldNormal), world.xyz) + emissiveOf(eid);`,
            fs: /* wgsl */ `col = vec4<f32>(litColor, 1.0);`,
        });
        // the typed twin — the varyings mechanism's first live consumer (`litColor` crosses vs→fs
        // through `typedVaryingVs`/`typedVaryingFs`'s per-surface copier, `pipelines.ts`); drawn typed
        // in every pass, like `default` above.
        typedRegister(state, {
            name: "vertex",
            layout: typedDefaultLayout,
            varyings: typedVertexVaryings,
            vs: typedVertexVs,
            fs: typedVertexFs,
        });
        Surfaces.register({
            name: "unlit",
            bindings: colorBindings,
            fs: /* wgsl */ `col = vec4<f32>(unpackLdrColor(color[eid]).rgb, 1.0);`,
        });
        // the typed twin, drawn typed in every pass like `default` above.
        typedRegister(state, {
            name: "unlit",
            layout: typedColorLayout,
            fs: typedUnlitFs,
        });
    },

    async warm() {
        if (!Compute.device) return;
        await prepareSear(Compute.device);
    },

    dispose(state) {
        destroyPointShadows(state);
        destroyCascades(state);
        disposeSear();
    },
};
