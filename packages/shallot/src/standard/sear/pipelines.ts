// Sear's pipeline compilation: the compiled-surface / compiled-background caches and the async
// `createRenderPipelineAsync` calls that fill them. `codegen.ts` supplies the WGSL text; `atlas.ts`
// supplies the two shadow-atlas bind-group layouts (`shadowLayout()` / `pointLayout()`) every color +
// point + cascade pipeline binds group 1 against. `forward.ts` owns bind-group *resolution* per draw
// (`record()`) — this file only compiles pipelines and caches them by `${surface}#${variant}`.

import type { Configurable, TgpuBindGroupLayout, TgpuRenderPipeline } from "typegpu";
import tgpu from "typegpu";
import type { AnyWgslData } from "typegpu/data";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute, type Registry } from "../../engine";
import {
    decodePos,
    decodeUv,
    MeshQuant,
    meshIdOf,
    octDecodeNormal,
    Xform,
    xformNormal,
    xformPoint,
} from "../../engine/utils/core";
import type { Binding, Draw, Mesh, Surface } from "../render/core";
import {
    Draws,
    Frame,
    LightCull,
    Lighting,
    Meshes,
    Render,
    Surfaces,
    VIEW_BYTES,
} from "../render/core";
import {
    cascadeLayoutTyped,
    pointLayout,
    pointLayoutTyped,
    shadowLayout,
    shadowLayoutTyped,
    shadowReady,
} from "./atlas";
import {
    BG_BASE,
    backgroundCode,
    bindingEntry,
    DEPTH_FORMAT,
    FRAME,
    LIGHTING,
    laneKey,
    laneSubsets,
    pointShadowCode,
    prepassEntry,
    SAMPLE_COUNT,
    SURFACE_BASE,
    surfaceCode,
    TAG_FORMAT,
    TAG_NONE,
    UNIFORM_LAYOUT,
    VIEW,
    VS_FS,
} from "./codegen";
import type {
    SurfaceLayout,
    TypedBackground,
    Binding as TypedBinding,
    TypedSurface,
} from "./contract";
import { BgCtx, fsCtxSchema, TypedBackgrounds, TypedSurfaces, VsIn } from "./contract";
import {
    engineLayout,
    fragCoord,
    fragWorld,
    pointScale,
    pointShadowSlot,
    pointShadowStub,
    sunVisibility,
} from "./engine";
import type { Background } from "./forward";
import { COMBO_SHIFT, EID_MASK } from "./regather";
import { sampleSunShadow } from "./shade";
import { cascadeAtlasSize, pointAtlasSize, sunCascades, sunResolution } from "./shadows";

export interface Compiled {
    // a surface compiles to one shape, by render mode. Opaque + `clip` cutout: `color` (the
    // single-target framebuffer) + `prepass` (one pipeline per color-lane subset — `""` is the
    // position-only depth pipeline the shadow map renders casters through, `"tag"` writes the id lane;
    // a `clip` surface's `""` runs the fragment to discard so it casts a holed shadow). `alpha`:
    // `transparent` alone — one blended color target, no prepass lanes (a transparent pixel has no
    // single owner, writes no prepass depth, casts nothing). The unused slots are null / an empty map,
    // so the shadow-map, prepass, and color passes each pick the pipeline that's theirs (the color pass
    // draws both `color` opaque and `transparent` blended, in that order, within one render pass)
    color: GPURenderPipeline | null;
    transparent: GPURenderPipeline | null;
    // the single-sample (AA-off) twin, compiled lazily by `ensureSingle` the first time a camera with
    // `Camera.antialias` off renders — null until then. An all-AA-on scene (the default) never touches it.
    // Differs from `color`/`transparent` only in `multisample.count` (1 vs SAMPLE_COUNT); `lazy` carries
    // the shared inputs to compile it
    single: { color: GPURenderPipeline | null; transparent: GPURenderPipeline | null } | null;
    singlePending: boolean;
    colorArgs: ColorArgs;
    prepass: Map<string, GPURenderPipeline>;
    // the point-shadow atlas pipeline (depth-only): one indirect draw per casting mesh, the VS reading the
    // re-gathered packed instance list (per-combo culled, concatenated mesh-major) at the eids lane and
    // remapping clip XY into each combo's atlas tile. null for `alpha` (a transparent pixel casts nothing)
    // and `screen` surfaces (2D overlays have no atlas placement)
    point: GPURenderPipeline | null;
    // the CSM cascade atlas pipeline (depth-only): the point pipeline's twin, the VS reading the cascade
    // re-gathered list and remapping clip XY into each cascade's atlas tile. Same gating as `point` (null for
    // `alpha`/`screen`/non-instanced); they differ only in the per-cascade vs per-(caster, face) tile index
    cascade: GPURenderPipeline | null;
    layout: GPUBindGroupLayout;
    slots: { name: string; type: Binding["type"] }[];
}

// straight (non-premultiplied) alpha. The swapchain is sRGB, so the blend unit linearizes the
// stored color before compositing — `src·α + dst·(1−α)` is gamma-correct with the fs writing
// linear `col`. The alpha channel keeps the framebuffer's coverage sensible for a later read
const ALPHA_BLEND: GPUBlendState = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};

export type BindResource = GPUBuffer | GPUTexture | GPUSampler;
export type GroupEntry = {
    main: GPUBuffer;
    position: GPUBuffer;
    layout: GPUBindGroupLayout;
    // every group-0 entry but VERTICES (slot 3) and VIEW — the two that vary per bind (vertex stream by
    // pass, View buffer by slot). Per-slot bind groups are built lazily off this shared prefix, keyed on
    // the slot the draw is bound for (a scene is typically one shading camera, so `colorCache`/
    // `prepassCache` usually hold one entry — the lazy-slot-key law, Approach 4a)
    shared: GPUBindGroupEntry[];
    colorCache: Map<number, GPUBindGroup>;
    prepassCache: Map<number, GPUBindGroup>;
    // the point-shadow pass's group 0 — the prepass group with the `eids` lane swapped to the point
    // re-gathered packed instance list (`pointRegather.eids()`); null for a non-casting surface, or
    // until the re-gather buffer is allocated (first casting frame). Bound to slot 0's View buffer as an
    // unread placeholder — the point VS projects by its own tile viewProj, never `view`
    pointGroup: GPUBindGroup | null;
    // the cascade-atlas pass's group 0 — the same swap, to the cascade re-gathered list
    cascadeGroup: GPUBindGroup | null;
    resources: BindResource[];
};
type GroupCache = Map<string, GroupEntry>;

// pipelines keyed `${surface}#${variant}` — one entry per (surface, material map-set) a scene actually
// draws (Bevy's on-demand specialization). A non-specializing surface is always variant 0 (compiled eagerly
// at warm); a specializing surface (the glTF importer) compiles each map-set variant lazily on first draw
const _compiled = new Map<string, Compiled>();
// the variant keys whose async compile is in flight (or permanently failed), so `record` triggers each
// compile once — the draw skips (returns null) until the pipeline lands, the same shape as an unpublished
// texture. A failed key stays in the set: a variant's WGSL is fixed for the State's life, so retrying would
// recompile + re-warn every frame; a fresh build clears the set (prepareSear)
const _compiling = new Set<string>();
const variantKey = (surface: string, variant: number) => `${surface}#${variant}`;
const _groups: GroupCache = new Map();

// a compiled background: the 4× MSAA + single-sample backdrop pipelines (one camera picks one by AA mode),
// the shared group-0 layout, and its binding slots. `groupCache` builds lazily per view slot on first use
// in `renderColor` (bg × slot, Approach 4a) and holds for the State's life — the frame/view/lighting +
// bg-binding buffers are stable post-warm, and a rebuild recompiles from scratch (`_backgrounds.clear()`
// in `resetPipelineCaches`), so no in-build invalidation
export type CompiledBg = {
    name: string;
    color: GPURenderPipeline;
    single: GPURenderPipeline;
    layout: GPUBindGroupLayout;
    slots: { name: string; type: Binding["type"] }[];
    groupCache: Map<number, GPUBindGroup>;
};
const _backgrounds = new Map<string, CompiledBg>();

/** the compiled pipeline set for a `(surface, variant)`, or `undefined` until it lands ({@link ensureVariant} triggers the compile). */
export function getCompiled(surface: string, variant: number): Compiled | undefined {
    return _compiled.get(variantKey(surface, variant));
}

/** the cached per-draw bind-group state for a Draw name, or `undefined` on a cache miss (`record` rebuilds it). */
export function getGroup(name: string): GroupEntry | undefined {
    return _groups.get(name);
}

/** cache a draw's resolved bind-group state (`record`, on a resource-identity change). */
export function setGroup(name: string, entry: GroupEntry): void {
    _groups.set(name, entry);
}

/** drop every cached per-draw bind-group state (legacy + typed) — a rebuild (`resetPipelineCaches`) or a
 * shadow-atlas re-gather reallocation (the `eids` lane both caches bind moved). */
export function clearGroups(): void {
    _groups.clear();
    _typedGroups.clear();
}

/** a compiled background by name, or `undefined` if it never registered / failed to compile. */
export function getBackground(name: string): CompiledBg | undefined {
    return _backgrounds.get(name);
}

/** drop every compiled background (plugin dispose — a rebuild's `resetPipelineCaches` also covers this). */
export function clearBackgrounds(): void {
    _backgrounds.clear();
}

/** clear every pipeline-compilation cache for a rebuild: compiled variants (legacy + typed), in-flight
 * compiles, resolved bind groups, and compiled backgrounds (legacy + typed). Called once at the top of
 * `prepareSear`. */
export function resetPipelineCaches(): void {
    _compiled.clear();
    _compiling.clear();
    _groups.clear();
    _backgrounds.clear();
    _compiledTyped.clear();
    _compiledTypedBg.clear();
    _typedGroups.clear();
    _bgQuant?.destroy();
    _bgQuant = null;
}

// the inputs to build a surface's color/transparent pipelines at any sample count. Fixed per variant
// (only the count varies), so `compileVariant` stashes them on `Compiled` for `ensureSingle` to compile
// the single-sample twin without re-deriving the shader module (the expensive part)
type ColorArgs = {
    name: string;
    variant: number;
    module: GPUShaderModule;
    colorLayout: GPUPipelineLayout;
    primitive: GPUPrimitiveState;
    blend: Surface["blend"];
};

// build the color pass's pipelines at a given sample count — the opaque `color` (a `clip` surface is
// opaque too) or the blended `transparent` (`alpha`), whichever the surface's blend mode selects; the
// other stays null. `multisample.count` is the only thing that varies with AA mode, so the same shader
// module + pipeline layout produce both the 4× (`compileVariant`) and 1× (`ensureSingle`) twins
export async function colorPipelines(
    device: GPUDevice,
    args: ColorArgs,
    samples: number,
): Promise<{ color: GPURenderPipeline | null; transparent: GPURenderPipeline | null }> {
    const { name, variant, module, colorLayout, primitive, blend } = args;
    const suffix = samples === 1 ? "-1x" : "";
    if (blend === "alpha") {
        const transparent = await device.createRenderPipelineAsync({
            label: `sear-transparent-${name}#${variant}${suffix}`,
            layout: colorLayout,
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: Render.format, blend: ALPHA_BLEND }],
            },
            primitive,
            depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: false,
                depthCompare: "greater-equal",
            },
            multisample: { count: samples },
        });
        return { color: null, transparent };
    }
    const color = await device.createRenderPipelineAsync({
        label: `sear-${name}#${variant}${suffix}`,
        layout: colorLayout,
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format: Render.format }] },
        primitive,
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "greater" },
        multisample: { count: samples },
    });
    return { color, transparent: null };
}

/**
 * compile one background's backdrop pipelines into the background cache: the 4× MSAA + single-sample twins
 * (a camera binds whichever its `Camera.antialias` selects), sharing one shader module + group-0 layout
 * (frame / one whole per-slot view buffer, no dynamic offset / lighting + the background's own bindings
 * at {@link BG_BASE}). The pipeline draws the fullscreen triangle at the reverse-Z far plane with
 * `depthCompare: "greater-equal"`
 * and **no depth write**: at clip z = 0 an un-rendered pixel (cleared depth 0) passes `0 >= 0`, a
 * geometry pixel (depth > 0) fails, so the backdrop fills only background pixels with no readback. Under
 * MSAA the per-sample test resolves the sky↔geometry silhouette antialiased. Both twins compile eagerly
 * (backgrounds are few; the camera's AA mode is known only at draw time).
 */
export async function compileBackground(device: GPUDevice, bg: Background): Promise<void> {
    const entries = Object.entries(bg.bindings ?? {});
    const slots = entries.map(([n, b]) => ({ name: n, type: b.type }));
    const layout = device.createBindGroupLayout({
        label: `sear-bg-${bg.name}`,
        entries: [
            { binding: FRAME, visibility: VS_FS, buffer: { type: "uniform" } },
            // one whole per-slot View buffer, no dynamic offset (bg × slot groups, `bgGroup`)
            {
                binding: VIEW,
                visibility: VS_FS,
                buffer: { type: "uniform", minBindingSize: VIEW_BYTES },
            },
            { binding: LIGHTING, visibility: VS_FS, buffer: { type: "uniform" } },
            ...entries.map(([, b], k) => bindingEntry(b, k + BG_BASE)),
        ],
    });
    const module = device.createShaderModule({
        label: `sear-bg-${bg.name}`,
        code: backgroundCode(bg),
    });
    // group 1 is the shadow seam — the background shader never references it, but the layout declares it
    // (unused) so the bg pipeline has the same two-group shape every color pipeline does. That keeps the
    // shadow group (bound once per pass) alive across the opaque → backdrop → blend pipeline switches: a
    // bg pipeline with only group 0 would be a group-count mismatch that drops group 1 for the blend draws
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [layout, shadowLayout()!],
    });
    const pipe = (samples: number) =>
        device.createRenderPipelineAsync({
            label: `sear-bg-${bg.name}${samples === 1 ? "-1x" : ""}`,
            layout: pipelineLayout,
            vertex: { module, entryPoint: "vs" },
            fragment: { module, entryPoint: "fs", targets: [{ format: Render.format }] },
            // a fullscreen triangle has no consistent winding — disable back-face culling
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: false,
                depthCompare: "greater-equal",
            },
            multisample: { count: samples },
        });
    const [color, single] = await Promise.all([pipe(SAMPLE_COUNT), pipe(1)]);
    _backgrounds.set(bg.name, {
        name: bg.name,
        color,
        single,
        layout,
        slots,
        groupCache: new Map(),
    });
}

/**
 * compile one surface's pipelines for a material map-set `variant` (the color + transparent + per-lane-set
 * prepass pipelines + the shared bind-group layout), keyed `${surface}#${variant}` in the compiled-surface
 * cache. A non-specializing surface only ever uses variant 0 (compiled eagerly at warm); a specializing
 * surface (the glTF importer) gets one entry per map-set a scene draws (Bevy's on-demand specialization). The
 * bindings are variant-invariant, so every variant shares the same layout shape and `record`'s one cached
 * bind group. The color pipelines compile at 4× (the AA-on default); the single-sample twin compiles lazily
 * in `ensureSingle`.
 */
export async function compileVariant(
    device: GPUDevice,
    surface: Surface,
    variant: number,
): Promise<void> {
    const name = surface.name;
    const entries = Object.entries(surface.bindings ?? {});
    const slots = entries.map(([n, b]) => ({ name: n, type: b.type }));

    const layout = device.createBindGroupLayout({
        label: `sear-${name}`,
        entries: [
            ...UNIFORM_LAYOUT,
            ...entries.map(([, b], k) => bindingEntry(b, k + SURFACE_BASE)),
        ],
    });
    // color binds group 0 (per-draw, camera-independent) + group 1 (the sun shadow map +
    // sampler + params); the prepass pipelines bind group 0 alone, since their fragments never
    // reference the group-1 shadow bindings, so the missing group 1 is valid
    const colorLayout = device.createPipelineLayout({
        bindGroupLayouts: [layout, shadowLayout()!],
    });
    const prepassLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    // two modules: the color entries reference the real group-1 shadow bindings; the
    // prepass entries compile against stubs so their group-0-only layout stays valid (and
    // the atlas render never samples the texture it's writing). `alpha` has no prepass
    // entries, so it compiles the color module alone
    const module = device.createShaderModule({
        label: `sear-${name}#${variant}`,
        code: surfaceCode(surface, "color", variant),
    });
    // a screen-space surface builds its own quads in clip space (lines), so their winding
    // flips with segment direction — back-face culling would drop half of them. World-space
    // surfaces keep back-face culling (the overdraw win + correct cutout/shadow facing)
    const primitive: GPUPrimitiveState = {
        topology: "triangle-list",
        cullMode: surface.screen ? "none" : "back",
        frontFace: "ccw",
    };
    const colorArgs: ColorArgs = {
        name,
        variant,
        module,
        colorLayout,
        primitive,
        blend: surface.blend,
    };

    // a `blend` surface is one non-opaque pipeline: a single blended color target, depth-*tested*
    // (`less-equal`) against the color pass's depth so nearer opaque geometry occludes it, but never
    // depth-*written* so it occludes nothing itself. No prepass lanes (a transparent pixel has no
    // single owner, writes no prepass depth, casts nothing) — `color` stays null, `prepass` an empty map
    if (surface.blend === "alpha") {
        const { color, transparent } = await colorPipelines(device, colorArgs, SAMPLE_COUNT);
        _compiled.set(variantKey(name, variant), {
            color,
            transparent,
            single: null,
            singlePending: false,
            colorArgs,
            prepass: new Map(),
            point: null, // a transparent pixel casts nothing
            cascade: null,
            layout,
            slots,
        });
        return;
    }

    // opaque and masked-opaque cutout (`clip`) share the color pipeline + the per-lane-set prepass
    // pipelines; they differ only in the empty-set prepass fragment stage (a `clip` surface runs
    // `fsPrepass` to discard, an opaque one is position-only)
    const clip = surface.blend === "clip";
    const prepassModule = device.createShaderModule({
        label: `sear-prepass-${name}#${variant}`,
        code: surfaceCode(surface, "prepass", variant),
    });
    // the prepass depth-stencil (reverse-Z `greater` + write, its own single-sample depth cleared each
    // frame). The color pass's depth lives inside `colorPipelines`; both are `greater` + write, never
    // cross-compared
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    // the point-shadow atlas pipeline (depth-only): one indirect draw per casting mesh into the shared
    // atlas, the VS reading the re-gathered per-combo culled instances + remapping clip XY to the tile.
    // cullMode "back" (as the depth pass): the tile remap applies two y-flips (face-NDC → atlas-uv →
    // atlas-NDC) that cancel, so the net winding is unchanged and back-face cull drops the same faces. It
    // must — a light sitting inside a caster (a lamp fixture sphere, a light marker) sees only that mesh's
    // back faces, so culling them is what stops the fixture from occluding its own light in every
    // direction (receiver-side bias carries the acne). Only an **instanced** surface casts (the re-gathered
    // list keys on the per-instance `eids` + `transforms`); a non-instanced producer or a `screen` overlay
    // has no per-instance member list, so it gets no point pipeline
    const instanced = !!(surface.bindings?.eids && surface.bindings?.transforms);
    const castable = !surface.screen && instanced;
    const pointModule = castable
        ? device.createShaderModule({
              label: `sear-point-${name}#${variant}`,
              code: pointShadowCode(surface, variant),
          })
        : null;
    // the cascade atlas pipeline is the point pipeline's twin (same depth-only shape + group-1 layout, the
    // per-cascade tile index the only difference), so it gates + compiles the same way
    const cascadeModule = castable
        ? device.createShaderModule({
              label: `sear-cascade-${name}#${variant}`,
              code: pointShadowCode(surface, variant, true),
          })
        : null;
    const castLayout = castable
        ? device.createPipelineLayout({ bindGroupLayouts: [layout, pointLayout()!] })
        : null;

    const prepass = new Map<string, GPURenderPipeline>();
    const [{ color, transparent }, point, cascade] = await Promise.all([
        // single-target color at the AA-on sample count, resolved into the offscreen framebuffer. Owns
        // its own depth (`less` + write); the single-sample twin compiles lazily in `ensureSingle`
        colorPipelines(device, colorArgs, SAMPLE_COUNT),
        pointModule && castLayout
            ? device.createRenderPipelineAsync({
                  label: `sear-point-${name}#${variant}`,
                  layout: castLayout,
                  vertex: { module: pointModule, entryPoint: "vs" },
                  fragment: { module: pointModule, entryPoint: "fsPoint", targets: [] },
                  primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
                  depthStencil,
                  multisample: { count: 1 },
              })
            : Promise.resolve(null),
        cascadeModule && castLayout
            ? device.createRenderPipelineAsync({
                  label: `sear-cascade-${name}#${variant}`,
                  layout: castLayout,
                  vertex: { module: cascadeModule, entryPoint: "vs" },
                  fragment: { module: cascadeModule, entryPoint: "fsPoint", targets: [] },
                  primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
                  depthStencil,
                  multisample: { count: 1 },
              })
            : Promise.resolve(null),
        // one single-sample prepass pipeline per color-lane subset (`less` + write, its own depth
        // cleared each frame so the front-most fragment stamps with no prepass below it). The
        // empty subset is position-only depth (the shadow map + a depth-only camera render through
        // it; a `clip` surface adds the `fsPrepass` discard so it holes the depth and casts a
        // holed shadow); `tag` writes the id lane. Binds the group-0 layout alone — the prepass
        // fragments never reference the group-1 shadow bindings, so they carry no lighting
        ...laneSubsets().map((laneSet) =>
            device
                .createRenderPipelineAsync({
                    label: `sear-prepass-${laneKey(laneSet) || "depth"}-${name}#${variant}`,
                    layout: prepassLayout,
                    vertex: { module: prepassModule, entryPoint: "vs" },
                    ...(laneSet.length > 0 || clip
                        ? {
                              fragment: {
                                  module: prepassModule,
                                  entryPoint: prepassEntry(laneSet),
                                  targets: laneSet.map((l) => ({ format: l.format })),
                              },
                          }
                        : {}),
                    primitive,
                    depthStencil,
                })
                .then((p) => {
                    prepass.set(laneKey(laneSet), p);
                    return p;
                }),
        ),
    ]);
    _compiled.set(variantKey(name, variant), {
        color,
        transparent,
        single: null,
        singlePending: false,
        colorArgs,
        prepass,
        point,
        cascade,
        layout,
        slots,
    });
}

// trigger the lazy compile of a specializing surface's `variant` once (the draw skips until it lands).
// Deduped via `_compiling`; on success the key drops so `_compiled` is the sole record, on failure it stays
// (warn once, never retry — the WGSL is deterministic, so a recompile would just fail + spam again)
export function ensureVariant(surface: Surface, variant: number): void {
    const key = variantKey(surface.name, variant);
    if (_compiled.has(key) || _compiling.has(key)) return;
    const device = Compute.device;
    if (!device || !shadowReady()) return;
    _compiling.add(key);
    compileVariant(device, surface, variant).then(
        () => _compiling.delete(key),
        (e) =>
            console.warn(`sear: surface "${surface.name}" variant ${variant} failed to compile`, e),
    );
}

// compile a variant's single-sample (AA-off) twin, once, the first frame a no-AA camera draws it (deduped
// via `singlePending`; until it lands the draw skips for that camera, the same shape as a lazy variant).
// The shader module + layout are already compiled, so this is just the 1× pipeline. An all-AA-on scene
// never calls this — `renderColor` invokes it only for a camera whose `Camera.antialias` is off
export function ensureSingle(c: Compiled): void {
    if (c.single || c.singlePending) return;
    const device = Compute.device;
    if (!device) return;
    c.singlePending = true;
    colorPipelines(device, c.colorArgs, 1).then(
        (pipes) => {
            c.single = pipes;
            c.singlePending = false;
        },
        (e) => {
            c.singlePending = false;
            const msg = `sear: surface "${c.colorArgs.name}" single-sample pipeline failed to compile`;
            console.warn(msg, e);
        },
    );
}

// ---- the typed pipeline builder (4a-ii-c, the template + its extensions): compiles a `TypedSurface`'s
// `vs`/`fs` TGSL fns against the canonical `engineLayout` (group 0), the typed shadow group (group 1,
// `shadowLayoutTyped` — the atlas passes bind `pointLayoutTyped`/`cascadeLayoutTyped` there instead), and
// the surface's own synthesized `layout` (group 2 — `surfaceLayout()`'s $idx(2) synthesis, c-1): the
// opaque color / transparent pipelines, the prepass depth + tag pipelines, and the point/cascade
// shadow-atlas pipelines. A surface in `TypedSurfaces` DRAWS through these in every pass — `record()`
// consults the typed registry first (the built-in flip, c-3b), and `forward.ts`/`atlas.ts` issue the
// draws via `.with(pass)` on sear's own render passes. `screen` surfaces project through their own `vs`
// chunk's `patch.clip` and draw un-culled (d-1); `"clip"` blend and `specialize` are carried for the glTF
// typed migration.
//
// Cached beside the legacy `_compiled` map by name + material variant. `preparePipelines` compiles every
// non-specializing `TypedSurfaces` entry and force-unwraps it, so the resolve + the sync
// `createRenderPipeline` (the
// 0a/1c spike finding) validate against the real device at warm — a malformed group split, a name
// collision, a binding-limit breach all throw there, not mid-frame at first draw.
export interface CompiledTyped {
    /** exact registry spec + layout this entry was compiled from; name/variant alone cannot distinguish
     * a same-name replacement after warm (or an in-place layout swap). */
    owner: AnyTypedSurface;
    layout: SurfaceLayout<Record<string, TypedBinding>>;
    color: TgpuRenderPipeline<d.Vec4f> | null;
    transparent: TgpuRenderPipeline<d.Vec4f> | null;
    // the prepass pipelines for this typed surface, keyed like the legacy `Compiled.prepass` map
    // (`laneKey` — `""` the position-only depth pipeline, `"tag"` the id lane); empty for a `blend:
    // "alpha"` surface (a transparent pixel has no single owner, writes no prepass depth, casts nothing —
    // the same legacy rule `compileVariant` applies). Compiled off `layout.depthVariant` — a DISTINCT
    // `TgpuBindGroupLayout` object from `layout` (the c-2 caching verdict), so a draw-time bind-group
    // cache for these needs its own key space, never `layout`'s (`TypedGroupEntry.depth`, `record`).
    // the map holds both the depth-only (`Void` output) and the tag-lane (`u32` output) pipeline under one
    // key space — the same widening the legacy `Compiled.prepass: Map<string, GPURenderPipeline>` uses (a
    // bare `GPURenderPipeline` has no output type parameter either)
    prepass: Map<string, TgpuRenderPipeline<any>>;
    // the point/cascade shadow-atlas pipelines (4a-ii-c-3a-3) — `pointShadowCode`'s typed twin. `null` for a
    // non-instanced surface (only an instanced surface casts, the `castable` law `compileVariant` applies) —
    // never a silent gap, since a non-instanced typed surface has no per-instance `eids`/`transforms` to
    // re-gather against in the first place.
    point: TgpuRenderPipeline<any> | null;
    cascade: TgpuRenderPipeline<any> | null;
    // the single-sample (AA-off) color twin, compiled lazily by `ensureTypedSingle` the first frame a
    // no-AA camera draws this surface — the legacy `Compiled.single` shape, sync here (typegpu pipeline
    // wrappers are cheap; the real resolve+create defers to first draw regardless)
    single: {
        color: TgpuRenderPipeline<d.Vec4f> | null;
        transparent: TgpuRenderPipeline<d.Vec4f> | null;
    } | null;
    // the fixed inputs `ensureTypedSingle` re-compiles the 1× twin from (the legacy `ColorArgs` shape —
    // the entry fns are reused, so the twin shares the authored vs/fs, differing only in multisample)
    args: {
        vertex: ReturnType<typeof typedColorVs>;
        fragment: ReturnType<typeof typedColorFs>;
        blend: TypedSurface["blend"];
        // the raster state the 4× twin compiled with — carries the `screen` cull decision, which
        // `ensureTypedSingle` can't re-derive (it never sees the surface)
        primitive: GPUPrimitiveState;
        name: string;
    };
}
const _compiledTyped = new Map<string, CompiledTyped>();

/** the per-draw group-2 state a typed draw binds (4a-ii-c-3b) — the typed twin of {@link GroupEntry}.
 * `color` builds against `layout` (the 16 B main stream at the `vertices` slot); opaque depth/atlas
 * groups use the DISTINCT `layout.depthVariant` object (the 8 B position stream), while clipped groups
 * use the full layout/main stream because their fragment cutoff consumes material UVs. `point`/`cascade`
 * swap the `eids` lane to that atlas's re-gathered packed list (the legacy `eidsSwap`). View rides group
 * 0 per slot, so this state is slot-independent — one
 * instance per draw. `engineCache` holds the draw's engine group-0 instances per view slot (lazy, the
 * legacy `colorCache`/`prepassCache` lifetime: entry-scoped so a quant-buffer churn — a glTF import's
 * per-import buffers — drops the old groups with the overwritten entry, never a module map keyed on
 * buffer identity that grows for the app's life). `atlasG0` is its prebuilt slot-0 instance the
 * shadow-atlas passes bind (slot 0's View buffer as an unread placeholder — the atlas VS projects by
 * its own tile viewProj, never `view`). */
export type TypedGroupEntry = {
    /** exact registry spec this group was built for — resource identity alone is insufficient when a
     * same-name surface replacement carries a different layout with the same buffers. */
    owner: AnyTypedSurface;
    layout: SurfaceLayout<Record<string, TypedBinding>>;
    quant: GPUBuffer;
    color: GPUBindGroup;
    depth: GPUBindGroup | null;
    point: GPUBindGroup | null;
    cascade: GPUBindGroup | null;
    engineCache: Map<number, GPUBindGroup>;
    atlasG0: GPUBindGroup;
    resources: BindResource[];
};
const _typedGroups = new Map<string, TypedGroupEntry>();

/** the cached typed per-draw group-2 state for a Draw name, or `undefined` on a cache miss (`record`
 * rebuilds it). Supplying the current surface also invalidates a same-name replacement: bind groups
 * are layout-object-specific even when every resolved GPU resource is unchanged. */
export function getTypedGroup(
    name: string,
    surface?: AnyTypedSurface,
): TypedGroupEntry | undefined {
    const entry = _typedGroups.get(name);
    if (entry && surface && (entry.owner !== surface || entry.layout !== surface.layout)) {
        _typedGroups.delete(name);
        return undefined;
    }
    return entry;
}

/** cache a typed draw's resolved group-2 state (`record`, on a resource-identity change). */
export function setTypedGroup(name: string, entry: TypedGroupEntry): void {
    _typedGroups.set(name, entry);
}

/** the engine group-0 bind group for a view slot against one meshQuant buffer — the shared live
 * `engineLayout` instance a typed draw at that slot binds (frame / per-slot View / lighting /
 * light-cull outputs / the dequant table), built lazily into the caller-owned `cache` (a
 * `TypedGroupEntry.engineCache` or a `CompiledTypedBg.engineCache` — never a module map keyed on the
 * quant buffer, whose entries would outlive a churned buffer for the app's life). */
export function typedEngineGroup(
    cache: Map<number, GPUBindGroup>,
    slot: number,
    quant: GPUBuffer,
): GPUBindGroup {
    const cached = cache.get(slot);
    if (cached) return cached;
    const group = Compute.root.unwrap(
        Compute.root.createBindGroup(engineLayout, {
            frame: Frame.buffer!,
            view: Render.viewBuffers[slot],
            lighting: Lighting.buffer!,
            pointLights: LightCull.lights!,
            lightGrid: LightCull.grid!,
            lightIndices: LightCull.indices!,
            meshQuant: quant,
        }),
    );
    cache.set(slot, group);
    return group;
}

// a background reads no mesh, but `engineLayout` (the shared group-0 instance the Backgrounds lock
// names) still carries the `meshQuant` slot — a one-record placeholder buffer fills it, never read (the
// slot-0 View placeholder precedent, `record`'s eidsSwap)
let _bgQuant: GPUBuffer | null = null;

/** the never-read `meshQuant` placeholder a typed background's engine group binds. */
export function bgQuant(): GPUBuffer {
    _bgQuant ??= Compute.device.createBuffer({
        label: "sear-bg-quant",
        size: d.sizeOf(MeshQuant),
        usage: GPUBufferUsage.STORAGE,
    });
    return _bgQuant;
}

/** the widest `TypedSurface` shape (any bindings, any varyings) — the bare `TypedSurface` default pins
 *  varyings to `Record<string, never>`, so every internal typed-pipeline helper takes this wider alias to
 *  accept a real varyings-carrying surface (`vertex`) at the call boundary. */
type AnyTypedSurface = TypedSurface<Record<string, TypedBinding>, Record<string, AnyWgslData>>;

function typedVariant<
    B extends Record<string, TypedBinding>,
    V extends Record<string, AnyWgslData>,
>(surface: TypedSurface<B, V>, variant: number): TypedSurface<B, V> {
    const spec = surface.specialize?.(variant);
    return (spec ? { ...surface, ...spec, specialize: undefined } : surface) as TypedSurface<B, V>;
}

const identityXform = tgpu
    .fn(
        [],
        Xform,
    )(() => {
        "use gpu";
        return Xform({ pos: d.vec3f(0), quat: d.vec4f(0, 0, 0, 1), scale: d.vec3f(1) });
    })
    .$name("identityXform");

// the first interstage location a custom varying pins to — after the five fixed non-builtin fields
// (worldNormal/eid/world/uv/localPos at 0–4); gpu.md's 4-slot custom budget keeps 5+ within the 16 cap
const VARYING_BASE = 5;

// gpu.md rule 9's hard budget: 4 custom interpolator slots per surface. The vs side is N-general (the
// copier templates over `Object.keys`), so this bound is the fragment entry's — its transpiled body must
// statically name `input.v0`…`input.v3`, one arm per count (`typedVaryingFs`)
const MAX_VARYINGS = 4;

/** the raster state every typed surface pipeline shares (color, its single-sample twin, prepass, atlas).
 * A `screen` surface builds its own quads in clip space (lines), so their winding flips with segment
 * direction and back-face culling would drop half of them; world-space surfaces keep the cull (the
 * overdraw win + correct cutout/shadow facing) — `compileVariant`'s own law, one source of truth here so
 * the sites can't drift. */
export function surfacePrimitive(screen?: boolean): GPUPrimitiveState {
    return { topology: "triangle-list", cullMode: screen ? "none" : "back", frontFace: "ccw" };
}

/** whether a typed surface's own `layout` carries the `eids` + `transforms` instancing convention — the
 * typed twin of `record()`/`surfaceCode`'s test over a legacy `Surface`'s `bindings`, run over the typed
 * layout's `entries` instead. */
function typedInstanced(surface: AnyTypedSurface): boolean {
    return "eids" in surface.layout.entries && "transforms" in surface.layout.entries;
}

/**
 * the typed color-pass vertex entry: pull the quantized 16 B vertex from the surface's own `vertices` slot
 * (group 2 — the design-lock move off the engine group), decode it against `engineLayout`'s shared
 * `meshQuant` table, apply the standard instance transform when the surface declares `eids` + `transforms`
 * (`INSTANCE_VS`'s typed twin), then splice the surface's own `vs` chunk when it has one (varying-free —
 * see the module header). `uv`/`localPos` cross for real (c-3) — every typed surface pays the two
 * interpolator slots this stage, unlike the raw path's per-surface prune (gpu.md rule 9).
 *
 * A `screen` surface projects itself: its `vs` writes `patch.clip` and that value IS the clip position
 * (`surfaceCode`'s `out.clip = clipPos`), so nothing else supplies one — hence
 * {@link compileTypedVariant}'s screen-without-vs guard.
 */
function typedColorVs(surface: AnyTypedSurface) {
    const instanced = typedInstanced(surface);
    const screen = !!surface.screen;
    // captured as a plain boolean (not `surface.vs` itself) — the outline `maskFragment` "captured JS
    // boolean" precedent needs a real `boolean`, not an `undefined`-valued reference, to fold the branch
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout;
    // `TypedSurface`'s `layout` carries the wide, unconstrained default `B = Record<string, Binding>`
    // (the registry erases a registrant's own concrete binding-name generic, `contract.ts`'s `register`),
    // so `layout.$.<name>` for a consumer-declared name (not the fixed `vertices` field) can't narrow past
    // the mapped type's value union — a host-side cast, checked instead by `typedInstanced` above (the
    // runtime shape a `default`-shaped surface actually registers).
    const bound = layout.$ as unknown as { eids: any[]; transforms: any[] };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                world: d.vec3f,
                // real crossings (c-3) — c-2 zero-filled both on the fs side; TGSL has no object-spread
                // (validated: `tinyest-for-wgsl` throws "Spread elements are not supported in TGSL" on a
                // dynamic-key return), so a single shared vs/fs pair can't vary its interstage struct shape
                // per surface's declared `varyings` — but `uv`/`localPos` are c-1's own FIXED base fields
                // (`fsCtxSchema`'s two non-varying members), not part of that dynamic set, so crossing them
                // unconditionally needs no genericity. Unlike the raw path's per-surface prune (gpu.md rule
                // 9 — cross only when the fs reads them), the typed template crosses both for every surface
                // this stage: correctness over the bandwidth optimization, which the raw path's prune
                // precedent shows is safe to defer (`default`'s fs reads neither, so its cost here is two
                // always-unread interpolator slots — a real, disclosed regression from the raw path,
                // acceptable until the template learns to prune per surface)
                uv: d.vec2f,
                localPos: d.vec3f,
            },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            const localNormal = octDecodeNormal(v.z);
            const uv = decodeUv(v.w, mq);
            let eid = d.u32(0);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            let xform = identityXform();
            if (instanced) {
                eid = bound.eids[input.iid];
                xform = Xform(bound.transforms[eid]);
                world = d.vec4f(xformPoint(xform, world.xyz), world.w);
                worldNormal = d.vec3f(xformNormal(xform, worldNormal));
            }
            let clip = d.vec4f(0);
            if (hasVs) {
                // `hasVs` folds this branch at build time — `vsFn` is defined whenever the branch is live
                const patched = vsFn!(
                    VsIn({
                        localPos,
                        localNormal,
                        uv,
                        vidx: input.vidx,
                        eid,
                        iid: input.iid,
                        xform,
                        world,
                        worldNormal,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
                // `screen` folds this the same way — the patch's `clip` lane is the surface's own
                // projection, unread (and left zero-filled by every world-space consumer) otherwise
                if (screen) clip = d.vec4f(patched.clip);
            }
            if (!screen) clip = d.vec4f(std.mul(engineLayout.$.view.viewProj, world));
            return {
                pos: clip,
                worldNormal: std.normalize(worldNormal),
                eid,
                world: world.xyz,
                uv,
                localPos,
            };
        })
        .$name(`${surface.name}Vs`);
}

/**
 * the typed color-pass fragment entry: fills the four `sear/engine.ts` shading-seam privateVars exactly
 * as the raw scaffold's `fragmentBody` does (`sunVisibility` via a real {@link sampleSunShadow} call —
 * matching the raw path's inline sample — `fragWorld`, `fragCoord`, `pointScale`), builds the surface's
 * `fsCtxSchema` context (`uv`/`localPos` cross for real from the vs — c-3), and returns the surface's own
 * `fs` chunk's result verbatim (sear's `col` return,
 * unwrapped — a typed `fs` already returns `vec4f`, no lane locals: the prepass tag/depth lanes are a
 * separate pipeline, still unported).
 */
function typedColorFs(surface: AnyTypedSurface) {
    // hoisted outside the "use gpu" body — `fsCtxSchema()` is a plain host function (no varyings this
    // stage), and a call inside GPU code must resolve to a `tgpu.fn`/schema value, never a bare JS call
    const CtxSchema = fsCtxSchema();
    return tgpu
        .fragmentFn({
            in: {
                pos: d.builtin.position,
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                world: d.vec3f,
                // real crossings (c-3) — see `typedColorVs`'s matching fields for the why
                uv: d.vec2f,
                localPos: d.vec3f,
            },
            out: d.vec4f,
        })((input) => {
            "use gpu";
            const worldNormal = std.normalize(input.worldNormal);
            sunVisibility.$ = sampleSunShadow(input.world, worldNormal);
            fragWorld.$ = d.vec3f(input.world);
            fragCoord.$ = d.vec4f(input.pos);
            pointScale.$ = 1;
            // force `shadowLayoutTyped`'s group-1 bindings into scope: `sampleSunShadow`/the transitively
            // pulled `pointShadowOf` (via `litPbr`) read `shadowMap`/`shadowSamp`/`sunShadow`/`pointAtlas`/
            // `pointShadows`/`tileRects` as free names inside their own WGSL bodies, invisible to
            // `tgpu.resolve`'s call-graph walk (the fog `fogKernel` forcedZero precedent) — folded into a
            // value the return genuinely uses, not a discarded local (the JS→WGSL transpiler would prune
            // a dead one)
            const forcedZero =
                (shadowLayoutTyped.$.pointShadows.casters[0].pos.x +
                    shadowLayoutTyped.$.tileRects.rects[0].x +
                    shadowLayoutTyped.$.sunShadow.enabled +
                    std.textureSampleCompareLevel(
                        shadowLayoutTyped.$.pointAtlas,
                        shadowLayoutTyped.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    ) +
                    std.textureSampleCompareLevel(
                        shadowLayoutTyped.$.shadowMap,
                        shadowLayoutTyped.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    )) *
                0;
            // `fsCtxSchema()`'s no-varyings default type param (`Record<string, never>`) carries a
            // `[x: string]: never` index signature that a plain object literal's own known keys never
            // structurally satisfy (a TS quirk over a generic-default index signature, not a real type
            // error) — the escape is the same shape as the `layout.$` cast above.
            //
            // `uv`/`localPos` now cross for real (c-3) — resolved from the vs's own interpolated output,
            // not a zero-fill stand-in.
            const ctx = CtxSchema({
                eid: input.eid,
                world: input.world,
                worldNormal,
                uv: input.uv,
                localPos: input.localPos,
            } as any);
            const col = surface.fs(ctx);
            return d.vec4f(std.add(col, d.vec4f(forcedZero)));
        })
        .$name(`${surface.name}Fs`);
}

/**
 * the position-only typed prepass vertex entry (empty lane set — the shadow map's own shape too): pulls
 * the 8 B position-only vertex from the surface's `layout.depthVariant` (a DISTINCT `TgpuBindGroupLayout`
 * instance from `layout` — the c-2 caching verdict), decodes position alone (normal defaults `+Z`, uv `0`
 * — the raw prepass's own shape, `pass === "prepass"` in `codegen.ts`'s `surfaceCode`), applies the
 * standard instance transform, then splices the surface's own `vs` chunk when present. Inlined rather than
 * factored through a shared helper (probed live: a plain function marked `"use gpu"` can't take a host
 * object like `surface` as an argument — "Shellless functions can only accept arguments representing WGSL
 * resources" — so this duplicates {@link typedTagVs}'s math, matching `typedColorVs`'s own inline shape
 * rather than inventing a new factoring pattern this file doesn't otherwise use).
 */
function typedPrepassVs(surface: AnyTypedSurface) {
    const instanced = typedInstanced(surface);
    const screen = !!surface.screen;
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as { eids: any[]; transforms: any[] };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            // the raw prepass's own default (`codegen.ts`'s `surfaceCode`, `pass === "prepass"`) — pinned
            // for the life of the vs, never touched by the instance transform below (`INSTANCE_VS` mutates
            // only `world`/`worldNormal`); a `vs` chunk reading `vsIn.localNormal` must see this default,
            // not the transformed `worldNormal` (a real bug caught in review — passing `worldNormal` here
            // silently fed world-space data into a field the raw path documents as local-space)
            const localNormal = d.vec3f(0, 0, 1);
            const uv = d.vec2f(0, 0);
            let eid = d.u32(0);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            let xform = identityXform();
            if (instanced) {
                eid = bound.eids[input.iid];
                xform = Xform(bound.transforms[eid]);
                world = d.vec4f(xformPoint(xform, world.xyz), world.w);
                worldNormal = d.vec3f(xformNormal(xform, worldNormal));
            }
            let clip = d.vec4f(0);
            if (hasVs) {
                const patched = vsFn!(
                    VsIn({
                        localPos,
                        localNormal,
                        uv,
                        vidx: input.vidx,
                        eid,
                        iid: input.iid,
                        xform,
                        world,
                        worldNormal,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
                if (screen) clip = d.vec4f(patched.clip);
            }
            if (!screen) clip = d.vec4f(std.mul(engineLayout.$.view.viewProj, world));
            // force ONE group-1 binding into scope: typegpu's pipeline layout is group-indexed
            // (`usedBindGroupLayouts[idx]`), and a hole at 1 (groups 0 + 2 used, 1 untouched) emits a
            // sparse layout Dawn rejects at createRenderPipeline — a real read, folded to zero, keeps
            // the prepass pipelines' group set dense (`renderPrepass` binds `shadowGroupTyped()` at 1,
            // inert: the stub receiver means nothing samples it)
            const forcedZero = shadowLayoutTyped.$.tileRects.rects[0].x * 0;
            return { pos: std.add(clip, d.vec4f(forcedZero)) };
        })
        .$name(`${surface.name}PrepassVs`);
}

/** the id-lane typed prepass vertex entry: {@link typedPrepassVs}'s twin, crossing the flat `eid` varying
 * the tag fragment ({@link typedTagFs}) writes verbatim — `instanced ? eid : TAG_NONE`, `COLOR_LANES`'s
 * own tag default (`codegen.ts`). */
function typedTagVs(surface: AnyTypedSurface) {
    const instanced = typedInstanced(surface);
    const screen = !!surface.screen;
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as { eids: any[]; transforms: any[] };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position, eid: d.interpolate("flat", d.u32) },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            // pinned default — never touched by the instance transform, matching `typedPrepassVs`'s law
            const localNormal = d.vec3f(0, 0, 1);
            const uv = d.vec2f(0, 0);
            let eid = d.u32(instanced ? 0 : TAG_NONE);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            let xform = identityXform();
            if (instanced) {
                eid = bound.eids[input.iid];
                xform = Xform(bound.transforms[eid]);
                world = d.vec4f(xformPoint(xform, world.xyz), world.w);
                worldNormal = d.vec3f(xformNormal(xform, worldNormal));
            }
            let clip = d.vec4f(0);
            if (hasVs) {
                const patched = vsFn!(
                    VsIn({
                        localPos,
                        localNormal,
                        uv,
                        vidx: input.vidx,
                        eid,
                        iid: input.iid,
                        xform,
                        world,
                        worldNormal,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
                if (screen) clip = d.vec4f(patched.clip);
            }
            if (!screen) clip = d.vec4f(std.mul(engineLayout.$.view.viewProj, world));
            // same group-1 hole fill as `typedPrepassVs` — the tag pipeline shares the prepass's
            // groups-0+2 shape
            const forcedZero = shadowLayoutTyped.$.tileRects.rects[0].x * 0;
            return { pos: std.add(clip, d.vec4f(forcedZero)), eid };
        })
        .$name(`${surface.name}PrepassTagVs`);
}

/**
 * the typed tag-lane fragment entry: the front-most fragment's `eid` verbatim ({@link typedTagVs}'s
 * varying already resolved the instanced/non-instanced default — `COLOR_LANES`'s own tag-init shape,
 * `codegen.ts`). Outputs `vec4u` rather than the raw path's bare `u32` — typegpu's `fragmentFn` constrains
 * every color output to a `vec4` family type (`FragmentOutConstrained`'s `FragmentColorValue = Vec4f |
 * Vec4i | Vec4u`, probed live: a bare `d.u32` fails the type constraint before the body even resolves), so
 * the eid rides lane 0 with the other three padded zero — WebGPU spec-legal against the single-channel
 * `r32uint` target (a fragment output may carry more components than the attachment's format; the excess
 * are dropped), and a real, disclosed WGSL-shape deviation the differential test must account for. No fs
 * override this stage — a typed surface authoring its own tag (terrain's `capacity + cell` shape) is a
 * real capability {@link compileTypedVariant} doesn't carry yet (the sprite migration's open contract
 * question).
 */
function typedTagFs(surface: AnyTypedSurface) {
    return tgpu
        .fragmentFn({ in: { eid: d.interpolate("flat", d.u32) }, out: d.vec4u })((input) => {
            "use gpu";
            return d.vec4u(input.eid, 0, 0, 0);
        })
        .$name(`${surface.name}PrepassTagFs`);
}

function typedClipVs(surface: AnyTypedSurface) {
    const instanced = typedInstanced(surface);
    const screen = !!surface.screen;
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    // A clipped depth/tag fragment executes the authored material cutoff, so it needs the same
    // UV/normal-bearing vertex stream as color. Opaque prepass variants keep the compact depth stream.
    const layout = surface.layout;
    const bound = layout.$ as unknown as { eids: any[]; transforms: any[] };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                world: d.vec3f,
                uv: d.vec2f,
                localPos: d.vec3f,
            },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            const localNormal = octDecodeNormal(v.z);
            const uv = decodeUv(v.w, mq);
            // Material indexing follows the color/legacy path: a non-instanced surface owns slot zero.
            // The tag attachment's no-owner sentinel is selected only by typedClipFs's tag output.
            let eid = d.u32(0);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            let xform = identityXform();
            if (instanced) {
                eid = bound.eids[input.iid];
                xform = Xform(bound.transforms[eid]);
                world = d.vec4f(xformPoint(xform, world.xyz), world.w);
                worldNormal = d.vec3f(xformNormal(xform, worldNormal));
            }
            let clip = d.vec4f(0);
            if (hasVs) {
                const patched = vsFn!(
                    VsIn({
                        localPos,
                        localNormal,
                        uv,
                        vidx: input.vidx,
                        eid,
                        iid: input.iid,
                        xform,
                        world,
                        worldNormal,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
                if (screen) clip = d.vec4f(patched.clip);
            }
            if (!screen) clip = d.vec4f(std.mul(engineLayout.$.view.viewProj, world));
            const forcedZero = shadowLayoutTyped.$.tileRects.rects[0].x * 0;
            return {
                pos: std.add(clip, d.vec4f(forcedZero)),
                worldNormal: std.normalize(worldNormal),
                eid,
                world: world.xyz,
                uv,
                localPos,
            };
        })
        .$name(`${surface.name}ClipVs`);
}

function typedClipFs(surface: AnyTypedSurface, tag: boolean) {
    const instanced = typedInstanced(surface);
    const Ctx = (surface.fs as unknown as { shell: { argTypes: [ReturnType<typeof fsCtxSchema>] } })
        .shell.argTypes[0];
    const input = {
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
    };
    if (tag) {
        return tgpu
            .fragmentFn({ in: input, out: d.vec4u })((fin) => {
                "use gpu";
                const ctx = Ctx({
                    eid: fin.eid,
                    world: fin.world,
                    worldNormal: std.normalize(fin.worldNormal),
                    uv: fin.uv,
                    localPos: fin.localPos,
                });
                surface.fs(ctx);
                return d.vec4u(instanced ? fin.eid : TAG_NONE, 0, 0, 0);
            })
            .$name(`${surface.name}ClipTagFs`);
    }
    return tgpu
        .fragmentFn({ in: input, out: d.Void })((fin) => {
            "use gpu";
            const ctx = Ctx({
                eid: fin.eid,
                world: fin.world,
                worldNormal: std.normalize(fin.worldNormal),
                uv: fin.uv,
                localPos: fin.localPos,
            });
            surface.fs(ctx);
        })
        .$name(`${surface.name}ClipFs`);
}

/** Build the locked one-varying cutoff copier: the fragment entry exposes a fixed `v0` slot while this
 * raw helper reconstructs the author's exact FsCtx positionally. The schema comes from the authored fs
 * shell, never a freshly minted lookalike. */
function clipVaryingCopier(surface: AnyTypedSurface) {
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    if (keys.length !== 1) {
        throw new Error(
            `sear: typed surface "${surface.name}" declares ${keys.length} varyings — the typed clip copier carries exactly one custom varying`,
        );
    }
    const varyingSchema = varyings[keys[0]];
    const varyingType = (varyingSchema as unknown as { type: string }).type;
    const fsFn = surface.fs;
    const CtxSchema = (fsFn as unknown as { shell: { argTypes: [unknown] } }).shell.argTypes[0];
    const copier = tgpu
        .fn(
            [d.vec3f, d.u32, d.vec3f, d.vec2f, d.vec3f, varyingSchema],
            d.Void,
        )(/* wgsl */ `(worldNormalIn: vec3f, eid: u32, world: vec3f, uv: vec2f, localPos: vec3f, v0: ${varyingType}) {
    let ctx = Ctx(eid, world, normalize(worldNormalIn), uv, localPos, v0);
    fs(ctx);
}`)
        .$uses({ Ctx: CtxSchema, fs: fsFn })
        .$name(`${surface.name}ClipFsCopier`);
    return { varyingSchema, copier };
}

function varyingClipFs(surface: AnyTypedSurface, tag: boolean) {
    const instanced = typedInstanced(surface);
    const { varyingSchema, copier } = clipVaryingCopier(surface);
    const entryIn = {
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        v0: d.location(VARYING_BASE, varyingSchema as d.Vec3f),
    };
    if (tag) {
        return tgpu
            .fragmentFn({
                in: entryIn as unknown as typeof entryIn & { v0: d.Vec3f },
                out: d.vec4u,
            })((input) => {
                "use gpu";
                copier(
                    input.worldNormal,
                    input.eid,
                    input.world,
                    input.uv,
                    input.localPos,
                    input.v0,
                );
                return d.vec4u(instanced ? input.eid : TAG_NONE, 0, 0, 0);
            })
            .$name(`${surface.name}ClipTagFs`);
    }
    return tgpu
        .fragmentFn({
            in: entryIn as unknown as typeof entryIn & { v0: d.Vec3f },
            out: d.Void,
        })((input) => {
            "use gpu";
            copier(input.worldNormal, input.eid, input.world, input.uv, input.localPos, input.v0);
        })
        .$name(`${surface.name}ClipFs`);
}

/**
 * the varyings-carrying typed color/prepass vertex entry (4a-ii-c-3a, the varyings mechanism lock): TGSL has
 * no object-spread and no dynamic-key struct construction, so a shared "use gpu" body can't vary its
 * return shape per surface — a surface declaring `varyings` gets its own **WGSL-bodied copier**, distinct
 * from `typedColorVs`'s shared body. The copier does the real vertex math (vertex pull, quantized decode,
 * instance transform, the surface's own `vs` chunk) AND constructs the whole per-surface out struct in one
 * raw fn, its `out.<varying> = patched.<varying>;` lines JS-string-templated from `Object.keys(surface.
 * varyings)` (never through the transpiler). The thin entry stays real TGSL (typegpu's own header/
 * location machinery) and only binds + returns the copier's result — `const r = copier(...); return r;`,
 * since a direct return hits "Cannot resolve struct cast" (the c-1/c-2 API laws, probed live via
 * `tgpu.resolve` before this landed).
 */
function typedVaryingVs(surface: AnyTypedSurface, clip = false) {
    const varyings = surface.varyings ?? {};
    const vsFn = surface.vs;
    if (!vsFn) {
        throw new Error(
            `sear: typed surface "${surface.name}" declares varyings with no vs — a varying can only be written by the surface's own vs chunk`,
        );
    }
    const instanced = typedInstanced(surface);
    // a `screen` surface's own projection is the patch's `clip` lane (`surfaceCode`'s `out.clip =
    // clipPos`); everything else projects `view.viewProj * world` after the chunk, so displacing `world`
    // still projects
    const screen = !!surface.screen;
    const layout = surface.layout;
    const bound = layout.bound as unknown as Record<string, unknown>;
    const OutStruct = d
        .struct({
            pos: d.vec4f,
            worldNormal: d.vec3f,
            eid: d.u32,
            world: d.vec3f,
            uv: d.vec2f,
            localPos: d.vec3f,
            // no type-directed `@interpolate(flat)` insertion — an INTEGER varying is unsupported and
            // fails loudly at resolve/device compile; every shipped varying is float-typed.
            ...varyings,
        })
        .$name(`${surface.name}VsOut`);
    const assigns = Object.keys(varyings)
        .map((k) => `    out.${k} = patched.${k};`)
        .join("\n");
    const uses: Record<string, unknown> = {
        Out: OutStruct,
        vertices: bound.vertices,
        meshQuant: engineLayout.bound.meshQuant,
        decodePos,
        decodeUv,
        meshIdOf,
        octDecodeNormal,
        xformPoint,
        xformNormal,
        Xform,
        VsIn,
        vs: vsFn,
    };
    // a `screen` copier projects nothing — `view` would resolve as an unused external (a warning plus a
    // dead group-0 declaration in the emitted module)
    if (!screen) uses.view = engineLayout.bound.view;
    if (clip) uses.tileRects = shadowLayoutTyped.bound.tileRects;
    if (instanced) {
        uses.eids = bound.eids;
        uses.transforms = bound.transforms;
    }
    const copier = tgpu
        .fn(
            [d.u32, d.u32],
            OutStruct,
        )(/* wgsl */ `(vidx: u32, iid: u32) -> Out {
    let v = vertices[vidx];
    let mq = meshQuant[meshIdOf(v.y)];
    let localPos = decodePos(v.x, v.y, mq);
    let localNormal = octDecodeNormal(v.z);
    let uv = decodeUv(v.w, mq);
    var eid: u32 = 0u;
    var xform = Xform(vec3f(0.0), vec4f(0.0, 0.0, 0.0, 1.0), vec3f(1.0));
    var world = vec4f(localPos, 1.0);
    var worldNormal = vec3f(localNormal);
${
    instanced
        ? `    eid = eids[iid];
    xform = transforms[eid];
    world = vec4f(xformPoint(xform, world.xyz), world.w);
    worldNormal = vec3f(xformNormal(xform, worldNormal));
`
        : ""
}    let patched = vs(VsIn(localPos, localNormal, uv, vidx, eid, iid, xform, world, worldNormal));
    world = patched.world;
    worldNormal = patched.worldNormal;
    var out: Out;
    out.pos = ${screen ? "patched.clip" : "view.viewProj * world"}${clip ? " + vec4f(tileRects.rects[0].x * 0.0)" : ""};
    out.worldNormal = normalize(worldNormal);
    out.eid = eid;
    out.world = world.xyz;
    out.uv = uv;
    out.localPos = localPos;
${assigns}
    return out;
}`)
        .$uses(uses)
        .$name(`${surface.name}${clip ? "Clip" : ""}Copier`);

    // explicit interstage locations from VARYING_BASE up: the fs entry carries the varying under the
    // fixed internal name `v0`, which typegpu's pipeline connection can't match to the vs side's real
    // name — an unmatched fs field auto-assigns from 0 and collides with the matched fixed fields.
    // Pinning both sides to the same explicit slot makes the location, not the name, the contract
    // (auto-assignment skips explicitly-taken locations)
    const located = Object.fromEntries(
        Object.entries(varyings).map(([k, s], i) => [k, d.location(VARYING_BASE + i, s)]),
    );
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                world: d.vec3f,
                uv: d.vec2f,
                localPos: d.vec3f,
                // same awareness as the copier's `OutStruct` above — no flat-interpolate insertion, so an
                // integer varying is unsupported.
                ...located,
            },
        })((input) => {
            "use gpu";
            const r = copier(input.vidx, input.iid);
            return r;
        })
        .$name(`${surface.name}${clip ? "Clip" : ""}Vs`);
}

/** the group-1 forcing touch as one callable fold: `sampleSunShadow` and the transitively-pulled
 * `pointShadowOf` read `shadowMap`/`shadowSamp`/`sunShadow`/`pointAtlas`/`pointShadows`/`tileRects` as free
 * names inside their own WGSL bodies, invisible to `tgpu.resolve`'s call-graph walk (the fog `fogKernel`
 * forcedZero precedent). {@link typedColorFs} and {@link typedBgFs} inline the same expression; the
 * varyings-carrying fragment entry calls this instead so its four arity arms don't each carry a copy. */
const shadowForce = tgpu
    .fn(
        [],
        d.f32,
    )(() => {
        "use gpu";
        return (
            (shadowLayoutTyped.$.pointShadows.casters[0].pos.x +
                shadowLayoutTyped.$.tileRects.rects[0].x +
                shadowLayoutTyped.$.sunShadow.enabled +
                std.textureSampleCompareLevel(
                    shadowLayoutTyped.$.pointAtlas,
                    shadowLayoutTyped.$.shadowSamp,
                    d.vec2f(0, 0),
                    0,
                ) +
                std.textureSampleCompareLevel(
                    shadowLayoutTyped.$.shadowMap,
                    shadowLayoutTyped.$.shadowSamp,
                    d.vec2f(0, 0),
                    0,
                )) *
            0
        );
    })
    .$name("shadowForce");

/**
 * the varyings-carrying typed color-pass fragment entry — `typedColorFs`'s twin for a surface declaring
 * `varyings` ({@link typedVaryingVs}'s matching half). typegpu's entry-input router can't cross a whole
 * `input` value into an ordinary function by value ("Cannot convert value of type 'entry-input-router'" —
 * probed live), and a real "use gpu" entry body can't read a per-surface dynamic field name either (the
 * same source-is-static constraint that forces the copier in the first place) — so the fs entry declares
 * its interstage varying slots under fixed internal names (`v0`…`v3`), independent of the surface's own
 * varying keys, and passes every field to the copier **positionally** (fixed base fields, then the slots in
 * declaration order). The copier's raw body constructs the real `fsCtxSchema`-shaped ctx via a
 * **positional** struct-constructor call (`Ctx(eid, world, worldNormal, uv, localPos, v0, …)` — WGSL struct
 * constructors are positional, so each slot's value lands in the ctx's real varying field with no name
 * matching needed) before calling the surface's own `fs` chunk (`$uses`).
 *
 * The entry body is transpiled TGSL, so it must *statically* name each `input.v<i>` — hence the bounded
 * per-count dispatch below: one explicit arm per count, 1 through {@link MAX_VARYINGS}, and a loud throw
 * past it (gpu.md rule 9's hard 4-slot custom interpolator budget). Everything else — the copier's
 * signature, the interstage locations, the vs side — is already N-general.
 *
 * The `Ctx` the copier constructs must be the exact schema instance `surface.fs` was declared against —
 * `fsCtxSchema(varyings)` mints a fresh struct object on every call (the `pointCastersSchema`/
 * `tileRectsSchema` precedent), so re-minting one here would pass a structurally-identical but
 * distinct-identity struct into `fs`'s call, which resolved clean device-free but is real risk at device
 * compile (WGSL struct-argument typing isn't purely structural) — reading it off `fsFn.shell.argTypes[0]`
 * (the schema the author's own `tgpu.fn([fsCtxSchema(...)], ...)` call recorded) is the one source of truth.
 */
function typedVaryingFs(surface: AnyTypedSurface) {
    const varyings = surface.varyings ?? {};
    const varyingKeys = Object.keys(varyings);
    if (varyingKeys.length < 1 || varyingKeys.length > MAX_VARYINGS) {
        throw new Error(
            `sear: typed surface "${surface.name}" declares ${varyingKeys.length} varyings — the typed fs entry carries 1 to ${MAX_VARYINGS} (gpu.md rule 9's custom interpolator budget)`,
        );
    }
    const schemas = varyingKeys.map((k) => varyings[k]);
    const params = schemas
        .map((s, i) => `v${i}: ${(s as unknown as { type: string }).type}`)
        .join(", ");
    const fsFn = surface.fs;
    const CtxSchema = (fsFn as unknown as { shell: { argTypes: [unknown] } }).shell.argTypes[0];
    const copier = tgpu
        .fn(
            [d.vec4f, d.vec3f, d.u32, d.vec3f, d.vec2f, d.vec3f, ...schemas],
            d.vec4f,
        )(/* wgsl */ `(pos: vec4f, worldNormalIn: vec3f, eid: u32, world: vec3f, uv: vec2f, localPos: vec3f, ${params}) -> vec4f {
    let worldNormal = normalize(worldNormalIn);
    let ctx = Ctx(eid, world, worldNormal, uv, localPos, ${schemas.map((_, i) => `v${i}`).join(", ")});
    return fs(ctx);
}`)
        .$uses({ Ctx: CtxSchema, fs: fsFn })
        .$name(`${surface.name}FsCopier`);

    // a slot's schema is the widened `AnyWgslData` — real (any concrete vector/scalar schema at runtime),
    // but too loose for `fragmentFn`'s `in:` constraint and the resulting `input`'s field types to
    // type-check without a cast, the same escape `typedColorVs`'s `layout.$` cast uses. The explicit
    // location pairs with the vs side's (`VARYING_BASE + i` — see typedVaryingVs's why: the internal
    // `v<i>` names are unmatchable, the slot is the contract). No flat-interpolate insertion, so an
    // integer varying is unsupported and fails loudly at resolve/device compile.
    const slot = (i: number) => d.location(VARYING_BASE + i, schemas[i] as d.Vec3f);
    const base = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
    };
    const name = `${surface.name}Fs`;
    if (varyingKeys.length === 1) {
        const entryIn = { ...base, v0: slot(0) } as unknown as typeof base & { v0: d.Vec3f };
        return tgpu
            .fragmentFn({ in: entryIn, out: d.vec4f })((input) => {
                "use gpu";
                const col = copier(
                    input.pos,
                    input.worldNormal,
                    input.eid,
                    input.world,
                    input.uv,
                    input.localPos,
                    input.v0,
                );
                return d.vec4f(std.add(col, d.vec4f(shadowForce())));
            })
            .$name(name);
    }
    if (varyingKeys.length === 2) {
        const entryIn = { ...base, v0: slot(0), v1: slot(1) } as unknown as typeof base & {
            v0: d.Vec3f;
            v1: d.Vec3f;
        };
        return tgpu
            .fragmentFn({ in: entryIn, out: d.vec4f })((input) => {
                "use gpu";
                const col = copier(
                    input.pos,
                    input.worldNormal,
                    input.eid,
                    input.world,
                    input.uv,
                    input.localPos,
                    input.v0,
                    input.v1,
                );
                return d.vec4f(std.add(col, d.vec4f(shadowForce())));
            })
            .$name(name);
    }
    if (varyingKeys.length === 3) {
        const entryIn = {
            ...base,
            v0: slot(0),
            v1: slot(1),
            v2: slot(2),
        } as unknown as typeof base & {
            v0: d.Vec3f;
            v1: d.Vec3f;
            v2: d.Vec3f;
        };
        return tgpu
            .fragmentFn({ in: entryIn, out: d.vec4f })((input) => {
                "use gpu";
                const col = copier(
                    input.pos,
                    input.worldNormal,
                    input.eid,
                    input.world,
                    input.uv,
                    input.localPos,
                    input.v0,
                    input.v1,
                    input.v2,
                );
                return d.vec4f(std.add(col, d.vec4f(shadowForce())));
            })
            .$name(name);
    }
    const entryIn = {
        ...base,
        v0: slot(0),
        v1: slot(1),
        v2: slot(2),
        v3: slot(3),
    } as unknown as typeof base & { v0: d.Vec3f; v1: d.Vec3f; v2: d.Vec3f; v3: d.Vec3f };
    return tgpu
        .fragmentFn({ in: entryIn, out: d.vec4f })((input) => {
            "use gpu";
            const col = copier(
                input.pos,
                input.worldNormal,
                input.eid,
                input.world,
                input.uv,
                input.localPos,
                input.v0,
                input.v1,
                input.v2,
                input.v3,
            );
            return d.vec4f(std.add(col, d.vec4f(shadowForce())));
        })
        .$name(name);
}

/**
 * compile a `TypedSurface`'s color-pass pipeline(s): the opaque `color` pipeline, or — for a `blend:
 * "alpha"` surface (c-3) — the single blended `transparent` pipeline instead (the legacy `colorPipelines`
 * split: exactly one of the two compiles, never both, matching the raw path's "one non-opaque pipeline"
 * shape). Cached by name + material variant plus exact source-surface/layout identity, so replacing a
 * registry entry after warm cannot inherit the previous owner's pipelines. A `screen` surface projects
 * through its own `vs` chunk (`patch.clip`) and rasterizes un-culled; `"clip"` and a surface's
 * `specialize` factory are resolved here for the glTF typed migration.
 */
export function compileTypedVariant<
    B extends Record<string, TypedBinding>,
    V extends Record<string, AnyWgslData>,
>(surface: TypedSurface<B, V>, variant = 0): CompiledTyped {
    const key = variantKey(surface.name, surface.specialize ? variant : 0);
    const cached = _compiledTyped.get(key);
    if (cached?.owner === surface && cached.layout === surface.layout) return cached;
    const resolved = typedVariant(surface, variant);
    // a `screen` surface's clip position comes from its own `vs` chunk's `patch.clip` and from nowhere
    // else — with no `vs` every vertex would collapse to the origin, silently drawing nothing
    if (resolved.screen && !resolved.vs) {
        throw new Error(
            `sear: typed surface "${surface.name}" is a screen surface with no vs — only its own vs chunk can supply the clip position`,
        );
    }
    const primitive = surfacePrimitive(resolved.screen);
    // TS can't narrow `vertex`/`fragment` as a matched pair across the ternary (their varying-record types
    // only agree structurally, proven at runtime by the differential + bench gates, not by the branch's
    // static shape) — the same class of escape `typedColorVs`'s `layout.$` cast uses elsewhere in this file.
    const hasVaryings = !!resolved.varyings && Object.keys(resolved.varyings).length > 0;
    const vertex = (hasVaryings ? typedVaryingVs(resolved) : typedColorVs(resolved)) as ReturnType<
        typeof typedColorVs
    >;
    const fragment = (
        hasVaryings ? typedVaryingFs(resolved) : typedColorFs(resolved)
    ) as ReturnType<typeof typedColorFs>;
    const args: CompiledTyped["args"] = {
        vertex,
        fragment,
        blend: resolved.blend,
        primitive,
        name: `${surface.name}#${variant}`,
    };
    let compiled: CompiledTyped;
    if (resolved.blend === "alpha") {
        const transparent = Compute.root
            .createRenderPipeline({
                vertex,
                fragment,
                targets: { format: Render.format, blend: ALPHA_BLEND },
                primitive,
                depthStencil: {
                    format: DEPTH_FORMAT,
                    depthWriteEnabled: false,
                    depthCompare: "greater-equal",
                },
                multisample: { count: SAMPLE_COUNT },
            })
            .$name(`sear-typed-transparent-${surface.name}`);
        // `blend: "alpha"` casts nothing (a transparent pixel has no single owner, `compileVariant`'s own
        // rule) — the same reason its prepass map stays empty
        compiled = {
            owner: surface as AnyTypedSurface,
            layout: surface.layout as SurfaceLayout<Record<string, TypedBinding>>,
            color: null,
            transparent,
            prepass: new Map(),
            point: null,
            cascade: null,
            single: null,
            args,
        };
    } else {
        const color = Compute.root
            .createRenderPipeline({
                vertex,
                fragment,
                targets: { format: Render.format },
                primitive,
                depthStencil: {
                    format: DEPTH_FORMAT,
                    depthWriteEnabled: true,
                    depthCompare: "greater",
                },
                multisample: { count: SAMPLE_COUNT },
            })
            .$name(`sear-typed-${surface.name}`);
        compiled = {
            owner: surface as AnyTypedSurface,
            layout: surface.layout as SurfaceLayout<Record<string, TypedBinding>>,
            color,
            transparent: null,
            prepass: new Map(),
            point: null,
            cascade: null,
            single: null,
            args,
        };
    }
    compiled.prepass = compileTypedPrepass(resolved);
    if (resolved.blend !== "alpha") {
        const { point, cascade } = compileTypedShadow(resolved);
        compiled.point = point;
        compiled.cascade = cascade;
    }
    _compiledTyped.set(key, compiled);
    return compiled;
}

/**
 * compile a typed surface's single-sample (AA-off) color twin, once, the first frame a no-AA camera
 * draws it — the legacy {@link ensureSingle}'s typed twin, sync (the wrapper is cheap; the real
 * resolve+create lands at the twin's first draw). Reuses the compiled entry fns, so only
 * `multisample.count` differs.
 */
export function ensureTypedSingle(t: CompiledTyped): void {
    if (t.single) return;
    const { vertex, fragment, blend, primitive, name } = t.args;
    if (blend === "alpha") {
        const transparent = Compute.root
            .createRenderPipeline({
                vertex,
                fragment,
                targets: { format: Render.format, blend: ALPHA_BLEND },
                primitive,
                depthStencil: {
                    format: DEPTH_FORMAT,
                    depthWriteEnabled: false,
                    depthCompare: "greater-equal",
                },
                multisample: { count: 1 },
            })
            .$name(`sear-typed-transparent-${name}-1x`);
        t.single = { color: null, transparent };
        return;
    }
    const color = Compute.root
        .createRenderPipeline({
            vertex,
            fragment,
            targets: { format: Render.format },
            primitive,
            depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: true,
                depthCompare: "greater",
            },
            multisample: { count: 1 },
        })
        .$name(`sear-typed-${name}-1x`);
    t.single = { color, transparent: null };
}

/**
 * compile a `TypedSurface`'s prepass pipelines (4a-ii-c-3a-2): the position-only depth pipeline (key `""`,
 * vertex-only except for a `clip` surface's cutoff fragment) and the id-lane pipeline (key `"tag"`, the
 * raw `COLOR_LANES` id lane). Opaque surfaces compile off `layout.depthVariant`; clipped surfaces execute
 * their authored cutoff and therefore use the full layout/main vertex stream. `TypedGroupEntry.depth`
 * mirrors that split. A `blend: "alpha"` surface casts no prepass at all — same rule
 * `compileVariant` applies (a transparent pixel has no single owner, writes no prepass depth) — so its map
 * stays empty.
 */
function compileTypedPrepass(surface: AnyTypedSurface): Map<string, TgpuRenderPipeline<any>> {
    const prepass = new Map<string, TgpuRenderPipeline<any>>();
    if (surface.blend === "alpha") return prepass;
    const primitive = surfacePrimitive(surface.screen);
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    // the receiver stub bound per pipeline (`SHADOW_STUB_WGSL`'s typed twin): a vs-chunk surface's
    // `litPbr` statically reaches `pointShadowOf`, whose free names the depth passes never declare or
    // bind — the stub keeps these modules group-0/2-only, exactly like the raw prepass module
    const root = Compute.root.with(pointShadowSlot, pointShadowStub);
    const clip = surface.blend === "clip";
    const varying = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const depthOnly = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? typedVaryingVs(surface, true)
                    : typedClipVs(surface)
                : typedPrepassVs(surface),
            ...(clip
                ? {
                      fragment: (varying
                          ? varyingClipFs(surface, false)
                          : typedClipFs(surface, false)) as never,
                  }
                : {}),
            primitive,
            depthStencil,
        })
        .$name(`sear-typed-prepass-${surface.name}`);
    prepass.set("", depthOnly);
    const tag = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? typedVaryingVs(surface, true)
                    : typedClipVs(surface)
                : typedTagVs(surface),
            fragment: clip
                ? ((varying ? varyingClipFs(surface, true) : typedClipFs(surface, true)) as never)
                : typedTagFs(surface),
            targets: { format: TAG_FORMAT },
            primitive,
            depthStencil,
        })
        .$name(`sear-typed-prepass-tag-${surface.name}`);
    prepass.set("tag", tag);
    return prepass;
}

/**
 * the shared typed point/cascade fragment entry (4a-ii-c-3a-3): the tile-seam discard alone, `pointShadowCode`'s
 * `fsPoint` twin — clamped to `layout.depthVariant` position + the `tileBox` varying its matching vs writes.
 * Atlas-size-independent (the vs bakes the atlas scale into `tileBox` already), so ONE instance serves every
 * typed surface's point pipeline AND every typed surface's cascade pipeline (the raw path's `fsPoint` is
 * textually identical between the two atlases too — only the VS's rect-index formula + atlas constant
 * differ, `codegen.ts`'s `pointShadowCode`). A `clip` surface uses the wider per-surface fragment below so
 * the same material cutoff holes its atlas depth.
 */
const typedShadowFs = tgpu
    .fragmentFn({
        in: { pos: d.builtin.position, tileBox: d.interpolate("flat", d.vec4f) },
        out: d.Void,
    })((input) => {
        "use gpu";
        const p = input.pos.xy;
        const mn = input.tileBox.xy;
        const sz = input.tileBox.z;
        if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) {
            std.discard();
        }
    })
    .$name("shadowAtlasFs");

/**
 * the typed point/cascade shadow-atlas vertex entry (4a-ii-c-3a-3): `pointShadowCode`'s VS, pinned
 * statement-for-statement — pulls the 8 B position-only vertex from `layout.depthVariant` (the
 * `typedPrepassVs` shape), reads the re-gathered `(combo << COMBO_SHIFT) | eid` packed instance at the
 * surface's `eids` lane, applies the instance transform, splices the surface's own `vs` chunk when present,
 * then projects by that combo's tile-folded viewProj (`shadowLayout.$.faceVP.m[combo]`) and computes the
 * `tileBox` seam-discard bounds from `shadowLayout.$.tileRects` (indexed `slot·6+face` for the point atlas,
 * `slot` alone for the cascade atlas — `pointShadowCode`'s `rectExpr` split) scaled by the atlas's pixel
 * size. Only an **instanced** surface reaches here (only `eids`+`transforms` gives a per-instance member to
 * re-gather against) — `compileTypedShadow` gates the call, so this never runs for a non-instanced surface.
 */
function typedShadowVs(
    surface: AnyTypedSurface,
    shadowGroup: TgpuBindGroupLayout<any>,
    atlas: number,
    cascade: boolean,
) {
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as { eids: any[]; transforms: any[] };
    const shadowBound = shadowGroup.$ as unknown as {
        faceVP: { m: any[] };
        comboMeta: { m: any[] };
        tileRects: { rects: any[] };
    };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position, tileBox: d.interpolate("flat", d.vec4f) },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            // the raw prepass/point default — never touched by the instance transform, matching
            // `typedPrepassVs`'s pinned law
            const localNormal = d.vec3f(0, 0, 1);
            const uv = d.vec2f(0, 0);
            const packed = bound.eids[input.iid];
            const eid = packed & EID_MASK;
            const combo = packed >> COMBO_SHIFT;
            const xform = Xform(bound.transforms[eid]);
            let world = d.vec4f(xformPoint(xform, localPos), 1);
            let worldNormal = d.vec3f(xformNormal(xform, localNormal));
            if (hasVs) {
                const patched = vsFn!(
                    VsIn({
                        localPos,
                        localNormal,
                        uv,
                        vidx: input.vidx,
                        eid,
                        iid: input.iid,
                        xform,
                        world,
                        worldNormal,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
            }
            const m = shadowBound.comboMeta.m[combo];
            const rect = cascade
                ? shadowBound.tileRects.rects[m.x]
                : shadowBound.tileRects.rects[m.x * 6 + m.y];
            const clip = std.mul(shadowBound.faceVP.m[combo], world);
            const tileBox = d.vec4f(std.mul(atlas, rect.xy), rect.z * atlas, 0);
            return { pos: clip, tileBox };
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}Vs`);
}

/** the clipped shadow-atlas vertex: {@link typedShadowVs} plus the fixed fs context fields required to
 * execute the surface's cutoff fragment. Cutoff consumes the full vertex stream so authored UVs and
 * normals match the color pass. */
function clipShadowVs(
    surface: AnyTypedSurface,
    shadowGroup: TgpuBindGroupLayout<any>,
    atlas: number,
    cascade: boolean,
) {
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout;
    const bound = layout.$ as unknown as { eids: any[]; transforms: any[] };
    const shadowBound = shadowGroup.$ as unknown as {
        faceVP: { m: any[] };
        comboMeta: { m: any[] };
        tileRects: { rects: any[] };
    };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                tileBox: d.interpolate("flat", d.vec4f),
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                world: d.vec3f,
                uv: d.vec2f,
                localPos: d.vec3f,
            },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            const localNormal = octDecodeNormal(v.z);
            const uv = decodeUv(v.w, mq);
            const packed = bound.eids[input.iid];
            const eid = packed & EID_MASK;
            const combo = packed >> COMBO_SHIFT;
            const xform = Xform(bound.transforms[eid]);
            let world = d.vec4f(xformPoint(xform, localPos), 1);
            let worldNormal = d.vec3f(xformNormal(xform, localNormal));
            if (hasVs) {
                const patched = vsFn!(
                    VsIn({
                        localPos,
                        localNormal,
                        uv,
                        vidx: input.vidx,
                        eid,
                        iid: input.iid,
                        xform,
                        world,
                        worldNormal,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
            }
            const m = shadowBound.comboMeta.m[combo];
            const rect = cascade
                ? shadowBound.tileRects.rects[m.x]
                : shadowBound.tileRects.rects[m.x * 6 + m.y];
            const pos = std.mul(shadowBound.faceVP.m[combo], world);
            const tileBox = d.vec4f(std.mul(atlas, rect.xy), rect.z * atlas, 0);
            return {
                pos,
                tileBox,
                worldNormal: std.normalize(worldNormal),
                eid,
                world: world.xyz,
                uv,
                localPos,
            };
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipVs`);
}

/** The atlas-projecting half of the locked one-varying clip mechanism. Dynamic varying names stay in a
 * raw copier; the thin entry assigns the authored field and the fragment receives it as fixed `v0` at
 * the same explicit location. */
function varyingShadowVs(
    surface: AnyTypedSurface,
    shadowGroup: TgpuBindGroupLayout<any>,
    atlas: number,
    cascade: boolean,
) {
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    if (keys.length !== 1 || !surface.vs) {
        throw new Error(
            `sear: typed surface "${surface.name}" needs one authored varying for its clip shadow copier`,
        );
    }
    const layout = surface.layout;
    const bound = layout.bound as unknown as Record<string, unknown>;
    const shadow = shadowGroup.bound as unknown as Record<string, unknown>;
    const Out = d
        .struct({
            pos: d.vec4f,
            tileBox: d.vec4f,
            worldNormal: d.vec3f,
            eid: d.u32,
            world: d.vec3f,
            uv: d.vec2f,
            localPos: d.vec3f,
            ...varyings,
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipOut`);
    const assigns = keys.map((key) => `    out.${key} = patched.${key};`).join("\n");
    const rect = cascade ? "m.x" : "m.x * 6u + m.y";
    const copier = tgpu
        .fn(
            [d.u32, d.u32],
            Out,
        )(/* wgsl */ `(vidx: u32, iid: u32) -> Out {
    let v = vertices[vidx];
    let mq = meshQuant[meshIdOf(v.y)];
    let localPos = decodePos(v.x, v.y, mq);
    let localNormal = octDecodeNormal(v.z);
    let uv = decodeUv(v.w, mq);
    let packed = eids[iid];
    let eid = packed & ${EID_MASK}u;
    let combo = packed >> ${COMBO_SHIFT}u;
    let xform = transforms[eid];
    var world = vec4f(xformPoint(xform, localPos), 1.0);
    var worldNormal = vec3f(xformNormal(xform, localNormal));
    let patched = vs(VsIn(localPos, localNormal, uv, vidx, eid, iid, xform, world, worldNormal));
    world = patched.world;
    worldNormal = patched.worldNormal;
    let m = comboMeta.m[combo];
    let rect = tileRects.rects[${rect}];
    var out: Out;
    out.pos = faceVP.m[combo] * world;
    out.tileBox = vec4f(${atlas}.0 * rect.xy, rect.z * ${atlas}.0, 0.0);
    out.worldNormal = normalize(worldNormal);
    out.eid = eid;
    out.world = world.xyz;
    out.uv = uv;
    out.localPos = localPos;
${assigns}
    return out;
}`)
        .$uses({
            Out,
            vertices: bound.vertices,
            eids: bound.eids,
            transforms: bound.transforms,
            meshQuant: engineLayout.bound.meshQuant,
            faceVP: shadow.faceVP,
            comboMeta: shadow.comboMeta,
            tileRects: shadow.tileRects,
            decodePos,
            decodeUv,
            meshIdOf,
            octDecodeNormal,
            xformPoint,
            xformNormal,
            VsIn,
            vs: surface.vs,
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipCopier`);
    const located = Object.fromEntries(
        Object.entries(varyings).map(([key, schema], i) => [
            key,
            d.location(VARYING_BASE + i, schema),
        ]),
    );
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                tileBox: d.interpolate("flat", d.vec4f),
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                world: d.vec3f,
                uv: d.vec2f,
                localPos: d.vec3f,
                ...located,
            },
        })((input) => {
            "use gpu";
            const out = copier(input.vidx, input.iid);
            return out;
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipVs`);
}

function varyingShadowFs(surface: AnyTypedSurface) {
    const { varyingSchema, copier } = clipVaryingCopier(surface);
    const entryIn = {
        pos: d.builtin.position,
        tileBox: d.interpolate("flat", d.vec4f),
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        v0: d.location(VARYING_BASE, varyingSchema as d.Vec3f),
    };
    return tgpu
        .fragmentFn({
            in: entryIn as unknown as typeof entryIn & { v0: d.Vec3f },
            out: d.Void,
        })((input) => {
            "use gpu";
            const p = input.pos.xy;
            const mn = input.tileBox.xy;
            const sz = input.tileBox.z;
            if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) {
                std.discard();
            }
            copier(input.worldNormal, input.eid, input.world, input.uv, input.localPos, input.v0);
        })
        .$name(`${surface.name}ClipShadowFs`);
}

/** the clipped atlas fragment: preserve the tile-seam discard, then call the surface fs solely for its
 * cutoff discard. Its color result is intentionally ignored by this depth-only pipeline. */
function clipShadowFs(surface: AnyTypedSurface) {
    const Ctx = (surface.fs as unknown as { shell: { argTypes: [ReturnType<typeof fsCtxSchema>] } })
        .shell.argTypes[0];
    return tgpu
        .fragmentFn({
            in: {
                pos: d.builtin.position,
                tileBox: d.interpolate("flat", d.vec4f),
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                world: d.vec3f,
                uv: d.vec2f,
                localPos: d.vec3f,
            },
            out: d.Void,
        })((input) => {
            "use gpu";
            const p = input.pos.xy;
            const mn = input.tileBox.xy;
            const sz = input.tileBox.z;
            if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) {
                std.discard();
            }
            const ctx = Ctx({
                eid: input.eid,
                world: input.world,
                worldNormal: std.normalize(input.worldNormal),
                uv: input.uv,
                localPos: input.localPos,
            });
            surface.fs(ctx);
        })
        .$name(`${surface.name}ClipShadowFs`);
}

/**
 * compile a `TypedSurface`'s point + cascade shadow-atlas pipelines (4a-ii-c-3a-3) — `null` for a
 * non-instanced or `screen` surface (only an instanced, non-`screen` surface casts — the raw `castable`
 * law `compileVariant` applies, `!surface.screen && instanced` — a 2D overlay has no atlas placement).
 * Opaque surfaces share {@link typedShadowFs}; clipped surfaces use their wider cutoff
 * vertex/fragment pair. Each closes over its own `pointLayoutTyped` / `cascadeLayoutTyped` group-1 and its
 * own atlas pixel size (the
 * two atlases are sized independently — the point atlas's live caster cap vs the cascade atlas's fixed
 * resolution × grid).
 */
function compileTypedShadow(surface: AnyTypedSurface): {
    point: TgpuRenderPipeline<any> | null;
    cascade: TgpuRenderPipeline<any> | null;
} {
    if (!typedInstanced(surface) || surface.screen) return { point: null, cascade: null };
    const primitive = surfacePrimitive(false);
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    // the receiver stub, as in `compileTypedPrepass` — doubly load-bearing here: the real receiver
    // would sample the very atlas this pipeline renders into (a usage hazard the raw path's stubs
    // exist to prevent)
    const root = Compute.root.with(pointShadowSlot, pointShadowStub);
    const clip = surface.blend === "clip";
    const varying = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const point = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingShadowVs(surface, pointLayoutTyped, pointAtlasSize(), false)
                    : clipShadowVs(surface, pointLayoutTyped, pointAtlasSize(), false)
                : typedShadowVs(surface, pointLayoutTyped, pointAtlasSize(), false),
            fragment: clip
                ? ((varying ? varyingShadowFs(surface) : clipShadowFs(surface)) as never)
                : typedShadowFs,
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`sear-typed-point-${surface.name}`);
    const cascade = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingShadowVs(
                          surface,
                          cascadeLayoutTyped,
                          cascadeAtlasSize(sunResolution(), sunCascades()),
                          true,
                      )
                    : clipShadowVs(
                          surface,
                          cascadeLayoutTyped,
                          cascadeAtlasSize(sunResolution(), sunCascades()),
                          true,
                      )
                : typedShadowVs(
                      surface,
                      cascadeLayoutTyped,
                      cascadeAtlasSize(sunResolution(), sunCascades()),
                      true,
                  ),
            fragment: clip
                ? ((varying ? varyingShadowFs(surface) : clipShadowFs(surface)) as never)
                : typedShadowFs,
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`sear-typed-cascade-${surface.name}`);
    return { point, cascade };
}

// the same receiver stub the depth pipelines bind, so the differential seams emit the pipelines' text
const stubReceiver = (cfg: Configurable) => cfg.with(pointShadowSlot, pointShadowStub);

/** the typed point/cascade shadow-atlas pipelines' emitted vs+fs WGSL for one `TypedSurface` — device-free
 * (`typedShadowVs`/`typedShadowFs` are pure resolve inputs), the structural seam `pipelines.test.ts`'s
 * differential resolves against `pointShadowCode(surface, 0)` / `pointShadowCode(surface, 0, true)`. */
export function typedShadowWgsl(
    surface: AnyTypedSurface,
    variant = 0,
): { point: string; cascade: string } {
    const resolved = typedVariant(surface, variant);
    const clip = resolved.blend === "clip";
    const varying = !!resolved.varyings && Object.keys(resolved.varyings).length > 0;
    return {
        point: tgpu.resolve(
            clip
                ? [
                      varying
                          ? varyingShadowVs(resolved, pointLayoutTyped, pointAtlasSize(), false)
                          : clipShadowVs(resolved, pointLayoutTyped, pointAtlasSize(), false),
                      varying ? varyingShadowFs(resolved) : clipShadowFs(resolved),
                  ]
                : [
                      typedShadowVs(resolved, pointLayoutTyped, pointAtlasSize(), false),
                      typedShadowFs,
                  ],
            { names: "strict", config: stubReceiver },
        ),
        cascade: tgpu.resolve(
            clip
                ? [
                      varying
                          ? varyingShadowVs(
                                resolved,
                                cascadeLayoutTyped,
                                cascadeAtlasSize(sunResolution(), sunCascades()),
                                true,
                            )
                          : clipShadowVs(
                                resolved,
                                cascadeLayoutTyped,
                                cascadeAtlasSize(sunResolution(), sunCascades()),
                                true,
                            ),
                      varying ? varyingShadowFs(resolved) : clipShadowFs(resolved),
                  ]
                : [
                      typedShadowVs(
                          resolved,
                          cascadeLayoutTyped,
                          cascadeAtlasSize(sunResolution(), sunCascades()),
                          true,
                      ),
                      typedShadowFs,
                  ],
            { names: "strict", config: stubReceiver },
        ),
    };
}

/** the compiled typed pipeline(s) for a `TypedSurfaces` entry, or `undefined` until
 * {@link compileTypedVariant} has run for it. */
export function getCompiledTyped(name: string, variant = 0): CompiledTyped | undefined {
    return _compiledTyped.get(variantKey(name, variant));
}

/** the typed color/transparent pipeline's emitted vs+fs WGSL for one `TypedSurface` — device-free
 * (`typedColorVs`/`typedColorFs` are pure resolve inputs, shared by both pipelines exactly like the raw
 * path's one `surfaceCode(surface, "color")` module backs both `colorPipelines` outputs), the structural
 * seam `pipelines.test.ts`'s differential resolves against. */
export function typedSurfaceWgsl(surface: AnyTypedSurface, variant = 0): string {
    const resolved = typedVariant(surface, variant);
    const hasVaryings = !!resolved.varyings && Object.keys(resolved.varyings).length > 0;
    const vertex = hasVaryings ? typedVaryingVs(resolved) : typedColorVs(resolved);
    const fragment = hasVaryings ? typedVaryingFs(resolved) : typedColorFs(resolved);
    return tgpu.resolve([vertex, fragment], { names: "strict" });
}

/** the typed prepass pipelines' emitted WGSL for one `TypedSurface` — device-free (`typedPrepassVs`/
 * `typedTagFs` are pure resolve inputs), the structural seam `pipelines.test.ts`'s differential resolves
 * against. `""` is the position-only depth pipeline (vertex-only, no fragment); `"tag"` the id-lane pair. */
export function typedPrepassWgsl(
    surface: AnyTypedSurface,
    variant = 0,
): { "": string; tag: string } {
    const resolved = typedVariant(surface, variant);
    const clip = resolved.blend === "clip";
    const varying = !!resolved.varyings && Object.keys(resolved.varyings).length > 0;
    return {
        "": tgpu.resolve(
            clip
                ? [
                      varying ? typedVaryingVs(resolved, true) : typedClipVs(resolved),
                      varying ? varyingClipFs(resolved, false) : typedClipFs(resolved, false),
                  ]
                : [typedPrepassVs(resolved)],
            {
                names: "strict",
                config: stubReceiver,
            },
        ),
        tag: tgpu.resolve(
            clip
                ? [
                      varying ? typedVaryingVs(resolved, true) : typedClipVs(resolved),
                      varying ? varyingClipFs(resolved, true) : typedClipFs(resolved, true),
                  ]
                : [typedTagVs(resolved), typedTagFs(resolved)],
            { names: "strict", config: stubReceiver },
        ),
    };
}

// ---- the typed `Backgrounds` contract's pipeline builder (4a-ii-c-3a-4, the Backgrounds bindings lock):
// the Surfaces contract minus mesh machinery, same group scheme. Group 0 = the shared `engineLayout`
// instance (the color pass's own — `BG_BASE` dies on the typed path, matching the raw path's own group-0
// sharing note in `compileBackground`'s docblock); group 1 = `shadowLayoutTyped`, declared-but-unused via
// the `forcedZero` scope-forcing precedent (preserving `compileBackground`'s documented group-count-
// compatibility reason: a bg pipeline with only group 0 would drop group 1 for the blend draws that follow
// in the same pass); group 2 = the background's own bindings, through `contract.ts`'s `bgLayout` — the
// SAME `layoutEntry` synthesis `layout()` uses for a surface, minus the `vertices` injection (no mesh).

/** the widest `TypedBackground` shape (any bindings) — the bare `TypedBackground` default pins `B` to
 *  `Record<string, Binding>` already, so this alias exists only to name the widened form at call
 *  boundaries, matching {@link AnyTypedSurface}'s shape. */
type AnyTypedBackground = TypedBackground<Record<string, TypedBinding>>;

/**
 * the engine-owned fullscreen-triangle vertex entry every typed background shares — no per-background
 * variance (no mesh, no varyings, per the Backgrounds bindings lock), so ONE instance serves every typed
 * background's pipeline. Statement-for-statement `backgroundCode`'s raw `vs` (codegen.ts): the three
 * corners come from `@builtin(vertex_index)` alone, emitted at the reverse-Z far plane (clip z = 0) so
 * {@link compileTypedBg}'s `depthCompare: "greater-equal"` + no-depth-write test admits only
 * un-rendered pixels.
 */
const typedBgVs = tgpu
    .vertexFn({ in: { vidx: d.builtin.vertexIndex }, out: { pos: d.builtin.position } })(
        (input) => {
            "use gpu";
            const c = d.vec2f(d.f32((input.vidx << 1) & 2), d.f32(input.vidx & 2));
            return { pos: d.vec4f(c.x * 2 - 1, c.y * 2 - 1, 0, 1) };
        },
    )
    .$name("bgVs");

/**
 * a typed background's fragment entry: reconstructs the normalized world-space view ray `dir` from
 * `@builtin(position)` + `engineLayout`'s `view.invViewProj` — operand-for-operand `backgroundCode`'s raw
 * reconstruct (codegen.ts), not an interstage varying (gpu.md rule 9) — forces `shadowLayoutTyped`'s
 * group-1 bindings into scope via the `forcedZero` fold (`typedColorFs`'s precedent, same reason:
 * `sampleSunShadow`/`pointShadowOf`'s free names are invisible to `tgpu.resolve`'s call-graph walk
 * otherwise), then calls the background's own `fs` chunk and wraps its `vec3f` result opaque (`vec4f(col,
 * 1)`, `backgroundCode`'s own contract).
 */
function typedBgFs(bg: AnyTypedBackground) {
    return tgpu
        .fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })((input) => {
            "use gpu";
            const uv = std.div(input.pos.xy, engineLayout.$.view.resolution);
            const ndc = d.vec3f(uv.x * 2 - 1, 1 - uv.y * 2, 0);
            const far = std.mul(engineLayout.$.view.invViewProj, d.vec4f(ndc, 1));
            const dir = std.normalize(
                std.sub(std.div(far.xyz, far.w), engineLayout.$.view.eye.xyz),
            );
            // see `typedColorFs`'s matching comment — the same forcing-touch precedent, folded into a
            // value the return genuinely uses so the transpiler can't prune it as dead
            const forcedZero =
                (shadowLayoutTyped.$.pointShadows.casters[0].pos.x +
                    shadowLayoutTyped.$.tileRects.rects[0].x +
                    shadowLayoutTyped.$.sunShadow.enabled +
                    std.textureSampleCompareLevel(
                        shadowLayoutTyped.$.pointAtlas,
                        shadowLayoutTyped.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    ) +
                    std.textureSampleCompareLevel(
                        shadowLayoutTyped.$.shadowMap,
                        shadowLayoutTyped.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    )) *
                0;
            const col = bg.fs(BgCtx({ dir }));
            return d.vec4f(std.add(col, d.vec3f(forcedZero)), 1);
        })
        .$name(`${bg.name}Fs`);
}

/** a compiled typed background: the 4× MSAA + single-sample twins (a camera binds whichever its
 *  `Camera.antialias` selects — `CompiledBg`'s raw twin). */
export interface CompiledTypedBg {
    color: TgpuRenderPipeline<d.Vec4f>;
    single: TgpuRenderPipeline<d.Vec4f>;
    // the background's own group-2 bind group, built lazily on first draw (`renderColor`'s typed
    // backdrop pick) and cached on the resolved resource identities; null for a binding-free background
    // (its empty layout never enters the pipeline layout, so no group is bound at 2)
    group2: { group: GPUBindGroup; resources: BindResource[] } | null;
    // the background's engine group-0 instances per view slot (`typedEngineGroup`'s caller-owned cache;
    // the bg's quant fill is the stable per-build `bgQuant()`, so this never sees buffer churn)
    engineCache: Map<number, GPUBindGroup>;
}
const _compiledTypedBg = new Map<string, CompiledTypedBg>();

/**
 * compile one typed background's color pipelines — both the 4× MSAA + single-sample twins, eagerly
 * (`compileBackground`'s own reason: backgrounds are few, the camera's AA mode is known only at draw
 * time). `depthCompare: "greater-equal"` + no depth write: at clip z = 0 an un-rendered pixel (cleared
 * depth 0) passes `0 >= 0`, a geometry pixel (depth > 0) fails, matching the raw pipeline's shape. Cached
 * by name.
 */
export function compileTypedBg(bg: AnyTypedBackground): CompiledTypedBg {
    const cached = _compiledTypedBg.get(bg.name);
    if (cached) return cached;
    const fragment = typedBgFs(bg);
    const primitive: GPUPrimitiveState = { topology: "triangle-list", cullMode: "none" };
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: false,
        depthCompare: "greater-equal",
    };
    const color = Compute.root
        .createRenderPipeline({
            vertex: typedBgVs,
            fragment,
            targets: { format: Render.format },
            primitive,
            depthStencil,
            multisample: { count: SAMPLE_COUNT },
        })
        .$name(`sear-typed-bg-${bg.name}`);
    const single = Compute.root
        .createRenderPipeline({
            vertex: typedBgVs,
            fragment,
            targets: { format: Render.format },
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`sear-typed-bg-${bg.name}-1x`);
    const compiled: CompiledTypedBg = { color, single, group2: null, engineCache: new Map() };
    _compiledTypedBg.set(bg.name, compiled);
    return compiled;
}

/** the compiled typed pipeline(s) for a `TypedBackgrounds` entry, or `undefined` until
 * {@link compileTypedBg} has run for it. */
export function getTypedBg(name: string): CompiledTypedBg | undefined {
    return _compiledTypedBg.get(name);
}

/** the typed background's emitted vs+fs WGSL — device-free (`typedBgVs`/`typedBgFs` are pure resolve
 * inputs), the structural seam `pipelines.test.ts`'s differential resolves against `backgroundCode` for an
 * equivalent string background. */
export function typedBgWgsl(bg: AnyTypedBackground): string {
    return tgpu.resolve([typedBgVs, typedBgFs(bg)], { names: "strict" });
}

/**
 * eagerly compile every non-specializing legacy surface's variant 0, except a migration descriptor whose
 * name is shadowed by `TypedSurfaces` (the Part materials, lines, sprite, …, so the bare happy path renders
 * on the first frame), plus every registered background — the async half of
 * `prepareSear`. Non-specializing `TypedSurfaces` entries compile here; specializing variants queue after
 * Part publishes its draw pairs. `compileTypedVariant` is sync (typegpu pipelines are sync-created), so it
 * runs beside the legacy compiles rather than joining the `Promise.all`; a real device (`bun bench`) is
 * what proves the typed color pipeline's group split actually validates, every run. Then every registered
 * `TypedBackgrounds` entry (4a-ii-c-3a-4) the same way — `compileTypedBg` is sync too, so any background
 * registered
 * through the typed contract compiles for real at warm the moment a consumer registers one (none does yet:
 * additive only, matching `TypedSurfaces`' own boundary until a consumer conversion lands).
 */
/** variants already reachable through registered draws at warm, deduped per surface. */
export function knownTypedVariants(
    surface: AnyTypedSurface,
    draws: Registry<Draw> = Draws,
    meshes: Registry<Mesh> = Meshes,
): number[] {
    if (!surface.specialize) return [0];
    const variants = new Set<number>();
    for (const draw of draws) {
        if (draw.surface !== surface.name) continue;
        const mesh = meshes.get(draw.mesh);
        if (mesh) variants.add(mesh.variant ?? 0);
    }
    return [...variants];
}

export async function preparePipelines(
    device: GPUDevice,
    backgrounds: Iterable<Background>,
): Promise<void> {
    await Promise.all([
        ...Array.from(Surfaces, (surface) =>
            TypedSurfaces.has(surface.name) || surface.specialize
                ? null
                : compileVariant(device, surface, 0),
        ),
        ...Array.from(backgrounds, (bg) => compileBackground(device, bg)),
    ]);
    // force each typed pipeline's memo at warm (`root.unwrap` runs the resolve + the sync
    // `createRenderPipeline`) — typegpu defers both to first use, which would otherwise land mid-frame
    // on the first draw and hide a resolution/validation error until then (the force-compile-at-warm
    // lock, Approach 0a)
    for (const surface of TypedSurfaces) {
        if (surface.specialize) continue;
        const t = compileTypedVariant(surface);
        for (const p of [t.color, t.transparent, t.point, t.cascade, ...t.prepass.values()]) {
            if (p) Compute.root.unwrap(p);
        }
    }
    for (const bg of TypedBackgrounds) {
        const cb = compileTypedBg(bg);
        Compute.root.unwrap(cb.color);
        Compute.root.unwrap(cb.single);
    }
}
