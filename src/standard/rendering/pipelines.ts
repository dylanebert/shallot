import { materialTable } from "./material";
// StandardRenderer's pipeline compilation: the compiled-surface / compiled-background caches and the async
// TypeGPU pipeline factories that fill them. `atlas.ts` supplies the shadow-atlas bind-group layouts every color +
// point + cascade pipeline binds group 1 against. `forward.ts` owns bind-group *resolution* per draw
// (`record()`) — this file compiles pipelines and caches them by surface name and spec identity.

import type { TgpuBindGroupLayout, TgpuRenderPipeline } from "typegpu";
import tgpu from "typegpu";
import type { AnyData, AnyWgslData } from "typegpu/data";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    DEPTH_FORMAT,
    Frame,
    PICKING_ID_FORMAT,
    PICKING_ID_NONE,
    RenderContext,
    SAMPLE_COUNT,
} from "../../core/rendering";
import type { World } from "../../engine";
import {
    decodePos,
    decodeUv,
    MeshQuant,
    meshIdOf,
    octDecodeNormal,
    Xform,
    xformNormal,
    xformPoint,
} from "../../engine/utils";
import { cascadeLayout, pointLayout, shadowLayout } from "./atlas";
import { LightCull } from "./cluster";
import type { Background, BackgroundLayout, Binding, Surface, SurfaceLayout } from "./contract";
import { BackgroundContext, Backgrounds, type fsCtxSchema, Surfaces, VsIn } from "./contract";
import {
    engineLayout,
    fragCoord,
    fragWorld,
    pointScale,
    pointShadowSlot,
    pointShadowStub,
    sunVisibility,
} from "./engine";
import type { Recorded } from "./forward";
import { Lighting } from "./lighting";
import type { Draw } from "./registry";
import { sampleSunShadow } from "./shade";
import { cascadeAtlasSize, pointAtlasSize, sunCascades, sunResolution } from "./shadows";

const ALPHA_BLEND: GPUBlendState = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};

export type BindResource =
    | import("typegpu").TgpuBuffer<AnyData>
    | GPUBuffer
    | GPUTexture
    | GPUSampler;

interface PipelineState {
    compiledSurfaces: Map<string, CompiledSurface>;
    surfaceGroups: Map<string, SurfaceGroupEntry>;
    bgQuant: GPUBuffer | null;
    compiledBackgrounds: Map<string, CompiledBackground>;
}

const pipelineStateKey = { create: createPipelineState };

function createPipelineState(): PipelineState {
    return {
        compiledSurfaces: new Map(),
        surfaceGroups: new Map(),
        bgQuant: null,
        compiledBackgrounds: new Map(),
    };
}

function pipelineState(world: World): PipelineState {
    return world.resource(pipelineStateKey);
}

/** Create this world's StandardRenderer pipeline caches during plugin initialization. */
export function initializePipelineState(world: World): void {
    world.resource(pipelineStateKey);
}

export function clearGroups(world: World): void {
    pipelineState(world).surfaceGroups.clear();
}

export function resetPipelineCaches(world: World): void {
    pipelineState(world).compiledSurfaces.clear();
    pipelineState(world).compiledBackgrounds.clear();
    pipelineState(world).surfaceGroups.clear();
    pipelineState(world).bgQuant?.destroy();
    pipelineState(world).bgQuant = null;
}

// ---- the pipeline builder (the template + its extensions): compiles a `Surface`'s
// `vs`/`fs` TGSL fns against the canonical `engineLayout` (group 0), the shadow group (group 1,
// `shadowLayout` — the atlas passes bind `pointLayout`/`cascadeLayout` there instead), and
// the surface's own synthesized `layout` (group 2 — `surfaceLayout()`'s $idx(2) synthesis): the
// opaque color / transparent pipelines, the prepass depth + tag pipelines, and the point/cascade
// shadow-atlas pipelines. A surface in `Surfaces` DRAWS through these in every pass — `record()`
// looks the surface up in `Surfaces`, and `forward.ts`/`atlas.ts` issue the
// draws via `.with(pass)` on standard's own render passes. `screen` surfaces project through their own `vs`
// chunk's `patch.clip` and draw un-culled.
//
// Cached by name and spec identity. `preparePipelines` compiles every
// `Surfaces` entry and force-unwraps it, so the resolve + the sync
// `createRenderPipeline` validate against the real device at warm — a malformed group split, a name
// collision, a binding-limit breach all throw there, not mid-frame at first draw.
export interface CompiledSurface {
    /** exact registry spec + layout this entry was compiled from; name alone cannot distinguish
     * a same-name replacement after warm (or an in-place layout swap). */
    owner: AnySurface;
    layout: SurfaceLayout<Record<string, Binding>>;
    color: TgpuRenderPipeline<d.Vec4f> | null;
    transparent: TgpuRenderPipeline<d.Vec4f> | null;
    // the prepass pipelines for this surface (keyed by lane: `""` the position-only depth
    // pipeline, `"tag"` the id lane); empty for a `blend:
    // "alpha"` surface (a transparent pixel has no single owner, writes no prepass depth, casts nothing —
    // the same rule `compileSurface` applies). Compiled off `layout.depthVariant` — a DISTINCT
    // `TgpuBindGroupLayout` object from `layout`, so a draw-time bind-group
    // cache for these needs its own key space, never `layout`'s (`SurfaceGroupEntry.depth`, `record`).
    // the map holds both the depth-only (`Void` output) and the tag-lane (`u32` output) pipeline under one
    // key space — holding both depth and tag pipelines under a single key namespace (a
    // bare `GPURenderPipeline` has no output type parameter either)
    prepass: Map<string, TgpuRenderPipeline<any>>;
    // the point/cascade shadow-atlas pipelines. `null` for a
    // non-instanced surface (only an instanced surface casts) —
    // never a silent gap, since a non-instanced surface has no per-instance `eids`/`globalTransforms` to
    // re-gather against in the first place.
    point: TgpuRenderPipeline<any> | null;
    cascade: TgpuRenderPipeline<any> | null;
    // the single-sample (AA-off) color twin, compiled lazily by `ensureSingle` the first frame a
    // no-AA camera draws this surface — compiled here (typegpu pipeline
    // wrappers are cheap; the real resolve+create defers to first draw regardless)
    single: {
        color: TgpuRenderPipeline<d.Vec4f> | null;
        transparent: TgpuRenderPipeline<d.Vec4f> | null;
    } | null;
    // the fixed inputs `ensureSingle` re-compiles the 1× twin from; the entry fns are reused, so the
    // twin shares the authored vs/fs, differing only in multisample
    args: {
        vertex: ReturnType<typeof colorVs> | ReturnType<typeof varyingVs>;
        fragment: ReturnType<typeof colorFs>;
        blend: Surface["blend"];
        // the raster state the 4× twin compiled with — carries the `screen` cull decision, which
        // `ensureSingle` can't re-derive (it never sees the surface)
        primitive: GPUPrimitiveState;
        name: string;
    };
}

/** the per-draw group-2 state a draw binds.
 * `color` builds against `layout` (the 16 B main stream at the `vertices` slot); opaque depth/atlas
 * groups use the DISTINCT `layout.depthVariant` object (the 8 B position stream), while clipped groups
 * use the full layout/main stream because their fragment cutoff consumes material UVs. `point`/`cascade`
 * swap the `eids` lane to that atlas's re-gathered packed list. View rides group
 * 0 per slot, so this state is slot-independent — one
 * instance per draw. `engineCache` holds the draw's engine group-0 instances per view slot (lazy,
 * entry-scoped so a quant-buffer churn — a glTF import's
 * per-import buffers — drops the old groups with the overwritten entry, never a module map keyed on
 * buffer identity that grows for the app's life). Shadow-atlas passes resolve slot 0 through this cache,
 * using its ViewUniforms buffer as an unread placeholder: the atlas VS projects by its own tile viewProj. */
export type SurfaceGroupEntry = {
    /** exact registry spec this group was built for — resource identity alone is insufficient when a
     * same-name surface replacement carries a different layout with the same buffers. */
    owner: AnySurface;
    layout: SurfaceLayout<Record<string, Binding>>;
    quant: GPUBuffer;
    color: GPUBindGroup;
    depth: GPUBindGroup | null;
    /** full-stream group for an authored tag hook; opaque depth otherwise keeps the compact stream. */
    tag: GPUBindGroup | null;
    point: GPUBindGroup | null;
    cascade: GPUBindGroup | null;
    /** the surface's resolved instance-id source before the atlas swaps in its re-gathered list. */
    eids: GPUBuffer | null;
    engineCache: Map<number, GPUBindGroup>;
    resources: BindResource[];
    /** the layout's own binding names in `resources` order (after the four mesh streams), each with the
     * registry it resolves from, so a steady frame compares live identities without re-resolving. */
    names: string[];
    registries: ReadonlyMap<string, BindResource>[];
    /** per compiled pipeline, that pipeline with this entry's group 2 and index buffer bound, built on the
     * entry's first draw through it (the pass binds groups 0 and 1 per draw). */
    bound: Map<TgpuRenderPipeline<any>, TgpuRenderPipeline<any>>;
    /** the entry's frame-draw record, rewritten in place when a steady frame re-resolves the draw. */
    item: { draw: Draw; r: Recorded };
    /** the material variant the entry's compiled surface was looked up at. */
};

/** the cached per-draw group-2 state for a Draw name, or `undefined` on a cache miss (`record`
 * rebuilds it). Supplying the current surface also invalidates a same-name replacement: bind groups
 * are layout-object-specific even when every resolved GPU resource is unchanged. */
export function getGroup(
    world: World,
    name: string,
    surface?: AnySurface,
): SurfaceGroupEntry | undefined {
    const entry = pipelineState(world).surfaceGroups.get(name);
    if (entry && surface && (entry.owner !== surface || entry.layout !== surface.layout)) {
        pipelineState(world).surfaceGroups.delete(name);
        return undefined;
    }
    return entry;
}

/** cache a draw's resolved group-2 state (`record`, on a resource-identity change). */
export function setGroup(world: World, name: string, entry: SurfaceGroupEntry): void {
    pipelineState(world).surfaceGroups.set(name, entry);
}

const engineMaterialBuffers = { create: () => new WeakMap<Map<number, GPUBindGroup>, GPUBuffer>() };

/** the engine group-0 bind group for a view slot against one meshQuant buffer — the shared live
 * `engineLayout` instance a draw at that slot binds (frame / per-slot View / lighting /
 * light-cull outputs / materials / the dequant table), built lazily into the caller-owned `cache` (a
 * `SurfaceGroupEntry.engineCache` or a `CompiledBackground.engineCache` — never a module map keyed on the
 * quant buffer, whose entries would outlive a churned buffer for the app's life). */
export function engineGroup(
    world: World,
    cache: Map<number, GPUBindGroup>,
    slot: number,
    quant: GPUBuffer,
): GPUBindGroup {
    const _lightCull = world.resource(LightCull);

    const materials = materialTable(world).buffer;
    const buffers = world.resource(engineMaterialBuffers);
    if (buffers.get(cache) !== materials) {
        cache.clear();
        buffers.set(cache, materials);
    }
    const cached = cache.get(slot);
    if (cached) return cached;
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(engineLayout, {
            frame: world.resource(Frame).buffer!,
            view: world.resource(RenderContext).viewBuffers[slot],
            lighting: world.resource(Lighting).buffer!,
            pointLights: _lightCull.lights!,
            meshQuant: quant,
            materials,
        }),
    );
    cache.set(slot, group);
    return group;
}

// a background reads no mesh, but `engineLayout` (the shared group-0 instance the Backgrounds lock
// names) still carries the `meshQuant` slot — a one-record placeholder buffer fills it, never read (the
// slot-0 View placeholder precedent)

/** the never-read `meshQuant` placeholder a background's engine group binds. */
export function bgQuant(world: World): GPUBuffer {
    const resources = pipelineState(world);
    resources.bgQuant ??= world.gpu.device.createBuffer({
        label: "standard-bg-quant",
        size: d.sizeOf(MeshQuant),
        usage: GPUBufferUsage.STORAGE,
    });
    return resources.bgQuant;
}

/** the widest `Surface` shape (any bindings, any varyings) — the bare `Surface` default pins
 *  varyings to `Record<string, never>`, so every internal pipeline helper takes this wider alias to
 *  accept a real varyings-carrying surface (`vertex`) at the call boundary. */
type AnySurface = Surface<Record<string, Binding>, Record<string, AnyWgslData>>;

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
// (worldNormal/eid/world/uv/localPos at 0–4); the 4-slot custom budget keeps 5+ within the 16 cap
const VARYING_BASE = 5;

// the hard budget: 4 custom interpolator slots per surface. The vs side is N-general (the
// copier templates over `Object.keys`), so this bound is the fragment entry's — its transpiled body must
// statically name `input.v0`…`input.v3`, one arm per count (`varyingFs`)
const MAX_VARYINGS = 4;

function fragmentInterstage(surface: AnySurface): Record<string, AnyWgslData> {
    return {
        ...(surface.fragmentInputs?.uv ? { uv: d.vec2f } : {}),
        ...(surface.fragmentInputs?.localPos ? { localPos: d.vec3f } : {}),
    };
}

/** the raster state every surface pipeline shares (color, its single-sample twin, prepass, atlas).
 * A `screen` surface builds its own quads in clip space (lines), so their winding flips with segment
 * direction and back-face culling would drop half of them; world-space surfaces keep the cull (the
 * overdraw win + correct cutout/shadow facing) — `compileSurface`'s own law, one source of truth here so
 * the sites can't drift. */
export function surfacePrimitive(screen?: boolean): GPUPrimitiveState {
    return { topology: "triangle-list", cullMode: screen ? "none" : "back", frontFace: "ccw" };
}

/** whether a surface's own `layout` carries the `eids` + `globalTransforms` instancing convention —
 * mirrors `record()`'s test, run over the layout's `entries` instead. */
function isInstanced(surface: AnySurface): boolean {
    return "eids" in surface.layout.entries && "globalTransforms" in surface.layout.entries;
}

// A zero-custom-varying surface stays entirely TGSL. One fixed function computes the vertex payload;
// four thin entry shapes select only the optional built-in fragment inputs the surface declared. The
// payload is an ordinary function result, not an interstage struct, so carrying uv/localPos here costs no
// raster bandwidth. Custom varying names alone need the WGSL-bodied copier below.
const SurfaceVertex = d
    .struct({
        pos: d.vec4f,
        worldNormal: d.vec3f,
        eid: d.u32,
        world: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        color: d.vec4f,
        material: d.vec4f,
    })
    .$name("SurfaceVertex");

function colorVertex(surface: AnySurface, clip: boolean, suffix = clip ? "Clip" : "") {
    const instanced = isInstanced(surface);
    const screen = !!surface.screen;
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
        globalTransformRows: any[];
        partRowMap: any[];
        meshInstances: any[];
    };
    return tgpu
        .fn(
            [d.u32, d.u32],
            SurfaceVertex,
        )((vidx, iid) => {
            "use gpu";
            const v = layout.$.vertices[vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            const localNormal = octDecodeNormal(v.z);
            const uv = decodeUv(v.w, mq);
            let eid = d.u32(0);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            let color = d.vec4f(1);
            let material = d.vec4f(0, 1, 0, 1);
            let xform = identityXform();
            if (instanced) {
                const instance = bound.eids[iid];
                eid = instance.x;
                const encodedMeshInstance = instance.z;
                if (encodedMeshInstance !== 0) {
                    const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                    color = d.vec4f(engineLayout.$.materials[meshInstance.material].baseColor);
                    material = d.vec4f(engineLayout.$.materials[meshInstance.material].params);
                }
                xform = Xform(bound.globalTransforms[instance.y]);
                world = d.vec4f(xformPoint(xform, world.xyz), world.w);
                worldNormal = d.vec3f(xformNormal(xform, worldNormal));
            }
            let pos = d.vec4f(0);
            if (hasVs) {
                const patched = vsFn!(
                    VsIn({
                        localPos,
                        localNormal,
                        uv,
                        vidx,
                        eid,
                        iid,
                        xform,
                        world,
                        worldNormal,
                        color,
                        material,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
                if (screen) pos = d.vec4f(patched.clip);
            }
            if (!screen) pos = d.vec4f(std.mul(engineLayout.$.view.viewProj, world));
            if (clip) pos = d.vec4f(std.add(pos, d.vec4f(shadowLayout.$.tileRects.rects[0].x * 0)));
            return SurfaceVertex({
                pos,
                worldNormal: std.normalize(worldNormal),
                eid,
                world: world.xyz,
                uv,
                localPos,
                color,
                material,
            });
        })
        .$name(`${surface.name}${suffix}Vertex`);
}

function colorVs(surface: AnySurface, clip = false, suffix = clip ? "Clip" : "") {
    const vertex = colorVertex(surface, clip, suffix);
    const name = `${surface.name}${suffix}Vs`;
    const input = { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex };
    const fixed = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
    };
    const uv = !!surface.fragmentInputs?.uv;
    const localPos = !!surface.fragmentInputs?.localPos;
    if (uv && localPos) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, uv: d.vec2f, localPos: d.vec3f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    world: v.world,
                    uv: v.uv,
                    localPos: v.localPos,
                    color: v.color,
                    material: v.material,
                };
            })
            .$name(name);
    }
    if (uv) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, uv: d.vec2f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    world: v.world,
                    uv: v.uv,
                    color: v.color,
                    material: v.material,
                };
            })
            .$name(name);
    }
    if (localPos) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, localPos: d.vec3f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    world: v.world,
                    localPos: v.localPos,
                    color: v.color,
                    material: v.material,
                };
            })
            .$name(name);
    }
    return tgpu
        .vertexFn({ in: input, out: fixed })((i) => {
            "use gpu";
            const v = vertex(i.vidx, i.iid);
            return {
                pos: v.pos,
                worldNormal: v.worldNormal,
                eid: v.eid,
                world: v.world,
                color: v.color,
                material: v.material,
            };
        })
        .$name(name);
}

/**
 * the color-pass fragment entry: fills the four `standard/engine.ts` shading-seam privateVars
 * (`sunVisibility` via a real {@link sampleSunShadow} call, `fragWorld`, `fragCoord`, `pointScale`), builds the surface's
 * `fsCtxSchema` context (`uv`/`localPos` cross for real from the vs), and returns the surface's own
 * `fs` chunk's result verbatim (standard's `col` return,
 * unwrapped — a surface `fs` already returns `vec4f`, no lane locals: the prepass tag/depth lanes are a
 * separate pipeline, still unported).
 */
function colorFs(surface: AnySurface) {
    // Use the exact schema instance the author passed to `surface.fs`. Re-minting `fsCtxSchema()` here
    // is structurally equal but makes TypeGPU insert and warn about an implicit struct conversion.
    const CtxSchema = (
        surface.fs as unknown as { shell: { argTypes: [ReturnType<typeof fsCtxSchema>] } }
    ).shell.argTypes[0];
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const inputSchema = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
        ...fragmentInterstage(surface),
    };
    return tgpu
        .fragmentFn({
            in: inputSchema,
            out: d.vec4f,
        })((input: any) => {
            "use gpu";
            const worldNormal = std.normalize(input.worldNormal);
            sunVisibility.$ = sampleSunShadow(input.world, worldNormal);
            fragWorld.$ = d.vec3f(input.world);
            fragCoord.$ = d.vec4f(input.pos);
            pointScale.$ = 1;
            // force `shadowLayout`'s group-1 bindings into scope: `sampleSunShadow`/the transitively
            // pulled `pointShadowOf` (via `litPbr`) read `shadowMap`/`shadowSamp`/`sunShadow`/`pointAtlas`/
            // `pointShadows`/`tileRects` as free names inside their own WGSL bodies, invisible to
            // `tgpu.resolve`'s call-graph walk (the fog `fogKernel` forcedZero precedent) — folded into a
            // value the return genuinely uses, not a discarded local (the JS→WGSL transpiler would prune
            // a dead one)
            const forcedZero =
                (shadowLayout.$.pointShadows.casters[0].pos.x +
                    shadowLayout.$.tileRects.rects[0].x +
                    shadowLayout.$.sunShadow.enabled +
                    std.textureSampleCompareLevel(
                        shadowLayout.$.pointAtlas,
                        shadowLayout.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    ) +
                    std.textureSampleCompareLevel(
                        shadowLayout.$.shadowMap,
                        shadowLayout.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    )) *
                0;
            // `fsCtxSchema()`'s no-varyings default type param (`Record<string, never>`) carries a
            // `[x: string]: never` index signature that a plain object literal's own known keys never
            // structurally satisfy (a TS quirk over a generic-default index signature, not a real type
            // error) — the escape is the same shape as the `layout.$` cast above.
            //
            // `uv`/`localPos` now cross for real — resolved from the vs's own interpolated output,
            // not a zero-fill stand-in.
            const ctx = CtxSchema({
                eid: input.eid,
                world: input.world,
                worldNormal,
                uv: needUv ? input.uv : d.vec2f(0),
                localPos: needLocalPos ? input.localPos : d.vec3f(0),
                color: input.color,
                material: input.material,
            } as any);
            const col = surface.fs(ctx);
            return d.vec4f(std.add(col, d.vec4f(forcedZero)));
        })
        .$name(`${surface.name}Fs`);
}

/**
 * the position-only prepass vertex entry (empty lane set — the shadow map's own shape too): pulls
 * the 8 B position-only vertex from the surface's `layout.depthVariant` (a DISTINCT `TgpuBindGroupLayout`
 * instance from `layout`), decodes position alone (normal defaults `+Z`, uv `0`), applies the
 * standard instance transform, then splices the surface's own `vs` chunk when present. Inlined rather than
 * factored through a shared helper (probed live: a plain function marked `"use gpu"` can't take a host
 * object like `surface` as an argument — "Shellless functions can only accept arguments representing WGSL
 * resources" — so this duplicates {@link tagVs}'s math, matching the color copier's vertex math
 * rather than inventing a new factoring pattern this file doesn't otherwise use).
 */
function prepassVs(surface: AnySurface) {
    const instanced = isInstanced(surface);
    const screen = !!surface.screen;
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
        globalTransformRows: any[];
        partRowMap: any[];
        meshInstances: any[];
    };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: { pos: d.builtin.position },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            // the depth-only default, pinned
            // for the life of the vs, never touched by the instance transform below; a `vs` chunk reading `vsIn.localNormal` sees this default,
            // not the transformed `worldNormal`
            const localNormal = d.vec3f(0, 0, 1);
            const uv = d.vec2f(0, 0);
            let eid = d.u32(0);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            let color = d.vec4f(1);
            let material = d.vec4f(0, 1, 0, 1);
            let xform = identityXform();
            if (instanced) {
                const instance = bound.eids[input.iid];
                eid = instance.x;
                const encodedMeshInstance = instance.z;
                if (encodedMeshInstance !== 0) {
                    const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                    color = d.vec4f(engineLayout.$.materials[meshInstance.material].baseColor);
                    material = d.vec4f(engineLayout.$.materials[meshInstance.material].params);
                }
                xform = Xform(bound.globalTransforms[instance.y]);
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
                        color,
                        material,
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
            // the prepass pipelines' group set dense (`renderPrepass` binds `shadowGroup()` at 1,
            // inert: the stub receiver means nothing samples it)
            const forcedZero = shadowLayout.$.tileRects.rects[0].x * 0;
            return { pos: std.add(clip, d.vec4f(forcedZero)) };
        })
        .$name(`${surface.name}PrepassVs`);
}

/** the id-lane prepass vertex entry: {@link prepassVs}'s twin, crossing the flat `eid` varying
 * the tag fragment ({@link tagFs}) writes verbatim — `instanced ? eid : PICKING_ID_NONE`, the tag
 * lane's clear value in `COLOR_LANES`. */
function tagVs(surface: AnySurface) {
    const instanced = isInstanced(surface);
    const screen = !!surface.screen;
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
        globalTransformRows: any[];
        partRowMap: any[];
        meshInstances: any[];
    };
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                eid: d.interpolate("flat", d.u32),
                color: d.vec4f,
                material: d.interpolate("flat", d.vec4f),
            },
        })((input) => {
            "use gpu";
            const v = layout.$.vertices[input.vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            // pinned default — never touched by the instance transform, matching `prepassVs`'s law
            const localNormal = d.vec3f(0, 0, 1);
            const uv = d.vec2f(0, 0);
            let eid = d.u32(instanced ? 0 : PICKING_ID_NONE);
            let world = d.vec4f(localPos, 1);
            let worldNormal = d.vec3f(localNormal);
            let color = d.vec4f(1);
            let material = d.vec4f(0, 1, 0, 1);
            let xform = identityXform();
            if (instanced) {
                const instance = bound.eids[input.iid];
                eid = instance.x;
                const encodedMeshInstance = instance.z;
                if (encodedMeshInstance !== 0) {
                    const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                    color = d.vec4f(engineLayout.$.materials[meshInstance.material].baseColor);
                    material = d.vec4f(engineLayout.$.materials[meshInstance.material].params);
                }
                xform = Xform(bound.globalTransforms[instance.y]);
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
                        color,
                        material,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
                if (screen) clip = d.vec4f(patched.clip);
            }
            if (!screen) clip = d.vec4f(std.mul(engineLayout.$.view.viewProj, world));
            // same group-1 hole fill as `prepassVs` — the tag pipeline shares the prepass's
            // groups-0+2 shape
            const forcedZero = shadowLayout.$.tileRects.rects[0].x * 0;
            return { pos: std.add(clip, d.vec4f(forcedZero)), eid, color, material };
        })
        .$name(`${surface.name}PrepassTagVs`);
}

/**
 * the tag-lane fragment entry: the front-most fragment's `eid` verbatim ({@link tagVs}'s
 * varying already resolved the instanced/non-instanced default). Outputs `vec4u` rather than a bare `u32` — typegpu's `fragmentFn` constrains
 * every color output to a `vec4` family type (`FragmentOutConstrained`'s `FragmentColorValue = Vec4f |
 * Vec4i | Vec4u`, probed live: a bare `d.u32` fails the type constraint before the body even resolves), so
 * the eid rides lane 0 with the other three padded zero — WebGPU spec-legal against the single-channel
 * `r32uint` target (a fragment output may carry more components than the attachment's format; the excess
 * are dropped), and a real, disclosed WGSL-shape deviation the differential test must account for. This
 * compact entry is only the no-hook default; an authored `Surface.tag` uses the full-stream fragment
 * context through {@link authoredTagFs} or {@link varyingTagFs}.
 */
function tagFs(surface: AnySurface) {
    return tgpu
        .fragmentFn({ in: { eid: d.interpolate("flat", d.u32) }, out: d.vec4u })((input) => {
            "use gpu";
            return d.vec4u(input.eid, 0, 0, 0);
        })
        .$name(`${surface.name}PrepassTagFs`);
}

function authoredTagFs(surface: AnySurface) {
    const instanced = isInstanced(surface);
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const tagFn = surface.tag!;
    const Ctx = (tagFn as unknown as { shell: { argTypes: [unknown, unknown] } }).shell.argTypes[0];
    const input = {
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
        ...fragmentInterstage(surface),
    };
    return tgpu
        .fragmentFn({ in: input, out: d.vec4u })((fin) => {
            "use gpu";
            const ctx = (Ctx as ReturnType<typeof fsCtxSchema>)({
                eid: fin.eid,
                world: fin.world,
                worldNormal: std.normalize(fin.worldNormal),
                uv: needUv ? (fin as any).uv : d.vec2f(0),
                localPos: needLocalPos ? (fin as any).localPos : d.vec3f(0),
                color: fin.color,
                material: fin.material,
            });
            return d.vec4u(tagFn(ctx, instanced ? fin.eid : PICKING_ID_NONE), 0, 0, 0);
        })
        .$name(`${surface.name}PrepassTagFs`);
}

function clipFs(surface: AnySurface, tag: boolean) {
    const instanced = isInstanced(surface);
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const Ctx = (surface.fs as unknown as { shell: { argTypes: [ReturnType<typeof fsCtxSchema>] } })
        .shell.argTypes[0];
    const input = {
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
        ...fragmentInterstage(surface),
    };
    if (tag) {
        return tgpu
            .fragmentFn({ in: input, out: d.vec4u })((fin) => {
                "use gpu";
                const ctx = Ctx({
                    eid: fin.eid,
                    world: fin.world,
                    worldNormal: std.normalize(fin.worldNormal),
                    uv: needUv ? (fin as any).uv : d.vec2f(0),
                    localPos: needLocalPos ? (fin as any).localPos : d.vec3f(0),
                    color: fin.color,
                    material: fin.material,
                });
                surface.fs(ctx);
                return d.vec4u(instanced ? fin.eid : PICKING_ID_NONE, 0, 0, 0);
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
                uv: needUv ? (fin as any).uv : d.vec2f(0),
                localPos: needLocalPos ? (fin as any).localPos : d.vec3f(0),
                color: fin.color,
                material: fin.material,
            });
            surface.fs(ctx);
        })
        .$name(`${surface.name}ClipFs`);
}

/** Build the locked one-varying cutoff copier: the fragment entry exposes a fixed `v0` slot while this
 * raw helper reconstructs the author's exact FsCtx positionally. The schema comes from the authored fs
 * shell, never a freshly minted lookalike. */
function clipVaryingCopier(surface: AnySurface) {
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    if (keys.length !== 1) {
        throw new Error(
            `standard: surface "${surface.name}" declares ${keys.length} varyings — the clip copier carries exactly one custom varying`,
        );
    }
    const varyingSchema = varyings[keys[0]];
    const varyingType = (varyingSchema as unknown as { type: string }).type;
    const fsFn = surface.fs;
    const CtxSchema = (fsFn as unknown as { shell: { argTypes: [unknown] } }).shell.argTypes[0];
    const copier = tgpu
        .fn(
            [d.vec3f, d.u32, d.vec3f, d.vec2f, d.vec3f, d.vec4f, d.vec4f, varyingSchema],
            d.Void,
        )(/* wgsl */ `(worldNormalIn: vec3f, eid: u32, world: vec3f, uv: vec2f, localPos: vec3f, color: vec4f, material: vec4f, v0: ${varyingType}) {
    let ctx = Ctx(eid, world, normalize(worldNormalIn), uv, localPos, color, material, v0);
    fs(ctx);
}`)
        .$uses({ Ctx: CtxSchema, fs: fsFn })
        .$name(`${surface.name}ClipFsCopier`);
    return { varyingSchema, copier };
}

function varyingClipFs(surface: AnySurface, tag: boolean) {
    const instanced = isInstanced(surface);
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const { varyingSchema, copier } = clipVaryingCopier(surface);
    const entryIn = {
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
        ...fragmentInterstage(surface),
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
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
                    input.v0,
                );
                return d.vec4u(instanced ? input.eid : PICKING_ID_NONE, 0, 0, 0);
            })
            .$name(`${surface.name}ClipTagFs`);
    }
    return tgpu
        .fragmentFn({
            in: entryIn as unknown as typeof entryIn & { v0: d.Vec3f },
            out: d.Void,
        })((input) => {
            "use gpu";
            copier(
                input.worldNormal,
                input.eid,
                input.world,
                needUv ? (input as any).uv : d.vec2f(0),
                needLocalPos ? (input as any).localPos : d.vec3f(0),
                input.color,
                input.material,
                input.v0,
            );
        })
        .$name(`${surface.name}ClipFs`);
}

/**
 * the varyings-carrying color/prepass vertex entry (the varyings mechanism): TGSL has
 * no object-spread and no dynamic-key struct construction, so a shared "use gpu" body can't vary its
 * return shape per surface — a surface declaring `varyings` gets its own **WGSL-bodied copier**, distinct
 * from a fixed shared body. The copier does the real vertex math (vertex pull, quantized decode,
 * instance transform, the surface's own `vs` chunk) AND constructs the whole per-surface out struct in one
 * raw fn, its `out.<varying> = patched.<varying>;` lines JS-string-templated from `Object.keys(surface.
 * varyings)` (never through the transpiler). The thin entry stays real TGSL (typegpu's own header/
 * location machinery) and only binds + returns the copier's result — `const r = copier(...); return r;`,
 * since a direct return hits "Cannot resolve struct cast" (probed live via
 * `tgpu.resolve` before this landed).
 */
function varyingVs(surface: AnySurface, clip = false, suffix = clip ? "Clip" : "") {
    const varyings = surface.varyings ?? {};
    const vsFn = surface.vs;
    if (Object.keys(varyings).length === 0) {
        throw new Error(
            `standard: surface "${surface.name}" reached the raw varying copier without a custom varying`,
        );
    }
    if (!vsFn) {
        throw new Error(
            `standard: surface "${surface.name}" declares varyings with no vs — a varying can only be written by the surface's own vs chunk`,
        );
    }
    const hasVs = !!vsFn;
    const fragmentFields = fragmentInterstage(surface);
    const instanced = isInstanced(surface);
    // a `screen` surface's own projection is the patch's `clip` lane; everything else projects
    // `view.viewProj * world` after the chunk, so displacing `world` still projects
    const screen = !!surface.screen;
    const layout = surface.layout;
    // explicit interstage locations from VARYING_BASE up: the fs entry carries the varying under the
    // fixed internal name `v0`, which typegpu's pipeline connection can't match to the vs side's real
    // name — an unmatched fs field auto-assigns from 0 and collides with the matched fixed fields.
    // Pinning both sides to the same explicit slot makes the location, not the name, the contract
    // (auto-assignment skips explicitly-taken locations)
    const located = Object.fromEntries(
        Object.entries(varyings).map(([k, s], i) => [k, d.location(VARYING_BASE + i, s)]),
    );
    // one struct on the wire: the entry is the raw WGSL body itself, so its `Out` is the IO struct
    // typegpu mints for `out` and there is no second struct to convert into. A thin TGSL entry that
    // returned a separate copier's struct made typegpu convert one into the other at every resolve and
    // warn (`[implicit-conversion] … r: struct:<surface>VsOut`), one line per surface on every
    // boot. `pipelines.test.ts` spies the resolve for the warning.
    const assigns = Object.keys(varyings)
        .map((k) => `    out.${k} = patched.${k};`)
        .join("\n");
    const fragmentAssigns = `${surface.fragmentInputs?.uv ? "    out.uv = uv;\n" : ""}${surface.fragmentInputs?.localPos ? "    out.localPos = localPos;\n" : ""}`;
    // 0.12 removed `layout.bound`: the dereferenced `layout.$.x` throws outside an actual TGSL body
    // ("Direct access to buffer values..."), including inside a `tgpu.lazy` compute (it forces normal
    // mode). The fix a raw-WGSL-string `uses` external needs is to defer the per-field property read
    // to resolution time itself — pass the whole `layout.$` proxy as one external and let the WGSL
    // text's own `bound.vertices`-style dot chain do the (codegen-mode-safe) lookup.
    const bound = layout.$;
    const engine = engineLayout.$;
    const shadowG = shadowLayout.$;
    const uses: Record<string, unknown> = {
        bound,
        engine,
        decodePos,
        decodeUv,
        meshIdOf,
        octDecodeNormal,
        Xform,
    };
    if (hasVs) {
        uses.VsIn = VsIn;
        uses.vs = vsFn;
    }
    // `engine` rides unconditionally even for a `screen` copier, which projects nothing: resolution is
    // per dot-chain reference in the WGSL text, not per top-level external, so an unreferenced
    // `engine.view` emits no group-0 declaration (pinned by pipelines.test.ts's `not.toContain("var<uniform> view: ViewUniforms;")`)
    if (clip) uses.shadowG = shadowG;
    if (instanced) {
        uses.xformPoint = xformPoint;
        uses.xformNormal = xformNormal;
    }
    return tgpu
        .vertexFn({
            in: { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex },
            out: {
                pos: d.builtin.position,
                worldNormal: d.vec3f,
                eid: d.interpolate("flat", d.u32),
                world: d.vec3f,
                color: d.vec4f,
                material: d.interpolate("flat", d.vec4f),
                ...fragmentFields,
                // no type-directed `@interpolate(flat)` insertion — an INTEGER varying is unsupported
                // and fails loudly at resolve/device compile; every shipped varying is float-typed.
                ...located,
            },
        })(/* wgsl */ `{
    let v = bound.vertices[in.vidx];
    let mq = engine.meshQuant[meshIdOf(v.y)];
    let localPos = decodePos(v.x, v.y, mq);
    let localNormal = octDecodeNormal(v.z);
    let uv = decodeUv(v.w, mq);
    var eid: u32 = 0u;
    var xform = Xform(vec3f(0.0), vec4f(0.0, 0.0, 0.0, 1.0), vec3f(1.0));
    var world = vec4f(localPos, 1.0);
    var worldNormal = vec3f(localNormal);
    var color = vec4f(1.0);
    var material = vec4f(0.0, 1.0, 0.0, 1.0);
${
    instanced
        ? `    let instance = bound.eids[in.iid];
    eid = instance.x;
    let encodedMeshInstance = instance.z;
    if (encodedMeshInstance != 0u) {
        let meshInstance = bound.meshInstances[encodedMeshInstance - 1u];
        color = engine.materials[meshInstance.material].baseColor;
        material = engine.materials[meshInstance.material].params;
    }
    xform = bound.globalTransforms[instance.y];
    world = vec4f(xformPoint(xform, world.xyz), world.w);
    worldNormal = vec3f(xformNormal(xform, worldNormal));
`
        : ""
}${
    hasVs
        ? `    let patched = vs(VsIn(localPos, localNormal, uv, vidx, eid, iid, xform, world, worldNormal, color, material));
    world = patched.world;
    worldNormal = patched.worldNormal;
`
        : ""
}
    var out: Out;
    out.pos = ${screen ? "patched.clip" : "engine.view.viewProj * world"}${clip ? " + vec4f(shadowG.tileRects.rects[0].x * 0.0)" : ""};
    out.worldNormal = normalize(worldNormal);
    out.eid = eid;
    out.world = world.xyz;
    out.color = color;
    out.material = material;
${fragmentAssigns}
${assigns}
    return out;
}`)
        .$uses(uses)
        .$name(`${surface.name}${suffix}Vs`);
}

/** the group-1 forcing touch as one callable fold: `sampleSunShadow` and the transitively-pulled
 * `pointShadowOf` read `shadowMap`/`shadowSamp`/`sunShadow`/`pointAtlas`/`pointShadows`/`tileRects` as free
 * names inside their own WGSL bodies, invisible to `tgpu.resolve`'s call-graph walk (the fog `fogKernel`
 * forcedZero precedent). {@link colorFs} and {@link bgFs} inline the same expression; the
 * varyings-carrying fragment entry calls this instead so its four arity arms don't each carry a copy. */
const shadowForce = tgpu
    .fn(
        [],
        d.f32,
    )(() => {
        "use gpu";
        return (
            (shadowLayout.$.pointShadows.casters[0].pos.x +
                shadowLayout.$.tileRects.rects[0].x +
                shadowLayout.$.sunShadow.enabled +
                std.textureSampleCompareLevel(
                    shadowLayout.$.pointAtlas,
                    shadowLayout.$.shadowSamp,
                    d.vec2f(0, 0),
                    0,
                ) +
                std.textureSampleCompareLevel(
                    shadowLayout.$.shadowMap,
                    shadowLayout.$.shadowSamp,
                    d.vec2f(0, 0),
                    0,
                )) *
            0
        );
    })
    .$name("shadowForce");

/**
 * the varyings-carrying color-pass fragment entry — `colorFs`'s twin for a surface declaring
 * `varyings` ({@link varyingVs}'s matching half). typegpu's entry-input router can't cross a whole
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
 * past it (the hard 4-slot custom interpolator budget). Everything else — the copier's
 * signature, the interstage locations, the vs side — is already N-general.
 *
 * The `Ctx` the copier constructs must be the exact schema instance `surface.fs` was declared against —
 * `fsCtxSchema(varyings)` mints a fresh struct object on every call (the `pointCastersSchema`/
 * `tileRectsSchema` precedent), so re-minting one here would pass a structurally-identical but
 * distinct-identity struct into `fs`'s call, which resolved clean device-free but is real risk at device
 * compile (WGSL struct-argument typing isn't purely structural) — reading it off `fsFn.shell.argTypes[0]`
 * (the schema the author's own `tgpu.fn([fsCtxSchema(...)], ...)` call recorded) is the one source of truth.
 */
function varyingFs(surface: AnySurface) {
    const varyings = surface.varyings ?? {};
    const varyingKeys = Object.keys(varyings);
    if (varyingKeys.length < 1 || varyingKeys.length > MAX_VARYINGS) {
        throw new Error(
            `standard: surface "${surface.name}" declares ${varyingKeys.length} varyings — the fs entry carries 1 to ${MAX_VARYINGS} (gpu.md rule 9's custom interpolator budget)`,
        );
    }
    const schemas = varyingKeys.map((k) => varyings[k]);
    const params = schemas
        .map((s, i) => `v${i}: ${(s as unknown as { type: string }).type}`)
        .join(", ");
    const fsFn = surface.fs;
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const CtxSchema = (fsFn as unknown as { shell: { argTypes: [unknown] } }).shell.argTypes[0];
    const copier = tgpu
        .fn(
            [d.vec4f, d.vec3f, d.u32, d.vec3f, d.vec2f, d.vec3f, d.vec4f, d.vec4f, ...schemas],
            d.vec4f,
        )(/* wgsl */ `(pos: vec4f, worldNormalIn: vec3f, eid: u32, world: vec3f, uv: vec2f, localPos: vec3f, color: vec4f, material: vec4f, ${params}) -> vec4f {
    let worldNormal = normalize(worldNormalIn);
    let ctx = Ctx(eid, world, worldNormal, uv, localPos, color, material, ${schemas.map((_, i) => `v${i}`).join(", ")});
    return fs(ctx);
}`)
        .$uses({ Ctx: CtxSchema, fs: fsFn })
        .$name(`${surface.name}FsCopier`);

    // a slot's schema is the widened `AnyWgslData` — real (any concrete vector/scalar schema at runtime),
    // but too loose for `fragmentFn`'s `in:` constraint and the resulting `input`'s field types to
    // type-check without a cast, the same escape the vertex copier's `layout.$` cast uses. The explicit
    // location pairs with the vs side's (`VARYING_BASE + i` — see varyingVs's why: the internal
    // `v<i>` names are unmatchable, the slot is the contract). No flat-interpolate insertion, so an
    // integer varying is unsupported and fails loudly at resolve/device compile.
    const slot = (i: number) => d.location(VARYING_BASE + i, schemas[i] as d.Vec3f);
    const base = {
        pos: d.builtin.position,
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
        ...fragmentInterstage(surface),
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
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
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
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
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
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
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
                needUv ? (input as any).uv : d.vec2f(0),
                needLocalPos ? (input as any).localPos : d.vec3f(0),
                input.color,
                input.material,
                input.v0,
                input.v1,
                input.v2,
                input.v3,
            );
            return d.vec4f(std.add(col, d.vec4f(shadowForce())));
        })
        .$name(name);
}

function varyingTagFs(surface: AnySurface) {
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    if (keys.length < 1 || keys.length > MAX_VARYINGS) {
        throw new Error(
            `standard: surface "${surface.name}" declares ${keys.length} varyings — the tag entry carries 1 to ${MAX_VARYINGS} (gpu.md rule 9's custom interpolator budget)`,
        );
    }
    const schemas = keys.map((key) => varyings[key]);
    const params = schemas
        .map((schema, i) => `v${i}: ${(schema as unknown as { type: string }).type}`)
        .join(", ");
    const tagFn = surface.tag!;
    const Ctx = (tagFn as unknown as { shell: { argTypes: [unknown, unknown] } }).shell.argTypes[0];
    const copier = tgpu
        .fn(
            [d.vec3f, d.u32, d.vec3f, d.vec2f, d.vec3f, d.vec4f, d.vec4f, d.u32, ...schemas],
            d.u32,
        )(/* wgsl */ `(worldNormalIn: vec3f, eid: u32, world: vec3f, uv: vec2f, localPos: vec3f, color: vec4f, material: vec4f, defaultTag: u32, ${params}) -> u32 {
    let ctx = Ctx(eid, world, normalize(worldNormalIn), uv, localPos, color, material, ${schemas.map((_, i) => `v${i}`).join(", ")});
    return tag(ctx, defaultTag);
}`)
        .$uses({ Ctx, tag: tagFn })
        .$name(`${surface.name}TagCopier`);
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const instanced = isInstanced(surface);
    const slot = (i: number) => d.location(VARYING_BASE + i, schemas[i] as d.Vec3f);
    const base = {
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
        ...fragmentInterstage(surface),
    };
    const name = `${surface.name}PrepassTagFs`;
    if (keys.length === 1) {
        const entryIn = { ...base, v0: slot(0) } as unknown as typeof base & { v0: d.Vec3f };
        return tgpu
            .fragmentFn({ in: entryIn, out: d.vec4u })((input) => {
                "use gpu";
                const value = copier(
                    input.worldNormal,
                    input.eid,
                    input.world,
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
                    instanced ? input.eid : PICKING_ID_NONE,
                    input.v0,
                );
                return d.vec4u(value, 0, 0, 0);
            })
            .$name(name);
    }
    if (keys.length === 2) {
        const entryIn = { ...base, v0: slot(0), v1: slot(1) } as unknown as typeof base & {
            v0: d.Vec3f;
            v1: d.Vec3f;
        };
        return tgpu
            .fragmentFn({ in: entryIn, out: d.vec4u })((input) => {
                "use gpu";
                const value = copier(
                    input.worldNormal,
                    input.eid,
                    input.world,
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
                    instanced ? input.eid : PICKING_ID_NONE,
                    input.v0,
                    input.v1,
                );
                return d.vec4u(value, 0, 0, 0);
            })
            .$name(name);
    }
    if (keys.length === 3) {
        const entryIn = {
            ...base,
            v0: slot(0),
            v1: slot(1),
            v2: slot(2),
        } as unknown as typeof base & { v0: d.Vec3f; v1: d.Vec3f; v2: d.Vec3f };
        return tgpu
            .fragmentFn({ in: entryIn, out: d.vec4u })((input) => {
                "use gpu";
                const value = copier(
                    input.worldNormal,
                    input.eid,
                    input.world,
                    needUv ? (input as any).uv : d.vec2f(0),
                    needLocalPos ? (input as any).localPos : d.vec3f(0),
                    input.color,
                    input.material,
                    instanced ? input.eid : PICKING_ID_NONE,
                    input.v0,
                    input.v1,
                    input.v2,
                );
                return d.vec4u(value, 0, 0, 0);
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
        .fragmentFn({ in: entryIn, out: d.vec4u })((input) => {
            "use gpu";
            const value = copier(
                input.worldNormal,
                input.eid,
                input.world,
                needUv ? (input as any).uv : d.vec2f(0),
                needLocalPos ? (input as any).localPos : d.vec3f(0),
                input.color,
                input.material,
                instanced ? input.eid : PICKING_ID_NONE,
                input.v0,
                input.v1,
                input.v2,
                input.v3,
            );
            return d.vec4u(value, 0, 0, 0);
        })
        .$name(name);
}

/**
 * compile a `Surface`'s color-pass pipeline(s): the opaque `color` pipeline, or — for a `blend:
 * "alpha"` surface — the single blended `transparent` pipeline instead (exactly one of the two
 * compiles, never both). Cached by name +
 * exact source-surface/layout identity, so replacing a
 * registry entry after warm cannot inherit the previous owner's pipelines. A `screen` surface projects
 * through its own `vs` chunk (`patch.clip`) and rasterizes un-culled.
 */
export function compileSurface<
    B extends Record<string, Binding>,
    V extends Record<string, AnyWgslData>,
>(world: World, surface: Surface<B, V>, capacity: number): CompiledSurface {
    const _render = world.resource(RenderContext);

    const key = surface.name;
    const cached = pipelineState(world).compiledSurfaces.get(key);
    if (cached?.owner === surface && cached.layout === surface.layout) return cached;
    const resolved = surface;
    // a `screen` surface's clip position comes from its own `vs` chunk's `patch.clip` and from nowhere
    // else — with no `vs` every vertex would collapse to the origin, silently drawing nothing
    if (resolved.screen && !resolved.vs) {
        throw new Error(
            `standard: surface "${surface.name}" is a screen surface with no vs — only its own vs chunk can supply the clip position`,
        );
    }
    const primitive = surfacePrimitive(resolved.screen);
    // TS can't narrow `vertex`/`fragment` as a matched pair across the ternary (their varying-record types
    // only agree structurally, proven at runtime by the differential + bench gates, not by the branch's
    // static shape) — the same class of escape the vertex copier's `layout.$` cast uses elsewhere.
    const hasVaryings = !!resolved.varyings && Object.keys(resolved.varyings).length > 0;
    const vertex = hasVaryings ? varyingVs(resolved) : colorVs(resolved);
    const fragment = (hasVaryings ? varyingFs(resolved) : colorFs(resolved)) as ReturnType<
        typeof colorFs
    >;
    const args: CompiledSurface["args"] = {
        vertex,
        fragment,
        blend: resolved.blend,
        primitive,
        name: surface.name,
    };
    let compiled: CompiledSurface;
    if (resolved.blend === "alpha") {
        const transparent = world.gpu.root
            .createRenderPipeline({
                vertex,
                fragment,
                targets: { format: _render.format, blend: ALPHA_BLEND },
                primitive,
                depthStencil: {
                    format: DEPTH_FORMAT,
                    depthWriteEnabled: false,
                    depthCompare: "greater-equal",
                },
                multisample: { count: SAMPLE_COUNT },
            })
            .$name(`standard-transparent-${args.name}`);
        // `blend: "alpha"` casts nothing (a transparent pixel has no single owner, `compileSurface`'s own
        // rule) — the same reason its prepass map stays empty
        compiled = {
            owner: surface as AnySurface,
            layout: surface.layout as SurfaceLayout<Record<string, Binding>>,
            color: null,
            transparent,
            prepass: new Map(),
            point: null,
            cascade: null,
            single: null,
            args,
        };
    } else {
        const color = world.gpu.root
            .createRenderPipeline({
                vertex,
                fragment,
                targets: { format: _render.format },
                primitive,
                depthStencil: {
                    format: DEPTH_FORMAT,
                    depthWriteEnabled: true,
                    depthCompare: "greater",
                },
                multisample: { count: SAMPLE_COUNT },
            })
            .$name(`standard-${args.name}`);
        compiled = {
            owner: surface as AnySurface,
            layout: surface.layout as SurfaceLayout<Record<string, Binding>>,
            color,
            transparent: null,
            prepass: new Map(),
            point: null,
            cascade: null,
            single: null,
            args,
        };
    }
    compiled.prepass = compilePrepass(world, resolved);
    if (resolved.blend !== "alpha") {
        const { point, cascade } = compileShadow(world, resolved, capacity);
        compiled.point = point;
        compiled.cascade = cascade;
    }
    pipelineState(world).compiledSurfaces.set(key, compiled);
    return compiled;
}

/**
 * compile a surface's single-sample (AA-off) color twin, once, the first frame a no-AA camera
 * draws it (the wrapper is cheap; the real resolve+create lands at the twin's first draw). Reuses the compiled entry fns, so only
 * `multisample.count` differs.
 */
export function ensureSingle(world: World, t: CompiledSurface): void {
    const _render = world.resource(RenderContext);

    if (t.single) return;
    const { vertex, fragment, blend, primitive, name } = t.args;
    if (blend === "alpha") {
        const transparent = world.gpu.root
            .createRenderPipeline({
                vertex,
                fragment,
                targets: { format: _render.format, blend: ALPHA_BLEND },
                primitive,
                depthStencil: {
                    format: DEPTH_FORMAT,
                    depthWriteEnabled: false,
                    depthCompare: "greater-equal",
                },
                multisample: { count: 1 },
            })
            .$name(`standard-transparent-${name}-1x`);
        t.single = { color: null, transparent };
        return;
    }
    const color = world.gpu.root
        .createRenderPipeline({
            vertex,
            fragment,
            targets: { format: _render.format },
            primitive,
            depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: true,
                depthCompare: "greater",
            },
            multisample: { count: 1 },
        })
        .$name(`standard-${name}-1x`);
    t.single = { color, transparent: null };
}

/**
 * compile a `Surface`'s prepass pipelines: the position-only depth pipeline (key `""`,
 * vertex-only except for a `clip` surface's cutoff fragment) and the id-lane pipeline (key `"tag"`, the
 * `COLOR_LANES` tag lane). Opaque depth-only surfaces compile off `layout.depthVariant`; clipped
 * surfaces execute their authored cutoff and therefore use the full layout/main vertex stream. An
 * authored tag hook also uses the full stream, independently of the depth-only pipeline, through
 * `SurfaceGroupEntry.tag`. A `blend: "alpha"` surface casts no prepass at all — same rule
 * `compileSurface` applies (a transparent pixel has no single owner, writes no prepass depth) — so its map
 * stays empty.
 */
function compilePrepass(world: World, surface: AnySurface): Map<string, TgpuRenderPipeline<any>> {
    const prepass = new Map<string, TgpuRenderPipeline<any>>();
    if (surface.blend === "alpha") return prepass;
    const primitive = surfacePrimitive(surface.screen);
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    // the receiver stub bound per pipeline (`pointShadowStub`): a vs-chunk surface's
    // `litPbr` statically reaches `pointShadowOf`, whose free names the depth passes never declare or
    // bind — the stub keeps these modules group-0/2-only
    const root = world.gpu.root.with(pointShadowSlot, pointShadowStub);
    const clip = surface.blend === "clip";
    const varying = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const authoredTag = !!surface.tag;
    const depthOnly = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingVs(surface, true)
                    : colorVs(surface, true)
                : prepassVs(surface),
            ...(clip
                ? {
                      fragment: (varying
                          ? varyingClipFs(surface, false)
                          : clipFs(surface, false)) as never,
                  }
                : {}),
            primitive,
            depthStencil,
        })
        .$name(`standard-prepass-${surface.name}`);
    prepass.set("", depthOnly);
    const tag = root
        .createRenderPipeline({
            vertex: authoredTag
                ? varying
                    ? varyingVs(surface, true, "PrepassTag")
                    : colorVs(surface, true, "PrepassTag")
                : clip
                  ? varying
                      ? varyingVs(surface, true)
                      : colorVs(surface, true)
                  : tagVs(surface),
            fragment: authoredTag
                ? ((varying ? varyingTagFs(surface) : authoredTagFs(surface)) as never)
                : clip
                  ? ((varying ? varyingClipFs(surface, true) : clipFs(surface, true)) as never)
                  : tagFs(surface),
            targets: { format: PICKING_ID_FORMAT },
            primitive,
            depthStencil,
        })
        .$name(`standard-prepass-tag-${surface.name}`);
    prepass.set("tag", tag);
    return prepass;
}

/**
 * the shared point/cascade fragment entry: the tile-seam discard alone — clamped to
 * `layout.depthVariant` position + the `tileBox` varying its matching vs writes.
 * Atlas-size-independent (the vs bakes the atlas scale into `tileBox` already), so ONE instance serves every
 * surface's point pipeline AND every surface's cascade pipeline (the VS's rect-index formula
 * and atlas constants differ per atlas, passed to its builder). A `clip` surface uses the wider per-surface
 * fragment below so the same material cutoff holes its atlas depth.
 */
const shadowFs = tgpu
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
 * the point/cascade shadow-atlas vertex entry: pulls the 8 B position-only vertex from `layout.depthVariant` (the
 * `prepassVs` shape), reads the re-gathered `(eid, globalTransformRow, encodedMeshInstanceSlot, combo)` instance at the
 * surface's `eids` lane, applies the instance transform, splices the surface's own `vs` chunk when present,
 * then projects by that combo's tile-folded viewProj (`shadowLayout.$.faceVP.m[combo]`) and computes the
 * `tileBox` seam-discard bounds from `shadowLayout.$.tileRects` (indexed `slot·6+face` for the point atlas,
 * `slot` alone for the cascade atlas — indexed differently per atlas) scaled by the atlas's pixel size.
 * Only an **instanced** surface reaches here (only `eids`+`globalTransforms` gives a per-instance member to
 * re-gather against) — `compileShadow` gates the call, so this never runs for a non-instanced surface.
 */
function shadowVs(
    surface: AnySurface,
    shadowGroup: TgpuBindGroupLayout<any>,
    atlas: number,
    cascade: boolean,
    _capacity: number,
) {
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout.depthVariant;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
        globalTransformRows: any[];
        partRowMap: any[];
        meshInstances: any[];
    };
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
            // the depth-only default — never touched by the instance transform, matching
            // `prepassVs`'s pinned law
            const localNormal = d.vec3f(0, 0, 1);
            const uv = d.vec2f(0, 0);
            const instance = bound.eids[input.iid];
            const eid = instance.x;
            const combo = instance.w;
            let color = d.vec4f(1);
            let material = d.vec4f(0, 1, 0, 1);
            const encodedMeshInstance = instance.z;
            if (encodedMeshInstance !== 0) {
                const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                color = d.vec4f(engineLayout.$.materials[meshInstance.material].baseColor);
                material = d.vec4f(engineLayout.$.materials[meshInstance.material].params);
            }
            const xform = Xform(bound.globalTransforms[instance.y]);
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
                        color,
                        material,
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

const ClipShadowVertex = d
    .struct({
        pos: d.vec4f,
        tileBox: d.vec4f,
        worldNormal: d.vec3f,
        eid: d.u32,
        world: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        color: d.vec4f,
        material: d.vec4f,
    })
    .$name("ClipShadowVertex");

function clipShadowVertex(
    surface: AnySurface,
    shadowGroup: TgpuBindGroupLayout<any>,
    atlas: number,
    cascade: boolean,
    _capacity: number,
) {
    const hasVs = !!surface.vs;
    const vsFn = surface.vs;
    const layout = surface.layout;
    const bound = layout.$ as unknown as {
        eids: any[];
        globalTransforms: any[];
        globalTransformRows: any[];
        partRowMap: any[];
        meshInstances: any[];
    };
    const shadowBound = shadowGroup.$ as unknown as {
        faceVP: { m: any[] };
        comboMeta: { m: any[] };
        tileRects: { rects: any[] };
    };
    return tgpu
        .fn(
            [d.u32, d.u32],
            ClipShadowVertex,
        )((vidx, iid) => {
            "use gpu";
            const v = layout.$.vertices[vidx];
            const mq = engineLayout.$.meshQuant[meshIdOf(v.y)];
            const localPos = decodePos(v.x, v.y, mq);
            const localNormal = octDecodeNormal(v.z);
            const uv = decodeUv(v.w, mq);
            const instance = bound.eids[iid];
            const eid = instance.x;
            const combo = instance.w;
            let color = d.vec4f(1);
            let material = d.vec4f(0, 1, 0, 1);
            const encodedMeshInstance = instance.z;
            if (encodedMeshInstance !== 0) {
                const meshInstance = bound.meshInstances[encodedMeshInstance - 1];
                color = d.vec4f(engineLayout.$.materials[meshInstance.material].baseColor);
                material = d.vec4f(engineLayout.$.materials[meshInstance.material].params);
            }
            const xform = Xform(bound.globalTransforms[instance.y]);
            let world = d.vec4f(xformPoint(xform, localPos), 1);
            let worldNormal = d.vec3f(xformNormal(xform, localNormal));
            if (hasVs) {
                const patched = vsFn!(
                    VsIn({
                        localPos,
                        localNormal,
                        uv,
                        vidx,
                        eid,
                        iid,
                        xform,
                        world,
                        worldNormal,
                        color,
                        material,
                    }),
                );
                world = d.vec4f(patched.world);
                worldNormal = d.vec3f(patched.worldNormal);
            }
            const m = shadowBound.comboMeta.m[combo];
            const rect = cascade
                ? shadowBound.tileRects.rects[m.x]
                : shadowBound.tileRects.rects[m.x * 6 + m.y];
            return ClipShadowVertex({
                pos: std.mul(shadowBound.faceVP.m[combo], world),
                tileBox: d.vec4f(std.mul(atlas, rect.xy), rect.z * atlas, 0),
                worldNormal: std.normalize(worldNormal),
                eid,
                world: world.xyz,
                uv,
                localPos,
                color,
                material,
            });
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipVertex`);
}

function clipShadowVs(
    surface: AnySurface,
    shadowGroup: TgpuBindGroupLayout<any>,
    atlas: number,
    cascade: boolean,
    capacity: number,
) {
    const vertex = clipShadowVertex(surface, shadowGroup, atlas, cascade, capacity);
    const name = `${surface.name}${cascade ? "Cascade" : "Point"}ClipVs`;
    const input = { vidx: d.builtin.vertexIndex, iid: d.builtin.instanceIndex };
    const fixed = {
        pos: d.builtin.position,
        tileBox: d.interpolate("flat", d.vec4f),
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
    };
    const uv = !!surface.fragmentInputs?.uv;
    const localPos = !!surface.fragmentInputs?.localPos;
    if (uv && localPos) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, uv: d.vec2f, localPos: d.vec3f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    tileBox: v.tileBox,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    world: v.world,
                    uv: v.uv,
                    localPos: v.localPos,
                    color: v.color,
                    material: v.material,
                };
            })
            .$name(name);
    }
    if (uv) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, uv: d.vec2f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    tileBox: v.tileBox,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    world: v.world,
                    uv: v.uv,
                    color: v.color,
                    material: v.material,
                };
            })
            .$name(name);
    }
    if (localPos) {
        return tgpu
            .vertexFn({ in: input, out: { ...fixed, localPos: d.vec3f } })((i) => {
                "use gpu";
                const v = vertex(i.vidx, i.iid);
                return {
                    pos: v.pos,
                    tileBox: v.tileBox,
                    worldNormal: v.worldNormal,
                    eid: v.eid,
                    world: v.world,
                    localPos: v.localPos,
                    color: v.color,
                    material: v.material,
                };
            })
            .$name(name);
    }
    return tgpu
        .vertexFn({ in: input, out: fixed })((i) => {
            "use gpu";
            const v = vertex(i.vidx, i.iid);
            return {
                pos: v.pos,
                tileBox: v.tileBox,
                worldNormal: v.worldNormal,
                eid: v.eid,
                world: v.world,
                color: v.color,
                material: v.material,
            };
        })
        .$name(name);
}

/** The atlas-projecting half of the locked one-varying clip mechanism. Dynamic varying names stay in a
 * raw copier; the thin entry assigns the authored field and the fragment receives it as fixed `v0` at
 * the same explicit location. */
function varyingShadowVs(
    surface: AnySurface,
    shadowGroup: TgpuBindGroupLayout<any>,
    atlas: number,
    cascade: boolean,
    _capacity: number,
) {
    const varyings = surface.varyings ?? {};
    const keys = Object.keys(varyings);
    if (keys.length !== 1 || !surface.vs) {
        throw new Error(
            `standard: surface "${surface.name}" needs at most one authored varying for its clip shadow copier`,
        );
    }
    const hasVs = !!surface.vs;
    const fragmentFields = fragmentInterstage(surface);
    const layout = surface.layout;
    // see varyingVs's why: `layout.$.x` throws outside codegen mode (including inside `tgpu.lazy`),
    // so the whole `$` proxy rides as one external and the WGSL text's own dot chain defers the field read.
    const bound = layout.$;
    const shadow = shadowGroup.$;
    const Out = d
        .struct({
            pos: d.vec4f,
            tileBox: d.vec4f,
            worldNormal: d.vec3f,
            eid: d.u32,
            world: d.vec3f,
            color: d.vec4f,
            material: d.vec4f,
            ...fragmentFields,
            ...varyings,
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipOut`);
    const assigns = keys.map((key) => `    out.${key} = patched.${key};`).join("\n");
    const fragmentAssigns = `${surface.fragmentInputs?.uv ? "    out.uv = uv;\n" : ""}${surface.fragmentInputs?.localPos ? "    out.localPos = localPos;\n" : ""}`;
    const rect = cascade ? "m.x" : "m.x * 6u + m.y";
    const copier = tgpu
        .fn(
            [d.u32, d.u32],
            Out,
        )(/* wgsl */ `(vidx: u32, iid: u32) -> Out {
    let v = bound.vertices[vidx];
    let mq = engine.meshQuant[meshIdOf(v.y)];
    let localPos = decodePos(v.x, v.y, mq);
    let localNormal = octDecodeNormal(v.z);
    let uv = decodeUv(v.w, mq);
    let instance = bound.eids[iid];
    let eid = instance.x;
    let combo = instance.w;
    let encodedMeshInstance = instance.z;
    var color = vec4f(1.0);
    var material = vec4f(0.0, 1.0, 0.0, 1.0);
    if (encodedMeshInstance != 0u) {
        let meshInstance = bound.meshInstances[encodedMeshInstance - 1u];
        color = engine.materials[meshInstance.material].baseColor;
        material = engine.materials[meshInstance.material].params;
    }
    let xform = bound.globalTransforms[instance.y];
    var world = vec4f(xformPoint(xform, localPos), 1.0);
    var worldNormal = vec3f(xformNormal(xform, localNormal));
${
    hasVs
        ? `    let patched = vs(VsIn(localPos, localNormal, uv, vidx, eid, iid, xform, world, worldNormal, color, material));
    world = patched.world;
    worldNormal = patched.worldNormal;
`
        : ""
}
    let m = shadow.comboMeta.m[combo];
    let rect = shadow.tileRects.rects[${rect}];
    var out: Out;
    out.pos = shadow.faceVP.m[combo] * world;
    out.tileBox = vec4f(${atlas}.0 * rect.xy, rect.z * ${atlas}.0, 0.0);
    out.worldNormal = normalize(worldNormal);
    out.eid = eid;
    out.world = world.xyz;
    out.color = color;
    out.material = material;
${fragmentAssigns}
${assigns}
    return out;
}`)
        .$uses({
            Out,
            bound,
            engine: engineLayout.$,
            shadow,
            decodePos,
            decodeUv,
            meshIdOf,
            octDecodeNormal,
            xformPoint,
            xformNormal,
            ...(hasVs ? { VsIn, vs: surface.vs } : {}),
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
                color: d.vec4f,
                material: d.interpolate("flat", d.vec4f),
                ...fragmentFields,
                ...located,
            },
        })((input) => {
            "use gpu";
            const out = copier(input.vidx, input.iid);
            return out;
        })
        .$name(`${surface.name}${cascade ? "Cascade" : "Point"}ClipVs`);
}

function varyingShadowFs(surface: AnySurface) {
    const { varyingSchema, copier } = clipVaryingCopier(surface);
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
    const entryIn = {
        pos: d.builtin.position,
        tileBox: d.interpolate("flat", d.vec4f),
        worldNormal: d.vec3f,
        eid: d.interpolate("flat", d.u32),
        world: d.vec3f,
        color: d.vec4f,
        material: d.interpolate("flat", d.vec4f),
        ...fragmentInterstage(surface),
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
            copier(
                input.worldNormal,
                input.eid,
                input.world,
                needUv ? (input as any).uv : d.vec2f(0),
                needLocalPos ? (input as any).localPos : d.vec3f(0),
                input.color,
                input.material,
                input.v0,
            );
        })
        .$name(`${surface.name}ClipShadowFs`);
}

/** the clipped atlas fragment: preserve the tile-seam discard, then call the surface fs solely for its
 * cutoff discard. Its color result is intentionally ignored by this depth-only pipeline. */
function clipShadowFs(surface: AnySurface) {
    const needUv = !!surface.fragmentInputs?.uv;
    const needLocalPos = !!surface.fragmentInputs?.localPos;
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
                color: d.vec4f,
                material: d.interpolate("flat", d.vec4f),
                ...fragmentInterstage(surface),
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
                uv: needUv ? (input as any).uv : d.vec2f(0),
                localPos: needLocalPos ? (input as any).localPos : d.vec3f(0),
                color: input.color,
                material: input.material,
            });
            surface.fs(ctx);
        })
        .$name(`${surface.name}ClipShadowFs`);
}

/**
 * compile a `Surface`'s point + cascade shadow-atlas pipelines — `null` for a non-instanced or
 * `screen` surface (only an instanced, non-`screen` surface casts — a 2D overlay has no atlas placement).
 * Opaque surfaces share {@link shadowFs}; clipped surfaces use their wider cutoff
 * vertex/fragment pair. Each closes over its own `pointLayout` / `cascadeLayout` group-1 and its
 * own atlas pixel size, read when the app builds: `PointShadows.atlas` for the point atlas, the
 * per-cascade resolution × grid for the cascade atlas.
 */
function compileShadow(
    world: World,
    surface: AnySurface,
    capacity: number,
): {
    point: TgpuRenderPipeline<any> | null;
    cascade: TgpuRenderPipeline<any> | null;
} {
    if (!isInstanced(surface) || surface.screen) return { point: null, cascade: null };
    const primitive = surfacePrimitive(false);
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    // the receiver stub, as in `compilePrepass` — doubly load-bearing here: the real receiver
    // would sample the very atlas this pipeline renders into (a usage hazard)
    const root = world.gpu.root.with(pointShadowSlot, pointShadowStub);
    const clip = surface.blend === "clip";
    const varying = !!surface.varyings && Object.keys(surface.varyings).length > 0;
    const point = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingShadowVs(surface, pointLayout, pointAtlasSize(), false, capacity)
                    : clipShadowVs(surface, pointLayout, pointAtlasSize(), false, capacity)
                : shadowVs(surface, pointLayout, pointAtlasSize(), false, capacity),
            fragment: clip
                ? ((varying ? varyingShadowFs(surface) : clipShadowFs(surface)) as never)
                : shadowFs,
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`standard-point-${surface.name}`);
    const cascade = root
        .createRenderPipeline({
            vertex: clip
                ? varying
                    ? varyingShadowVs(
                          surface,
                          cascadeLayout,
                          cascadeAtlasSize(sunResolution(), sunCascades()),
                          true,
                          capacity,
                      )
                    : clipShadowVs(
                          surface,
                          cascadeLayout,
                          cascadeAtlasSize(sunResolution(), sunCascades()),
                          true,
                          capacity,
                      )
                : shadowVs(
                      surface,
                      cascadeLayout,
                      cascadeAtlasSize(sunResolution(), sunCascades()),
                      true,
                      capacity,
                  ),
            fragment: clip
                ? ((varying ? varyingShadowFs(surface) : clipShadowFs(surface)) as never)
                : shadowFs,
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`standard-cascade-${surface.name}`);
    return { point, cascade };
}

/** the compiled pipeline(s) for a `Surfaces` entry, or `undefined` until
 * {@link compileSurface} has run for it. */
export function getCompiledSurface(world: World, name: string): CompiledSurface | undefined {
    return pipelineState(world).compiledSurfaces.get(name);
}

// ---- the `Backgrounds` contract's pipeline builder (the Backgrounds bindings lock):
// the Surfaces contract minus mesh machinery, same group scheme. Group 0 = the shared `engineLayout`
// instance (the color pass's own); group 1 = `shadowLayout`, declared-but-unused via
// the `forcedZero` scope-forcing precedent (preserving `compileBackground`'s documented group-count-
// compatibility reason: a bg pipeline with only group 0 would drop group 1 for the blend draws that follow
// in the same pass); group 2 = the background's own bindings, through `contract.ts`'s `bgLayout` — the
// SAME `layoutEntry` synthesis `layout()` uses for a surface, minus the `vertices` injection (no mesh).

/** the widest `Background` shape (any bindings) — the bare `Background` default pins `B` to
 *  `Record<string, Binding>` already, so this alias exists only to name the widened form at call
 *  boundaries, matching {@link AnySurface}'s shape. */
type AnyBackground = Background<Record<string, Binding>>;

/**
 * the engine-owned fullscreen-triangle vertex entry every background shares — no per-background
 * variance (no mesh, no varyings, per the Backgrounds bindings lock), so ONE instance serves every
 * background's pipeline. The three corners come from `@builtin(vertex_index)` alone, emitted at the reverse-Z far plane (clip z = 0)
 * so {@link compileBackground}'s `depthCompare: "greater-equal"` + no-depth-write test admits only
 * un-rendered pixels.
 */
const bgVs = tgpu
    .vertexFn({ in: { vidx: d.builtin.vertexIndex }, out: { pos: d.builtin.position } })(
        (input) => {
            "use gpu";
            const c = d.vec2f(d.f32((input.vidx << 1) & 2), d.f32(input.vidx & 2));
            return { pos: d.vec4f(c.x * 2 - 1, c.y * 2 - 1, 0, 1) };
        },
    )
    .$name("bgVs");

/**
 * a background's fragment entry: reconstructs the normalized world-space view ray `dir` from
 * `@builtin(position)` + `engineLayout`'s `view.invViewProj`, not an interstage varying — forces
 * `shadowLayout`'s group-1 bindings into scope via the `forcedZero` fold (`colorFs`'s precedent, same
 * reason: `sampleSunShadow`/`pointShadowOf`'s free names are invisible to `tgpu.resolve`'s call-graph walk
 * otherwise), then calls the background's own `fs` chunk and wraps its `vec3f` result opaque (`vec4f(col, 1)`).
 */
function bgFs(bg: AnyBackground) {
    return tgpu
        .fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })((input) => {
            "use gpu";
            const uv = std.div(input.pos.xy, engineLayout.$.view.resolution);
            const ndc = d.vec3f(uv.x * 2 - 1, 1 - uv.y * 2, 0);
            const far = std.mul(engineLayout.$.view.invViewProj, d.vec4f(ndc, 1));
            const dir = std.normalize(
                std.sub(std.div(far.xyz, far.w), engineLayout.$.view.eye.xyz),
            );
            // see `colorFs`'s matching comment — the same forcing-touch precedent, folded into a
            // value the return genuinely uses so the transpiler can't prune it as dead
            const forcedZero =
                (shadowLayout.$.pointShadows.casters[0].pos.x +
                    shadowLayout.$.tileRects.rects[0].x +
                    shadowLayout.$.sunShadow.enabled +
                    std.textureSampleCompareLevel(
                        shadowLayout.$.pointAtlas,
                        shadowLayout.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    ) +
                    std.textureSampleCompareLevel(
                        shadowLayout.$.shadowMap,
                        shadowLayout.$.shadowSamp,
                        d.vec2f(0, 0),
                        0,
                    )) *
                0;
            const col = bg.fs(BackgroundContext({ dir }));
            return d.vec4f(std.add(col, d.vec3f(forcedZero)), 1);
        })
        .$name(`${bg.name}Fs`);
}

/** a compiled background: the 4× MSAA + single-sample twins (a camera binds whichever its
 *  `Camera.antialias` selects). */
export interface CompiledBackground {
    /** exact registry spec + layout this pipeline/group state was derived from. */
    owner: AnyBackground;
    layout: BackgroundLayout<Record<string, Binding>>;
    color: TgpuRenderPipeline<d.Vec4f>;
    single: TgpuRenderPipeline<d.Vec4f>;
    // the background's own group-2 bind group, built lazily on first draw (`renderColor`'s
    // backdrop pick) and cached on the resolved resource identities; null for a binding-free background
    // (its empty layout never enters the pipeline layout, so no group is bound at 2)
    group2: { group: GPUBindGroup; resources: BindResource[] } | null;
    // the background's engine group-0 instances per view slot (`engineGroup`'s caller-owned cache;
    // the bg's quant fill is the stable per-build `bgQuant()`, so this never sees buffer churn)
    engineCache: Map<number, GPUBindGroup>;
}

/**
 * compile one background's color pipelines — both the 4× MSAA + single-sample twins, eagerly
 * (`compileBackground`'s own reason: backgrounds are few, the camera's AA mode is known only at draw
 * time). `depthCompare: "greater-equal"` + no depth write: at clip z = 0 an un-rendered pixel (cleared
 * depth 0) passes `0 >= 0`, a geometry pixel (depth > 0) fails. Cached
 * by name plus exact source-spec/layout identity, so a same-name replacement cannot inherit pipelines
 * or layout-bound groups from its previous owner.
 */
export function compileBackground(world: World, bg: AnyBackground): CompiledBackground {
    const _render = world.resource(RenderContext);

    const cached = pipelineState(world).compiledBackgrounds.get(bg.name);
    if (cached?.owner === bg && cached.layout === bg.layout) return cached;
    const fragment = bgFs(bg);
    const primitive: GPUPrimitiveState = { topology: "triangle-list", cullMode: "none" };
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: false,
        depthCompare: "greater-equal",
    };
    const color = world.gpu.root
        .createRenderPipeline({
            vertex: bgVs,
            fragment,
            targets: { format: _render.format },
            primitive,
            depthStencil,
            multisample: { count: SAMPLE_COUNT },
        })
        .$name(`standard-bg-${bg.name}`);
    const single = world.gpu.root
        .createRenderPipeline({
            vertex: bgVs,
            fragment,
            targets: { format: _render.format },
            primitive,
            depthStencil,
            multisample: { count: 1 },
        })
        .$name(`standard-bg-${bg.name}-1x`);
    const compiled: CompiledBackground = {
        owner: bg,
        layout: bg.layout,
        color,
        single,
        group2: null,
        engineCache: new Map(),
    };
    pipelineState(world).compiledBackgrounds.set(bg.name, compiled);
    return compiled;
}

/** the compiled pipeline(s) for a `Backgrounds` entry, or `undefined` until
 * {@link compileBackground} has run for it. */
export function getBackground(
    world: World,
    name: string,
    bg?: AnyBackground,
): CompiledBackground | undefined {
    const compiled = pipelineState(world).compiledBackgrounds.get(name);
    if (compiled && bg && (compiled.owner !== bg || compiled.layout !== bg.layout)) {
        pipelineState(world).compiledBackgrounds.delete(name);
        return undefined;
    }
    return compiled;
}

/** Compile every surface and background at warm, before the first draw. */
export async function preparePipelines(world: World, capacity: number): Promise<void> {
    // force each pipeline's memo at warm (`root.unwrap` runs the resolve + the sync
    // `createRenderPipeline`) — typegpu defers both to first use, which would otherwise land mid-frame
    // on the first draw and hide a resolution/validation error until then (the force-compile-at-warm
    // lock)
    for (const surface of world.resource(Surfaces)) {
        const t = compileSurface(world, surface, capacity);
        for (const p of [t.color, t.transparent, t.point, t.cascade, ...t.prepass.values()]) {
            if (p) world.gpu.root.unwrap(p);
        }
    }
    for (const bg of world.resource(Backgrounds)) {
        const cb = compileBackground(world, bg);
        world.gpu.root.unwrap(cb.color);
        world.gpu.root.unwrap(cb.single);
    }
}
