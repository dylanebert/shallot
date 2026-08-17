// The renderer-neutral schema-backed surface/background contract. Layouts are created before their TGSL functions so
// shader code can close over `layout.$.name`; registration binds exact spec identity to a State lifetime.
//
// Group split: a typed surface's own bindings + the sear-injected `vertices` slot pin
// to **group 2** (engine 0 / shadow-or-atlas 1 / surface 2) — `surfaceLayout()`'s `$idx(SURFACE_GROUP)`. The
// `vertices` slot is pass-variant (color pass reads the 16 B main stream, prepass/shadow the 8 B
// position-only stream, same physical slot) — `surfaceLayout()` synthesizes both variants; typegpu resolves only
// what a given shader actually calls, so carrying the unused variant's declaration costs nothing (the
// "layout is a superset of what one shader declares" precedent, `engine.test.ts`).

import type { TgpuBindGroupLayout, TgpuFn } from "typegpu";
import tgpu, { isTgpuFn } from "typegpu";
import type { AnyWgslData, AnyWgslStruct, WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import { Registry, type State } from "../../engine";
import { Xform } from "../../engine/utils/core";

// Free functions (barrel-named — `layout`/`register` are too generic for a barrel, exports.md), not `Surfaces.layout`/`Surfaces.register` methods (the spec's literal wording):
// `Registry<T>` (`engine/utils/registry.ts`) is generic infra shared by `Draws`/`Meshes`, so it must stay
// free of typegpu types and sear's group scheme — a method on it would leak both into every registry
// consumer. `surfaceLayout`/`registerSurface` live here instead, against the plain surface registry.

/** typegpu's shader-stage literal, re-declared locally — `TgpuShaderStage` isn't re-exported from the
 *  top-level `typegpu` package (only from its internal `./types`). */
type ShaderStage = "compute" | "vertex" | "fragment";

/** the default visibility for an immutable surface resource. */
const VS_FS: ShaderStage[] = ["vertex", "fragment"];

/** group 2 (engine 0 / shadow-or-atlas 1 / surface 2) — where a typed surface's own
 *  bindings + the sear-injected `vertices` slot pin. */
export const SURFACE_GROUP = 2;

/**
 * the schema-carrying surface binding union. `uniform` takes the WGSL struct schema directly (`struct:
 * AnyWgslStruct`) and `storage` its element schema (`element: AnyWgslData`) — registry-kind resolution,
 * `mesh.bindings` overrides, and the `eids`+`transforms` instancing convention carry over unchanged;
 * only the payload each variant carries changes. Texture/sampler variants are
 * unchanged (no schema needed — `type` alone selects the WGSL type).
 */
export type Binding =
    | { type: "uniform"; struct: AnyWgslStruct }
    // `AnyWgslData`, not the wider `AnyData`: `AnyData` also admits loose vertex-format-only schemas that
    // can't back a storage declaration — the narrower type is deliberate, not an accident of the spec's wording
    | { type: "storage"; element: AnyWgslData; access?: "read" | "read_write" }
    | { type: "texture-2d" }
    | { type: "texture-2d-array" }
    | { type: "texture-depth-2d" }
    | { type: "sampler" }
    | { type: "sampler-comparison" };

/** the typed-layout entry a {@link Binding} converts to, discriminated the same way `Binding` itself is —
 *  so `SurfaceLayout<B>['$'][K]` recovers the precise per-binding value type (`layout.$.items` reads as
 *  an array of the declared element, not the widened `TgpuLayoutEntry` union every case would collapse
 *  to). Mirrors {@link layoutEntry}'s runtime shape exactly; keep the two in sync. */
// biome-ignore format: one row per Binding variant reads clearer un-wrapped
type EntryFor<B extends Binding> = B extends { type: "uniform" }
    ? { uniform: B["struct"]; visibility: ShaderStage[] }
    : B extends { type: "storage" }
      ? {
            // `d.arrayOf(element)` with no length is a runtime-sized-array *constructor* — this is the
            // unsized `array<T>` shape a storage binding needs
            storage: (elementCount: number) => WgslArray<B["element"]>;
            access: "mutable" | "readonly";
            visibility: ShaderStage[];
        }
      : B extends { type: "texture-2d" }
        ? { texture: d.WgslTexture2d<d.F32>; visibility: ShaderStage[] }
        : B extends { type: "texture-2d-array" }
          ? { texture: d.WgslTexture2dArray<d.F32>; visibility: ShaderStage[] }
          : B extends { type: "texture-depth-2d" }
            ? { texture: d.WgslTextureDepth2d; visibility: ShaderStage[] }
            : B extends { type: "sampler" }
              ? { sampler: "filtering"; visibility: ShaderStage[] }
              : { sampler: "comparison"; visibility: ShaderStage[] };

function layoutEntry<B extends Binding>(b: B): EntryFor<B> {
    switch (b.type) {
        case "uniform":
            return { uniform: b.struct, visibility: VS_FS } as EntryFor<B>;
        case "storage":
            return {
                // a storage binding is always `array<element>` — the typed contract keeps that
                // shape, so a bound name reads an array
                storage: d.arrayOf(b.element),
                access: b.access === "read_write" ? ("mutable" as const) : ("readonly" as const),
                // WebGPU forbids a read-write storage binding visible to the vertex stage — a surface
                // never runs compute, so a mutable binding narrows to fragment-only (typegpu's own
                // default for mutable storage is `["compute", "fragment"]`); a read-only binding stays
                // VS_FS like every other entry
                visibility: b.access === "read_write" ? (["fragment"] as ShaderStage[]) : VS_FS,
            } as EntryFor<B>;
        case "texture-2d":
            return { texture: d.texture2d(), visibility: VS_FS } as EntryFor<B>;
        case "texture-2d-array":
            return { texture: d.texture2dArray(), visibility: VS_FS } as EntryFor<B>;
        case "texture-depth-2d":
            return { texture: d.textureDepth2d(), visibility: VS_FS } as EntryFor<B>;
        case "sampler":
            return { sampler: "filtering" as const, visibility: VS_FS } as EntryFor<B>;
        case "sampler-comparison":
            return { sampler: "comparison" as const, visibility: VS_FS } as EntryFor<B>;
    }
}

/** the sear-injected `vertices` slot's two pass variants — the color pass's 16 B main stream
 *  (`array<vec4u>`: pos + meshId / oct normal / uv) and the prepass/shadow passes' 8 B position-only
 *  stream (`array<vec2u>`: pos + meshId). Same physical binding slot, distinct element type per pass —
 *  `render.md`'s `uniformWgsl(pass)` split, moved from the engine group into the surface group here. */
const verticesColor = {
    storage: d.arrayOf(d.vec4u),
    access: "readonly" as const,
    visibility: VS_FS,
};
const verticesDepth = {
    storage: d.arrayOf(d.vec2u),
    access: "readonly" as const,
    visibility: VS_FS,
};

/** a synthesized typed surface layout: group 2, a consumer's own bindings by name (`layout.$.name`) plus
 *  the sear-injected `vertices` slot (color-pass shape). {@link depthVariant} is the same bindings at the
 *  same `$idx`, `vertices` swapped to the prepass/shadow shape — the typed pipeline builder selects
 *  between them per pass, the same way `uniformWgsl(pass)` does today. */
export type SurfaceLayout<B extends Record<string, Binding>> = TgpuBindGroupLayout<
    { [K in keyof B]: EntryFor<B[K]> } & { vertices: typeof verticesColor }
> & {
    readonly depthVariant: TgpuBindGroupLayout<
        { [K in keyof B]: EntryFor<B[K]> } & { vertices: typeof verticesDepth }
    >;
};

/**
 * step one of the two-step typed registration: synthesize a surface's group-2 layout
 * from its own bindings, so a typed `vs`/`fs` can close over `layout.$.name` while it's being authored —
 * before {@link registerSurface} exists to call. Layouts are shareable across surfaces declaring the same
 * bindings (sprite ×6, gltf trios — register the same layout object on each).
 *
 * @example
 * const layout = surfaceLayout({ eids: { type: "storage", element: d.u32 }, transforms: { type: "storage", element: Xform } });
 * const fs = tgpu.fn([fsCtxSchema()], d.vec4f)((ctx) => layout.$.eids[ctx.eid] ? d.vec4f(1) : d.vec4f(0));
 */
/** the shared binding-map synthesis {@link surfaceLayout} and {@link backgroundLayout} both build on — a surface's own
 *  bindings mapped through {@link layoutEntry}, before either the `vertices` slot (surfaces) or nothing
 *  (backgrounds, which pull no mesh) is added. */
function ownEntries<B extends Record<string, Binding>>(
    bindings: B,
): { [K in keyof B]: EntryFor<B[K]> } {
    return Object.fromEntries(
        Object.entries(bindings).map(([name, b]) => [name, layoutEntry(b)]),
    ) as { [K in keyof B]: EntryFor<B[K]> };
}

export function surfaceLayout<B extends Record<string, Binding>>(bindings: B): SurfaceLayout<B> {
    const own = ownEntries(bindings);
    const color = tgpu.bindGroupLayout({ ...own, vertices: verticesColor }).$idx(SURFACE_GROUP);
    const depth = tgpu.bindGroupLayout({ ...own, vertices: verticesDepth }).$idx(SURFACE_GROUP);
    return Object.assign(color, { depthVariant: depth }) as SurfaceLayout<B>;
}

/** a synthesized typed background layout: group 2 (the same group a surface's own bindings pin to — the
 *  Backgrounds bindings lock), a background's own bindings by name (`layout.$.name`), through the SAME
 *  {@link layoutEntry} synthesis {@link surfaceLayout} uses — minus the `vertices` slot (a background pulls no
 *  mesh) and with no `depthVariant` (a background draws only in the color pass). */
export type BgLayout<B extends Record<string, Binding>> = TgpuBindGroupLayout<{
    [K in keyof B]: EntryFor<B[K]>;
}>;

/**
 * step one of a typed background's two-step registration, {@link surfaceLayout}'s twin for the Backgrounds seam:
 * synthesize a background's group-2 layout from its own
 * bindings, so its typed `fs` can close over `layout.$.name` while it's being authored.
 *
 * @example
 * const Tint = d.struct({ value: d.vec4f });
 * const layout = backgroundLayout({ tint: { type: "uniform", struct: Tint } });
 * const fs = tgpu.fn([BgCtx], d.vec3f)((ctx) => std.mul(layout.$.tint.value.xyz, ctx.dir));
 */
export function backgroundLayout<B extends Record<string, Binding>>(bindings: B): BgLayout<B> {
    return tgpu.bindGroupLayout(ownEntries(bindings)).$idx(SURFACE_GROUP) as BgLayout<B>;
}

/** what a `vs` chunk reads: the vertex-pull's pulled `localPos`/`localNormal`/`uv`, the
 *  `vidx`/`eid`/`iid` builtins, the resolved instance `xform`, and `world`/`worldNormal` — already carrying
 *  that transform when the surface is instanced (the `eids`+`transforms` convention). `xform` is identity for a
 *  non-instanced surface and lets a deforming vs replace local geometry without closing over one pass's
 *  concrete surface layout. */
export const VsIn = d
    .struct({
        localPos: d.vec3f,
        localNormal: d.vec3f,
        uv: d.vec2f,
        vidx: d.u32,
        eid: d.u32,
        iid: d.u32,
        xform: Xform,
        world: d.vec4f,
        worldNormal: d.vec3f,
    })
    .$name("VsIn");

/** a `vs` chunk's output schema, folding a surface's own `varyings` in beside the fixed patch fields
 *  (`world`/`worldNormal` override, `clip` for a `screen` surface's own projection) — sear builds this
 *  struct per surface rather than carrying one fixed shape, since the varying set is per-surface. */
export function vsPatchSchema<V extends Record<string, AnyWgslData> = Record<string, never>>(
    varyings: V = {} as V,
) {
    return d.struct({ world: d.vec4f, worldNormal: d.vec3f, clip: d.vec4f, ...varyings });
}

/** an `fs` chunk's input schema: the built-in fields sear rebinds as locals today (`eid`/`world`/
 *  `worldNormal`/`uv`/`localPos`) plus the surface's own `varyings` — the typed twin of
 *  `codegen.ts`'s `builtinFields`/`fragmentBody` rebind, built per surface for the same reason as
 *  {@link vsPatchSchema}. */
export function fsCtxSchema<V extends Record<string, AnyWgslData> = Record<string, never>>(
    varyings: V = {} as V,
) {
    return d.struct({
        eid: d.u32,
        world: d.vec3f,
        worldNormal: d.vec3f,
        uv: d.vec2f,
        localPos: d.vec3f,
        ...varyings,
    });
}

/** a surface's typed vertex chunk: `VsIn` in, its own `vsPatchSchema(varyings)` out. Optional — a surface
 *  with no `vs` leaves `world`/`worldNormal` at `VsIn`'s (the identity transform, or the instance
 *  transform's result). */
export type VsFn<V extends Record<string, AnyWgslData> = Record<string, never>> = TgpuFn<
    (vsIn: typeof VsIn) => ReturnType<typeof vsPatchSchema<V>>
>;

/** a surface's typed fragment chunk: its own `fsCtxSchema(varyings)` in, the shaded RGBA out. */
export type FsFn<V extends Record<string, AnyWgslData> = Record<string, never>> = TgpuFn<
    (ctx: ReturnType<typeof fsCtxSchema<V>>) => d.Vec4f
>;

/** a surface's optional typed id-lane hook: the same fragment context as {@link FsFn}, plus the
 * renderer's default (`eid` for an instanced surface, the no-surface sentinel otherwise), returning
 * the u32 written to `view.tag`. */
export type TagFn<V extends Record<string, AnyWgslData> = Record<string, never>> = TgpuFn<
    (ctx: ReturnType<typeof fsCtxSchema<V>>, defaultTag: d.U32) => d.U32
>;

/** compile-time pipeline specialization — a JS
 *  factory returning variant-folded fns (the outline `maskFragment` precedent: a captured JS boolean/number
 *  folds a branch at build time, no runtime cost), replacing gltf's string splicing. */
export type Specialize<V extends Record<string, AnyWgslData> = Record<string, never>> = (
    variant: number,
) => { vs?: VsFn<V>; fs: FsFn<V> };

/**
 * a surface authored against the typed contract: TGSL fns as the code (a synthesized `surfaceLayout()` is what
 * lets `vs`/`fs` close over `layout.$.name`, the accessor chicken-egg {@link surfaceLayout} solves). Structural
 * facts (`blend`/`screen`) route the renderer, they're not code.
 */
export interface Surface<
    B extends Record<string, Binding> = Record<string, Binding>,
    V extends Record<string, AnyWgslData> = Record<string, never>,
> {
    /** registry key referenced by a `Draw.surface`. */
    name: string;
    /** group-2 layout created before the shader functions so they can close over `layout.$`. */
    layout: SurfaceLayout<B>;
    /** optional mesh fields that must cross the rasterizer into {@link fs}. Undeclared fields remain
     * available to {@link vs} but are zero-filled in the fragment context instead of consuming an
     * interpolator. */
    fragmentInputs?: { uv?: true; localPos?: true };
    /** custom vertex-to-fragment fields; pass the same object to both IO schema factories. */
    varyings?: V;
    /** optional vertex patch over {@link VsIn}; omit to keep the engine-computed world values. */
    vs?: VsFn<V>;
    /** fragment shader returning linear HDR RGBA. */
    fs: FsFn<V>;
    /** optional id-lane shader evaluated by the tag prepass. It receives the fragment context and the
     * renderer's default tag, and replaces the color fragment function for that pass. */
    tag?: TagFn<V>;
    /** compile-time mesh-variant shader factory. */
    specialize?: Specialize<V>;
    /** alpha blends without depth writes; clip retains opaque depth/shadow routing and honors any
     * `discard` authored by {@link fs}. */
    blend?: "alpha" | "clip";
    /** lets {@link vs} author clip space directly instead of using the engine world projection. Requires
     * {@link vs}; registration without one fails when the variant compiles. */
    screen?: boolean;
}

/** every schema-backed surface, keyed by name with a stable renderer-owned id. */
export const Surfaces: Registry<Surface> = new Registry<Surface>();

/**
 * brand-check one incoming TGSL fn against this engine's own resolution of typegpu, at the seam where a
 * consumer-built object first meets the engine ({@link registerSurface}/{@link registerBackground}).
 * typegpu's brand markers are a per-copy `Symbol(...)` (never `Symbol.for`, `typegpu/shared/symbols.js`),
 * so `isTgpuFn`, imported from *this* engine's own resolution, reads `false` for a foreign copy's fn
 * even when its shape matches exactly (de-risked 2026-08-10, spec `shallot-peer-identity` stage 1: same
 * copy `true`, cross-copy `false`). This closes the ordering gap the module-load write-counter
 * (`checkTgsl`, `engine/runtime/gpu.ts`) can't: that counter can fold a duplicate's write into its own
 * baseline depending on which module evaluates first, but a foreign fn arriving here fails the brand
 * check regardless of evaluation order.
 */
export function assertOwnFn(label: string, fn: unknown): void {
    if (fn == null || isTgpuFn(fn)) return;
    throw new Error(
        `${label}: this isn't a TGSL function the engine can recognize. ` +
            "If it came from tgpu.fn, it resolved from a foreign copy of typegpu, not the one this " +
            "engine built against. Two physical copies in one bundle stamp different internal " +
            "markers even when the code is identical, so a kernel built from this fn resolves " +
            "against the wrong metadata map, or not at all. Known triggers: a bundler dedupe miss " +
            "(Vite prebundling) and pnpm's isolated node_modules. Dedupe typegpu to a single copy; " +
            "it is the engine's peerDependency for exactly this reason. " +
            "If it never came from tgpu.fn, build it with tgpu.fn(args, ret)(body).",
    );
}

function owned<T extends { name: string }>(registry: Registry<T>) {
    const byState = new WeakMap<State, Map<string, T>>();
    return (state: State, spec: T): number => {
        let entries = byState.get(state);
        if (!entries) {
            entries = new Map();
            byState.set(state, entries);
            state.onDispose(() => {
                for (const [name, owner] of entries!) {
                    if (registry.get(name) === owner) registry.delete(name);
                }
                byState.delete(state);
            });
        }
        entries.set(spec.name, spec);
        return registry.register(spec);
    };
}

const ownSurface = owned(Surfaces);

/**
 * register a surface for the lifetime of its owning State. Disposal removes it only while this exact
 * spec still owns the name, so a rebuilt State cannot delete its replacement.
 * @example registerSurface(state, { name: "tinted", layout, fs });
 */
export function registerSurface<
    B extends Record<string, Binding>,
    V extends Record<string, AnyWgslData>,
>(state: State, spec: Surface<B, V>): number {
    assertOwnFn(`registerSurface "${spec.name}" vs`, spec.vs);
    assertOwnFn(`registerSurface "${spec.name}" fs`, spec.fs);
    assertOwnFn(`registerSurface "${spec.name}" tag`, spec.tag);
    return ownSurface(state, spec as Surface);
}

// Background bindings use the same group-2 scheme, minus the mesh vertex slot and depth variant.

/** a background's fragment-input context: the engine-reconstructed normalized world-space view ray
 *  `dir` (from `@builtin(position)` + `view.invViewProj` at the reverse-Z far plane, not an interstage varying).
 *  Fixed shape, no per-background
 *  varyings — a background has no vertex stage of its own to write one from (the engine owns the one
 *  fullscreen-triangle `vertexFn`), so unlike {@link fsCtxSchema} this schema never folds anything in. */
export const BgCtx = d.struct({ dir: d.vec3f }).$name("BgCtx");

/** a background's TGSL fragment function: {@link BgCtx} in and HDR RGB out; Sear adds opaque alpha. */
export type BgFn = TgpuFn<(ctx: typeof BgCtx) => d.Vec3f>;

/**
 * a schema-backed background recipe: {@link backgroundLayout} declares the group-2 resources its TGSL
 * {@link BgFn} closes over. It has no mesh, vertex function, or interpolators; Sear owns the fullscreen
 * triangle and reconstructs {@link BgCtx.dir} per fragment.
 */
export interface Background<B extends Record<string, Binding> = Record<string, Binding>> {
    /** registry key referenced by a camera's `Backdrop.background`. */
    name: string;
    /** group-2 resources the fragment function closes over. */
    layout: BgLayout<B>;
    /** fragment shader returning linear HDR RGB for the reconstructed view ray. */
    fs: BgFn;
}

/** every registered background, keyed by name. */
export const Backgrounds: Registry<Background> = new Registry<Background>();

const ownBackground = owned(Backgrounds);

/**
 * register a background for the lifetime of its owning State.
 * @example registerBackground(state, { name: "sky", layout, fs });
 */
export function registerBackground<B extends Record<string, Binding>>(
    state: State,
    spec: Background<B>,
): number {
    assertOwnFn(`registerBackground "${spec.name}" fs`, spec.fs);
    return ownBackground(state, spec as Background);
}
