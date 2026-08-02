// The typed `Surfaces`/`Backgrounds` contract (4a-ii-c-1): the schema-carrying `Binding` union, the
// two-step `surfaceLayout()`/`registerSurface()` registration that solves the accessor chicken-egg (a typed `vs`/`fs`
// closes over `layout.$.name`, so the layout must exist before the code does), and the TGSL-fn code
// contract types (`VsIn` in, a per-surface `VsPatch`/`FsCtx` out — both fold the surface's own
// `varyings` in, since sear builds each surface's IO struct dynamically rather than a single fixed shape).
//
// **No pipeline changes, no codegen rewrite, no built-in ports this stage** (4a-ii-c-1's boundary). Nothing
// here is wired into `codegen.ts` / `pipelines.ts` / `forward.ts` — the raw string contract (`Surface` in
// `render/core`, spliced by `surfaceCode`) is untouched and stays the only path a real pipeline compiles
// against until c-2 teaches the pipeline builder to branch on the two shapes. `registerSurface()`'s dual-accept
// shim is the seam that makes that landing safe: a legacy-shaped spec still lands in the real `Surfaces`
// registry (unchanged behavior), a typed-shaped spec lands in a separate typed registry c-2 wires in.
//
// Group split (4a-ii design lock): a typed surface's own bindings + the sear-injected `vertices` slot pin
// to **group 2** (engine 0 / shadow-or-atlas 1 / surface 2) — `surfaceLayout()`'s `$idx(SURFACE_GROUP)`. The
// `vertices` slot is pass-variant (color pass reads the 16 B main stream, prepass/shadow the 8 B
// position-only stream, same physical slot) — `surfaceLayout()` synthesizes both variants; typegpu resolves only
// what a given shader actually calls, so carrying the unused variant's declaration costs nothing (the 2b
// "layout is a superset of what one shader declares" precedent, `engine.test.ts`).

import type { TgpuBindGroupLayout, TgpuFn } from "typegpu";
import tgpu from "typegpu";
import type { AnyWgslData, AnyWgslStruct, WgslArray } from "typegpu/data";
import * as d from "typegpu/data";
import { Registry, type State } from "../../engine";
import { Xform } from "../../engine/utils/core";
import type { Surface as LegacySurface } from "../render/core";
import { Surfaces } from "../render/core";

// Free functions (barrel-named — `layout`/`register` are too generic for a barrel, exports.md), not `Surfaces.layout`/`Surfaces.register` methods (the spec's literal wording):
// `Registry<T>` (`engine/utils/registry.ts`) is generic infra shared by `Draws`/`Meshes`, so it must stay
// free of typegpu types and sear's group scheme — a method on it would leak both into every registry
// consumer. `surfaceLayout`/`registerSurface` live here instead, against the plain `TypedSurfaces: Registry<TypedSurface>`.

/** typegpu's shader-stage literal, re-declared locally — `TgpuShaderStage` isn't re-exported from the
 *  top-level `typegpu` package (only from its internal `./types`). */
type ShaderStage = "compute" | "vertex" | "fragment";

/** every typed binding is visible to both stages — a surface's own bindings are always VS_FS, matching
 *  the legacy contract's `bindingEntry` (codegen.ts). */
const VS_FS: ShaderStage[] = ["vertex", "fragment"];

/** group 2 (4a-ii design lock: engine 0 / shadow-or-atlas 1 / surface 2) — where a typed surface's own
 *  bindings + the sear-injected `vertices` slot pin. */
export const SURFACE_GROUP = 2;

/**
 * the typed contract's binding union — the schema-carrying twin of `render/core`'s string-carrying
 * `Binding` (dual-accepted at {@link registerSurface} during migration; codegen.ts's legacy splicing keeps
 * reading the string form untouched). `uniform` takes the WGSL struct schema directly (`struct:
 * AnyWgslStruct`) and `storage` its element schema (`element: AnyWgslData`) — registry-kind resolution,
 * `mesh.bindings` overrides, and the `eids`+`transforms` instancing convention all carry over unchanged
 * from the legacy contract; only the payload each variant carries changes. Texture/sampler variants are
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
            // unsized `array<T>` shape a storage binding needs (the legacy contract's `array<${element}>`)
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
                // the legacy contract's storage binding is always `array<element>` (`bindingDecl`,
                // codegen.ts) — the typed contract keeps that shape, so a bound name reads an array
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
 *  same `$idx`, `vertices` swapped to the prepass/shadow shape — the typed pipeline builder (c-2) selects
 *  between them per pass, the same way `uniformWgsl(pass)` does today. */
export type SurfaceLayout<B extends Record<string, Binding>> = TgpuBindGroupLayout<
    { [K in keyof B]: EntryFor<B[K]> } & { vertices: typeof verticesColor }
> & {
    readonly depthVariant: TgpuBindGroupLayout<
        { [K in keyof B]: EntryFor<B[K]> } & { vertices: typeof verticesDepth }
    >;
};

/**
 * step one of the two-step typed registration (4a-ii design lock): synthesize a surface's group-2 layout
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
 * step one of a typed background's two-step registration, {@link surfaceLayout}'s twin for the Backgrounds seam
 * (the Backgrounds bindings lock, 4a-ii-c-3a-4): synthesize a background's group-2 layout from its own
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
 *  that transform when the surface is instanced (the `eids`+`transforms` convention), matching the legacy
 *  contract's `INSTANCE_VS` running before the surface's own vs chunk today. `xform` is identity for a
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

/** a surface's typed fragment chunk: its own `fsCtxSchema(varyings)` in, the shaded RGBA out (prepass
 *  lanes — `tag`, … — stay engine-computed, never authored here). */
export type FsFn<V extends Record<string, AnyWgslData> = Record<string, never>> = TgpuFn<
    (ctx: ReturnType<typeof fsCtxSchema<V>>) => d.Vec4f
>;

/** compile-time pipeline specialization, the typed twin of the legacy `Surface.specialize` — a JS
 *  factory returning variant-folded fns (the outline `maskFragment` precedent: a captured JS boolean/number
 *  folds a branch at build time, no runtime cost), replacing gltf's string splicing. */
export type Specialize<V extends Record<string, AnyWgslData> = Record<string, never>> = (
    variant: number,
) => { vs?: VsFn<V>; fs: FsFn<V> };

/**
 * a surface authored against the typed contract: TGSL fns as the code (a synthesized `surfaceLayout()` is what
 * lets `vs`/`fs` close over `layout.$.name`, the accessor chicken-egg {@link surfaceLayout} solves). Structural
 * facts (`blend`/`screen`) carry over unchanged from the legacy `Surface` — they route the renderer,
 * they're not code.
 */
export interface TypedSurface<
    B extends Record<string, Binding> = Record<string, Binding>,
    V extends Record<string, AnyWgslData> = Record<string, never>,
> {
    name: string;
    layout: SurfaceLayout<B>;
    varyings?: V;
    vs?: VsFn<V>;
    fs: FsFn<V>;
    specialize?: Specialize<V>;
    blend?: "alpha" | "clip";
    screen?: boolean;
}

/** every registered typed surface, keyed by name — separate from `render/core`'s `Surfaces` registry
 *  until 4a-ii-d deletes the string path (a `TypedSurface` can't satisfy the legacy `Surface` shape
 *  `compileVariant` reads: `vs`/`fs` are TGSL fns, not WGSL strings). `record()` consults this registry
 *  FIRST, so a name registered in both draws typed (the built-in flip); Part's automatic
 *  per-`(surface, mesh)` draw registration still reads `render/core`'s `Surfaces`, so a typed-only
 *  surface gets no automatic Part draws until 4a-ii-d. */
export const TypedSurfaces: Registry<TypedSurface> = new Registry<TypedSurface>();

function isTyped(spec: LegacySurface | TypedSurface): spec is TypedSurface {
    // `"layout" in spec` alone misroutes a legacy spec carrying an incidental `layout` key (the string
    // contract has no reserved meaning for that name) into the typed registry, where it silently draws
    // nothing — the `fs` shape is the second, load-bearing discriminant: legacy `fs` is always a WGSL
    // string, typed `fs` is always a TgpuFn.
    return "layout" in spec && typeof spec.fs !== "string";
}

/**
 * the dual-accept shim (4a-ii design lock, the 3b-iv two-copy precedent): accepts either the legacy
 * string-contract `Surface` (`render/core`) or a typed spec, discriminated by `layout` presence plus
 * `fs`'s shape (a string is always legacy, a TgpuFn always typed) — `layout` alone isn't safe, since a
 * legacy spec carrying an incidental `layout` key would otherwise misroute. A legacy spec
 * delegates straight to the real `Surfaces.register` (unchanged behavior, every existing gate stays
 * green); a typed spec lands in {@link TypedSurfaces} for c-2 to compile. Supplying its owning State ties
 * that exact registration to teardown; the one-argument compatibility arm stays caller-owned until
 * 4a-ii-d deletes the string path.
 *
 * @example
 * registerSurface({ name: "checker", bindings: { color: { type: "uniform", struct: "vec4<f32>" } }, fs: "col = color;" }); // legacy
 * registerSurface(state, { name: "typed-checker", layout, fs }); // typed
 */
export function registerSurface<
    B extends Record<string, Binding>,
    V extends Record<string, AnyWgslData>,
>(state: State, spec: TypedSurface<B, V>): number;
export function registerSurface<
    B extends Record<string, Binding>,
    V extends Record<string, AnyWgslData>,
>(spec: TypedSurface<B, V>): number;
export function registerSurface(spec: LegacySurface): number;
export function registerSurface(
    stateOrSpec: State | LegacySurface | TypedSurface,
    typed?: TypedSurface,
): number {
    if (typed) {
        const id = TypedSurfaces.register(typed);
        const state = stateOrSpec as State;
        state.onDispose(() => {
            if (TypedSurfaces.get(typed.name) === typed) TypedSurfaces.delete(typed.name);
        });
        return id;
    }
    const spec = stateOrSpec as LegacySurface | TypedSurface;
    if (isTyped(spec)) return TypedSurfaces.register(spec);
    return Surfaces.register(spec);
}

// ---- the typed Backgrounds contract (4a-ii-c-3a-4, the Backgrounds bindings lock) — additive only: no
// pipeline wiring, no built-in conversion. The dual-accept `registerBackground` shim lives in `forward.ts`
// beside the legacy `Backgrounds` registry (this module can't import it — `forward.ts` already imports
// this one, and a background's own bindings pin to the SAME group-2 layout the surface's own bindings do
// (`bgLayout`, above), minus the `vertices` slot (no mesh) and with no `depthVariant` (color-pass only).

/** a typed background's fragment-input context: the engine-reconstructed normalized world-space view ray
 *  `dir` (from `@builtin(position)` + `view.invViewProj` at the reverse-Z far plane — `codegen.ts`'s
 *  `backgroundCode` reconstruct, not an interstage varying — gpu.md rule 9). Fixed shape, no per-background
 *  varyings — a background has no vertex stage of its own to write one from (the engine owns the one
 *  fullscreen-triangle `vertexFn`), so unlike {@link fsCtxSchema} this schema never folds anything in. */
export const BgCtx = d.struct({ dir: d.vec3f }).$name("BgCtx");

/** a typed background's fragment chunk: {@link BgCtx} in, the shaded HDR color out (the engine wraps it
 *  opaque — `vec4f(col, 1)` — the same as the raw `backgroundCode`'s `col: vec3<f32>` contract). */
export type BgFn = TgpuFn<(ctx: typeof BgCtx) => d.Vec3f>;

/**
 * a background authored against the typed contract: the Surfaces contract minus mesh machinery (Approach
 * 4a's Backgrounds bindings lock). `layout` is {@link backgroundLayout}'s synthesis — solves the same
 * accessor-chicken-egg {@link surfaceLayout} does for a surface's `vs`/`fs`.
 */
export interface TypedBackground<B extends Record<string, Binding> = Record<string, Binding>> {
    name: string;
    layout: BgLayout<B>;
    fs: BgFn;
}

/** every registered typed background, keyed by name — separate from `forward.ts`'s `Backgrounds`
 *  registry until 4a-ii-d; the `Backdrop` trait stores a typed entry's id offset by the typed base, and
 *  `renderColor`'s backdrop pick draws it through `compileTypedBg`'s pipelines. */
export const TypedBackgrounds: Registry<TypedBackground> = new Registry<TypedBackground>();
