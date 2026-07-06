import { Registry } from "../../engine";

// the render-contract registries a producer registers into and a consumer
// renderer iterates: Surfaces (shading recipes) and Draws (draw calls). Meshes
// (geometry) lives in mesh.ts alongside its packing logic. Render defines these
// but never iterates them — they're the bridge between any producer and any
// renderer.

/**
 * surface binding spec. Each variant resolves to one WGSL declaration plus
 * one bind-group layout entry. Consumers assign binding indices in
 * registration order; the surface chunk references resources by name. Buffer
 * bindings (`uniform` / `storage`) resolve from `Compute.buffers`; texture
 * bindings (`texture-2d` / `texture-2d-array` / `texture-depth-2d`) from
 * `Compute.textures`; sampler bindings (`sampler` / `sampler-comparison`) from `Compute.samplers`.
 * All resolution happens at bind-group build time: slabs with a name
 * self-register their buffer; producers register their static buffers,
 * textures, and samplers explicitly
 */
export type Binding =
    | { type: "uniform"; struct: string }
    | { type: "storage"; element: string; access?: "read" | "read_write" }
    | { type: "texture-2d" }
    | { type: "texture-2d-array" }
    | { type: "texture-depth-2d" }
    | { type: "sampler" }
    | { type: "sampler-comparison" };

/**
 * renderer-agnostic shading recipe. `vs` and `fs` are WGSL chunks the
 * consumer renderer inlines into its `@vertex` / `@fragment` bodies; the
 * renderer provides vertex pull, struct definitions, lighting helpers, and
 * entry points: the surface's `vs` chunk owns any per-instance transform via
 * the bindings it declares, and the `fs` chunk writes the final color
 * (`col`), shading explicitly via the renderer's lighting helpers. `preamble`
 * is an optional WGSL chunk inlined at module scope after bindings, for helper
 * functions / structs / constants the entry chunks call. `interpolators` are
 * custom vertex→fragment varyings (name → WGSL type, e.g.
 * `{ litFactor: "vec3<f32>" }`): the renderer adds them to its output struct,
 * exposes them as mutable locals the `vs` chunk writes, and rebinds them as
 * locals the `fs` chunk reads: the path for per-vertex lighting, world-space
 * position, tangent frames, and the like. `bindings` enumerates every resource
 * the surface uses beyond Frame + View + Lighting; the renderer looks each
 * name up in the matching registry (`Compute.buffers` / `Compute.textures` /
 * `Compute.samplers`) by binding type. A surface is mesh-agnostic: geometry
 * is chosen per-{@link Draw} (`Draw.mesh`); for Part entities that's the
 * per-entity `Part.mesh` field, so one surface shades any mesh.
 *
 * `blend` is the one structural fact the renderer needs to route the surface: omitted =
 * opaque (color pass + the opt-in prepass id lane), `"alpha"` = non-opaque (sorted forward
 * blend, depth-tested but not depth-written, no prepass lanes), `"clip"` = masked-opaque cutout
 * (opaque — writes depth, has an id lane — but its empty-lane prepass fragment runs the chunk, so a
 * `discard` cuts holes the prepass depth and shadowmap both honor; the foliage / fence case that casts a
 * holed shadow). The trio maps to Unity's Transparent / Cutout / Opaque. It declares *what* the
 * surface is, not *how* the renderer composites it — `render/` and `part/` carry the field and
 * never learn what a pass is. A plain `discard` is orthogonal: an `"alpha"` surface that writes
 * alpha 1 and discards is a hard-edged *decal* (depth-tested, casts nothing); `"clip"` is the
 * variant that writes depth and casts.
 *
 * `screen` opts the surface into screen-space projection: the `vs` chunk writes the clip-space
 * position into `clipPos` itself (e.g. constant-pixel-width lines projecting their own endpoints),
 * and the renderer emits `out.clip = clipPos`. Omitted (the default) = world-space: the renderer
 * projects `view.viewProj * world` after the `vs` chunk, so a chunk that displaces `world` still
 * projects correctly. Like `blend`, it's a structural fact the renderer routes on — `render/` and
 * `part/` carry it without learning what projection means.
 */
export interface Surface {
    name: string;
    bindings?: Record<string, Binding>;
    preamble?: string;
    interpolators?: Record<string, string>;
    blend?: "alpha" | "clip";
    screen?: boolean;
    vs?: string;
    fs?: string;
    /**
     * compile-time pipeline specialization (Bevy's `specialize` / the `#ifdef USE_*MAP` idiom). When set,
     * the renderer compiles one pipeline per `variant` value a draw requests (the draw's `Mesh.variant`),
     * splicing the returned `preamble` / `fs` over the surface's defaults: on demand, deduped by variant,
     * so only the variants a scene actually loads are compiled. The `bindings` + `interpolators` are
     * variant-invariant (a variant samples fewer of the *same* bindings), so every variant shares one
     * bind-group layout. Omitted = one pipeline, variant ignored. The glTF importer specializes its
     * textured surfaces by material map-set so a draw samples only the maps its material carries.
     */
    specialize?: (variant: number) => { preamble?: string; fs?: string };
}

/** every registered surface, keyed by name with a stable numeric ID */
export const Surfaces: Registry<Surface> = new Registry<Surface>();

/**
 * draw arguments: always indirect. `indirect` is a GPU buffer holding the
 * standard 20-byte `DrawIndexedIndirect` record (`{ indexCount, instanceCount,
 * firstIndex, baseVertex, firstInstance }`) at `offset`; the renderer binds the
 * mesh's index buffer and issues `pass.drawIndexedIndirect(indirect, offset)`. Everything is GPU-driven, so the draw
 * count lives in GPU memory: a CPU-known draw just `writeBuffer`s its record
 * into a small buffer rather than passing literals.
 *
 * `viewStride`, when set, makes the record per-view: a producer that culls into
 * a separate `DrawIndexedIndirect` per camera lays the records out `slot`-major, and
 * the renderer reads `offset + view.slot * viewStride` so each camera draws its
 * own culled instances. Omit it (or `0`) for a view-independent draw
 */
export interface DrawArgs {
    indirect: GPUBuffer;
    offset?: number;
    viewStride?: number;
}

/**
 * one rendered thing. `surface` references a registered Surface by name;
 * `mesh` references a registered Mesh by name: the consumer renderer pulls
 * indexed vertices from that mesh's `vertices` + `indices` buffers in WGSL.
 * `args` points at the indirect draw record. Surface bindings beyond mesh
 * resolve by name against `Compute.buffers`
 */
export interface Draw {
    name: string;
    surface: string;
    mesh: string;
    args: DrawArgs;
}

/** every registered draw, keyed by name */
export const Draws: Registry<Draw> = new Registry<Draw>();
