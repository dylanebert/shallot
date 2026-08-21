// Stage 5 — posts (`roads-interactive.md` stage 5). A row of abstract posts along the road, positioned
// entirely on the GPU: a compute kernel writes one `Post` record per slot (station along the chord,
// lateral offset alternating sides, `y` from `flatten.ts`'s own `flattenedHeightAt` TGSL fn, scale 0 for
// slots past the chord), and a custom surface's VS maps the standard `capsule` mesh onto the post's own
// dimensions and translates by the record. The fountain showcase's shape: a typed record buffer written
// by compute, read by a custom surface's VS at `input.iid`, one shared mesh, `Draws.register` with a
// fixed `instanceCount`. The records never touch Part or the Transform slabs.
//
// `POST_COUNT` is sized off the world's own diagonal (`WORLD_EXTENT · √2` ≈ 1448 m, since
// `ROAD_MAX_LENGTH` was deleted in stage 4c) divided by `POST_SPACING`, so `instanceCount` stays fixed
// across edits and short chords leave most slots at scale 0. Re-dispatched after every `setNetwork` (edit
// and reseed), never per frame — `terrain.ts`'s `warm`, `regenerate`, and `editDocument` each call
// `dispatchPosts(seed)` after `setNetwork`.
//
// Stage 11 re-derived every constant off **one referent** (see the constant block's header) and rewrote
// the VS's mesh mapping as a core/cap decomposition with a buried base (see {@link postVertexOffset}).
// The kernel and the placement record are untouched by that pass, so the placement arms are its null
// control.

import { Compute, type State } from "@dylanebert/shallot";
import {
    type Draw,
    DrawIndexedIndirect,
    type DrawIndirectBuffer,
    Draws,
    Meshes,
} from "@dylanebert/shallot/render/core";
import { precompile, precompileScope } from "@dylanebert/shallot/runtime";
import {
    fsCtxSchema,
    lit,
    registerSurface,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "@dylanebert/shallot/sear/core";
import tgpu, { type StorageFlag, type TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { flattenFieldAt } from "./flatness";
import type { Check } from "./harness";
import { ROAD_MIN_LENGTH } from "./overlay/network";
import {
    buildNetworkGeometry,
    computeFalloff,
    flattenedHeightAt,
    networkBindGroup,
    networkLayout,
} from "./terrain/flatten";
import { FLAT_CORE_MARGIN } from "./terrain/flatten-math";
import { WORLD_EXTENT } from "./terrain/grid";
import { makePermutation, noiseLayout, PermData, RELIEF } from "./terrain/noise";
import { heightAtCpu } from "./terrain/profile";
import { getCurrentSeed, getDocument } from "./terrain/terrain";

// --- constants --------------------------------------------------------------
//
// THE REFERENT (stage 11, `roads-interactive.md`'s Locked decision): **one kerbside pipe bollard**, the
// galvanized/painted steel pipe set in a footing at the kerb line of an urban street edge — a single
// object, and every `POST_*` constant below is read off that one object rather than off a per-dimension
// citation. Stage 10 cited a pipe OD for the radius, a foundry catalogue for the height and NACTO
// channelization for the spacing: three numbers from three different objects, which is what left the row
// still reading wrong (spec: "a decorative element's referent is a single thing in the world, and every
// constant describing it is read off that one thing").
//
// The referent's own dimensions, as one object (`[snippet]`-grade, 1800bollards.com + the OSHA colour-use
// interpretation, read 2026-08-22, recorded in the spec's Locked decision): ~1 m tall above grade, set at
// the **kerb line** rather than out in the grass, **1.5–2.0 m** apart along the kerb, finished in RAL 1023
// traffic yellow (the high-visibility default), RAL 9005 jet black, or bare galvanized grey. Its shaft is a
// steel pipe — 8–10 in Schedule 40, OD 219–273 mm — closed with a domed cap and **embedded in a footing**,
// so it meets the pavement at a line, never at a visible bulge.
//
// What that one object fixes, constant by constant: `POST_HEIGHT` (~1 m above grade), `POST_RADIUS`
// (the pipe's own OD/2), `POST_SPACING` (the kerb row's 2.0 m), `POST_OFFSET` (the kerb line, ~0.4 m off
// the pavement edge), `POST_COLOR` (RAL 1023), and `POST_BURIAL_DEPTH` (the footing — derived below from
// the grade the road carries, not chosen by eye).

/** metres between posts along the chord — the kerb row's own spacing.
 *
 *  Held at 2 m by stage 11 rather than moved: 2.0 m is the **upper bound of the referent's own 1.5–2.0 m
 *  kerbside range**, so the value stage 10 shipped is already read off this object and needs no change.
 *  Below 1.5 m the row reads as a barrier rather than a kerb line; above 2.0 m it stops reading as a row. */
export const POST_SPACING = 2;

/** the lateral offset from the road edge (pavement edge) to the post centre, in metres — the **kerb line**.
 *
 *  Referent value: a kerbside bollard stands at the kerb, i.e. immediately off the pavement edge, not out
 *  in the verge. ~0.4 m from the pavement edge puts the post's near face ~0.28 m clear of the asphalt —
 *  a kerb-and-gutter's own width, which is what "at the kerb line" means in this scene, where the road is
 *  a bare 8 m strip with no modelled kerb. Stage 5's `POST_OFFSET = SPACING = 4 m` (13 ft) was the flat
 *  core band's convenience and never a fact about bollards — it is why the row read as posts standing in
 *  a field.
 *
 *  Admissibility, re-measured at stage 11's claim against `flatten-math.ts` rather than assumed safer for
 *  being smaller (the *lower* end of the band is the one 0.4 m approaches). The flat core is the region
 *  `coreDist <= 0` (`flatten.ts`'s `networkCore`), i.e. perpendicular distance from the centreline up to
 *  `halfWidth + FLAT_CORE_MARGIN`, where `FLAT_CORE_MARGIN = √2 · SPACING = √2 · 4 = 5.65685 m`. Inside
 *  it the flatten field is exactly affine, so the rendered mesh reproduces it exactly and a post's foot
 *  meets the rendered surface — the property the placement oracle reads. A post at `halfWidth +
 *  POST_OFFSET` occupies the band `[halfWidth + POST_OFFSET − POST_RADIUS, halfWidth + POST_OFFSET +
 *  POST_RADIUS]`, so the whole footing is strictly inside the flat core and strictly off the pavement iff
 *
 *      POST_RADIUS < POST_OFFSET < FLAT_CORE_MARGIN − POST_RADIUS
 *      0.12 m      < 0.4 m       < 5.53685 m
 *
 *  Measured margins at 0.4 m: **0.28 m** of clearance from the pavement edge to the post's near face and
 *  **5.13685 m** from its far face to the flat core's outer edge. 0.4 m is admissible, so it stands as
 *  the referent's own value; no smallest-admissible fallback was needed. (The band's earlier statement,
 *  `0 < POST_OFFSET < FLAT_CORE_MARGIN`, ignored the post's own radius — correct for the record's point,
 *  too loose for the footing.) */
export const POST_OFFSET = 0.4;

/** the post's **above-grade** height in metres — the referent's own quantity ("36 in above grade"), and
 *  the burial depth below is added *below* it rather than taken out of it, so the height a person sees is
 *  this number exactly.
 *
 *  Referent value: the kerbside pipe bollard stands ~1 m above grade (the R-7181 pipe bollard's 36 in =
 *  0.914 m is the same object's catalogue instance). Held at 1 m by stage 11. */
export const POST_HEIGHT = 1;

/** the post's radius in metres — the referent's pipe OD / 2, and the radius of **both** spherical caps.
 *
 *  Referent value: the same kerbside pipe bollard is 8–10 in Schedule 40 steel pipe, OD 219 mm (r =
 *  0.110 m) to 273 mm (r = 0.137 m); 0.12 m (OD 240 mm) sits between the two. Held at 0.12 m by stage 11. */
export const POST_RADIUS = 0.12;

/** the bollard's finish colour, in the same 0–1 linear-albedo convention as `ROAD_ALBEDO`/`EDGE_ALBEDO`
 *  (unconverted).
 *
 *  Referent value: **RAL 1023 traffic yellow** ≈ `[0.941, 0.792, 0.0]` — the high-visibility default
 *  finish for a kerbside safety bollard. The referent's other two catalogue finishes are the one-constant
 *  swap if the look prefers them: **RAL 9005 jet black** ≈ `[0.039, 0.039, 0.039]` and **bare galvanized
 *  grey** ≈ `[0.66, 0.67, 0.68]`. Stage 5's `[0.5, 0.4, 0.3]` was an unexplained brown triple with no
 *  referent at all, and it is half of why the row read as fence posts. */
export const POST_COLOR: readonly [number, number, number] = [0.941, 0.792, 0.0];

/** the steepest grade (rise/run) the flatten field can carry *along* a chord the drag admits — the input
 *  the footing depth is derived from.
 *
 *  The chord's target height is linear between its endpoints' natural heights (`buildPolylineProfile`), so
 *  the field's along-chord grade is exactly `|Δh| / chordLength`. `heightAtCpu` is `GROUND_LEVEL + fbm2 ·
 *  RELIEF` with `|fbm2| ≤ 1`, so `|Δh| ≤ 2 · RELIEF = 80 m`, and the drag clamps the chord at
 *  `ROAD_MIN_LENGTH = 80 m` — hence a ceiling of exactly 1.0. This is an analytic ceiling over the whole
 *  admissible domain, not a fitted number; the *measured* worst case over a 5-seed × 400-chord scan of
 *  admissible chords (half of them drawn short, since grade is |Δh| / length) is **0.1508** — `posts.test.ts`
 *  asserts the scan stays under this ceiling and that it reaches grades worth burying. So the ceiling is
 *  ~6.6× the measurement, and the footing is sized against the domain rather than against a sample. */
export const MAX_CHORD_GRADE = (2 * RELIEF) / ROAD_MIN_LENGTH;

/** the footing: how far below the surface the shaft's base sits, in metres — so the shaft meets the
 *  pavement at a **line** and the bottom cap is entirely underground, which is what a bollard set in a
 *  footing looks like and what stage 5's shipped mapping got wrong (its lowest mesh point sat *on* the
 *  surface, leaving a 0.25 m dome bulging under the shaft).
 *
 *  Derived, not eyeballed. The flatten field is exactly flat *laterally* inside the flat core, so the only
 *  rise across the post's own footprint is the road's along-chord grade. The base ring's uphill side sits
 *  `grade · POST_RADIUS` above the field height sampled at the post's centre, so burying the ring needs
 *  `POST_BURIAL_DEPTH ≥ MAX_CHORD_GRADE · POST_RADIUS = 0.12 m`. Taking the rise across the post's whole
 *  **diameter** instead of its radius gives exactly 2× that requirement — the margin is that factor of
 *  two, stated rather than added as an epsilon:
 *
 *      POST_BURIAL_DEPTH = MAX_CHORD_GRADE · 2 · POST_RADIUS = 1.0 · 0.24 = 0.24 m
 *
 *  It costs nothing visible: the depth is added *below* `POST_HEIGHT` (the shaft grows downward), so the
 *  above-grade extent stays at the referent's 1 m exactly. */
export const POST_BURIAL_DEPTH = MAX_CHORD_GRADE * 2 * POST_RADIUS;

/** the cylindrical shaft's length in metres — a derived quantity, not a referent one: the shaft spans from
 *  the footing's base (`POST_BURIAL_DEPTH` below the surface) to the underside of the `POST_RADIUS` dome,
 *  and the dome's top is the referent's above-grade height. So
 *  `POST_SHAFT_LENGTH = POST_HEIGHT + POST_BURIAL_DEPTH − POST_RADIUS` = 1 + 0.24 − 0.12 = 1.12 m. */
export const POST_SHAFT_LENGTH = POST_HEIGHT + POST_BURIAL_DEPTH - POST_RADIUS;

/** the world's own diagonal — the maximum chord length the world contains (`WORLD_EXTENT · √2` ≈ 1448 m),
 *  since `ROAD_MAX_LENGTH` was deleted in stage 4c. This is the ceiling `POST_COUNT` is sized against. */
const WORLD_DIAGONAL = WORLD_EXTENT * Math.SQRT2;

/** the fixed slot count — sized off the world's own diagonal divided by `POST_SPACING`, so `instanceCount`
 *  stays fixed across edits and short chords leave most slots at scale 0. At `POST_SPACING = 2` this is
 *  ≈725 slots (up from ~73 at 20 m spacing); the instanced draw handles this trivially — 725 instances of
 *  a capsule mesh is well within WebGPU's instancing limits, and the record buffer is 725 × 16 B ≈ 11 KB. */
export const POST_COUNT = Math.ceil(WORLD_DIAGONAL / POST_SPACING);

const POST_WORKGROUP = 64;
const POST_DISPATCH = Math.ceil(POST_COUNT / POST_WORKGROUP);

// --- pure derivation functions (device-free, for the test seam and the gate) -----

/** the station (metres along the chord from endpoint A) of slot `i`. The first post sits at
 *  `POST_SPACING` (not station 0, the endpoint), so the live-slot count is
 *  `floor(chordLength / POST_SPACING)` — matching Validation's assertion, not `floor(...) + 1`.
 *  (Open question 2 — reported to the spec owner for correction: Approach wrote `i · POST_SPACING`,
 *  which puts a post at station 0 and makes the count `floor(...) + 1`.) */
export function postStation(i: number): number {
    return (i + 1) * POST_SPACING;
}

/** the lateral offset sign for slot `i`: +1 for even slots, −1 for odd — alternating sides of the road. */
export function postLateralSign(i: number): number {
    return i % 2 === 0 ? 1 : -1;
}

/** the number of live (non-zero-scale) slots for a chord of `chordLength` metres:
 *  `floor(chordLength / POST_SPACING)`. */
export function liveSlotCount(chordLength: number): number {
    return Math.floor(chordLength / POST_SPACING);
}

/** whether slot `i` is live (scale ≠ 0) for a chord of `chordLength` metres: the station is within the
 *  chord (`station <= chordLength`). Slots past the chord are scale 0. */
export function isLiveSlot(i: number, chordLength: number): boolean {
    return postStation(i) <= chordLength;
}

// --- Post record + layouts ---------------------------------------------------

/** the compute kernel's output / the surface VS's input: world position (x, y, z) + scale in `w`
 *  (0 = hidden, 1 = visible). The fountain's `Particle` shape — a typed record buffer written by
 *  compute, read by a custom surface's VS at `input.iid`. */
const Post = d.struct({ pos: d.vec4f }).$name("Post");
const PostArray = d.arrayOf(Post, POST_COUNT);
const POST_BYTES = d.sizeOf(PostArray);

/** the compute kernel's bind group layout — one mutable storage buffer of `POST_COUNT` `Post` records. */
const postsComputeLayout = tgpu.bindGroupLayout({
    posts: { storage: PostArray, access: "mutable" },
});

/** the surface's bind group layout — the VS reads the same buffer by instance id. The binding name
 *  matches the `Compute.buffers`/`Compute.typed` key the renderer resolves by name (fountain's shape). */
const postsSurfaceLayout = surfaceLayout({
    posts: { type: "storage", element: Post },
});

// --- compute kernel ---------------------------------------------------------

/** one thread per slot: read the chord from `networkLayout`'s segment 0, compute station and lateral
 *  offset, get `y` from `flattenedHeightAt` (the height kernel's own TGSL fn, bound through the same
 *  network bind group — never re-derived on the CPU for this consumer), and write scale 0 for slots past
 *  the chord's length. */
const postsKernel = tgpu
    .computeFn({
        workgroupSize: [POST_WORKGROUP],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const i = input.gid.x;
        if (i >= d.u32(POST_COUNT)) return;

        const seg = networkLayout.$.segments[0];
        const ax = seg.a.x;
        const az = seg.a.y;
        const bx = seg.b.x;
        const bz = seg.b.y;
        const halfWidth = seg.halfWidth;

        const abx = bx - ax;
        const abz = bz - az;
        const chordLength = std.sqrt(abx * abx + abz * abz);

        const station = (d.f32(i) + d.f32(1)) * d.f32(POST_SPACING);

        if (station > chordLength) {
            postsComputeLayout.$.posts[i] = Post({ pos: d.vec4f(0, 0, 0, 0) });
            return;
        }

        const ux = abx / chordLength;
        const uz = abz / chordLength;
        const nx = -uz;
        const nz = ux;

        const cx = ax + ux * station;
        const cz = az + uz * station;

        const sign = std.select(d.f32(1), d.f32(-1), (i & d.u32(1)) === d.u32(1));
        const lateral = (halfWidth + d.f32(POST_OFFSET)) * sign;

        const px = cx + nx * lateral;
        const pz = cz + nz * lateral;
        const py = flattenedHeightAt(px, pz);

        postsComputeLayout.$.posts[i] = Post({ pos: d.vec4f(px, py, pz, d.f32(1)) });
    })
    .$name("posts-kernel");

/** the emitted posts WGSL — the device-free structural seam `posts.test.ts` resolves. Contains
 *  `flattenedHeightAt` (the named oracle, Validation's "the resolved posts WGSL contains
 *  `flattenedHeightAt`"). */
export function postsWgsl(): string {
    return tgpu.resolve([flattenedHeightAt, postsKernel], { names: "strict" });
}

/** the emitted posts **surface** WGSL — the VS's core/cap decomposition and the FS's finish colour
 *  (stage 11). A second seam from {@link postsWgsl} because the surface functions resolve on their own
 *  schemas, not through the compute entry point. */
export function postsSurfaceWgsl(): string {
    return tgpu.resolve([postsVs, postsFs], { names: "strict" });
}

// --- surface (VS + FS) -------------------------------------------------------

const postsPatch = vsPatchSchema({});

/** the core/cap decomposition of the standard `capsule` mesh (stage 11), and the CPU twin of exactly the
 *  arithmetic {@link postsVs} performs — `posts.test.ts`'s mesh arms read this rather than a
 *  re-derivation, and the VS's own structural arm pins the emitted WGSL beside it.
 *
 *  The mesh (`part/mesh.ts`'s `capsule()`) is `radius = 0.5`, `halfHeight = 0.5`: a cylinder section over
 *  `y ∈ [−0.5, 0.5]` with a hemisphere cap beyond each end, total extent `[−1, 1]`. Stage 5 scaled the
 *  whole thing anisotropically — y by `POST_HEIGHT/2`, x/z by `POST_RADIUS · 2` — which stretched both
 *  caps into ellipsoids taller than they are wide (a bullet nose, not a hemisphere). The decomposition
 *  fixes that at any radius/length ratio:
 *
 *    - **core** `= clamp(localY, −0.5, 0.5)`, the cylinder section, scaled by the **shaft length** — it
 *      spans one unit, so `core · shaftLength` spans `shaftLength`;
 *    - **cap** `= localY − core`, the hemisphere remainder (`|cap| ≤ 0.5`), scaled by `radius · 2` — the
 *      **same factor x/z take**, which is what makes both caps spheres of exactly `radius`, independent of
 *      the shaft length;
 *    - **translate** by `shaftLength/2 − burial`, putting the shaft's base ring `burial` below the record's
 *      own surface height and its top at `shaftLength − burial`, so the dome's apex lands at
 *      `shaftLength + radius − burial = POST_HEIGHT` above grade exactly.
 *
 *  Everything is multiplied by the record's `scale`, so a scale-0 slot collapses to a point at the record
 *  position (the fixed-`instanceCount` mechanism stage 5 shipped, unchanged). */
export interface PostMeshParams {
    readonly shaftLength: number;
    readonly radius: number;
    readonly burial: number;
}

/** the production mesh mapping's parameters — what {@link postsVs} bakes as literals. */
export const POST_MESH: PostMeshParams = {
    shaftLength: POST_SHAFT_LENGTH,
    radius: POST_RADIUS,
    burial: POST_BURIAL_DEPTH,
};

/** the CPU twin of {@link postsVs}'s vertex arithmetic: the offset (in metres, relative to the record's
 *  own world position — i.e. relative to the terrain surface under the post) of the capsule vertex whose
 *  local position is `(lx, ly, lz)`. Independently parameterized so an arm can vary the shaft length and
 *  witness that the cap term does not move with it. */
export function postVertexOffset(
    lx: number,
    ly: number,
    lz: number,
    params: PostMeshParams = POST_MESH,
    scale = 1,
): [number, number, number] {
    const core = Math.min(Math.max(ly, -0.5), 0.5);
    const cap = ly - core;
    const radial = params.radius * 2;
    const y = core * params.shaftLength + cap * radial + (params.shaftLength / 2 - params.burial);
    return [lx * radial * scale, y * scale, lz * radial * scale];
}

const postsVs = tgpu.fn(
    [VsIn],
    postsPatch,
)((input) => {
    "use gpu";
    const record = postsSurfaceLayout.$.posts[input.iid];
    const scale = record.pos.w;
    const worldPos = record.pos.xyz;

    // the core/cap decomposition (see postVertexOffset, the CPU twin of exactly this arithmetic): the
    // cylinder section scales by the shaft length, the hemisphere remainder by the radius alone — so the
    // caps stay spherical at POST_RADIUS — and the whole post is translated down by POST_BURIAL_DEPTH so
    // the shaft meets the ground at a line with the bottom cap buried.
    const core = std.clamp(input.localPos.y, d.f32(-0.5), d.f32(0.5));
    const cap = input.localPos.y - core;
    const radial = d.f32(POST_RADIUS * 2);
    const sx = input.localPos.x * radial * scale;
    const sy =
        (core * d.f32(POST_SHAFT_LENGTH) +
            cap * radial +
            d.f32(POST_SHAFT_LENGTH / 2 - POST_BURIAL_DEPTH)) *
        scale;
    const sz = input.localPos.z * radial * scale;

    return postsPatch({
        world: d.vec4f(sx + worldPos.x, sy + worldPos.y, sz + worldPos.z, d.f32(1)),
        worldNormal: input.worldNormal,
        clip: d.vec4f(0),
    });
});

const postsFs = tgpu.fn(
    [fsCtxSchema({})],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const color = d.vec3f(d.f32(POST_COLOR[0]), d.f32(POST_COLOR[1]), d.f32(POST_COLOR[2]));
    return d.vec4f(lit(color, ctx.worldNormal), d.f32(1));
});

// --- lifecycle ---------------------------------------------------------------

type PostsRoot = typeof Compute.root;
type PostsPipeline = ReturnType<PostsRoot["createComputePipeline"]>;
type PostsBindGroup = ReturnType<PostsRoot["createBindGroup"]>;
type PostBuffer = TgpuBuffer<typeof PostArray> & StorageFlag;

let postsRaw: GPUBuffer | null = null;
let postsTyped: PostBuffer | null = null;
let postsPermRaw: GPUBuffer | null = null;
let argsRaw: GPUBuffer | null = null;
let argsTyped: DrawIndirectBuffer | null = null;
let pipeline: PostsPipeline | null = null;
let bindGroup: PostsBindGroup | null = null;
let draw: Draw | null = null;

function teardown(): void {
    if (draw) Draws.delete(draw.name);
    postsRaw?.destroy();
    postsPermRaw?.destroy();
    argsRaw?.destroy();
    postsRaw = null;
    postsTyped = null;
    postsPermRaw = null;
    argsRaw = null;
    argsTyped = null;
    pipeline = null;
    bindGroup = null;
    draw = null;
    postsWarmed = false;
}

/** allocate the posts buffer, pipeline, bind group, and registered draw. Call once from
 *  `terrain.ts`'s `warm()`, after the capsule mesh is registered (Part plugin). Ties cleanup to
 *  `state.onDispose` (the same in-place-rebuild-safe pattern `terrain.ts` uses). */
export function warmPosts(state: State): void {
    teardown();
    state.onDispose(teardown);
    const { device, root } = Compute;

    postsRaw = device.createBuffer({
        label: "posts-records",
        size: POST_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    postsTyped = root.createBuffer(PostArray, postsRaw).$usage("storage");

    argsRaw = device.createBuffer({
        label: "posts-draw-args",
        size: d.sizeOf(DrawIndexedIndirect),
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    argsTyped = root.createBuffer(DrawIndexedIndirect, argsRaw).$usage("indirect");

    const capsule = Meshes.get("capsule");
    if (!capsule) throw new Error("posts: capsule mesh not registered");
    argsTyped.write({
        indexCount: capsule.indexCount,
        instanceCount: POST_COUNT,
        firstIndex: capsule.indexBase,
        baseVertex: 0,
        firstInstance: 0,
    });

    pipeline = root.createComputePipeline({ compute: postsKernel }).$name("posts-dispatch");
    bindGroup = root.createBindGroup(postsComputeLayout, { posts: postsTyped });

    const drawSpec: Draw = {
        name: "posts",
        surface: "posts",
        mesh: "capsule",
        args: { indirect: argsTyped },
    };
    draw = drawSpec;
    Draws.register(drawSpec);

    registerSurface(state, {
        name: "posts",
        layout: postsSurfaceLayout,
        vs: postsVs,
        fs: postsFs,
    });

    // publish the posts buffer so the surface renderer resolves the "posts" binding by name
    Compute.buffers.set("posts", postsRaw);
    Compute.typed.set("posts", postsTyped);
}

let postsWarmed = false;

/** dispatch the posts compute kernel — one thread per slot, writing every `Post` record. Call after
 *  every `setNetwork` (the network bind group must be up-to-date so the kernel reads the current chord
 *  and `flattenedHeightAt` uses the current falloff). Never per frame. Precompiles the pipeline on the
 *  first call (the same `precompile` pattern `generate.ts`'s `run` uses), binding all three layouts the
 *  kernel closes over — `postsComputeLayout`, `networkLayout`, and `noiseLayout` (through
 *  `flattenedHeightAt` → `heightAt`) — with temporary warm buffers that are destroyed after. */
export async function dispatchPosts(seed: number): Promise<void> {
    if (!pipeline || !bindGroup) throw new Error("posts: dispatch before warmPosts");
    const activePipeline = pipeline;
    const activeBindGroup = bindGroup;
    const { device, root } = Compute;

    if (!postsWarmed) {
        const warmLabel = precompileScope("posts-dispatch");
        const owned: { raw: GPUBuffer | null } = { raw: null };
        try {
            await precompile(warmLabel, () => {
                owned.raw = device.createBuffer({
                    label: `${warmLabel}-perm`,
                    size: d.sizeOf(PermData),
                    usage: GPUBufferUsage.STORAGE,
                });
                const warmPerm = root.createBuffer(PermData, owned.raw).$usage("storage");
                const noiseGroup = root.createBindGroup(noiseLayout, { perm: warmPerm });
                const enc = device.createCommandEncoder({ label: warmLabel });
                const pass = enc.beginComputePass({ label: warmLabel });
                activePipeline
                    .with(noiseGroup)
                    .with(activeBindGroup)
                    .with(networkBindGroup())
                    .with(pass)
                    .dispatchWorkgroups(0);
                pass.end();
                device.queue.submit([enc.finish()]);
                return activePipeline;
            });
        } finally {
            owned.raw?.destroy();
        }
        postsWarmed = true;
    }

    // The kernel closes over three layouts: postsComputeLayout (the output buffer), networkLayout
    // (the chord segments + falloff, set by setNetwork), and noiseLayout (the permutation, through
    // flattenedHeightAt → heightAt). The precompile above attaches all three; the actual dispatch must
    // attach all three too — the noise bind group carries the seed's permutation so heightAt matches
    // the CPU twin flattenFieldAt's makePermutation(seed). Mirrors generate.ts's run(), which creates
    // a fresh permutation buffer per dispatch and keeps it alive until the next one.
    const perm = makePermutation(seed);
    const nextRaw = device.createBuffer({
        label: "posts-perm",
        size: perm.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    try {
        const nextPerm = root.createBuffer(PermData, nextRaw).$usage("storage");
        device.queue.writeBuffer(nextRaw, 0, perm as Uint32Array<ArrayBuffer>);
        const noiseGroup = root.createBindGroup(noiseLayout, { perm: nextPerm });
        const enc = device.createCommandEncoder({ label: "posts-dispatch" });
        const pass = enc.beginComputePass({ label: "posts-dispatch" });
        activePipeline
            .with(noiseGroup)
            .with(activeBindGroup)
            .with(networkBindGroup())
            .with(pass)
            .dispatchWorkgroups(POST_DISPATCH);
        pass.end();
        device.queue.submit([enc.finish()]);
    } catch (cause) {
        nextRaw.destroy();
        throw cause;
    }
    postsPermRaw?.destroy();
    postsPermRaw = nextRaw;
}

// --- readback + gate --------------------------------------------------------

/** one post record as read back from the GPU — the bridge's return shape. */
interface PostRecord {
    x: number;
    y: number;
    z: number;
    scale: number;
}

/** one-shot GPU→CPU readback of the posts buffer — the device gate's readback arm. Mirrors the
 *  fountain's `readParticles` and `terrain.ts`'s `readVertices`; an assert-only bridge, never a
 *  per-frame readback. */
async function readPosts(): Promise<PostRecord[]> {
    if (!postsRaw) throw new Error("posts: readPosts before warmPosts");
    const { device } = Compute;
    const staging = device.createBuffer({
        label: "posts-readback",
        size: POST_BYTES,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    let mapped = false;
    try {
        const enc = device.createCommandEncoder({ label: "posts-readback" });
        enc.copyBufferToBuffer(postsRaw, 0, staging, 0, POST_BYTES);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        mapped = true;
        const data = new Float32Array(staging.getMappedRange().slice(0));
        const records: PostRecord[] = [];
        for (let i = 0; i < POST_COUNT; i++) {
            const off = i * 4;
            records.push({
                x: data[off],
                y: data[off + 1],
                z: data[off + 2],
                scale: data[off + 3],
            });
        }
        return records;
    } finally {
        if (mapped) staging.unmap();
        staging.destroy();
    }
}

/** the perpendicular distance from (px, pz) to the line through (ax, az) and (bx, bz). */
function perpDist(px: number, pz: number, ax: number, az: number, bx: number, bz: number): number {
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz);
    if (len === 0) return Math.hypot(px - ax, pz - az);
    return Math.abs((dx * (pz - az) - dz * (px - ax)) / len);
}

/** the signed perpendicular distance from (px, pz) to the line through (ax, az) and (bx, bz): positive on
 *  one side, negative on the other. The kernel's lateral normal is `(-uz, ux)` (the left normal of the
 *  chord direction), so a post with `lateral > 0` (even slot, `postLateralSign = +1`) lands on the
 *  positive side and vice versa — the sign of this value is what `checkPosts` compares against
 *  `postLateralSign(i)`. */
function signedPerpDist(
    px: number,
    pz: number,
    ax: number,
    az: number,
    bx: number,
    bz: number,
): number {
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz);
    if (len === 0) return 0;
    return (dx * (pz - az) - dz * (px - ax)) / len;
}

/** the device gate's posts check: reads back the posts buffer and verifies every Validation criterion —
 *  every live slot's `y` equals CPU `flattenFieldAt` at its `(x, z)` (baked against the live seed via
 *  `getCurrentSeed`, not the boot `SEED` — an F9 reseed changes the permutation and the falloff, so a
 *  stale-seed twin produces a false negative), the lateral offset is at the kerb line (`halfWidth +
 *  POST_OFFSET`, re-anchored by stage 11) with the whole footing inside the flat core, each live
 *  slot is on the side `postLateralSign(i)` predicts (the alternation the distance band alone never
 *  checks), the live-slot count is `floor(chordLength / POST_SPACING)`, and every slot past the chord
 *  is at scale 0. Called from `gate.ts` at boot and from `boot.ts`'s `__roadsPostsCheck` bridge after an
 *  edit. */
export async function checkPosts(): Promise<Check> {
    const posts = await readPosts();
    const doc = getDocument();
    const seed = getCurrentSeed();
    const perm = makePermutation(seed);
    const { segments, cutDepth } = buildNetworkGeometry(doc, seed);
    const falloff = computeFalloff(cutDepth);
    const natural = (x: number, z: number) => heightAtCpu(x, z, perm);

    const line = doc.polylines[0];
    const [a, b] = line.points;
    const chordLength = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const halfWidth = line.halfWidth;

    let liveCount = 0;
    let maxDiff = 0;
    let lateralOk = true;
    let lateralSignOk = true;
    let scale0Ok = true;

    for (let i = 0; i < posts.length; i++) {
        const rec = posts[i];
        const station = postStation(i);
        if (rec.scale !== 0) {
            liveCount++;
            const expectedY = flattenFieldAt(rec.x, rec.z, segments, falloff, natural);
            maxDiff = Math.max(maxDiff, Math.abs(rec.y - expectedY));
            // the lateral band, re-anchored on stage 11's kerb-line offset: the record sits at exactly
            // `halfWidth + POST_OFFSET` from the centreline (the old assertion — anywhere between
            // `halfWidth` and `halfWidth + FLAT_CORE_MARGIN` — passed for any offset in a 5.66 m band and
            // so could not witness the offset at all), and the post's whole footing, ±POST_RADIUS about
            // that, is strictly inside the flat core and strictly off the pavement.
            const dist = perpDist(rec.x, rec.z, a[0], a[1], b[0], b[1]);
            if (Math.abs(dist - (halfWidth + POST_OFFSET)) > 0.01) lateralOk = false;
            if (dist - POST_RADIUS <= halfWidth) lateralOk = false;
            if (dist + POST_RADIUS >= halfWidth + FLAT_CORE_MARGIN) lateralOk = false;
            // the alternation check: each live slot must be on the side `postLateralSign(i)` predicts
            // (even → +1, odd → −1). The kernel's `select(1f, -1f, (i & 1u) == 1u)` and this CPU twin
            // must agree on which side of the road every post sits — the distance band alone never
            // catches a swapped or constant sign. This gives `postLateralSign` a production reader.
            const signed = signedPerpDist(rec.x, rec.z, a[0], a[1], b[0], b[1]);
            if (Math.sign(signed) !== postLateralSign(i)) lateralSignOk = false;
        } else {
            if (station <= chordLength) scale0Ok = false;
        }
    }

    const expectedLive = liveSlotCount(chordLength);
    const pass =
        maxDiff < 0.01 && liveCount === expectedLive && lateralOk && lateralSignOk && scale0Ok;
    return {
        name: "post-placement",
        pass,
        detail: `live=${liveCount}/${expectedLive} maxDiff=${maxDiff.toFixed(6)} lateralOk=${lateralOk} lateralSignOk=${lateralSignOk} scale0Ok=${scale0Ok}`,
    };
}
