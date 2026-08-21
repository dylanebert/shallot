// Stage 5 — posts (`roads-interactive.md` stage 5). A row of abstract posts along the road, positioned
// entirely on the GPU: a compute kernel writes one `Post` record per slot (station along the chord,
// lateral offset alternating sides, `y` from `flatten.ts`'s own `flattenedHeightAt` TGSL fn, scale 0 for
// slots past the chord), and a custom surface's VS scales the standard `capsule` mesh by the post
// dimensions and translates by the record. The fountain showcase's shape: a typed record buffer written
// by compute, read by a custom surface's VS at `input.iid`, one shared mesh, `Draws.register` with a
// fixed `instanceCount`. The records never touch Part or the Transform slabs.
//
// `POST_COUNT` is sized off the world's own diagonal (`WORLD_EXTENT · √2` ≈ 1448 m, since
// `ROAD_MAX_LENGTH` was deleted in stage 4c) divided by `POST_SPACING`, so `instanceCount` stays fixed
// across edits and short chords leave most slots at scale 0. Re-dispatched after every `setNetwork` (edit
// and reseed), never per frame — `terrain.ts`'s `warm`, `regenerate`, and `editDocument` each call
// `dispatchPosts(seed)` after `setNetwork`.

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
import {
    buildNetworkGeometry,
    computeFalloff,
    flattenedHeightAt,
    networkBindGroup,
    networkLayout,
} from "./terrain/flatten";
import { FLAT_CORE_MARGIN } from "./terrain/flatten-math";
import { SPACING, WORLD_EXTENT } from "./terrain/grid";
import { makePermutation, noiseLayout, PermData } from "./terrain/noise";
import { heightAtCpu } from "./terrain/profile";
import { getCurrentSeed, getDocument } from "./terrain/terrain";

// --- constants --------------------------------------------------------------

/** metres between posts along the chord.
 *
 *  Referent: NACTO Urban Street Design Guide — traffic-calming / channelizing bollards are spaced
 *  1.5–2.0 m apart for pedestrian channelization (nacto.org, accessed 2026-08-22). 2.0 m is the upper
 *  bound of that range, within the spec's ~2 m band. */
export const POST_SPACING = 2;

/** the lateral offset from the road edge to the post centre, in metres.
 *
 *  Derivation of the admissible band. The flat core is the region where `coreDist <= 0`, where
 *  `coreDist = distance(point, segment) − (halfWidth + FLAT_CORE_MARGIN)` (`flatten.ts`'s `networkCore`).
 *  The flat core extends from the centreline (perpendicular distance 0) to `halfWidth + FLAT_CORE_MARGIN`
 *  from it. A post at `halfWidth + POST_OFFSET` from the centreline is inside the flat core iff
 *  `POST_OFFSET <= FLAT_CORE_MARGIN`. For the post to sit outside the road surface (not on the road),
 *  `POST_OFFSET > 0`. For the post's foot to meet the *rendered surface* rather than the field's idea of
 *  it — the property Validation names: inside the flat core the field is affine and the rendered mesh
 *  reproduces it exactly — the post must be strictly inside, so `POST_OFFSET < FLAT_CORE_MARGIN`.
 *  Therefore the admissible band is `0 < POST_OFFSET < FLAT_CORE_MARGIN`.
 *  `FLAT_CORE_MARGIN = √2 · SPACING = √2 · 4 ≈ 5.657 m` (`flatten-math.ts`).
 *  `POST_OFFSET = SPACING = 4 m` sits well inside the band (0 < 4 < 5.657).
 *
 *  The spec's Approach wrote the offset as `halfWidth + FLAT_CORE_MARGIN − √2·SPACING` from the
 *  centreline, but `FLAT_CORE_MARGIN = √2 · SPACING`, so that expression reduces to exactly `halfWidth` —
 *  the flat core's *inner* edge, not a point inside it. The real constraint is `0 < POST_OFFSET <
 *  FLAT_CORE_MARGIN`, derived above. (Open question 1 — reported to the spec owner for correction.) */
export const POST_OFFSET = SPACING;

/** the post's visual height in metres. The capsule mesh spans y ∈ [−1, 1] (height 2), so the VS scales
 *  y by `POST_HEIGHT / 2`.
 *
 *  Referent: a standard roadside bollard is ~1.0 m above grade. Reliance Foundry R-7181 cast-iron
 *  bollard: 36 in (0.914 m) above grade (reliance-fd.com, accessed 2026-08-22); UK DfT Traffic Signs
 *  Manual Chapter 4 describes traffic bollards at approximately 1.0 m. 1.0 m is within the spec's
 *  ~1 m band. */
const POST_HEIGHT = 1;

/** the post's visual radius in metres. The capsule mesh has radius 0.5, so the VS scales x/z by
 *  `POST_RADIUS / 0.5 = POST_RADIUS * 2`.
 *
 *  Referent: a standard pipe bollard made from 10 in Schedule 40 steel pipe has an OD of 10.75 in
 *  (273 mm, radius 0.137 m); 8 in Schedule 40 has OD 8.625 in (219 mm, radius 0.11 m). 0.12 m (diameter
 *  240 mm) sits between the two, within the spec's ~0.1–0.15 m band. */
const POST_RADIUS = 0.12;

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

// --- surface (VS + FS) -------------------------------------------------------

const postsPatch = vsPatchSchema({});

const postsVs = tgpu.fn(
    [VsIn],
    postsPatch,
)((input) => {
    "use gpu";
    const record = postsSurfaceLayout.$.posts[input.iid];
    const scale = record.pos.w;
    const worldPos = record.pos.xyz;

    // scale the unit capsule (radius 0.5, y ∈ [−1, 1]) to post dimensions, then translate.
    // y is offset by POST_HEIGHT/2 so the foot (bottom) sits at worldPos.y (the terrain surface).
    const sx = input.localPos.x * d.f32(POST_RADIUS * 2) * scale;
    const sy = input.localPos.y * d.f32(POST_HEIGHT / 2) * scale + d.f32(POST_HEIGHT / 2) * scale;
    const sz = input.localPos.z * d.f32(POST_RADIUS * 2) * scale;

    return postsPatch({
        world: d.vec4f(sx + worldPos.x, sy + worldPos.y, sz + worldPos.z, d.f32(1)),
        worldNormal: input.worldNormal,
        clip: d.vec4f(0),
    });
});

const POST_COLOR: readonly [number, number, number] = [0.5, 0.4, 0.3];

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
 *  stale-seed twin produces a false negative), the lateral offset is inside the flat core, each live
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
            const dist = perpDist(rec.x, rec.z, a[0], a[1], b[0], b[1]);
            if (dist > halfWidth + FLAT_CORE_MARGIN || dist < halfWidth) lateralOk = false;
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
