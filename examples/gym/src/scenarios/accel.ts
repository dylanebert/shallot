import {
    AmbientLight,
    arrow,
    box,
    Camera,
    CameraMode,
    Compute,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    LinesPlugin,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    type Plugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    type System,
    segment,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import {
    BVH_FEATURES,
    BVH_TRAIL_LEVELS,
    type Bvh,
    createBvh,
    createRadixSort,
    KEYS_PER_BLOCK,
    type RadixSort,
} from "@dylanebert/shallot/bvh/core";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { BeginFrameSystem, Render } from "@dylanebert/shallot/render/core";
// test scaffolding (tests/, out of the published src/) reached by relative path — the executable spec.
import {
    allFixtures,
    clumps,
    clustered,
    coincident,
    coplanar,
    giantAndTiny,
    PRIM_F32,
    type Prims,
    primMax,
    primMin,
    slivers,
    uniformRandom,
} from "../../../../packages/shallot/tests/bvh/fixtures";
import {
    type Bvh2,
    compareRays,
    invariants,
    mortonCodes,
    nearestHitBrute,
    refit as oracleRefit,
    type Ray,
    rays,
    sceneBounds,
    sortMorton,
    treeMaxDepth,
} from "../../../../packages/shallot/tests/bvh/oracle";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";
import {
    createTracer,
    MAX_PRIMS,
    packRays,
    readbackBvh,
    shuffledPrims,
    type Tracer,
} from "./bvh-lib";

// accel — the GPU acceleration-structure pipeline as one composable scene. sort, build, and traverse are
// the same pipeline (a Morton radix sort feeds the LBVH build, the tree feeds the ray traverse), so they
// live in one orbitable demonstration with URL-param / control-panel layers: the **curve** (a polyline
// through the primitives in sorted Morton order — the space-filling curve the sort produces, the viz the
// bar-chart test port never had), the **tree** (every BVH node's AABB as a wireframe box), and the
// **rays** (a batch traced into the scene, hits as arrows). The builder + traverser run every frame on
// the frame encoder, so ProfilePlugin times `bvh:bounds/morton/sort/build` + `bvh:trace`. The assert
// concatenates three oracle gates: sort stability + partition boundaries (a standalone radix sort run
// during the assert, the DF-sort regression guard), build invariants + refit + coherence + depth, and
// traverse closest/any-hit vs brute force. References: the validated `bvh/core` + the `tests/bvh`
// oracle, the executable spec these reuse unchanged.

const INTERNAL_COLOR = 0x3b82f6; // blue — the BVH hierarchy
const LEAF_COLOR = 0xf59e0b; // amber — the prim AABBs (leaf j bounds prim j)
const CURVE_LOW = 0x1e40af; // deep blue — curve start
const CURVE_HIGH = 0xfbbf24; // amber — curve end
const GEO_COLOR = 0x52525b; // dim grey — the primitive boxes the rays test
const HIT_COLOR = 0x22c55e; // green — a ray to its hit point
const MISS_COLOR = 0x3f3f46; // faint grey — a ray that found nothing
const PRIM_COLOR = 0xf59e0b; // amber — a pierced primitive
const INVALID = 0xffffffff;

// the live scene the params select. The asserts sweep the whole fixture matrix regardless; this is the
// scene a live tab orbits + the steady build/trace the profiler measures.
function fixture(dist: string, n: number, seed: number): Prims {
    switch (dist) {
        case "clustered":
            return clustered(n, seed);
        case "slivers":
            return slivers(n, seed);
        case "coplanar":
            return coplanar(n, seed);
        case "coincident":
            return coincident(n, seed);
        case "clumps":
            return clumps(Math.max(1, Math.floor(n / 32)), 32, seed);
        case "giant":
            return giantAndTiny(n, seed);
        default:
            return uniformRandom(n, seed);
    }
}

// one page = one scenario instance, so the GPU structures + Mirrors + scene live in module scope.
let bvh: Bvh | null = null;
let tracer: Tracer | null = null;
let nodes: Mirror | null = null;
let hits: Mirror | null = null;
let scene: Prims | null = null;
let curveOrder: number[] = []; // prim indices in Morton-sorted order — the curve viz path
let vizRays: Ray[] = [];
let rayCount = 0;
let buildMode: "build" | "refit" = "build"; // the assert flips to refit for that sub-gate
let assertMode: number | null = null; // the traverse assert locks the trace mode; null → follow params
let params: Params | null = null; // live values the per-frame pass + viz read

// the standalone radix sort the sort assert exercises (the DF-sort boundary + stability guard). Kept off
// the per-frame path — the build's internal sort carries the `bvh:sort` span — and run by hand during the
// assert, so its mirror copies don't compete with the measured spine.
const MAX_KEYS = 1 << 16;
let rs: RadixSort | null = null;
let countBuf: GPUBuffer | null = null;
let keysMirror: Mirror | null = null;
let payloadMirror: Mirror | null = null;

function liveMode(): number {
    return assertMode ?? (params?.mode === "any" ? 1 : 0);
}

function uploadScene(prims: Prims): void {
    if (!bvh) return;
    Compute.device.queue.writeBuffer(bvh.prims, 0, prims.data);
    Compute.device.queue.writeBuffer(bvh.count, 0, new Uint32Array([prims.count]));
}

function uploadRays(batch: Ray[]): void {
    if (!tracer) return;
    rayCount = batch.length;
    Compute.device.queue.writeBuffer(tracer.rays, 0, packRays(batch));
}

function centroid(prims: Prims, i: number): [number, number, number] {
    const lo = primMin(prims, i);
    const hi = primMax(prims, i);
    return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
}

// ============================ per-frame pipeline (the measured spine) ============================

// one arm = a full builder + tracer + standalone sort set. warm builds the live arm (the one the
// profiler times); the assert builds the other arm on the side to gate it in the same run.
interface ArmGpu {
    bvh: Bvh;
    tracer: Tracer;
    rs: RadixSort;
    countBuf: GPUBuffer;
}

function armName(sub: boolean): string {
    return sub ? "subgroup" : "lds";
}

async function makeArm(sub: boolean, cap: number): Promise<ArmGpu> {
    const b = await createBvh(Compute.device, cap, undefined, sub);
    const t = await createTracer(Compute.device, b.nodes, b.count);
    const cb = Compute.device.createBuffer({
        label: "gym-accel-sort-count",
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const r = await createRadixSort(Compute.device, MAX_KEYS, { count: cb }, sub);
    return { bvh: b, tracer: t, rs: r, countBuf: cb };
}

function disposeArm(a: ArmGpu): void {
    a.tracer.destroy();
    a.bvh.destroy();
    a.rs.destroy();
    a.countBuf.destroy();
}

function accelPlugin(): Plugin {
    const pass: System = {
        group: "draw",
        annotations: { mode: "always" },
        after: [BeginFrameSystem], // opens Render.encoder; build is a pure-compute producer pass on it
        update() {
            if (!bvh || !tracer || !Render.encoder) return;
            if (buildMode === "refit") bvh.refit(Render.encoder);
            else bvh.build(Render.encoder);
            // build then trace on the same encoder → the nodes the trace reads are in-order visible
            if (rayCount > 0) tracer.trace(Render.encoder, rayCount, liveMode());
        },
    };
    const draw: System = {
        group: "simulation",
        annotations: { mode: "always" },
        update: drawOverlay,
    };
    // the arm the live tab measures + orbits; `subgroups=false` forces the LDS kernels (bounds reduce
    // + radix sort). The assert gates BOTH arms every run regardless of this, so it only selects which
    // one the profiler times. `subgroups` is requested as a *preferred* feature (below), so the device
    // carries it where present and either arm builds — the assert can stand up the opposite arm on the
    // side without a second device.
    const sub = (params?.subgroups as boolean) ?? true;
    // `cap` sizes the per-frame builder — set it == `count` to measure the sort/bounds at a fair
    // scale (the LDS sort dispatches at capacity, so a capacity == live-count run compares equal
    // per-block work, the regime a physics broadphase BVH actually runs in).
    const cap = Math.max(1, (params?.cap as number) ?? MAX_PRIMS);
    return {
        name: "GymAccel",
        systems: [pass, draw],
        dependencies: [RenderPlugin, LinesPlugin],
        preferredFeatures: BVH_FEATURES, // best-effort subgroups; both arms build either way
        async warm() {
            const arm = await makeArm(sub, cap);
            bvh = arm.bvh;
            tracer = arm.tracer;
            rs = arm.rs;
            countBuf = arm.countBuf;
            const dist = (params?.dist as string) ?? "uniform";
            const n = (params?.count as number) ?? 256;
            const seed = (params?.seed as number) ?? 1;
            setScene(fixture(dist, n, seed));
            vizRays = rays(sceneBounds(scene!), 12, 0x5eed); // 24 rays — a coherent shell batch
            uploadRays(vizRays);
        },
    };
}

// adopt a new live scene: upload it, recompute the Morton-order curve (the CPU reference matches what the
// GPU sort produces — derivable, deterministic), and keep it for the overlay + assert restore.
function setScene(prims: Prims): void {
    scene = prims;
    curveOrder = Array.from(sortMorton(mortonCodes(prims, sceneBounds(prims))));
    uploadScene(prims);
}

// ============================ overlay viz (curve → tree → rays) ============================

function lerpColor(a: number, b: number, t: number): number {
    const lerp = (sh: number) =>
        Math.round(((a >> sh) & 255) + (((b >> sh) & 255) - ((a >> sh) & 255)) * t);
    return (lerp(16) << 16) | (lerp(8) << 8) | lerp(0);
}

function drawCurve(prims: Prims): void {
    if (curveOrder.length < 2) return;
    let prev = centroid(prims, curveOrder[0]);
    for (let i = 1; i < curveOrder.length; i++) {
        const cur = centroid(prims, curveOrder[i]);
        segment(prev, cur, lerpColor(CURVE_LOW, CURVE_HIGH, i / (curveOrder.length - 1)), 2);
        prev = cur;
    }
}

// one wireframe box per node — leaves (the prim AABBs) amber, internal nodes (the hierarchy) blue.
function drawTree(tree: Bvh2, n: number): void {
    const totalNodes = Math.max(1, 2 * n - 1);
    for (let node = 0; node < totalNodes; node++) {
        const o = node * 8;
        const leaf = tree.child[o + 3] === INVALID;
        box(
            [tree.bounds[o], tree.bounds[o + 1], tree.bounds[o + 2]],
            [tree.bounds[o + 4], tree.bounds[o + 5], tree.bounds[o + 6]],
            leaf ? LEAF_COLOR : INTERNAL_COLOR,
        );
    }
}

// the primitive boxes dim, the viz rays cast over them: a closest hit is a green arrow to its hit point
// with the pierced box lit amber; a miss is a faint grey ray. Only meaningful in closest mode.
function drawRays(prims: Prims): void {
    for (let i = 0; i < prims.count; i++) box(primMin(prims, i), primMax(prims, i), GEO_COLOR);
    const snap = hits?.snapshot;
    if (!snap || liveMode() !== 0) return;
    const t = new Float32Array(snap.bytes);
    const u = new Uint32Array(snap.bytes);
    for (let i = 0; i < vizRays.length; i++) {
        const { origin: o, dir: d } = vizRays[i];
        const prim = u[i * 2 + 1];
        if (prim === INVALID) {
            segment(o, [o[0] + d[0], o[1] + d[1], o[2] + d[2]], MISS_COLOR);
            continue;
        }
        const k = t[i * 2];
        arrow(o, [o[0] + d[0] * k, o[1] + d[1] * k, o[2] + d[2] * k], HIT_COLOR, 2);
        box(primMin(prims, prim), primMax(prims, prim), PRIM_COLOR, 2);
    }
}

// the layered overlay — `viz` is the master switch (a clean frame for the headless span read), each layer
// its own toggle. The build + trace passes always run (the measured spine); this only draws.
function drawOverlay(): void {
    if (!params?.viz || !scene) return;
    if (params.curve) drawCurve(scene);
    if (params.tree) {
        const snap = nodes?.snapshot;
        if (snap) drawTree(readbackBvh(snap.bytes, scene.count), scene.count);
    }
    if (params.rays) drawRays(scene);
}

// ============================ sort assert (standalone DF-sort guard) ============================

type Pattern = "random" | "equal" | "sorted" | "reverse" | "halfMax";

// counts spanning sub-block / single-block / multi-block regimes (the partition boundaries the Decoupled
// Fallback sort must get right), then the pathological distributions at a fixed multi-block size.
const SORT_CASES: { count: number; pattern: Pattern }[] = [
    { count: 1, pattern: "random" },
    { count: 2, pattern: "reverse" },
    { count: KEYS_PER_BLOCK - 1, pattern: "random" },
    { count: KEYS_PER_BLOCK, pattern: "random" },
    { count: KEYS_PER_BLOCK + 1, pattern: "random" },
    { count: 8192, pattern: "random" },
    { count: MAX_KEYS, pattern: "random" },
    { count: 50000, pattern: "equal" }, // one bin every pass — worst-case ranking contention
    { count: 50000, pattern: "sorted" },
    { count: 50000, pattern: "reverse" },
    { count: 50000, pattern: "halfMax" }, // real 0xffffffff keys must sort before the tail padding
];

function genKeys(n: number, pattern: Pattern, seed: number): Uint32Array<ArrayBuffer> {
    const keys = new Uint32Array(n);
    let s = seed >>> 0 || 1;
    const rand = (): number => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s;
    };
    for (let i = 0; i < n; i++) {
        switch (pattern) {
            case "random":
                keys[i] = rand();
                break;
            case "equal":
                keys[i] = 0x9e3779b9;
                break;
            case "sorted":
                keys[i] = i;
                break;
            case "reverse":
                keys[i] = n - 1 - i;
                break;
            case "halfMax":
                keys[i] = rand() & 1 ? 0xffffffff : rand();
                break;
        }
    }
    return keys;
}

// upload an unsorted case to the standalone sort: keys + identity payload + GPU count, the [count,
// blockMultiple) tail sentinel-padded to 0xffffffff (the Morton pass does this in production).
function uploadCase(
    c: { count: number; pattern: Pattern },
    seed: number,
): Uint32Array<ArrayBuffer> {
    if (!rs || !countBuf) return new Uint32Array(0);
    const keys = genKeys(c.count, c.pattern, seed);
    const payload = new Uint32Array(c.count);
    for (let i = 0; i < c.count; i++) payload[i] = i;
    Compute.device.queue.writeBuffer(rs.keys, 0, keys);
    Compute.device.queue.writeBuffer(rs.payload, 0, payload);
    Compute.device.queue.writeBuffer(countBuf, 0, new Uint32Array([c.count]));
    const padded = Math.ceil(c.count / KEYS_PER_BLOCK) * KEYS_PER_BLOCK;
    if (padded > c.count) {
        Compute.device.queue.writeBuffer(
            rs.keys,
            c.count * 4,
            new Uint32Array(padded - c.count).fill(0xffffffff),
        );
    }
    return keys;
}

// gate the sorted readback against the oracle: keys ascending, equal keys in ascending payload order
// (stability) — payload was the original index, the contract the build's leaf order depends on.
function gateSort(
    c: { count: number },
    input: Uint32Array,
    keysBytes: ArrayBuffer,
    payloadBytes: ArrayBuffer,
): string {
    const gotKeys = new Uint32Array(keysBytes);
    const gotPayload = new Uint32Array(payloadBytes);
    const order = sortMorton(input); // stable permutation by key — the executable spec
    for (let i = 0; i < c.count; i++) {
        const ek = input[order[i]];
        const ep = order[i];
        if (gotKeys[i] !== ek || gotPayload[i] !== ep) {
            return `at ${i}: got (key ${gotKeys[i] >>> 0}, payload ${gotPayload[i]}), expected (key ${ek >>> 0}, payload ${ep})`;
        }
    }
    return "";
}

async function sortCase(c: { count: number; pattern: Pattern }, seed: number): Promise<string> {
    if (!rs || !keysMirror || !payloadMirror) return "no sort";
    const input = uploadCase(c, seed);
    const enc = Compute.device.createCommandEncoder({ label: "gym-accel-sort" });
    rs.sortIndirect(enc); // run by hand — off the per-frame spine, so no competing span
    Compute.device.queue.submit([enc.finish()]);
    await frames(3);
    await settle(keysMirror);
    await settle(payloadMirror);
    const k = keysMirror.snapshot?.bytes;
    const p = payloadMirror.snapshot?.bytes;
    if (!k || !p) return "no snapshot";
    return gateSort(c, input, k, p);
}

// ============================ build + traverse asserts ============================

// translate every prim by a deterministic per-prim offset — same count + leaf-index order, so leaf j
// still maps to prim j and a refit applies (the jitter forces every node bound to recompute).
function move(prims: Prims, seed: number): Prims {
    let s = seed >>> 0 || 1;
    const rand = (): number => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return (s / 0x100000000 - 0.5) * 6;
    };
    const data = new Float32Array(prims.data);
    for (let i = 0; i < prims.count; i++) {
        const o = i * PRIM_F32;
        const dx = rand();
        const dy = rand();
        const dz = rand();
        data[o] += dx;
        data[o + 1] += dy;
        data[o + 2] += dz;
        data[o + 4] += dx;
        data[o + 5] += dy;
        data[o + 6] += dz;
    }
    return { count: prims.count, data };
}

// upload a scene, let the per-frame pass (build or refit, per `buildMode`) run over it, read the tree
// back through the Mirror. Returns a stable copy (the mirror ring slot is reused).
async function pump(prims: Prims): Promise<Bvh2 | null> {
    uploadScene(prims);
    await frames(3);
    await settle(nodes!);
    const snap = nodes?.snapshot;
    return snap ? readbackBvh(snap.bytes.slice(0), prims.count) : null;
}

// gate one built tree on the oracle: structural invariants + ray-vs-brute-force agreement. Both exact.
function gateBuild(tree: Bvh2 | null, prims: Prims, seed: number): string {
    if (!tree) return "no snapshot";
    const errs = invariants(tree, prims);
    if (errs.length) return `invariants: ${errs.slice(0, 3).join("; ")}`;
    const rayErrs = compareRays(tree, prims, rays(sceneBounds(prims), 64, seed));
    if (rayErrs.length) return `rays: ${rayErrs.slice(0, 3).join("; ")}`;
    return "";
}

// build → readback the GPU topology → refit moved prims twice over it → gate bit-exact vs the oracle
// refit over that same topology, plus invariants + rays on the moved tree.
async function refitCheck(): Promise<Check> {
    const base = uniformRandom(1024, 0xc0ffee);
    buildMode = "build";
    const built = await pump(base);
    if (!built) return { name: "refit", pass: false, detail: "no build snapshot" };

    buildMode = "refit";
    await pump(move(base, 0x9e3779b9)); // first refit — leaves the arrival flags reset for the second
    const movedB = move(base, 0x85ebca6b);
    const gpu = await pump(movedB);
    buildMode = "build";
    if (!gpu) return { name: "refit", pass: false, detail: "no refit snapshot" };

    oracleRefit(built, movedB); // the oracle refits the GPU's own topology — a correct climb is bit-exact
    for (let node = 0; node < gpu.count; node++) {
        for (const w of [0, 1, 2, 4, 5, 6]) {
            const idx = node * 8 + w;
            if (built.bounds[idx] !== gpu.bounds[idx]) {
                return {
                    name: "refit",
                    pass: false,
                    detail: `node ${node} bound ${w}: gpu ${gpu.bounds[idx]}, oracle ${built.bounds[idx]}`,
                };
            }
        }
    }
    const errs = invariants(gpu, movedB);
    if (errs.length) return { name: "refit", pass: false, detail: `invariants: ${errs[0]}` };
    const rayErrs = compareRays(gpu, movedB, rays(sceneBounds(movedB), 64, 0xfeed));
    if (rayErrs.length) return { name: "refit", pass: false, detail: `rays: ${rayErrs[0]}` };
    return { name: "refit", pass: true, detail: "bit-exact vs oracle refit + invariants + rays" };
}

// upload a scene + ray batch, let the per-frame build → trace run over them, read the hits back. The
// trace mode is locked for the sweep so the per-frame pass uses it regardless of the live `mode` control.
async function retrace(prims: Prims, batch: Ray[], m: number): Promise<ArrayBuffer | null> {
    assertMode = m;
    uploadScene(prims);
    uploadRays(batch);
    await frames(3);
    await settle(hits!);
    return hits?.snapshot?.bytes ?? null;
}

// closest-hit (m=0): distance must agree within the f32-vs-f64 tolerance (the prim may differ on a tie).
function gateClosest(bytes: ArrayBuffer | null, prims: Prims, batch: Ray[]): string {
    if (!bytes) return "no snapshot";
    const t = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const sb = sceneBounds(prims);
    const span = Math.max(sb.max[0] - sb.min[0], sb.max[1] - sb.min[1], sb.max[2] - sb.min[2], 1);
    const tol = span * 1e-4 + 1e-4;
    for (let i = 0; i < batch.length; i++) {
        const brute = nearestHitBrute(prims, batch[i]);
        const prim = u[i * 2 + 1];
        const gpuMiss = prim === INVALID;
        if (brute === null) {
            if (!gpuMiss) return `ray ${i}: brute miss, gpu hit prim ${prim}`;
            continue;
        }
        if (gpuMiss) return `ray ${i}: brute hit prim ${brute.prim}, gpu miss`;
        if (Math.abs(t[i * 2] - brute.t) > tol)
            return `ray ${i}: gpu t=${t[i * 2]}, brute t=${brute.t} (tol ${tol.toFixed(5)})`;
    }
    return "";
}

// any-hit (m=1): with tMax = ∞ the occlusion answer is exactly "brute found a hit".
function gateAny(bytes: ArrayBuffer | null, prims: Prims, batch: Ray[]): string {
    if (!bytes) return "no snapshot";
    const u = new Uint32Array(bytes);
    for (let i = 0; i < batch.length; i++) {
        const want = nearestHitBrute(prims, batch[i]) !== null;
        if ((u[i * 2] === 1) !== want) return `ray ${i}: gpu ${u[i * 2] === 1}, brute ${want}`;
    }
    return "";
}

// the full pipeline gate (sort → build → refit → empty → depth → coherence → traverse) over the
// current module-global structures, each check tagged with the arm under test. The assert calls it
// once per arm; the structures the helpers read (`bvh`/`tracer`/`rs` + the four mirrors) are the
// arm's, swapped in by the caller. Returns the checks; restoring the live scene is the caller's job.
async function gate(arm: string): Promise<Check[]> {
    const checks: Check[] = [];
    const tag = (name: string): string => `${arm} ${name}`;

    // --- sort: stability + partition boundaries + pathological distributions (the DF-sort guard) ---
    let seed = 0x5eed;
    for (const c of SORT_CASES) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        const detail = await sortCase(c, seed);
        checks.push({
            name: tag(`sort ${c.pattern} N=${c.count}`),
            pass: detail === "",
            detail: detail || "sorted + stable",
        });
    }

    // --- build: the fixture matrix gated on invariants + rays, then refit, empty, depth, coherence ---
    buildMode = "build";
    for (const { name, prims } of allFixtures()) {
        const detail = gateBuild(await pump(prims), prims, 0xabcdef ^ prims.count);
        checks.push({
            name: tag(`build ${name}`),
            pass: detail === "",
            detail: detail || `${prims.count} prims ok`,
        });
    }

    const refit = await refitCheck();
    checks.push({ ...refit, name: tag(refit.name) });

    // empty: the count=0 a producer writes when its last entity leaves must build as a safe no-op.
    buildMode = "build";
    uploadScene({ count: 0, data: new Float32Array(0) });
    await frames(4);
    await settle(nodes!);
    checks.push({
        name: tag("empty (count=0)"),
        pass: nodes?.snapshot != null,
        detail: nodes?.snapshot ? "built without device error" : "no snapshot",
    });

    // tree depth vs the restart-trail's provable coverage — a deeper tree means a malformed build.
    const deep = await pump(coincident(4096, 0xdd01));
    const depth = deep ? treeMaxDepth(deep) : -1;
    checks.push({
        name: tag("tree-depth"),
        pass: depth >= 0 && depth < BVH_TRAIL_LEVELS,
        detail: deep ? `depth ${depth} (trail covers ${BVH_TRAIL_LEVELS})` : "no snapshot",
    });

    // coherence guard: rebuild an adversarial scene from a fresh permutation each pass — a stale
    // cross-workgroup read surfaces as a broken union here, where a single build per scene can't.
    const reps = 16;
    for (const { name, prims } of [
        { name: "coincident-4096", prims: coincident(4096, 0xdd01) },
        { name: "clumps-degenerate", prims: clumps(64, 32, 0xc1c1, true) },
    ]) {
        let detail = "";
        for (let r = 0; r < reps && detail === ""; r++) {
            const sh = shuffledPrims(prims, 0x51a7 + r);
            detail = gateBuild(await pump(sh), sh, 0x9911 ^ r);
        }
        checks.push({
            name: tag(`coherence ${name}`),
            pass: detail === "",
            detail: detail || `${reps}× reshuffled rebuilds ok`,
        });
    }

    // --- traverse: closest-hit on every fixture, any-hit on the larger ones (the occlusion path) ---
    for (const { name, prims } of allFixtures()) {
        const batch = rays(sceneBounds(prims), 64, 0x5eed ^ prims.count);
        const detail = gateClosest(await retrace(prims, batch, 0), prims, batch);
        checks.push({
            name: tag(`closest ${name}`),
            pass: detail === "",
            detail: detail || `${batch.length} rays ok`,
        });
    }
    for (const { name, prims } of allFixtures().filter((f) => f.prims.count >= 256)) {
        const batch = rays(sceneBounds(prims), 64, 0xace ^ prims.count);
        const detail = gateAny(await retrace(prims, batch, 1), prims, batch);
        checks.push({
            name: tag(`any-hit ${name}`),
            pass: detail === "",
            detail: detail || `${batch.length} rays ok`,
        });
    }

    return checks;
}

const scenario: Scenario = {
    name: "accel",
    params: [
        { key: "count", type: "number", default: 256, min: 1, step: 64, rebuild: true },
        // selects which arm the profiler times (false = the subgroup-free LDS kernels); the assert
        // gates both arms regardless. `cap` sizes the builder (set == count for a fair full-capacity scale).
        { key: "subgroups", type: "bool", default: true, rebuild: true },
        { key: "cap", type: "number", default: MAX_PRIMS, min: 1, rebuild: true },
        {
            key: "dist",
            type: "select",
            default: "uniform",
            options: [
                "uniform",
                "clustered",
                "slivers",
                "coplanar",
                "coincident",
                "clumps",
                "giant",
            ],
            rebuild: true,
        },
        { key: "seed", type: "number", default: 1, rebuild: true },
        { key: "viz", type: "bool", default: true },
        { key: "curve", type: "bool", default: true },
        { key: "tree", type: "bool", default: true },
        { key: "rays", type: "bool", default: true },
        { key: "mode", type: "select", default: "closest", options: ["closest", "any"] },
    ],

    async build(_canvas, p: Params) {
        params = p;

        const { state, dispose } = await run({
            defaults: false,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                MirrorPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                SearPlugin,
                GlazePlugin,
                LinesPlugin,
                accelPlugin(),
            ],
        });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);

        // orbit the scene's origin from a 3/4 angle. The distance bounds bracket the start so the first
        // zoom doesn't clamp-snap past the default 30.
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 5);
        Orbit.pitch.set(cam, Math.PI / 7);
        Orbit.distance.set(cam, 44);
        Orbit.minDistance.set(cam, 4);
        Orbit.maxDistance.set(cam, 400);

        await frames(2);
        nodes = mirror(bvh!.nodes);
        hits = mirror(tracer!.hits);
        keysMirror = mirror(rs!.keys);
        payloadMirror = mirror(rs!.payload);
        await frames(3);
        // warm allocates raw GPU structures outside State — destroy them on teardown (before the State
        // dispose, while the device is live) so an HMR reload doesn't leak a builder + tracer + sort set.
        return {
            state,
            dispose() {
                bvh?.destroy();
                tracer?.destroy();
                rs?.destroy();
                countBuf?.destroy();
                dispose();
            },
        };
    },

    // The three oracle gates of the pipeline, swept through the live structures via Mirror readback,
    // run once per builder arm. The subgroup→LDS fallback is a documented exception to the
    // no-conditional-path rule, justified only if the LDS arm can't bitrot — so one `bun bench
    // --scenario accel` run gates BOTH arms: the live arm (the one the profiler timed) and a fresh
    // opposite arm stood up on the side, repointing the per-frame pass + mirrors at it. One headless
    // run covers every check the bvh-build / bvh-traverse / sort scenarios ran, on both arms.
    async assert(): Promise<Check[]> {
        const liveSub = (params?.subgroups as boolean) ?? true;
        const cap = Math.max(1, (params?.cap as number) ?? MAX_PRIMS);
        const checks: Check[] = [];

        // live arm — the structures warm built + the mirrors `build` wired, the arm `measure` timed
        checks.push(...(await gate(armName(liveSub))));

        // the other arm — build a fresh set, repoint the per-frame pass (it reads the module-global
        // `bvh`/`tracer`) and the four mirrors at it, gate it, then tear it down and restore the live
        // arm. The subgroup arm needs the feature present; a device without it (a real WebKit tier)
        // can only run LDS, so skip the subgroup arm there rather than fail the build.
        const otherSub = !liveSub;
        if (otherSub && !Compute.device.features.has("subgroups")) {
            checks.push({
                name: `${armName(otherSub)} arm`,
                pass: true,
                detail: "skipped — device has no subgroups",
            });
        } else {
            const other = await makeArm(otherSub, cap);
            const saved = { bvh, tracer, rs, countBuf, nodes, hits, keysMirror, payloadMirror };
            bvh = other.bvh;
            tracer = other.tracer;
            rs = other.rs;
            countBuf = other.countBuf;
            nodes = mirror(other.bvh.nodes);
            hits = mirror(other.tracer.hits);
            keysMirror = mirror(other.rs.keys);
            payloadMirror = mirror(other.rs.payload);
            await frames(3); // let the per-frame pass build the new arm + its mirrors start resolving
            checks.push(...(await gate(armName(otherSub))));
            // stop the opposite-arm mirrors, restore the live globals, then free the opposite arm —
            // no `await` between restore and dispose, so the frame loop can't build a freed buffer
            nodes.dispose();
            hits.dispose();
            keysMirror.dispose();
            payloadMirror.dispose();
            bvh = saved.bvh;
            tracer = saved.tracer;
            rs = saved.rs;
            countBuf = saved.countBuf;
            nodes = saved.nodes;
            hits = saved.hits;
            keysMirror = saved.keysMirror;
            payloadMirror = saved.payloadMirror;
            disposeArm(other);
        }

        // restore the live scene for the HUD + viz (gate left an adversarial scene + a locked trace mode)
        assertMode = null;
        buildMode = "build";
        if (scene) {
            setScene(scene);
            uploadRays(vizRays);
            await pump(scene);
        }
        return checks;
    },

    live(): string {
        const n = scene?.count ?? 0;
        return [
            "accel",
            `dist     ${params?.dist ?? "uniform"}`,
            `prims    ${n}`,
            `nodes    ${Math.max(1, 2 * n - 1)}`,
            `rays     ${vizRays.length} (${params?.mode ?? "closest"})`,
        ].join("\n");
    },
};

register(scenario);
