// Floor-vanish instrument for the tumble sample host (spec tumble-inline stage 6b / F1). Two surfaces, one
// shared readback core:
//
//   • the passive page hooks — `window.__tumbleProbe` (a drawArgs readback + a NaN/Inf scan over body poses
//     AND Part instance transforms), `window.__tumbleAim` (a grabbable dynamic body's client pixel), and
//     `window.__tumbleInject` (a self-test poison for the red-first proof). Zero per-frame cost: they run
//     only when a driver calls them. The trusted-input Playwright driver (`scripts/tumble-repro-driver.mjs`)
//     reads them over CDP to check the invariant after each mouse burst.
//   • the always-available auto-dump watcher (`watch=1`) — a per-frame system that runs the same readback and,
//     the moment any draw pair's drawn count drops below the distinct-mesh count OR any pose/transform goes
//     non-finite, latches a full dump onto `window.__tumbleFloorDump` and console.error's it. So a by-hand
//     native reproduction leaves the trace behind with no Playwright in the loop.
//
// The watcher is opt-in behind `watch=1`, not always-on, for one reason: it does a per-frame GPU→CPU drawArgs
// readback, which stalls the pipeline and would skew the profiler frame timing that `bun run bench:tumble`
// gates on across every twin. Passive hooks cost nothing idle, so they install unconditionally.
//
// Diagnostic only — nothing here feeds the gold oracle, and it reads world/Part state, never mutates it
// (except `__tumbleInject`, which exists solely to prove the watcher fires and is never called in a real run).

import {
    Camera,
    Color,
    Compute,
    Material,
    Orbit,
    Part,
    type State,
    Transform,
} from "@dylanebert/shallot";
import { Parts } from "@dylanebert/shallot/part/core";
import { screenToRay } from "@dylanebert/shallot/physics/core";
import {
    CLUSTER_COUNT,
    CULL_VOLUME_FLOATS,
    clusterView,
    LightCull,
    Lighting,
    Meshes,
    Render,
    Surfaces,
    Views,
} from "@dylanebert/shallot/render/core";
import {
    type Body,
    BodyType,
    type DebugDraw,
    defaultDebugDraw,
    type Mesh,
    type World,
    type WorldTransform,
} from "@dylanebert/shallot/tumble/core";
import type { ProbeContext } from "./tumble-probe";
import { cameraPose, worldToScreen } from "./tumble-project";

const FAR = 1000; // pick-ray length, matching the host's grab cast
const H = 1e9; // an effectively-unbounded draw box so a far body still counts (TOTAL_DRAW_BOUNDS)
const MAX_HITS = 8; // cap the non-finite list — the first few pin the source

/** one non-finite value found in the scan: where it lives and the offending numbers. */
export interface NonFiniteHit {
    source: "body-pos" | "body-rot" | "part-pos" | "part-rot" | "part-scale";
    index: number;
    values: number[];
}

/** the invariant break the driver + the auto-dump watcher both key on. `static-missing` ranks first — it's
 *  the root world-truth loss (a registered static gone from the broadphase, the confirmed floor-vanish cause);
 *  `non-finite` next (a poisoned transform); `draw-drop` the downstream symptom (a pair stops rasterizing). */
export interface Breach {
    kind: "static-missing" | "non-finite" | "draw-drop";
    drawing: number;
    meshes: number;
    nonFinite: NonFiniteHit[];
    /** for `static-missing`: the stable ids of the valid registered statics absent from the broadphase. */
    missing?: number[];
}

/** one instrument sample: the render-level draw count, the CPU derivation count, and the non-finite scan. */
export interface Snapshot {
    drawing: number;
    meshes: number;
    bodyCount: number;
    partCount: number;
    nonFinite: NonFiniteHit[];
}

/**
 * The pure break rule, factored out so a unit test can pin it without a GPU. A non-finite hit is a break
 * regardless of the draw count; absent that, a drawing count below the distinct-mesh count is the draw-drop
 * break (`drawing < 0` means the readback couldn't run — not a break). Returns null when the frame is clean.
 */
export function detectBreach(
    drawing: number,
    meshes: number,
    nonFinite: NonFiniteHit[],
): Breach | null {
    if (nonFinite.length > 0) return { kind: "non-finite", drawing, meshes, nonFinite };
    if (drawing >= 0 && drawing < meshes) return { kind: "draw-drop", drawing, meshes, nonFinite };
    return null;
}

/** one registered static body's state this frame: its stable id, whether it's still a live body, and whether
 *  it's present in the live broadphase query. */
export interface StaticState {
    id: number;
    valid: boolean;
    present: boolean;
}

/**
 * The registered-static invariant break (spec 6b/F3′). The floor-vanish was a broadphase-tree corruption that
 * dropped the static bodies from `overlapAABB` / `world.draw` — so a detector derived from those (the mesh
 * count {@link detectBreach} compares against) is structurally blind to it: the drawn count AND the derived
 * count both read the corrupted query, so they fell together and looked consistent. This checks the live
 * broadphase against the static set captured at scenario build instead — the one source `world.draw` can't
 * corrupt. A registered static that is still a live body (`valid`) but absent from the query is the corruption.
 * One that is `!valid` was legitimately destroyed (`--inject statics`, a re-typed body) and drops from the set
 * — that live-handle distinction is exactly what keeps a real destroy from reading as a corruption breach.
 * @returns the ids of the valid-but-missing statics, or null when the registered set is intact.
 */
export function staticBreach(statics: StaticState[]): number[] | null {
    const missing = statics.filter((s) => s.valid && !s.present).map((s) => s.id);
    return missing.length > 0 ? missing : null;
}

// distinct solid geometries the pack must draw, keyed as the solid layer dedupes them, read UNBOUNDED.
function distinctMeshes(world: World): number {
    const keys = new Set<object>();
    const dd: DebugDraw = {
        ...defaultDebugDraw(),
        drawingBounds: { lowerBound: { x: -H, y: -H, z: -H }, upperBound: { x: H, y: H, z: H } },
        drawShapes: true,
        drawSolidSphere: (_xf, sphere) => keys.add(sphere),
        drawSolidCapsule: (_xf, cap) => keys.add(cap),
        drawSolidHull: (_xf, hull) => keys.add(hull),
        drawSolidMesh: (_xf, mesh: Mesh) => keys.add(mesh.data),
    };
    world.draw(dd);
    return keys.size;
}

// the frame's peak dynamic-body linear/angular speed + any non-finite velocity. Velocity is the grab-spring's
// actual output and is NOT covered by the pose non-finite scan (a body can carry an unbounded velocity for a
// step while its pose is still finite — exactly the frame that fattens its broadphase AABB past what the tree
// survives). The per-frame watcher tracks the running peak so a transient spike (gone by the breach frame) is
// still captured. `nonFiniteVel` counts NaN/Inf velocities the pose scan misses.
function frameVel(world: World): { maxVel: number; maxAngVel: number; nonFiniteVel: number } {
    const box = {
        lowerBound: { x: -H, y: -H, z: -H },
        upperBound: { x: H, y: H, z: H },
    };
    let maxVel = 0;
    let maxAngVel = 0;
    let nonFiniteVel = 0;
    const seen = new Set<Body>();
    world.overlapAABB(box, (shape) => {
        const b = shape.getBody();
        if (b.getType() !== BodyType.Dynamic || seen.has(b)) return true;
        seen.add(b);
        const v = b.getLinearVelocity();
        const a = b.getAngularVelocity();
        const vm = Math.hypot(v.x, v.y, v.z);
        const am = Math.hypot(a.x, a.y, a.z);
        if (!(Number.isFinite(vm) && Number.isFinite(am))) nonFiniteVel++;
        else {
            if (vm > maxVel) maxVel = vm;
            if (am > maxAngVel) maxAngVel = am;
        }
        return true;
    });
    return { maxVel, maxAngVel, nonFiniteVel };
}

// scan every world solid pose (position + orientation) for a non-finite component — the physics-side signal.
function scanBodies(world: World, hits: NonFiniteHit[]): number {
    let n = 0;
    const check = (xf: WorldTransform): void => {
        const p = xf.p;
        const q = xf.q;
        if (
            hits.length < MAX_HITS &&
            !(Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z))
        ) {
            hits.push({ source: "body-pos", index: n, values: [p.x, p.y, p.z] });
        }
        if (
            hits.length < MAX_HITS &&
            !(
                Number.isFinite(q.v.x) &&
                Number.isFinite(q.v.y) &&
                Number.isFinite(q.v.z) &&
                Number.isFinite(q.s)
            )
        ) {
            hits.push({ source: "body-rot", index: n, values: [q.v.x, q.v.y, q.v.z, q.s] });
        }
        n++;
    };
    const dd: DebugDraw = {
        ...defaultDebugDraw(),
        drawingBounds: { lowerBound: { x: -H, y: -H, z: -H }, upperBound: { x: H, y: H, z: H } },
        drawShapes: true,
        drawSolidSphere: (xf) => check(xf),
        drawSolidCapsule: (xf) => check(xf),
        drawSolidHull: (xf) => check(xf),
        drawSolidMesh: (xf) => check(xf),
    };
    world.draw(dd);
    return n;
}

// scan every Part instance's transform (what the pack reads into the shared scan) for a non-finite component.
function scanParts(state: State, hits: NonFiniteHit[]): number {
    let n = 0;
    for (const eid of state.query([Part])) {
        const px = Transform.pos.x.get(eid);
        const py = Transform.pos.y.get(eid);
        const pz = Transform.pos.z.get(eid);
        if (
            hits.length < MAX_HITS &&
            !(Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz))
        ) {
            hits.push({ source: "part-pos", index: eid, values: [px, py, pz] });
        }
        const rx = Transform.rot.x.get(eid);
        const ry = Transform.rot.y.get(eid);
        const rz = Transform.rot.z.get(eid);
        const rw = Transform.rot.w.get(eid);
        if (
            hits.length < MAX_HITS &&
            !(
                Number.isFinite(rx) &&
                Number.isFinite(ry) &&
                Number.isFinite(rz) &&
                Number.isFinite(rw)
            )
        ) {
            hits.push({ source: "part-rot", index: eid, values: [rx, ry, rz, rw] });
        }
        const sx = Transform.scale.x.get(eid);
        const sy = Transform.scale.y.get(eid);
        const sz = Transform.scale.z.get(eid);
        if (
            hits.length < MAX_HITS &&
            !(Number.isFinite(sx) && Number.isFinite(sy) && Number.isFinite(sz))
        ) {
            hits.push({ source: "part-scale", index: eid, values: [sx, sy, sz] });
        }
        n++;
    }
    return n;
}

// count the Part pack's drawing pairs from a one-shot `Parts.drawArgs` readback (lane 1 = instanceCount of
// the 5-u32 DrawIndexedIndirect stride). One shading camera, no shadow atlas, so a whole-buffer nonzero
// tally is the drawing-pair count. `-1` when no device / pack output exists yet.
async function drawingPairs(): Promise<number> {
    const device = Compute.device;
    const src = Parts.drawArgs;
    if (!device || !src) return -1;
    const staging = device.createBuffer({
        size: src.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, staging, 0, src.size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const args = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    let drawing = 0;
    for (let i = 1; i < args.length; i += 5) if (args[i] > 0) drawing++;
    return drawing;
}

/** the full instrument sample — the draw count, the derivation count, and the non-finite scan of both the
 *  world poses and the Part instance transforms. Async for the GPU readback. */
async function snapshot(ctx: ProbeContext): Promise<Snapshot> {
    const nonFinite: NonFiniteHit[] = [];
    const bodyCount = scanBodies(ctx.world, nonFinite);
    const partCount = scanParts(ctx.state, nonFinite);
    const meshes = distinctMeshes(ctx.world);
    const drawing = await drawingPairs();
    return { drawing, meshes, bodyCount, partCount, nonFinite };
}

// ── F2 deep probe ──────────────────────────────────────────────────────────────────────────────────────
// The layer-bisection instrument: it answers "did the render boundary lose truth the CPU had, or was truth
// already gone upstream?" by replaying the pack's own `visible()` cull on the CPU — against the exact frustum
// `BeginFrameSystem` packed (`Render.cullVolumeStaging`) and the exact CPU transforms — and comparing that
// prediction to the GPU `drawArgs` the pack actually wrote. A pair the CPU predicts visible but the GPU drew
// zero = truth lost AT the GPU pack (a sub-frame poison the per-frame reconcile already overwrote). A pair
// the CPU also predicts culled = truth lost UPSTREAM (a real transform / frustum the pack culled correctly).

/** one (surface, mesh) pair's slot-0 draw state: what the GPU pack wrote vs what a CPU replay of `visible()`
 *  predicts. `gpuCount < cpuVisible` localizes the loss to the GPU pack; equal (both 0) localizes it upstream. */
export interface PairState {
    pair: number;
    mesh: string;
    surface: string;
    gpuCount: number;
    gpuFirst: number;
    cpuVisible: number;
}

/** the deep readback: the slot-0 frustum finiteness, the per-pair GPU-vs-CPU cull comparison, and any body
 *  the grab flung far (|pos| > FLUNG_M) — the three signals that separate a render-boundary loss from an
 *  upstream one. */
export interface DeepReadback {
    viewCount: number;
    surfaceCount: number;
    meshCount: number;
    pairCount: number;
    viewDim: number;
    frustumFinite: boolean;
    frustumExtreme: number; // max |plane component| across all slots — a huge-but-finite frustum poison
    pairs: PairState[];
    flung: { eid: number; mesh: string; pos: number[] }[];
    mismatch: PairState[]; // pairs where gpuCount < cpuVisible — the render-boundary loss, if any
}

const FLUNG_M = 50; // a body past this is "flung" — the grab can't legitimately carry one here (camera depth)

// CPU twin of the pack's `xformPoint` + `visible`: rotate the local bound center by the instance quaternion,
// scale, translate, then test the world sphere against the six frustum planes exactly as the WGSL does.
function rotate(
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    x: number,
    y: number,
    z: number,
): number[] {
    const tx = 2 * (qy * z - qz * y);
    const ty = 2 * (qz * x - qx * z);
    const tz = 2 * (qx * y - qy * x);
    return [
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    ];
}

function culled(planes: number[][], cx: number, cy: number, cz: number, r: number): boolean {
    for (const p of planes) {
        if (p[0] * cx + p[1] * cy + p[2] * cz + p[3] < -r) return true;
    }
    return false;
}

// read slot 0's six frustum planes out of the CPU staging the pack's cull reads (post-`BeginFrameSystem`).
function slotPlanes(slot: number): number[][] {
    const staging = Render.cullVolumeStaging;
    const base = slot * CULL_VOLUME_FLOATS + 4; // +4: skip the header vec4 (the tag)
    const planes: number[][] = [];
    for (let i = 0; i < 6; i++) {
        const o = base + i * 4;
        planes.push([staging[o], staging[o + 1], staging[o + 2], staging[o + 3]]);
    }
    return planes;
}

/** the full deep readback. Async for the `drawArgs` GPU→CPU map. Diagnostic only — never mutates. */
export async function deepProbe(ctx: ProbeContext): Promise<DeepReadback> {
    const surfaceCount = Surfaces.size;
    const meshCount = Meshes.size;
    const pairCount = surfaceCount * meshCount;
    const viewCount = Render.viewCount;

    // frustum finiteness + the extreme magnitude, across every packed slot (a poisoned camera would show here)
    const staging = Render.cullVolumeStaging;
    let frustumFinite = true;
    let frustumExtreme = 0;
    for (let s = 0; s < viewCount; s++) {
        const base = s * CULL_VOLUME_FLOATS + 4;
        for (let i = 0; i < 24; i++) {
            const v = staging[base + i];
            if (!Number.isFinite(v)) frustumFinite = false;
            else frustumExtreme = Math.max(frustumExtreme, Math.abs(v));
        }
    }

    // per-mesh local bound sphere, keyed by mesh id (the pack's meshBounds; a boundless mesh never culls)
    const bounds = new Map<number, [number, number, number, number] | null>();
    for (const m of Meshes) bounds.set(Meshes.id(m.name)!, m.bounds ?? null);

    // CPU cull replay against slot 0 (the shading camera) — tally survivors per pair
    const planes0 = slotPlanes(0);
    const cpu = new Uint32Array(pairCount);
    const flung: { eid: number; mesh: string; pos: number[] }[] = [];
    for (const eid of ctx.state.query([Part])) {
        const mid = Part.mesh.get(eid);
        const sid = Part.surface.get(eid);
        if (sid >= surfaceCount) continue;
        const pair = mid * surfaceCount + sid;
        if (pair >= pairCount) continue;
        const px = Transform.pos.x.get(eid);
        const py = Transform.pos.y.get(eid);
        const pz = Transform.pos.z.get(eid);
        if (Math.hypot(px, py, pz) > FLUNG_M) {
            flung.push({ eid, mesh: Meshes.name(mid) ?? `#${mid}`, pos: [px, py, pz] });
        }
        const b = bounds.get(mid);
        let visible = true;
        if (b) {
            const sx = Transform.scale.x.get(eid);
            const sy = Transform.scale.y.get(eid);
            const sz = Transform.scale.z.get(eid);
            const [wx, wy, wz] = rotate(
                Transform.rot.x.get(eid),
                Transform.rot.y.get(eid),
                Transform.rot.z.get(eid),
                Transform.rot.w.get(eid),
                b[0] * sx,
                b[1] * sy,
                b[2] * sz,
            );
            const r = b[3] * Math.max(Math.abs(sx), Math.abs(sy), Math.abs(sz));
            visible = !culled(planes0, px + wx, py + wy, pz + wz, r);
        }
        if (visible) cpu[pair]++;
    }

    // GPU drawArgs readback — slot 0's records (slot-major: slot * pairCount + pair, stride 5 u32)
    const device = Compute.device;
    const src = Parts.drawArgs;
    let gpu: Uint32Array | null = null;
    let viewDim = 0;
    if (device && src) {
        const st = device.createBuffer({
            size: src.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(src, 0, st, 0, src.size);
        device.queue.submit([enc.finish()]);
        await st.mapAsync(GPUMapMode.READ);
        gpu = new Uint32Array(st.getMappedRange().slice(0));
        st.unmap();
        st.destroy();
        viewDim = pairCount > 0 ? Math.floor(gpu.length / 5 / pairCount) : 0;
    }

    const pairs: PairState[] = [];
    const mismatch: PairState[] = [];
    for (let pair = 0; pair < pairCount; pair++) {
        const gpuCount = gpu ? gpu[pair * 5 + 1] : -1; // slot 0 record
        const gpuFirst = gpu ? gpu[pair * 5 + 4] : -1;
        const cpuVisible = cpu[pair];
        if (cpuVisible === 0 && gpuCount <= 0) continue;
        const mid = Math.floor(pair / surfaceCount);
        const sid = pair % surfaceCount;
        const rec: PairState = {
            pair,
            mesh: Meshes.name(mid) ?? `#${mid}`,
            surface: Surfaces.name(sid) ?? `#${sid}`,
            gpuCount,
            gpuFirst,
            cpuVisible,
        };
        pairs.push(rec);
        if (gpuCount >= 0 && gpuCount < cpuVisible) mismatch.push(rec);
    }

    return {
        viewCount,
        surfaceCount,
        meshCount,
        pairCount,
        viewDim,
        frustumFinite,
        frustumExtreme,
        pairs,
        flung,
        mismatch,
    };
}

// the forward-most grabbable dynamic body + its client pixel, aimed top-down (a top plank is unobstructed).
// Both `aim` (the driver's mouse target) and `inject` (the red-first poison site) key on this one cast so the
// self-test poisons exactly the body a real grab would land on.
function findTarget(ctx: ProbeContext): { body: Body; clientX: number; clientY: number } | null {
    const { world, cam, canvas } = ctx;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w < 1 || h < 1) return null;
    const fovDeg = Camera.fov.get(cam);
    const near = Camera.near.get(cam);
    const camPos: [number, number, number] = [
        Transform.pos.x.get(cam),
        Transform.pos.y.get(cam),
        Transform.pos.z.get(cam),
    ];
    const camQuat: [number, number, number, number] = [
        Transform.rot.x.get(cam),
        Transform.rot.y.get(cam),
        Transform.rot.z.get(cam),
        Transform.rot.w.get(cam),
    ];
    const candidates = poses(world)
        .map((p) => ({ p, s: worldToScreen(camPos, camQuat, fovDeg, w, h, p) }))
        .filter((c) => c.s.front && c.s.x >= 0 && c.s.x <= w && c.s.y >= 0 && c.s.y <= h)
        .sort((a, b) => b.p.y - a.p.y);
    for (const c of candidates) {
        const ray = screenToRay(c.s.x, c.s.y, w, h, fovDeg, near, camPos, camQuat);
        const r = world.castRayClosest(
            { x: ray.origin[0], y: ray.origin[1], z: ray.origin[2] },
            { x: ray.dir[0] * FAR, y: ray.dir[1] * FAR, z: ray.dir[2] * FAR },
        );
        if (r.hit && r.shape && r.shape.getBody().getType() === BodyType.Dynamic) {
            return {
                body: r.shape.getBody(),
                clientX: rect.left + c.s.x,
                clientY: rect.top + c.s.y,
            };
        }
    }
    return null;
}

// every solid body pose the walk draws (positions only) — the aim projects these to find a grab pixel.
function poses(world: World): { x: number; y: number; z: number }[] {
    const out: { x: number; y: number; z: number }[] = [];
    const push = (xf: WorldTransform): void => {
        out.push({ x: xf.p.x, y: xf.p.y, z: xf.p.z });
    };
    const dd: DebugDraw = {
        ...defaultDebugDraw(),
        drawingBounds: { lowerBound: { x: -H, y: -H, z: -H }, upperBound: { x: H, y: H, z: H } },
        drawShapes: true,
        drawSolidSphere: (xf) => push(xf),
        drawSolidCapsule: (xf) => push(xf),
        drawSolidHull: (xf) => push(xf),
        drawSolidMesh: (xf) => push(xf),
    };
    world.draw(dd);
    return out;
}

// poison one live WORLD body pose, the self-test that proves the watcher fires (red-first). Injecting into the
// world (not a Part transform, which the per-frame solid reconcile immediately overwrites with the finite
// world pose) is both persistent AND faithful to the hypothesis: a non-finite body pose is exactly the
// grab-spring explosion's outcome, and it flows into the Part transform on the next reconcile — so the scan
// catches it on both sides. Never called in a real run; the driver calls it once through `__tumbleInject`.
const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };
function inject(ctx: ProbeContext, kind: string): boolean {
    const target = findTarget(ctx);
    if (!target) return false;
    if (kind === "nan") {
        target.body.setTransform({ x: Number.NaN, y: Number.NaN, z: Number.NaN }, IDENT);
        return true;
    }
    if (kind === "inf") {
        target.body.setTransform({ x: Number.POSITIVE_INFINITY, y: 6, z: 0 }, IDENT);
        return true;
    }
    // F2 synthetic: the traced mechanism — a huge-but-FINITE displacement (what the grab-spring explosion
    // actually produces) pushing every dynamic body out of the camera frustum. It reproduces the observed
    // multi-pair draw-drop deterministically: each dynamic pair culls to instanceCount 0 (the CPU replay
    // predicts the same → mismatch 0, a correct cull), while the static ground's ~70-unit bound sphere at
    // the origin stays visible. Proves the loss is displacement→cull, not a poisoned transform/scan/frustum.
    if (kind === "far") {
        const box = {
            lowerBound: { x: -H, y: -H, z: -H },
            upperBound: { x: H, y: H, z: H },
        };
        const dynamics: Body[] = [];
        ctx.world.overlapAABB(box, (shape) => {
            const body = shape.getBody();
            if (body.getType() === BodyType.Dynamic) dynamics.push(body);
            return true;
        });
        let n = 0;
        for (const body of dynamics) {
            // spread them so no two overlap (a stacked spawn would crash the solver), all well past the frustum
            body.setTransform({ x: 500 + n * 5, y: 500, z: 500 }, IDENT);
            n++;
        }
        return n > 0;
    }
    // F2′ synthetic: the TRACED state — remove the static bodies (ground + posts) from the world directly, no
    // grab, no steep pitch. If the ground then shows the same near-black pixel signature the grab breach did,
    // it proves the chain "static bodies absent from the world → their solids drop → ground blacks out" is the
    // whole mechanism, and the render/shading/shadow layers are faithful (they draw only what the world reports).
    if (kind === "statics") {
        const box = {
            lowerBound: { x: -H, y: -H, z: -H },
            upperBound: { x: H, y: H, z: H },
        };
        const statics: Body[] = [];
        const seen = new Set<Body>();
        ctx.world.overlapAABB(box, (shape) => {
            const b = shape.getBody();
            if (b.getType() === BodyType.Static && !seen.has(b)) {
                seen.add(b);
                statics.push(b);
            }
            return true;
        });
        for (const b of statics) b.destroy();
        return statics.length > 0;
    }
    // F2′ cross-tree probe: does a huge dynamic-body velocity (the grab-spring's actual output — a finite pose
    // with an unbounded velocity the pose scan can't see) drop the STATIC bodies? The broadphase keeps a
    // separate tree per BodyType (`bp.trees[type]`), so a dynamic body's velocity-fattened AABB should NOT
    // reach the static tree — a static drop here would show the trees are not actually isolated.
    if (kind === "whipvel") {
        const box = {
            lowerBound: { x: -H, y: -H, z: -H },
            upperBound: { x: H, y: H, z: H },
        };
        let n = 0;
        ctx.world.overlapAABB(box, (shape) => {
            const body = shape.getBody();
            if (body.getType() === BodyType.Dynamic) {
                body.setLinearVelocity({ x: 1e7, y: 1e7, z: 1e7 });
                n++;
            }
            return true;
        });
        return n > 0;
    }
    return false;
}

// ── F2′ render-layer readback ────────────────────────────────────────────────────────────────────────────
// The pixel repro (F1′) proved the ground blacks out with drawArgs GREEN — a render/shading-layer defect. This
// hook reads the render inputs the FS shades from, so the driver can capture them at baseline vs breach and
// bisect WHICH layer lost truth: the shared Lighting UBO (its CPU staging mirror), the camera/view + froxel
// cluster params, the ground vs a deck plank's own per-instance Part data (transform / color / material / mesh /
// surface — the ground's shading is view-independent, so a black ground implicates one of these or a uniform),
// and the clustered-light buffers (lightGrid + the compacted pointLights, both COPY_SRC) — a stale/garbage
// froxel entry pointing at a non-finite light would poison litPbr's point loop. Diagnostic only, never mutates.

interface PartState {
    eid: number;
    pos: number[];
    rot: number[];
    scale: number[];
    color: number[];
    material: number[];
    mesh: string | number;
    surface: string | number;
}

function partState(eid: number): PartState {
    return {
        eid,
        pos: [Transform.pos.x.get(eid), Transform.pos.y.get(eid), Transform.pos.z.get(eid)],
        rot: [
            Transform.rot.x.get(eid),
            Transform.rot.y.get(eid),
            Transform.rot.z.get(eid),
            Transform.rot.w.get(eid),
        ],
        scale: [Transform.scale.x.get(eid), Transform.scale.y.get(eid), Transform.scale.z.get(eid)],
        color: [
            Color.rgba.x.get(eid),
            Color.rgba.y.get(eid),
            Color.rgba.z.get(eid),
            Color.rgba.w.get(eid),
        ],
        material: [
            Material.params.x.get(eid),
            Material.params.y.get(eid),
            Material.params.z.get(eid),
            Material.params.w.get(eid),
        ],
        mesh: Meshes.name(Part.mesh.get(eid)) ?? Part.mesh.get(eid),
        surface: Surfaces.name(Part.surface.get(eid)) ?? Part.surface.get(eid),
    };
}

// read a COPY_SRC GPU buffer back into a fresh typed array (diagnostic only). Null when no device.
async function readU32(src: GPUBuffer | null, bytes: number): Promise<Uint32Array | null> {
    const device = Compute.device;
    if (!device || !src) return null;
    const size = Math.min(bytes, src.size);
    const staging = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
}

/** the render-layer snapshot the F2′ driver captures at baseline vs breach — the inputs the color FS shades
 *  from, so a bisection can name the layer that lost truth. Async for the light-buffer readbacks. */
export interface RenderState {
    lighting: { ambient: number[]; sunDir: number[]; sunColor: number[] };
    camera: {
        pos: number[];
        quat: number[];
        near: number;
        far: number;
        fov: number;
        mode: number;
        antialias: number;
        pitch: number;
        yaw: number;
        distance: number;
    };
    cluster: { perspective: boolean; near: number; far: number };
    viewCount: number;
    partCount: number;
    // world-layer truth (upstream of render): the body scan count, the distinct geometries `world.draw` visits
    // under UNBOUNDED bounds, and every static body's position — so a "ground gone" breach separates a physics
    // destroy (the ground body absent from `statics`) from a draw-walk drop (present in `statics`, absent from
    // `worldGeoms`) from a render-layer drop (present in both, absent from the Part query).
    bodyCount: number;
    worldGeoms: number;
    statics: number[][];
    ground: PartState | null;
    deck: PartState | null;
    // slot-0 froxel occupancy: how many cells carry a nonzero light count, and the max — nonzero with no
    // point lights = a cull-pass poison the litPbr loop would read (garbage light → non-finite radiance)
    grid: { nonzeroCells: number; maxCount: number };
    // the compacted point-light list: any nonzero float (with zero PointLights authored, all-zero is correct;
    // a nonzero here that the froxel loop reaches is the poison source)
    lights: { nonzeroWords: number; firstNonzero: number[] };
}

async function renderState(ctx: ProbeContext): Promise<RenderState> {
    const s = Lighting.staging;
    const cam = ctx.cam;
    const view = Views.get(cam);
    const aspect = view && view.height > 0 ? view.width / view.height : 1;
    const cv = clusterView(cam, aspect);

    let minY = Number.POSITIVE_INFINITY;
    let deckY = Number.NEGATIVE_INFINITY;
    let ground: PartState | null = null;
    let deck: PartState | null = null;
    let partCount = 0;
    for (const eid of ctx.state.query([Part])) {
        partCount++;
        const py = Transform.pos.y.get(eid);
        if (py < minY) {
            minY = py;
            ground = partState(eid);
        }
        if (py > deckY && py < 12) {
            deckY = py;
            deck = partState(eid);
        }
    }

    // slot-0 froxel counts (vec2<u32> per cell, .y = count) + the compacted lights
    const grid = await readU32(LightCull.grid, CLUSTER_COUNT * 8);
    let nonzeroCells = 0;
    let maxCount = 0;
    if (grid) {
        for (let i = 0; i < CLUSTER_COUNT; i++) {
            const c = grid[i * 2 + 1];
            if (c > 0) nonzeroCells++;
            if (c > maxCount) maxCount = c;
        }
    }
    // world-layer truth: the body scan, the unbounded geometry count, and the static-body positions
    const bodyScan: NonFiniteHit[] = [];
    const bodyCount = scanBodies(ctx.world, bodyScan);
    const worldGeoms = distinctMeshes(ctx.world);
    const staticPositions = statics(ctx);

    const lightsU = await readU32(LightCull.lights, 4096);
    let nonzeroWords = 0;
    const firstNonzero: number[] = [];
    if (lightsU) {
        const f = new Float32Array(lightsU.buffer);
        for (let i = 0; i < f.length; i++) {
            if (f[i] !== 0) {
                nonzeroWords++;
                if (firstNonzero.length < 8) firstNonzero.push(f[i]);
            }
        }
    }

    return {
        lighting: {
            ambient: [s[0], s[1], s[2], s[3]],
            sunDir: [s[4], s[5], s[6], s[7]],
            sunColor: [s[8], s[9], s[10]],
        },
        camera: {
            pos: [Transform.pos.x.get(cam), Transform.pos.y.get(cam), Transform.pos.z.get(cam)],
            quat: [
                Transform.rot.x.get(cam),
                Transform.rot.y.get(cam),
                Transform.rot.z.get(cam),
                Transform.rot.w.get(cam),
            ],
            near: Camera.near.get(cam),
            far: Camera.far.get(cam),
            fov: Camera.fov.get(cam),
            mode: Camera.mode.get(cam),
            antialias: Camera.antialias.get(cam),
            pitch: Orbit.pitch.get(cam),
            yaw: Orbit.yaw.get(cam),
            distance: Orbit.distance.get(cam),
        },
        cluster: { perspective: cv.perspective, near: cv.near, far: cv.far },
        viewCount: Render.viewCount,
        partCount,
        bodyCount,
        worldGeoms,
        statics: staticPositions,
        ground,
        deck,
        grid: { nonzeroCells, maxCount },
        lights: { nonzeroWords, firstNonzero },
    };
}

// ── F1′ recipe instrument ────────────────────────────────────────────────────────────────────────────────
// The sustained-downward-drag recipe (spec 6b/F1′) asserts at the layer the user SEES — pixels — because the
// reported symptom (the static ground AND the static end posts going black while a plank is dragged below the
// ground plane) is invisible to every drawArgs-layer check: a shadow/regather/tonemap corruption keeps the
// instanceCounts intact. These hooks give the trusted-input driver what it needs to place reference pixel
// patches over known static surfaces, aim the grab at a central plank, and read the grab anchor's below-ground
// depth. Diagnostic only — reads live state, never mutates.

/** a projected world point in CLIENT pixel coords + whether it's in front of the camera and on-screen. */
export interface ScreenPoint {
    x: number;
    y: number;
    front: boolean;
    inView: boolean;
}

/** project world points to client pixels through the live orbit camera (the pixel-patch driver's placement). */
function projectPoints(ctx: ProbeContext, pts: [number, number, number][]): ScreenPoint[] {
    const rect = ctx.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    const pose = cameraPose(ctx.cam);
    return pts.map(([x, y, z]) => {
        const s = worldToScreen(pose.pos, pose.quat, pose.fovDeg, w, h, { x, y, z });
        return {
            x: rect.left + s.x,
            y: rect.top + s.y,
            front: s.front,
            inView: s.front && s.x >= 0 && s.x <= w && s.y >= 0 && s.y <= h,
        };
    });
}

/** the grab pixel + hit for a world point (the driver aims the central-plank grab through `(0,6,0)`): project
 *  the point, cast the pick ray, and report whether the closest hit is a dynamic body (a grabbable plank). */
function aimAt(
    ctx: ProbeContext,
    wx: number,
    wy: number,
    wz: number,
): {
    clientX: number;
    clientY: number;
    hitDynamic: boolean;
    bodyPos: number[] | null;
    hitPoint: number[] | null;
} | null {
    const rect = ctx.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w < 1 || h < 1) return null;
    const pose = cameraPose(ctx.cam);
    const s = worldToScreen(pose.pos, pose.quat, pose.fovDeg, w, h, { x: wx, y: wy, z: wz });
    if (!s.front || s.x < 0 || s.x > w || s.y < 0 || s.y > h) return null;
    const ray = screenToRay(s.x, s.y, w, h, pose.fovDeg, pose.near, pose.pos, pose.quat);
    const r = ctx.world.castRayClosest(
        { x: ray.origin[0], y: ray.origin[1], z: ray.origin[2] },
        { x: ray.dir[0] * FAR, y: ray.dir[1] * FAR, z: ray.dir[2] * FAR },
    );
    const hitDynamic = !!(r.hit && r.shape && r.shape.getBody().getType() === BodyType.Dynamic);
    let bodyPos: number[] | null = null;
    if (hitDynamic && r.shape) {
        const p = r.shape.getBody().getPosition();
        bodyPos = [p.x, p.y, p.z];
    }
    return {
        clientX: rect.left + s.x,
        clientY: rect.top + s.y,
        hitDynamic,
        bodyPos,
        hitPoint: r.hit ? [r.point.x, r.point.y, r.point.z] : null,
    };
}

/** the static bodies' world positions (the ground + the two end posts) — the driver patches the pixels over
 *  these known-static surfaces, so a black-out there can't be a frustum cull of a flung dynamic body. */
function statics(ctx: ProbeContext): number[][] {
    const box = {
        lowerBound: { x: -H, y: -H, z: -H },
        upperBound: { x: H, y: H, z: H },
    };
    const out: number[][] = [];
    const seen = new Set<Body>();
    ctx.world.overlapAABB(box, (shape) => {
        const b = shape.getBody();
        if (b.getType() === BodyType.Static && !seen.has(b)) {
            seen.add(b);
            const p = b.getPosition();
            out.push([p.x, p.y, p.z]);
        }
        return true;
    });
    return out;
}

const WORLD_BOX = { lowerBound: { x: -H, y: -H, z: -H }, upperBound: { x: H, y: H, z: H } };

// The stable body ids of the static bodies present at build — the registered set the static-body invariant
// (`staticBreach`) checks the live broadphase against. Captured from the clean, freshly-built world (the one
// point before any interaction can corrupt the tree), keyed on the engine's index1 body id so a later query's
// fresh Body wrappers still match. This is the source `world.draw` can't corrupt: the corruption drops the
// static bodies FROM the tree, so a set derived from the tree would silently shrink with them.
function captureStatics(ctx: ProbeContext): Body[] {
    const seen = new Set<number>();
    const out: Body[] = [];
    ctx.world.overlapAABB(WORLD_BOX, (shape) => {
        const b = shape.getBody();
        if (b.getType() === BodyType.Static && !seen.has(b.id.index1)) {
            seen.add(b.id.index1);
            out.push(b);
        }
        return true;
    });
    return out;
}

/** the registered static set's live state + any breach — the world-level invariant the drawArgs/mesh checks
 *  are blind to. `present` re-scans the broadphase each call (the corruption's own symptom); `valid` reads the
 *  body handle (a legit destroy → `!valid` → not a breach). */
function staticInvariant(
    ctx: ProbeContext,
    registered: Body[],
): { statics: StaticState[]; missing: number[] | null } {
    const present = new Set<number>();
    ctx.world.overlapAABB(WORLD_BOX, (shape) => {
        const b = shape.getBody();
        if (b.getType() === BodyType.Static) present.add(b.id.index1);
        return true;
    });
    const statics: StaticState[] = registered.map((b) => ({
        id: b.id.index1,
        valid: b.isValid(),
        present: present.has(b.id.index1),
    }));
    return { statics, missing: staticBreach(statics) };
}

/** the live orbit camera pose the driver records on a breach (and reads to steepen pitch before an attempt). */
function camInfo(ctx: ProbeContext): {
    pos: number[];
    quat: number[];
    fovDeg: number;
    yaw: number;
    pitch: number;
    distance: number;
} {
    const pose = cameraPose(ctx.cam);
    return {
        pos: pose.pos,
        quat: pose.quat,
        fovDeg: pose.fovDeg,
        yaw: Orbit.yaw.get(ctx.cam),
        pitch: Orbit.pitch.get(ctx.cam),
        distance: Orbit.distance.get(ctx.cam),
    };
}

/** one sampled pixel patch: which static surface it covers and its mean luminance (0..255). */
export interface PatchLum {
    surface: string;
    lum: number;
}

// a reference patch dimmer than this is discarded — it's not reliably lit, so a "went dark" test on it would
// fire on noise. Ground + posts in the lit scene sit well above this.
const MIN_REF_LUM = 25;
// a patch this far below its reference is "dark-breached"; this far above is "wash-breached" (a NaN colour blow-out).
const DARK_FRACTION = 0.35;
const WASH_MULTIPLE = 2.5;

/**
 * The pure pixel-breach rule (spec 6b/F1′), factored out so a unit test pins it without a GPU. A single patch
 * going dark is a body passing in front of it; a BREACH is EVERY reliably-lit patch of one static surface
 * (≥2 of them) dropping near-black — or all blowing out bright — in the SAME sample. Grouped by surface so a
 * plank occluding one ground patch never trips it, only a whole-surface black-out (the reported symptom) does.
 * `ref`/`sample` are index-aligned; returns the first breached surface or null.
 */
export function pixelBreach(
    ref: PatchLum[],
    sample: PatchLum[],
): { surface: string; kind: "dark" | "wash"; patches: number } | null {
    const surfaces = [...new Set(ref.map((r) => r.surface))];
    for (const s of surfaces) {
        const idx = ref
            .map((_, i) => i)
            .filter((i) => ref[i].surface === s && ref[i].lum >= MIN_REF_LUM);
        if (idx.length < 2) continue;
        if (idx.every((i) => sample[i].lum < DARK_FRACTION * ref[i].lum)) {
            return { surface: s, kind: "dark", patches: idx.length };
        }
        if (idx.every((i) => sample[i].lum > WASH_MULTIPLE * ref[i].lum)) {
            return { surface: s, kind: "wash", patches: idx.length };
        }
    }
    return null;
}

/** the dump the auto-dump watcher latches on the first break — the artifact a by-hand reproduction leaves.
 *  `deep` (F2) carries the layer-bisection readback captured at the break frame. */
export interface FloorDump {
    frame: number;
    breach: Breach;
    snapshot: Snapshot;
    deep?: DeepReadback;
}

declare global {
    interface Window {
        __tumbleProbe?: () => Promise<Snapshot>;
        __tumbleAim?: () => { clientX: number; clientY: number } | null;
        __tumbleInject?: (kind: string) => boolean;
        __tumbleDeep?: () => Promise<DeepReadback>;
        __tumbleFloorDump?: FloorDump;
        // F1′ recipe instrument (spec 6b/F1′)
        __tumbleAimAt?: (wx: number, wy: number, wz: number) => ReturnType<typeof aimAt>;
        __tumbleProject?: (pts: [number, number, number][]) => ScreenPoint[];
        __tumbleStatics?: () => number[][];
        __tumbleCam?: () => ReturnType<typeof camInfo>;
        __tumbleGrab?: () => ReturnType<NonNullable<ProbeContext["grab"]>>;
        // F2′ render-layer readback (spec 6b/F2′)
        __tumbleRender?: () => Promise<RenderState>;
        __tumbleVelPeak?: (reset?: boolean) => {
            maxVel: number;
            maxAngVel: number;
            nonFiniteVel: number;
        };
        // F3′ world-level static-set invariant (spec 6b/F3′)
        __tumbleStaticBreach?: () => { statics: StaticState[]; missing: number[] | null };
    }
}

/**
 * Install the floor-vanish instrument on `ctx`'s live scene. The passive page hooks install always (idle
 * cost is zero — they run only when a driver calls them); the per-frame auto-dump watcher installs only under
 * `watch=1` (its per-frame GPU readback would skew bench timing otherwise). Registers its own teardown.
 */
export function installWatch(ctx: ProbeContext): void {
    const w = window;
    // running peak dynamic-body velocity since the last reset — accumulated by the watcher (watch=1). The grab
    // spike is transient (gone by the breach frame a single-shot readback catches), so the peak is what pins it.
    let velPeak = { maxVel: 0, maxAngVel: 0, nonFiniteVel: 0 };
    w.__tumbleProbe = () => snapshot(ctx);
    w.__tumbleAim = () => {
        const t = findTarget(ctx);
        return t ? { clientX: t.clientX, clientY: t.clientY } : null;
    };
    w.__tumbleInject = (kind: string) => inject(ctx, kind);
    w.__tumbleDeep = () => deepProbe(ctx);
    w.__tumbleAimAt = (wx, wy, wz) => aimAt(ctx, wx, wy, wz);
    w.__tumbleProject = (pts) => projectPoints(ctx, pts);
    w.__tumbleStatics = () => statics(ctx);
    w.__tumbleCam = () => camInfo(ctx);
    w.__tumbleGrab = () => (ctx.grab ? ctx.grab() : null);
    // the registered static set, captured from the clean world at install (post-build, pre-interaction) — the
    // one source the broadphase corruption can't shrink out from under the invariant.
    const registeredStatics = captureStatics(ctx);
    w.__tumbleStaticBreach = () => staticInvariant(ctx, registeredStatics);
    w.__tumbleRender = () => renderState(ctx);
    w.__tumbleVelPeak = (reset?: boolean) => {
        const r = { ...velPeak };
        if (reset) velPeak = { maxVel: 0, maxAngVel: 0, nonFiniteVel: 0 };
        return r;
    };
    ctx.state.onDispose(() => {
        w.__tumbleProbe = undefined;
        w.__tumbleAim = undefined;
        w.__tumbleInject = undefined;
        w.__tumbleDeep = undefined;
        w.__tumbleFloorDump = undefined;
        w.__tumbleAimAt = undefined;
        w.__tumbleProject = undefined;
        w.__tumbleStatics = undefined;
        w.__tumbleCam = undefined;
        w.__tumbleGrab = undefined;
        w.__tumbleRender = undefined;
        w.__tumbleVelPeak = undefined;
        w.__tumbleStaticBreach = undefined;
    });

    if (new URL(location.href).searchParams.get("watch") !== "1") return;

    let frame = 0;
    let inFlight = false;
    let lastDrawing = -1;
    let dumped = false;
    ctx.state.addSystem({
        name: "tumble-floor-watch",
        group: "simulation",
        update() {
            frame++;
            // the CPU non-finite scan runs every frame (cheap); the drawArgs readback rides a single-in-flight
            // slot so no frame ever stalls on the GPU→CPU map (its result lags a frame or two — fine for a break).
            const nonFinite: NonFiniteHit[] = [];
            const bodyCount = scanBodies(ctx.world, nonFinite);
            const partCount = scanParts(ctx.state, nonFinite);
            const meshes = distinctMeshes(ctx.world);
            // track the peak dynamic velocity — the transient grab spike the pose scan can't see (velocity, not
            // pose, is what fattens the broadphase AABB the frame the static bodies drop out)
            const fv = frameVel(ctx.world);
            if (fv.maxVel > velPeak.maxVel) velPeak.maxVel = fv.maxVel;
            if (fv.maxAngVel > velPeak.maxAngVel) velPeak.maxAngVel = fv.maxAngVel;
            velPeak.nonFiniteVel += fv.nonFiniteVel;
            if (!inFlight) {
                inFlight = true;
                drawingPairs()
                    .then((d) => {
                        lastDrawing = d;
                    })
                    .catch(() => {})
                    .finally(() => {
                        inFlight = false;
                    });
            }
            // the world-level static invariant: a valid registered static missing from the broadphase is the
            // corruption, and it's caught HERE even when the drawArgs/mesh counts fell together and looked whole
            // (both read the same corrupted query). Ranks ahead of the drawArgs check — it's the root, not a symptom.
            const missing = staticInvariant(ctx, registeredStatics).missing;
            const breach: Breach | null =
                missing !== null
                    ? { kind: "static-missing", drawing: lastDrawing, meshes, nonFinite, missing }
                    : detectBreach(lastDrawing, meshes, nonFinite);
            if (breach && !dumped) {
                dumped = true;
                const dump: FloorDump = {
                    frame,
                    breach,
                    snapshot: { drawing: lastDrawing, meshes, bodyCount, partCount, nonFinite },
                };
                w.__tumbleFloorDump = dump;
                console.error(
                    `[tumble-floor-watch] BREACH frame ${frame}: ${breach.kind} — drawing ${breach.drawing} of ${breach.meshes} mesh(es), ${nonFinite.length} non-finite ${JSON.stringify(nonFinite.slice(0, 4))}`,
                );
                // capture the layer-bisection readback at the break frame (async, one-shot — the latched dump
                // gains `deep` a frame or two later; the driver reads it after the sequence settles).
                deepProbe(ctx)
                    .then((deep) => {
                        dump.deep = deep;
                        console.error(
                            `[tumble-floor-watch] deep: frustumFinite=${deep.frustumFinite} extreme=${deep.frustumExtreme.toExponential(2)} flung=${deep.flung.length} mismatch=${deep.mismatch.length} ${JSON.stringify(deep.mismatch.slice(0, 4))}`,
                        );
                    })
                    .catch(() => {});
            }
        },
    });
}
