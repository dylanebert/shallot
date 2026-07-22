import {
    AmbientLight,
    Backdrop,
    Body,
    Camera,
    CameraMode,
    Color,
    Compute,
    compose,
    composeTransform,
    Depth,
    DirectionalLight,
    Fog,
    FogPlugin,
    GlazePlugin,
    InputPlugin,
    invert,
    Material,
    type Mirror,
    MirrorPlugin,
    mirror,
    multiply,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    Physics,
    type Plugin,
    PointLight,
    quat,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    Shadow,
    ShapeKind,
    SlabPlugin,
    Spot,
    type State,
    type System,
    Tag,
    Transform,
    TransformsPlugin,
    Tumble,
    TumblePlugin,
    unpackColor,
    Volumetric,
} from "@dylanebert/shallot";
import {
    GltfPlugin,
    loadGltf,
    Profile,
    ProfilePlugin,
    placeGltf,
    placeScene,
    Skin,
    Sky,
    SkyPlugin,
    Textured,
} from "@dylanebert/shallot/extras";
import {
    FOG_FLOATS,
    FOG_INSCATTER_WGSL,
    FOG_MARCH_WGSL,
    FOG_MAX_STEPS,
    FOG_STRUCT_WGSL,
    type FogLight,
    FogSystem,
    fogComposite,
    fogInScatter,
    fogSunInScatter,
    fogTransmittance,
    packFog,
} from "@dylanebert/shallot/fog/core";
import { GlazeSystem } from "@dylanebert/shallot/glaze";
import {
    ALBEDO_NAMES,
    type DecodedGltf,
    decode,
    decodeInWorker,
    gltfCacheStats,
    LiveSkin,
    skinMatrix,
    unionPending,
} from "@dylanebert/shallot/gltf/core";
// Parts.drawArgs is the pack's GPU output — the cull readback reads it through the part/core extension surface
import { Parts } from "@dylanebert/shallot/part/core";
import { qRotate } from "@dylanebert/shallot/physics/core";
import {
    BeginFrameSystem,
    type Binding,
    CLUSTER_COUNT,
    CLUSTER_X,
    CLUSTER_Y,
    CLUSTER_Z,
    Clusters,
    clusterAabb,
    clusterCoord,
    clusterIndex,
    clusterView,
    computeViewProj,
    FRUSTUM_FLOATS,
    frustumPlanes,
    LIGHT_POOL,
    LightCull,
    type Mesh,
    Meshes,
    POINT_LIGHTS_STRUCT_WGSL,
    quantizeMeshes,
    Render,
    Surfaces,
    spotParams,
    VERTEX_FLOATS,
    Views,
} from "@dylanebert/shallot/render/core";
import {
    Backgrounds,
    ColorSystem,
    // the point-shadow allocator — the atlas metadata Mirror pins the caster params + the importance-sized
    // tile rects' invariants; the pooled cull-slot eids pin per-cascade/per-combo survivor counts
    cascadeComboEids,
    cascadeCount,
    LIGHT_EVAL_WGSL,
    pointAtlasSize,
    pointCasters,
    pointComboCount,
    pointComboEids,
} from "@dylanebert/shallot/sear/core";
// the ragdoll pose producer render-interpolates readBody poses at fixedAlpha with the same shortest-arc
// nlerp the tumble compose uses (the tumble/core CPU pose-compose surface)
import { nlerpShortest } from "@dylanebert/shallot/tumble/core";
import {
    OCT_ENCODE_WGSL,
    octDecodeNormal,
    octEncodeNormal,
    packLdrColor,
} from "@dylanebert/shallot/utils/core";
import {
    type Check,
    frames,
    type Params,
    packCounts,
    register,
    type Scenario,
    settle,
} from "../gym";

// render — the forward-pipeline atom. One `mode`-selected scene per pass through the renderer, each
// asserting the slice of the pipeline it exercises. Two coverage mechanisms, by mode:
//
//   structural (mode `cull`): a flat field of boxes you orbit while the per-view frustum cull → indirect-
//   draw spine drops off-screen instances. Mirror-readback oracles pin (1) the cull verdict — survivor
//   counts vs a by-construction frustum rectangle; (2) the cluster AABB grid + clustered light-bucket
//   membership vs a brute-force CPU oracle; (3) the point/spot shadow caster metadata + per-combo cull +
//   the importance-sized atlas tile rects; (4) light-pool overflow degrading loud; (5) slab transport
//   round-trip (CPU write → GPU → readback); (6) the prepass `Tag`/`Depth` lanes appearing independently
//   on one `sear:prepass`.
//
//   shaded look (modes `lit`/`spec`/`spot`/`spotShadow`/`pointShadow`/`acne`/`zfight`): a small lit scene
//   whose HDR scene color a compute probe reads mid-frame (`view.framebuffer`, before present — the one
//   thing the structural metadata gates can't see). The probe pins the shaded result: a caster light
//   reaching a floor (no self-occlusion), the representative-point sphere BRDF widening a soft lamp's
//   highlight, spot-cone confinement, spot/point shadow occlusion, shadow-acne freedom, and the reverse-Z
//   front-slab depth win. No separate clustered-light span check exists because the `cull`-mode
//   light-bucket oracle above is strictly stronger (membership vs the GPU, not just "the pass fired").

// mulberry32 — a fast seeded [0,1) generator. Determinism is the gym contract: a fixed seed reproduces
// the exact same colors, so the scene is byte-identical run to run.
function rng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// one page = one scenario instance, so the camera + readback + scene live in module scope.
let cam = 0;
let mode = "cull";
let params: Params | null = null; // live values — the wave reads `k`, the probe live HUD reads the snapshot

// the core plugin stack shared by every mode (the structural transport plugin is inert in the probe
// modes — `count` is 0 there, so the wave never fires).
const corePlugins = [
    ProfilePlugin,
    SlabPlugin,
    MirrorPlugin,
    TransformsPlugin,
    InputPlugin,
    OrbitPlugin,
    RenderPlugin,
    PartPlugin,
    SearPlugin,
    GlazePlugin,
];

// ============================================================================
// structural mode (`cull`) — the flat field + the Mirror-readback oracles
// ============================================================================

// The scene: a flat square XZ field at y=0 viewed straight down by a top-down camera. Top-down keeps the
// camera on-axis, so the survivor set is the closed-form frustum rectangle — the assert never re-extracts
// the production planes from the view-projection matrix. A live tab orbits off this start freely; the
// assert leaves the pose alone and varies only fov/near/far, which orbit doesn't own.
//
// At pitch π/2 the camera sits at (0, D_CAM, 0) looking -Y, so world X maps to the screen's horizontal
// axis (frustum half-width = depth·tan(fov/2)·aspect) and world Z to the vertical (half-height =
// depth·tan(fov/2), no aspect). Every box sits at one depth (D_CAM), so the side planes carve the field
// into a rectangle while the near/far planes keep all of it or drop all.
const D_CAM = 24; // camera height (orbit distance at pitch π/2)
const SPACING = 4; // grid spacing — coarse enough that a frustum boundary at a gap clears every box
const HALF_N = 3; // field is (2·HALF_N+1)² boxes: x, z ∈ {-12, -8, … , 12}

// A box this far (axial) from a side boundary clears the slanted side plane by more than its radius, so
// the GPU's sphere-vs-plane test and the center-in-rect predicate below always agree (no straddler).
// Perpendicular clearance = axial·cos(halfFov); the binding case is the aspect-widened X plane (halfFov ≲
// 45° for sane aspects), needing axial > 0.866 / cos(45°) ≈ 1.2. 1.5 is comfortable. Don't drop it below
// ~1.3 — wide aspects straddle.
const MARGIN = 1.5;

// box world positions, filled in build() alongside the entities so the assert predicate reads the same
// layout the GPU culls.
let boxes: [number, number, number][] = [];

let drawArgs: Mirror | null = null;
let clusterBuf: Mirror | null = null; // mirror of the cluster AABB grid — the CPU↔GPU oracle gate
let colorBuf: Mirror | null = null; // mirror of the "color" slab — the transport round-trip readback
let lightsBuf: Mirror | null = null; // mirror of the compacted light list (GPU-built)
let gridBuf: Mirror | null = null; // mirror of the per-cluster lightGrid (offset, count)
let idxBuf: Mirror | null = null; // mirror of the flat light index pool (+ counter/overflow header)
let pointBuf: Mirror | null = null; // mirror of sear's PointCaster params ("pointShadows")
let tileBuf: Mirror | null = null; // mirror of the per-(caster, face) atlas tile rects ("pointTileRects")
let lightEids: number[] = [];
let shadowEids: number[] = []; // the lights carrying Shadow — sear's point-shadow casters
let lightPos: [number, number, number][] = [];
let lightRange: number[] = [];
let pairCount = 1;
let pair = 0;
let total = 0;
let count = 0; // filler population size (the transport + pack-load instances)
let gridEids: number[] = []; // the verdict's box grid — the transport round-trip writes these
let fillerEids: number[] = []; // the off-screen filler — the per-frame transport wave rewrites these

function liveCounts(): { visible: number; draws: number } | null {
    if (!drawArgs) return null;
    const counts = packCounts(drawArgs, 0, pairCount);
    if (!counts) return null;
    return { visible: counts[pair], draws: counts.reduce((n, c) => n + (c > 0 ? 1 : 0), 0) };
}

// survivors by construction: a box's center sits inside the frustum (independently derived from
// fov/aspect/near-far, NOT the production plane extraction). Exact for the count whenever `safe()` holds,
// since then no sphere straddles a plane.
function expected(fovDeg: number, near: number, far: number, aspect: number): number {
    const tan = Math.tan((fovDeg * Math.PI) / 360);
    let n = 0;
    for (const [x, y, z] of boxes) {
        const depth = D_CAM - y;
        if (depth < near || depth > far) continue;
        const halfV = depth * tan;
        if (Math.abs(x) <= halfV * aspect && Math.abs(z) <= halfV) n++;
    }
    return n;
}

// no box's bounding sphere straddles an active plane at this config — the precondition for `expected` to
// match the GPU exactly. Side boundaries scale together by aspect, so a single fov rarely lands both
// clear; the assert scans fov until one does.
function safe(fovDeg: number, near: number, far: number, aspect: number): boolean {
    const tan = Math.tan((fovDeg * Math.PI) / 360);
    for (const [x, y, z] of boxes) {
        const depth = D_CAM - y;
        if (Math.abs(depth - near) < MARGIN || Math.abs(depth - far) < MARGIN) return false;
        if (depth < near || depth > far) continue;
        const halfV = depth * tan;
        if (Math.abs(Math.abs(x) - halfV * aspect) < MARGIN) return false;
        if (Math.abs(Math.abs(z) - halfV) < MARGIN) return false;
    }
    return true;
}

// an fov that cuts the field on both screen axes with no straddler, for the real aspect. `null` if the
// field's spacing leaves no clear band (shouldn't happen for sane aspects).
function sideCutFov(near: number, far: number, aspect: number): number | null {
    for (let halfV = HALF_N * SPACING - 0.5; halfV >= SPACING; halfV -= 0.05) {
        const fov = ((2 * Math.atan(halfV / D_CAM)) / Math.PI) * 180;
        if (!safe(fov, near, far, aspect)) continue;
        const c = expected(fov, near, far, aspect);
        if (c > 1 && c < (2 * HALF_N + 1) ** 2) return fov; // partial cut on the field
    }
    return null;
}

// settle the loop at the current camera config, read the survivor count back through the Mirror, and
// compare to the by-construction expectation. `draws` is how many (surface, mesh) pairs survived with
// instances — the indirect-draw count, 1 when anything is visible.
async function cullCheck(name: string, visible: number, draws: number): Promise<Check> {
    await settle(drawArgs!);
    const got = liveCounts();
    if (!got) return { name, pass: false, detail: "no mirror snapshot" };
    return {
        name,
        pass: got.visible === visible && got.draws === draws,
        // denominator is the whole field, so a passing line shows the real cull ratio (e.g. 35/51)
        detail: `${got.visible}/${total} survive (expected ${visible}), draws ${got.draws}/${draws}`,
    };
}

// the sear render-pass names the profiler resolved on the most recent frame, filtered to `sear:` so the
// compute passes (slab, part, glaze) don't muddy the structural prepass check.
function searPasses(): string[] {
    return [...Profile.gpu.keys()].filter((k) => k.startsWith("sear:")).sort();
}

// settle a few frames, then read the camera's published lanes. `view.tag` / `view.depth` are set
// synchronously by the prepass each frame from the camera's current markers, so a couple frames suffice.
async function lanes(): Promise<{ tag: boolean; depth: boolean }> {
    await frames(4);
    const view = Views.get(cam);
    return { tag: !!view?.tag, depth: !!view?.depth };
}

function laneCheck(name: string, tag: boolean, depth: boolean): Promise<Check> {
    return lanes().then((got) => ({
        name,
        pass: got.tag === tag && got.depth === depth,
        detail: `tag ${got.tag} (want ${tag}), depth ${got.depth} (want ${depth})`,
    }));
}

// the transport wave: each frame rewrite `k · count` filler colors (advancing a cursor), marking the
// "color" slab dirty so SlabSystem's flush scatters them — the slab-write coverage at a swept load. The
// filler is off-screen (always culled), so this is pure transport, not a viz; `part:pack` + the slab
// flush span carry it. `k = 0` (default) leaves the field static.
let waveCursor = 0;
function wave(): void {
    const k = (params?.k as number) ?? 0;
    const n = Math.floor(k * count);
    if (n <= 0) return;
    for (let i = 0; i < n; i++) {
        const eid = fillerEids[(waveCursor + i) % fillerEids.length];
        const t = (Compute.frame + i) * 0.013;
        Color.rgba.set(eid, 0.5 + 0.5 * Math.sin(t), 0.5 + 0.5 * Math.sin(t + 2), 0.7, 1);
    }
    waveCursor = (waveCursor + n) % fillerEids.length;
}

const transportPlugin: Plugin = {
    name: "GymRenderTransport",
    systems: [{ group: "simulation", annotations: { mode: "always" }, update: wave } as System],
};

// the cluster grid's CPU↔GPU boundary gate: Mirror-read the GPU AABB buffer and
// compare every cluster against the TS oracle (`clusterAabb`) for the camera's
// current projection. Run at a projection no earlier frame used, so a stale
// cache (the rebuild-on-projection-change path failing to dispatch) can't pass.
// Tolerance: the GPU slice depth is `near · pow(far/near, z/Z)` in f32 — pow's
// relative error is ≤ ~|s·log2(r)| · 2^-21 (WGSL exp2/log2 ulp bounds), with
// far/near = 150 → log2 ≈ 7.3 → ≤ 2.5e-6 relative; the f32 rounding of the
// packed params adds ~2^-24 per factor. 1e-5 relative covers both with margin.
async function clusterCheck(): Promise<Check> {
    if (!clusterBuf) return { name: "clusters", pass: false, detail: "no cluster mirror" };
    await settle(clusterBuf);
    const snap = clusterBuf.snapshot;
    if (!snap) return { name: "clusters", pass: false, detail: "no cluster snapshot" };
    const view = Views.get(cam);
    if (!view) return { name: "clusters", pass: false, detail: "no view" };
    const oracle = clusterView(cam, view.width / view.height);
    const aabbs = new Float32Array(snap.bytes);
    const base = view.slot * CLUSTER_COUNT * 8;
    let worst = 0;
    let bad = "";
    for (let y = 0; y < CLUSTER_Y; y++) {
        for (let x = 0; x < CLUSTER_X; x++) {
            for (let z = 0; z < CLUSTER_Z; z++) {
                const want = clusterAabb(oracle, x, y, z);
                const o = base + clusterIndex(x, y, z) * 8;
                const got = [
                    aabbs[o],
                    aabbs[o + 1],
                    aabbs[o + 2],
                    aabbs[o + 4],
                    aabbs[o + 5],
                    aabbs[o + 6],
                ];
                const flat = [...want.min, ...want.max];
                for (let i = 0; i < 6; i++) {
                    const err =
                        Math.abs(got[i] - flat[i]) / Math.max(Math.abs(flat[i]), oracle.near);
                    if (err > worst) {
                        worst = err;
                        bad = `cluster (${x},${y},${z}) lane ${i}: got ${got[i]}, want ${flat[i]}`;
                    }
                }
            }
        }
    }
    return {
        name: "clusters",
        pass: worst <= 1e-5,
        detail:
            worst <= 1e-5
                ? `${CLUSTER_COUNT} AABBs match oracle (worst rel ${worst.toExponential(1)})`
                : bad,
    };
}

// the light-cull CPU↔GPU boundary gate: Mirror-read the compacted light list, the
// lightGrid, and the index pool, and compare every (light, cluster) assignment
// against the TS oracle — a light at P with range R lands in exactly the clusters
// whose AABB its sphere intersects (clustered == brute-force membership). The
// compare is banded: a cluster whose box-to-center distance sits within `EPS`
// (relative to range²) of the boundary is skipped — the GPU's f32 pow/clamp and
// the f64 oracle legitimately disagree in that shell (the AABB itself carries
// ~1e-5 relative error, amplified by the distance derivative); outside it the
// sets must match exactly.
const LIGHT_EPS = 5e-3;

async function lightCullCheck(): Promise<Check> {
    const name = "light-cull";
    if (!lightsBuf || !gridBuf || !idxBuf) return { name, pass: false, detail: "no mirrors" };
    await settle(lightsBuf);
    await settle(gridBuf);
    await settle(idxBuf);
    const view = Views.get(cam);
    if (!view || !lightsBuf.snapshot || !gridBuf.snapshot || !idxBuf.snapshot) {
        return { name, pass: false, detail: "no snapshots" };
    }
    const listF = new Float32Array(lightsBuf.snapshot.bytes);
    const listU = new Uint32Array(lightsBuf.snapshot.bytes);
    const grid = new Uint32Array(gridBuf.snapshot.bytes);
    const pool = new Uint32Array(idxBuf.snapshot.bytes);

    // GPU list index → scene light, matched by world position (append order is nondeterministic)
    const count = listU[0];
    if (count !== lightPos.length) {
        return { name, pass: false, detail: `list count ${count}, expected ${lightPos.length}` };
    }
    const toScene: number[] = [];
    // per-light stride is 12 floats: posRange + color + params (POINT_LIGHTS_STRUCT_WGSL)
    const LightStride = 12;
    for (let i = 0; i < count; i++) {
        const o = 4 + i * LightStride;
        const k = lightPos.findIndex(
            (p) =>
                Math.abs(p[0] - listF[o]) < 1e-3 &&
                Math.abs(p[1] - listF[o + 1]) < 1e-3 &&
                Math.abs(p[2] - listF[o + 2]) < 1e-3,
        );
        if (k < 0) {
            return {
                name,
                pass: false,
                detail: `list entry ${i} matches no scene light: (${listF[o].toFixed(3)}, ${listF[o + 1].toFixed(3)}, ${listF[o + 2].toFixed(3)}) w ${listF[o + 3].toExponential(2)}; scene[0] (${lightPos[0]})`,
            };
        }
        toScene.push(k);
    }

    // per-scene-light GPU cluster sets, from grid + pool
    const gpuSets = lightPos.map(() => new Set<number>());
    let reserved = 0;
    for (let c = 0; c < CLUSTER_COUNT; c++) {
        const g = (view.slot * CLUSTER_COUNT + c) * 2;
        const off = grid[g];
        const n = grid[g + 1];
        reserved += n;
        for (let j = 0; j < n; j++) gpuSets[toScene[pool[off + j]]].add(c);
    }

    // pool conservation: this is the only shading view, so the camera's slot accounts for every
    // reserved pool entry. The scene carries 12 depth-only shadow-face views — if the cull binned
    // for them too (the shade-mask regression), the counter runs ~13× the slot total and the
    // shared pool overflows in light-heavy scenes (random froxels losing lights, frame to frame)
    if (pool[0] !== reserved) {
        return {
            name,
            pass: false,
            detail: `pool counter ${pool[0]} ≠ shading slot total ${reserved} — off-screen views are binning`,
        };
    }

    // the oracle: view-space light position via the camera's inverted CPU world matrix
    const world = composeTransform(cam, new Float32Array(16));
    const viewMat = invert(world, new Float32Array(16));
    const cv = clusterView(cam, view.width / view.height);
    let checked = 0;
    for (let k = 0; k < lightPos.length; k++) {
        const [wx, wy, wz] = lightPos[k];
        const vx = viewMat[0] * wx + viewMat[4] * wy + viewMat[8] * wz + viewMat[12];
        const vy = viewMat[1] * wx + viewMat[5] * wy + viewMat[9] * wz + viewMat[13];
        const vz = viewMat[2] * wx + viewMat[6] * wy + viewMat[10] * wz + viewMat[14];
        const rangeSq = lightRange[k] * lightRange[k];
        for (let c = 0; c < CLUSTER_COUNT; c++) {
            const { x, y, z } = clusterCoord(c);
            const a = clusterAabb(cv, x, y, z);
            let distSq = 0;
            const p = [vx, vy, vz];
            for (let i = 0; i < 3; i++) {
                const q = Math.min(Math.max(p[i], a.min[i]), a.max[i]);
                distSq += (q - p[i]) ** 2;
            }
            const rel = distSq / rangeSq;
            if (Math.abs(rel - 1) < LIGHT_EPS) continue; // boundary shell — f32 vs f64
            const want = rel < 1;
            if (gpuSets[k].has(c) !== want) {
                return {
                    name,
                    pass: false,
                    detail: `light ${k} cluster (${x},${y},${z}): gpu ${gpuSets[k].has(c)}, oracle ${want} (rel ${rel.toFixed(4)})`,
                };
            }
            checked++;
        }
    }
    return {
        name,
        pass: true,
        detail: `${count} lights × ${CLUSTER_COUNT} clusters match oracle (${checked} decisive)`,
    };
}

// point shadows: the caster metadata sear wrote for the FS — light eids, world positions, the range-derived
// clip planes (the "pointShadows" Mirror) — and the importance-sized atlas tile rects (the "pointTileRects"
// Mirror). This is the CPU↔GPU boundary of the feature (the atlas depth itself is depth32float, not copyable;
// the look is the render `pointShadow` / `spotShadow` probe modes). The tile rects are pinned by their
// allocator invariants — square, in-bounds, a point's six faces one size, globally non-overlapping — not
// the exact placement, which is a policy detail (importance sizing) the shaded-look probe validates end to end.
async function pointShadowCheck(): Promise<Check> {
    const name = "point-shadow";
    if (!pointBuf || !tileBuf) return { name, pass: false, detail: "no mirror" };
    await settle(pointBuf);
    await settle(tileBuf);
    if (!pointBuf.snapshot || !tileBuf.snapshot)
        return { name, pass: false, detail: "no snapshot" };
    const f = new Float32Array(pointBuf.snapshot.bytes);
    const r = new Float32Array(tileBuf.snapshot.bytes); // per-(caster, face) rects, slot·6 + face
    // PointCaster stride: pos + nf + the spot basis (spotA/B/C), 5 vec4 = 20 f32 (rects are separate now)
    const Caster = 20;
    const atlas = pointAtlasSize();
    const want = new Set(shadowEids);
    const tiles: { x: number; y: number; size: number }[] = []; // every placed face tile, for overlap
    for (let k = 0; k < Math.min(shadowEids.length, pointCasters()); k++) {
        const o = k * Caster;
        const eid = f[o + 3];
        if (!want.has(eid))
            return { name, pass: false, detail: `slot ${k} eid ${eid} not a caster` };
        const i = lightEids.indexOf(eid);
        const [px, py, pz] = lightPos[i];
        if (
            Math.abs(f[o] - px) > 1e-3 ||
            Math.abs(f[o + 1] - py) > 1e-3 ||
            Math.abs(f[o + 2] - pz) > 1e-3
        ) {
            return {
                name,
                pass: false,
                detail: `slot ${k} pos (${f[o]}, ${f[o + 1]}, ${f[o + 2]})`,
            };
        }
        const range = lightRange[i];
        if (Math.abs(f[o + 4] - range / 1000) > 1e-6 || Math.abs(f[o + 5] - range) > 1e-5) {
            return { name, pass: false, detail: `slot ${k} clip [${f[o + 4]}, ${f[o + 5]}]` };
        }
        // a point caster's six face rects: square, in bounds, and one shared tile size
        const size0 = r[k * 6 * 4 + 2];
        if (!(size0 > 0)) return { name, pass: false, detail: `slot ${k} face 0 has no tile` };
        for (let face = 0; face < 6; face++) {
            const ri = (k * 6 + face) * 4;
            const [u0, v0, du, dv] = [r[ri], r[ri + 1], r[ri + 2], r[ri + 3]];
            if (Math.abs(du - dv) > 1e-7 || Math.abs(du - size0) > 1e-7)
                return { name, pass: false, detail: `slot ${k} face ${face} tile ${du}×${dv}` };
            if (u0 < -1e-7 || v0 < -1e-7 || u0 + du > 1 + 1e-7 || v0 + dv > 1 + 1e-7)
                return { name, pass: false, detail: `slot ${k} face ${face} rect out of bounds` };
            tiles.push({ x: u0 * atlas, y: v0 * atlas, size: du * atlas });
        }
    }
    // every remaining slot is empty (pos.w = -1, matching no light)
    for (let k = shadowEids.length; k < pointCasters(); k++) {
        if (f[k * Caster + 3] !== -1) {
            return { name, pass: false, detail: `slot ${k} not empty (${f[k * Caster + 3]})` };
        }
    }
    // the buddy packer must place every face tile without overlap (the no-leak guarantee's geometric half)
    for (let a = 0; a < tiles.length; a++) {
        for (let b = a + 1; b < tiles.length; b++) {
            const ta = tiles[a];
            const tb = tiles[b];
            const sep =
                ta.x + ta.size <= tb.x ||
                tb.x + tb.size <= ta.x ||
                ta.y + ta.size <= tb.y ||
                tb.y + tb.size <= ta.y;
            if (!sep) return { name, pass: false, detail: `tiles ${a} and ${b} overlap` };
        }
    }
    const span = Profile.gpu.has("sear:pointshadow");
    if (!span) return { name, pass: false, detail: "no sear:pointshadow span" };
    return {
        name,
        pass: true,
        detail: `${shadowEids.length} casters, ${tiles.length} tiles packed`,
    };
}

// per-combo cull: each combo (a caster's cube face / spot cone) is its own depth-only frustum-culled view,
// so the Part pack writes a per-combo survivor count at that combo's slot — a member rasterizes only the
// faces it actually hits, not all six. Pin every combo's GPU survivor count (Mirror of drawArgs at the
// combo's slot) to a CPU frustum test: the cube bound sphere vs the combo's six planes, the same predicate
// part.ts `visible()` runs — the planes from the combo camera's `computeViewProj`, exactly what
// BeginFrameSystem packed its cull volume from. A box within EPS of a plane is a straddler (f32 GPU vs f64
// oracle), so the count must land in [decisive-in, decisive-in + straddlers]. The over-draw cut vs the
// deleted amplify path (in-range members × combos) is the win this unified cull buys.
async function comboCullCheck(): Promise<Check> {
    const name = "per-combo-cull";
    const combos = pointComboEids();
    if (combos.length === 0) return { name, pass: false, detail: "no combo cameras" };
    await settle(drawArgs!);
    if (!drawArgs!.snapshot) return { name, pass: false, detail: "no drawArgs snapshot" };
    const cb = Meshes.get("cube")?.bounds;
    const cubeR = cb?.[3] ?? 0;
    const [ccx, ccy, ccz] = [cb?.[0] ?? 0, cb?.[1] ?? 0, cb?.[2] ?? 0]; // bound center (≈0 for the cube)
    const Eps = 1e-4; // world units — above f32 noise (~1e-6), well below the 4-unit grid spacing
    const planes = new Float32Array(FRUSTUM_FLOATS);
    const vp = new Float32Array(16);
    let totalSurvivors = 0;
    for (let c = 0; c < combos.length; c++) {
        const slot = Views.get(combos[c])?.slot;
        if (slot === undefined) return { name, pass: false, detail: `combo ${c} has no view slot` };
        // the combo camera's frustum the way render packed the cull volume (computeViewProj → frustumPlanes)
        computeViewProj(combos[c], 1, vp);
        frustumPlanes(vp, planes);
        // a box's signed margin to its nearest exclusion boundary: min over planes of dot(n,c)+w+radius.
        // ≥ 0 survives (identity rot + scale 1, so center = box pos + bound center, radius = cubeR)
        let decisiveIn = 0;
        let straddle = 0;
        for (const [bx, by, bz] of boxes) {
            const cx = bx + ccx;
            const cy = by + ccy;
            const cz = bz + ccz;
            let margin = Infinity;
            for (let p = 0; p < 6; p++) {
                const o = p * 4;
                const d =
                    planes[o] * cx +
                    planes[o + 1] * cy +
                    planes[o + 2] * cz +
                    planes[o + 3] +
                    cubeR;
                if (d < margin) margin = d;
            }
            if (margin > Eps) decisiveIn++;
            else if (margin >= -Eps) straddle++;
        }
        const counts = packCounts(drawArgs!, slot, pairCount);
        if (!counts) return { name, pass: false, detail: "no counts" };
        const got = counts[pair];
        totalSurvivors += got;
        if (got < decisiveIn || got > decisiveIn + straddle) {
            return {
                name,
                pass: false,
                detail: `combo ${c} (slot ${slot}): got ${got}, expected [${decisiveIn}, ${decisiveIn + straddle}]`,
            };
        }
    }
    // the over-draw the per-combo cull eliminates: the deleted amplify path fanned every member in range of
    // ≥1 caster into every combo — `inRange × comboCount` box-instances rasterized — where per-combo cull
    // draws only Σ survivors. Compute the in-range set on CPU (the sum-of-radii test the deleted union slot
    // ran — this subsumes the old range-cull check) and report the cut; assert it's a real one (not ~1×)
    const nCast = Math.min(pointCasters(), lightEids.length);
    let inRange = 0;
    for (const [bx, by, bz] of boxes) {
        for (let k = 0; k < nCast; k++) {
            const [lx, ly, lz] = lightPos[k];
            const reach = lightRange[k] + cubeR;
            if ((bx - lx) ** 2 + (by - ly) ** 2 + (bz - lz) ** 2 <= reach * reach) {
                inRange++;
                break;
            }
        }
    }
    const amplified = inRange * pointComboCount();
    const cut = amplified / Math.max(totalSurvivors, 1);

    // the headline invariant: the re-gather holds the atlas at one `drawIndexedIndirect` per casting
    // (surface, mesh) pair — NOT fanned to per-(combo, pair). The per-frame `sear:pointshadow` draw count
    // (cumulative / fires) must stay ≤ pairCount; a collapse regression (drawing per combo) blows past it to
    // ~pairCount × comboCount, the Dawn ~1µs/draw floor the whole unified path protects (gpu.md)
    const fires = Profile.indirectFires.get("sear:pointshadow") ?? 0;
    const draws = fires > 0 ? (Profile.indirectCount.get("sear:pointshadow") ?? 0) / fires : 0;
    const collapsed = draws > 0 && draws <= pairCount;
    return {
        name,
        pass: cut > 1.5 && collapsed,
        detail: `${combos.length} combos cull exactly; ${totalSurvivors} vs ${amplified} (${cut.toFixed(1)}× cut, ${inRange} in range); ${draws.toFixed(0)} draws/frame ≤ ${pairCount} pairs`,
    };
}

// per-cascade cull: the sun joins the unified culled-combo spine — each CSM cascade is its own depth-only
// frustum-culled ortho view (`updateCascades` poses the cameras), so the Part pack writes a per-cascade
// survivor count at that cascade's slot. Pin every cascade slot's GPU survivor count to the same CPU frustum
// test `comboCullCheck` runs (the cube bound sphere vs the cascade camera's six planes from `computeViewProj`),
// and confirm the cascade atlas collapses to one `drawIndexedIndirect` per casting mesh (the re-gather, the
// point atlas's win carried to the sun).
async function cascadeCullCheck(): Promise<Check> {
    const name = "per-cascade-cull";
    const cascades = cascadeComboEids();
    if (cascades.length !== cascadeCount() || cascades.length === 0)
        return {
            name,
            pass: false,
            detail: `no cascade cameras (${cascades.length}/${cascadeCount()})`,
        };
    await settle(drawArgs!);
    if (!drawArgs!.snapshot) return { name, pass: false, detail: "no drawArgs snapshot" };
    const cb = Meshes.get("cube")?.bounds;
    const cubeR = cb?.[3] ?? 0;
    const [ccx, ccy, ccz] = [cb?.[0] ?? 0, cb?.[1] ?? 0, cb?.[2] ?? 0];
    const Eps = 1e-4;
    const planes = new Float32Array(FRUSTUM_FLOATS);
    const vp = new Float32Array(16);
    for (let c = 0; c < cascades.length; c++) {
        const slot = Views.get(cascades[c])?.slot;
        if (slot === undefined)
            return { name, pass: false, detail: `cascade ${c} has no view slot` };
        computeViewProj(cascades[c], 1, vp);
        frustumPlanes(vp, planes);
        let decisiveIn = 0;
        let straddle = 0;
        for (const [bx, by, bz] of boxes) {
            const cx = bx + ccx;
            const cy = by + ccy;
            const cz = bz + ccz;
            let margin = Infinity;
            for (let p = 0; p < 6; p++) {
                const o = p * 4;
                const d =
                    planes[o] * cx +
                    planes[o + 1] * cy +
                    planes[o + 2] * cz +
                    planes[o + 3] +
                    cubeR;
                if (d < margin) margin = d;
            }
            if (margin > Eps) decisiveIn++;
            else if (margin >= -Eps) straddle++;
        }
        const counts = packCounts(drawArgs!, slot, pairCount);
        if (!counts) return { name, pass: false, detail: "no counts" };
        const got = counts[pair];
        if (got < decisiveIn || got > decisiveIn + straddle)
            return {
                name,
                pass: false,
                detail: `cascade ${c} (slot ${slot}): got ${got}, expected [${decisiveIn}, ${decisiveIn + straddle}]`,
            };
    }
    // the cascade atlas collapses to one indirect draw per casting mesh (the re-gather), like the point atlas
    const fires = Profile.indirectFires.get("sear:cascadeshadow") ?? 0;
    const draws = fires > 0 ? (Profile.indirectCount.get("sear:cascadeshadow") ?? 0) / fires : 0;
    const collapsed = draws > 0 && draws <= pairCount;
    return {
        name,
        pass: collapsed,
        detail: `${cascades.length} cascades cull exactly; ${draws.toFixed(0)} draws/frame ≤ ${pairCount} pairs`,
    };
}

// overflow degrades loud, never silently: blow the index pool (every light covering every
// cluster, more cluster-light pairs than the pool holds) and verify the overflow counter
// reports the drop while every grid entry stays inside the pool. Restores the scene after.
async function lightOverflowCheck(state: State): Promise<Check> {
    const name = "light-overflow";
    if (!idxBuf || !gridBuf) return { name, pass: false, detail: "no mirrors" };
    const added: number[] = [];
    const need = Math.ceil(LIGHT_POOL / CLUSTER_COUNT) + 8;
    for (let i = lightEids.length; i < need; i++) {
        const eid = state.create();
        state.add(eid, PointLight);
        state.add(eid, Transform);
        Transform.pos.set(eid, i * 0.5, 1, 0, 0);
        added.push(eid);
    }
    for (const eid of [...lightEids, ...added]) PointLight.range.set(eid, 1e4);
    await settle(idxBuf);
    await settle(gridBuf);
    const pool = new Uint32Array(idxBuf.snapshot!.bytes);
    const grid = new Uint32Array(gridBuf.snapshot!.bytes);
    const overflow = pool[1];
    let inBounds = true;
    for (let c = 0; c < CLUSTER_COUNT; c++) {
        const g = (Views.get(cam)!.slot * CLUSTER_COUNT + c) * 2;
        // an empty entry's offset may sit past the pool (the reserve overshot); only
        // entries that will actually be read must stay inside
        if (grid[g + 1] > 0 && grid[g] + grid[g + 1] > LIGHT_POOL + 2) inBounds = false;
    }
    for (const eid of added) state.destroy(eid);
    for (let k = 0; k < lightEids.length; k++) PointLight.range.set(lightEids[k], lightRange[k]);
    await frames(2);
    return {
        name,
        pass: overflow > 0 && inBounds,
        detail: `overflow ${overflow} entries dropped, grid ${inBounds ? "inside" : "OUTSIDE"} pool`,
    };
}

const SENTINEL: [number, number, number, number] = [0.123, 0.456, 0.789, 1];

// round-trip integrity on the real GPU: write a sentinel color to the even grid slots, settle the color
// mirror, and verify the scatter landed — even slots hold the sentinel, odd slots (never written) don't.
// This is the slab-write `verifyScatter` essence — the gym's real-GPU scatter-integrity gate. The
// wave only touches filler eids, so grid writes never collide with it.
async function transportCheck(): Promise<Check> {
    if (!colorBuf) return { name: "transport", pass: false, detail: "no color mirror" };
    for (let i = 0; i < gridEids.length; i += 2) {
        Color.rgba.set(gridEids[i], ...SENTINEL);
    }
    await settle(colorBuf);
    const snap = colorBuf.snapshot;
    if (!snap) return { name: "transport", pass: false, detail: "no color snapshot" };
    // the color slab mirrors as one sRGB-packed u32 per entity (srgb8x4); compare in the packed
    // domain — the scatter is a lossless u32 copy, so the readback equals the CPU pack of the sentinel
    const packed = new Uint32Array(snap.bytes);
    const expected = packLdrColor(...SENTINEL);
    const near = (eid: number) => packed[eid] === expected;
    for (let i = 0; i < gridEids.length; i += 2) {
        if (!near(gridEids[i])) {
            return {
                name: "transport",
                pass: false,
                detail: `grid slot ${gridEids[i]} got 0x${packed[gridEids[i]].toString(16)}, want 0x${expected.toString(16)}`,
            };
        }
    }
    for (let i = 1; i < gridEids.length; i += 2) {
        if (near(gridEids[i])) {
            return {
                name: "transport",
                pass: false,
                detail: `untouched grid slot ${gridEids[i]} clobbered`,
            };
        }
    }
    return {
        name: "transport",
        pass: true,
        detail: `${Math.ceil(gridEids.length / 2)} writes scattered, untouched intact`,
    };
}

async function buildCull(state: State, p: Params): Promise<void> {
    const seed = p.seed as number;
    count = p.count as number;

    // bare lights — defaults match the kitchen scene's `<a ambient-light />`. The sun carries a `Shadow` so
    // the CSM cascade cameras pose + cull the box field (the `per-cascade-cull` oracle reads their survivors)
    state.add(state.create(), AmbientLight);
    const sun = state.create();
    state.add(sun, DirectionalLight);
    DirectionalLight.direction.set(sun, -0.4, -0.8, -0.45, 0);
    state.add(sun, Shadow);

    // flat square field at y=0 (depth D_CAM). This fixed field is what the verdict counts; `count`
    // drives a separate always-culled filler population (below) for the throughput + transport load.
    boxes = [];
    gridEids = [];
    for (let ix = -HALF_N; ix <= HALF_N; ix++)
        for (let iz = -HALF_N; iz <= HALF_N; iz++) boxes.push([ix * SPACING, 0, iz * SPACING]);

    const rand = rng(seed);
    for (const [x, y, z] of boxes) {
        const eid = state.create();
        state.add(eid, Part); // surface "default", mesh "cube" by trait default
        state.add(eid, Transform);
        state.add(eid, Color);
        Transform.pos.set(eid, x, y, z, 0);
        Color.rgba.set(eid, 0.4 + rand() * 0.6, 0.4 + rand() * 0.6, 0.4 + rand() * 0.6, 1);
        gridEids.push(eid);
    }

    // pack-load + transport population: `count` instances parked far off +X, always outside the
    // frustum at every test config. They never enter a survivor set (the verdict reads only the field
    // above), but the cull pass scans them — so `part:pack` times a realistic instance count — and the
    // transport wave rewrites their colors at the swept `k` load.
    fillerEids = [];
    for (let i = 0; i < count; i++) {
        const eid = state.create();
        state.add(eid, Part);
        state.add(eid, Transform);
        state.add(eid, Color);
        Transform.pos.set(eid, 1e4 + i, 0, 0, 0);
        Color.rgba.set(eid, 0.4 + rand() * 0.6, 0.4 + rand() * 0.6, 0.4 + rand() * 0.6, 1);
        fillerEids.push(eid);
    }

    // point lights scattered through the field at varied heights and ranges — the
    // light-cull oracle's known layout (deterministic, margin from grid alignment)
    const nLights = p.lights as number;
    lightEids = [];
    lightPos = [];
    lightRange = [];
    for (let i = 0; i < nLights; i++) {
        const eid = state.create();
        state.add(eid, PointLight);
        state.add(eid, Transform);
        const pos: [number, number, number] = [
            ((i * 1.73) % 22) - 11,
            0.6 + (i % 3) * 1.4,
            ((i * 2.31) % 18) - 9,
        ];
        Transform.pos.set(eid, ...pos, 0);
        PointLight.color.set(eid, [0xffd9a0, 0xa0c8ff, 0xffa0d0][i % 3]);
        PointLight.intensity.set(eid, 2);
        const range = 2.4 + (i % 4) * 0.9;
        PointLight.range.set(eid, range);
        lightEids.push(eid);
        lightPos.push(pos);
        lightRange.push(range);
    }

    // the first pointCasters() lights cast — the metadata the point-shadow check pins
    shadowEids = lightEids.slice(0, pointCasters());
    for (const eid of shadowEids) state.add(eid, Shadow);

    // Top-down on-axis start (pitch π/2 → camera at (0, D_CAM, 0) looking -Y). Headless holds this so
    // the assert reads the closed-form rectangle; a live tab orbits off it. The assert never fights
    // it — it varies fov/near/far, which orbit doesn't own.
    cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Tag); // the id lane — the prepass-lane gate toggles this + Depth
    state.add(cam, Orbit);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 60);
    Camera.near.set(cam, 1);
    Camera.far.set(cam, 200);
    Orbit.distance.set(cam, D_CAM);
    Orbit.yaw.set(cam, 0);
    // the assert needs the exact top-down pose (pitch π/2). The default maxPitch is π/2 − 0.01, so a
    // pitch above it clamps on the first orbit drag; raise the limit to keep π/2 in range.
    Orbit.pitch.set(cam, Math.PI / 2);
    Orbit.maxPitch.set(cam, Math.PI / 2);
    Camera.antialias.set(cam, p.antialias ? 1 : 0); // off → exercise the single-sample color pass
    // the camera auto-binds to the first <canvas> in BeginFrameSystem — no attachCanvas needed.

    total = boxes.length;
    pairCount = Surfaces.size * Meshes.size;
    pair = Meshes.id("cube")! * Surfaces.size + Surfaces.id("default")!;

    // first frames allocate Parts.drawArgs + the color slab (the pack runs once a camera + Parts
    // exist); then mirror them for the assert + live HUD readback.
    await frames(2);
    drawArgs = mirror(Parts.drawArgs!);
    colorBuf = mirror(Color.rgba.gpu!); // allocated by SlabPlugin.warm, non-null by now
    clusterBuf = mirror(Clusters.aabbs!); // allocated by RenderPlugin.warm
    lightsBuf = mirror(LightCull.lights!);
    gridBuf = mirror(LightCull.grid!);
    idxBuf = mirror(LightCull.indices!);
    pointBuf = mirror(Compute.buffers.get("pointShadows")!); // published by SearPlugin.warm
    tileBuf = mirror(Compute.buffers.get("pointTileRects")!); // the importance-sized atlas tile rects
    await frames(3);
}

// the structural coverage. The cull edges (each known by construction, none re-deriving the frustum test
// it checks; aspect read from the live View so the predicate matches the planes the renderer built), the
// cluster + light-cull oracles, the shadow metadata, the transport round-trip, and the prepass-lane toggle
// gate. The pose stays put throughout.
async function assertCull(state: State): Promise<Check[]> {
    const view = Views.get(cam);
    const aspect = view ? view.width / view.height : 1;
    const checks: Check[] = [];

    const cull = async (name: string, fov: number, near: number, far: number) => {
        Camera.fov.set(cam, fov);
        Camera.near.set(cam, near);
        Camera.far.set(cam, far);
        const v = expected(fov, near, far, aspect);
        checks.push(await cullCheck(name, v, v > 0 ? 1 : 0));
    };

    // wide fov + planes bracketing the field depth → nothing culled
    await cull("all-visible", 90, 1, 100);

    // narrow the fov until the side planes cut the field cleanly on both axes (near/far stay wide, so
    // this isolates the side planes)
    const sideFov = sideCutFov(1, 100, aspect);
    if (sideFov === null) {
        checks.push({
            name: "side-cull",
            pass: false,
            detail: `no clear fov @ aspect ${aspect.toFixed(3)}`,
        });
    } else {
        await cull("side-cull", sideFov, 1, 100);
    }

    // far plane in front of the field → far plane drops every box
    await cull("far-cull", 90, 1, 20);
    // near plane behind the field → near plane drops every box
    await cull("near-cull", 90, 30, 100);

    // clusters: a projection no earlier check used, so the rebuild path is what's measured
    Camera.fov.set(cam, 75);
    Camera.near.set(cam, 2);
    Camera.far.set(cam, 150);
    checks.push(await clusterCheck());

    Camera.fov.set(cam, 60); // restore for the live HUD
    Camera.near.set(cam, 1);
    Camera.far.set(cam, 200);
    await settle(drawArgs!);

    // light cull: clustered assignment == oracle membership at the restored
    // projection, then the pool-overflow path degrades loud
    checks.push(await lightCullCheck());
    checks.push(await pointShadowCheck());
    checks.push(await comboCullCheck());
    checks.push(await cascadeCullCheck());
    checks.push(await lightOverflowCheck(state));

    // transport: CPU → GPU scatter integrity through the color slab.
    checks.push(await transportCheck());

    // prepass lanes: toggle the camera's markers and verify the lanes appear/vanish independently and
    // ride one `sear:prepass` (no per-output `sear:tag` / `sear:depth`). The camera starts Tag-only.
    checks.push(await laneCheck("tag-only", true, false));
    state.add(cam, Depth);
    checks.push(await laneCheck("tag+depth", true, true));
    const passes = searPasses();
    checks.push({
        name: "one-prepass",
        pass:
            passes.includes("sear:prepass") &&
            !passes.some((p) => p === "sear:tag" || p === "sear:depth"),
        detail: `sear passes: ${passes.join(", ") || "(none resolved)"}`,
    });
    state.remove(cam, Tag); // depth requestable on its own
    checks.push(await laneCheck("depth-only", false, true));
    state.remove(cam, Depth); // bare camera → no prepass output
    checks.push(await laneCheck("bare", false, false));
    state.add(cam, Tag); // restore the id lane for the live HUD
    await frames(2);

    return checks;
}

// ============================================================================
// shaded-look modes — the mid-frame framebuffer luminance probe
// ============================================================================

const AMBIENT = 0.05;
const ALBEDO = 0.8; // the floor's linear grey — bright enough that the falloff reads
const LIGHT_Y = 3;
const INTENSITY = 12;
const RANGE = 30;
const FIXTURE_R = 0.2; // the caster sphere colocated with the light — the self-occlusion trigger

// out[0] = the average linear luminance of the offscreen scene color over a 96×96 sample grid (the lit
// check); out[1..4] = the peak luminance in the left / right / top / bottom screen half (the spec check).
// Both axes are reported because the top-down camera's straight-down look has a degenerate up, so which
// world axis the two spheres separate along on screen isn't known a priori — the assert reads whichever
// axis actually splits them. One thread, no atomics. `view.framebuffer` is the HDR linear color the color
// pass resolved, so the readback is pre-tonemap luminance, unclamped at the specular highlight (the peak).
const PROBE_WGSL = /* wgsl */ `
@group(0) @binding(0) var fb: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
    let dim = vec2<f32>(textureDimensions(fb));
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let N = 96;
    var sum = 0.0;
    var pL = 0.0; var pR = 0.0; var pT = 0.0; var pB = 0.0;
    for (var y = 0; y < N; y = y + 1) {
        for (var x = 0; x < N; x = x + 1) {
            let uv = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) / f32(N);
            let lum = dot(textureLoad(fb, vec2<i32>(uv * dim), 0).rgb, luma);
            sum = sum + lum;
            if (uv.x < 0.5) { pL = max(pL, lum); } else { pR = max(pR, lum); }
            if (uv.y < 0.5) { pT = max(pT, lum); } else { pB = max(pB, lum); }
        }
    }
    out[0] = sum / f32(N * N);
    out[1] = pL; out[2] = pR; out[3] = pT; out[4] = pB;
    // spot row: the cone hits the floor under the spot (screen centre); the screen corner is the far floor
    // outside the cone, lit by ambient only. The gap between them is the cone confinement. The centre
    // (out[5]) also reads which surface won the depth test in the z-fight row (front = bright, back = dark)
    out[5] = dot(textureLoad(fb, vec2<i32>(vec2<f32>(0.5, 0.5) * dim), 0).rgb, luma);
    out[6] = dot(textureLoad(fb, vec2<i32>(vec2<f32>(0.08, 0.08) * dim), 0).rgb, luma);
    // acne row: the MINIMUM luminance over the central window (the directly-lit, in-cone flat floor).
    // Shadow acne speckles a self-receiving flat surface with wrongly-shadowed fragments that drop to
    // ambient-only — so a clean floor's worst pixel still reads well above the ambient ceiling, an acned
    // one doesn't. Min, not average: acne averages out, the worst fragment is the signal
    var lo = 1e9;
    for (var y = 0; y < N; y = y + 1) {
        for (var x = 0; x < N; x = x + 1) {
            let uv = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) / f32(N);
            if (uv.x < 0.35 || uv.x > 0.65 || uv.y < 0.35 || uv.y > 0.65) { continue; }
            lo = min(lo, dot(textureLoad(fb, vec2<i32>(uv * dim), 0).rgb, luma));
        }
    }
    out[7] = lo;
}`;

let probeBuf: GPUBuffer | null = null;
let probeMirror: Mirror | null = null;
let probePipeline: GPUComputePipeline | null = null;
let probeBg: GPUBindGroup | null = null;

// dispatch the probe after the color pass has written the framebuffer (glaze reads it next but never
// rewrites it). The bind group binds late — `view.framebuffer` exists only once the first color pass ran.
const ProbeSystem: System = {
    name: "render-probe",
    group: "draw",
    after: [ColorSystem],
    update() {
        if (!Render.encoder || !probePipeline || !probeBuf) return;
        const view = Views.get(cam);
        if (!view?.framebuffer) return;
        if (!probeBg) {
            probeBg = Compute.device.createBindGroup({
                label: "render-probe",
                layout: probePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: view.framebuffer },
                    { binding: 1, resource: { buffer: probeBuf } },
                ],
            });
        }
        const pass = Render.encoder.beginComputePass({ label: "render-probe" });
        pass.setPipeline(probePipeline);
        pass.setBindGroup(0, probeBg);
        pass.dispatchWorkgroups(1);
        pass.end();
    },
};

// a big bright floor filling the top-down frame — the receiver for the lit / spot / shadow / acne rows.
// The spec + zfight rows bring their own geometry instead.
function probeFloor(state: State): void {
    const floor = state.create();
    state.add(floor, Transform);
    Transform.pos.set(floor, 0, 0, 0, 0);
    Transform.scale.set(floor, 40, 0.2, 40, 0);
    state.add(floor, Part); // default surface: diffuse lit
    state.add(floor, Color);
    Color.rgba.set(floor, ALBEDO, ALBEDO, ALBEDO, 1);
}

async function buildProbe(state: State, p: Params): Promise<void> {
    const ambient = state.create();
    state.add(ambient, AmbientLight);
    AmbientLight.color.set(ambient, 0xffffff);
    AmbientLight.intensity.set(ambient, AMBIENT);

    // the receiver floor is shared by every row except the ones that supply their own geometry
    if (mode !== "spec" && mode !== "zfight" && mode !== "cascade-boundary") probeFloor(state);

    if (mode === "spec") {
        // the representative-point specular row: two metallic spheres on opposite halves of the frame,
        // each lit by an overhead lamp identical but for source radius. The sharp lamp keeps a tight
        // bright highlight; the soft lamp spreads it and the energy-conserving renormalization drops its
        // peak — so the two halves' peak luminance diverge (plain Cook-Torrance, radius-blind, wouldn't)
        const metalSphere = (x: number, lampRadius: number) => {
            const s = state.create();
            state.add(s, Transform);
            Transform.pos.set(s, x, 1.4, 0, 0);
            Transform.scale.set(s, 2.6, 2.6, 2.6, 0);
            state.add(s, Part);
            Part.mesh.set(s, Meshes.id("sphere") ?? 0);
            state.add(s, Color);
            Color.rgba.set(s, 0.9, 0.9, 0.9, 1);
            state.add(s, Material);
            Material.params.set(s, 1, 0.5, 0, 1); // metallic 1, roughness 0.5 — broad highlight, no diffuse

            // the lamp directly above its sphere (own distance ~3.6). A metallic sphere reflects every
            // light in range, so range-limit each lamp to ~6: it reaches its own sphere but the falloff
            // window is exactly zero at the cross sphere (~8.8 away), or the brighter sharp lamp's
            // reflection would dominate BOTH spheres and the peaks wouldn't diverge
            const lamp = state.create();
            state.add(lamp, Transform);
            Transform.pos.set(lamp, x, 5, 0, 0);
            state.add(lamp, PointLight);
            PointLight.color.set(lamp, 0xffffff);
            PointLight.intensity.set(lamp, 30);
            PointLight.range.set(lamp, 6);
            PointLight.radius.set(lamp, lampRadius);
        };
        metalSphere(-4, 0.05); // sharp source
        metalSphere(4, 1.0); // soft source
    } else if (mode === "spot") {
        // a spot aimed straight down at the floor (forward = -Y, a -90° pitch about X). The cone lights
        // a ~2.3 m circle under it (outer 25° at height 5); the floor corners sit outside it, ambient
        // only — what the `cone` check reads. Pre-spot this was a plain omni point light flooding the
        // whole floor, so the dark corner is the new behavior the check pins.
        const spot = state.create();
        state.add(spot, Transform);
        Transform.pos.set(spot, 0, 5, 0, 0);
        Transform.rot.set(spot, -Math.SQRT1_2, 0, 0, Math.SQRT1_2); // -90° about X → local -Z points down
        state.add(spot, PointLight);
        PointLight.color.set(spot, 0xffffff);
        PointLight.intensity.set(spot, 20);
        PointLight.range.set(spot, 20);
        state.add(spot, Spot);
        Spot.inner.set(spot, 15);
        Spot.outer.set(spot, 25);
    } else if (mode === "spotShadow") {
        // a shadow-casting spot straight down + a blocker over the floor centre. The blocker's shadow
        // darkens the centre (which the cone would otherwise light brightest), so a cone-lit ring
        // surrounds a shadowed core — the spot's single-tile atlas caster doing its job
        const spot = state.create();
        state.add(spot, Transform);
        Transform.pos.set(spot, 0, 6, 0, 0);
        Transform.rot.set(spot, -Math.SQRT1_2, 0, 0, Math.SQRT1_2); // straight down
        state.add(spot, PointLight);
        PointLight.color.set(spot, 0xffffff);
        PointLight.intensity.set(spot, 25);
        PointLight.range.set(spot, 25);
        state.add(spot, Spot);
        Spot.inner.set(spot, 20);
        Spot.outer.set(spot, 35); // wide cone — the lit ring extends well past the blocker's shadow
        state.add(spot, Shadow);

        // the blocker, between the spot and the floor centre, casting a shadow onto (0,0,0)
        const blocker = state.create();
        state.add(blocker, Transform);
        Transform.pos.set(blocker, 0, 3, 0, 0);
        Transform.scale.set(blocker, 1, 1, 1, 0);
        state.add(blocker, Part);
        Part.mesh.set(blocker, Meshes.id("sphere") ?? 0);
        state.add(blocker, Color);
        Color.rgba.set(blocker, 0.7, 0.7, 0.7, 1);
    } else if (mode === "pointShadow") {
        // an OFF-AXIS point caster at (-6, 3, 0) + a blocker sphere at (-3, 1.5, 0); the ray through the
        // blocker meets the floor at the origin, so the blocker's shadow lands at screen centre. The
        // floor→light direction at the origin is (6, -3, 0), dominant +X → the +X cube face, not -Y — so
        // the shadow is sampled from an off-axis face (the case a per-face-clipping bug would corrupt).
        // The check reads the shadowed centre against the lit floor ring (the spot-shadow shape, point caster)
        const lamp = state.create();
        state.add(lamp, Transform);
        Transform.pos.set(lamp, -6, 3, 0, 0);
        state.add(lamp, PointLight);
        PointLight.color.set(lamp, 0xffffff);
        PointLight.intensity.set(lamp, INTENSITY);
        PointLight.range.set(lamp, RANGE);
        state.add(lamp, Shadow);

        const blocker = state.create();
        state.add(blocker, Transform);
        Transform.pos.set(blocker, -3, 1.5, 0, 0);
        Transform.scale.set(blocker, 1.2, 1.2, 1.2, 0);
        state.add(blocker, Part);
        Part.mesh.set(blocker, Meshes.id("sphere") ?? 0);
        state.add(blocker, Color);
        Color.rgba.set(blocker, 0.7, 0.7, 0.7, 1);
    } else if (mode === "cascade" || mode === "cascade-ortho") {
        // the CSM sun: a bright directional Shadow light + a blocker positioned up-sun of the floor centre,
        // so the blocker's cascade shadow lands at the origin (screen centre). The flat floor is the receiver;
        // a sun-lit ring surrounds the shadowed core — the directional twin of the spotShadow / pointShadow
        // rows, exercising the cascade atlas + the receiver's view-z cascade select + the unified culled spine.
        // `cascade-ortho` reuses this scene under an orthographic camera (the camera block below): the ortho
        // path's single footprint box must shadow the centre where the frustum-slice fit didn't reach the ground
        const sun = state.create();
        state.add(sun, DirectionalLight);
        DirectionalLight.color.set(sun, 0xffffff);
        DirectionalLight.intensity.set(sun, 3);
        DirectionalLight.direction.set(sun, 0.45, -0.85, 0, 0); // tilted in +x so the shadow offsets in screen
        state.add(sun, Shadow);
        Shadow.distance.set(sun, 50);

        // the blocker up-sun of the origin: the shadow ray (sun dir) through it meets y=0 at (0,0,0), so its
        // cascade shadow lands at screen centre. Up + to the side, clear of the camera's low-angle line in
        const blocker = state.create();
        state.add(blocker, Transform);
        Transform.pos.set(blocker, -2.12, 4, 0, 0);
        Transform.scale.set(blocker, 1.5, 1.5, 1.5, 0);
        state.add(blocker, Part);
        state.add(blocker, Color);
        Color.rgba.set(blocker, 0.7, 0.7, 0.7, 1);
    } else if (mode === "acne") {
        // a shadow-casting spot straight down over the flat floor, no blocker — the floor is its own
        // only receiver, so any self-shadow speckle (acne) on the directly-lit centre is the precision
        // signal. The floor fragment under the spot has its normal parallel to the light, the dead-on
        // case the normal offset can't help — only float depth + reverse-Z + the residual bias keep it clean
        const spot = state.create();
        state.add(spot, Transform);
        Transform.pos.set(spot, 0, 6, 0, 0);
        Transform.rot.set(spot, -Math.SQRT1_2, 0, 0, Math.SQRT1_2); // straight down
        state.add(spot, PointLight);
        PointLight.color.set(spot, 0xffffff);
        PointLight.intensity.set(spot, 25);
        PointLight.range.set(spot, 50);
        state.add(spot, Spot);
        Spot.inner.set(spot, 22);
        Spot.outer.set(spot, 32); // covers the central window the acne min samples
        state.add(spot, Shadow);
    } else if (mode === "cascade-boundary") {
        // a high roof shadows the ENTIRE receding floor, which spans the cascade splits up the frame. With a
        // correct fit every visible floor fragment is uniformly shadowed; the multi-cascade boundary bug lets a
        // near cascade's tight box clip the high roof occluder, so the near floor (cascade 0) reads lit — a
        // bright full-sun rim at the seam. A straight-down sun makes the shadowed floor read ambient and a leak
        // read full sun, the maximum contrast the "no rim anywhere" peak check reads.
        const sun = state.create();
        state.add(sun, DirectionalLight);
        DirectionalLight.color.set(sun, 0xffffff);
        DirectionalLight.intensity.set(sun, 3);
        DirectionalLight.direction.set(sun, 0, -1, 0, 0);
        state.add(sun, Shadow);
        Shadow.distance.set(sun, 50);

        // a big floor receding across the splits + a high roof covering it all (the occluder a near cascade's
        // tight box clips). The roof sits well above cascade 0's pre-fix box top (a few metres) but within the
        // shadow distance, so only the near-plane-toward-light extension captures it. Floor and roof share the
        // ±50 footprint, so no floor is ever directly sun-lit — every visible fragment should read ambient.
        const floor = state.create();
        state.add(floor, Transform);
        Transform.pos.set(floor, 0, 0, 0, 0);
        Transform.scale.set(floor, 100, 0.2, 100, 0);
        state.add(floor, Part);
        state.add(floor, Color);
        Color.rgba.set(floor, ALBEDO, ALBEDO, ALBEDO, 1);

        const roof = state.create();
        state.add(roof, Transform);
        Transform.pos.set(roof, 0, 10, 0, 0);
        Transform.scale.set(roof, 100, 0.2, 100, 0);
        state.add(roof, Part);
        state.add(roof, Color);
        Color.rgba.set(roof, 0.7, 0.7, 0.7, 1);
    } else if (mode === "zfight") {
        // two big flat slabs 0.5 m apart in Y (front higher, nearer the overhead camera at ~5000 m),
        // unlit so the centre pixel reads the winning surface's colour directly: front bright, back dark.
        // The front must win the depth test — reverse-Z + float resolves the separation at 5000 m
        const slab = (y: number, grey: number) => {
            const s = state.create();
            state.add(s, Transform);
            Transform.pos.set(s, 0, y, 0, 0);
            Transform.scale.set(s, 4000, 0.05, 4000, 0);
            state.add(s, Part);
            Part.surface.set(s, Surfaces.id("unlit") ?? 0);
            state.add(s, Color);
            Color.rgba.set(s, grey, grey, grey, 1);
        };
        slab(-0.25, 0.05); // back — dark
        slab(0.25, 0.9); // front — bright; nearer the camera, so it must win
    } else {
        // mode "lit": the lamp — a caster point light wrapped in a small unlit sphere at the same point,
        // the fixture whose interior shell the atlas must cull, or it occludes its own light
        const lamp = state.create();
        state.add(lamp, Transform);
        Transform.pos.set(lamp, 0, LIGHT_Y, 0, 0);
        Transform.scale.set(lamp, FIXTURE_R * 2, FIXTURE_R * 2, FIXTURE_R * 2, 0);
        state.add(lamp, Part);
        Part.mesh.set(lamp, Meshes.id("sphere") ?? 0);
        Part.surface.set(lamp, Surfaces.id("unlit") ?? 0);
        state.add(lamp, Color);
        Color.rgba.set(lamp, 1, 0.9, 0.7, 1);
        state.add(lamp, PointLight);
        PointLight.color.set(lamp, 0xffffff);
        PointLight.intensity.set(lamp, INTENSITY);
        PointLight.range.set(lamp, RANGE);
        PointLight.radius.set(lamp, p.radius as number);
        if (p.shadows) state.add(lamp, Shadow);
    }

    cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Orbit);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 60);
    Camera.clearColor.set(cam, 0x000000);
    // most rows view top-down so the floor fills the frame; the spec row pulls back to frame both
    // spheres, the spot row pulls back so the corners sit outside the cone. The spot-shadow row views
    // at a low angle and aims at the floor centre (the shadowed point), so the blocker — up at y=3 —
    // doesn't occlude the camera's line to the shadow, and the shadowed centre lands at screen centre
    Orbit.distance.set(
        cam,
        mode === "zfight"
            ? 5000
            : mode === "spotShadow"
              ? 7.5
              : mode === "pointShadow"
                ? 11
                : mode === "cascade"
                  ? 9
                  : mode === "cascade-ortho"
                    ? 70 // posed far — forward distance to the ground > Shadow.distance, the ortho regression
                    : mode === "cascade-boundary"
                      ? 8
                      : mode === "spec"
                        ? 12
                        : mode === "spot"
                          ? 10
                          : 8,
    );
    // the point-shadow row views at a low angle (like the spot-shadow row), aimed at the shadowed origin
    // with the off-axis light + blocker to the side, not between the camera and the shadow. The acne +
    // z-fight rows view straight down (π/2) so the flat receiver / slabs fill the frame
    Orbit.pitch.set(
        cam,
        mode === "spotShadow"
            ? 0.35
            : mode === "pointShadow"
              ? 0.45
              : mode === "cascade" || mode === "cascade-ortho"
                ? 0.4
                : // cascade-boundary 0.7: steep enough the whole frame lands on floor within Shadow.distance
                  // (no beyond-distance lit floor), yet shallow enough it recedes across the near splits
                  mode === "cascade-boundary"
                  ? 0.7
                  : Math.PI / 2,
    );
    Orbit.maxPitch.set(cam, Math.PI / 2);
    if (mode === "zfight") {
        Orbit.maxDistance.set(cam, 20000); // the default caps distance at 30; the slabs sit ~5000 m out
        Camera.near.set(cam, 0.1);
        Camera.far.set(cam, 20000); // a wide near:far is what starves a forward-Z buffer's far precision
    }
    if (mode === "cascade-ortho") {
        // the orrstead regression case: an orthographic camera posed far from the scene. The frustum-slice fit
        // ran cascades along [near, distance] that never reach the ground at this forward distance — only the
        // single footprint box does. Orbit caps distance at 30 by default, so lift it for the distance-70 pose
        Camera.mode.set(cam, CameraMode.Orthographic);
        Camera.size.set(cam, 8);
        Orbit.maxDistance.set(cam, 120);
    }
    // a definite near so the cascade splits (read from Camera.near) are well-conditioned, not ~0
    if (mode === "cascade" || mode === "cascade-ortho" || mode === "cascade-boundary")
        Camera.near.set(cam, 0.1);

    await setupFramebufferProbe(state);
}

// the shared mid-frame framebuffer luminance probe: PROBE_WGSL over view.framebuffer, read back through a
// Mirror. Used by the shaded-look modes and by transparency (the blend composite lands in view.framebuffer).
async function setupFramebufferProbe(state: State): Promise<void> {
    probePipeline = await Compute.device.createComputePipelineAsync({
        label: "render-probe",
        layout: "auto",
        compute: {
            module: Compute.device.createShaderModule({ label: "render-probe", code: PROBE_WGSL }),
            entryPoint: "main",
        },
    });
    probeBuf = Compute.device.createBuffer({
        label: "render-probe",
        size: 32, // 8 floats: avg + 4 half-peaks (L/R/T/B) + centre + corner + acne min
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    probeBg = null;
    state.addSystem(ProbeSystem);
    await frames(4);
    probeMirror = mirror(probeBuf);
    await frames(3);
}

async function assertProbe(): Promise<Check[]> {
    if (!probeMirror) return [{ name: "probe", pass: false, detail: "no probe mirror" }];
    await settle(probeMirror);
    if (!probeMirror.snapshot) return [{ name: "probe", pass: false, detail: "no snapshot" }];
    const out = new Float32Array(probeMirror.snapshot.bytes);
    const ceiling = AMBIENT * ALBEDO;

    if (mode === "spec") {
        // the two metallic spheres differ only in their lamp's source radius, so a radius-blind BRDF
        // would render equal peaks. The representative-point sphere BRDF widens the soft lamp's lobe
        // and renormalizes its peak down, so the sharp sphere is markedly brighter (analytic peak ratio
        // ~10× at these params). The spheres separate along one screen axis (unknown which, given the
        // degenerate top-down up), so read both axes and assert on whichever actually splits them.
        const hHi = Math.max(out[1], out[2]);
        const hLo = Math.min(out[1], out[2]);
        const vHi = Math.max(out[3], out[4]);
        const vLo = Math.min(out[3], out[4]);
        const hRatio = hHi / Math.max(hLo, 1e-4);
        const vRatio = vHi / Math.max(vLo, 1e-4);
        const axis = hRatio >= vRatio ? "L/R" : "T/B";
        const ratio = Math.max(hRatio, vRatio);
        const peak = Math.max(hHi, vHi);
        return [
            {
                name: "spec",
                pass: peak > 0.5 && ratio > 2,
                detail: `sharp/soft peak ratio ${ratio.toFixed(2)} on ${axis} (peak ${peak.toFixed(2)}, want ratio > 2)`,
            },
        ];
    }
    if (mode === "spot") {
        // inside the cone (screen centre) is spot-lit well above the ambient floor; outside (screen
        // corner) is ambient only — a plain omni point light would flood the corner too, so the dark
        // corner is what the cone confines. ceiling = ambient × albedo (the unlit floor luminance).
        const center = out[5];
        const corner = out[6];
        return [
            {
                name: "cone",
                pass: center > ceiling * 3 && corner < ceiling * 1.5,
                detail: `centre ${center.toFixed(3)} (lit, want > ${(ceiling * 3).toFixed(3)}), corner ${corner.toFixed(3)} (ambient, want < ${(ceiling * 1.5).toFixed(3)})`,
            },
        ];
    }
    if (mode === "spotShadow") {
        // the camera aims at the floor centre (screen centre), which the blocker's shadow darkens to
        // ~ambient; a cone-lit ring (the brightest half-peak) surrounds it. Without the spot casting,
        // the cone would light the centre as bright as the ring, so the dark centre IS the shadow.
        const center = out[5]; // the shadowed floor centre
        const ring = Math.max(out[1], out[2], out[3], out[4]); // the brightest cone-lit surroundings
        return [
            {
                name: "shadow",
                pass: ring > ceiling * 4 && center < ceiling * 2,
                detail: `shadowed centre ${center.toFixed(3)} (want < ${(ceiling * 2).toFixed(3)}) under a lit ring ${ring.toFixed(3)} (want > ${(ceiling * 4).toFixed(3)})`,
            },
        ];
    }
    if (mode === "pointShadow") {
        // the off-axis point caster's blocker shadows the floor centre (screen centre); a lit floor ring
        // surrounds it. The point shadow must be present (centre dark) under a lit ring — the same shape
        // as the spot-shadow check, exercising a point caster's off-axis (+X) cube face
        const center = out[5]; // the shadowed floor centre
        const ring = Math.max(out[1], out[2], out[3], out[4]); // the brightest lit floor
        return [
            {
                name: "pointShadow",
                pass: ring > ceiling * 4 && center < ceiling * 2,
                detail: `shadowed centre ${center.toFixed(3)} (want < ${(ceiling * 2).toFixed(3)}) under a lit ring ${ring.toFixed(3)} (want > ${(ceiling * 4).toFixed(3)})`,
            },
        ];
    }
    if (mode === "cascade" || mode === "cascade-ortho") {
        // the blocker's directional cascade shadow darkens the floor centre (screen centre) to ~ambient; a
        // sun-lit floor ring surrounds it. Without the cascade atlas casting, the sun would light the centre
        // as bright as the ring, so the dark centre IS the sun shadow — the directional twin of pointShadow.
        // `cascade-ortho` is the same assert under an ortho camera posed far from the scene: pre-fix its
        // frustum-slice fit never reached the ground, so the centre read lit (no shadow) — the regression
        const center = out[5]; // the sun-shadowed floor centre
        const ring = Math.max(out[1], out[2], out[3], out[4]); // the brightest sun-lit floor
        return [
            {
                name: mode === "cascade-ortho" ? "cascade-ortho" : "cascade",
                pass: ring > ceiling * 4 && center < ceiling * 2,
                detail: `shadowed centre ${center.toFixed(3)} (want < ${(ceiling * 2).toFixed(3)}) under a sun-lit ring ${ring.toFixed(3)} (want > ${(ceiling * 4).toFixed(3)})`,
            },
        ];
    }
    if (mode === "cascade-boundary") {
        // the high roof shadows the entire receding floor, which spans the cascade splits. Every visible floor
        // fragment must read ambient: a cascade-boundary leak lets the near cascade fail to capture the high
        // roof occluder, lighting a bright full-sun rim at the seam. So the brightest pixel anywhere in the
        // frame stays near the ambient floor (ceiling = ambient × albedo) — a rim spikes it toward the lit
        // value (~(ambient + sun)×albedo, tens× the ceiling). ×3 is the margin over sampling + MSAA edges
        const peak = Math.max(out[1], out[2], out[3], out[4]);
        return [
            {
                name: "boundary",
                pass: peak < ceiling * 3,
                detail: `brightest floor pixel ${peak.toFixed(3)} (want < ${(ceiling * 3).toFixed(3)} — uniformly shadowed, no cascade-seam rim)`,
            },
        ];
    }
    if (mode === "acne") {
        // the directly-lit flat floor (central in-cone window) must have NO self-shadowed fragment: its
        // worst (min) pixel stays well above the ambient-only ceiling. Acne speckles fragments down toward
        // ambient; ceiling×3 is the existing "clearly lit" bar (the cone check's), read on the worst pixel
        const lo = out[7];
        return [
            {
                name: "acne",
                pass: lo > ceiling * 3,
                detail: `worst lit pixel ${lo.toFixed(3)} (want > ${(ceiling * 3).toFixed(3)} — no self-shadow speckle)`,
            },
        ];
    }
    if (mode === "zfight") {
        // the front (bright, linear lum ~0.9) slab must win the depth test at ~5000 m over the back
        // (dark, ~0.05) slab 0.5 m behind it. reverse-Z + float resolves the separation → centre reads
        // the front; a forward-Z float buffer can't, so the back z-fights through and the centre drops
        // toward the dark back. 0.4 cleanly separates a stable front win from the fight
        const center = out[5];
        return [
            {
                name: "zfight",
                pass: center > 0.4,
                detail: `centre ${center.toFixed(3)} (want > 0.4 — front slab won; z-fight drops it toward the dark back ~0.05)`,
            },
        ];
    }
    // mode "lit": the floor can't exceed ambient × albedo without a point-light contribution (ambient is
    // flat, its half-Lambert is 1). Clear above that ceiling ⇒ the caster light reached the floor (no
    // self-occlusion). 2× is the margin over the small unlit fixture + the coarse sample grid.
    const avg = out[0];
    const threshold = ceiling * 2;
    return [
        {
            name: "lit",
            pass: avg > threshold,
            detail: `floor avg ${avg.toFixed(4)} ${avg > threshold ? ">" : "≤"} ${threshold.toFixed(4)} (ambient ceiling ${ceiling.toFixed(4)})`,
        },
    ];
}

// ============================================================================
// fog mode — the volumetric atmosphere, gated by a synthetic march probe
// ============================================================================
//
// The rendered scene (ground + receding box row + a Volumetric spot with an occluder + a Volumetric+Shadow
// sun) is the live visual. The GATE doesn't pixel-read the framebuffer (null post-present): a compute probe
// marches one synthetic ray through the SAME WGSL the production fog pass splices — extinction
// (FOG_MARCH_WGSL) + in-scatter (FOG_INSCATTER_WGSL: one shadow-free spot + the directional sun) — and the
// assert pins its readback to the TS oracle (fogTransmittance / fogComposite / fogInScatter / fogSunInScatter).
// GPU == TS. So the gate covers extinction (haze), spot in-scatter (shafts), and sun in-scatter (sun) in one
// page — independent of the rendered scene.

// a synthetic camera→fragment ray through the foggy near-ground region (transmittance ~0.8 at the defaults).
// Offset 0.5 (midpoint) makes both the probe and the oracle deterministic, jitter-independent.
const RAY_ORIGIN: [number, number, number] = [2, 8, 5];
const RAY_TARGET: [number, number, number] = [0, 0, 0];
const PROBE_SCENE: [number, number, number] = [0.8, 0.3, 0.2];
const PROBE_OFFSET = 0.5;

// the probe's single volumetric spot — above the origin, aimed down, so the probe ray sweeps from outside the
// cone into the inner cone. Shadow is forced 1 (no occluder) — the analytic-cone gate. params.y is the
// oct-packed axis the FS decodes, so the oracle reads the same oct round-trip.
const SPOT_POS: [number, number, number] = [0, 10, 0];
const SPOT_RANGE = 30;
const SPOT_RADIUS = 0.1;
const SPOT_COLOR: [number, number, number] = [6, 5.2, 4];
const SPOT_AXIS: [number, number, number] = [0, -1, 0];
const SPOT_INNER = 30;
const SPOT_OUTER = 55;
const { scale: SPOT_SCALE, offset: SPOT_OFFSET } = spotParams(SPOT_INNER, SPOT_OUTER);
const SPOT_AXIS_ENC = octEncodeNormal({ x: SPOT_AXIS[0], y: SPOT_AXIS[1], z: SPOT_AXIS[2] });
const SPOT_AXIS_DEC = octDecodeNormal(SPOT_AXIS_ENC);

// the oracle's light in the decoded terms the GPU march reads — coneAxis is the oct round-trip the FS applies
const PROBE_LIGHT: FogLight = {
    pos: SPOT_POS,
    invRangeSq: 1 / (SPOT_RANGE * SPOT_RANGE),
    radius: SPOT_RADIUS,
    color: SPOT_COLOR,
    coneAxis: [SPOT_AXIS_DEC.x, SPOT_AXIS_DEC.y, SPOT_AXIS_DEC.z],
    coneScale: SPOT_SCALE,
    coneOffset: SPOT_OFFSET,
};

// the probe's synthetic sun — a directional shaft marched shadow-free. `dir` is the travel direction
// (normalized like writeLighting); `color` has intensity baked, as the Lighting uniform carries it.
const SUN_DIR_RAW: [number, number, number] = [-0.4, -0.8, -0.45];
const SUN_DIR_LEN = Math.hypot(SUN_DIR_RAW[0], SUN_DIR_RAW[1], SUN_DIR_RAW[2]);
const PROBE_SUN_DIR: [number, number, number] = [
    SUN_DIR_RAW[0] / SUN_DIR_LEN,
    SUN_DIR_RAW[1] / SUN_DIR_LEN,
    SUN_DIR_RAW[2] / SUN_DIR_LEN,
];
const PROBE_SUN_COLOR: [number, number, number] = [3, 2.7, 2.2]; // rgb · intensity, baked

function fogProbeCode(): string {
    return /* wgsl */ `
${FOG_STRUCT_WGSL}
${POINT_LIGHTS_STRUCT_WGSL}
${OCT_ENCODE_WGSL}
${LIGHT_EVAL_WGSL}
${FOG_MARCH_WGSL}
${FOG_INSCATTER_WGSL}

struct Probe { origin: vec4<f32>, dir: vec4<f32>, scene: vec4<f32>, cfg: vec4<f32> }
struct Sun { dir: vec4<f32>, color: vec4<f32> }

@group(0) @binding(0) var<uniform> fog: Fog;
@group(0) @binding(1) var<uniform> probe: Probe;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> light: PointLightGpu;
@group(0) @binding(4) var<uniform> sun: Sun;

@compute @workgroup_size(1)
fn main() {
    let steps = max(u32(fog.extra.x), 1u);
    let t = fogTransmittance(
        probe.origin.xyz, probe.dir.xyz, probe.cfg.x,
        fog.march.x, fog.march.y, fog.march.z, steps, probe.cfg.y,
    );
    let c = fogComposite(probe.scene.rgb, fog.color.rgb, t);
    out[0] = vec4<f32>(t, c.r, c.g, c.b);

    let g = fog.extra.y;
    let absorption = fog.extra.z;
    let gain = fog.extra.w;
    let albedo = 1.0 - absorption;
    let ds = probe.cfg.x / f32(steps);
    var trans = 1.0;
    var inScatter = vec3<f32>(0.0);
    var sunScatter = vec3<f32>(0.0);
    for (var i = 0u; i < ${FOG_MAX_STEPS}u; i = i + 1u) {
        if (i >= steps) { break; }
        let p = probe.origin.xyz + probe.dir.xyz * ((f32(i) + probe.cfg.y) * ds);
        let dens = fogDensity(p, fog.march.x, fog.march.y, fog.march.z);
        let sampleTrans = exp(-dens * ds);
        let w = trans * albedo * gain * (1.0 - sampleTrans);
        inScatter += w * inScatterContribution(light, p, probe.dir.xyz, g);
        sunScatter += w * sunInScatter(sun.color.rgb, sun.dir.xyz, probe.dir.xyz, g);
        trans = trans * sampleTrans;
    }
    out[1] = vec4<f32>(inScatter, 0.0);
    out[2] = vec4<f32>(sunScatter, 0.0);
}`;
}

let fogEid = -1;
let fogRayOriginY = 0;
let fogRayDirY = 0;
let fogRayDist = 0;
let fogRayDir: [number, number, number] = [0, 0, 0];
let fogProbePipeline: GPUComputePipeline | null = null;
let fogBuf: GPUBuffer | null = null;
let fogCfgBuf: GPUBuffer | null = null;
let fogLightBuf: GPUBuffer | null = null;
let fogSunBuf: GPUBuffer | null = null;
let fogOutBuf: GPUBuffer | null = null;
let fogOutMirror: Mirror | null = null;
let fogProbeBg: GPUBindGroup | null = null;

// dispatch the probe each frame so Mirror has a fresh readback. A synthetic ray, independent of the rendered
// frame, so it only needs the encoder — `after: [BeginFrameSystem]`.
const FogProbeSystem: System = {
    name: "render-fog-probe",
    group: "draw",
    after: [BeginFrameSystem],
    update() {
        if (
            !Render.encoder ||
            !fogProbePipeline ||
            !fogBuf ||
            !fogCfgBuf ||
            !fogLightBuf ||
            !fogSunBuf ||
            !fogOutBuf
        )
            return;
        if (!fogProbeBg) {
            fogProbeBg = Compute.device.createBindGroup({
                label: "render-fog-probe",
                layout: fogProbePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: fogBuf } },
                    { binding: 1, resource: { buffer: fogCfgBuf } },
                    { binding: 2, resource: { buffer: fogOutBuf } },
                    { binding: 3, resource: { buffer: fogLightBuf } },
                    { binding: 4, resource: { buffer: fogSunBuf } },
                ],
            });
        }
        const pass = Render.encoder.beginComputePass({ label: "render-fog-probe" });
        pass.setPipeline(fogProbePipeline);
        pass.setBindGroup(0, fogProbeBg);
        pass.dispatchWorkgroups(1);
        pass.end();
    },
};

// ----------------------------------------------------------------------------
// the fogged-framebuffer gate: PROBE_WGSL twice over view.framebuffer — once pre-fog (FogClearProbeSystem,
// before the march) for the clear baseline, once post-fog (FogFoggedProbeSystem, after it) for the composite —
// so assertFog reads the SAME screen-centre pixel before and after the march. The synthetic oracle above pins
// the march *math*; this pins the *composite that lands on the scene*, catching a march that destroys the
// surface shading (the reverse-Z inversion regression, fog/index.ts reconstructWorld(uv, 1.0)) — a wash-out
// the math-only oracle passes green through. Two dedicated systems (not the shared ProbeSystem) so the pre/post
// ordering is pinned by `before: [FogSystem]` / `after: [FogSystem]`, which sit on either side of the march.
//
// The reference target is a bright box at the orbit pivot (screen centre, PROBE_WGSL's out[5]); the sun is
// non-volumetric in this scene so that box's view ray carries ZERO in-scatter (the spot's cone never reaches
// it), leaving pure extinction `centre·T + haze·(1−T)` to compare against the fogTransmittance oracle. Sun
// in-scatter math stays covered by the synthetic oracle (its ray is independent of the scene's Volumetric set).
let fogFbPipeline: GPUComputePipeline | null = null;
let fogClearBuf: GPUBuffer | null = null;
let fogClearMirror: Mirror | null = null;
let fogClearBg: GPUBindGroup | null = null;
let fogFoggedBuf: GPUBuffer | null = null;
let fogFoggedMirror: Mirror | null = null;
let fogFoggedBg: GPUBindGroup | null = null;

function fogFbProbe(buf: GPUBuffer, bg: GPUBindGroup | null, label: string): GPUBindGroup | null {
    if (!Render.encoder || !fogFbPipeline) return bg;
    const view = Views.get(cam);
    if (!view?.framebuffer) return bg;
    const group =
        bg ??
        Compute.device.createBindGroup({
            label,
            layout: fogFbPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: view.framebuffer },
                { binding: 1, resource: { buffer: buf } },
            ],
        });
    const pass = Render.encoder.beginComputePass({ label });
    pass.setPipeline(fogFbPipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(1);
    pass.end();
    return group;
}

const FogClearProbeSystem: System = {
    name: "render-fog-clear-probe",
    group: "draw",
    after: [ColorSystem],
    before: [FogSystem], // pre-march: reads the lit scene before the haze composites
    update() {
        if (fogClearBuf) fogClearBg = fogFbProbe(fogClearBuf, fogClearBg, "render-fog-clear-probe");
    },
};

const FogFoggedProbeSystem: System = {
    name: "render-fog-fogged-probe",
    group: "draw",
    after: [FogSystem], // post-march: reads the composited haze
    before: [GlazeSystem], // before glaze tonemaps + presents
    update() {
        if (fogFoggedBuf)
            fogFoggedBg = fogFbProbe(fogFoggedBuf, fogFoggedBg, "render-fog-fogged-probe");
    },
};

async function setupFoggedProbe(state: State): Promise<void> {
    fogFbPipeline = await Compute.device.createComputePipelineAsync({
        label: "render-fog-fb-probe",
        layout: "auto",
        compute: {
            module: Compute.device.createShaderModule({
                label: "render-fog-fb-probe",
                code: PROBE_WGSL,
            }),
            entryPoint: "main",
        },
    });
    fogClearBuf = Compute.device.createBuffer({
        label: "render-fog-clear-probe",
        size: 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    fogFoggedBuf = Compute.device.createBuffer({
        label: "render-fog-fogged-probe",
        size: 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    fogClearBg = null;
    fogFoggedBg = null;
    state.addSystem(FogClearProbeSystem);
    state.addSystem(FogFoggedProbeSystem);
    await frames(4);
    fogClearMirror = mirror(fogClearBuf);
    fogFoggedMirror = mirror(fogFoggedBuf);
    await frames(3);
}

// the screen-centre reference box (= the orbit pivot, so it projects to uv (0.5, 0.5) = PROBE_WGSL's out[5]).
// White + lit, so its clear luminance sits clearly above the haze — the band `(fogged − haze)/(clear − haze)`
// is then well-conditioned and the extinction loss is a luminance DROP toward haze the inverted march can't fake.
const FOG_REF_CENTER: [number, number, number] = [0, 2, -18];
const FOG_REF_HALF = 1.5;

async function buildFog(state: State, p: Params): Promise<void> {
    const ambient = state.create();
    state.add(ambient, AmbientLight);
    AmbientLight.color.set(ambient, 0xdfe7f0);
    AmbientLight.intensity.set(ambient, 0.5);

    const sun = state.create();
    state.add(sun, DirectionalLight);
    DirectionalLight.color.set(sun, 0xfff2dc);
    DirectionalLight.intensity.set(sun, 1.4);
    DirectionalLight.direction.set(sun, -0.4, -0.8, -0.45, 0);
    // the sun casts surface shadows but is deliberately NOT Volumetric: that keeps the reference-box ray
    // (the extinction gate below) free of in-scatter. Sun-shaft math is covered by the synthetic oracle.
    state.add(sun, Shadow);

    const floor = state.create();
    state.add(floor, Transform);
    Transform.pos.set(floor, 0, 0, -40, 0);
    Transform.scale.set(floor, 30, 0.2, 120, 0);
    state.add(floor, Part);
    state.add(floor, Color);
    Color.rgba.set(floor, 0.34, 0.42, 0.3, 1);

    // a row of boxes receding into the distance — the near ones read clear, the far ones fade into haze
    for (let i = 0; i < 10; i++) {
        const z = -8 - i * 11;
        for (const x of [-6, 6]) {
            const box = state.create();
            state.add(box, Transform);
            Transform.pos.set(box, x, 1.5, z, 0);
            Transform.scale.set(box, 2, 3, 2, 0);
            state.add(box, Part);
            state.add(box, Color);
            Color.rgba.set(box, 0.7, 0.5, 0.4, 1);
        }
    }

    // the framebuffer gate's reference: a bright box at the orbit pivot (screen centre), brighter than the
    // haze so the march reads as a luminance drop toward haze. x=0 is clear of the x=±6 box row, so the
    // centre ray hits it first.
    const ref = state.create();
    state.add(ref, Transform);
    Transform.pos.set(ref, FOG_REF_CENTER[0], FOG_REF_CENTER[1], FOG_REF_CENTER[2], 0);
    Transform.scale.set(ref, FOG_REF_HALF * 2, FOG_REF_HALF * 2, FOG_REF_HALF * 2, 0);
    state.add(ref, Part);
    state.add(ref, Color);
    Color.rgba.set(ref, 1, 1, 1, 1);

    // a volumetric spot + occluder for the shadowed-shaft live visual (the probe gate is independent)
    const spot = state.create();
    state.add(spot, Transform);
    Transform.pos.set(spot, 0, 12, -30, 0);
    const aim = quat(-90, 0, 0); // aim the entity forward (-Z) straight down (-Y)
    Transform.rot.set(spot, aim.x, aim.y, aim.z, aim.w);
    state.add(spot, PointLight);
    PointLight.color.set(spot, 0xfff0d0);
    PointLight.intensity.set(spot, 40);
    PointLight.range.set(spot, 45);
    state.add(spot, Spot);
    Spot.inner.set(spot, 14);
    Spot.outer.set(spot, 30);
    state.add(spot, Shadow);
    state.add(spot, Volumetric);

    const occluder = state.create();
    state.add(occluder, Transform);
    Transform.pos.set(occluder, 0, 7, -30, 0);
    Transform.scale.set(occluder, 2.5, 0.5, 2.5, 0);
    state.add(occluder, Part);
    state.add(occluder, Color);
    Color.rgba.set(occluder, 0.3, 0.3, 0.35, 1);

    const fog = state.create();
    state.add(fog, Fog);
    Fog.density.set(fog, p.density as number);
    Fog.color.set(fog, 0xb5c4d8);
    Fog.heightBase.set(fog, 0);
    Fog.heightFalloff.set(fog, p.falloff as number);
    Fog.steps.set(fog, p.steps as number);
    Fog.jitter.set(fog, 1);
    // the in-scatter knobs the march reads — non-default so the probe + oracle exercise them
    Fog.absorption.set(fog, 0.1);
    Fog.scattering.set(fog, 8);
    Fog.anisotropy.set(fog, 0.6);
    Fog.scatterIntensity.set(fog, 2);
    fogEid = fog;

    cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Depth); // the fog march needs the depth lane — no auto-add
    state.add(cam, Orbit);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 60);
    Camera.far.set(cam, 300);
    Camera.clearColor.set(cam, 0xb5c4d8); // sky = haze color, so the horizon dissolves into fog
    Orbit.distance.set(cam, 18);
    Orbit.pitch.set(cam, 0.2);
    Orbit.yaw.set(cam, 0.3);
    Orbit.pan.set(cam, 0, 2, -18, 0);

    // the probe ray + its packed uniforms (static — the Fog config doesn't change within a build)
    const dx = RAY_TARGET[0] - RAY_ORIGIN[0];
    const dy = RAY_TARGET[1] - RAY_ORIGIN[1];
    const dz = RAY_TARGET[2] - RAY_ORIGIN[2];
    fogRayDist = Math.hypot(dx, dy, dz);
    fogRayOriginY = RAY_ORIGIN[1];
    fogRayDirY = dy / fogRayDist;
    fogRayDir = [dx / fogRayDist, fogRayDirY, dz / fogRayDist];

    fogBuf = Compute.device.createBuffer({
        label: "render-fog-config",
        size: FOG_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const fogStaging = new Float32Array(FOG_FLOATS);
    packFog(fogEid, fogStaging);
    Compute.device.queue.writeBuffer(fogBuf, 0, fogStaging);

    fogCfgBuf = Compute.device.createBuffer({
        label: "render-fog-ray",
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const cfg = new Float32Array(16);
    cfg.set([RAY_ORIGIN[0], RAY_ORIGIN[1], RAY_ORIGIN[2], 0], 0);
    cfg.set([dx / fogRayDist, fogRayDirY, dz / fogRayDist, 0], 4);
    cfg.set([PROBE_SCENE[0], PROBE_SCENE[1], PROBE_SCENE[2], 0], 8);
    cfg.set([fogRayDist, PROBE_OFFSET, 0, 0], 12);
    Compute.device.queue.writeBuffer(fogCfgBuf, 0, cfg);

    // the single volumetric spot the in-scatter probe marches (a PointLightGpu uniform). params.y is the
    // oct-packed cone axis (bitcast), so the FS spotFactor decodes the same axis PROBE_LIGHT.coneAxis holds
    fogLightBuf = Compute.device.createBuffer({
        label: "render-fog-light",
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const lightF = new Float32Array(12);
    lightF.set([SPOT_POS[0], SPOT_POS[1], SPOT_POS[2], 1 / (SPOT_RANGE * SPOT_RANGE)], 0);
    lightF.set([SPOT_COLOR[0], SPOT_COLOR[1], SPOT_COLOR[2], 0], 4);
    // params.x = -radius (the Volumetric sign flag; squared on read, sign-immune), z/w = cone scale/offset
    lightF.set([-SPOT_RADIUS, 0, SPOT_SCALE, SPOT_OFFSET], 8);
    new Uint32Array(lightF.buffer)[9] = SPOT_AXIS_ENC;
    Compute.device.queue.writeBuffer(fogLightBuf, 0, lightF);

    // the synthetic directional sun the sun-in-scatter probe marches: dir (travel direction) + color
    fogSunBuf = Compute.device.createBuffer({
        label: "render-fog-sun",
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const sunF = new Float32Array(8);
    sunF.set([PROBE_SUN_DIR[0], PROBE_SUN_DIR[1], PROBE_SUN_DIR[2], 0], 0);
    sunF.set([PROBE_SUN_COLOR[0], PROBE_SUN_COLOR[1], PROBE_SUN_COLOR[2], 0], 4);
    Compute.device.queue.writeBuffer(fogSunBuf, 0, sunF);

    // three vec4 results: [0] transmittance + composite, [1] spot in-scatter, [2] sun in-scatter
    fogOutBuf = Compute.device.createBuffer({
        label: "render-fog-out",
        size: 48,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    fogProbeBg = null;
    fogProbePipeline = await Compute.device.createComputePipelineAsync({
        label: "render-fog-probe",
        layout: "auto",
        compute: {
            module: Compute.device.createShaderModule({
                label: "render-fog-probe",
                code: fogProbeCode(),
            }),
            entryPoint: "main",
        },
    });
    state.addSystem(FogProbeSystem);

    await frames(4);
    fogOutMirror = mirror(fogOutBuf);
    await frames(3);

    // the framebuffer gate: the pre-fog clear baseline + the post-fog composite, read at screen centre (out[5])
    await setupFoggedProbe(state);
}

// nearest-hit distance of the centre ray against the reference box's AABB (slab method). The ray origin is
// the near-plane point the fog march itself starts from, so the derived march distance matches the shader's.
function fogRefHit(): { originY: number; dirY: number; dist: number } | null {
    const view = Views.get(cam);
    if (!view) return null;
    const vp = new Float32Array(16);
    computeViewProj(cam, view.width / view.height, vp);
    const inv = invert(vp, new Float32Array(16));
    const unproject = (z: number): [number, number, number] => {
        // centre pixel ndc = (0, 0, z); reverse-Z near plane is z=1, like the fog shader's reconstructWorld.
        // ndc.xy = 0 drops the first two matrix columns, leaving the z-column (8..11) + translation (12..15).
        const x = inv[8] * z + inv[12];
        const y = inv[9] * z + inv[13];
        const w2 = inv[10] * z + inv[14];
        const ww = inv[11] * z + inv[15];
        return [x / ww, y / ww, w2 / ww];
    };
    const o = unproject(1);
    const f = unproject(0);
    const d: [number, number, number] = [f[0] - o[0], f[1] - o[1], f[2] - o[2]];
    const dl = Math.hypot(d[0], d[1], d[2]);
    d[0] /= dl;
    d[1] /= dl;
    d[2] /= dl;
    let tNear = -Infinity;
    let tFar = Infinity;
    for (let i = 0; i < 3; i++) {
        const lo = (FOG_REF_CENTER[i] - FOG_REF_HALF - o[i]) / d[i];
        const hi = (FOG_REF_CENTER[i] + FOG_REF_HALF - o[i]) / d[i];
        tNear = Math.max(tNear, Math.min(lo, hi));
        tFar = Math.min(tFar, Math.max(lo, hi));
    }
    if (tNear > tFar || tFar < 0) return null;
    return { originY: o[1], dirY: d[1], dist: Math.max(tNear, 0) };
}

async function assertFog(): Promise<Check[]> {
    const checks: Check[] = [];

    const steps = Math.min(Math.max(Fog.steps.get(fogEid), 1), FOG_MAX_STEPS);

    // perf: the march span must fire with real GPU time, and its PER-STEP cost (against the color pass, both
    // full-frame) stays bounded. Per-step, not absolute, because the march is linear in step count: dividing the
    // fog/color ratio by `steps` is the device-tolerant invariant (~0.5 on lovelace across steps 8/32/256), and
    // it's what cleanly separates a legitimate high `steps` from the one steady regression a mean can see — a
    // failed loop unroll (256 iterations regardless of the active count), which spikes per-step cost ~8× at the
    // default 32 and far more at low counts. Only the per-pass MEAN (gpuTime/gpuFires) is readable in-assert; the
    // occP99 spike lives on the benchmark window the harness owns, read from the bench print + the stress sweep.
    const fogFires = Profile.gpuFires.get("fog:march") ?? 0;
    const colorFires = Profile.gpuFires.get("sear:color") ?? 0;
    const fogOcc = fogFires > 0 ? (Profile.gpuTime.get("fog:march") ?? 0) / fogFires : 0;
    const colorOcc = colorFires > 0 ? (Profile.gpuTime.get("sear:color") ?? 0) / colorFires : 0;
    const perStep = colorOcc > 0 ? fogOcc / colorOcc / steps : Infinity;
    const PerStepMax = 1.5;
    checks.push({
        name: "march-span",
        pass: fogOcc > 0,
        detail: fogOcc > 0 ? `fog:march ${fogOcc.toFixed(4)}ms/occ` : "fog:march never fired",
    });
    checks.push({
        name: "march-cost",
        pass: perStep < PerStepMax,
        detail: `fog:march/sear:color/step ${perStep.toFixed(3)} (want < ${PerStepMax} — unroll-failure tripwire)`,
    });

    // the framebuffer extinction gate: the bright reference box at screen centre must survive the march as
    // `centre·T + haze·(1−T)` (its zero-in-scatter ray makes this exact). The inverted-march regression
    // over-extincts it toward haze, collapsing the band far below T — the wash-out the synthetic oracle misses.
    const hit = fogRefHit();
    if (fogClearMirror && fogFoggedMirror && hit) {
        await settle(fogClearMirror);
        await settle(fogFoggedMirror);
        const clear = fogClearMirror.snapshot
            ? new Float32Array(fogClearMirror.snapshot.bytes)[5]
            : Number.NaN;
        const fogged = fogFoggedMirror.snapshot
            ? new Float32Array(fogFoggedMirror.snapshot.bytes)[5]
            : Number.NaN;
        const fc = unpackColor(Fog.color.get(fogEid));
        const hazeLum = 0.2126 * fc.r + 0.7152 * fc.g + 0.0722 * fc.b; // Rec709, matching PROBE_WGSL
        const refT = fogTransmittance(
            hit.originY,
            hit.dirY,
            hit.dist,
            Fog.density.get(fogEid),
            Fog.heightBase.get(fogEid),
            Fog.heightFalloff.get(fogEid),
            steps,
            PROBE_OFFSET,
        );
        const composite = clear * refT + hazeLum * (1 - refT);
        // tolerance: flat lit face → near-uniform window, so the budget is MSAA silhouette + box-depth ΔT +
        // the f32-vs-f64 march, ~5% + a floor. The regression margin is |haze − composite| ≈ 27% — far outside.
        const tol = 0.05 * composite + 0.02;
        const bright = clear > hazeLum * 1.1; // the reference must be brighter than haze for the band to read
        checks.push({
            name: "extinction",
            pass: bright && Math.abs(fogged - composite) < tol,
            detail: `centre fogged ${fogged.toFixed(3)} vs composite ${composite.toFixed(3)} (clear ${clear.toFixed(3)}·T${refT.toFixed(2)} + haze ${hazeLum.toFixed(3)}, tol ${tol.toFixed(3)})`,
        });
    } else {
        checks.push({ name: "extinction", pass: false, detail: "no framebuffer probe / ray miss" });
    }

    if (!fogOutMirror)
        return [...checks, { name: "probe", pass: false, detail: "no probe mirror" }];
    await settle(fogOutMirror);
    if (!fogOutMirror.snapshot)
        return [...checks, { name: "probe", pass: false, detail: "no probe snapshot" }];

    const out = new Float32Array(fogOutMirror.snapshot.bytes);
    const gpuT = out[0];
    const gpuC: [number, number, number] = [out[1], out[2], out[3]];

    // the oracle reads the SAME Fog values packFog packed (f32 component reads), so only the march arithmetic
    // diverges (f32 vs f64). 1e-4 bounds ~32 f32 march steps + the exp; a real divergence is far larger.
    const density = Fog.density.get(fogEid);
    const base = Fog.heightBase.get(fogEid);
    const falloff = Fog.heightFalloff.get(fogEid);
    const c = unpackColor(Fog.color.get(fogEid));
    const wantT = fogTransmittance(
        fogRayOriginY,
        fogRayDirY,
        fogRayDist,
        density,
        base,
        falloff,
        steps,
        PROBE_OFFSET,
    );
    const wantC = fogComposite(PROBE_SCENE, [c.r, c.g, c.b], wantT);
    const Tol = 1e-4;

    checks.push({
        name: "transmittance",
        pass: Math.abs(gpuT - wantT) < Tol,
        detail: `gpu ${gpuT.toFixed(6)} vs oracle ${wantT.toFixed(6)}`,
    });
    const cErr = Math.max(
        Math.abs(gpuC[0] - wantC[0]),
        Math.abs(gpuC[1] - wantC[1]),
        Math.abs(gpuC[2] - wantC[2]),
    );
    checks.push({
        name: "composite",
        pass: cErr < Tol,
        detail: `max channel err ${cErr.toExponential(2)} (gpu ${gpuC.map((v) => v.toFixed(3)).join(",")})`,
    });

    const scatter = {
        density,
        base,
        falloff,
        absorption: Fog.absorption.get(fogEid),
        gain: Fog.scattering.get(fogEid) * Fog.scatterIntensity.get(fogEid),
        anisotropy: Fog.anisotropy.get(fogEid),
    };

    // in-scatter (S2): the GPU march of the single volumetric spot (shadow-free) vs the TS oracle, same
    // midpoint samples. A relative bound (per-step f32 rounding accumulates); `mag > 1e-5` guards a vacuous
    // all-zero pass (the ray must be lit). Observed ~3e-7 relative on lovelace, so 1e-4 is a ~300× margin.
    const gpuIn: [number, number, number] = [out[4], out[5], out[6]];
    const wantIn = fogInScatter(
        RAY_ORIGIN,
        fogRayDir,
        fogRayDist,
        scatter,
        PROBE_LIGHT,
        steps,
        PROBE_OFFSET,
    );
    const inMag = Math.max(Math.abs(wantIn[0]), Math.abs(wantIn[1]), Math.abs(wantIn[2]));
    const inErr = Math.max(
        Math.abs(gpuIn[0] - wantIn[0]),
        Math.abs(gpuIn[1] - wantIn[1]),
        Math.abs(gpuIn[2] - wantIn[2]),
    );
    checks.push({
        name: "in-scatter",
        pass: inMag > 1e-5 && inErr <= 1e-9 + 1e-4 * inMag,
        detail: `err ${inErr.toExponential(2)} / mag ${inMag.toExponential(2)}`,
    });

    // sun in-scatter (S3): the GPU march of the synthetic directional sun (shadow-free) vs the TS oracle
    const gpuSun: [number, number, number] = [out[8], out[9], out[10]];
    const wantSun = fogSunInScatter(
        RAY_ORIGIN,
        fogRayDir,
        fogRayDist,
        scatter,
        { direction: PROBE_SUN_DIR, color: PROBE_SUN_COLOR },
        steps,
        PROBE_OFFSET,
    );
    const sunMag = Math.max(Math.abs(wantSun[0]), Math.abs(wantSun[1]), Math.abs(wantSun[2]));
    const sunErr = Math.max(
        Math.abs(gpuSun[0] - wantSun[0]),
        Math.abs(gpuSun[1] - wantSun[1]),
        Math.abs(gpuSun[2] - wantSun[2]),
    );
    checks.push({
        name: "sun-in-scatter",
        pass: sunMag > 1e-5 && sunErr <= 1e-9 + 1e-4 * sunMag,
        detail: `err ${sunErr.toExponential(2)} / mag ${sunMag.toExponential(2)}`,
    });
    return checks;
}

// ============================================================================
// gltf modes — the asset import path (structural reads, no probe)
// ============================================================================
//
// `gltf-model` loads the Sponza atrium under a few lanterns; `gltf-animated` loads the Fox baked to a vertex
// animation texture. The gate is structural — it reads the published import artifacts (Part / Material counts,
// the bucketed baseColor `texture_2d_array`s, the per-mesh VAT textures, the material palette) plus the right
// passes firing — coverage the CPU-deviceless `parse` tests can't reach. `variant` (gltf-model) picks the
// codec path; `ktx`/`ktx-draco` drive the compressed bucket path; `gltf-spill` (below) the over-cap union.

// Sponza decodes to 124 node instances; every material is textured. Its 25 distinct baseColor images become 25
// array layers across the per-size buckets. Materials 1/2/19 are alphaMode MASK → 6 ride the `clip` cutout.
const SPONZA_INSTANCES = 124;
const ALBEDO_LAYERS = 25;
const CLIP_INSTANCES = 6;
// a transcoded KTX2 baseColor bucket lands as one of these block families; the RGBA last resort is
// rgba8unorm-srgb, so matching here is exactly "the compressed bucketed path ran, not the fallback"
const COMPRESSED = /^(bc\d+|etc2|astc)-/;

// the same atrium in four packed-asset variants — same decoded scene, only the codec path differs
const GLTF_VARIANTS: Record<string, { file: string; compressed: boolean }> = {
    gltf: { file: "sponza/Sponza.gltf", compressed: false },
    draco: { file: "sponza/Sponza-Draco.glb", compressed: false },
    ktx: { file: "sponza/Sponza-KTX.glb", compressed: true },
    "ktx-draco": { file: "sponza/Sponza-KTX-Draco.glb", compressed: true },
};

const FOX = "gltf-samples/Fox/glTF/Fox.gltf";
const FOX_CLIPS = ["survey", "walk", "run"];

// the size-bucket spill case: a union of two KTX assets whose baseColor textures span SIX distinct compressed
// sizes (StainedGlassLamp {1024², 512², 2048², 2048×1024} + ChronographWatch {1024×256, 512×256, 2048²}) — past
// the ALBEDO_BUCKETS (4) cap, so the three rarest sizes spill, decoded to RGBA into the shared bitmap bucket.
const SPILL_ASSETS = [
    "gltf-samples/StainedGlassLamp/glTF-KTX-BasisU/StainedGlassLamp.gltf",
    "gltf-samples/ChronographWatch/glTF-KTX-BasisU/ChronographWatch.gltf",
];

const gltfVariant = (p: Params) =>
    GLTF_VARIANTS[(p.variant as string) ?? "gltf"] ?? GLTF_VARIANTS.gltf;

// gltf-multi: two textured sources (DamagedHelmet, WaterBottle) + two skinned (Fox, CesiumMan) loaded
// imperatively after build (placeMulti) so the union accumulates all four; the gate reads palette/VAT.
const MULTI = {
    helmet: "gltf-samples/DamagedHelmet/glTF/DamagedHelmet.gltf",
    bottle: "gltf-samples/WaterBottle/glTF/WaterBottle.gltf",
    fox: "gltf-samples/Fox/glTF/Fox.gltf",
    cesium: "gltf-samples/CesiumMan/glTF/CesiumMan.gltf",
};

// gltf-worker fixtures — the off-thread decode-pool proof. Box/Draco + AnisotropyBarnLamp/KTX live in the
// gltf-samples submodule; box-meshopt.glb is a local fixture (Box through `gltfpack -c`: EXT_meshopt +
// KHR_mesh_quantization). CORRUPT is a real 200 that isn't valid glTF — the worker error-boundary case.
const WORKER_DRACO = "gltf-samples/Box/glTF-Draco/Box.gltf";
const WORKER_KTX = "gltf-samples/AnisotropyBarnLamp/glTF-KTX-BasisU/AnisotropyBarnLamp.gltf";
const WORKER_MESHOPT = "box-meshopt.glb";
const WORKER_CORRUPT = "gltf-samples/Box/glTF-Draco/Box.bin";
// the declarative textured case — authored by name with no import code, so the preloader + route sync
// carry it: import, surface routing, and the Textured decoration all scene-derived
const WORKER_TEXTURED = "gltf-samples/BoxTextured/glTF-Binary/BoxTextured.glb";

// the gltf-model env: a couple of lanterns down the nave (one casting) so light:cull + sear:pointshadow fire
// on the textured scene, plus a casting sun and an orbit framed on the atrium. The Fox env is the `gltf-animated` mode's.
function gltfScene(): string {
    if (mode === "gltf-multi") {
        // an empty stage — placeMulti loads the four assets imperatively after build, so the shared union
        // accumulates all four (a single scene-authored asset can't drive the multi-source accumulation)
        return `<scene>
            <a ambient-light="intensity: 0.6" />
            <a directional-light="intensity: 1.6" shadow="distance: 30" />
            <a camera="clear-color: 0x6a7a8a" sear
               orbit="pan: 0 1.2 0; distance: 13; max-distance: 60; min-distance: 4; yaw: 0.5; pitch: 0.18"
               transform />
        </scene>`;
    }
    if (mode === "gltf-worker") {
        // both boxes are authored by name with NO import code — the declarative gate: GltfPlugin's
        // preloader imports them before load, and the route sync decorates the textured one. The look
        // isn't the point; the worker round-trip runs in assert (decode / decodeInWorker bypass the cache).
        return `<scene>
            <a ambient-light="intensity: 0.6" />
            <a directional-light="intensity: 1.8; direction: -0.5 -0.8 -0.35" />
            <a camera="clear-color: 0x202428" sear
               orbit="distance: 5; yaw: 0.6; pitch: 0.3" transform />
            <a part="mesh: ${WORKER_DRACO}#0" transform="pos: -1.2 0 0" color="rgba: 0.72 0.7 0.78" />
            <a part="mesh: ${WORKER_TEXTURED}#0" transform="pos: 1.2 0 0" color />
        </scene>`;
    }
    if (mode === "gltf-animated") {
        return `<scene>
            <a ambient-light="intensity: 0.55" />
            <a directional-light="intensity: 1.6" shadow="distance: 220" />
            <a camera="clear-color: 0x6a7a8a" sear
               orbit="pan: 0 35 -10; distance: 240; max-distance: 700; min-distance: 20; yaw: 0.9; pitch: 0.32"
               transform />
        </scene>`;
    }
    if (mode === "gltf-spill") {
        // two KTX assets whose union exceeds the size-bucket cap → the real-device spill path (KTX→RGBA
        // transcode into the shared bitmap bucket). Overlap is fine — the gate reads the texture buckets.
        return `<scene>
            <a ambient-light="intensity: 0.6" />
            <a directional-light="intensity: 1.4; direction: -0.4 -0.8 -0.35" />
            <a camera="clear-color: 0x202428" sear
               orbit="pan: 0 0.4 0; distance: 5; yaw: 0.6; pitch: 0.25" transform />
        </scene>`;
    }
    return `<scene>
        <a ambient-light="color: 0x39435a; intensity: 0.4" />
        <a directional-light="intensity: 0.6; direction: -0.4 -0.8 -0.45" shadow="distance: 30" />
        <a point-light="color: 0xffe6c8; range: 11; intensity: 6" transform="pos: -0.5 5.5 -0.3" shadow />
        <a point-light="color: 0xffc080; range: 7; intensity: 3" transform="pos: -6 3 -3.9" />
        <a point-light="color: 0xffc080; range: 7; intensity: 3" transform="pos: 5 3 3.3" />
        <a camera="clear-color: 0x0a0c12" sear
           orbit="pan: 0 3 0; distance: 14; max-distance: 40; yaw: 0.6; pitch: 0.25" transform />
    </scene>`;
}

// load each gltf mode's asset(s) imperatively after build — the importer is a one-way utility (it registers
// meshes + returns a descriptor, no entities), so the scene authors only the env + camera and placeScene
// spawns the asset. The multi/spill modes load several so the shared union accumulates across them.
async function placeGltfAssets(state: State, p: Params): Promise<void> {
    if (mode === "gltf-multi") return placeMulti(state);
    if (mode === "gltf-worker") return; // the scene authors both boxes by name; the preloader imports them
    if (mode === "gltf-animated") {
        const clip = Math.max(0, FOX_CLIPS.indexOf((p.clip as string) ?? "walk"));
        placeScene(state, await loadGltf(state, FOX, { clip }));
        return;
    }
    if (mode === "gltf-spill") {
        for (const src of SPILL_ASSETS) placeScene(state, await loadGltf(state, src));
        return;
    }
    placeScene(state, await loadGltf(state, gltfVariant(p).file)); // gltf-model
}

// the skinned mesh's VAT is bound per-mesh (Mesh.bindings), not published globally, so read it off the
// skinned entity's mesh. Returns its position/normal textures + params buffer, or null.
function skinVat(state: State): { pos?: GPUTexture; norm?: GPUTexture; params?: GPUBuffer } | null {
    const eid = [...state.query([Skin])][0];
    if (eid === undefined) return null;
    const b = Meshes.get(Meshes.name(Part.mesh.get(eid)) ?? "")?.bindings;
    if (!b) return null;
    return {
        pos: b.vatPos as GPUTexture,
        norm: b.vatNorm as GPUTexture,
        params: b.vatParams as GPUBuffer,
    };
}

function clipCount(state: State): number {
    const clip = Surfaces.id("gltf-albedo-clip");
    let n = 0;
    for (const eid of state.query([Part])) if (Part.surface.get(eid) === clip) n++;
    return n;
}

async function assertGltfModel(state: State): Promise<Check[]> {
    while (unionPending()) await frames(1); // the union uploads across frames now — wait for the published set
    const compressed = params ? gltfVariant(params).compressed : false;
    const parts = [...state.query([Part])].length;
    const textured = [...state.query([Textured])].length;
    const clip = clipCount(state);
    // the real baseColor buckets (a 1×1 is an unused-bucket fallback); sum their layers + check the format
    const buckets = ALBEDO_NAMES.map((n) => Compute.textures.get(n)).filter(
        (t): t is GPUTexture => !!t && (t.width > 1 || t.depthOrArrayLayers > 1),
    );
    const albedoLayers = buckets.reduce((s, t) => s + t.depthOrArrayLayers, 0);
    const materialData = Compute.buffers.get("materialData");
    await frames(4);
    const cull = Profile.gpu.has("light:cull");
    const color = Profile.gpu.has("sear:color");
    const point = Profile.gpu.has("sear:pointshadow");
    return [
        {
            name: "sponza imported",
            pass: parts === SPONZA_INSTANCES,
            detail: `${parts} Part entities (expected ${SPONZA_INSTANCES})`,
        },
        {
            name: "all instances textured",
            pass: textured === SPONZA_INSTANCES,
            detail: `${textured} Material instances (expected ${SPONZA_INSTANCES})`,
        },
        {
            // layers sum to 25 across the size buckets; each KTX bucket stays a compressed block format (not
            // the RGBA last resort), squares only checked on the resize path. The compressed branch subsumes
            // the ktx-array gate (the compressedAlbedoArray path ran, not the fallback).
            name: "albedo size-buckets built",
            pass:
                albedoLayers === ALBEDO_LAYERS &&
                (!compressed || buckets.every((t) => COMPRESSED.test(t.format))) &&
                (compressed || buckets.every((t) => t.width === t.height)),
            detail: buckets.length
                ? `${buckets.length} bucket(s) [${buckets.map((t) => `${t.format} ${t.width}²×${t.depthOrArrayLayers}`).join(", ")}], ${albedoLayers} layers (expected ${ALBEDO_LAYERS})`
                : "no albedo buckets published",
        },
        {
            name: "material palette published",
            pass: !!materialData,
            detail: materialData
                ? `materialData ${materialData.size}B`
                : "materialData not published",
        },
        {
            name: "MASK foliage on the clip surface",
            pass: clip === CLIP_INSTANCES,
            detail: `${clip} clip instances (expected ${CLIP_INSTANCES})`,
        },
        {
            name: "point-light passes",
            pass: cull && color && point,
            detail: `light:cull ${cull}, sear:color ${color}, sear:pointshadow ${point}`,
        },
    ];
}

async function assertGltfAnimated(state: State): Promise<Check[]> {
    while (unionPending()) await frames(1); // the union uploads across frames now — wait for the published set
    const parts = [...state.query([Part])].length;
    const skinned = [...state.query([Skin])].length;
    const vat = skinVat(state);
    const vatPos = vat?.pos;
    const vatNorm = vat?.norm;
    const vatParams = vat?.params;
    const albedo = Compute.textures.get("albedo0");
    await frames(4);
    const color = Profile.gpu.has("sear:color");
    return [
        { name: "fox imported", pass: parts === 1, detail: `${parts} Part entities (expected 1)` },
        {
            name: "skinned instance",
            pass: skinned === 1,
            detail: `${skinned} Skin instances (expected 1)`,
        },
        {
            // a real VAT (more than the 1×1 fallback): vertices × frames, the unorm16-AABB position map
            name: "VAT position texture",
            pass:
                !!vatPos &&
                vatPos.format === "rgba16float" &&
                vatPos.width > 1 &&
                vatPos.height > 1,
            detail: vatPos
                ? `${vatPos.format} ${vatPos.width}×${vatPos.height} (verts × frames)`
                : "vatPos not published",
        },
        {
            // the vec3 normal map, same vertex extent as the position map
            name: "VAT normal texture",
            pass: !!vatNorm && vatNorm.format === "rgba16float" && vatNorm.width === vatPos?.width,
            detail: vatNorm
                ? `${vatNorm.format} ${vatNorm.width}×${vatNorm.height}`
                : "vatNorm not published",
        },
        {
            name: "VAT params + albedo published",
            pass: !!vatParams && !!albedo,
            detail: `vatParams ${vatParams ? `${vatParams.size}B` : "—"}, albedo ${albedo ? "yes" : "no"}`,
        },
        { name: "skin surface draws", pass: color, detail: `sear:color ${color}` },
    ];
}

// the size-bucket spill end-to-end gate: the union of two KTX assets spans 6 distinct compressed baseColor
// sizes, past the 4-bucket cap, so the three rarest spill — decoded to RGBA into the shared bitmap bucket on a
// real device (the `assembleUnion` spill branch the pure `planAlbedoBuckets` unit test can't reach). The proof
// is a compressed bucket and a real (non-fallback) rgba8unorm-srgb spill bucket coexisting among the published
// albedo arrays, the textured scene rendering. (The pure bucket-planning logic is `union.test.ts`.)
async function assertGltfSpill(state: State): Promise<Check[]> {
    while (unionPending()) await frames(1); // the union uploads across frames now — wait for the published set
    const textured = [...state.query([Textured])].length;
    const buckets = ALBEDO_NAMES.map((n) => Compute.textures.get(n)).filter(
        (t): t is GPUTexture => !!t && (t.width > 1 || t.depthOrArrayLayers > 1),
    );
    const compressed = buckets.filter((t) => COMPRESSED.test(t.format));
    // the spill bucket: a real (resized, width > 1 — not the 1×1 fallback) RGBA array holding the transcoded
    // overflow. arrayFromBitmaps publishes albedo as rgba8unorm-srgb, the format the compressed regex excludes.
    const spill = buckets.filter((t) => t.format === "rgba8unorm-srgb" && t.width > 1);
    await frames(4);
    const color = Profile.gpu.has("sear:color");
    const layers = buckets.reduce((s, t) => s + t.depthOrArrayLayers, 0);
    return [
        {
            name: "compressed bucket built",
            pass: compressed.length >= 1,
            detail: `${compressed.length} compressed bucket(s) [${compressed.map((t) => `${t.format} ${t.width}²×${t.depthOrArrayLayers}`).join(", ") || "none"}]`,
        },
        {
            // ≥ 2 layers in the RGBA bucket = the rarest sizeS (plural) genuinely overflowed the cap, not a
            // single edge image — the 6-distinct-size union spills its 3 rarest here.
            name: "rarest sizes spilled to an RGBA bucket",
            pass: spill.length >= 1 && spill[0].depthOrArrayLayers >= 2,
            detail: spill.length
                ? `spill bucket ${spill[0].format} ${spill[0].width}×${spill[0].height}×${spill[0].depthOrArrayLayers}`
                : "no RGBA spill bucket — the over-cap union didn't spill",
        },
        {
            name: "textured union renders",
            pass: textured > 0 && color,
            detail: `${textured} Material instances, ${layers} albedo layers, sear:color ${color}`,
        },
    ];
}

// ============================================================================
// gltf-multi — multi-asset coexistence: shared albedo arrays + palette bases accumulate, distinct per-mesh VATs
// ============================================================================
//
// Two textured sources (DamagedHelmet, WaterBottle) + two skinned (Fox, CesiumMan) in one scene. Before the
// union path a second source clobbered the first (fixed global binding names, one global VAT); now the shared
// albedo arrays + material palette accumulate every active asset (per-instance index = the asset's palette
// base + local id) and each skinned mesh binds its own VAT. The structural gate reads what the live look shows:
// the albedo arrays accumulate both textured sources' layers, the palette indices span both bases (not all 0),
// the two skinned meshes carry distinct real VATs. The pure bucket-planning logic is `union.test.ts`.

// load a source, normalize to ~2 units and offset to its slot along X
async function placeCluster(state: State, src: string, x: number, scale: number): Promise<void> {
    const eids = placeScene(state, await loadGltf(state, src));
    for (const eid of eids) {
        const sx = Transform.scale.x.get(eid);
        const sy = Transform.scale.y.get(eid);
        const sz = Transform.scale.z.get(eid);
        Transform.scale.set(eid, sx * scale, sy * scale, sz * scale, 0);
        Transform.pos.set(
            eid,
            Transform.pos.x.get(eid) * scale + x,
            Transform.pos.y.get(eid) * scale,
            Transform.pos.z.get(eid) * scale,
            0,
        );
    }
}

async function placeMulti(state: State): Promise<void> {
    await placeCluster(state, MULTI.helmet, -4.2, 1);
    await placeCluster(state, MULTI.bottle, -1.4, 8);
    await placeCluster(state, MULTI.fox, 1.4, 0.016);
    await placeCluster(state, MULTI.cesium, 4.2, 1.3);
}

// every per-instance palette index — `Textured.id` for a static textured instance, the skin slab's y lane for
// a skinned one. Both index the one shared union palette; before the union path each asset based at 0.
function paletteIndices(state: State): number[] {
    const ids: number[] = [];
    for (const eid of state.query([Textured])) ids.push(Textured.id.get(eid));
    for (const eid of state.query([Skin])) ids.push(Skin.anim.y.get(eid));
    return ids;
}

// the per-mesh VAT position texture for each skinned instance (Mesh.bindings) — distinct objects prove each
// skinned mesh binds its own VAT, not a single global one.
function skinnedVats(state: State): GPUTexture[] {
    const vats: GPUTexture[] = [];
    for (const eid of state.query([Skin])) {
        const pos = Meshes.get(Meshes.name(Part.mesh.get(eid)) ?? "")?.bindings?.vatPos;
        if (pos) vats.push(pos as GPUTexture);
    }
    return vats;
}

async function assertGltfMulti(state: State): Promise<Check[]> {
    while (unionPending()) await frames(1); // the union uploads across frames now — wait for the published set
    const parts = [...state.query([Part])].length;
    const textured = [...state.query([Textured])].length;
    const skinned = [...state.query([Skin])].length;
    const ids = paletteIndices(state);
    const maxMid = ids.length ? Math.max(...ids) : -1;
    const vats = skinnedVats(state);
    const distinctVats = new Set(vats).size;
    const albedoLayers = ALBEDO_NAMES.map((n) => Compute.textures.get(n))
        .filter((t): t is GPUTexture => !!t && (t.width > 1 || t.depthOrArrayLayers > 1))
        .reduce((s, t) => s + t.depthOrArrayLayers, 0);
    await frames(4);
    const color = Profile.gpu.has("sear:color");
    return [
        {
            name: "four sources imported",
            pass: parts >= 4 && textured >= 2 && skinned === 2,
            detail: `${parts} parts, ${textured} textured, ${skinned} skinned (expected ≥4 / ≥2 / 2)`,
        },
        {
            // the union accumulates every active asset's albedo (≥1 baseColor each), not just the last — a
            // clobbering single-active publish would leave 1 layer
            name: "shared albedo arrays accumulate",
            pass: albedoLayers >= 4,
            detail: `${albedoLayers} albedo layers across buckets (expected ≥4, one+ per source)`,
        },
        {
            // a clobbering publish bases every asset at 0 (max near 0); accumulation reaches the last offset (≥3)
            name: "palette bases accumulate (no clobber)",
            pass: new Set(ids).size >= 4 && maxMid >= 3,
            detail: `indices [${ids.join(", ")}], ${new Set(ids).size} distinct, max ${maxMid} (expected ≥4 distinct, max ≥3)`,
        },
        {
            name: "two skinned meshes carry distinct VATs",
            pass: distinctVats === 2 && vats.every((t) => t.width > 1 && t.height > 1),
            detail: `${distinctVats} distinct VAT textures (expected 2), sizes ${vats.map((t) => `${t.width}×${t.height}`).join(", ")}`,
        },
        { name: "multi-asset color pass draws", pass: color, detail: `sear:color ${color}` },
    ];
}

// ============================================================================
// gltf-worker — the off-thread decode pool + the declarative by-name gate
// ============================================================================
//
// The mode's scene authors two boxes purely by name (no import code), so it doubles as the declarative
// end-to-end gate: the preloader imports the sources before load, and the route sync decorates the
// textured one (gltf-albedo surface + Textured). On top of that it adds the proofs only a programmatic
// comparison can make: the pooled (worker) decode is byte-identical to an inline `decode`, the Draco /
// meshopt / KTX2 codec wasm each resolve + run off-thread, a concurrent batch decodes with no clobber, a
// cached scene load registers one asset per source (url normalized to src), and a corrupt decode rejects
// across the postMessage boundary. bun-webgpu has no Worker, so this is real-Chrome-only (the unit twins
// are `worker.test.ts` / `meshopt.test.ts` / `basis.test.ts` / `routes.test.ts`).

function sameBytes(a: ArrayBufferView, b: ArrayBufferView): boolean {
    if (a.byteLength !== b.byteLength) return false;
    const ua = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const ub = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
    return true;
}

// every geometry stream the worker transfers, paired inline-vs-worker — the byte-identical surface
function geometryPairs(
    inline: DecodedGltf,
    worker: DecodedGltf,
): [string, ArrayBufferView, ArrayBufferView][] {
    const si = inline.geometry.static!;
    const sw = worker.geometry.static!;
    const pairs: [string, ArrayBufferView, ArrayBufferView][] = [
        ["quant.main", si.quant.main, sw.quant.main],
        ["quant.position", si.quant.position, sw.quant.position],
        ["quant.quant", si.quant.quant, sw.quant.quant],
        ["indices", si.indices, sw.indices],
    ];
    for (let i = 0; i < inline.scene.meshes.length; i++) {
        pairs.push([
            `mesh${i}.vertices`,
            inline.scene.meshes[i].vertices,
            worker.scene.meshes[i].vertices,
        ]);
        pairs.push([
            `mesh${i}.indices`,
            inline.scene.meshes[i].indices,
            worker.scene.meshes[i].indices,
        ]);
    }
    return pairs;
}

async function assertGltfWorker(state: State): Promise<Check[]> {
    // the declarative gate: no import code ran — the preloader scanned the scene and imported both boxes
    // before load, so the scene-authored Parts resolved their `…#0` names, not the cube default
    const boxMesh = Meshes.id(`${WORKER_DRACO}#0`);
    const cube = Meshes.id("cube");
    const byName =
        boxMesh !== undefined &&
        boxMesh !== cube &&
        [...state.query([Part])].some((e) => Part.mesh.get(e) === boxMesh);

    // the route sync decorated the textured by-name Part: gltf-albedo surface + the Textured material id,
    // with the real (non-fallback) albedo bucket published once the union settles
    while (unionPending()) await frames(1);
    const texMesh = Meshes.id(`${WORKER_TEXTURED}#0`);
    const texEid = [...state.query([Part])].find((e) => Part.mesh.get(e) === texMesh);
    const routed =
        texEid !== undefined &&
        Part.surface.get(texEid) === Surfaces.id("gltf-albedo") &&
        state.has(texEid, Textured);
    const texBucket = ALBEDO_NAMES.map((n) => Compute.textures.get(n)).some(
        (t) => !!t && (t.width > 1 || t.depthOrArrayLayers > 1),
    );

    // a concurrent multi-asset load through the pool — two Draco + two KTX2 + meshopt decoded in parallel across
    // the pool's worker slots. decodeInWorker bypasses the cache, so each is a fresh decode.
    const [worker, lamp, mesho, worker2, lamp2] = await Promise.all([
        decodeInWorker(WORKER_DRACO),
        decodeInWorker(WORKER_KTX),
        decodeInWorker(WORKER_MESHOPT),
        decodeInWorker(WORKER_DRACO),
        decodeInWorker(WORKER_KTX),
    ]);

    // byte-identical reference — inline, after the batch; Draco needs no transcode target, a clean reference
    const inline = await decode(WORKER_DRACO);
    const decodedOffThread = !!worker.geometry.static && !!inline.geometry.static;
    const mismatch = decodedOffThread
        ? geometryPairs(inline, worker).find(([, a, b]) => !sameBytes(a, b))
        : undefined;

    // meshopt: the same byte-identical gate proves the meshopt decoder wasm bundles into the worker chunk and
    // decompresses + dequantizes off-thread identically to inline (deviceless, geometry needs no target)
    const inlineMesho = await decode(WORKER_MESHOPT);
    const meshoOffThread = !!mesho.geometry.static && !!inlineMesho.geometry.static;
    const meshoMismatch = meshoOffThread
        ? geometryPairs(inlineMesho, mesho).find(([, a, b]) => !sameBytes(a, b))
        : undefined;

    // KTX2/Basis: the worker resolved the device target + ran the transcoder off-thread, both lamps
    const compressedAlbedo = (d: DecodedGltf) => {
        const a = d.textures.albedo[0];
        return !!a && a.kind === "compressed" && COMPRESSED.test(a.image.format);
    };
    const transcoded = compressedAlbedo(lamp) && compressedAlbedo(lamp2);
    const albedo = lamp.textures.albedo[0];

    // every asset in the concurrent batch produced usable geometry — the pool didn't drop or clobber one
    const batch = [worker, lamp, mesho, worker2, lamp2];
    const allDecoded = batch.every((d) => !!d.geometry.static || d.geometry.skinned.length > 0);

    // the preloader loaded the two scene-named sources through the cache (ensureDecoded → pool → register),
    // normalizing the worker's absolute url back to src — so register found each entry once, no duplicates.
    // The decode / decodeInWorker calls above bypass the cache, so the count stays at the two scene assets.
    const sceneEntries = gltfCacheStats().assets === 2;

    // a worker decode error crosses the postMessage boundary as a rejection carrying decode's message (the
    // worker stays alive: it catches, posts {ok:false}, the pool rewraps in an Error and frees the slot).
    let workerError: string | null = null;
    try {
        await decodeInWorker(WORKER_CORRUPT);
    } catch (e) {
        workerError = e instanceof Error ? e.message : String(e);
    }

    return [
        {
            name: "Draco geometry decoded off the main thread (pool worker)",
            pass: decodedOffThread,
            detail: decodedOffThread
                ? "worker + inline both decoded"
                : "worker decode missing geometry",
        },
        {
            name: "worker geometry is byte-identical to inline",
            pass: decodedOffThread && !mismatch,
            detail: mismatch ? `mismatch in ${mismatch[0]}` : "all geometry streams identical",
        },
        {
            name: "meshopt geometry decoded + byte-identical to inline (worker codec wasm)",
            pass: meshoOffThread && !meshoMismatch,
            detail: !meshoOffThread
                ? "meshopt worker decode missing geometry"
                : meshoMismatch
                  ? `mismatch in ${meshoMismatch[0]}`
                  : "all meshopt geometry streams identical",
        },
        {
            name: "KTX2 baseColor transcoded off-thread to a compressed albedo",
            pass: transcoded,
            detail:
                albedo && albedo.kind === "compressed"
                    ? `compressed ${albedo.image.format}`
                    : "no compressed albedo decoded",
        },
        {
            name: "concurrent multi-asset load decoded every asset (no clobber)",
            pass: allDecoded,
            detail: `${batch.length} assets decoded in parallel through the pool`,
        },
        {
            name: "cached scene loads registered one asset each (url normalized to src)",
            pass: sceneEntries,
            detail: `${gltfCacheStats().assets} cached assets (expected 2)`,
        },
        {
            name: "worker decode error rejects across the boundary with a message",
            pass: !!workerError,
            detail: workerError ? `rejected: ${workerError}` : "corrupt decode did not reject",
        },
        {
            name: "scene resolved the glTF mesh by name (declarative preloader, no import code)",
            pass: byName,
            detail: byName
                ? `Part → "${WORKER_DRACO}#0" (mesh ${boxMesh}, not cube ${cube})`
                : `no scene Part resolved "${WORKER_DRACO}#0" (mesh ${boxMesh}, cube ${cube})`,
        },
        {
            name: "route sync decorated the textured by-name Part (surface + Textured)",
            pass: routed && texBucket,
            detail: routed
                ? `surface gltf-albedo + Textured, albedo bucket ${texBucket ? "published" : "missing"}`
                : `Part for "${WORKER_TEXTURED}#0" not routed (eid ${texEid ?? "—"})`,
        },
    ];
}

// ============================================================================
// transparency mode — the alpha-blend `blend` surface mode, framebuffer-probed
// ============================================================================
//
// A bright opaque white wall fills the frame; a translucent black panel (the `alpha` blend surface, α = 0.5)
// covers the screen centre, leaving the corners on bare wall. A correct straight-alpha "over" composite
// transmits exactly (1 − α) of the wall through the panel — so the centre reads half the corner. The probe
// pins centre/corner ≈ 0.5: an opaque panel drives it to 0 (wall hidden), a missing panel to 1 (wall bare),
// only the blend lands at the alpha factor. The `clip` cutout mode is gated by gltf-model's MASK foliage, so
// this carries the `blend` mode the other modes don't reach.

const PANEL_ALPHA = 0.5;
const PANEL_BINDINGS: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    color: { type: "storage", element: "u32" },
};

// an unlit straight-alpha surface: writes the entity color rgb + its (linear) alpha lane; sear's blend unit
// does the "over" composite. `unpackLdrColor` is sear-spliced for every surface.
const TransparencyPlugin: Plugin = {
    name: "GymRenderTransparency",
    dependencies: [RenderPlugin],
    initialize() {
        Surfaces.register({
            name: "alphaPanel",
            blend: "alpha",
            bindings: PANEL_BINDINGS,
            fs: /* wgsl */ `
                let c = unpackLdrColor(color[eid]);
                col = vec4<f32>(c.rgb, c.a);`,
        });
    },
};

function buildTransparency(state: State): void {
    // a big bright opaque white wall behind everything (unlit → luma 1.0 in the HDR framebuffer)
    const wall = state.create();
    state.add(wall, Transform);
    Transform.pos.set(wall, 0, 0, -30, 0);
    Transform.scale.set(wall, 80, 50, 1, 0);
    state.add(wall, Part);
    Part.surface.set(wall, Surfaces.id("unlit") ?? 0);
    state.add(wall, Color);
    Color.rgba.set(wall, 1, 1, 1, 1);

    // the translucent panel over the screen centre — black so its only contribution is dimming the wall by α
    const panel = state.create();
    state.add(panel, Transform);
    Transform.pos.set(panel, 0, 0, -12, 0);
    Transform.scale.set(panel, 9, 9, 1, 0);
    state.add(panel, Part);
    Part.surface.set(panel, Surfaces.id("alphaPanel") ?? 0);
    state.add(panel, Color);
    Color.rgba.set(panel, 0, 0, 0, PANEL_ALPHA);

    cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Orbit);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 60);
    Camera.clearColor.set(cam, 0x000000);
    Orbit.distance.set(cam, 12);
    Orbit.yaw.set(cam, 0);
    Orbit.pitch.set(cam, 0);
    Orbit.pan.set(cam, 0, 0, -12, 0); // look straight at the panel; the wall fills the frame behind it
}

async function assertTransparency(): Promise<Check[]> {
    if (!probeMirror) return [{ name: "transparency", pass: false, detail: "no probe mirror" }];
    await settle(probeMirror);
    if (!probeMirror.snapshot)
        return [{ name: "transparency", pass: false, detail: "no snapshot" }];
    const out = new Float32Array(probeMirror.snapshot.bytes);
    const center = out[5]; // panel over wall = (1 − α) · wall
    const corner = out[6]; // bare wall ≈ 1.0
    const ratio = center / Math.max(corner, 1e-4);
    // the corner reads the bright bare wall; the centre/corner ratio reads the transmitted fraction (1 − α).
    // A 0.15 band around 0.5 separates the blend cleanly from opaque (0) and absent (1).
    return [
        {
            name: "blend transmits the background",
            pass: corner > 0.8 && Math.abs(ratio - (1 - PANEL_ALPHA)) < 0.15,
            detail: `corner ${corner.toFixed(3)} (bare wall), centre ${center.toFixed(3)}, centre/corner ${ratio.toFixed(3)} (want ≈ ${(1 - PANEL_ALPHA).toFixed(2)})`,
        },
    ];
}

// ============================================================================
// background mode — the injectable backdrop seam, gated by the framebuffer probe
// ============================================================================

// a vertical sky gradient driven by the world-space view ray's elevation — the minimal backdrop recipe (no
// bindings, reads only the sear-reconstructed `dir`). Bright across the frame so a painted background pixel
// reads well above the black clear, and an opaque dark box overdrawing it reads well below.
const GRADIENT_BG_FS = /* wgsl */ `
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    col = mix(vec3<f32>(0.7, 0.8, 1.0), vec3<f32>(0.2, 0.4, 0.9), t);`;

// depends on SearPlugin so its initialize runs *after* SearPlugin clears the Backgrounds registry (the
// reload-safety wipe) — the same ordering TransparencyPlugin uses against RenderPlugin's surface clear
const BackgroundPlugin: Plugin = {
    name: "GymRenderBackground",
    dependencies: [SearPlugin],
    initialize() {
        Backgrounds.register({ name: "gradient", fs: GRADIENT_BG_FS });
    },
};

function buildBackground(state: State): void {
    // a dark opaque box dead-centre. unlit → its luminance is deterministic (no lights). It overdraws the
    // backdrop where geometry is — the far-plane fill is depth-masked to un-rendered pixels, so the centre
    // reads the dark box while the corners read the bright sky
    const box = state.create();
    state.add(box, Transform);
    Transform.scale.set(box, 4, 4, 4, 0);
    state.add(box, Part);
    Part.surface.set(box, Surfaces.id("unlit") ?? 0);
    state.add(box, Color);
    Color.rgba.set(box, 0.05, 0.05, 0.05, 1); // dark grey — far below any sky-gradient luminance

    cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Orbit);
    state.add(cam, Backdrop);
    Backdrop.name.set(cam, Backgrounds.id("gradient") ?? 0);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 60);
    // black clear: a camera with no backdrop reads ~0 on background pixels, so a painted corner is the signal
    Camera.clearColor.set(cam, 0x000000);
    Orbit.distance.set(cam, 12);
    Orbit.yaw.set(cam, 0);
    Orbit.pitch.set(cam, 0); // horizontal look → the top corners see the up-tilted sky (dir.y > 0)
}

async function assertBackground(): Promise<Check[]> {
    if (!probeMirror) return [{ name: "background", pass: false, detail: "no probe mirror" }];
    await settle(probeMirror);
    if (!probeMirror.snapshot) return [{ name: "background", pass: false, detail: "no snapshot" }];
    const out = new Float32Array(probeMirror.snapshot.bytes);
    const sky = out[6]; // top-left corner — pure backdrop, no geometry
    const box = out[5]; // screen centre — the dark opaque box
    // two facts in one frame: the backdrop painted the un-rendered pixels (a black clear reads ~0 at the
    // corner), and opaque geometry overdrew it at the centre (if the far-plane fill ignored depth, the centre
    // would read the bright sky too). So the corner is bright sky, the centre the dark box well below it.
    return [
        {
            name: "backdrop fills background, geometry overdraws it",
            pass: sky > 0.3 && box < sky * 0.4,
            detail: `corner ${sky.toFixed(3)} (sky, want > 0.3), centre ${box.toFixed(3)} (box, want < ${(sky * 0.4).toFixed(3)})`,
        },
    ];
}

// ============================================================================
// sky mode — extras/sky on the backdrop seam, the uniform-binding path the bindings-free `background` mode
// doesn't exercise (the recipe reads its own `Sky` uniform, resolved by name from Compute.buffers)
// ============================================================================

// the same overdraw geometry as `background`, but the backdrop is extras/sky's procedural recipe — so a
// painted corner proves the `Sky` uniform binding resolved and the sky shaded a real gradient, not black.
function buildSky(state: State): void {
    state.add(state.create(), Sky); // the singleton config (daytime defaults)

    // a sun so the recipe's lighting.sunDirection read is non-degenerate (the disk sits opposite it)
    const sun = state.create();
    state.add(sun, DirectionalLight);
    DirectionalLight.direction.set(sun, -0.4, -0.8, -0.45, 0);

    const box = state.create();
    state.add(box, Transform);
    Transform.scale.set(box, 4, 4, 4, 0);
    state.add(box, Part);
    Part.surface.set(box, Surfaces.id("unlit") ?? 0);
    state.add(box, Color);
    Color.rgba.set(box, 0.05, 0.05, 0.05, 1); // dark — far below the daytime sky gradient

    cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Orbit);
    state.add(cam, Backdrop);
    Backdrop.name.set(cam, Backgrounds.id("sky") ?? 0);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 60);
    Camera.clearColor.set(cam, 0x000000); // black clear → an unpainted background pixel reads ~0 and fails
    Orbit.distance.set(cam, 12);
    Orbit.yaw.set(cam, 0);
    Orbit.pitch.set(cam, 0); // horizontal → the top corners see up-sky (dir.y > 0)
}

async function assertSky(): Promise<Check[]> {
    if (!probeMirror) return [{ name: "sky", pass: false, detail: "no probe mirror" }];
    await settle(probeMirror);
    if (!probeMirror.snapshot) return [{ name: "sky", pass: false, detail: "no snapshot" }];
    const out = new Float32Array(probeMirror.snapshot.bytes);
    const sky = out[6]; // top-left corner — pure sky
    const box = out[5]; // centre — the dark overdrawing box
    // the sky uniform resolved and painted a bright gradient on the un-rendered pixels (a black clear or an
    // unbound uniform reads ~0 at the corner), and opaque geometry overdrew it at the centre
    return [
        {
            name: "sky backdrop fills background, geometry overdraws it",
            pass: sky > 0.3 && box < sky * 0.4,
            detail: `corner ${sky.toFixed(3)} (sky, want > 0.3), centre ${box.toFixed(3)} (box, want < ${(sky * 0.4).toFixed(3)})`,
        },
    ];
}

// ============================================================================
// skin-live — the live joint-palette substrate on the real GPU (specs/tumble-shallot.md stage 6d)
// ============================================================================
//
// A by-construction 2-bone rig posed entirely through the public `LiveSkin` seam — no glTF asset, no
// physics, the non-physics producer that proves the substrate independently. A skin-live Part carries its
// own palette block; a scripted driver writes per-joint object-space skin matrices each frame (bend the
// upper bone about an elbow), and the surface's `vs` blends the palette per vertex. Because skinning is a
// VS warp, sear's shadow + prepass passes deform for free — the whole point the mode pins on hardware:
//
//   deform: bending swings the upper arm into a screen region that was background (fills) while the bind-
//   pose tip region empties — the palette blend rendering the live pose. shadow: the limb's shadow on the
//   floor moves between poses, pinning the free prepass/shadow deformation (a static-geometry shadow pass
//   couldn't). reach: at the bent (max-extension) pose the instance still packs a survivor and renders its
//   extended arm — the reach-sphere cull bound covers the pose envelope, not the tight rest AABB.
//
// GltfPlugin supplies the substrate (registers the skin-live surfaces + LiveSkinSystem + the material/albedo
// fallbacks + resets LiveSkin), so the mode rides the gltf plugin set but authors its rig by hand.

const RIG_MESH = "skin-live-rig";
const RIG_RINGS = [0, 0.5, 1, 1.5, 2, 2.5, 3]; // beam heights (y); the elbow blend sits at y≈1
const RIG_HALF = 0.2; // cross-section half-extent (a thin square beam along +Y)
// the 4 cross-section corners CCW viewed from +Y — the order the side-face winding below reads as outward
const RIG_CORNERS: [number, number][] = [
    [RIG_HALF, -RIG_HALF],
    [RIG_HALF, RIG_HALF],
    [-RIG_HALF, RIG_HALF],
    [-RIG_HALF, -RIG_HALF],
];

// the upper bone's influence at height `y`: 0 below the elbow blend band, ramping to 1 above it, so the
// lower half stays with the fixed root (a stable shadow anchor) and the upper half swings with the elbow
function rigWeight(y: number): number {
    return Math.min(1, Math.max(0, (y - 0.7) / 0.6));
}

// build the beam: 4 side faces, each a vertical quad strip over the rings, with the per-vertex joints
// (slot 0 = root joint 0, slot 1 = elbow joint 1) + weights (unorm8 pair summing to 255, so the surface
// skips a runtime renorm, matching the importer). `reach` is the object-space reach radius the importer's
// bound derives — max over joints (w>0) of |bⱼ| + |restPos − bⱼ|, bⱼ the bone bind origin (0 / (0,1,0)).
function buildRigGeometry(): {
    vertices: Float32Array;
    indices: Uint32Array;
    joints: Uint32Array;
    weights: Uint32Array;
    reach: number;
} {
    const verts: number[] = [];
    const idx: number[] = [];
    const joints: number[] = [];
    const weights: number[] = [];
    let reach = 0;
    const R = RIG_RINGS.length;
    for (let f = 0; f < 4; f++) {
        const c0 = RIG_CORNERS[f];
        const c1 = RIG_CORNERS[(f + 1) % 4];
        const nx = (c0[0] + c1[0]) * 0.5;
        const nz = (c0[1] + c1[1]) * 0.5;
        const nl = Math.hypot(nx, nz) || 1; // face midpoint direction = its outward normal (axis-aligned)
        const base = verts.length / VERTEX_FLOATS;
        for (let r = 0; r < R; r++) {
            const y = RIG_RINGS[r];
            const w1 = rigWeight(y);
            const u1 = Math.round(w1 * 255);
            for (const c of [c0, c1]) {
                verts.push(c[0], y, c[1], 0, nx / nl, 0, nz / nl, 0); // posU + normalV (uv unused)
                joints.push(0 | (1 << 8)); // slots [0, 1, 0, 0]
                weights.push((255 - u1) | (u1 << 8)); // unorm8 [w0, w1, 0, 0], sum 255
                const d0 = Math.hypot(c[0], y, c[1]); // |restPos − b0|, b0 = origin
                const d1 = 1 + Math.hypot(c[0], y - 1, c[1]); // |b1| + |restPos − b1|, b1 = (0,1,0)
                reach = Math.max(reach, w1 < 1 ? d0 : 0, w1 > 0 ? d1 : 0);
            }
        }
        // outward-wound quads (derived: cross(edge) points along the face's outward normal for this order)
        for (let r = 0; r < R - 1; r++) {
            const v0 = base + r * 2;
            idx.push(v0, v0 + 2, v0 + 1, v0 + 1, v0 + 2, v0 + 3);
        }
    }
    return {
        vertices: new Float32Array(verts),
        indices: new Uint32Array(idx),
        joints: new Uint32Array(joints),
        weights: new Uint32Array(weights),
        reach,
    };
}

let rigEid = -1;
let rigMeshId = -1;
let rigReach = 0;
let rigMaxExcursion = 0; // the actual max distance any posed vertex reaches from the origin (brute force)
let heldAngle: number | null = null; // the assert pins a pose here; null → the live tab oscillates
let rigDrawArgs: Mirror | null = null;

const RIG_IDENT = compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
const RIG_T1 = compose(0, 1, 0, 0, 0, 0, 1, 1, 1, 1); // elbow at y=1
const RIG_INV1 = compose(0, -1, 0, 0, 0, 0, 1, 1, 1, 1); // joint-1 inverse bind
const _rigSkin = new Float32Array(32);
const _rigRz = new Float32Array(16);
const _rigTmp = new Float32Array(16);

// write the 2-bone pose into the rig's palette: joint 0 stays identity (the root is fixed), joint 1 rotates
// `angle` about Z at the elbow (y=1). skin = jointWorld · inverseBind — the object-space matrices a producer
// (a physics ragdoll, this scripted driver) hands `writePalette`; LiveSkin decomposes each to its Xform entry.
function writeRigPose(angle: number): void {
    _rigSkin.set(RIG_IDENT, 0);
    const h = angle / 2;
    compose(0, 0, 0, 0, 0, Math.sin(h), Math.cos(h), 1, 1, 1, _rigRz);
    multiply(RIG_T1, _rigRz, _rigTmp); // T1·Rz
    multiply(_rigTmp, RIG_INV1, _rigRz); // ·inv1 → distinct out (multiply isn't alias-safe)
    _rigSkin.set(_rigRz, 16);
    LiveSkin.writePalette(rigEid, _rigSkin);
}

// the actual max distance any vertex reaches from the origin across the pose range — a geometry-side
// cross-check (independent of the analytic reach formula) that the reach bound covers the envelope. LBS each
// rest vertex at the bind (rest) and bent (90°) extremes: joint 0 is identity, joint 1 is `skin1`, so the
// posed point is `w0·p + w1·(skin1·p)`. The bent tip lands *closer* to the origin than the rest tip, so for
// this interior-elbow rig the rest pose is the max extent (the note in the reach check spells this out).
function maxPosedExcursion(g: ReturnType<typeof buildRigGeometry>): number {
    const skin1 = new Float32Array(16);
    compose(0, 0, 0, 0, 0, Math.SQRT1_2, Math.SQRT1_2, 1, 1, 1, _rigRz); // Rz(90°)
    multiply(RIG_T1, _rigRz, _rigTmp);
    multiply(_rigTmp, RIG_INV1, skin1);
    let max = 0;
    for (let v = 0; v < g.vertices.length / VERTEX_FLOATS; v++) {
        const px = g.vertices[v * 8];
        const py = g.vertices[v * 8 + 1];
        const pz = g.vertices[v * 8 + 2];
        const w1 = ((g.weights[v] >> 8) & 0xff) / 255;
        const jx = skin1[0] * px + skin1[4] * py + skin1[8] * pz + skin1[12];
        const jy = skin1[1] * px + skin1[5] * py + skin1[9] * pz + skin1[13];
        const jz = skin1[2] * px + skin1[6] * py + skin1[10] * pz + skin1[14];
        const bx = (1 - w1) * px + w1 * jx;
        const by = (1 - w1) * py + w1 * jy;
        const bz = (1 - w1) * pz + w1 * jz;
        max = Math.max(max, Math.hypot(px, py, pz), Math.hypot(bx, by, bz));
    }
    return max;
}

// pose the rig each frame — a held angle (the assert's pinned pose) or a 0→90° oscillation for the live tab.
// `simulation` group so the write lands before SlabSystem + LiveSkinSystem flush it (draw group).
const RigDriverSystem: System = {
    name: "skin-live-driver",
    group: "simulation",
    annotations: { mode: "always" },
    update(state: State) {
        if (rigEid < 0) return;
        const t = state.time.elapsed;
        writeRigPose(heldAngle ?? (Math.PI / 2) * (0.5 - 0.5 * Math.cos(t * 1.2)));
    },
};

// the probe classifies each framebuffer pixel by colour, not luminance: the warm-tinted limb (r ≫ b) and
// the neutral-grey floor read nearly the same luminance, so a luminance region can't tell them apart. It
// scans a coarse grid and reports the limb's coloured-pixel centroid + count (out[0..2]) and the shadowed-
// floor centroid + count (out[3..4]). The neutral floor makes the shadow a genuine LUMINANCE separation
// (the `chroma < 0.04` clause is a tiebreaker, not the discriminator): the pre-tonemap HDR framebuffer reads
// the lit floor at ~0.4 (albedo 0.214 × [ambient 0.35 + sun 1.6·halfLambert]), the sun-shadowed floor at
// ~0.075 (ambient only), the clear-colour background at ~0.004. The `0.02 < lum < 0.2` band captures the
// shadowed floor with ≥2.5× margin either side, excluding both the lit floor and the background — so no
// coloured floor is needed to keep the lit floor out. The assert reads these across a bind vs bent pose: the
// limb centroid swings left when the arm bends, and the shadow footprint shifts/grows.
const SKIN_PROBE_WGSL = /* wgsl */ `
@group(0) @binding(0) var fb: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
const luma = vec3<f32>(0.2126, 0.7152, 0.0722);
@compute @workgroup_size(1)
fn main() {
    let dim = vec2<f32>(textureDimensions(fb));
    let W = 160; let H = 100;
    var limbN = 0.0; var limbX = 0.0; var limbY = 0.0;
    var shadN = 0.0; var shadX = 0.0;
    for (var j = 0; j < H; j = j + 1) {
        for (var i = 0; i < W; i = i + 1) {
            let u = (f32(i) + 0.5) / f32(W);
            let v = (f32(j) + 0.5) / f32(H);
            let c = textureLoad(fb, vec2<i32>(vec2<f32>(u, v) * dim), 0).rgb;
            let lum = dot(c, luma);
            let chroma = c.r - c.b; // warm limb: r ≫ b; neutral floor / background: ≈ 0
            if (chroma > 0.06 && lum > 0.03) {
                limbN += 1.0; limbX += u; limbY += v;
            } else if (abs(chroma) < 0.04 && lum > 0.02 && lum < 0.2) {
                shadN += 1.0; shadX += u; // grey + mid-dark (below lit floor, above background) = shadow
            }
        }
    }
    out[0] = limbN;
    out[1] = select(0.5, limbX / limbN, limbN > 0.0);
    out[2] = select(0.5, limbY / limbN, limbN > 0.0);
    out[3] = shadN;
    out[4] = select(0.5, shadX / shadN, shadN > 0.0);
}`;

// publish a white 1×1 albedo so the skin-live limb reads its Color tint: the limb samples material 0 (the
// gltf fallback palette entry — bucket 0, layer 0), whose albedo texture is otherwise the unwritten black
// fallback, so base = albedo × tint would be black. A real asset publishes its own union here; a hand-built
// rig has none, so the mode supplies a usable albedo.
function whiteAlbedo(device: GPUDevice): void {
    const tex = device.createTexture({
        label: "skin-live-albedo",
        size: { width: 1, height: 1 },
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
        { texture: tex },
        new Uint8Array([255, 255, 255, 255]),
        { bytesPerRow: 4 },
        { width: 1, height: 1 },
    );
    Compute.textures.set("albedo0", tex);
}

async function buildSkinLive(state: State): Promise<void> {
    const device = Compute.device;

    const ambient = state.create();
    state.add(ambient, AmbientLight);
    AmbientLight.color.set(ambient, 0xffffff);
    AmbientLight.intensity.set(ambient, 0.35);

    // a casting sun, mostly overhead (slightly -X) so the limb's shadow projects onto the floor and shifts
    // when the arm bends left — the moving-shadow probe reads that
    const sun = state.create();
    state.add(sun, DirectionalLight);
    DirectionalLight.color.set(sun, 0xffffff);
    DirectionalLight.intensity.set(sun, 1.6);
    DirectionalLight.direction.set(sun, -0.35, -1, -0.12, 0);
    state.add(sun, Shadow);
    Shadow.distance.set(sun, 20);

    // the receiver floor (default surface — independent of the gltf material path), top face at y=0
    const floor = state.create();
    state.add(floor, Transform);
    Transform.pos.set(floor, 0, -0.1, 0, 0);
    Transform.scale.set(floor, 14, 0.2, 14, 0);
    state.add(floor, Part);
    state.add(floor, Color);
    Color.rgba.set(floor, 0.5, 0.5, 0.5, 1); // neutral grey — the shadow probe separates by luminance, not tint

    // above + in front (+Z) looking down at the rig; yaw 0 keeps world X = screen X, world Y = screen Y, so
    // the bind-pose beam runs up the centre and the bent arm swings to screen-left (the probe regions)
    cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Orbit);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 50);
    Camera.near.set(cam, 0.1);
    Camera.far.set(cam, 100);
    Camera.clearColor.set(cam, 0x0a0c12);
    Orbit.distance.set(cam, 8);
    Orbit.pan.set(cam, 0, 1.2, 0, 0);
    Orbit.yaw.set(cam, 0);
    Orbit.pitch.set(cam, 0.32);

    // the rig geometry as its OWN buffers (not the shared mesh() family) so `vidx` is local [0, vertCount) —
    // the axis the live JW block is keyed by, the same own-stream a skinned import uses
    const g = buildRigGeometry();
    rigReach = g.reach;
    rigMaxExcursion = maxPosedExcursion(g);
    const q = quantizeMeshes(g.vertices, [
        { vertexBase: 0, vertexCount: g.vertices.length / VERTEX_FLOATS },
    ]);
    const bufOf = (label: string, data: Uint32Array): GPUBuffer => {
        const b = device.createBuffer({
            label,
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(b, 0, data);
        return b;
    };
    const spec: Mesh = {
        name: RIG_MESH,
        vertices: bufOf("skin-live-rig-main", q.main),
        position: bufOf("skin-live-rig-pos", q.position),
        quant: device.createBuffer({
            label: "skin-live-rig-quant",
            size: q.quant.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        indices: bufOf("skin-live-rig-idx", g.indices),
        indexBase: 0,
        indexCount: g.indices.length,
        bounds: [0, 0, 0, g.reach], // the reach sphere (origin-centred), not the tight rest AABB
        variant: 0,
    };
    device.queue.writeBuffer(spec.quant!, 0, q.quant);
    rigMeshId = Meshes.register(spec);
    LiveSkin.registerMesh(rigMeshId, g.joints, g.weights);
    const registered = Meshes.get(RIG_MESH);
    if (registered) registered.bindings = { skinParams: LiveSkin.paramsBuffer(device, rigMeshId) };
    whiteAlbedo(device);

    // the skin-live Part carrying its own palette block (seeded to the bind pose until the driver poses it)
    rigEid = state.create();
    state.add(rigEid, Transform);
    state.add(rigEid, Part);
    state.add(rigEid, Color);
    state.add(rigEid, Skin);
    Part.mesh.set(rigEid, rigMeshId);
    Part.surface.set(rigEid, Surfaces.id("skin-live")!);
    Color.rgba.set(rigEid, 0.95, 0.7, 0.35, 1);
    const paletteBase = LiveSkin.alloc(rigEid, 2, state.stamp(rigEid));
    Skin.anim.set(rigEid, paletteBase, 0, 0, 0); // palette base, material 0, unused, w=0 (SkinSystem skips)

    state.addSystem(RigDriverSystem);

    probePipeline = await device.createComputePipelineAsync({
        label: "skin-live-probe",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "skin-live-probe", code: SKIN_PROBE_WGSL }),
            entryPoint: "main",
        },
    });
    probeBuf = device.createBuffer({
        label: "skin-live-probe",
        size: 20, // limb count/x/y + shadow count/x
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    probeBg = null;
    state.addSystem(ProbeSystem);
    await frames(4);
    probeMirror = mirror(probeBuf);
    rigDrawArgs = mirror(Parts.drawArgs!);
    await frames(3);
}

// read the probe stats once the currently-held pose has propagated through the pipeline (the driver ticks
// on a sim step ~0.5/frame → flush → render → probe → the Mirror ring). `.slice()` copies out of the
// mirror's reused staging buffer — a bare `new Float32Array(snapshot.bytes)` is a *view*, so two stored
// reads would alias the same buffer and always compare equal.
async function skinProbe(): Promise<Float32Array | null> {
    if (!probeMirror) return null;
    await frames(12);
    await settle(probeMirror);
    return probeMirror.snapshot ? new Float32Array(probeMirror.snapshot.bytes).slice() : null;
}

// the (skin-live, live-rig) pair's survivor count at the camera slot — 1 when the instance passes its
// frustum cull, 0 if the reach bound wrongly dropped it
function rigSurvivors(): number | null {
    if (!rigDrawArgs?.snapshot) return null;
    const slot = Views.get(cam)?.slot ?? 0;
    const pairCount = Surfaces.size * Meshes.size;
    const pair = rigMeshId * Surfaces.size + (Surfaces.id("skin-live") ?? 0);
    const counts = packCounts(rigDrawArgs, slot, pairCount);
    return counts ? counts[pair] : null;
}

async function assertSkinLive(): Promise<Check[]> {
    const checks: Check[] = [];

    // structural: the substrate published its buffers and the rig's JW landed in region B
    const skinData = Compute.buffers.has("skinData");
    const jwRegistered = LiveSkin.meshes.has(rigMeshId) && LiveSkin.jwEnd > 0;
    checks.push({
        name: "skinData + JW published",
        pass: skinData && jwRegistered,
        detail: `skinData ${skinData}, mesh JW ${jwRegistered} (jwEnd ${LiveSkin.jwEnd})`,
    });

    // the mesh carries the origin-centred reach sphere (not a mid-centred rest AABB), and that reach covers
    // the actual posed geometry — a formula-vs-brute-force cross-check of the bound the cull tests against
    const meshBound = Meshes.get(RIG_MESH)?.bounds;
    checks.push({
        name: "reach bound carried by the mesh + covers the posed geometry",
        pass:
            meshBound?.[3] === rigReach && meshBound[0] === 0 && rigReach >= rigMaxExcursion - 1e-3,
        detail: `mesh bound r ${meshBound?.[3]?.toFixed(2)} = reach ${rigReach.toFixed(2)} ≥ max posed excursion ${rigMaxExcursion.toFixed(2)}`,
    });

    // bind pose (straight)
    heldAngle = 0;
    const bind = await skinProbe();
    // bent pose (~90° — the arm swings horizontally left; the max-extension reach)
    heldAngle = Math.PI / 2;
    const bent = await skinProbe();
    const survivors = rigSurvivors();

    if (!bind || !bent) {
        checks.push({ name: "probe", pass: false, detail: "no probe snapshot" });
        return checks;
    }

    // deform: the limb renders in both poses (coloured pixels present) and its centroid swings left when the
    // arm bends about the elbow — the palette blend rendering the live pose in the color pass
    const visible = bind[0] > 80 && bent[0] > 80;
    checks.push({
        name: "bending swings the limb left",
        pass: visible && bent[1] < bind[1] - 0.03,
        detail: `limb pixels bind ${bind[0]} / bent ${bent[0]}, centroid x ${bind[1].toFixed(3)} → ${bent[1].toFixed(3)}`,
    });

    // shadow: the limb's shadow on the floor shifts/grows with the pose (the free prepass/shadow deform — a
    // static-geometry shadow pass couldn't) — a differential on the shadowed-floor centroid + area
    const shadowMoved =
        bind[3] > 15 &&
        bent[3] > 15 &&
        (Math.abs(bind[4] - bent[4]) > 0.02 || Math.abs(bind[3] - bent[3]) > 0.2 * bind[3]);
    checks.push({
        name: "the limb's shadow moves with the pose",
        pass: shadowMoved,
        detail: `shadow pixels bind ${bind[3]} / bent ${bent[3]}, centroid x ${bind[4].toFixed(3)} → ${bent[4].toFixed(3)}`,
    });

    // reach: the max-extension pose packs a frustum-cull survivor and its extended arm renders — the reach-
    // bounded instance draws its live pose. Honest scope: the camera frames the whole rig, so survivors=1
    // doesn't adversarially force the reach bound over a tighter one (and this interior-elbow bend stays
    // within the origin-centred rest extent anyway); the reach formula is unit-tested in gltf.test.ts. The
    // pin here is that the substrate packs + draws the extended pose on the real GPU.
    checks.push({
        name: "max-extension pose packs a survivor and renders",
        pass: survivors === 1 && bent[0] > 80,
        detail: `survivors ${survivors}, limb pixels at extension ${bent[0]}`,
    });

    // the skin-live color pass fired
    await frames(4);
    const color = Profile.gpu.has("sear:color");
    checks.push({ name: "skin-live color pass draws", pass: color, detail: `sear:color ${color}` });

    heldAngle = null; // hand the pose back to the live oscillation
    return checks;
}

// ============================================================================
// ragdoll — the live palette's first physics producer (specs/tumble-shallot.md stage 7b)
// ============================================================================
//
// RiggedFigure imported `{live}` + an 11-capsule tumble ragdoll driving its 19-joint palette. Bones are
// substrate `Body` capsule entities (so writeback, pick, and the character sweep see them); joints ride
// the `Tumble.world` escape hatch via `Tumble.body(eid)` handles (spherical cone/twist, revolute,
// filter — deliberately richer than the substrate `Spring`/`Joint` mapping). The pose producer reads
// `readBody` per bone each fixed tick, nlerps prev→curr at `fixedAlpha`, and writes each glTF joint's
// palette entry as its bone's rigid delta: `palette_j = T_inst⁻¹ · boneNow · boneObjBind⁻¹` — the glTF
// inverse-bind cancels (a joint rigidly follows its bone), so the producer needs only each BONE's
// object-space bind (captured at build) + the bone→joints name map, never per-joint IBMs. The instance
// `Transform` tracks the pelvis (`T_inst = pelvisNow · pelvisObjBind⁻¹`) so palette entries stay bounded
// and the importer's reach-sphere cull holds as the ragdoll falls.

const RIG_SRC = "rig/RiggedFigure.gltf";
const RAG_DROP = 1.0; // spawn height (feet above the floor)
const RAG_TILT = 0.25; // spawn lean (rad about world Z) so the topple is deterministic, not knife-edge

type V3 = [number, number, number];
type Q4 = [number, number, number, number];

// quaternion helpers over [x, y, z, w] arrays (the engine's quat layout); rotation via physics/core's qRotate
function qMul(a: Q4, b: Q4): Q4 {
    return [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
}
const qConj = (q: Q4): Q4 => [-q[0], -q[1], -q[2], q[3]];
const qRotA = (q: Q4, v: V3): V3 => qRotate(q[0], q[1], q[2], q[3], v[0], v[1], v[2]);

// the quat rotating unit vector `a` onto unit vector `b` (shortest arc); antiparallel flips about X or Y
function qFromTo(a: V3, b: V3): Q4 {
    const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    if (d > 1 - 1e-8) return [0, 0, 0, 1];
    if (d < -1 + 1e-8) {
        // 180° about any axis perpendicular to `a`
        const ax: V3 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
        const px = a[1] * ax[2] - a[2] * ax[1];
        const py = a[2] * ax[0] - a[0] * ax[2];
        const pz = a[0] * ax[1] - a[1] * ax[0];
        const pl = Math.hypot(px, py, pz);
        return [px / pl, py / pl, pz / pl, 0];
    }
    const cx = a[1] * b[2] - a[2] * b[1];
    const cy = a[2] * b[0] - a[0] * b[2];
    const cz = a[0] * b[1] - a[1] * b[0];
    const w = 1 + d;
    const l = Math.hypot(cx, cy, cz, w);
    return [cx / l, cy / l, cz / l, w / l];
}

// the 11 capsule bones in the rig's object space (glTF Z-up, feet at z = 0), placed to RiggedFigure's
// skeleton (segment ends at its joint bind positions); fixed creation order — the tumble determinism
// contract. Radii keep non-jointed bones clear of each other (jointed pairs don't collide).
const RAG_BONES: { name: string; a: V3; b: V3; r: number; mass: number }[] = [
    { name: "pelvis", a: [-0.09, 0, 0.66], b: [0.09, 0, 0.66], r: 0.09, mass: 2.5 },
    { name: "chest", a: [0, 0, 0.78], b: [0, 0, 1.04], r: 0.11, mass: 3.5 },
    { name: "head", a: [0, 0, 1.21], b: [0, 0, 1.33], r: 0.09, mass: 1 },
    { name: "upperArmL", a: [0.088, 0.01, 1.074], b: [0.306, 0.023, 0.964], r: 0.05, mass: 0.5 },
    { name: "upperArmR", a: [-0.088, 0.01, 1.074], b: [-0.306, 0.023, 0.964], r: 0.05, mass: 0.5 },
    { name: "lowerArmL", a: [0.306, 0.023, 0.964], b: [0.447, -0.065, 0.882], r: 0.045, mass: 0.4 },
    {
        name: "lowerArmR",
        a: [-0.306, 0.023, 0.964],
        b: [-0.447, -0.065, 0.882],
        r: 0.045,
        mass: 0.4,
    },
    { name: "thighL", a: [0.068, -0.001, 0.614], b: [0.077, -0.058, 0.354], r: 0.055, mass: 1.4 },
    { name: "thighR", a: [-0.068, -0.001, 0.614], b: [-0.077, -0.058, 0.354], r: 0.055, mass: 1.4 },
    { name: "calfL", a: [0.077, -0.058, 0.354], b: [0.078, 0.002, 0.085], r: 0.05, mass: 0.9 },
    { name: "calfR", a: [-0.077, -0.058, 0.354], b: [-0.078, 0.002, 0.085], r: 0.05, mass: 0.9 },
];

// bone → glTF joint names (RiggedFigure's 19-joint skeleton) — every joint follows exactly one bone
const RAG_MAP: Record<string, string[]> = {
    pelvis: ["torso_joint_1"],
    chest: ["torso_joint_2", "torso_joint_3"],
    head: ["neck_joint_1", "neck_joint_2"],
    upperArmL: ["arm_joint_L_1"],
    upperArmR: ["arm_joint_R_1"],
    lowerArmL: ["arm_joint_L_2", "arm_joint_L_3"],
    lowerArmR: ["arm_joint_R_2", "arm_joint_R_3"],
    thighL: ["leg_joint_L_1"],
    thighR: ["leg_joint_R_1"],
    calfL: ["leg_joint_L_2", "leg_joint_L_3", "leg_joint_L_5"],
    calfR: ["leg_joint_R_2", "leg_joint_R_3", "leg_joint_R_5"],
};

// the 11 joints in fixed order (the tumble samples' human.ts recipe): object-space pivots at the rig's
// anatomical joints; `axis` is the joint frame's Z in object space — tumble's cone axis (spherical) and
// hinge axis (revolute) both read frame Z. Shoulder cones point along the arm; hips down; spine/neck up.
const RAG_JOINTS: {
    kind: "ball" | "hinge" | "filter";
    a: string;
    b: string;
    pivot?: V3;
    axis?: V3;
}[] = [
    { kind: "ball", a: "pelvis", b: "chest", pivot: [0, 0, 0.72], axis: [0, 0, 1] },
    { kind: "ball", a: "chest", b: "head", pivot: [0, 0, 1.13], axis: [0, 0, 1] },
    {
        kind: "ball",
        a: "chest",
        b: "upperArmL",
        pivot: [0.088, 0.01, 1.074],
        axis: [0.218, 0.013, -0.11],
    },
    {
        kind: "ball",
        a: "chest",
        b: "upperArmR",
        pivot: [-0.088, 0.01, 1.074],
        axis: [-0.218, 0.013, -0.11],
    },
    {
        kind: "hinge",
        a: "upperArmL",
        b: "lowerArmL",
        pivot: [0.306, 0.023, 0.964],
        axis: [0, 1, 0],
    },
    {
        kind: "hinge",
        a: "upperArmR",
        b: "lowerArmR",
        pivot: [-0.306, 0.023, 0.964],
        axis: [0, -1, 0],
    },
    { kind: "ball", a: "pelvis", b: "thighL", pivot: [0.068, -0.001, 0.614], axis: [0, 0, -1] },
    {
        kind: "ball",
        a: "pelvis",
        b: "thighR",
        pivot: [-0.068, -0.001, 0.614],
        axis: [0, 0, -1],
    },
    { kind: "hinge", a: "thighL", b: "calfL", pivot: [0.077, -0.058, 0.354], axis: [1, 0, 0] },
    { kind: "hinge", a: "thighR", b: "calfR", pivot: [-0.077, -0.058, 0.354], axis: [1, 0, 0] },
    { kind: "filter", a: "thighL", b: "thighR" },
];

interface RagBone {
    eid: number;
    invBind: Float32Array; // the bone's object-space bind inverse — the skinMatrix right factor
    joints: number[]; // palette indices (skin-joint order) this bone drives
    spawnPos: V3;
    spawnQuat: Q4; // world spawn pose — local joint frames + the bones-moved assert derive from it
}

let ragEid = -1;
let ragMeshId = -1;
let ragBones: RagBone[] = [];
let ragBindPos: V3 = [0, 0, 0]; // the pelvis bone's object-space bind (T_inst's fixed right factor)
let ragBindQuat: Q4 = [0, 0, 0, 1];
let ragR0: Q4 = [0, 0, 0, 1]; // spawn orientation: object Z-up → world Y-up, plus the lean
let ragJointsWired = 0;
let ragPalette = new Float32Array(0);
// per-bone prev/curr fixed-tick poses (7 lanes: pos xyz + quat xyzw), the render-interpolation pair
let ragPrev = new Float32Array(0);
let ragCurr = new Float32Array(0);
let ragTick = -1;
let ragSampled = false;
let ragShowBind = false; // assert A/B: true → the driver writes the identity (bind) palette instead
let ragDrawArgs: Mirror | null = null;

// wire the escape-hatch joints once every bone has marshaled (Tumble.body non-null after the first
// fixed tick). Local frames derive from the SPAWN pose analytically — the bodies have already stepped
// by wire time, so a live getLocalPoint would fold the first ticks' free-fall into the anchors.
function wireRagdoll(): void {
    const world = Tumble.world;
    if (!world) return;
    const handles = ragBones.map((b) => Tumble.body(b.eid));
    if (handles.some((h) => !h)) return;
    const index = new Map(RAG_BONES.map((b, i) => [b.name, i]));
    const origin: V3 = [0, RAG_DROP, 0];
    for (const j of RAG_JOINTS) {
        const ia = index.get(j.a)!;
        const ib = index.get(j.b)!;
        const A = handles[ia]!;
        const B = handles[ib]!;
        if (j.kind === "filter") {
            world.createFilterJoint(A, B);
            ragJointsWired++;
            continue;
        }
        const pl = Math.hypot(j.axis![0], j.axis![1], j.axis![2]);
        const axisW = qRotA(ragR0, [j.axis![0] / pl, j.axis![1] / pl, j.axis![2] / pl]);
        const qJ = qFromTo([0, 0, 1], axisW); // both frames share one world orientation → rest rotation = identity
        const pivotW: V3 = [0, 0, 0];
        const rot = qRotA(ragR0, j.pivot!);
        for (let k = 0; k < 3; k++) pivotW[k] = origin[k] + rot[k];
        const frame = (i: number) => {
            const b = ragBones[i];
            const local = qRotA(qConj(b.spawnQuat), [
                pivotW[0] - b.spawnPos[0],
                pivotW[1] - b.spawnPos[1],
                pivotW[2] - b.spawnPos[2],
            ]);
            const q = qMul(qConj(b.spawnQuat), qJ);
            return {
                p: { x: local[0], y: local[1], z: local[2] },
                q: { v: { x: q[0], y: q[1], z: q[2] }, s: q[3] },
            };
        };
        if (j.kind === "ball") {
            world.createSphericalJoint(A, B, {
                localFrameA: frame(ia),
                localFrameB: frame(ib),
                enableConeLimit: true,
                coneAngle: 0.9,
                enableTwistLimit: true,
                lowerTwistAngle: -0.4,
                upperTwistAngle: 0.4,
                enableMotor: true,
                maxMotorTorque: 1.5,
                motorVelocity: { x: 0, y: 0, z: 0 },
            });
        } else {
            world.createRevoluteJoint(A, B, {
                localFrameA: frame(ia),
                localFrameB: frame(ib),
                enableLimit: true,
                lowerAngle: -0.1,
                upperAngle: 2.2,
                enableMotor: true,
                maxMotorTorque: 1.5,
                motorSpeed: 0,
            });
        }
        ragJointsWired++;
    }
}

const _ragQTi: Q4 = [0, 0, 0, 1];

// the pose producer: sample readBody per bone on each fixed tick (prev/curr pair), then every render
// frame nlerp at fixedAlpha, track the instance Transform to the pelvis, and write the palette.
// `simulation` group so the writes land before the Transform compose + LiveSkinSystem flush (draw group).
const RagdollSystem: System = {
    name: "ragdoll-driver",
    group: "simulation",
    annotations: { mode: "always" },
    update(state: State) {
        if (ragEid < 0 || ragBones.length === 0) return;
        if (ragJointsWired === 0) wireRagdoll();
        const backend = Physics.backend;
        if (!backend) return;
        if (state.time.fixedTick !== ragTick) {
            ragTick = state.time.fixedTick;
            for (let i = 0; i < ragBones.length; i++) {
                const s = backend.readBody(ragBones[i].eid);
                if (!s) return; // not marshaled yet — keep the seeded bind pose
                const o = i * 7;
                if (ragSampled) ragPrev.set(ragCurr.subarray(o, o + 7), o);
                ragCurr[o] = s.pos[0];
                ragCurr[o + 1] = s.pos[1];
                ragCurr[o + 2] = s.pos[2];
                ragCurr[o + 3] = s.quat[0];
                ragCurr[o + 4] = s.quat[1];
                ragCurr[o + 5] = s.quat[2];
                ragCurr[o + 6] = s.quat[3];
                if (!ragSampled) ragPrev.set(ragCurr.subarray(o, o + 7), o);
            }
            ragSampled = true;
        }
        if (!ragSampled) return;
        const t = state.time.fixedAlpha;
        const pose = (i: number): { p: V3; q: Q4 } => {
            const o = i * 7;
            return {
                p: [
                    ragPrev[o] + (ragCurr[o] - ragPrev[o]) * t,
                    ragPrev[o + 1] + (ragCurr[o + 1] - ragPrev[o + 1]) * t,
                    ragPrev[o + 2] + (ragCurr[o + 2] - ragPrev[o + 2]) * t,
                ],
                q: nlerpShortest(
                    [ragPrev[o + 3], ragPrev[o + 4], ragPrev[o + 5], ragPrev[o + 6]],
                    [ragCurr[o + 3], ragCurr[o + 4], ragCurr[o + 5], ragCurr[o + 6]],
                    t,
                ),
            };
        };
        const pelvis = pose(0);
        // T_inst = pelvisNow · pelvisObjBind⁻¹ — the instance rides the pelvis, so frustum cull, pick,
        // and the firehose keep a meaningful root while the palette stays bounded
        const instQ = qMul(pelvis.q, qConj(ragBindQuat));
        const off = qRotA(instQ, ragBindPos);
        Transform.pos.set(
            ragEid,
            pelvis.p[0] - off[0],
            pelvis.p[1] - off[1],
            pelvis.p[2] - off[2],
            0,
        );
        Transform.rot.set(ragEid, instQ[0], instQ[1], instQ[2], instQ[3]);
        if (ragShowBind) {
            // the assert's A/B arm: a pure-translation palette (bind rotation, lifted 1.2 along
            // WORLD up expressed in object space — independent of how the ragdoll landed) under the
            // SAME instance transform. The mesh visibly rises only if the vertices actually flow
            // through skinData, so the screen-centroid delta is unambiguous.
            const lift = qRotA(qConj(instQ), [0, 1.2, 0]);
            ragPalette.fill(0);
            for (let j = 0; j * 16 < ragPalette.length; j++) {
                const o = j * 16;
                ragPalette[o] = 1;
                ragPalette[o + 5] = 1;
                ragPalette[o + 10] = 1;
                ragPalette[o + 12] = lift[0];
                ragPalette[o + 13] = lift[1];
                ragPalette[o + 14] = lift[2];
                ragPalette[o + 15] = 1;
            }
            LiveSkin.writePalette(ragEid, ragPalette);
            return;
        }
        // T_inst⁻¹ · boneNow as a pos/quat pair, fed through skinMatrix with the bone's bind inverse
        const qTi = qMul(ragBindQuat, qConj(pelvis.q));
        _ragQTi[0] = qTi[0];
        _ragQTi[1] = qTi[1];
        _ragQTi[2] = qTi[2];
        _ragQTi[3] = qTi[3];
        for (let i = 0; i < ragBones.length; i++) {
            const bone = ragBones[i];
            const b = pose(i);
            const d = qRotA(_ragQTi, [
                b.p[0] - pelvis.p[0],
                b.p[1] - pelvis.p[1],
                b.p[2] - pelvis.p[2],
            ]);
            const relP: V3 = [ragBindPos[0] + d[0], ragBindPos[1] + d[1], ragBindPos[2] + d[2]];
            const relQ = qMul(_ragQTi, b.q);
            for (const j of bone.joints) {
                skinMatrix(relP, relQ, bone.invBind, ragPalette.subarray(j * 16, j * 16 + 16));
            }
        }
        LiveSkin.writePalette(ragEid, ragPalette);
    },
};

// the probe classifies the warm-tinted figure by chroma (the skin-live classifier) and reports its pixel
// count, centroid, and vertical screen extent — the crumple differential the deform assert reads — plus
// the lit neutral-grey floor's pixel count (chroma ≈ 0, luminance well above the dark background), the
// gate for the static Body+Part floor rendering at all (the membership-gated transforms compose).
const RAG_PROBE_WGSL = /* wgsl */ `
@group(0) @binding(0) var fb: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
const luma = vec3<f32>(0.2126, 0.7152, 0.0722);
@compute @workgroup_size(1)
fn main() {
    let dim = vec2<f32>(textureDimensions(fb));
    let W = 160; let H = 120;
    var n = 0.0; var cx = 0.0; var cy = 0.0;
    var lo = 1.0; var hi = 0.0;
    var floorN = 0.0;
    for (var j = 0; j < H; j = j + 1) {
        for (var i = 0; i < W; i = i + 1) {
            let u = (f32(i) + 0.5) / f32(W);
            let v = (f32(j) + 0.5) / f32(H);
            let c = textureLoad(fb, vec2<i32>(vec2<f32>(u, v) * dim), 0).rgb;
            let lum = dot(c, luma);
            let chroma = c.r - c.b;
            if (chroma > 0.06 && lum > 0.03) {
                n += 1.0; cx += u; cy += v;
                lo = min(lo, v); hi = max(hi, v);
            } else if (abs(chroma) < 0.04 && lum > 0.15) {
                floorN += 1.0;
            }
        }
    }
    out[0] = n;
    out[1] = select(0.5, cx / n, n > 0.0);
    out[2] = select(0.5, cy / n, n > 0.0);
    out[3] = select(0.0, hi - lo, n > 0.0);
    out[4] = floorN;
}`;

async function buildRagdoll(state: State): Promise<void> {
    const device = Compute.device;
    ragBones = [];
    ragJointsWired = 0;
    ragTick = -1;
    ragSampled = false;
    ragShowBind = false;
    ragEid = -1;

    const ambient = state.create();
    state.add(ambient, AmbientLight);
    AmbientLight.color.set(ambient, 0xffffff);
    AmbientLight.intensity.set(ambient, 0.35);
    const sun = state.create();
    state.add(sun, DirectionalLight);
    DirectionalLight.color.set(sun, 0xffffff);
    DirectionalLight.intensity.set(sun, 1.6);
    DirectionalLight.direction.set(sun, -0.35, -1, -0.12, 0);
    state.add(sun, Shadow);
    Shadow.distance.set(sun, 20);

    // the physics floor doubles as the receiver: a static Body box + Part renders at the body's pose
    const floor = state.create();
    state.add(floor, Body);
    Body.pos.set(floor, 0, -0.1, 0, 0);
    Body.halfExtents.set(floor, 7, 0.1, 7, 0);
    Body.mass.set(floor, 0);
    state.add(floor, Part);
    state.add(floor, Color);
    Color.rgba.set(floor, 0.5, 0.5, 0.5, 1);

    cam = state.create();
    state.add(cam, Transform);
    state.add(cam, Camera);
    state.add(cam, Sear);
    state.add(cam, Orbit);
    Camera.mode.set(cam, CameraMode.Perspective);
    Camera.fov.set(cam, 50);
    Camera.near.set(cam, 0.1);
    Camera.far.set(cam, 100);
    Camera.clearColor.set(cam, 0x0a0c12);
    Orbit.distance.set(cam, 4.5);
    Orbit.pan.set(cam, 0, 0.9, 0, 0);
    Orbit.yaw.set(cam, 0.6);
    Orbit.pitch.set(cam, 0.3);

    // import the rig on the live path + place its skinned instance (palette seeded to the bind pose)
    const asset = await loadGltf(state, RIG_SRC, { live: true });
    const handle = asset.meshes.find((m) => m.live);
    if (!handle) throw new Error("[ragdoll] RiggedFigure did not import live");
    ragMeshId = handle.mesh;
    ragEid = placeGltf(state, handle);
    // warm tint the probe classifies the figure by chroma. RiggedFigure is untextured, so its material
    // carries palette layer -1 and `sampleAlbedo` returns white (the glTF default) — `white × tint` is the
    // warm base with no albedo texture to supply (shade.ts). The by-construction skin-live mode still hands
    // its rig an explicit white 1×1: it hand-builds a material with no palette entry, so no -1 to key on.
    Color.rgba.set(ragEid, 0.95, 0.7, 0.35, 1);
    ragPalette = new Float32Array(16 * handle.jointCount);
    ragPrev = new Float32Array(7 * RAG_BONES.length);
    ragCurr = new Float32Array(7 * RAG_BONES.length);

    // palette order is the skin's joint list — resolve the bone map's names against the raw glTF json
    const json = await (await fetch(RIG_SRC)).json();
    const names: string[] = json.skins[0].joints.map((n: number) => json.nodes[n].name);
    const jointIdx = new Map(names.map((n, i) => [n, i] as const));

    // spawn frame: object Z-up → world Y-up, leaned RAG_TILT about world Z, feet RAG_DROP above the floor
    const S = Math.SQRT1_2;
    ragR0 = qMul([0, 0, Math.sin(RAG_TILT / 2), Math.cos(RAG_TILT / 2)], [-S, 0, 0, S]);
    const origin: V3 = [0, RAG_DROP, 0];
    for (const bone of RAG_BONES) {
        const mid: V3 = [
            (bone.a[0] + bone.b[0]) / 2,
            (bone.a[1] + bone.b[1]) / 2,
            (bone.a[2] + bone.b[2]) / 2,
        ];
        const dx = bone.b[0] - bone.a[0];
        const dy = bone.b[1] - bone.a[1];
        const dz = bone.b[2] - bone.a[2];
        const len = Math.hypot(dx, dy, dz);
        // the substrate capsule's segment runs along local +Y — align it to the bone direction
        const qAlign = qFromTo([0, 1, 0], [dx / len, dy / len, dz / len]);
        const wr = qRotA(ragR0, mid);
        const spawnPos: V3 = [origin[0] + wr[0], origin[1] + wr[1], origin[2] + wr[2]];
        const spawnQuat = qMul(ragR0, qAlign);
        const eid = state.create();
        state.add(eid, Body);
        Body.shape.set(eid, ShapeKind.Capsule);
        Body.pos.set(eid, spawnPos[0], spawnPos[1], spawnPos[2], 0);
        Body.quat.set(eid, spawnQuat[0], spawnQuat[1], spawnQuat[2], spawnQuat[3]);
        Body.halfExtents.set(eid, 0, len / 2, 0, bone.r);
        Body.mass.set(eid, bone.mass);
        const joints = RAG_MAP[bone.name].map((n) => {
            const j = jointIdx.get(n);
            if (j === undefined) throw new Error(`[ragdoll] rig has no joint named ${n}`);
            return j;
        });
        const objBind = compose(
            mid[0],
            mid[1],
            mid[2],
            qAlign[0],
            qAlign[1],
            qAlign[2],
            qAlign[3],
            1,
            1,
            1,
        );
        ragBones.push({
            eid,
            invBind: invert(objBind, new Float32Array(16)),
            joints,
            spawnPos,
            spawnQuat,
        });
        if (bone.name === "pelvis") {
            ragBindPos = mid;
            ragBindQuat = qAlign;
        }
    }

    state.addSystem(RagdollSystem);

    probePipeline = await device.createComputePipelineAsync({
        label: "ragdoll-probe",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "ragdoll-probe", code: RAG_PROBE_WGSL }),
            entryPoint: "main",
        },
    });
    probeBuf = device.createBuffer({
        label: "ragdoll-probe",
        size: 20, // figure count/x/y + vertical extent + lit-floor count
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    probeBg = null;
    state.addSystem(ProbeSystem);
    await frames(4);
    probeMirror = mirror(probeBuf);
    ragDrawArgs = mirror(Parts.drawArgs!);
    await frames(3);
}

// probe read with the pipeline-depth settling + `.slice()` copy the skin-live mode established (a bare
// Float32Array over snapshot.bytes is a view into the reused staging buffer — two reads would alias)
async function ragProbe(): Promise<Float32Array | null> {
    if (!probeMirror) return null;
    await frames(12);
    await settle(probeMirror);
    return probeMirror.snapshot ? new Float32Array(probeMirror.snapshot.bytes).slice() : null;
}

// the (skin-live surface, rig mesh) pair's survivor count at the camera slot — the reach-bound cull gate
function ragSurvivors(): number | null {
    if (!ragDrawArgs?.snapshot) return null;
    const slot = Views.get(cam)?.slot ?? 0;
    const pairCount = Surfaces.size * Meshes.size;
    const pair = Part.mesh.get(ragEid) * Surfaces.size + Part.surface.get(ragEid);
    const counts = packCounts(ragDrawArgs, slot, pairCount);
    return counts ? counts[pair] : null;
}

async function assertRagdoll(): Promise<Check[]> {
    const checks: Check[] = [];

    // structural: the live substrate published + all 11 bones marshaled + all 11 joints wired
    const skinData = Compute.buffers.has("skinData");
    const jw = LiveSkin.meshes.has(ragMeshId);
    const marshaled = ragBones.filter((b) => Tumble.body(b.eid)).length;
    checks.push({
        name: "live substrate + ragdoll wired",
        pass:
            skinData &&
            jw &&
            marshaled === RAG_BONES.length &&
            ragJointsWired === RAG_JOINTS.length,
        detail: `skinData ${skinData}, JW ${jw}, bodies ${marshaled}/${RAG_BONES.length}, joints ${ragJointsWired}/${RAG_JOINTS.length}`,
    });

    // the ragdoll fell + settled during the bench's measured frames; give a fresh scene a margin too
    await frames(90);

    // the palette received a genuinely crumpled pose: bones rotated relative to the pelvis vs bind
    let moved = 0;
    const backend = Physics.backend;
    const pelvisNow = backend?.readBody(ragBones[0].eid);
    if (pelvisNow) {
        for (let i = 1; i < ragBones.length; i++) {
            const s = backend?.readBody(ragBones[i].eid);
            if (!s) continue;
            const relNow = qMul(qConj([...pelvisNow.quat] as Q4), [...s.quat] as Q4);
            const relBind = qMul(qConj(ragBones[0].spawnQuat), ragBones[i].spawnQuat);
            const d = Math.abs(
                relNow[0] * relBind[0] +
                    relNow[1] * relBind[1] +
                    relNow[2] * relBind[2] +
                    relNow[3] * relBind[3],
            );
            if (2 * Math.acos(Math.min(1, d)) > 0.3) moved++;
        }
    }
    checks.push({
        name: "the ragdoll crumples the palette",
        pass: moved >= 3,
        detail: `${moved}/${ragBones.length - 1} bones rotated >0.3 rad relative to the pelvis`,
    });

    // the settled crumple renders
    const settled = await ragProbe();
    checks.push({
        name: "the fallen figure renders",
        pass: !!settled && settled[0] > 120,
        detail: settled ? `${settled[0]} px, extent ${settled[3].toFixed(3)}` : "no probe snapshot",
    });

    // the static Body+Part floor renders — pins the membership-gated transforms compose on the real
    // GPU (an ungated scatter stomps the tumble backend's CPU-written record: an invisible floor)
    checks.push({
        name: "static physics floor renders",
        pass: !!settled && settled[4] > 2000,
        detail: `lit-floor pixels ${settled?.[4] ?? 0} / 19200`,
    });

    // the palette drives the pixels: swap in a lifted pure-translation palette under the SAME
    // instance transform and the figure's screen centroid must rise. A mesh ignoring the palette
    // (stuck at bind, or not sampling skinData) renders identically in both arms.
    ragShowBind = true;
    const lifted = await ragProbe();
    ragShowBind = false;
    const rose = !!settled && !!lifted && lifted[0] > 120 && settled[2] - lifted[2] > 0.06;
    checks.push({
        name: "the palette drives the rendered mesh (lifted-palette A/B)",
        pass: rose,
        detail:
            settled && lifted
                ? `crumpled ${settled[0]}px @v ${settled[2].toFixed(3)} vs lifted ${lifted[0]}px @v ${lifted[2].toFixed(3)}`
                : "no probe snapshot",
    });

    // the fallen pose still packs a frustum survivor — the pelvis-tracked reach bound covers the crumple
    const survivors = ragSurvivors();
    checks.push({
        name: "fallen pose packs a survivor and draws",
        pass: survivors === 1 && !!settled && settled[0] > 120,
        detail: `survivors ${survivors}, figure pixels ${settled?.[0] ?? 0}`,
    });

    await frames(4);
    const color = Profile.gpu.has("sear:color");
    checks.push({ name: "color pass draws", pass: color, detail: `sear:color ${color}` });
    return checks;
}

// ============================================================================

// the modes grouped by the build path each drives — the select lists their union (one home for the list)
// and each knob's `when` shows it only for the modes whose scene actually reads it. `shaded` is the
// code-authored probe set (lit through zfight); the rest map one mode-prefix to one builder.
const MODES = {
    cull: ["cull"],
    shaded: [
        "lit",
        "spec",
        "spot",
        "spotShadow",
        "pointShadow",
        "cascade",
        "cascade-ortho",
        "cascade-boundary",
        "acne",
        "zfight",
    ],
    fog: ["fog"],
    gltf: ["gltf-model", "gltf-animated", "gltf-spill", "gltf-multi", "gltf-worker"],
    // the live joint-palette substrate — rides the gltf plugin set, but authors its rig by hand (no asset)
    skinLive: ["skin-live"],
    // the live palette's physics producer: RiggedFigure {live} driven by an 11-capsule tumble ragdoll
    ragdoll: ["ragdoll"],
    transparency: ["transparency"],
    background: ["background"],
    sky: ["sky"],
};
const ALL_MODES = Object.values(MODES).flat();
const isMode = (v: Params, m: string) => v.mode === m;

const scenario: Scenario = {
    name: "render",
    params: [
        { key: "mode", type: "select", default: "cull", options: ALL_MODES, rebuild: true },
        // cull-mode knobs
        {
            key: "count",
            type: "number",
            default: 4096,
            min: 0,
            step: 256,
            rebuild: true,
            when: (v) => isMode(v, "cull"),
        },
        {
            key: "lights",
            type: "number",
            default: 12,
            min: 0,
            max: 256,
            step: 4,
            rebuild: true,
            when: (v) => isMode(v, "cull"),
        },
        {
            key: "seed",
            type: "number",
            default: 1,
            rebuild: true,
            when: (v) => isMode(v, "cull"),
        },
        {
            key: "k",
            type: "number",
            default: 0,
            min: 0,
            max: 1,
            step: 0.05,
            label: "transport-k",
            when: (v) => isMode(v, "cull"),
        },
        // 4× MSAA (default) vs single-sample — exercises the AA-off color pass on the real GPU
        {
            key: "antialias",
            type: "bool",
            default: true,
            rebuild: true,
            when: (v) => isMode(v, "cull"),
        },
        // shaded-look knobs: whether the `lit` lamp casts, and its source-sphere radius (m) — the
        // soft-bulb falloff clamp. The floor is far-field (~3 m), so radius doesn't move the lit assert
        {
            key: "shadows",
            type: "bool",
            default: true,
            rebuild: true,
            when: (v) => MODES.shaded.includes(v.mode as string),
        },
        {
            key: "radius",
            type: "number",
            default: 0.1,
            min: 0.01,
            max: 2,
            step: 0.01,
            rebuild: true,
            when: (v) => MODES.shaded.includes(v.mode as string),
        },
        // fog-mode knobs — the march the probe oracle reads (density / height falloff / step count)
        {
            key: "density",
            type: "number",
            default: 0.04,
            min: 0,
            max: 0.5,
            step: 0.005,
            rebuild: true,
            when: (v) => isMode(v, "fog"),
        },
        {
            key: "falloff",
            type: "number",
            default: 0.15,
            min: 0,
            max: 1,
            step: 0.01,
            rebuild: true,
            when: (v) => isMode(v, "fog"),
        },
        {
            key: "steps",
            type: "number",
            default: 32,
            min: 1,
            max: FOG_MAX_STEPS,
            step: 1,
            rebuild: true,
            when: (v) => isMode(v, "fog"),
        },
        // gltf-mode knobs — the Sponza codec variant (gltf-model; ktx drives the compressed bucket path) and
        // the Fox animation clip (gltf-animated)
        {
            key: "variant",
            type: "select",
            default: "gltf",
            options: ["gltf", "draco", "ktx", "ktx-draco"],
            rebuild: true,
            when: (v) => isMode(v, "gltf-model"),
        },
        {
            key: "clip",
            type: "select",
            default: "walk",
            options: FOX_CLIPS,
            rebuild: true,
            when: (v) => isMode(v, "gltf-animated"),
        },
    ],

    async build(_canvas, p: Params) {
        params = p;
        mode = p.mode as string;
        count = 0; // non-cull modes carry no filler — the transport wave is a no-op there

        const gltf = MODES.gltf.includes(mode);
        const skinLive = mode === "skin-live";
        const ragdoll = mode === "ragdoll";
        const plugins = [...corePlugins];
        if (mode === "cull") plugins.push(transportPlugin);
        else if (mode === "fog") plugins.push(FogPlugin);
        else if (gltf || skinLive)
            plugins.push(GltfPlugin); // GltfPlugin owns the live-skin substrate
        else if (ragdoll) plugins.push(GltfPlugin, TumblePlugin);
        else if (mode === "transparency") plugins.push(TransparencyPlugin);
        else if (mode === "background") plugins.push(BackgroundPlugin);
        else if (mode === "sky") plugins.push(SkyPlugin);

        // gltf modes author only the env + camera in the scene; the asset is imported imperatively after
        // build (placeGltfAssets), except gltf-worker which authors both boxes by name (the declarative
        // preloader imports them). The importer is a one-way utility that creates no entities. The other
        // modes author their scene in code.
        const scene = gltf ? gltfScene() : undefined;
        const { state, dispose } = await run({ defaults: false, plugins, scene });

        if (mode === "cull") await buildCull(state, p);
        else if (mode === "fog") await buildFog(state, p);
        else if (gltf) {
            await placeGltfAssets(state, p);
            await frames(2); // settle a couple frames before asserting
        } else if (skinLive) await buildSkinLive(state);
        else if (ragdoll) await buildRagdoll(state);
        else if (mode === "transparency") {
            buildTransparency(state);
            await setupFramebufferProbe(state);
        } else if (mode === "background") {
            buildBackground(state);
            await setupFramebufferProbe(state);
        } else if (mode === "sky") {
            buildSky(state);
            await setupFramebufferProbe(state);
        } else await buildProbe(state, p);

        return { state, dispose };
    },

    assert(state): Promise<Check[]> {
        if (mode === "cull") return assertCull(state);
        if (mode === "fog") return assertFog();
        if (mode === "gltf-model") return assertGltfModel(state);
        if (mode === "gltf-animated") return assertGltfAnimated(state);
        if (mode === "gltf-spill") return assertGltfSpill(state);
        if (mode === "gltf-multi") return assertGltfMulti(state);
        if (mode === "gltf-worker") return assertGltfWorker(state);
        if (mode === "skin-live") return assertSkinLive();
        if (mode === "ragdoll") return assertRagdoll();
        if (mode === "transparency") return assertTransparency();
        if (mode === "background") return assertBackground();
        if (mode === "sky") return assertSky();
        return assertProbe();
    },

    live(state): string {
        if (mode === "transparency") {
            if (!probeMirror?.snapshot) return "render — transparency\nprobe …";
            const out = new Float32Array(probeMirror.snapshot.bytes);
            return `render — transparency\ncentre ${out[5].toFixed(3)} / corner ${out[6].toFixed(3)} (want ≈ ${1 - PANEL_ALPHA})`;
        }
        if (mode === "background") {
            if (!probeMirror?.snapshot) return "render — background\nprobe …";
            const out = new Float32Array(probeMirror.snapshot.bytes);
            return `render — background\nsky corner ${out[6].toFixed(3)} (backdrop) / box centre ${out[5].toFixed(3)} (overdraw)`;
        }
        if (mode === "sky") {
            if (!probeMirror?.snapshot) return "render — sky\nprobe …";
            const out = new Float32Array(probeMirror.snapshot.bytes);
            return `render — sky\nsky corner ${out[6].toFixed(3)} (backdrop) / box centre ${out[5].toFixed(3)} (overdraw)`;
        }
        if (mode === "skin-live") {
            const s = rigSurvivors();
            const p = probeMirror?.snapshot ? new Float32Array(probeMirror.snapshot.bytes) : null;
            const regions = p
                ? `limb ${p[0].toFixed(0)}px @ x${p[1].toFixed(2)}  shadow ${p[3].toFixed(0)}px`
                : "probe …";
            return `render — skin-live\nsurvivors ${s ?? "…"}  reach ${rigReach.toFixed(2)}\n${regions}`;
        }
        if (mode === "ragdoll") {
            const marshaled = ragBones.filter((b) => Tumble.body(b.eid)).length;
            const p = probeMirror?.snapshot ? new Float32Array(probeMirror.snapshot.bytes) : null;
            const fig = p ? `figure ${p[0].toFixed(0)}px extent ${p[3].toFixed(2)}` : "probe …";
            return `render — ragdoll\nbodies ${marshaled}/${ragBones.length || 11}  joints ${ragJointsWired}/${RAG_JOINTS.length}\n${fig}`;
        }
        if (mode.startsWith("gltf-")) {
            const parts = [...state.query([Part])].length;
            const textured = [...state.query([Textured])].length;
            const skinned = [...state.query([Skin])].length;
            return `render — ${mode}\nparts ${parts}  textured ${textured}  skinned ${skinned}`;
        }
        if (mode === "fog") {
            if (!fogOutMirror?.snapshot) return "render — fog\nprobe …";
            const out = new Float32Array(fogOutMirror.snapshot.bytes);
            return `render — fog\nT ${out[0].toFixed(4)} · spot ${out[4].toExponential(2)} · sun ${out[8].toExponential(2)}`;
        }
        if (mode !== "cull") {
            const avg = probeMirror?.snapshot
                ? new Float32Array(probeMirror.snapshot.bytes)[0].toFixed(4)
                : "—";
            return `render — ${mode}\nfloor avg luminance ${avg} (ambient ceiling ${(AMBIENT * ALBEDO).toFixed(4)})`;
        }
        const got = liveCounts();
        const k = (params?.k as number) ?? 0;
        const head = [
            "render — cull",
            `count    ${total} + ${count} filler`,
            `lights   ${lightEids.length}`,
            `transport-k  ${k}`,
        ];
        if (!got) return [...head, "visible  …"].join("\n");
        return [...head, `visible  ${got.visible} / ${total}`, `draws    ${got.draws}`].join("\n");
    },
};

register(scenario);
