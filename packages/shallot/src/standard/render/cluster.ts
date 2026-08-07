import tgpu, { type StorageFlag, type TgpuBuffer, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { State, System } from "../../engine";
import { Compute, capacity } from "../../engine";
import { precompile } from "../../engine/runtime";
import {
    idiv,
    octEncodeNormal,
    srgbToLinear1,
    uniformLoad,
    Xform,
    xformQuat,
} from "../../engine/utils/core";
import { Camera, CameraMode } from "./camera";
import {
    MAX_POINT_LIGHTS,
    PointLight,
    PointLights,
    PointLightsRw,
    Spot,
    Volumetric,
    warnLightOverflow,
} from "./lighting";
import { Render } from "./render";
import { MAX_VIEWS } from "./view";

// The froxel cluster substrate: the grid (per-view view-space cluster AABBs)
// and the per-frame light passes that bin into it (compact + cull, below) —
// what sear's clustered loop reads and what volumetric fog / decals / probes
// read later. 16×9×24 with logarithmic Z-slicing (DOOM 2016 / Olsson 2012):
// log slicing counters NDC depth non-linearity, where linear slicing bands
// everything near the camera into one slice. The view-space AABB per cluster
// depends only on the projection (not the pose), so the GPU build runs only
// on projection change.

/** cluster grid: horizontal screen-space tiles */
export const CLUSTER_X = 16;
/** cluster grid: vertical screen-space tiles */
export const CLUSTER_Y = 9;
/** cluster grid: logarithmic depth slices (DOOM 2016 / Olsson log-Z) */
export const CLUSTER_Z = 24;
/** total froxels per view: `CLUSTER_X * CLUSTER_Y * CLUSTER_Z` */
export const CLUSTER_COUNT = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;

/**
 * a view's cluster-space parameters, derived from its camera projection.
 * `halfW`/`halfH` are the view-space frustum half-extents: at unit view depth
 * for a perspective camera (`tan(fov/2)`, aspect-widened), absolute for an
 * orthographic one (`size`, aspect-widened)
 * @expand
 */
export interface ClusterView {
    perspective: boolean;
    halfW: number;
    halfH: number;
    near: number;
    far: number;
}

/** the camera entity's {@link ClusterView}, from its Camera fields + the view aspect */
export function clusterView(eid: number, aspect: number): ClusterView {
    const perspective = Camera.mode.get(eid) !== CameraMode.Orthographic;
    const halfH = perspective
        ? Math.tan((Camera.fov.get(eid) * Math.PI) / 360)
        : Camera.size.get(eid);
    return {
        perspective,
        halfW: halfH * aspect,
        halfH,
        near: Camera.near.get(eid),
        far: Camera.far.get(eid),
    };
}

/**
 * linearize cluster coords: `(y·X + x)·Z + z`, so a tile's Z-slices are
 * contiguous, so the FS walks depth within a tile without striding
 */
export function clusterIndex(x: number, y: number, z: number): number {
    return (y * CLUSTER_X + x) * CLUSTER_Z + z;
}

/** inverse of {@link clusterIndex} */
export function clusterCoord(index: number): { x: number; y: number; z: number } {
    const z = index % CLUSTER_Z;
    const xy = (index - z) / CLUSTER_Z;
    return { x: xy % CLUSTER_X, y: Math.floor(xy / CLUSTER_X), z };
}

/**
 * the log-slice boundary depth: positive view-space depth where slice `z`
 * begins: `near · (far/near)^(z/Z)`, so slice 0 starts at `near` and slice
 * `CLUSTER_Z` (one past the last) lands exactly on `far`
 */
export function sliceDepth(view: ClusterView, z: number): number {
    return view.near * (view.far / view.near) ** (z / CLUSTER_Z);
}

/**
 * the slot-major froxel index for a pixel at `(fx, fy)` in `[0,1]` (y-down) and positive view depth
 * `viewZ`: the {@link zSlice} log slice, the screen tile, and the view's slot folded into the one index
 * the light cull binned into. Tile `(0, 0)` is NDC `(-1, -1)` — bottom-left — so the y tile flips from
 * the top-down screen y. Sear's color FS passes fragCoord-derived args; the fog march passes its pixel
 * plus the per-step view depth (the tile xy is fixed along the ray, the z slice moves per step).
 * Relocatable, spliced by both (`lightEvalWgsl`, `sear/core`).
 *
 * @example let cell = clusterCell(fx, fy, viewZ, near, far, slot);
 */
export const clusterCell = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32, d.f32, d.u32],
    d.u32,
)((fx, fy, viewZ, near, far, slot) => {
    "use gpu";
    // clamp in float space, then truncate once: a pre-clamp log ratio goes negative for a viewZ just
    // inside near, but the clamp's 0 floor dominates before the truncation ever sees it, so the u32
    // conversion needs no signed intermediate.
    const zs = d.u32(
        std.clamp((std.log(viewZ / near) / std.log(far / near)) * CLUSTER_Z, 0, CLUSTER_Z - 1),
    );
    const tx = std.min(d.u32(fx * CLUSTER_X), d.u32(CLUSTER_X - 1));
    const tyTop = std.min(d.u32(fy * CLUSTER_Y), d.u32(CLUSTER_Y - 1));
    const ty = d.u32(CLUSTER_Y - 1) - tyTop;
    const cluster = (ty * d.u32(CLUSTER_X) + tx) * d.u32(CLUSTER_Z) + zs;
    return slot * d.u32(CLUSTER_COUNT) + cluster;
});

/** the slice containing a positive view-space depth, clamped to the grid */
export function zSlice(view: ClusterView, viewZ: number): number {
    const s = Math.floor(
        (Math.log(viewZ / view.near) / Math.log(view.far / view.near)) * CLUSTER_Z,
    );
    return Math.min(Math.max(s, 0), CLUSTER_Z - 1);
}

/**
 * cluster `(x, y, z)`'s view-space AABB (camera looks down −Z, so `min.z` is
 * the slice's far boundary). Tile `(0, 0)` spans NDC `(-1, -1)`; a perspective
 * frustum's tile corners scale with depth, so the AABB takes min/max across
 * the slice's two boundary depths. The GPU pass is the WGSL twin; the gym
 * Mirror assert pins them together
 */
export function clusterAabb(
    view: ClusterView,
    x: number,
    y: number,
    z: number,
): { min: [number, number, number]; max: [number, number, number] } {
    const loX = (-1 + (2 * x) / CLUSTER_X) * view.halfW;
    const hiX = (-1 + (2 * (x + 1)) / CLUSTER_X) * view.halfW;
    const loY = (-1 + (2 * y) / CLUSTER_Y) * view.halfH;
    const hiY = (-1 + (2 * (y + 1)) / CLUSTER_Y) * view.halfH;
    const dNear = sliceDepth(view, z);
    const dFar = sliceDepth(view, z + 1);
    if (!view.perspective) {
        return { min: [loX, loY, -dFar], max: [hiX, hiY, -dNear] };
    }
    return {
        min: [Math.min(loX * dNear, loX * dFar), Math.min(loY * dNear, loY * dFar), -dFar],
        max: [Math.max(hiX * dNear, hiX * dFar), Math.max(hiY * dNear, hiY * dFar), -dNear],
    };
}

/**
 * the cluster indices a point light's influence sphere touches:
 * sphere-vs-AABB by squared distance from the view-space center to each
 * cluster's box. The TS twin of the light-cull WGSL test; the gym Mirror
 * assert pins them together. `center` is the light's view-space position
 */
export function lightClusters(
    view: ClusterView,
    center: [number, number, number],
    range: number,
): number[] {
    const hit: number[] = [];
    const rangeSq = range * range;
    for (let y = 0; y < CLUSTER_Y; y++) {
        for (let x = 0; x < CLUSTER_X; x++) {
            for (let z = 0; z < CLUSTER_Z; z++) {
                const { min, max } = clusterAabb(view, x, y, z);
                let distSq = 0;
                for (let i = 0; i < 3; i++) {
                    const p = Math.min(Math.max(center[i], min[i]), max[i]);
                    distSq += (p - center[i]) ** 2;
                }
                if (distSq <= rangeSq) hit.push(clusterIndex(x, y, z));
            }
        }
    }
    return hit;
}

// per-view packed ClusterView: two vec4 — (halfW, halfH, near, far) +
// (perspective, 0, 0, 0)
const CLUSTER_VIEW_FLOATS = 8;

/**
 * GPU cluster substrate. `aabbs` holds each cluster's view-space AABB as two
 * `vec4<f32>` (min, max; w unused), slot-major at
 * `(slot · CLUSTER_COUNT + cluster) · 2`, published to `Compute.buffers` as
 * `"clusterAabbs"`. Rebuilt by {@link ClusterSystem} only when a view's
 * projection changes
 * @expand
 */
export interface Clusters {
    aabbs: GPUBuffer | null;
    views: GPUBuffer | null;
    staging: Float32Array;
    last: Float32Array;
}

export const Clusters: Clusters = {
    aabbs: null,
    views: null,
    staging: new Float32Array(MAX_VIEWS * CLUSTER_VIEW_FLOATS),
    last: new Float32Array(MAX_VIEWS * CLUSTER_VIEW_FLOATS),
};

/**
 * pack a camera's {@link ClusterView} into the staging slot, called per view by
 * `BeginFrameSystem`, which reuses the returned view for the View.cluster pack
 */
export function packClusterView(eid: number, aspect: number, slot: number): ClusterView {
    const v = clusterView(eid, aspect);
    const o = slot * CLUSTER_VIEW_FLOATS;
    const s = Clusters.staging;
    s[o] = v.halfW;
    s[o + 1] = v.halfH;
    s[o + 2] = v.near;
    s[o + 3] = v.far;
    s[o + 4] = v.perspective ? 1 : 0;
    return v;
}

const gridLayout = tgpu.bindGroupLayout({
    clusterViews: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    aabbs: { storage: d.arrayOf(d.vec4f), access: "mutable" },
});

// the TGSL twin of clusterAabb — one thread per (cluster, view slot). The grid dimensions are module
// constants, so they fold to literals; `idiv` is the integer division (TGSL's `/` is float division —
// `idiv` is not an over-2²⁴ precaution here, it's what makes the quotient integral at all)
const gridKernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const cluster = input.gid.x;
    if (cluster >= CLUSTER_COUNT) return;
    const slot = input.gid.y;
    const p = gridLayout.$.clusterViews[slot * 2];
    const perspective = gridLayout.$.clusterViews[slot * 2 + 1].x > 0.5;

    const z = cluster % CLUSTER_Z;
    const xy = idiv(cluster, CLUSTER_Z);
    const x = xy % CLUSTER_X;
    const y = idiv(xy, CLUSTER_X);

    const near = p.z;
    const far = p.w;
    const dNear = near * std.pow(far / near, d.f32(z) / CLUSTER_Z);
    const dFar = near * std.pow(far / near, d.f32(z + 1) / CLUSTER_Z);

    const half = d.vec2f(p.x, p.y);
    const lo = std.mul(
        d.vec2f(-1 + (2 * d.f32(x)) / CLUSTER_X, -1 + (2 * d.f32(y)) / CLUSTER_Y),
        half,
    );
    const hi = std.mul(
        d.vec2f(-1 + (2 * d.f32(x + 1)) / CLUSTER_X, -1 + (2 * d.f32(y + 1)) / CLUSTER_Y),
        half,
    );

    let mn = d.vec2f(lo);
    let mx = d.vec2f(hi);
    if (perspective) {
        mn = std.min(std.mul(lo, dNear), std.mul(lo, dFar));
        mx = std.max(std.mul(hi, dNear), std.mul(hi, dFar));
    }
    const base = (slot * CLUSTER_COUNT + cluster) * 2;
    gridLayout.$.aabbs[base] = d.vec4f(mn.x, mn.y, -dFar, 0);
    gridLayout.$.aabbs[base + 1] = d.vec4f(mx.x, mx.y, -dNear, 0);
});

/** the emitted cluster-AABB WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function gridWgsl(): string {
    return tgpu.resolve([gridKernel], { names: "strict" });
}

let _pipe: TgpuComputePipeline | null = null;
let _bound: TgpuComputePipeline | null = null;
let _typedViews: (TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag) | null = null;
let _typedAabbs: (TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag) | null = null;

/**
 * rebuilds the cluster AABB buffer when any active view's projection changed
 * since the last build (the staging prefix is the dirty signal: pose changes
 * never touch it, so a static-projection frame dispatches nothing). Runs after
 * `BeginFrameSystem` (the `first` bucket sorts ahead of every normal system),
 * which packed the staging prefix this frame
 */
export const ClusterSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    update() {
        if (!Render.encoder || !_pipe || Render.shadeCount === 0) return;
        const used = Render.shadeCount * CLUSTER_VIEW_FLOATS;
        let changed = false;
        for (let i = 0; i < used; i++) {
            if (Clusters.staging[i] !== Clusters.last[i]) {
                changed = true;
                break;
            }
        }
        if (!changed) return;
        Clusters.last.set(Clusters.staging.subarray(0, used));
        Compute.device.queue.writeBuffer(
            Clusters.views!,
            0,
            Clusters.staging as Float32Array<ArrayBuffer>,
            0,
            used,
        );
        const pass = Render.encoder.beginComputePass({
            label: "kitchen-cluster-aabbs",
            timestampWrites: Compute.span?.("cluster:aabbs"),
        });
        bindGrid()
            .with(pass)
            .dispatchWorkgroups(Math.ceil(CLUSTER_COUNT / 64), Render.shadeCount);
        pass.end();
    },
};

// bound once, on the forced precompile (which drains after every plugin has warmed). Every input is
// this module's own, allocated in `warmClusters` before the forcer is registered — so a missing one is
// a wiring bug and throws, never a silently skipped frame (`ecs.md` Anti-patterns)
function bindGrid(): TgpuComputePipeline {
    if (_bound) return _bound;
    if (!_pipe || !_typedViews || !_typedAabbs)
        throw new Error("[render] cluster grid used before warmClusters");
    _bound = _pipe.with(
        Compute.root.createBindGroup(gridLayout, {
            clusterViews: _typedViews,
            aabbs: _typedAabbs,
        }),
    );
    return _bound;
}

/** allocate the cluster buffers + compile the AABB-build pipeline */
export function warmClusters(): void {
    if (!Compute.device) return;
    const root = Compute.root;
    Clusters.last.fill(0);
    _bound = null;

    _typedViews = root
        .createBuffer(d.arrayOf(d.vec4f, MAX_VIEWS * (CLUSTER_VIEW_FLOATS / 4)))
        .$usage("storage")
        .$name("kitchen-cluster-views");
    Clusters.views = root.unwrap(_typedViews);
    // typegpu grants COPY_SRC on every buffer it creates, which is what the gym Mirror assert against
    // the TS oracle reads the AABBs back through
    _typedAabbs = root
        .createBuffer(d.arrayOf(d.vec4f, MAX_VIEWS * CLUSTER_COUNT * 2))
        .$usage("storage")
        .$name("kitchen-cluster-aabbs");
    Clusters.aabbs = root.unwrap(_typedAabbs);
    Compute.buffers.set("clusterAabbs", Clusters.aabbs);
    Compute.typed.set("clusterAabbs", _typedAabbs);

    _pipe = root.createComputePipeline({ compute: gridKernel }).$name("kitchen-cluster-aabbs");
    // the bind, not just the dispatch, is deferred into the forcer: it runs after every plugin's warm
    // has resolved (warm hooks run under `Promise.all`), the first moment every input buffer is up
    precompile("kitchen-cluster-aabbs", () => {
        const bound = bindGrid();
        bound.dispatchWorkgroups(0);
        return bound;
    });
}

// The per-frame light passes: compact + cull, the GPU-driven deviation from
// Bevy's CPU light assignment (the firehose has no CPU loop over lights). The
// compact pass scans capacity gated on PointLight membership and atomic-appends
// the live lights — world position from the transforms firehose, params from the
// PointLight slabs — into the compacted list. The cull pass then bins that list
// into the cluster grid (one thread per cluster per view): each light transforms
// to view space once per workgroup batch (shared memory, the DaveH355/logdahl
// structure), sphere-vs-AABB tests against the landed cluster AABBs, and the
// survivors atomic-append into one flat index pool, `lightGrid` recording each
// cluster's (offset, count). Sear's FS reads grid + pool — the per-fragment
// light loop is the cluster's shortlist, not the whole list.

/** per-cluster light index pool: 32 × CLUSTER_COUNT entries shared across views */
export const LIGHT_POOL = CLUSTER_COUNT * 32;

// pool header: [0] next-free counter, [1] overflow (entries that didn't fit).
// Data entries start at element 2; grid offsets are absolute, so the FS indexes
// the same binding without offset arithmetic
const POOL_HEADER = 2;

/**
 * GPU light-cull state. `lights` is the compacted world-space light list
 * (POINT_LIGHTS_STRUCT_WGSL: count header + posRange/color entries), GPU-written
 * each frame by the compact pass. `grid` holds an (offset, count) entry per
 * (view slot, cluster), slot-major; `indices` is the flat index pool the offsets
 * point into ([0] counter, [1] overflow, data from element 2). `viewMats` is the
 * per-slot world→view matrix, staged by `BeginFrameSystem`: the cull pass
 * transforms world-space lights into each view's cluster space with it
 * @expand
 */
export interface LightCull {
    lights: GPUBuffer | null;
    grid: GPUBuffer | null;
    indices: GPUBuffer | null;
    viewMats: GPUBuffer | null;
    viewStaging: Float32Array;
}

export const LightCull: LightCull = {
    lights: null,
    grid: null,
    indices: null,
    viewMats: null,
    viewStaging: new Float32Array(MAX_VIEWS * 16),
};

const compactLayout = tgpu.bindGroupLayout({
    membership: { storage: d.arrayOf(d.u32), access: "readonly" },
    transforms: { storage: d.arrayOf(Xform), access: "readonly" },
    colorF: { storage: d.arrayOf(d.f32), access: "readonly" },
    intensityF: { storage: d.arrayOf(d.f32), access: "readonly" },
    rangeF: { storage: d.arrayOf(d.f32), access: "readonly" },
    radiusF: { storage: d.arrayOf(d.f32), access: "readonly" },
    spotInnerF: { storage: d.arrayOf(d.f32), access: "readonly" },
    spotOuterF: { storage: d.arrayOf(d.f32), access: "readonly" },
    lights: { storage: PointLightsRw, access: "mutable" },
});

const cullLayout = tgpu.bindGroupLayout({
    aabbs: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    lights: { storage: PointLights, access: "readonly" },
    viewMats: { storage: d.arrayOf(d.mat4x4f), access: "readonly" },
    grid: { storage: d.arrayOf(d.vec2u), access: "mutable" },
    pool: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
});

// the GPU twin of the deleted CPU pack: membership-gated scan over capacity, world position from the
// transforms firehose, hex sRGB color decoded to linear with intensity pre-baked, posRange.w = 1/range².
// The three membership gates (PointLight, Spot, Volumetric) come in as captured row bases + masks, so
// they fold to literals — no uniform to bind.
function compactKernel(
    light: { base: number; mask: number },
    spot: { base: number; mask: number },
    vol: { base: number; mask: number },
) {
    // a factory-returned kernel has no binding for `names: "strict"` to read, so it resolves to `fn item`
    // unless named — and that name is what a Tint compile error and every GPU `console.log` line reports
    return tgpu
        .computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const eid = input.gid.x;
            if (eid >= capacity) return;
            if ((compactLayout.$.membership[light.base + eid] & light.mask) === 0) return;
            const range = compactLayout.$.rangeF[eid];
            if (range <= 0) return;
            const i = std.atomicAdd(compactLayout.$.lights.count[0], 1);
            if (i >= MAX_POINT_LIGHTS) return;
            const hex = d.u32(compactLayout.$.colorF[eid]);
            const intensity = compactLayout.$.intensityF[eid];
            const rgb = std.mul(
                d.vec3f(
                    srgbToLinear1(d.f32((hex >> 16) & 0xff) / 255),
                    srgbToLinear1(d.f32((hex >> 8) & 0xff) / 255),
                    srgbToLinear1(d.f32(hex & 0xff) / 255),
                ),
                intensity,
            );
            const pos = compactLayout.$.transforms[eid].pos;
            compactLayout.$.lights.lights[i].posRange = d.vec4f(
                pos.x,
                pos.y,
                pos.z,
                1 / (range * range),
            );
            // color.a carries the source entity id (exact in f32 up to 2^24 ≫ capacity) — the hook a
            // consumer matches per-entity light extensions on (sear's point-shadow casters)
            compactLayout.$.lights.lights[i].color = d.vec4f(rgb.x, rgb.y, rgb.z, d.f32(eid));

            // params.x = source radius (the soft-sphere falloff clamp + representative-point spec). Its sign
            // is the Volumetric opt-in flag: the lit path only ever reads radiusSq = params.x·params.x
            // (sign-immune), so a negated radius leaves shading unchanged while the fog march reads
            // params.x < 0 as "scatter this light" through the haze. max(.,1e-4) keeps the flag a nonzero
            // negative for a radius-0 light
            let radius = compactLayout.$.radiusF[eid];
            if ((compactLayout.$.membership[vol.base + eid] & vol.mask) !== 0) {
                radius = -std.max(radius, 1e-4);
            }
            // the spot lanes (y = cone-axis oct, z/w = angular scale/offset) are (0, 0, 1) for a plain point
            // light so the FS angular factor is 1; a Spot bakes the cone here (axis = the entity's forward,
            // scale/offset = Frostbite getAngleAtt from the inner/outer half-angles — the spotParams oracle's
            // twin)
            let params = d.vec4f(radius, 0, 0, 1);
            if ((compactLayout.$.membership[spot.base + eid] & spot.mask) !== 0) {
                const dir = std.normalize(
                    xformQuat(compactLayout.$.transforms[eid].quat, d.vec3f(0, 0, -1)),
                );
                const cosInner = std.cos(std.radians(compactLayout.$.spotInnerF[eid]));
                const cosOuter = std.cos(std.radians(compactLayout.$.spotOuterF[eid]));
                const scale = 1 / std.max(cosInner - cosOuter, 1e-4);
                params = d.vec4f(
                    radius,
                    std.bitcastU32toF32(octEncodeNormal(dir)),
                    scale,
                    -cosOuter * scale,
                );
            }
            compactLayout.$.lights.lights[i].params = d.vec4f(params);
        })
        .$name("lightCompact");
}

const wgCount = tgpu.workgroupVar(d.u32);
const batch = tgpu.workgroupVar(d.arrayOf(d.vec4f, 64));

// view-space sphere vs cluster AABB: squared distance from the box to the center against range²
// (posRange.w carries 1/range²)
const hits = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec4f],
    d.bool,
)((mn, mx, l) => {
    "use gpu";
    const c = d.vec3f(l.x, l.y, l.z);
    const p = std.clamp(c, mn, mx);
    const delta = std.sub(p, c);
    return std.dot(delta, delta) * l.w <= 1;
});

// one thread per (cluster, view slot). Lights batch through shared memory: each thread of the workgroup
// transforms one light to this view's space, then every thread tests the whole batch against its cluster
// AABB — the mat4 transform runs once per workgroup, not once per cluster. Two sweeps (count, then
// reserve + write) avoid a function-private index array (the Metal dynamically-indexed-private-array
// miscompile, gpu.md). The batch loop bound comes through `uniformLoad` so the in-loop barriers pass
// uniformity analysis; out-of-range threads mask on `live` instead of returning, for the same reason.
const cullKernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationId },
})((input) => {
    "use gpu";
    const cluster = input.gid.x;
    // the dispatch's y covers the shading slots alone (depth-only shadow views sit above
    // Render.shadeCount and never bin — binning them would overflow the shared index pool)
    const slot = input.gid.y;
    const live = cluster < CLUSTER_COUNT;
    if (input.lid.x === 0) wgCount.$ = std.min(cullLayout.$.lights.count.x, MAX_POINT_LIGHTS);
    const n = uniformLoad(wgCount.$);
    const base = (slot * CLUSTER_COUNT + std.min(cluster, CLUSTER_COUNT - 1)) * 2;
    const lo = cullLayout.$.aabbs[base];
    const hi = cullLayout.$.aabbs[base + 1];
    const mn = d.vec3f(lo.x, lo.y, lo.z);
    const mx = d.vec3f(hi.x, hi.y, hi.z);
    const viewMat = cullLayout.$.viewMats[slot];

    let cnt = d.u32(0);
    let b = d.u32(0);
    while (b < n) {
        const li = b + input.lid.x;
        if (li < n) {
            const l = cullLayout.$.lights.lights[li];
            const v = std.mul(viewMat, d.vec4f(l.posRange.x, l.posRange.y, l.posRange.z, 1));
            batch.$[input.lid.x] = d.vec4f(v.x, v.y, v.z, l.posRange.w);
        }
        std.workgroupBarrier();
        const m = std.min(n - b, 64);
        if (live) {
            let j = d.u32(0);
            while (j < m) {
                if (hits(mn, mx, batch.$[j])) cnt = cnt + 1;
                j = j + 1;
            }
        }
        std.workgroupBarrier();
        b = b + 64;
    }

    let off = d.u32(0);
    let take = d.u32(0);
    if (live && cnt > 0) {
        off = std.atomicAdd(cullLayout.$.pool[0], cnt);
        const avail = std.select(d.u32(0), LIGHT_POOL - off, off < LIGHT_POOL);
        take = std.min(cnt, avail);
        if (cnt > take) std.atomicAdd(cullLayout.$.pool[1], cnt - take);
    }

    let w = d.u32(0);
    let b2 = d.u32(0);
    while (b2 < n) {
        const li = b2 + input.lid.x;
        if (li < n) {
            const l = cullLayout.$.lights.lights[li];
            const v = std.mul(viewMat, d.vec4f(l.posRange.x, l.posRange.y, l.posRange.z, 1));
            batch.$[input.lid.x] = d.vec4f(v.x, v.y, v.z, l.posRange.w);
        }
        std.workgroupBarrier();
        const m = std.min(n - b2, 64);
        if (live) {
            let j = d.u32(0);
            while (j < m) {
                if (w < take && hits(mn, mx, batch.$[j])) {
                    std.atomicStore(cullLayout.$.pool[POOL_HEADER + off + w], b2 + j);
                    w = w + 1;
                }
                j = j + 1;
            }
        }
        std.workgroupBarrier();
        b2 = b2 + 64;
    }

    if (live) {
        cullLayout.$.grid[slot * CLUSTER_COUNT + cluster] = d.vec2u(POOL_HEADER + off, take);
    }
});

/** the emitted light compact + cull WGSL — the device-free structural seam their tests resolve.
 *  @internal */
export function lightCullWgsl(
    light: { base: number; mask: number },
    spot: { base: number; mask: number },
    vol: { base: number; mask: number },
): { compact: string; cull: string } {
    return {
        compact: tgpu.resolve([compactKernel(light, spot, vol)], { names: "strict" }),
        cull: tgpu.resolve([cullKernel], { names: "strict" }),
    };
}

let _typedLights: (TgpuBuffer<typeof PointLightsRw> & StorageFlag) | null = null;
let _compactPipe: TgpuComputePipeline | null = null;
let _cullPipe: TgpuComputePipeline | null = null;
let _compactBound: TgpuComputePipeline | null = null;
let _cullBound: TgpuComputePipeline | null = null;

// pool-overflow surfacing: the reserve counter lives GPU-side, so a throttled
// 8-byte readback (copy one frame, map the next) carries the warn — never
// silent truncation, never a per-frame stall
let _overflowStaging: GPUBuffer | null = null;
let _overflowPending = false;
let _overflowInFlight = false;
let _overflowWarned = false;
const OVERFLOW_PERIOD = 240;

// bound once, on the forced precompile. A typed bind group takes a raw GPUBuffer, which is what keeps
// the slab mirrors' and `membership`'s reach-in open. Every input is stable post-warm, so a missing one
// is a wiring bug and gets the named throw — never a skipped frame (`ecs.md` Anti-patterns)
function bindCompact(): TgpuComputePipeline {
    if (_compactBound) return _compactBound;
    if (!_compactPipe || !_typedLights)
        throw new Error("[render] light compact used before warmLightCull");
    const inputs = {
        membership: Compute.buffers.get("membership"),
        transforms: Compute.buffers.get("transforms"),
        colorF: PointLight.color.gpu,
        intensityF: PointLight.intensity.gpu,
        rangeF: PointLight.range.gpu,
        radiusF: PointLight.radius.gpu,
        spotInnerF: Spot.inner.gpu,
        spotOuterF: Spot.outer.gpu,
    };
    const missing = Object.entries(inputs)
        .filter(([, buffer]) => !buffer)
        .map(([name]) => name);
    if (missing.length > 0) {
        throw new Error(
            `[render] light compact inputs missing (${missing.join(", ")}) — SlabPlugin + TransformsPlugin must be loaded`,
        );
    }
    _compactBound = _compactPipe.with(
        Compute.root.createBindGroup(compactLayout, {
            ...(inputs as Required<{ [K in keyof typeof inputs]: GPUBuffer }>),
            lights: _typedLights,
        }),
    );
    return _compactBound;
}

function bindCull(): TgpuComputePipeline {
    if (_cullBound) return _cullBound;
    if (!_cullPipe || !_typedAabbs || !LightCull.lights)
        throw new Error("[render] light cull used before warmLightCull");
    // the light list binds RAW here and typed in the compact group: same buffer, two schemas (the
    // writer's count word is atomic, which WGSL forbids in a read-only binding — `PointLightsRw` vs
    // `PointLights`, layouts pinned equal in lighting.test.ts)
    _cullBound = _cullPipe.with(
        Compute.root.createBindGroup(cullLayout, {
            aabbs: _typedAabbs,
            lights: LightCull.lights,
            viewMats: LightCull.viewMats!,
            grid: LightCull.grid!,
            pool: LightCull.indices!,
        }),
    );
    return _cullBound;
}

function checkOverflow(): void {
    if (!_overflowStaging) return;
    // copy was submitted with last frame's encoder — safe to map now
    if (_overflowInFlight) return;
    _overflowInFlight = true;
    _overflowStaging
        .mapAsync(GPUMapMode.READ)
        .then(() => {
            const words = new Uint32Array(_overflowStaging!.getMappedRange());
            const dropped = words[1];
            if (dropped > 0) {
                if (!_overflowWarned) {
                    _overflowWarned = true;
                    console.warn(
                        `kitchen: light index pool overflow — ${dropped} cluster-light entries dropped this frame (pool ${LIGHT_POOL})`,
                    );
                }
            } else {
                _overflowWarned = false;
            }
            _overflowStaging!.unmap();
            _overflowInFlight = false;
        })
        .catch(() => {
            _overflowInFlight = false;
        });
}

/**
 * per-frame light compact + cull: builds the compacted light list from the
 * PointLight slabs + transforms firehose, then bins it into the cluster grid.
 * Runs after `ClusterSystem` by registration order, before the renderers
 * (which sort after `BeginFrameSystem` in the same registration stream)
 */
export const LightCullSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    update(state) {
        if (!Render.encoder || !_compactPipe || !_cullPipe || Render.shadeCount === 0) return;
        warnLightOverflow(state);

        Compute.device.queue.writeBuffer(
            LightCull.viewMats!,
            0,
            LightCull.viewStaging as Float32Array<ArrayBuffer>,
            0,
            Render.shadeCount * 16,
        );
        Render.encoder.clearBuffer(LightCull.lights!, 0, 16);
        Render.encoder.clearBuffer(LightCull.indices!, 0, POOL_HEADER * 4);
        const pass = Render.encoder.beginComputePass({
            label: "kitchen-light-cull",
            timestampWrites: Compute.span?.("light:cull"),
        });
        bindCompact()
            .with(pass)
            .dispatchWorkgroups(Math.ceil(capacity / 64));
        bindCull()
            .with(pass)
            .dispatchWorkgroups(Math.ceil(CLUSTER_COUNT / 64), Render.shadeCount);
        pass.end();

        if (_overflowPending) {
            _overflowPending = false;
            checkOverflow();
        } else if (Compute.frame % OVERFLOW_PERIOD === 0 && !_overflowInFlight) {
            Render.encoder.copyBufferToBuffer(LightCull.indices!, 0, _overflowStaging!, 0, 8);
            _overflowPending = true;
        }
    },
};

/** allocate the light-cull buffers + compile the compact and cull pipelines */
export function warmLightCull(state: State): void {
    if (!Compute.device) return;
    const device = Compute.device;
    const root = Compute.root;
    _compactBound = null;
    _cullBound = null;
    _overflowPending = false;

    _typedLights = root.createBuffer(PointLightsRw).$usage("storage").$name("kitchen-lights");
    LightCull.lights = root.unwrap(_typedLights);
    // COPY_SRC throughout for the gym Mirror asserts against the TS oracle (typegpu grants it on the
    // buffers it creates)
    LightCull.grid = device.createBuffer({
        label: "kitchen-light-grid",
        size: MAX_VIEWS * CLUSTER_COUNT * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    LightCull.indices = device.createBuffer({
        label: "kitchen-light-indices",
        size: (POOL_HEADER + LIGHT_POOL) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    LightCull.viewMats = device.createBuffer({
        label: "kitchen-light-views",
        size: MAX_VIEWS * 64,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _overflowInFlight = false;
    _overflowStaging = device.createBuffer({
        label: "kitchen-light-overflow",
        size: 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    Compute.buffers.set("lightGrid", LightCull.grid);
    Compute.buffers.set("lightIndices", LightCull.indices);
    Compute.typed.set(
        "lightGrid",
        root
            .createBuffer(d.arrayOf(d.vec2u, MAX_VIEWS * CLUSTER_COUNT), LightCull.grid)
            .$usage("storage"),
    );
    Compute.typed.set(
        "lightIndices",
        root
            .createBuffer(d.arrayOf(d.u32, POOL_HEADER + LIGHT_POOL), LightCull.indices)
            .$usage("storage"),
    );

    const bit = state.membership.bit(PointLight);
    const spotBit = state.membership.bit(Spot);
    const volBit = state.membership.bit(Volumetric);
    _compactPipe = root
        .createComputePipeline({
            compute: compactKernel(
                { base: bit.gen * capacity, mask: bit.mask },
                { base: spotBit.gen * capacity, mask: spotBit.mask },
                { base: volBit.gen * capacity, mask: volBit.mask },
            ),
        })
        .$name("kitchen-light-compact");
    _cullPipe = root.createComputePipeline({ compute: cullKernel }).$name("kitchen-light-cull");
    precompile("kitchen-light-compact", () => {
        const bound = bindCompact();
        bound.dispatchWorkgroups(0);
        return bound;
    });
    precompile("kitchen-light-cull", () => {
        const bound = bindCull();
        bound.dispatchWorkgroups(0);
        return bound;
    });
}
