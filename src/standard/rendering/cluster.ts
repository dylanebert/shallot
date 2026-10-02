import tgpu, { type StorageFlag, type TgpuBuffer, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    BeginFrameSystem,
    Camera,
    CameraMode,
    computeViewProj,
    MAX_VIEWS,
    PointLight,
    RenderContext,
    SpotLight,
    Views,
    VolumetricLight,
} from "../../core/rendering";
import type { System, World } from "../../engine";
import { globalTransformTable } from "../../engine";
import { precompile, probeBuffer } from "../../engine/runtime";
import {
    idiv,
    octEncodeNormal,
    srgbToLinear1,
    uniformLoad,
    Xform,
    xformQuat,
} from "../../engine/utils";
import { MAX_POINT_LIGHTS, PointLights, PointLightsRw, warnLightOverflow } from "./lighting";

interface ClusterGpuState {
    clusters: Clusters;
    lightCull: LightCull;
    pipe: TgpuComputePipeline | null;
    bound: { pipeline: GPUComputePipeline; group: GPUBindGroup } | null;
    typedViews: (TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag) | null;
    typedAabbs: (TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag) | null;
    compactPipe: TgpuComputePipeline | null;
    cullPipe: TgpuComputePipeline | null;
    compactBound: { pipeline: GPUComputePipeline; group: GPUBindGroup } | null;
    compactGeneration: Int32Array;
    lightCountBuffer: GPUBuffer | null;
    lightCountValue: number;
    cullBound: { pipeline: GPUComputePipeline; group: GPUBindGroup } | null;
    viewProj: Float32Array;
    clusterView: ClusterView;
    viewMatrices: Float32Array[];
    gridPass: GPUComputePassDescriptor;
    cullPass: GPUComputePassDescriptor;
}

export const clusterGpuKey = { create: createClusterGpuState };

function createClusterGpuState(): ClusterGpuState {
    return {
        clusters: {
            aabbs: null,
            views: null,
            staging: new Float32Array(MAX_VIEWS * CLUSTER_VIEW_FLOATS),
            last: new Float32Array(MAX_VIEWS * CLUSTER_VIEW_FLOATS),
        },
        lightCull: {
            lights: null,
            viewMats: null,
            viewStaging: new Float32Array(MAX_VIEWS * 16),
        },
        pipe: null,
        bound: null,
        typedViews: null,
        typedAabbs: null,
        compactPipe: null,
        cullPipe: null,
        compactBound: null,
        compactGeneration: new Int32Array(4).fill(-1),
        lightCountBuffer: null,
        lightCountValue: -1,
        cullBound: null,
        viewProj: new Float32Array(16),
        clusterView: { perspective: false, halfW: 0, halfH: 0, near: 0, far: 0 },
        viewMatrices: [],
        gridPass: { label: "shallot-cluster-aabbs" },
        cullPass: { label: "shallot-light-cull" },
    };
}

function _clusterGpu(world: World): ClusterGpuState {
    return world.resource(clusterGpuKey);
}

const LightInput = d
    .struct({
        color: d.f32,
        intensity: d.f32,
        range: d.f32,
        radius: d.f32,
        spotInner: d.f32,
        spotOuter: d.f32,
        flags: d.u32,
        padding: d.u32,
    })
    .$name("LightInput");
const LIGHT_SPOT = 1;
const LIGHT_VOLUMETRIC = 2;
export const lightInputKey = { create: createLightInputTable };
const lightCountData = new Uint32Array(1);

function createLightInputTable(world: World) {
    return world.table("lightInputs", LightInput);
}

function lightInputTable(world: World) {
    return world.resource(lightInputKey);
}

/** Create this world's cluster and dense light-input state during StandardRenderingPlugin initialization. */
export function initializeClusterState(world: World): void {
    world.resource(clusterGpuKey);
    const table = lightInputTable(world);
    table.bindFields(PointLight, {
        color: "color",
        intensity: "intensity",
        range: "range",
        radius: "radius",
    });
    table.bindMembership(PointLight);
    table.bindFields(SpotLight, {
        color: "color",
        intensity: "intensity",
        range: "range",
        radius: "radius",
        spotInner: "innerAngle",
        spotOuter: "outerAngle",
    });
    table.bindMembership(SpotLight);
    table.bindPresence(SpotLight, "flags", LIGHT_SPOT);
    table.bindPresence(VolumetricLight, "flags", LIGHT_VOLUMETRIC);
}

// The froxel cluster substrate: the grid (per-view view-space cluster AABBs)
// and the per-frame light passes that bin into it (compact + cull, below) —
// what standard's clustered loop reads and what volumetric fog / decals / probes
// read later. 16×9×24 with logarithmic Z-slicing (DOOM 2016 / Olsson 2012):
// log slicing counters NDC depth non-linearity, where linear slicing bands
// everything near the camera into one slice. The view-space AABB per cluster
// depends only on the projection (not the view position), so the GPU build runs only
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
 */
export interface ClusterView {
    perspective: boolean;
    halfW: number;
    halfH: number;
    near: number;
    far: number;
}

/** the camera entity's {@link ClusterView}, from its Camera fields + the view aspect */
export function clusterView(world: World, eid: number, aspect: number): ClusterView {
    return readClusterView(world, eid, aspect, {} as ClusterView);
}

function readClusterView(world: World, eid: number, aspect: number, out: ClusterView): ClusterView {
    const perspective = world.storage(Camera).mode.get(eid) !== CameraMode.Orthographic;
    const halfH = perspective
        ? Math.tan((world.storage(Camera).fov.get(eid) * Math.PI) / 360)
        : world.storage(Camera).size.get(eid);
    out.perspective = perspective;
    out.halfW = halfH * aspect;
    out.halfH = halfH;
    out.near = world.storage(Camera).near.get(eid);
    out.far = world.storage(Camera).far.get(eid);
    return out;
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
 * the top-down screen y. StandardRenderer's color FS passes fragCoord-derived args; the fog march passes its pixel
 * plus the per-step view depth (the tile xy is fixed along the ray, the z slice moves per step).
 * Relocatable, spliced by both (`lightEvalWgsl`, `standard`).
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
 * the slice's two boundary depths. The GPU pass is the WGSL twin
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
 * cluster's box. The TS twin of the light-cull WGSL test.
 * `center` is the light's view-space position
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
 * `(slot · CLUSTER_COUNT + cluster) · 2`, published to `world.gpu.buffers` as
 * `"clusterAabbs"`. Rebuilt by {@link UpdateLightClustersSystem} only when a view's
 * projection changes
 */
export interface Clusters {
    aabbs: GPUBuffer | null;
    views: GPUBuffer | null;
    staging: Float32Array;
    last: Float32Array;
}

export const Clusters: import("../../engine").Resource<Clusters> = {
    create: (world) => world.resource(clusterGpuKey).clusters,
};

/**
 * Pack a camera's cluster projection into standard's slot-major staging.
 */
export function packClusterView(
    world: World,
    eid: number,
    aspect: number,
    slot: number,
): ClusterView {
    const v = readClusterView(world, eid, aspect, _clusterGpu(world).clusterView);
    const o = slot * CLUSTER_VIEW_FLOATS;
    const s = world.resource(Clusters).staging;
    s[o] = v.halfW;
    s[o + 1] = v.halfH;
    s[o + 2] = v.near;
    s[o + 3] = v.far;
    s[o + 4] = v.perspective ? 1 : 0;
    return v;
}

// group 0 is declared, not inferred: the dispatch is issued on a raw pass, which addresses a bind group
// by index, so the kernel's one group pins its index here
const gridLayout = tgpu
    .bindGroupLayout({
        clusterViews: { storage: d.arrayOf(d.vec4f), access: "readonly" },
        aabbs: { storage: d.arrayOf(d.vec4f), access: "mutable" },
    })
    .$idx(0);

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

/**
 * rebuilds the cluster AABB buffer when any active view's projection changed
 * since the last build (the staging prefix is the dirty signal: GlobalTransform changes
 * never touch it, so a static-projection frame dispatches nothing). Runs after
 * `BeginFrameSystem` (the `first` bucket sorts ahead of every normal system),
 * after which standard packs its own cluster projection and world→view matrices
 */
export const UpdateLightClustersSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(world: World) {
        const _render = world.resource(RenderContext);
        const _clusterGpu = world.resource(clusterGpuKey);
        const _clusters = world.resource(Clusters);

        if (!_render.encoder || !_clusterGpu.pipe || _render.shadeCount === 0) return;
        for (const [eid, view] of world.resource(Views)) {
            if (!view.framebuffer || view.slot >= _render.shadeCount) continue;
            packClusterView(world, eid, view.width / view.height, view.slot);
            computeViewProj(
                world,
                eid,
                view.width / view.height,
                _clusterGpu.viewProj,
                _clusterGpu.viewMatrices[view.slot],
            );
        }
        const used = _render.shadeCount * CLUSTER_VIEW_FLOATS;
        let changed = false;
        for (let i = 0; i < used; i++) {
            if (_clusters.staging[i] !== _clusters.last[i]) {
                changed = true;
                break;
            }
        }
        if (!changed) return;
        _clusters.last.set(_clusters.staging.subarray(0, used));
        world.gpu.device.queue.writeBuffer(
            _clusters.views!,
            0,
            _clusters.staging as Float32Array<ArrayBuffer>,
            0,
            used,
        );
        _clusterGpu.gridPass.timestampWrites = world.gpu.span?.("cluster:aabbs");
        const grid = bindGrid(world);
        const pass = _render.encoder.beginComputePass(_clusterGpu.gridPass);
        pass.setPipeline(grid.pipeline);
        pass.setBindGroup(0, grid.group);
        pass.dispatchWorkgroups(Math.ceil(CLUSTER_COUNT / 64), _render.shadeCount);
        pass.end();
    },
};

// Pass descriptors are part of the world's mutable dispatch state.

// bound once, on the forced precompile (which drains after every plugin has warmed). Every input is
// this module's own, allocated in `warmClusters` before the forcer is registered — so a missing one is
// a wiring bug and throws, never a silently skipped frame
function bindGrid(world: World): { pipeline: GPUComputePipeline; group: GPUBindGroup } {
    const _clusterGpu = world.resource(clusterGpuKey);

    if (_clusterGpu.bound) return _clusterGpu.bound;
    if (!_clusterGpu.pipe || !_clusterGpu.typedViews || !_clusterGpu.typedAabbs)
        throw new Error("[render] cluster grid used before warmClusters");
    _clusterGpu.bound = {
        pipeline: world.gpu.root.unwrap(_clusterGpu.pipe),
        group: world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(gridLayout, {
                clusterViews: _clusterGpu.typedViews,
                aabbs: _clusterGpu.typedAabbs,
            }),
        ),
    };
    return _clusterGpu.bound;
}

/** allocate the cluster buffers + compile the AABB-build pipeline */
export function warmClusters(world: World): void {
    const _clusters = world.resource(Clusters);
    const _clusterGpu = world.resource(clusterGpuKey);

    if (!world.gpu.device) return;
    const root = world.gpu.root;
    _clusters.last.fill(0);
    _clusterGpu.bound = null;

    _clusterGpu.typedViews = root
        .createBuffer(d.arrayOf(d.vec4f, MAX_VIEWS * (CLUSTER_VIEW_FLOATS / 4)))
        .$usage("storage")
        .$name("shallot-cluster-views");
    _clusters.views = root.unwrap(_clusterGpu.typedViews);
    // typegpu grants COPY_SRC on every buffer it creates, which is what a requested readback
    // reads the AABBs back through
    _clusterGpu.typedAabbs = root
        .createBuffer(d.arrayOf(d.vec4f, MAX_VIEWS * CLUSTER_COUNT * 2))
        .$usage("storage")
        .$name("shallot-cluster-aabbs");
    _clusters.aabbs = root.unwrap(_clusterGpu.typedAabbs);
    world.gpu.buffers.set("clusterAabbs", _clusters.aabbs);
    world.gpu.typed.set("clusterAabbs", _clusterGpu.typedAabbs);

    _clusterGpu.pipe = root
        .createComputePipeline({ compute: gridKernel })
        .$name("shallot-cluster-aabbs");
    // the bind is deferred into the forcer: it runs after every plugin's warm
    // has resolved (warm hooks run under `Promise.all`), the first moment every input buffer is up
    precompile(world, "shallot-cluster-aabbs", () => {
        // the raw pipeline, already unwrapped for the dispatch: the forcer's raw-pipeline shape, which
        // Dawn compiles on the drain like any other
        return [bindGrid(world).pipeline];
    });
}

// The per-frame light passes: compact + cull, the GPU-driven deviation from
// Bevy's CPU light assignment (the firehose has no CPU loop over lights). The
// compact pass reads active PointLight table rows and atomic-appends
// the live lights — world position from the GlobalTransform table, params from the
// PointLight table — into the compacted list. The cull pass then bins that list
// into the cluster grid (one thread per cluster per view): each light is transformed
// to view space once per workgroup batch (shared memory, the DaveH355/logdahl
// structure), sphere-vs-AABB tests against the landed cluster AABBs, and the
// survivors atomic-append into one flat index pool, `lightGrid` recording each
// cluster's (offset, count). StandardRenderer's FS reads grid + pool — the per-fragment
// light loop is the cluster's shortlist, not the whole list.

/** per-cluster light index pool: 32 × CLUSTER_COUNT entries shared across views */
export const LIGHT_POOL = CLUSTER_COUNT * 32;

// pool header: [0] next-free counter, [1] overflow (entries that didn't fit).
// Data entries start at element 2; grid offsets are absolute, so the FS indexes
// the same binding without offset arithmetic
const POOL_HEADER = 2;

/** GPU-written light list, per-view grid and index pool in one binding. The runtime tail
 * preserves the pool's exact size without end padding; offsets are unchanged within each table. */
export const LightClusters = d.struct({
    lights: PointLights,
    grid: d.arrayOf(d.vec2u, MAX_VIEWS * CLUSTER_COUNT),
    indices: d.arrayOf(d.u32, 0),
});
const LightClustersRw = d.struct({
    lights: PointLightsRw,
    grid: d.arrayOf(d.vec2u, MAX_VIEWS * CLUSTER_COUNT),
    indices: d.arrayOf(d.atomic(d.u32), 0),
});
export const LIGHT_GRID_OFFSET = d.sizeOf(PointLights);
export const LIGHT_INDICES_OFFSET = LIGHT_GRID_OFFSET + MAX_VIEWS * CLUSTER_COUNT * 8;

/**
 * GPU light-cull state. `lights` holds {@link LightClusters}: the compacted light list,
 * slot-major grid and flat index pool. The pool starts with a counter and overflow word;
 * grid offsets address its data from element 2. Compact and cull write the same allocation
 * in command order. `viewMats` is the
 * per-slot world→view matrix, staged by `BeginFrameSystem`: the cull pass
 * transforms world-space lights into each view's cluster space with it
 */
export interface LightCull {
    lights: GPUBuffer | null;
    viewMats: GPUBuffer | null;
    viewStaging: Float32Array;
}

export const LightCull: import("../../engine").Resource<LightCull> = {
    create: (world) => world.resource(clusterGpuKey).lightCull,
};

const compactLayout = tgpu
    .bindGroupLayout({
        lightRows: { storage: d.arrayOf(d.vec2u), access: "readonly" },
        lightInput: { storage: d.arrayOf(LightInput), access: "readonly" },
        globalTransforms: { storage: d.arrayOf(Xform), access: "readonly" },
        globalTransformRows: { storage: d.arrayOf(d.u32), access: "readonly" },
        lightCount: { uniform: d.u32 },
        lights: { storage: LightClustersRw, access: "mutable" },
    })
    .$idx(0);

const cullLayout = tgpu
    .bindGroupLayout({
        aabbs: { storage: d.arrayOf(d.vec4f), access: "readonly" },
        lights: { storage: LightClustersRw, access: "mutable" },
        viewMats: { storage: d.arrayOf(d.mat4x4f), access: "readonly" },
    })
    .$idx(0);

// Compact only active point-light rows. Dense table slots feed the record fields; the optional eid map is
// a point lookup into the GlobalTransform table, never a capacity-sized pass. Hex sRGB is decoded on GPU.
function compactKernel() {
    return tgpu
        .computeFn({
            workgroupSize: [64],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const index = input.gid.x;
            if (index >= compactLayout.$.lightCount) return;
            const entry = compactLayout.$.lightRows[index];
            const eid = entry.x;
            const record = compactLayout.$.lightInput[entry.y];
            const globalTransformEncoded = compactLayout.$.globalTransformRows[eid];
            if (globalTransformEncoded === 0 || record.range <= 0) return;
            const globalTransform = compactLayout.$.globalTransforms[globalTransformEncoded - 1];
            const i = std.atomicAdd(compactLayout.$.lights.lights.count[0], 1);
            if (i >= MAX_POINT_LIGHTS) return;
            const hex = d.u32(record.color);
            const rgb = std.mul(
                d.vec3f(
                    srgbToLinear1(d.f32((hex >>> 16) & 0xff) / 255),
                    srgbToLinear1(d.f32((hex >>> 8) & 0xff) / 255),
                    srgbToLinear1(d.f32(hex & 0xff) / 255),
                ),
                record.intensity,
            );
            const pos = globalTransform.pos;
            compactLayout.$.lights.lights.lights[i].posRange = d.vec4f(
                pos.x,
                pos.y,
                pos.z,
                1 / (record.range * record.range),
            );
            // color.a carries the source entity id for per-entity light extensions.
            compactLayout.$.lights.lights.lights[i].color = d.vec4f(
                rgb.x,
                rgb.y,
                rgb.z,
                d.f32(eid),
            );

            let radius = record.radius;
            if ((record.flags & LIGHT_VOLUMETRIC) !== 0) radius = -std.max(radius, 1e-4);
            let params = d.vec4f(radius, 0, 0, 1);
            if ((record.flags & LIGHT_SPOT) !== 0) {
                const dir = std.normalize(xformQuat(globalTransform.quat, d.vec3f(0, 0, -1)));
                const cosInner = std.cos(std.radians(record.spotInner));
                const cosOuter = std.cos(std.radians(record.spotOuter));
                const scale = 1 / std.max(cosInner - cosOuter, 1e-4);
                params = d.vec4f(
                    radius,
                    std.bitcastU32toF32(octEncodeNormal(dir)),
                    scale,
                    -cosOuter * scale,
                );
            }
            compactLayout.$.lights.lights.lights[i].params = d.vec4f(params);
        })
        .$name("lightCompact");
}

const wgCount = tgpu.workgroupVar(d.u32);
const wgCountUniform = uniformLoad(wgCount);
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
// miscompile). The batch loop bound comes through `uniformLoad` so the in-loop barriers pass
// uniformity analysis; out-of-range threads mask on `live` instead of returning, for the same reason.
const cullKernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationId },
})((input) => {
    "use gpu";
    const cluster = input.gid.x;
    // the dispatch's y covers the shading slots alone (depth-only shadow views sit above
    // RenderContext.shadeCount and never bin — binning them would overflow the shared index pool)
    const slot = input.gid.y;
    const live = cluster < CLUSTER_COUNT;
    if (input.lid.x === 0)
        wgCount.$ = std.min(std.atomicLoad(cullLayout.$.lights.lights.count[0]), MAX_POINT_LIGHTS);
    const n = wgCountUniform.$;
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
            const l = cullLayout.$.lights.lights.lights[li];
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
        off = std.atomicAdd(cullLayout.$.lights.indices[0], cnt);
        const avail = std.select(d.u32(0), LIGHT_POOL - off, off < LIGHT_POOL);
        take = std.min(cnt, avail);
        if (cnt > take) std.atomicAdd(cullLayout.$.lights.indices[1], cnt - take);
    }

    let w = d.u32(0);
    let b2 = d.u32(0);
    while (b2 < n) {
        const li = b2 + input.lid.x;
        if (li < n) {
            const l = cullLayout.$.lights.lights.lights[li];
            const v = std.mul(viewMat, d.vec4f(l.posRange.x, l.posRange.y, l.posRange.z, 1));
            batch.$[input.lid.x] = d.vec4f(v.x, v.y, v.z, l.posRange.w);
        }
        std.workgroupBarrier();
        const m = std.min(n - b2, 64);
        if (live) {
            let j = d.u32(0);
            while (j < m) {
                if (w < take && hits(mn, mx, batch.$[j])) {
                    std.atomicStore(cullLayout.$.lights.indices[POOL_HEADER + off + w], b2 + j);
                    w = w + 1;
                }
                j = j + 1;
            }
        }
        std.workgroupBarrier();
        b2 = b2 + 64;
    }

    if (live) {
        cullLayout.$.lights.grid[slot * CLUSTER_COUNT + cluster] = d.vec2u(POOL_HEADER + off, take);
    }
});

// Keep bind groups until a table buffer generation changes; row membership alone never rebuilds one.
function bindCompact(world: World): { pipeline: GPUComputePipeline; group: GPUBindGroup } {
    const _clusterGpu = world.resource(clusterGpuKey);

    const buffer = world.resource(LightCull).lights;
    if (!_clusterGpu.compactPipe || !buffer || !_clusterGpu.lightCountBuffer)
        throw new Error("[render] light compact used before warmLightCull");
    const lights = lightInputTable(world);
    const globalTransforms = globalTransformTable(world);
    const generation = _clusterGpu.compactGeneration;
    if (
        _clusterGpu.compactBound &&
        generation[0] === lights.generation &&
        generation[1] === lights.activeGeneration &&
        generation[2] === globalTransforms.generation &&
        generation[3] === globalTransforms.mapGeneration
    )
        return _clusterGpu.compactBound;
    const inputs = {
        lightRows: lights.activeRowsBuffer,
        lightInput: lights.buffer,
        globalTransforms: globalTransforms.buffer,
        globalTransformRows: globalTransforms.eidToRowBuffer,
        lightCount: _clusterGpu.lightCountBuffer,
    };
    const missing = Object.entries(inputs)
        .filter(([, buffer]) => !buffer)
        .map(([name]) => name);
    if (missing.length > 0) {
        throw new Error(`[render] light compact table inputs missing (${missing.join(", ")})`);
    }
    _clusterGpu.compactBound = {
        pipeline: world.gpu.root.unwrap(_clusterGpu.compactPipe),
        group: world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(compactLayout, {
                ...(inputs as Required<{ [K in keyof typeof inputs]: GPUBuffer }>),
                lights: buffer,
            }),
        ),
    };
    generation[0] = lights.generation;
    generation[1] = lights.activeGeneration;
    generation[2] = globalTransforms.generation;
    generation[3] = globalTransforms.mapGeneration;
    return _clusterGpu.compactBound;
}

function bindCull(world: World): { pipeline: GPUComputePipeline; group: GPUBindGroup } {
    const _clusterGpu = world.resource(clusterGpuKey);
    const _lightCull = world.resource(LightCull);

    if (_clusterGpu.cullBound) return _clusterGpu.cullBound;
    if (!_clusterGpu.cullPipe || !_clusterGpu.typedAabbs || !_lightCull.lights)
        throw new Error("[render] light cull used before warmLightCull");
    // the light list binds RAW here and typed in the compact group: same buffer, two schemas (the
    // writer's count word is atomic, which WGSL forbids in a read-only binding — `PointLightsRw` vs
    // `PointLights`, layouts pinned equal in lighting.test.ts)
    _clusterGpu.cullBound = {
        pipeline: world.gpu.root.unwrap(_clusterGpu.cullPipe),
        group: world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(cullLayout, {
                aabbs: _clusterGpu.typedAabbs,
                lights: _lightCull.lights,
                viewMats: _lightCull.viewMats!,
            }),
        ),
    };
    return _clusterGpu.cullBound;
}

/** Request the latest submitted light-pool overflow count for diagnostics.
 * The cull pass clamps its writes independently of this request. */
export async function requestLightOverflow(world: World) {
    const indices = world.resource(clusterGpuKey).lightCull.lights;
    if (!indices) throw new Error("light overflow diagnostic requested before rendering warm");
    const result = await probeBuffer(world, indices, {
        offset: LIGHT_INDICES_OFFSET + 4,
        size: 4,
        label: "light-pool-overflow",
    });
    return {
        get frame() {
            return result.frame;
        },
        get fixedTick() {
            return result.fixedTick;
        },
        get dropped() {
            return new Uint32Array(result.bytes)[0];
        },
    };
}

/**
 * per-frame light compact + cull: builds the compacted list from active light-table rows, then bins it.
 * Runs after `UpdateLightClustersSystem` by registration order, before the renderers
 * (which sort after `BeginFrameSystem` in the same registration stream)
 */
export const CullLightsSystem: System = {
    group: "draw",
    after: [UpdateLightClustersSystem],
    update(world) {
        const _render = world.resource(RenderContext);
        const _clusterGpu = world.resource(clusterGpuKey);
        const _lightCull = world.resource(LightCull);

        if (
            !_render.encoder ||
            !_clusterGpu.compactPipe ||
            !_clusterGpu.cullPipe ||
            _render.shadeCount === 0
        )
            return;
        warnLightOverflow(world);

        world.gpu.device.queue.writeBuffer(
            _lightCull.viewMats!,
            0,
            _lightCull.viewStaging as Float32Array<ArrayBuffer>,
            0,
            _render.shadeCount * 16,
        );
        _render.encoder.clearBuffer(_lightCull.lights!, 0, 16);
        _render.encoder.clearBuffer(_lightCull.lights!, LIGHT_INDICES_OFFSET, POOL_HEADER * 4);
        _clusterGpu.cullPass.timestampWrites = world.gpu.span?.("light:cull");
        const lightCount = lightInputTable(world).count;
        if (lightCount !== _clusterGpu.lightCountValue) {
            lightCountData[0] = lightCount;
            world.gpu.device.queue.writeBuffer(_clusterGpu.lightCountBuffer!, 0, lightCountData);
            _clusterGpu.lightCountValue = lightCount;
        }
        const compact = lightCount > 0 ? bindCompact(world) : null;
        const cull = bindCull(world);
        const pass = _render.encoder.beginComputePass(_clusterGpu.cullPass);
        if (compact) {
            pass.setPipeline(compact.pipeline);
            pass.setBindGroup(0, compact.group);
            pass.dispatchWorkgroups(Math.ceil(lightCount / 64));
        }
        pass.setPipeline(cull.pipeline);
        pass.setBindGroup(0, cull.group);
        pass.dispatchWorkgroups(Math.ceil(CLUSTER_COUNT / 64), _render.shadeCount);
        pass.end();
    },
};

/** allocate the light-cull buffers + compile the compact and cull pipelines */
export function warmLightCull(world: World): void {
    const _clusterGpu = world.resource(clusterGpuKey);
    const _lightCull = world.resource(LightCull);

    if (!world.gpu.device) return;
    const device = world.gpu.device;
    const root = world.gpu.root;
    _clusterGpu.viewMatrices = Array.from({ length: MAX_VIEWS }, (_, slot) =>
        _lightCull.viewStaging.subarray(slot * 16, slot * 16 + 16),
    );
    _clusterGpu.compactBound = null;
    _clusterGpu.compactGeneration.fill(-1);
    _clusterGpu.cullBound = null;

    _lightCull.lights = device.createBuffer({
        label: "shallot-light-clusters",
        size: LIGHT_INDICES_OFFSET + (POOL_HEADER + LIGHT_POOL) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    // COPY_SRC throughout for requested readback (typegpu grants it on the
    // buffers it creates)
    _lightCull.viewMats = device.createBuffer({
        label: "shallot-light-views",
        size: MAX_VIEWS * 64,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _clusterGpu.lightCountBuffer = device.createBuffer({
        label: "shallot-light-count",
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _clusterGpu.lightCountValue = -1;
    world.gpu.buffers.set("lightClusters", _lightCull.lights);
    world.gpu.buffers.set("lightCount", _clusterGpu.lightCountBuffer);

    _clusterGpu.compactPipe = root
        .createComputePipeline({ compute: compactKernel() })
        .$name("shallot-light-compact");
    _clusterGpu.cullPipe = root
        .createComputePipeline({ compute: cullKernel })
        .$name("shallot-light-cull");
    precompile(world, "shallot-light-compact", () => [bindCompact(world).pipeline]);
    precompile(world, "shallot-light-cull", () => [bindCull(world).pipeline]);
}
