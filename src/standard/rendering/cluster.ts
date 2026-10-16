import tgpu, { type StorageFlag, type TgpuBuffer, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    BeginFrameSystem,
    Camera,
    CameraMode,
    computeViewFromWorld,
    globalTransformTable,
    INITIAL_VIEW_CAPACITY,
    MAX_VIEWS,
    PointLight,
    RenderContext,
    SpotLight,
    type View,
    Views,
    VolumetricLight,
    Xform,
    xformQuat,
} from "../../core/rendering";
import { ClearChangeMarksSystem, type System, type World } from "../../engine";
import {
    beginComputePass,
    beginRenderPass,
    dispatchWorkgroups,
    dispatchWorkgroups2,
    drawIndexedIndirect,
    endComputePass,
    endRenderPass,
    precompile,
    probeBuffer,
    setComputeBindGroup,
    setComputePipeline,
    setIndexBuffer,
    setRenderBindGroup,
    setRenderPipeline,
    setViewport,
} from "../../engine/runtime";
import { idiv, octEncodeNormal, srgbToLinear1 } from "../../engine/utils";
import {
    MAX_POINT_LIGHTS,
    PointLightGpu,
    PointLights,
    PointLightsRw,
    spotParams,
    warnLightOverflow,
} from "./lighting";

interface ClusterGpuState {
    clusters: Clusters;
    lightCull: LightCull;
    viewCapacity: number;
    pipe: TgpuComputePipeline | null;
    bound: { pipeline: GPUComputePipeline; group: GPUBindGroup } | null;
    typedViews: (TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag) | null;
    typedAabbs: (TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag) | null;
    compactPipe: TgpuComputePipeline | null;
    compactBound: { pipeline: GPUComputePipeline; group: GPUBindGroup } | null;
    compactGeneration: Int32Array;
    lightCountBuffer: GPUBuffer | null;
    lightCountValue: number;
    rasterLights: GPUBuffer | null;
    zSlicePipe: GPUComputePipeline | null;
    zSliceBound: { pipeline: GPUComputePipeline; group: GPUBindGroup } | null;
    allocationLocalPipe: GPUComputePipeline | null;
    allocationGlobalPipe: GPUComputePipeline | null;
    allocationBound: GPUBindGroup | null;
    allocationSlots: number;
    countPipe: GPURenderPipeline | null;
    countBound: GPUBindGroup | null;
    populatePipe: GPURenderPipeline | null;
    populateBound: GPUBindGroup | null;
    zSliceLayout: GPUBindGroupLayout | null;
    allocationLayout: GPUBindGroupLayout | null;
    countLayout: GPUBindGroupLayout | null;
    populateLayout: GPUBindGroupLayout | null;
    clusterCounts: GPUBuffer | null;
    zSlices: GPUBuffer | null;
    rasterArgs: GPUBuffer | null;
    rasterTexture: GPUTexture | null;
    rasterTextureView: GPUTextureView | null;
    rasterIndexBuffer: GPUBuffer | null;
    clusterView: ClusterView;
    viewMatrices: Float32Array[];
    gridPass: GPUComputePassDescriptor;
    compactPass: GPUComputePassDescriptor;
    zSlicePass: GPUComputePassDescriptor;
    allocationLocalPass: GPUComputePassDescriptor;
    allocationGlobalPass: GPUComputePassDescriptor;
    countPass: GPURenderPassDescriptor;
    populatePass: GPURenderPassDescriptor;
}

export const clusterGpuKey = { create: createClusterGpuState };

function createClusterGpuState(): ClusterGpuState {
    return {
        clusters: {
            aabbs: null,
            views: null,
            staging: new Float32Array(INITIAL_VIEW_CAPACITY * CLUSTER_VIEW_FLOATS),
            last: new Float32Array(INITIAL_VIEW_CAPACITY * CLUSTER_VIEW_FLOATS),
            pending: new Float32Array(INITIAL_VIEW_CAPACITY * CLUSTER_VIEW_FLOATS),
            pendingUsed: 0,
        },
        lightCull: {
            lights: null,
            viewMats: null,
            viewStaging: new Float32Array(INITIAL_VIEW_CAPACITY * 16),
        },
        pipe: null,
        bound: null,
        typedViews: null,
        typedAabbs: null,
        viewCapacity: INITIAL_VIEW_CAPACITY,
        compactPipe: null,
        compactBound: null,
        compactGeneration: new Int32Array(4).fill(-1),
        lightCountBuffer: null,
        lightCountValue: -1,
        rasterLights: null,
        zSlicePipe: null,
        zSliceBound: null,
        allocationLocalPipe: null,
        allocationGlobalPipe: null,
        allocationBound: null,
        allocationSlots: -1,
        countPipe: null,
        countBound: null,
        populatePipe: null,
        populateBound: null,
        zSliceLayout: null,
        allocationLayout: null,
        countLayout: null,
        populateLayout: null,
        clusterCounts: null,
        zSlices: null,
        rasterArgs: null,
        rasterTexture: null,
        rasterTextureView: null,
        rasterIndexBuffer: null,
        clusterView: { perspective: false, halfW: 0, halfH: 0, near: 0, far: 0 },
        viewMatrices: [],
        gridPass: { label: "shallot-cluster-aabbs" },
        compactPass: { label: "shallot-light-compact" },
        zSlicePass: { label: "shallot-light-z-slice" },
        allocationLocalPass: { label: "shallot-light-allocation-local" },
        allocationGlobalPass: { label: "shallot-light-allocation-global" },
        countPass: { label: "shallot-light-count", colorAttachments: [] },
        populatePass: { label: "shallot-light-populate", colorAttachments: [] },
    };
}

function _clusterGpu(world: World): ClusterGpuState {
    return world.resource(clusterGpuKey);
}

/** Authoring input rows preserve lumens; the compact pass divides by 4π to store linear-RGB candela.
 * Spot rows carry {@link spotParams} from the CPU, written by {@link PackSpotAttenuationSystem}. */
const LightInput = d
    .struct({
        color: d.f32,
        intensity: d.f32,
        range: d.f32,
        radius: d.f32,
        spotScale: d.f32,
        spotOffset: d.f32,
        flags: d.u32,
        padding: d.u32,
    })
    .$name("LightInput");
const LIGHT_SPOT = 1;
const LIGHT_VOLUMETRIC = 2;
const INV_FOUR_PI = 1 / (4 * Math.PI);
export const lightInputKey = { create: createLightInputTable };
const lightCountData = new Uint32Array(1);

function createLightInputTable(world: World) {
    return world.table("lightInputs", LightInput);
}

function lightInputTable(world: World) {
    return world.resource(lightInputKey);
}

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
    });
    table.bindMembership(SpotLight);
    table.bindPresence(SpotLight, "flags", LIGHT_SPOT);
    table.bindPresence(VolumetricLight, "flags", LIGHT_VOLUMETRIC);
}

const SPOT_SCALE_OFFSET = d.memoryLayoutOf(LightInput, (row) => row.spotScale).offset;
const SPOT_LIGHTS = [SpotLight];

/**
 * Spot coefficients come from {@link spotParams} on the CPU, as in Bevy, before the draw-head upload. WGSL
 * bounds f32 `cos` only to an absolute 2⁻¹¹, and the scale multiplies that error by up to 1e4; f32 also loses
 * a narrow cone's `cos inner − cos outer` to cancellation.
 */
export const PackSpotAttenuationSystem: System = {
    group: "draw",
    first: true,
    before: [ClearChangeMarksSystem],
    update(world) {
        const table = lightInputTable(world);
        const spot = world.storage(SpotLight);
        const view = new DataView(table.bytes.buffer);
        for (const eid of world.query(SPOT_LIGHTS)) {
            const row = table.rowIndex(eid);
            if (row < 0) continue;
            const { scale, offset } = spotParams(
                spot.innerAngle.get(eid),
                spot.outerAngle.get(eid),
            );
            const at = row * table.rowBytes + SPOT_SCALE_OFFSET;
            if (
                Object.is(view.getFloat32(at, true), Math.fround(scale)) &&
                Object.is(view.getFloat32(at + 4, true), Math.fround(offset))
            )
                continue;
            view.setFloat32(at, scale, true);
            view.setFloat32(at + 4, offset, true);
            table.markRange(row, 1);
        }
    },
};

// Logarithmic depth slices distribute near-camera lights across clusters.
// View-space AABBs depend only on projection, so camera motion alone needs no rebuild.

/** cluster grid: horizontal screen-space tiles */
export const CLUSTER_X = 16;
/** cluster grid: vertical screen-space tiles */
export const CLUSTER_Y = 9;
/** cluster grid: logarithmic depth slices (DOOM 2016 / Olsson log-Z) */
export const CLUSTER_Z = 24;
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
 * Slot-major froxel index for normalized pixel coordinates [0,1] (y down) and positive view depth.
 * Tile (0,0) is NDC (-1,-1), so screen y is flipped.
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
 * View-space cluster AABB: camera looks down -Z, min.z is the far boundary; tile (0,0) starts at NDC (-1,-1).
 * Perspective tile corners widen with depth, so bounds must include both slice boundaries.
 * Keep this CPU calculation aligned with gridKernel's GPU calculation.
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
 * Clusters touched by the light's influence sphere; center and range are in view space.
 * Uses the sphere-to-AABB distance predicate shared with GPU light culling.
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
    /** Projection values encoded in the current draw, committed to `last` after submission. */
    pending: Float32Array;
    pendingUsed: number;
    /** Projection values last acknowledged after a successful frame submission. */
    last: Float32Array;
    /** Stable callback for the current draw's encoded grid rebuild. */
    commitOnSubmit?: () => void;
}

export const Clusters: import("../../engine").Resource<Clusters> = {
    create: (world) => {
        const clusters = world.resource(clusterGpuKey).clusters;
        clusters.commitOnSubmit = () => {
            if (!clusters.pendingUsed) return;
            clusters.last.set(clusters.pending.subarray(0, clusters.pendingUsed));
            clusters.pendingUsed = 0;
        };
        return clusters;
    },
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

// TGSL's `/` divides as floats; idiv is required for integral grid coordinates.
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

function growClusterCapacity(world: World, shadeCount: number): void {
    const state = world.resource(clusterGpuKey);
    if (shadeCount <= state.viewCapacity) return;
    // Geometric growth bounds reallocations when cameras arrive one at a time.
    const capacity = Math.min(MAX_VIEWS, Math.max(shadeCount, state.viewCapacity * 2));
    const clusters = world.resource(Clusters);
    const lightCull = world.resource(LightCull);
    const previousBuffers = [
        clusters.views,
        clusters.aabbs,
        lightCull.lights,
        lightCull.viewMats,
        state.clusterCounts,
        state.zSlices,
        state.rasterArgs,
    ].filter((buffer): buffer is GPUBuffer => buffer !== null);
    const nextFloats = (source: Float32Array, length: number, newFill = 0): Float32Array => {
        const target = new Float32Array(length);
        target.set(source);
        if (newFill !== 0) target.fill(newFill, source.length);
        return target;
    };
    clusters.staging = nextFloats(clusters.staging, capacity * CLUSTER_VIEW_FLOATS);
    clusters.last = nextFloats(clusters.last, capacity * CLUSTER_VIEW_FLOATS, Number.NaN);
    clusters.pending = nextFloats(clusters.pending, capacity * CLUSTER_VIEW_FLOATS);
    lightCull.viewStaging = nextFloats(lightCull.viewStaging, capacity * 16);
    state.viewCapacity = capacity;
    allocateGridBuffers(world, capacity);
    allocatePerViewLightBuffers(world, capacity);
    state.bound = null;
    bindLightCullGroups(world);
    for (const buffer of previousBuffers) buffer.destroy();
    state.lightCountValue = -1;
}

/**
 * Rebuild the AABB table when an active projection changes. Its runtime-sized storage buffers grow with the
 * presenting prefix; the pipeline and layout remain fixed. The staging prefix is the dirty signal, so a stable
 * projection dispatches nothing on steady frames.
 */
function updateClusterView(this: World, view: View, eid: number): void {
    const render = this.resource(RenderContext);
    if (!view.framebuffer || view.slot >= render.shadeCount) return;
    packClusterView(this, eid, view.width / view.height, view.slot);
    const state = this.resource(clusterGpuKey);
    const viewFromWorld = view.viewFromWorld;
    if (viewFromWorld) state.viewMatrices[view.slot]!.set(viewFromWorld);
    else computeViewFromWorld(this, eid, state.viewMatrices[view.slot]!);
}

export const UpdateLightClustersSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    update(world: World) {
        const _render = world.resource(RenderContext);
        const _clusterGpu = world.resource(clusterGpuKey);
        const _clusters = world.resource(Clusters);

        _clusters.pendingUsed = 0;
        if (_render.shadeCount === 0 || !_clusterGpu.pipe) return;
        growClusterCapacity(world, _render.shadeCount);
        world.resource(Views).forEach(updateClusterView, world);
        const used = _render.shadeCount * CLUSTER_VIEW_FLOATS;
        let changed = false;
        for (let i = 0; i < used; i++) {
            if (_clusters.staging[i] !== _clusters.last[i]) {
                changed = true;
                break;
            }
        }
        if (!changed) return;
        world.gpu.device.queue.writeBuffer(
            _clusters.views!,
            0,
            _clusters.staging as Float32Array<ArrayBuffer>,
            0,
            used,
        );
        _clusterGpu.gridPass.timestampWrites = world.gpu.span?.("cluster:aabbs");
        const grid = bindGrid(world);
        const pass = beginComputePass(world.frameEncoder()!, _clusterGpu.gridPass);
        setComputePipeline(pass, grid.pipeline);
        setComputeBindGroup(pass, 0, grid.group);
        dispatchWorkgroups2(pass, Math.ceil(CLUSTER_COUNT / 64), _render.shadeCount);
        endComputePass(pass);
        _clusters.pending.set(_clusters.staging.subarray(0, used));
        _clusters.pendingUsed = used;
        world.onFrameSubmit(_clusters.commitOnSubmit!);
    },
};

// Pass descriptors are part of the world's mutable dispatch state.

// Precompile runs after all warm hooks; missing cluster inputs at that point are an error.
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

function ownedBuffer(
    world: World,
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
): GPUBuffer {
    const buffer = world.gpu.device.createBuffer({ label, size, usage });
    world.own(buffer);
    return buffer;
}

function scratchCount(capacity: number): number {
    return (
        Math.ceil((capacity * CLUSTER_COUNT) / ALLOCATION_WORKGROUP_SIZE) *
        ALLOCATION_WORKGROUP_SIZE
    );
}

function allocateGridBuffers(world: World, capacity: number): void {
    const clusters = world.resource(Clusters);
    const state = world.resource(clusterGpuKey);
    const root = world.gpu.root;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const views = ownedBuffer(
        world,
        "shallot-cluster-views",
        capacity * CLUSTER_VIEW_FLOATS * 4,
        usage,
    );
    const aabbs = ownedBuffer(world, "shallot-cluster-aabbs", capacity * CLUSTER_COUNT * 32, usage);
    state.typedViews = root.createBuffer(d.arrayOf(d.vec4f, 0), views).$usage("storage");
    state.typedAabbs = root.createBuffer(d.arrayOf(d.vec4f, 0), aabbs).$usage("storage");
    clusters.views = views;
    clusters.aabbs = aabbs;
    world.gpu.buffers.set("clusterAabbs", aabbs);
    world.gpu.typed.set("clusterAabbs", state.typedAabbs);
}

/** Allocate one initial cluster slot and compile the count-independent AABB pipeline. */
export function warmClusters(world: World): void {
    const _clusters = world.resource(Clusters);
    const _clusterGpu = world.resource(clusterGpuKey);

    if (!world.gpu.device) return;
    const root = world.gpu.root;
    _clusters.last.fill(0);
    _clusters.pendingUsed = 0;
    _clusterGpu.viewCapacity = INITIAL_VIEW_CAPACITY;
    _clusterGpu.bound = null;
    allocateGridBuffers(world, INITIAL_VIEW_CAPACITY);

    _clusterGpu.pipe = root
        .createComputePipeline({ compute: gridKernel })
        .$name("shallot-cluster-aabbs");
    // the bind is deferred into the forcer: it runs after every plugin's warm
    // has resolved (warm hooks run under `Promise.all`), the first moment every input buffer is up
    precompile(world, "shallot-cluster-aabbs", () => {
        return [bindGrid(world).pipeline];
    });
}

/** per-cluster light index pool: 32 × CLUSTER_COUNT entries shared across views */
export const LIGHT_POOL = CLUSTER_COUNT * 32;

// pool header: [0] allocated entries, [1] overflow (memberships that didn't fit).
// Data entries start at element 2; grid offsets index this binding directly.
const POOL_HEADER = 2;

/** GPU-written lights, fixed index pool and runtime-sized per-view grid in one storage binding. */
export const LightClusters = d.struct({
    lights: PointLights,
    indices: d.arrayOf(d.u32, LIGHT_POOL + POOL_HEADER),
    grid: d.arrayOf(d.vec2u, 0),
});
const LightClustersRw = d.struct({
    lights: PointLightsRw,
    indices: d.arrayOf(d.atomic(d.u32), LIGHT_POOL + POOL_HEADER),
    grid: d.arrayOf(d.vec2u, 0),
});
export const LIGHT_INDICES_OFFSET = d.sizeOf(PointLights);
export const LIGHT_GRID_OFFSET = LIGHT_INDICES_OFFSET + (LIGHT_POOL + POOL_HEADER) * 4;

/**
 * GPU light state. `lights` holds {@link LightClusters}: compacted lights, the slot-major grid and the fixed
 * index pool. `viewMats` is the per-slot world→view matrix staged by `BeginFrameSystem`.
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
        rasterLights: { storage: d.arrayOf(PointLightGpu), access: "mutable" },
    })
    .$idx(0);

const Z_SLICE_CAPACITY = MAX_POINT_LIGHTS * CLUSTER_Z;
const ALLOCATION_WORKGROUP_SIZE = 256;
const RASTER_ARGS_SIZE = 20;

const zSliceShader = `
struct PointLight { posRange: vec4f, color: vec4f, params: vec4f };
struct ClusterOutput {
    count: array<atomic<u32>, 4>,
    lights: array<PointLight, ${MAX_POINT_LIGHTS}>,
    indices: array<atomic<u32>, ${POOL_HEADER + LIGHT_POOL}>,
    grid: array<vec2u>,
};
struct RasterArgs {
    indexCount: u32,
    instanceCount: atomic<u32>,
    firstIndex: u32,
    baseVertex: i32,
    firstInstance: u32,
};
struct ZSlice { light: u32, z: u32 };
@group(0) @binding(0) var<storage, read_write> cluster: ClusterOutput;
@group(0) @binding(1) var<storage, read> rasterLights: array<PointLight, ${MAX_POINT_LIGHTS}>;
@group(0) @binding(2) var<storage, read> viewMats: array<mat4x4f>;
@group(0) @binding(3) var<storage, read> clusterViews: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> rasterArgs: array<RasterArgs>;
@group(0) @binding(5) var<storage, read_write> zSlices: array<ZSlice>;

fn viewScale(view: mat4x4f) -> f32 {
    let x = length(vec3f(view[0].x, view[1].x, view[2].x));
    let y = length(vec3f(view[0].y, view[1].y, view[2].y));
    let z = length(vec3f(view[0].z, view[1].z, view[2].z));
    return max(x, max(y, z));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let lightCount = min(atomicLoad(&cluster.count[0]), ${MAX_POINT_LIGHTS}u);
    if (gid.x >= lightCount || gid.y >= arrayLength(&viewMats)) { return; }
    let light = rasterLights[gid.x];
    let center = (viewMats[gid.y] * vec4f(light.posRange.xyz, 1.0)).xyz;
    let inverseRangeSquared = light.posRange.w;
    let radius = select(1e20, inverseSqrt(inverseRangeSquared), inverseRangeSquared > 0.0) * viewScale(viewMats[gid.y]);
    let view = clusterViews[gid.y * 2u];
    let nearest = -center.z - radius;
    let farthest = -center.z + radius;
    if (farthest < view.z || nearest > view.w) { return; }
    let nearDepth = max(nearest, view.z);
    let farDepth = min(farthest, view.w);
    let scale = log(view.w / view.z);
    // Include the preceding slice when the sphere is tangent to a boundary. The fine sphere/AABB test
    // removes the harmless extra candidates; rounding down alone drops the touching froxel.
    let firstZ = u32(clamp(floor(log(nearDepth / view.z) / scale * ${CLUSTER_Z}.0) - 1.0, 0.0, ${CLUSTER_Z - 1}.0));
    let lastZ = u32(clamp(floor(log(farDepth / view.z) / scale * ${CLUSTER_Z}.0), 0.0, ${CLUSTER_Z - 1}.0));
    for (var z = firstZ; z <= lastZ; z += 1u) {
        let at = gid.y * ${Z_SLICE_CAPACITY}u + atomicAdd(&rasterArgs[gid.y].instanceCount, 1u);
        zSlices[at] = ZSlice(gid.x, z);
    }
}
`;

const allocationShader = `
struct ClusterOutput {
    count: array<atomic<u32>, 4>,
    lights: array<vec4f, ${MAX_POINT_LIGHTS * 3}>,
    indices: array<atomic<u32>, ${POOL_HEADER + LIGHT_POOL}>,
    grid: array<vec2u>,
};
struct ClusterCounter { count: atomic<u32>, cursor: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> counters: array<ClusterCounter>;
@group(0) @binding(1) var<storage, read_write> cluster: ClusterOutput;
var<workgroup> scan: array<u32, ${ALLOCATION_WORKGROUP_SIZE}>;
var<workgroup> blockSize: u32;
var<workgroup> blockBase: u32;

@compute @workgroup_size(${ALLOCATION_WORKGROUP_SIZE})
fn allocateLocal(
    @builtin(global_invocation_id) global: vec3u,
    @builtin(local_invocation_id) local: vec3u,
) {
    let gridCount = arrayLength(&cluster.grid);
    let count = atomicLoad(&counters[global.x].count);
    scan[local.x] = select(0u, count, global.x < gridCount);
    workgroupBarrier();
    for (var offset = 1u; offset < ${ALLOCATION_WORKGROUP_SIZE}u; offset *= 2u) {
        var term = 0u;
        if (local.x >= offset) { term = scan[local.x - offset]; }
        workgroupBarrier();
        scan[local.x] += term;
        workgroupBarrier();
    }
    if (global.x < gridCount) {
        cluster.grid[global.x] = vec2u(scan[local.x] - count, count);
    }
    atomicStore(&counters[global.x].cursor, 0u);
}

@compute @workgroup_size(${ALLOCATION_WORKGROUP_SIZE})
fn allocateGlobal(@builtin(local_invocation_id) local: vec3u) {
    if (local.x == 0u) { blockBase = 0u; }
    workgroupBarrier();
    let gridCount = arrayLength(&cluster.grid);
    let counterCount = arrayLength(&counters);
    for (var start = 0u; start < counterCount; start += ${ALLOCATION_WORKGROUP_SIZE}u) {
        if (local.x == 0u) {
            let tail = min(start + ${ALLOCATION_WORKGROUP_SIZE - 1}u, gridCount - 1u);
            blockSize = cluster.grid[tail].x + cluster.grid[tail].y;
        }
        workgroupBarrier();
        let index = start + local.x;
        if (index < gridCount) {
            let raw = cluster.grid[index];
            let offset = blockBase + raw.x;
            let boundedOffset = min(offset, ${LIGHT_POOL}u);
            cluster.grid[index] = vec2u(
                ${POOL_HEADER}u + boundedOffset,
                min(raw.y, ${LIGHT_POOL}u - boundedOffset),
            );
        }
        storageBarrier();
        if (local.x == 0u) {
            blockBase += blockSize;
            if (start + ${ALLOCATION_WORKGROUP_SIZE}u >= counterCount) {
                atomicStore(&cluster.indices[0], min(blockBase, ${LIGHT_POOL}u));
                atomicStore(&cluster.indices[1], blockBase - min(blockBase, ${LIGHT_POOL}u));
            }
        }
        workgroupBarrier();
    }
}
`;

const rasterCommon = `
struct PointLight { posRange: vec4f, color: vec4f, params: vec4f };
struct ZSlice { light: u32, z: u32 };
struct ClusterCounter { count: atomic<u32>, cursor: atomic<u32> };
struct ClusterOutput {
    count: array<atomic<u32>, 4>,
    lights: array<PointLight, ${MAX_POINT_LIGHTS}>,
    indices: array<atomic<u32>, ${POOL_HEADER + LIGHT_POOL}>,
    grid: array<vec2u>,
};
struct Varyings {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) light: u32,
    @location(1) @interpolate(flat) slot: u32,
    @location(2) @interpolate(flat) z: u32,
    @location(3) @interpolate(flat) center: vec3f,
    @location(4) @interpolate(flat) inverseRangeSquared: f32,
};
@group(0) @binding(0) var<storage, read> zSlices: array<ZSlice>;
@group(0) @binding(1) var<storage, read> rasterLights: array<PointLight, ${MAX_POINT_LIGHTS}>;
@group(0) @binding(2) var<storage, read> viewMats: array<mat4x4f>;
@group(0) @binding(3) var<storage, read> clusterViews: array<vec4f>;

fn viewScale(view: mat4x4f) -> f32 {
    let x = length(vec3f(view[0].x, view[1].x, view[2].x));
    let y = length(vec3f(view[0].y, view[1].y, view[2].y));
    let z = length(vec3f(view[0].z, view[1].z, view[2].z));
    return max(x, max(y, z));
}

fn projectSphereCorner(point: vec3f, view: vec4f, perspective: bool) -> vec2f {
    if (perspective) { return point.xy / (-point.z * view.xy); }
    return point.xy / view.xy;
}

fn ndcCluster(position: f32, dimension: u32) -> u32 {
    return min(u32(floor(clamp((position + 1.0) * 0.5, 0.0, 1.0) * f32(dimension))), dimension - 1u);
}

fn sliceBoundary(view: vec4f, z: u32) -> f32 {
    return view.z * pow(view.w / view.z, f32(z) / ${CLUSTER_Z}.0);
}

// A cluster AABB spans its whole log slice, so project the sphere's XY bounds at the slice ends too.
// Using only the sphere's own Z extent misses AABB overlaps where a far slice widens toward its far plane.
fn rasterBounds(center: vec3f, radius: f32, view: vec4f, perspective: bool, slice: u32) -> vec4u {
    let viewMin = center - vec3f(radius);
    let viewMax = center + vec3f(radius);
    let sliceNear = sliceBoundary(view, slice);
    let sliceFar = sliceBoundary(view, slice + 1u);
    let zMin = min(viewMin.z, -sliceFar);
    let zMax = max(viewMax.z, -sliceNear);
    let nearZ = min(zMin, -1e-5);
    let farZ = min(zMax, -1e-5);
    let xyMin = viewMin.xy;
    let xyMax = viewMax.xy;
    let a = projectSphereCorner(vec3f(xyMin, nearZ), view, perspective);
    let b = projectSphereCorner(vec3f(xyMin, farZ), view, perspective);
    let c = projectSphereCorner(vec3f(xyMax, nearZ), view, perspective);
    let d = projectSphereCorner(vec3f(xyMax, farZ), view, perspective);
    let ndcMin = clamp(min(min(a, b), min(c, d)), vec2f(-1.0), vec2f(1.0));
    let ndcMax = clamp(max(max(a, b), max(c, d)), vec2f(-1.0), vec2f(1.0));
    let minX = ndcCluster(ndcMin.x, ${CLUSTER_X}u);
    let maxX = ndcCluster(ndcMax.x, ${CLUSTER_X}u);
    let minY = ndcCluster(ndcMin.y, ${CLUSTER_Y}u);
    let maxY = ndcCluster(ndcMax.y, ${CLUSTER_Y}u);
    let top = ${CLUSTER_Y}u - maxY - 1u;
    let bottom = ${CLUSTER_Y}u - minY;
    return vec4u(
        select(minX, minX - 1u, minX > 0u),
        select(top, top - 1u, top > 0u),
        min(maxX + 2u, ${CLUSTER_X}u),
        min(bottom + 1u, ${CLUSTER_Y}u),
    );
}

@vertex
fn vertexMain(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
) -> Varyings {
    let slot = instance / ${Z_SLICE_CAPACITY}u;
    let slice = zSlices[instance];
    let light = rasterLights[slice.light];
    let center = (viewMats[slot] * vec4f(light.posRange.xyz, 1.0)).xyz;
    let worldInverseRangeSquared = light.posRange.w;
    let radius = select(1e20, inverseSqrt(worldInverseRangeSquared), worldInverseRangeSquared > 0.0) * viewScale(viewMats[slot]);
    let inverseRangeSquared = select(0.0, 1.0 / (radius * radius), worldInverseRangeSquared > 0.0);
    let view = clusterViews[slot * 2u];
    let perspective = clusterViews[slot * 2u + 1u].x > 0.5;
    let bounds = rasterBounds(center, radius, view, perspective, slice.z);
    let right = vertex == 1u || vertex == 3u;
    let bottom = vertex >= 2u;
    let x = select(bounds.x, bounds.z, right);
    let y = select(bounds.y, bounds.w, bottom);
    let position = vec2f(
        f32(x) * (2.0 / ${CLUSTER_X}.0) - 1.0,
        1.0 - f32(y) * (2.0 / ${CLUSTER_Y}.0),
    );
    return Varyings(
        vec4f(position, 0.0, 1.0),
        slice.light,
        slot,
        slice.z,
        center,
        inverseRangeSquared,
    );
}

fn fragmentCluster(input: Varyings) -> u32 {
    let x = min(u32(input.position.x), ${CLUSTER_X - 1}u);
    let yTop = min(u32(input.position.y), ${CLUSTER_Y - 1}u);
    let y = ${CLUSTER_Y - 1}u - yTop;
    return (input.slot * ${CLUSTER_COUNT}u) + ((y * ${CLUSTER_X}u + x) * ${CLUSTER_Z}u) + input.z;
}

fn intersects(clusterIndex: u32, center: vec3f, inverseRangeSquared: f32) -> bool {
    let base = (clusterIndex * 2u);
    let mn = clusterAabbs[base].xyz;
    let mx = clusterAabbs[base + 1u].xyz;
    let nearest = clamp(center, mn, mx);
    let delta = nearest - center;
    return dot(delta, delta) * inverseRangeSquared <= 1.0;
}
`;

const countRasterShader = `${rasterCommon}
@group(0) @binding(4) var<storage, read> clusterAabbs: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> counters: array<ClusterCounter>;
@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4f {
    let index = fragmentCluster(input);
    if (intersects(index, input.center, input.inverseRangeSquared)) {
        atomicAdd(&counters[index].count, 1u);
    }
    return vec4f(0.0);
}
`;

const populateRasterShader = `${rasterCommon}
@group(0) @binding(4) var<storage, read> clusterAabbs: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> counters: array<ClusterCounter>;
@group(0) @binding(6) var<storage, read_write> output: ClusterOutput;
@fragment
fn fragmentMain(input: Varyings) -> @location(0) vec4f {
    let index = fragmentCluster(input);
    if (intersects(index, input.center, input.inverseRangeSquared)) {
        let local = atomicAdd(&counters[index].cursor, 1u);
        let list = output.grid[index];
        if (local < list.y) {
            atomicStore(&output.indices[list.x + local], input.light);
        }
    }
    return vec4f(0.0);
}
`;

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
                record.intensity * INV_FOUR_PI,
            );
            const pos = globalTransform.pos;
            const posRange = d.vec4f(pos.x, pos.y, pos.z, 1 / (record.range * record.range));
            // color.a carries the source entity id for per-entity light extensions.
            const color = d.vec4f(rgb.x, rgb.y, rgb.z, d.f32(eid));
            compactLayout.$.lights.lights.lights[i].posRange = d.vec4f(posRange);
            compactLayout.$.lights.lights.lights[i].color = d.vec4f(color);

            let radius = record.radius;
            if ((record.flags & LIGHT_VOLUMETRIC) !== 0) radius = -std.max(radius, 1e-4);
            let params = d.vec4f(radius, 0, 0, 1);
            if ((record.flags & LIGHT_SPOT) !== 0) {
                const dir = std.normalize(xformQuat(globalTransform.quat, d.vec3f(0, 0, -1)));
                params = d.vec4f(
                    radius,
                    std.bitcastU32toF32(octEncodeNormal(dir)),
                    record.spotScale,
                    record.spotOffset,
                );
            }
            compactLayout.$.lights.lights.lights[i].params = d.vec4f(params);
            compactLayout.$.rasterLights[i].posRange = d.vec4f(posRange);
            compactLayout.$.rasterLights[i].color = d.vec4f(color);
            compactLayout.$.rasterLights[i].params = d.vec4f(params);
        })
        .$name("lightCompact");
}

// Keep bind groups until a table buffer generation changes; row membership alone never rebuilds one.
function bindCompact(world: World): { pipeline: GPUComputePipeline; group: GPUBindGroup } {
    const _clusterGpu = world.resource(clusterGpuKey);

    const buffer = world.resource(LightCull).lights;
    if (
        !_clusterGpu.compactPipe ||
        !buffer ||
        !_clusterGpu.lightCountBuffer ||
        !_clusterGpu.rasterLights
    )
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
        rasterLights: _clusterGpu.rasterLights,
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

async function warmLightRaster(world: World): Promise<void> {
    const _clusterGpu = world.resource(clusterGpuKey);
    const _lightCull = world.resource(LightCull);
    const _clusters = world.resource(Clusters);
    const device = world.gpu.device;
    const lights = _lightCull.lights;
    const rasterLights = _clusterGpu.rasterLights;
    const viewMats = _lightCull.viewMats;
    const aabbs = _clusterGpu.typedAabbs && world.gpu.root.unwrap(_clusterGpu.typedAabbs);
    const zSlices = _clusterGpu.zSlices;
    const rasterArgs = _clusterGpu.rasterArgs;
    const clusterCounts = _clusterGpu.clusterCounts;
    if (
        !lights ||
        !rasterLights ||
        !viewMats ||
        !aabbs ||
        !zSlices ||
        !rasterArgs ||
        !clusterCounts
    )
        throw new Error("[render] light raster inputs missing during warmLightCull");

    const storage = (
        binding: number,
        visibility: GPUShaderStageFlags,
        type: "storage" | "read-only-storage",
    ): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type } });
    const computeLayout = (entries: GPUBindGroupLayoutEntry[]) =>
        device.createBindGroupLayout({ entries });
    const zSliceLayout = computeLayout([
        storage(0, GPUShaderStage.COMPUTE, "storage"),
        storage(1, GPUShaderStage.COMPUTE, "read-only-storage"),
        storage(2, GPUShaderStage.COMPUTE, "read-only-storage"),
        storage(3, GPUShaderStage.COMPUTE, "read-only-storage"),
        storage(4, GPUShaderStage.COMPUTE, "storage"),
        storage(5, GPUShaderStage.COMPUTE, "storage"),
    ]);
    const allocationLayout = computeLayout([
        storage(0, GPUShaderStage.COMPUTE, "storage"),
        storage(1, GPUShaderStage.COMPUTE, "storage"),
    ]);
    const countLayout = computeLayout([
        storage(0, GPUShaderStage.VERTEX, "read-only-storage"),
        storage(1, GPUShaderStage.VERTEX, "read-only-storage"),
        storage(2, GPUShaderStage.VERTEX, "read-only-storage"),
        storage(3, GPUShaderStage.VERTEX, "read-only-storage"),
        storage(4, GPUShaderStage.FRAGMENT, "read-only-storage"),
        storage(5, GPUShaderStage.FRAGMENT, "storage"),
    ]);
    const populateLayout = computeLayout([
        storage(0, GPUShaderStage.VERTEX, "read-only-storage"),
        storage(1, GPUShaderStage.VERTEX, "read-only-storage"),
        storage(2, GPUShaderStage.VERTEX, "read-only-storage"),
        storage(3, GPUShaderStage.VERTEX, "read-only-storage"),
        storage(4, GPUShaderStage.FRAGMENT, "read-only-storage"),
        storage(5, GPUShaderStage.FRAGMENT, "storage"),
        storage(6, GPUShaderStage.FRAGMENT, "storage"),
    ]);
    const zSliceModule = device.createShaderModule({
        label: "shallot-light-z-slice",
        code: zSliceShader,
    });
    const allocationModule = device.createShaderModule({
        label: "shallot-light-allocation",
        code: allocationShader,
    });
    const countModule = device.createShaderModule({
        label: "shallot-light-count",
        code: countRasterShader,
    });
    const populateModule = device.createShaderModule({
        label: "shallot-light-populate",
        code: populateRasterShader,
    });
    const zSlicePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [zSliceLayout] });
    const allocationPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [allocationLayout],
    });
    const countPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [countLayout] });
    const populatePipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [populateLayout],
    });

    const [zSlicePipe, allocationLocalPipe, allocationGlobalPipe, countPipe, populatePipe] =
        await Promise.all([
            device.createComputePipelineAsync({
                label: "shallot-light-z-slice",
                layout: zSlicePipelineLayout,
                compute: { module: zSliceModule, entryPoint: "main" },
            }),
            device.createComputePipelineAsync({
                label: "shallot-light-allocation-local",
                layout: allocationPipelineLayout,
                compute: { module: allocationModule, entryPoint: "allocateLocal" },
            }),
            device.createComputePipelineAsync({
                label: "shallot-light-allocation-global",
                layout: allocationPipelineLayout,
                compute: { module: allocationModule, entryPoint: "allocateGlobal" },
            }),
            device.createRenderPipelineAsync({
                label: "shallot-light-count",
                layout: countPipelineLayout,
                vertex: { module: countModule, entryPoint: "vertexMain" },
                primitive: { topology: "triangle-list" },
                fragment: {
                    module: countModule,
                    entryPoint: "fragmentMain",
                    targets: [{ format: "r8unorm", writeMask: 0 }],
                },
            }),
            device.createRenderPipelineAsync({
                label: "shallot-light-populate",
                layout: populatePipelineLayout,
                vertex: { module: populateModule, entryPoint: "vertexMain" },
                primitive: { topology: "triangle-list" },
                fragment: {
                    module: populateModule,
                    entryPoint: "fragmentMain",
                    targets: [{ format: "r8unorm", writeMask: 0 }],
                },
            }),
        ]);
    _clusterGpu.zSlicePipe = zSlicePipe;
    _clusterGpu.allocationLocalPipe = allocationLocalPipe;
    _clusterGpu.allocationGlobalPipe = allocationGlobalPipe;
    _clusterGpu.countPipe = countPipe;
    _clusterGpu.populatePipe = populatePipe;
    _clusterGpu.zSliceLayout = zSliceLayout;
    _clusterGpu.allocationLayout = allocationLayout;
    _clusterGpu.countLayout = countLayout;
    _clusterGpu.populateLayout = populateLayout;
    bindLightCullGroups(world);

    _clusterGpu.rasterTexture = device.createTexture({
        label: "shallot-light-raster-target",
        size: [CLUSTER_X, CLUSTER_Y],
        format: "r8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    world.own(_clusterGpu.rasterTexture);
    _clusterGpu.rasterTextureView = _clusterGpu.rasterTexture.createView();
    const attachment: GPURenderPassColorAttachment = {
        view: _clusterGpu.rasterTextureView,
        loadOp: "clear",
        storeOp: "discard",
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
    };
    _clusterGpu.countPass.colorAttachments.push(attachment);
    _clusterGpu.populatePass.colorAttachments.push(attachment);

    const indices = device.createBuffer({
        label: "shallot-light-raster-indices",
        size: 6 * Uint16Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    world.own(indices);
    device.queue.writeBuffer(indices, 0, new Uint16Array([0, 1, 2, 1, 3, 2]));
    _clusterGpu.rasterIndexBuffer = indices;
}

function bindLightCullGroups(world: World): void {
    const state = world.resource(clusterGpuKey);
    const lightCull = world.resource(LightCull);
    const clusters = world.resource(Clusters);
    const device = world.gpu.device;
    if (
        !device ||
        !state.zSliceLayout ||
        !state.allocationLayout ||
        !state.countLayout ||
        !state.populateLayout ||
        !state.zSlicePipe ||
        !state.allocationLocalPipe ||
        !state.allocationGlobalPipe ||
        !state.countPipe ||
        !state.populatePipe ||
        !lightCull.lights ||
        !lightCull.viewMats ||
        !clusters.views ||
        !clusters.aabbs ||
        !state.rasterLights ||
        !state.clusterCounts ||
        !state.zSlices ||
        !state.rasterArgs
    )
        throw new Error("[render] light-cull bind-group inputs missing");
    const lights = lightCull.lights;
    const rasterLights = state.rasterLights;
    const viewMats = lightCull.viewMats;
    const views = clusters.views;
    const aabbs = clusters.aabbs;
    const clusterCounts = state.clusterCounts;
    const zSlices = state.zSlices;
    const rasterArgs = state.rasterArgs;
    state.zSliceBound = {
        pipeline: state.zSlicePipe,
        group: device.createBindGroup({
            layout: state.zSliceLayout,
            entries: [
                { binding: 0, resource: { buffer: lights } },
                { binding: 1, resource: { buffer: rasterLights } },
                { binding: 2, resource: { buffer: viewMats } },
                { binding: 3, resource: { buffer: views } },
                { binding: 4, resource: { buffer: rasterArgs } },
                { binding: 5, resource: { buffer: zSlices } },
            ],
        }),
    };
    state.allocationBound = null;
    state.countBound = device.createBindGroup({
        layout: state.countLayout,
        entries: [
            { binding: 0, resource: { buffer: zSlices } },
            { binding: 1, resource: { buffer: rasterLights } },
            { binding: 2, resource: { buffer: viewMats } },
            { binding: 3, resource: { buffer: views } },
            { binding: 4, resource: { buffer: aabbs } },
            { binding: 5, resource: { buffer: clusterCounts } },
        ],
    });
    state.populateBound = device.createBindGroup({
        layout: state.populateLayout,
        entries: [
            { binding: 0, resource: { buffer: zSlices } },
            { binding: 1, resource: { buffer: rasterLights } },
            { binding: 2, resource: { buffer: viewMats } },
            { binding: 3, resource: { buffer: views } },
            { binding: 4, resource: { buffer: aabbs } },
            { binding: 5, resource: { buffer: clusterCounts } },
            { binding: 6, resource: { buffer: lights } },
        ],
    });
    state.compactBound = null;
    state.compactGeneration.fill(-1);
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
 * compact lights, then count, allocate and populate their cluster lists after the AABBs are current.
 * The two allocation passes always run for the fixed grid; the other passes skip when there are no local
 * light inputs. Rasterization only draws the shading-view prefix, never depth-only shadow slots.
 */
export const CullLightsSystem: System = {
    group: "draw",
    after: [UpdateLightClustersSystem],
    update(world) {
        const _render = world.resource(RenderContext);
        const _clusterGpu = world.resource(clusterGpuKey);
        const _lightCull = world.resource(LightCull);
        const zSlice = _clusterGpu.zSliceBound;
        const count = _clusterGpu.countBound;
        const populate = _clusterGpu.populateBound;
        const args = _clusterGpu.rasterArgs;

        if (
            !_clusterGpu.compactPipe ||
            !zSlice ||
            !_clusterGpu.allocationLocalPipe ||
            !_clusterGpu.allocationGlobalPipe ||
            !_clusterGpu.countPipe ||
            !count ||
            !_clusterGpu.populatePipe ||
            !populate ||
            !_clusterGpu.clusterCounts ||
            !args ||
            !_clusterGpu.rasterIndexBuffer ||
            !_clusterGpu.countPass.colorAttachments.length ||
            _render.shadeCount === 0
        )
            return;
        warnLightOverflow(world);
        const allocation = bindAllocation(world, _render.shadeCount);

        world.gpu.device.queue.writeBuffer(
            _lightCull.viewMats!,
            0,
            _lightCull.viewStaging as Float32Array<ArrayBuffer>,
            0,
            _render.shadeCount * 16,
        );
        const lightCount = lightInputTable(world).count;
        if (lightCount !== _clusterGpu.lightCountValue) {
            lightCountData[0] = lightCount;
            world.gpu.device.queue.writeBuffer(_clusterGpu.lightCountBuffer!, 0, lightCountData);
            _clusterGpu.lightCountValue = lightCount;
        }

        const encoder = world.frameEncoder()!;
        encoder.clearBuffer(_lightCull.lights!, 0, 16);
        encoder.clearBuffer(_lightCull.lights!, LIGHT_INDICES_OFFSET, POOL_HEADER * 4);
        encoder.clearBuffer(_clusterGpu.clusterCounts!, 0, scratchCount(_render.shadeCount) * 8);
        for (let slot = 0; slot < _render.shadeCount; slot++)
            encoder.clearBuffer(args, slot * RASTER_ARGS_SIZE + 4, 4);

        if (lightCount > 0) {
            const compact = bindCompact(world);
            _clusterGpu.compactPass.timestampWrites = world.gpu.span?.("light:compact");
            const compactPass = beginComputePass(encoder, _clusterGpu.compactPass);
            setComputePipeline(compactPass, compact.pipeline);
            setComputeBindGroup(compactPass, 0, compact.group);
            dispatchWorkgroups(compactPass, Math.ceil(lightCount / 64));
            endComputePass(compactPass);

            _clusterGpu.zSlicePass.timestampWrites = world.gpu.span?.("light:z-slice");
            const zSlicePass = beginComputePass(encoder, _clusterGpu.zSlicePass);
            setComputePipeline(zSlicePass, zSlice.pipeline);
            setComputeBindGroup(zSlicePass, 0, zSlice.group);
            dispatchWorkgroups2(zSlicePass, Math.ceil(MAX_POINT_LIGHTS / 64), _render.shadeCount);
            endComputePass(zSlicePass);

            encodeLightRaster(
                world,
                _clusterGpu.countPass,
                _clusterGpu.countPipe,
                count,
                _render.shadeCount,
                "light:count",
            );
        }

        _clusterGpu.allocationLocalPass.timestampWrites = world.gpu.span?.("light:allocate-local");
        const localPass = beginComputePass(encoder, _clusterGpu.allocationLocalPass);
        setComputePipeline(localPass, _clusterGpu.allocationLocalPipe);
        setComputeBindGroup(localPass, 0, allocation);
        dispatchWorkgroups(localPass, scratchCount(_render.shadeCount) / ALLOCATION_WORKGROUP_SIZE);
        endComputePass(localPass);

        _clusterGpu.allocationGlobalPass.timestampWrites =
            world.gpu.span?.("light:allocate-global");
        const globalPass = beginComputePass(encoder, _clusterGpu.allocationGlobalPass);
        setComputePipeline(globalPass, _clusterGpu.allocationGlobalPipe);
        setComputeBindGroup(globalPass, 0, allocation);
        dispatchWorkgroups(globalPass, 1);
        endComputePass(globalPass);

        if (lightCount > 0) {
            encodeLightRaster(
                world,
                _clusterGpu.populatePass,
                _clusterGpu.populatePipe,
                populate,
                _render.shadeCount,
                "light:populate",
            );
        }
    },
};

// The counters binding spans only the presenting prefix, so the allocation passes scan live slots,
// not the grown capacity; slots past it are never shaded.
function bindAllocation(world: World, shadeCount: number): GPUBindGroup {
    const state = world.resource(clusterGpuKey);
    if (state.allocationBound && state.allocationSlots === shadeCount) return state.allocationBound;
    state.allocationBound = world.gpu.device.createBindGroup({
        layout: state.allocationLayout!,
        entries: [
            {
                binding: 0,
                resource: { buffer: state.clusterCounts!, size: scratchCount(shadeCount) * 8 },
            },
            { binding: 1, resource: { buffer: world.resource(LightCull).lights! } },
        ],
    });
    state.allocationSlots = shadeCount;
    return state.allocationBound;
}

function encodeLightRaster(
    world: World,
    descriptor: GPURenderPassDescriptor,
    pipeline: GPURenderPipeline,
    group: GPUBindGroup,
    shadeCount: number,
    label: string,
): void {
    const _clusterGpu = world.resource(clusterGpuKey);
    descriptor.timestampWrites = world.gpu.span?.(label);
    const pass = beginRenderPass(world.frameEncoder()!, descriptor);
    setRenderPipeline(pass, pipeline);
    setRenderBindGroup(pass, 0, group);
    setIndexBuffer(pass, _clusterGpu.rasterIndexBuffer!, "uint16");
    setViewport(pass, 0, 0, CLUSTER_X, CLUSTER_Y, 0, 1);
    for (let slot = 0; slot < shadeCount; slot++)
        drawIndexedIndirect(pass, _clusterGpu.rasterArgs!, slot * RASTER_ARGS_SIZE);
    endRenderPass(pass);
    world.gpu.indirect?.(label, shadeCount);
}

function allocatePerViewLightBuffers(world: World, capacity: number): void {
    const state = world.resource(clusterGpuKey);
    const lightCull = world.resource(LightCull);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    lightCull.lights = ownedBuffer(
        world,
        "shallot-light-clusters",
        LIGHT_GRID_OFFSET + capacity * CLUSTER_COUNT * 8,
        usage | GPUBufferUsage.COPY_SRC,
    );
    lightCull.viewMats = ownedBuffer(world, "shallot-light-views", capacity * 64, usage);
    state.clusterCounts = ownedBuffer(
        world,
        "shallot-light-cluster-counts",
        scratchCount(capacity) * 8,
        usage,
    );
    state.zSlices = ownedBuffer(
        world,
        "shallot-light-z-slices",
        capacity * Z_SLICE_CAPACITY * 8,
        GPUBufferUsage.STORAGE,
    );
    state.rasterArgs = ownedBuffer(
        world,
        "shallot-light-raster-args",
        capacity * RASTER_ARGS_SIZE,
        GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    );
    state.viewMatrices = Array.from({ length: capacity }, (_, slot) =>
        lightCull.viewStaging.subarray(slot * 16, slot * 16 + 16),
    );
    const args = new Uint32Array(capacity * 5);
    for (let slot = 0; slot < capacity; slot++) {
        args[slot * 5] = 6;
        args[slot * 5 + 4] = slot * Z_SLICE_CAPACITY;
    }
    world.gpu.device.queue.writeBuffer(state.rasterArgs, 0, args);
    world.gpu.buffers.set("lightClusters", lightCull.lights);
}

/** Allocate one initial light-cluster slot and compile the count-independent pipelines. */
export async function warmLightCull(world: World): Promise<void> {
    const _clusterGpu = world.resource(clusterGpuKey);

    if (!world.gpu.device) return;
    _clusterGpu.compactBound = null;
    _clusterGpu.compactGeneration.fill(-1);
    _clusterGpu.zSliceBound = null;
    _clusterGpu.allocationBound = null;
    _clusterGpu.countBound = null;
    _clusterGpu.populateBound = null;

    _clusterGpu.lightCountBuffer = ownedBuffer(
        world,
        "shallot-light-count",
        4,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    _clusterGpu.rasterLights = ownedBuffer(
        world,
        "shallot-light-raster-lights",
        MAX_POINT_LIGHTS * 48,
        GPUBufferUsage.STORAGE,
    );
    allocatePerViewLightBuffers(world, _clusterGpu.viewCapacity);
    _clusterGpu.lightCountValue = -1;
    world.gpu.buffers.set("lightCount", _clusterGpu.lightCountBuffer);

    _clusterGpu.compactPipe = world.gpu.root
        .createComputePipeline({ compute: compactKernel() })
        .$name("shallot-light-compact");
    await warmLightRaster(world);
    precompile(world, "shallot-light-compact", () => [bindCompact(world).pipeline], {
        owner: "StandardRendering",
    });
}
