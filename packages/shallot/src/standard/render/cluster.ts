import type { State, System } from "../../engine";
import { Compute, capacity } from "../../engine";
import { OCT_ENCODE_WGSL, XFORM_WGSL } from "../../engine/utils/core";
import { Camera, CameraMode } from "./camera";
import {
    MAX_POINT_LIGHTS,
    POINT_LIGHTS_BUFFER_SIZE,
    POINT_LIGHTS_STRUCT_WGSL,
    PointLight,
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

let _pipe: GPUComputePipeline | null = null;
let _group: GPUBindGroup | null = null;

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
        pass.setPipeline(_pipe);
        pass.setBindGroup(0, _group!);
        pass.dispatchWorkgroups(Math.ceil(CLUSTER_COUNT / 64), Render.shadeCount);
        pass.end();
    },
};

/** allocate the cluster buffers + compile the AABB-build pipeline */
export async function warmClusters(): Promise<void> {
    if (!Compute.device) return;
    const device = Compute.device;
    Clusters.last.fill(0);

    Clusters.views = device.createBuffer({
        label: "kitchen-cluster-views",
        size: MAX_VIEWS * CLUSTER_VIEW_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // COPY_SRC for the gym Mirror assert against the TS oracle
    Clusters.aabbs = device.createBuffer({
        label: "kitchen-cluster-aabbs",
        size: MAX_VIEWS * CLUSTER_COUNT * 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    Compute.buffers.set("clusterAabbs", Clusters.aabbs);

    // the WGSL twin of clusterAabb — one thread per (cluster, view slot)
    const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> clusterViews: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> aabbs: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cluster = gid.x;
    if (cluster >= ${CLUSTER_COUNT}u) { return; }
    let slot = gid.y;
    let p = clusterViews[slot * 2u];
    let perspective = clusterViews[slot * 2u + 1u].x > 0.5;

    let z = cluster % ${CLUSTER_Z}u;
    let xy = cluster / ${CLUSTER_Z}u;
    let x = xy % ${CLUSTER_X}u;
    let y = xy / ${CLUSTER_X}u;

    let near = p.z;
    let far = p.w;
    let dNear = near * pow(far / near, f32(z) / ${CLUSTER_Z}.0);
    let dFar = near * pow(far / near, f32(z + 1u) / ${CLUSTER_Z}.0);

    let lo = vec2<f32>(-1.0 + 2.0 * f32(x) / ${CLUSTER_X}.0, -1.0 + 2.0 * f32(y) / ${CLUSTER_Y}.0) * p.xy;
    let hi = vec2<f32>(-1.0 + 2.0 * f32(x + 1u) / ${CLUSTER_X}.0, -1.0 + 2.0 * f32(y + 1u) / ${CLUSTER_Y}.0) * p.xy;

    var mn = vec2<f32>(lo);
    var mx = vec2<f32>(hi);
    if (perspective) {
        mn = min(lo * dNear, lo * dFar);
        mx = max(hi * dNear, hi * dFar);
    }
    let base = (slot * ${CLUSTER_COUNT}u + cluster) * 2u;
    aabbs[base] = vec4<f32>(mn, -dFar, 0.0);
    aabbs[base + 1u] = vec4<f32>(mx, -dNear, 0.0);
}
`;
    const layout = device.createBindGroupLayout({
        label: "kitchen-cluster-aabbs",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
    });
    _pipe = await device.createComputePipelineAsync({
        label: "kitchen-cluster-aabbs",
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: {
            module: device.createShaderModule({ label: "kitchen-cluster-aabbs", code }),
            entryPoint: "main",
        },
    });
    _group = device.createBindGroup({
        label: "kitchen-cluster-aabbs",
        layout,
        entries: [
            { binding: 0, resource: { buffer: Clusters.views } },
            { binding: 1, resource: { buffer: Clusters.aabbs } },
        ],
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

let _compactPipe: GPUComputePipeline | null = null;
let _cullPipe: GPUComputePipeline | null = null;
let _compactLayout: GPUBindGroupLayout | null = null;
let _cullLayout: GPUBindGroupLayout | null = null;
let _compactGroup: GPUBindGroup | null = null;
let _cullGroup: GPUBindGroup | null = null;

// pool-overflow surfacing: the reserve counter lives GPU-side, so a throttled
// 8-byte readback (copy one frame, map the next) carries the warn — never
// silent truncation, never a per-frame stall
let _overflowStaging: GPUBuffer | null = null;
let _overflowPending = false;
let _overflowInFlight = false;
let _overflowWarned = false;
const OVERFLOW_PERIOD = 240;

function compactGroup(): GPUBindGroup {
    const membership = Compute.buffers.get("membership");
    const transforms = Compute.buffers.get("transforms");
    const color = PointLight.color.gpu;
    const intensity = PointLight.intensity.gpu;
    const range = PointLight.range.gpu;
    const radius = PointLight.radius.gpu;
    const spotInner = Spot.inner.gpu;
    const spotOuter = Spot.outer.gpu;
    if (
        !membership ||
        !transforms ||
        !color ||
        !intensity ||
        !range ||
        !radius ||
        !spotInner ||
        !spotOuter
    ) {
        throw new Error(
            "[render] light compact inputs missing — SlabPlugin + TransformsPlugin must be loaded",
        );
    }
    return Compute.device.createBindGroup({
        label: "kitchen-light-compact",
        layout: _compactLayout!,
        entries: [
            membership,
            transforms,
            color,
            intensity,
            range,
            radius,
            spotInner,
            spotOuter,
            LightCull.lights!,
        ].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
}

function cullGroup(): GPUBindGroup {
    return Compute.device.createBindGroup({
        label: "kitchen-light-cull",
        layout: _cullLayout!,
        entries: [
            Clusters.aabbs!,
            LightCull.lights!,
            LightCull.viewMats!,
            LightCull.grid!,
            LightCull.indices!,
        ].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
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
        _compactGroup ??= compactGroup();
        _cullGroup ??= cullGroup();

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
        pass.setPipeline(_compactPipe);
        pass.setBindGroup(0, _compactGroup);
        pass.dispatchWorkgroups(Math.ceil(capacity / 64));
        pass.setPipeline(_cullPipe);
        pass.setBindGroup(0, _cullGroup!);
        pass.dispatchWorkgroups(Math.ceil(CLUSTER_COUNT / 64), Render.shadeCount);
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
export async function warmLightCull(state: State): Promise<void> {
    if (!Compute.device) return;
    const device = Compute.device;
    _compactGroup = null;
    _cullGroup = null;
    _overflowPending = false;

    // COPY_SRC throughout for the gym Mirror asserts against the TS oracle
    LightCull.lights = device.createBuffer({
        label: "kitchen-lights",
        size: POINT_LIGHTS_BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
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

    const bit = state.membership.bit(PointLight);
    const spotBit = state.membership.bit(Spot);
    const volBit = state.membership.bit(Volumetric);

    // the GPU twin of the deleted CPU pack: membership-gated scan over capacity,
    // world position from the transforms firehose, hex sRGB color decoded to
    // linear with intensity pre-baked, posRange.w = 1/range². OCT_ENCODE_WGSL is
    // spliced for octEncodeNormal (the spot cone axis packs into one params lane)
    const compactCode =
        XFORM_WGSL +
        OCT_ENCODE_WGSL +
        /* wgsl */ `
${POINT_LIGHTS_STRUCT_WGSL.replace("count: vec4<u32>", "count: atomic<u32>,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32")}

@group(0) @binding(0) var<storage, read> membership: array<u32>;
@group(0) @binding(1) var<storage, read> transforms: array<Xform>;
@group(0) @binding(2) var<storage, read> colorF: array<f32>;
@group(0) @binding(3) var<storage, read> intensityF: array<f32>;
@group(0) @binding(4) var<storage, read> rangeF: array<f32>;
@group(0) @binding(5) var<storage, read> radiusF: array<f32>;
@group(0) @binding(6) var<storage, read> spotInnerF: array<f32>;
@group(0) @binding(7) var<storage, read> spotOuterF: array<f32>;
@group(0) @binding(8) var<storage, read_write> lights: PointLights;

fn srgb1(c: f32) -> f32 {
    return select(pow(max((c + 0.055) / 1.055, 0.0), 2.4), c / 12.92, c <= 0.04045);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    if (eid >= ${capacity}u) { return; }
    if ((membership[${bit.gen}u * ${capacity}u + eid] & ${bit.mask}u) == 0u) { return; }
    let range = rangeF[eid];
    if (range <= 0.0) { return; }
    let i = atomicAdd(&lights.count, 1u);
    if (i >= ${MAX_POINT_LIGHTS}u) { return; }
    let hex = u32(colorF[eid]);
    let intensity = intensityF[eid];
    let rgb = vec3<f32>(
        srgb1(f32((hex >> 16u) & 0xffu) / 255.0),
        srgb1(f32((hex >> 8u) & 0xffu) / 255.0),
        srgb1(f32(hex & 0xffu) / 255.0)) * intensity;
    lights.lights[i].posRange = vec4<f32>(transforms[eid].pos, 1.0 / (range * range));
    // color.a carries the source entity id (exact in f32 up to 2^24 ≫ capacity) — the hook a
    // consumer matches per-entity light extensions on (sear's point-shadow casters)
    lights.lights[i].color = vec4<f32>(rgb, f32(eid));
    // params.x = source radius (the soft-sphere falloff clamp + representative-point spec). Its sign is the
    // Volumetric opt-in flag: the lit path only ever reads radiusSq = params.x·params.x (sign-immune), so a
    // negated radius leaves shading unchanged while the fog march reads params.x < 0 as "scatter this light"
    // through the haze. max(.,1e-4) keeps the flag a nonzero negative for a radius-0 light
    var radius = radiusF[eid];
    if ((membership[${volBit.gen}u * ${capacity}u + eid] & ${volBit.mask}u) != 0u) {
        radius = -max(radius, 1e-4);
    }
    // the spot lanes (y = cone-axis oct, z/w = angular scale/offset) are (0, 0, 1) for a plain point light
    // so the FS angular factor is 1; a Spot bakes the cone here (axis = the entity's forward, scale/offset =
    // Frostbite getAngleAtt from the inner/outer half-angles — the spotParams oracle's twin)
    var params = vec4<f32>(radius, 0.0, 0.0, 1.0);
    if ((membership[${spotBit.gen}u * ${capacity}u + eid] & ${spotBit.mask}u) != 0u) {
        let dir = normalize(xformQuat(transforms[eid].quat, vec3<f32>(0.0, 0.0, -1.0)));
        let cosInner = cos(radians(spotInnerF[eid]));
        let cosOuter = cos(radians(spotOuterF[eid]));
        let scale = 1.0 / max(cosInner - cosOuter, 1e-4);
        params.y = bitcast<f32>(octEncodeNormal(dir));
        params.z = scale;
        params.w = -cosOuter * scale;
    }
    lights.lights[i].params = params;
}`;

    // one thread per (cluster, view slot). Lights batch through shared memory:
    // each thread of the workgroup transforms one light to this view's space,
    // then every thread tests the whole batch against its cluster AABB — the
    // mat4 transform runs once per workgroup, not once per cluster. Two sweeps
    // (count, then reserve + write) avoid a function-private index array (the
    // Metal dynamically-indexed-private-array miscompile, gpu.md). The batch
    // loop bound comes through workgroupUniformLoad so the in-loop barriers
    // pass uniformity analysis; out-of-range threads mask on `active` instead
    // of returning, for the same reason
    const cullCode = /* wgsl */ `
${POINT_LIGHTS_STRUCT_WGSL}

@group(0) @binding(0) var<storage, read> aabbs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> lights: PointLights;
@group(0) @binding(2) var<storage, read> viewMats: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> grid: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> pool: array<atomic<u32>>;

var<workgroup> wgCount: u32;
var<workgroup> batch: array<vec4<f32>, 64>;

// view-space sphere vs cluster AABB: squared distance from the box to the
// center against range² (posRange.w carries 1/range²)
fn hits(mn: vec3<f32>, mx: vec3<f32>, l: vec4<f32>) -> bool {
    let d = clamp(l.xyz, mn, mx) - l.xyz;
    return dot(d, d) * l.w <= 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cluster = gid.x;
    // the dispatch's y covers the shading slots alone (depth-only shadow views sit above
    // Render.shadeCount and never bin — binning them would overflow the shared index pool)
    let slot = gid.y;
    let live = cluster < ${CLUSTER_COUNT}u;
    if (lid.x == 0u) { wgCount = min(lights.count.x, ${MAX_POINT_LIGHTS}u); }
    let n = workgroupUniformLoad(&wgCount);
    let base = (slot * ${CLUSTER_COUNT}u + min(cluster, ${CLUSTER_COUNT - 1}u)) * 2u;
    let mn = aabbs[base].xyz;
    let mx = aabbs[base + 1u].xyz;
    let viewMat = viewMats[slot];

    var cnt = 0u;
    for (var b = 0u; b < n; b = b + 64u) {
        let li = b + lid.x;
        if (li < n) {
            let l = lights.lights[li];
            batch[lid.x] = vec4<f32>((viewMat * vec4<f32>(l.posRange.xyz, 1.0)).xyz, l.posRange.w);
        }
        workgroupBarrier();
        let m = min(n - b, 64u);
        if (live) {
            for (var j = 0u; j < m; j = j + 1u) {
                if (hits(mn, mx, batch[j])) { cnt = cnt + 1u; }
            }
        }
        workgroupBarrier();
    }

    var off = 0u;
    var take = 0u;
    if (live && cnt > 0u) {
        off = atomicAdd(&pool[0], cnt);
        let avail = select(0u, ${LIGHT_POOL}u - off, off < ${LIGHT_POOL}u);
        take = min(cnt, avail);
        if (cnt > take) { atomicAdd(&pool[1], cnt - take); }
    }

    var w = 0u;
    for (var b = 0u; b < n; b = b + 64u) {
        let li = b + lid.x;
        if (li < n) {
            let l = lights.lights[li];
            batch[lid.x] = vec4<f32>((viewMat * vec4<f32>(l.posRange.xyz, 1.0)).xyz, l.posRange.w);
        }
        workgroupBarrier();
        let m = min(n - b, 64u);
        if (live) {
            for (var j = 0u; j < m; j = j + 1u) {
                if (w < take && hits(mn, mx, batch[j])) {
                    atomicStore(&pool[${POOL_HEADER}u + off + w], b + j);
                    w = w + 1u;
                }
            }
        }
        workgroupBarrier();
    }

    if (live) {
        grid[slot * ${CLUSTER_COUNT}u + cluster] = vec2<u32>(${POOL_HEADER}u + off, take);
    }
}`;

    const layout = (label: string, kinds: string) =>
        device.createBindGroupLayout({
            label,
            entries: [...kinds].map((k, binding) => ({
                binding,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: k === "w" ? "storage" : "read-only-storage",
                } as GPUBufferBindingLayout,
            })),
        });
    _compactLayout = layout("kitchen-light-compact", "rrrrrrrrw");
    _cullLayout = layout("kitchen-light-cull", "rrrww");

    const pipe = (label: string, code: string, l: GPUBindGroupLayout) =>
        device.createComputePipelineAsync({
            label,
            layout: device.createPipelineLayout({ bindGroupLayouts: [l] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
        });
    [_compactPipe, _cullPipe] = await Promise.all([
        pipe("kitchen-light-compact", compactCode, _compactLayout),
        pipe("kitchen-light-cull", cullCode, _cullLayout),
    ]);
}
