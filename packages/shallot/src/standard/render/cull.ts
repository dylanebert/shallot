import { MAX_BATCH_SLOTS, MAX_SHAPES, MAX_SURFACES } from "./mesh";

export const CULL_WORKGROUP_SIZE = 64;
export const SHAPE_AABB_STRIDE = 8;

export const CULL_SHARED_WGSL = /* wgsl */ `
struct CullParams {
    planes: array<vec4<f32>, 6>,
    entityCount: u32,
}

@group(0) @binding(0) var<uniform> params: CullParams;
@group(0) @binding(1) var<storage, read> matrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> sizes: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> shapeAABBs: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> cullEntities: array<vec2<u32>>;

@group(1) @binding(0) var<storage, read_write> indirect: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> entityIds: array<u32>;

struct WorldSphere {
    center: vec3<f32>,
    radius: f32,
    batchSlot: u32,
    eid: u32,
}

fn computeWorldSphere(gid: u32) -> WorldSphere {
    let packed = cullEntities[gid];
    let eid = packed.x;
    let batchSlot = packed.y;

    let shapeIdx = (batchSlot % ${MAX_BATCH_SLOTS}u) / ${MAX_SURFACES}u;
    let aabbIdx = shapeIdx * 2u;
    let aabbMin = shapeAABBs[aabbIdx];
    let aabbMax = shapeAABBs[aabbIdx + 1u];

    let size = sizes[eid];
    let localMin = aabbMin.xyz * size.xyz;
    let localMax = aabbMax.xyz * size.xyz;

    let localCenter = (localMin + localMax) * 0.5;
    let localExtent = (localMax - localMin) * 0.5;

    let world = matrices[eid];
    let worldCenter = (world * vec4<f32>(localCenter, 1.0)).xyz;

    let absCol0 = abs(world[0].xyz);
    let absCol1 = abs(world[1].xyz);
    let absCol2 = abs(world[2].xyz);
    let worldExtent = absCol0 * localExtent.x + absCol1 * localExtent.y + absCol2 * localExtent.z;
    let radius = length(worldExtent);

    return WorldSphere(worldCenter, radius, batchSlot, eid);
}

fn frustumTest(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = params.planes[i];
        let dist = dot(plane.xyz, center) + plane.w;
        if (dist < -radius) { return false; }
    }
    return true;
}

fn emitVisible(sphere: WorldSphere) {
    let indirectBase = sphere.batchSlot * 5u;
    let firstInstance = atomicLoad(&indirect[indirectBase + 4u]);
    let idx = atomicAdd(&indirect[indirectBase + 1u], 1u);
    entityIds[firstInstance + idx] = sphere.eid;
}
`;

export function packShapeAABBs(src: Float32Array, out: Float32Array): void {
    for (let shapeId = 0; shapeId < MAX_SHAPES; shapeId++) {
        const s = shapeId * 6;
        const d = shapeId * SHAPE_AABB_STRIDE;
        out[d] = src[s];
        out[d + 1] = src[s + 1];
        out[d + 2] = src[s + 2];
        out[d + 3] = 0;
        out[d + 4] = src[s + 3];
        out[d + 5] = src[s + 4];
        out[d + 6] = src[s + 5];
        out[d + 7] = 0;
    }
}
