import { capacity } from "../../engine";
import { beginComputePass } from "../compute";
import {
    createLBVH,
    dispatchLBVH,
    disposeLBVH,
    rebuildLBVHBuffers,
    INSTANCE_AABB_STRUCT_WGSL,
    AABB_SENTINEL_WGSL,
    type LBVH,
} from "../bvh";
import { gbuf, binding, type GBuf, type Binding } from "../compute";

export interface PhysicsLBVH {
    bodyAABBs: GBuf;
    countBuffer: GPUBuffer;
    lbvh: LBVH;
    computeAABBsPipeline: GPUComputePipeline;
    computeAABBsBindGroup: Binding;
    cachedCapacity: number;
}

export const BODY_STRUCT_WGSL = `struct Body {
    pos: vec3f,
    mass: f32,
    vel: vec3f,
    momentX: f32,
    angVel: vec3f,
    radius: f32,
    inertial: vec3f,
    friction: f32,
    initial: vec3f,
    hullId: u32,
    quat: vec4f,
    inertialQuat: vec4f,
    initialQuat: vec4f,
    prevVel: vec3f,
    momentY: f32,
    prevAngVel: vec3f,
    momentZ: f32,
    cumAng: vec3f,
    gravity: f32,
    halfExtents: vec3f,
    colliderType: f32,
    collisionGroup: u32,
}`;

const PARAMS_STRUCT_WGSL = /* wgsl */ `
struct Params {
    dt: f32,
    gravity: f32,
    iterations: u32,
    alpha: f32,
    beta: f32,
    gamma: f32,
    bodyCount: u32,
    jointCount: u32,
    capacity: u32,
    constraintMul: u32,
    hashMul: u32,
    _pad2: u32,
}`;

export const QUAT_WGSL = /* wgsl */ `
fn quatMul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

fn quatConj(q: vec4f) -> vec4f {
    return vec4f(-q.x, -q.y, -q.z, q.w);
}

fn quatRotate(q: vec4f, v: vec3f) -> vec3f {
    let u = q.xyz;
    let t = 2.0 * cross(u, v);
    return v + q.w * t + cross(u, t);
}`;

export const SHAPE_CONSTS_WGSL = /* wgsl */ `
const SHAPE_BOX: f32 = 0.0;
const SHAPE_SPHERE: f32 = 1.0;
const SHAPE_CAPSULE: f32 = 2.0;
const SHAPE_HULL: f32 = 3.0;
`;

const computeAABBsShader = /* wgsl */ `
${BODY_STRUCT_WGSL}
${INSTANCE_AABB_STRUCT_WGSL}
${PARAMS_STRUCT_WGSL}
${AABB_SENTINEL_WGSL}
${SHAPE_CONSTS_WGSL}

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read_write> bodyAABBs: array<InstanceAABB>;
@group(0) @binding(2) var<uniform> params: Params;
${QUAT_WGSL}

fn hasNaN(v: vec3f) -> bool {
    return v.x != v.x || v.y != v.y || v.z != v.z;
}

const BROADPHASE_MARGIN: f32 = 0.04;

fn primitiveAABB(body: Body) -> array<vec3f, 2> {
    let margin = vec3f(BROADPHASE_MARGIN);
    if (body.colliderType == SHAPE_SPHERE) {
        let r = vec3f(body.radius);
        return array(body.pos - r - margin, body.pos + r + margin);
    }
    if (body.colliderType == SHAPE_CAPSULE) {
        let axis = quatRotate(body.quat, vec3f(0, body.halfExtents.y, 0));
        let tipA = body.pos + axis;
        let tipB = body.pos - axis;
        let lo = min(tipA, tipB);
        let hi = max(tipA, tipB);
        let r = vec3f(body.radius);
        return array(lo - r - margin, hi + r + margin);
    }
    let h = body.halfExtents;
    let ax = abs(quatRotate(body.quat, vec3f(h.x, 0, 0)));
    let ay = abs(quatRotate(body.quat, vec3f(0, h.y, 0)));
    let az = abs(quatRotate(body.quat, vec3f(0, 0, h.z)));
    let ext = ax + ay + az;
    return array(body.pos - ext - margin, body.pos + ext + margin);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.bodyCount) { return; }

    let body = bodies[idx];
    let aabb = primitiveAABB(body);

    var out: InstanceAABB;
    out._pad0 = 0u;
    out._pad1 = 0u;
    if (hasNaN(aabb[0]) || hasNaN(aabb[1])) {
        out.minX = AABB_SENTINEL;
        out.minY = AABB_SENTINEL;
        out.minZ = AABB_SENTINEL;
        out.maxX = -AABB_SENTINEL;
        out.maxY = -AABB_SENTINEL;
        out.maxZ = -AABB_SENTINEL;
    } else {
        out.minX = aabb[0].x;
        out.minY = aabb[0].y;
        out.minZ = aabb[0].z;
        out.maxX = aabb[1].x;
        out.maxY = aabb[1].y;
        out.maxZ = aabb[1].z;
    }
    bodyAABBs[idx] = out;
}
`;

export async function createPhysicsLBVH(
    device: GPUDevice,
    bodyBuffer: GBuf,
    paramsBuffer: GPUBuffer,
): Promise<PhysicsLBVH> {
    const bodyAABBs = gbuf(
        device,
        "physics-lbvh-bodyAABBs",
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        (c) => c * 32,
    );

    const countBuffer = device.createBuffer({
        label: "physics-lbvh-count",
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const computeAABBsModule = device.createShaderModule({ code: computeAABBsShader });

    const [computeAABBsPipeline, lbvh] = await Promise.all([
        device.createComputePipelineAsync({
            label: "physics-aabb",
            layout: "auto",
            compute: { module: computeAABBsModule, entryPoint: "main" },
        }),
        createLBVH(device, {
            leafAABBs: bodyAABBs.buffer,
            countBuffer,
            maxLeaves: capacity(),
            label: "physics-lbvh",
        }),
    ]);

    const computeAABBsBindGroup = binding(
        device,
        computeAABBsPipeline.getBindGroupLayout(0),
        () => [
            { binding: 0, resource: { buffer: bodyBuffer.buffer } },
            { binding: 1, resource: { buffer: bodyAABBs.buffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
        ],
    );

    return {
        bodyAABBs,
        countBuffer,
        lbvh,
        computeAABBsPipeline,
        computeAABBsBindGroup,
        cachedCapacity: capacity(),
    };
}

const lbvhCountData = new Uint32Array(1);

export function dispatchPhysicsLBVH(
    phys: PhysicsLBVH,
    encoder: GPUCommandEncoder,
    device: GPUDevice,
    bodyCount: number,
    ts?: (name: string) => GPUComputePassTimestampWrites | undefined,
): void {
    const cap = capacity();
    if (cap !== phys.cachedCapacity) {
        phys.cachedCapacity = cap;
        rebuildLBVHBuffers(phys.lbvh, device, phys.bodyAABBs.buffer, cap);
    }

    lbvhCountData[0] = bodyCount;
    device.queue.writeBuffer(phys.countBuffer, 0, lbvhCountData);

    const aabbWG = Math.ceil(bodyCount / 64);
    const pass = beginComputePass(encoder, ts?.("phys:aabb"));
    pass.setPipeline(phys.computeAABBsPipeline);
    pass.setBindGroup(0, phys.computeAABBsBindGroup.group);
    pass.dispatchWorkgroups(aabbWG);
    pass.end();

    dispatchLBVH(phys.lbvh, encoder, device, bodyCount, ts);
}

export function disposePhysicsLBVH(phys: PhysicsLBVH): void {
    phys.bodyAABBs.buffer.destroy();
    phys.countBuffer.destroy();
    disposeLBVH(phys.lbvh);
}
