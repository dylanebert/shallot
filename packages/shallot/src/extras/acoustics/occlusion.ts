import type { ComputeNode, ExecutionContext } from "../../standard/compute";
import { beginComputePass } from "../../standard/compute";
import {
    TREE_NODE_STRUCT_WGSL,
    LEAF_FLAG_WGSL,
    LEAF_FUNCTIONS_WGSL,
    AABB_SENTINEL_WGSL,
} from "../../standard/bvh";
import {
    BODY_STRUCT_WGSL,
    QUAT_WGSL,
    SHAPE_CONSTS_WGSL,
    type PhysicsGPU,
} from "../../standard/physics/gpu";
import { AcousticMaterialData } from "./material";
import { MAX_SOURCES } from "./dsp";
import { write, capacity } from "../../engine";
const SOURCE_STRIDE = 4;
const PARAMS_SIZE = 32;

const BVH_UTILS_WGSL = /* wgsl */ `
const MAX_STACK_DEPTH: u32 = 32u;

fn quatRotateInverse(q: vec4f, v: vec3f) -> vec3f {
    return quatRotate(quatConj(q), v);
}

fn intersectAABBDist(origin: vec3f, invDir: vec3f, bmin: vec3f, bmax: vec3f) -> f32 {
    let t1 = (bmin - origin) * invDir;
    let t2 = (bmax - origin) * invDir;
    let tMin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tMax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tMax < max(tMin, 0.0)) { return AABB_SENTINEL; }
    return max(tMin, 0.0);
}
`;

const RAY_BODY_CLOSEST_WGSL = /* wgsl */ `
struct ClosestHitOcc {
    t: f32,
    bodyIdx: u32,
    hit: bool,
}

fn intersectBoxClosestOcc(localOrigin: vec3f, localDir: vec3f, halfExt: vec3f) -> f32 {
    let invDir = 1.0 / localDir;
    let t1 = (-halfExt - localOrigin) * invDir;
    let t2 = (halfExt - localOrigin) * invDir;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tFar < max(tNear, 0.0)) { return -1.0; }
    if (tNear < 0.0) { return -1.0; }
    return tNear;
}

fn intersectSphereClosestOcc(localOrigin: vec3f, localDir: vec3f, radius: f32) -> f32 {
    let a = dot(localDir, localDir);
    let b = 2.0 * dot(localOrigin, localDir);
    let c = dot(localOrigin, localOrigin) - radius * radius;
    if (c < 0.0) { return -1.0; }
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) { return -1.0; }
    let sqrtDisc = sqrt(disc);
    let t0 = (-b - sqrtDisc) / (2.0 * a);
    if (t0 < 0.0) { return -1.0; }
    return t0;
}

fn intersectCapsuleClosestOcc(localOrigin: vec3f, localDir: vec3f, halfHeight: f32, radius: f32) -> f32 {
    let clampedY = clamp(localOrigin.y, -halfHeight, halfHeight);
    let closestAxisPt = vec3f(0.0, clampedY, 0.0);
    if (dot(localOrigin - closestAxisPt, localOrigin - closestAxisPt) < radius * radius) {
        return -1.0;
    }
    var bestT = -1.0f;
    let ox = localOrigin.x; let oz = localOrigin.z;
    let dx = localDir.x; let dz = localDir.z;
    let a = dx * dx + dz * dz;
    let b = 2.0 * (ox * dx + oz * dz);
    let c = ox * ox + oz * oz - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if (disc >= 0.0 && a > 1e-8) {
        let sqrtDisc = sqrt(disc);
        for (var ci = 0; ci < 2; ci++) {
            let t = (-b + select(sqrtDisc, -sqrtDisc, ci == 0)) / (2.0 * a);
            if (t >= 0.0 && (bestT < 0.0 || t < bestT)) {
                let y = localOrigin.y + t * localDir.y;
                if (abs(y) <= halfHeight) { bestT = t; }
            }
        }
    }
    let centers = array(vec3f(0, halfHeight, 0), vec3f(0, -halfHeight, 0));
    for (var ci = 0; ci < 2; ci++) {
        let oc = localOrigin - centers[ci];
        let sa = dot(localDir, localDir);
        let sb = 2.0 * dot(oc, localDir);
        let sc = dot(oc, oc) - radius * radius;
        let sd = sb * sb - 4.0 * sa * sc;
        if (sd >= 0.0) {
            let sqd = sqrt(sd);
            let t0 = (-sb - sqd) / (2.0 * sa);
            if (t0 >= 0.0 && (bestT < 0.0 || t0 < bestT)) { bestT = t0; }
        }
    }
    return bestT;
}

fn intersectBodyClosestOcc(body: Body, rayOrigin: vec3f, rayDir: vec3f, tMax: f32) -> f32 {
    let localOrigin = quatRotateInverse(body.quat, rayOrigin - body.pos);
    let localDir = quatRotateInverse(body.quat, rayDir);
    var t = -1.0f;
    if (body.colliderType == SHAPE_BOX) {
        t = intersectBoxClosestOcc(localOrigin, localDir, body.halfExtents);
    } else if (body.colliderType == SHAPE_SPHERE) {
        t = intersectSphereClosestOcc(localOrigin, localDir, body.radius);
    } else if (body.colliderType == SHAPE_CAPSULE) {
        t = intersectCapsuleClosestOcc(localOrigin, localDir, body.halfExtents.y, body.radius);
    }
    if (t < 0.0 || t >= tMax) { return -1.0; }
    return t;
}
`;

const TRACE_CLOSEST_OCC_WGSL = /* wgsl */ `
fn traceOcclusionClosest(origin: vec3f, dir: vec3f, tMax: f32) -> ClosestHitOcc {
    let bodyCount = params.bodyCount;
    if (bodyCount == 0u) { return ClosestHitOcc(0.0, 0u, false); }

    let invDir = 1.0 / dir;
    var bestT = tMax;
    var bestBody = 0u;
    var found = false;

    var stack: array<u32, MAX_STACK_DEPTH>;
    var stackPtr = 0u;
    stack[stackPtr] = 0u;
    stackPtr++;

    while (stackPtr > 0u) {
        stackPtr--;
        let nodeIdx = stack[stackPtr];
        let node = treeNodes[nodeIdx];
        let leftChild = node.leftChild;
        let rightChild = node.rightChild;

        if (leftChild != 0xFFFFFFFFu) {
            if (isLeaf(leftChild)) {
                let bodyIdx = sortedIds[leafIndex(leftChild)];
                let t = intersectBodyClosestOcc(bodies[bodyIdx], origin, dir, bestT);
                if (t >= 0.0) {
                    bestT = t;
                    bestBody = bodyIdx;
                    found = true;
                }
            } else {
                let lNode = treeNodes[leftChild];
                let lDist = intersectAABBDist(origin, invDir,
                    vec3f(lNode.minX, lNode.minY, lNode.minZ),
                    vec3f(lNode.maxX, lNode.maxY, lNode.maxZ));
                if (lDist < bestT && stackPtr < MAX_STACK_DEPTH) {
                    stack[stackPtr] = leftChild;
                    stackPtr++;
                }
            }
        }

        if (rightChild != 0xFFFFFFFFu) {
            if (isLeaf(rightChild)) {
                let bodyIdx = sortedIds[leafIndex(rightChild)];
                let t = intersectBodyClosestOcc(bodies[bodyIdx], origin, dir, bestT);
                if (t >= 0.0) {
                    bestT = t;
                    bestBody = bodyIdx;
                    found = true;
                }
            } else {
                let rNode = treeNodes[rightChild];
                let rDist = intersectAABBDist(origin, invDir,
                    vec3f(rNode.minX, rNode.minY, rNode.minZ),
                    vec3f(rNode.maxX, rNode.maxY, rNode.maxZ));
                if (rDist < bestT && stackPtr < MAX_STACK_DEPTH) {
                    stack[stackPtr] = rightChild;
                    stackPtr++;
                }
            }
        }
    }

    return ClosestHitOcc(bestT, bestBody, found);
}
`;

const SHADER_CODE = /* wgsl */ `
${BODY_STRUCT_WGSL}
${TREE_NODE_STRUCT_WGSL}
${LEAF_FLAG_WGSL}
${LEAF_FUNCTIONS_WGSL}
${AABB_SENTINEL_WGSL}
${QUAT_WGSL}
${SHAPE_CONSTS_WGSL}

struct AcousticSource {
    posX: f32,
    posY: f32,
    posZ: f32,
    voiceSlot: u32,
}

struct OcclusionParams {
    listenerX: f32,
    listenerY: f32,
    listenerZ: f32,
    sourceCount: u32,
    bodyCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: OcclusionParams;
@group(0) @binding(1) var<storage, read> sources: array<AcousticSource>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

@group(1) @binding(0) var<storage, read> treeNodes: array<TreeNode>;
@group(1) @binding(1) var<storage, read> sortedIds: array<u32>;
@group(1) @binding(2) var<storage, read> bodies: array<Body>;

@group(2) @binding(0) var<storage, read> bodyEids: array<u32>;
@group(2) @binding(1) var<storage, read> materialProps: array<f32>;

${BVH_UTILS_WGSL}
${RAY_BODY_CLOSEST_WGSL}
${TRACE_CLOSEST_OCC_WGSL}

fn getTransmission3(bodyIdx: u32) -> vec3f {
    let eid = bodyEids[bodyIdx];
    let base = eid * 8u;
    return vec3f(materialProps[base + 4u], materialProps[base + 5u], materialProps[base + 6u]);
}

const SOURCE_RADIUS: f32 = 0.5;
const NUM_OCCLUSION_SAMPLES: u32 = 8u;

fn pcgOcc(seed: u32) -> u32 {
    var s = seed * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn sampleSphere(s1: u32, s2: u32, center: vec3f, radius: f32) -> vec3f {
    let u = f32(s1) / 4294967295.0;
    let v = f32(s2) / 4294967295.0;
    let z = 2.0 * u - 1.0;
    let phi = 6.2831853 * v;
    let r = sqrt(max(0.0, 1.0 - z * z)) * radius * pow(f32(s1 ^ s2) / 4294967295.0, 0.333);
    return center + vec3f(r * cos(phi), r * sin(phi), z * radius * pow(f32(s1 ^ s2) / 4294967295.0, 0.333));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.sourceCount) { return; }

    let src = sources[idx];
    let listener = vec3(params.listenerX, params.listenerY, params.listenerZ);
    let srcPos = vec3(src.posX, src.posY, src.posZ);

    let dir = srcPos - listener;
    let dist = length(dir);
    if (dist < 0.001) {
        results[idx * 4u] = 1.0;
        results[idx * 4u + 1u] = 0.0;
        results[idx * 4u + 2u] = 0.0;
        results[idx * 4u + 3u] = 0.0;
        return;
    }

    var visible = 0.0;
    var trans = vec3f(0.0);
    var occluded = 0u;
    var rng = idx * 1337u + params.bodyCount;

    for (var s = 0u; s < NUM_OCCLUSION_SAMPLES; s++) {
        rng = pcgOcc(rng);
        let s1 = rng;
        rng = pcgOcc(rng);
        let s2 = rng;
        let samplePt = sampleSphere(s1, s2, srcPos, SOURCE_RADIUS);
        let toTarget = samplePt - listener;
        let tDist = length(toTarget);
        if (tDist < 0.001) { visible += 1.0; continue; }

        let hit = traceOcclusionClosest(listener, normalize(toTarget), tDist - 0.01);
        if (!hit.hit) {
            visible += 1.0;
        } else {
            trans += getTransmission3(hit.bodyIdx);
            occluded += 1u;
        }
    }

    let nf = f32(NUM_OCCLUSION_SAMPLES);
    let visFrac = visible / nf;
    let avgTrans = select(vec3f(0.0), trans / f32(occluded), occluded > 0u);

    results[idx * 4u] = visFrac;
    results[idx * 4u + 1u] = avgTrans.x;
    results[idx * 4u + 2u] = avgTrans.y;
    results[idx * 4u + 3u] = avgTrans.z;
}
`;

export interface OcclusionState {
    sourceCount: number;
    listener: Float32Array;
    sources: Float32Array;
    sourcesU32: Uint32Array;
    slots: Uint32Array;
    readbackReady: boolean;
    readbackBuf: Float32Array;
    readbackCount: number;
    generation: number;
}

export function createOcclusionState(): OcclusionState {
    const sources = new Float32Array(MAX_SOURCES * SOURCE_STRIDE);
    return {
        sourceCount: 0,
        listener: new Float32Array(3),
        sources,
        sourcesU32: new Uint32Array(sources.buffer),
        slots: new Uint32Array(MAX_SOURCES),
        readbackReady: false,
        readbackBuf: new Float32Array(MAX_SOURCES * 4),
        readbackCount: 0,
        generation: 0,
    };
}

export function createOcclusionNode(physics: PhysicsGPU, occ: OcclusionState): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let paramsBuffer: GPUBuffer | null = null;
    let sourceBuffer: GPUBuffer | null = null;
    let resultBuffer: GPUBuffer | null = null;
    let bg0: GPUBindGroup | null = null;
    let bg1: GPUBindGroup | null = null;
    let bg2: GPUBindGroup | null = null;
    let cachedTreeNodes: GPUBuffer | null = null;
    let cachedEids: GPUBuffer | null = null;
    let materialPropsBuffer: GPUBuffer | null = null;
    let cachedMaterialSize = 0;
    let staging: GPUBuffer | null = null;
    let readbackPending = false;

    const paramsUpload = new ArrayBuffer(PARAMS_SIZE);
    const paramsF32 = new Float32Array(paramsUpload);
    const paramsU32 = new Uint32Array(paramsUpload);

    return {
        name: "acoustics-occlusion",
        scope: "frame",
        inputs: [],
        outputs: [],
        execute(ctx: ExecutionContext) {
            if (occ.sourceCount === 0 || readbackPending) return;

            const { device, encoder } = ctx;

            if (!pipeline) {
                pipeline = device.createComputePipeline({
                    label: "acoustics-occlusion",
                    layout: "auto",
                    compute: { module: device.createShaderModule({ code: SHADER_CODE }) },
                });
            }

            if (!paramsBuffer) {
                paramsBuffer = device.createBuffer({
                    label: "acoustics-params",
                    size: PARAMS_SIZE,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
            }

            if (!sourceBuffer) {
                sourceBuffer = device.createBuffer({
                    label: "acoustics-sources",
                    size: MAX_SOURCES * SOURCE_STRIDE * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
            }

            if (!resultBuffer) {
                resultBuffer = device.createBuffer({
                    label: "acoustics-results",
                    size: MAX_SOURCES * 4 * 4,
                    usage:
                        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                });
            }

            if (!staging) {
                staging = device.createBuffer({
                    label: "acoustics-occ-staging",
                    size: MAX_SOURCES * 4 * 4,
                    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                });
            }

            const bodyCount = physics.bodyEids.length;

            paramsF32[0] = occ.listener[0];
            paramsF32[1] = occ.listener[1];
            paramsF32[2] = occ.listener[2];
            paramsU32[3] = occ.sourceCount;
            paramsU32[4] = bodyCount;
            device.queue.writeBuffer(paramsBuffer, 0, paramsUpload);

            device.queue.writeBuffer(
                sourceBuffer,
                0,
                occ.sources.buffer,
                occ.sources.byteOffset,
                occ.sourceCount * SOURCE_STRIDE * 4,
            );

            const matSize = Math.max(capacity() * 8 * 4, 16);
            if (!materialPropsBuffer || cachedMaterialSize < matSize) {
                materialPropsBuffer?.destroy();
                materialPropsBuffer = device.createBuffer({
                    label: "occlusion-material-props",
                    size: matSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
                cachedMaterialSize = matSize;
                bg2 = null;
            }
            write(device.queue, materialPropsBuffer, 0, AcousticMaterialData, capacity());

            const treeNodesBuffer = physics.lbvh.lbvh.treeNodes;
            const eidsGPU = physics.eidsBuffer.buffer;

            if (
                !bg0 ||
                !bg1 ||
                !bg2 ||
                cachedTreeNodes !== treeNodesBuffer ||
                cachedEids !== eidsGPU
            ) {
                bg0 = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: sourceBuffer } },
                        { binding: 2, resource: { buffer: resultBuffer } },
                    ],
                });
                bg1 = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: treeNodesBuffer } },
                        { binding: 1, resource: { buffer: physics.lbvh.lbvh.sortedIds } },
                        { binding: 2, resource: { buffer: physics.bodyBuffer.buffer } },
                    ],
                });
                bg2 = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(2),
                    entries: [
                        { binding: 0, resource: { buffer: physics.eidsBuffer.buffer } },
                        { binding: 1, resource: { buffer: materialPropsBuffer } },
                    ],
                });
                cachedTreeNodes = treeNodesBuffer;
                cachedEids = eidsGPU;
            }

            const pass = beginComputePass(encoder, ctx.timestampWrites?.("acoustics-occlusion"));
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg0);
            pass.setBindGroup(1, bg1);
            pass.setBindGroup(2, bg2);
            pass.dispatchWorkgroups(1);
            pass.end();

            const readbackSize = occ.sourceCount * 4 * 4;
            encoder.copyBufferToBuffer(resultBuffer, 0, staging, 0, readbackSize);

            const count = occ.sourceCount;
            const gen = ++occ.generation;
            readbackPending = true;
            ctx.afterSubmit(async () => {
                try {
                    await staging!.mapAsync(GPUMapMode.READ);
                    if (occ.generation === gen) {
                        const src = new Float32Array(staging!.getMappedRange(), 0, count * 4);
                        occ.readbackBuf.set(src);
                        occ.readbackCount = count;
                        occ.readbackReady = true;
                    }
                    staging!.unmap();
                } catch {
                } finally {
                    readbackPending = false;
                }
            });
        },
    };
}
