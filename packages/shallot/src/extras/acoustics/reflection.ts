import type { ComputeNode, ExecutionContext } from "../../standard/compute";
import { beginComputePass } from "../../standard/compute";
import { write, capacity } from "../../engine";
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
import type { OcclusionState } from "./occlusion";
import { AcousticMaterialData } from "./material";
import { MAX_SOURCES, NUM_BINS } from "./dsp";

const NUM_RAYS = 4096;
const MAX_BOUNCES = 16;

const CLOSEST_HIT_WGSL = /* wgsl */ `
fn quatRotateInverse(q: vec4f, v: vec3f) -> vec3f {
    return quatRotate(quatConj(q), v);
}

struct ClosestHit {
    t: f32,
    normal: vec3f,
    bodyIdx: u32,
    hit: bool,
}

fn intersectBoxClosest(localOrigin: vec3f, localDir: vec3f, halfExt: vec3f) -> vec4f {
    let invDir = 1.0 / localDir;
    let t1 = (-halfExt - localOrigin) * invDir;
    let t2 = (halfExt - localOrigin) * invDir;
    let tMinVec = min(t1, t2);
    let tMaxVec = max(t1, t2);
    let tNear = max(max(tMinVec.x, tMinVec.y), tMinVec.z);
    let tFar = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
    if (tFar < max(tNear, 0.0)) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
    if (tNear < 0.0) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
    var normal = vec3f(0.0);
    if (tMinVec.x >= tMinVec.y && tMinVec.x >= tMinVec.z) {
        normal = vec3f(sign(-localDir.x), 0.0, 0.0);
    } else if (tMinVec.y >= tMinVec.x && tMinVec.y >= tMinVec.z) {
        normal = vec3f(0.0, sign(-localDir.y), 0.0);
    } else {
        normal = vec3f(0.0, 0.0, sign(-localDir.z));
    }
    return vec4f(tNear, normal);
}

fn intersectSphereClosest(localOrigin: vec3f, localDir: vec3f, radius: f32) -> vec4f {
    let a = dot(localDir, localDir);
    let b = 2.0 * dot(localOrigin, localDir);
    let c = dot(localOrigin, localOrigin) - radius * radius;
    if (c < 0.0) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
    let sqrtDisc = sqrt(disc);
    let t0 = (-b - sqrtDisc) / (2.0 * a);
    if (t0 < 0.0) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
    let hitPoint = localOrigin + localDir * t0;
    let normal = normalize(hitPoint);
    return vec4f(t0, normal);
}

fn intersectCapsuleClosest(localOrigin: vec3f, localDir: vec3f, halfHeight: f32, radius: f32) -> vec4f {
    let clampedY = clamp(localOrigin.y, -halfHeight, halfHeight);
    let closestAxisPt = vec3f(0.0, clampedY, 0.0);
    if (dot(localOrigin - closestAxisPt, localOrigin - closestAxisPt) < radius * radius) {
        return vec4f(-1.0, 0.0, 0.0, 0.0);
    }
    var bestT = -1.0f;
    var bestNormal = vec3f(0.0);
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
                if (abs(y) <= halfHeight) {
                    bestT = t;
                    let hp = localOrigin + localDir * t;
                    bestNormal = normalize(vec3f(hp.x, 0.0, hp.z));
                }
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
            if (t0 >= 0.0 && (bestT < 0.0 || t0 < bestT)) {
                bestT = t0;
                let hp = localOrigin + localDir * t0 - centers[ci];
                bestNormal = normalize(hp);
            }
        }
    }
    if (bestT < 0.0) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
    return vec4f(bestT, bestNormal);
}

fn intersectBodyClosest(body: Body, rayOrigin: vec3f, rayDir: vec3f, tMax: f32) -> ClosestHit {
    let localOrigin = quatRotateInverse(body.quat, rayOrigin - body.pos);
    let localDir = quatRotateInverse(body.quat, rayDir);
    var result = vec4f(-1.0, 0.0, 0.0, 0.0);
    if (body.colliderType == SHAPE_BOX) {
        result = intersectBoxClosest(localOrigin, localDir, body.halfExtents);
    } else if (body.colliderType == SHAPE_SPHERE) {
        result = intersectSphereClosest(localOrigin, localDir, body.radius);
    } else if (body.colliderType == SHAPE_CAPSULE) {
        result = intersectCapsuleClosest(localOrigin, localDir, body.halfExtents.y, body.radius);
    }
    if (result.x < 0.0 || result.x >= tMax) {
        return ClosestHit(0.0, vec3f(0.0), 0u, false);
    }
    let worldNormal = quatRotate(body.quat, result.yzw);
    return ClosestHit(result.x, worldNormal, 0u, true);
}
`;

const TRACE_CLOSEST_WGSL = /* wgsl */ `
const MAX_STACK_DEPTH: u32 = 32u;

fn intersectAABBDist(origin: vec3f, invDir: vec3f, bmin: vec3f, bmax: vec3f) -> f32 {
    let t1 = (bmin - origin) * invDir;
    let t2 = (bmax - origin) * invDir;
    let tMin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tMax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tMax < max(tMin, 0.0)) { return AABB_SENTINEL; }
    return max(tMin, 0.0);
}

fn traceClosest(origin: vec3f, dir: vec3f, maxDist: f32) -> ClosestHit {
    let bodyCount = params.bodyCount;
    if (bodyCount == 0u) { return ClosestHit(0.0, vec3f(0.0), 0u, false); }

    let invDir = 1.0 / dir;
    var bestHit = ClosestHit(maxDist, vec3f(0.0), 0u, false);

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
                var hit = intersectBodyClosest(bodies[bodyIdx], origin, dir, bestHit.t);
                if (hit.hit) {
                    hit.bodyIdx = bodyIdx;
                    bestHit = hit;
                }
            } else {
                let lNode = treeNodes[leftChild];
                let lDist = intersectAABBDist(origin, invDir,
                    vec3f(lNode.minX, lNode.minY, lNode.minZ),
                    vec3f(lNode.maxX, lNode.maxY, lNode.maxZ));
                if (lDist < bestHit.t && stackPtr < MAX_STACK_DEPTH) {
                    stack[stackPtr] = leftChild;
                    stackPtr++;
                }
            }
        }

        if (rightChild != 0xFFFFFFFFu) {
            if (isLeaf(rightChild)) {
                let bodyIdx = sortedIds[leafIndex(rightChild)];
                var hit = intersectBodyClosest(bodies[bodyIdx], origin, dir, bestHit.t);
                if (hit.hit) {
                    hit.bodyIdx = bodyIdx;
                    bestHit = hit;
                }
            } else {
                let rNode = treeNodes[rightChild];
                let rDist = intersectAABBDist(origin, invDir,
                    vec3f(rNode.minX, rNode.minY, rNode.minZ),
                    vec3f(rNode.maxX, rNode.maxY, rNode.maxZ));
                if (rDist < bestHit.t && stackPtr < MAX_STACK_DEPTH) {
                    stack[stackPtr] = rightChild;
                    stackPtr++;
                }
            }
        }
    }

    return bestHit;
}

fn traceAcousticShadow(origin: vec3f, dir: vec3f, tMax: f32) -> bool {
    let bodyCount = params.bodyCount;
    if (bodyCount == 0u) { return false; }

    let invDir = 1.0 / dir;

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
                let hit = intersectBodyClosest(bodies[bodyIdx], origin, dir, tMax);
                if (hit.hit) { return true; }
            } else {
                let lNode = treeNodes[leftChild];
                let lDist = intersectAABBDist(origin, invDir,
                    vec3f(lNode.minX, lNode.minY, lNode.minZ),
                    vec3f(lNode.maxX, lNode.maxY, lNode.maxZ));
                if (lDist < tMax && stackPtr < MAX_STACK_DEPTH) {
                    stack[stackPtr] = leftChild;
                    stackPtr++;
                }
            }
        }

        if (rightChild != 0xFFFFFFFFu) {
            if (isLeaf(rightChild)) {
                let bodyIdx = sortedIds[leafIndex(rightChild)];
                let hit = intersectBodyClosest(bodies[bodyIdx], origin, dir, tMax);
                if (hit.hit) { return true; }
            } else {
                let rNode = treeNodes[rightChild];
                let rDist = intersectAABBDist(origin, invDir,
                    vec3f(rNode.minX, rNode.minY, rNode.minZ),
                    vec3f(rNode.maxX, rNode.maxY, rNode.maxZ));
                if (rDist < tMax && stackPtr < MAX_STACK_DEPTH) {
                    stack[stackPtr] = rightChild;
                    stackPtr++;
                }
            }
        }
    }

    return false;
}
`;

const REFLECTION_SHADER = /* wgsl */ `
${BODY_STRUCT_WGSL}
${TREE_NODE_STRUCT_WGSL}
${LEAF_FLAG_WGSL}
${LEAF_FUNCTIONS_WGSL}
${AABB_SENTINEL_WGSL}
${QUAT_WGSL}
${SHAPE_CONSTS_WGSL}

struct ReflectionParams {
    listenerX: f32,
    listenerY: f32,
    listenerZ: f32,
    sourceCount: u32,
    bodyCount: u32,
    frameCounter: u32,
    _pad0: u32,
    _pad1: u32,
}

struct AcousticSource {
    posX: f32,
    posY: f32,
    posZ: f32,
    voiceSlot: u32,
}

const NUM_RAYS: u32 = ${NUM_RAYS}u;
const MAX_BOUNCES: u32 = ${MAX_BOUNCES}u;
const BIN_DURATION_MS: f32 = 10.0;
const NUM_BINS: u32 = ${NUM_BINS}u;
const SPEED_OF_SOUND: f32 = 340.0;
const FIXED_POINT_SCALE: f32 = 1000000.0;

@group(0) @binding(0) var<uniform> params: ReflectionParams;
@group(0) @binding(1) var<storage, read> sources: array<AcousticSource>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@group(1) @binding(0) var<storage, read> treeNodes: array<TreeNode>;
@group(1) @binding(1) var<storage, read> sortedIds: array<u32>;
@group(1) @binding(2) var<storage, read> bodies: array<Body>;

@group(2) @binding(0) var<storage, read> bodyEids: array<u32>;
@group(2) @binding(1) var<storage, read> materialProps: array<f32>;

${CLOSEST_HIT_WGSL}
${TRACE_CLOSEST_WGSL}

fn getAbsorption3(bodyIdx: u32) -> vec3f {
    let eid = bodyEids[bodyIdx];
    let base = eid * 8u;
    return vec3f(materialProps[base], materialProps[base + 1u], materialProps[base + 2u]);
}

fn getScattering(bodyIdx: u32) -> f32 {
    let eid = bodyEids[bodyIdx];
    return materialProps[eid * 8u + 3u];
}

fn pcg(seed: u32) -> u32 {
    var s = seed * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn cosineHemisphere(s1: u32, s2: u32, normal: vec3f) -> vec3f {
    let u1 = f32(s1) / 4294967295.0;
    let u2 = f32(s2) / 4294967295.0;
    let r = sqrt(u1);
    let phi = 6.2831853 * u2;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - u1));
    var up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
    let tangent = normalize(cross(normal, up));
    let bitangent = cross(normal, tangent);
    return tangent * x + bitangent * y + normal * z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rayIdx = gid.x;
    if (rayIdx >= NUM_RAYS) { return; }
    if (params.sourceCount == 0u) { return; }

    let listener = vec3f(params.listenerX, params.listenerY, params.listenerZ);
    let seed = rayIdx * 1337u + params.frameCounter * 7919u;

    var rng = pcg(seed);
    let initS1 = rng;
    rng = pcg(rng);
    let initS2 = rng;
    let z = 2.0 * f32(initS1) / 4294967295.0 - 1.0;
    let phi = 6.2831853 * f32(initS2) / 4294967295.0;
    let rr = sqrt(1.0 - z * z);
    var dir = vec3f(rr * cos(phi), rr * sin(phi), z);

    var origin = listener;
    var accumDist = 0.0f;
    var accumEnergy = vec3f(1.0, 1.0, 1.0);

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let hit = traceClosest(origin, dir, 100.0);
        if (!hit.hit) { break; }

        let hitPoint = origin + dir * hit.t;
        let absorption3 = getAbsorption3(hit.bodyIdx);
        let scattering = getScattering(hit.bodyIdx);

        for (var si = 0u; si < params.sourceCount; si++) {
            let src = sources[si];
            let srcPos = vec3f(src.posX, src.posY, src.posZ);
            let toSrc = srcPos - hitPoint;
            let shadowDist = length(toSrc);
            if (shadowDist < 0.001) { continue; }

            let shadowDir = toSrc / shadowDist;
            let cosTheta = max(dot(hit.normal, shadowDir), 0.0);
            if (cosTheta < 0.001) { continue; }

            let shadowed = traceAcousticShadow(hitPoint + hit.normal * 0.01, shadowDir, shadowDist - 0.02);
            if (shadowed) { continue; }

            let totalDist = accumDist + hit.t + shadowDist;
            let directDist = length(srcPos - listener);
            let delay = totalDist / SPEED_OF_SOUND - directDist / SPEED_OF_SOUND;
            if (delay < 0.0) { continue; }
            let binIdx = u32(floor(delay * 1000.0 / BIN_DURATION_MS));
            if (binIdx >= NUM_BINS) { continue; }

            let diffuse = 0.31831 * scattering * cosTheta;
            let halfVec = normalize(shadowDir - dir);
            let cosAlpha = max(dot(halfVec, hit.normal), 0.0);
            let specular = 4.0584 * (1.0 - scattering) * pow(cosAlpha, 100.0);
            let clampedDist = max(shadowDist, 1.0);
            let distAtten = 0.07958 / (clampedDist * clampedDist);
            let energy3 = accumEnergy * (vec3f(1.0) - absorption3) * (diffuse + specular) * distAtten;
            let histBase = (si * NUM_BINS + binIdx) * 3u;
            let fixedLow = u32(energy3.x * FIXED_POINT_SCALE);
            let fixedMid = u32(energy3.y * FIXED_POINT_SCALE);
            let fixedHigh = u32(energy3.z * FIXED_POINT_SCALE);
            if (fixedLow > 0u) { atomicAdd(&histogram[histBase], fixedLow); }
            if (fixedMid > 0u) { atomicAdd(&histogram[histBase + 1u], fixedMid); }
            if (fixedHigh > 0u) { atomicAdd(&histogram[histBase + 2u], fixedHigh); }
        }

        accumEnergy *= (vec3f(1.0) - absorption3);
        accumDist += hit.t;
        origin = hitPoint + hit.normal * 0.01;
        rng = pcg(rng);
        if (f32(rng) / 4294967295.0 < scattering) {
            rng = pcg(rng);
            let h1 = rng;
            rng = pcg(rng);
            dir = cosineHemisphere(h1, rng, hit.normal);
        } else {
            dir = reflect(dir, hit.normal);
        }
    }
}
`;

const PARAMS_SIZE = 32;
const HISTOGRAM_SIZE = MAX_SOURCES * NUM_BINS * 3 * 4;

export interface ReflectionState {
    histogram: Float32Array;
    histogramReady: boolean;
    smoothed: Float32Array;
    frameCounter: number;
    generation: number;
    prevListener: Float32Array;
    prevBodyCount: number;
    fixedTick: number;
    lastDispatchTick: number;
    needsReset: boolean;
}

export function createReflectionState(): ReflectionState {
    return {
        histogram: new Float32Array(MAX_SOURCES * NUM_BINS * 3),
        histogramReady: false,
        smoothed: new Float32Array(MAX_SOURCES * NUM_BINS * 3),
        frameCounter: 0,
        generation: 0,
        prevListener: new Float32Array(3),
        prevBodyCount: 0,
        fixedTick: 0,
        lastDispatchTick: 0,
        needsReset: true,
    };
}

export function createReflectionNode(
    physics: PhysicsGPU,
    occ: OcclusionState,
    refl: ReflectionState,
): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let paramsBuffer: GPUBuffer | null = null;
    let histogramBuffer: GPUBuffer | null = null;
    let bg0: GPUBindGroup | null = null;
    let bg1: GPUBindGroup | null = null;
    let bg2: GPUBindGroup | null = null;
    let cachedTreeNodes: GPUBuffer | null = null;
    let cachedEids: GPUBuffer | null = null;
    let sourceBuffer: GPUBuffer | null = null;
    let materialPropsBuffer: GPUBuffer | null = null;
    let cachedMaterialSize = 0;
    let staging: GPUBuffer | null = null;
    let readbackPending = false;

    const paramsUpload = new ArrayBuffer(PARAMS_SIZE);
    const paramsF32 = new Float32Array(paramsUpload);
    const paramsU32 = new Uint32Array(paramsUpload);
    const invScale = 1 / (1000000 * NUM_RAYS);

    return {
        name: "acoustics-reflection",
        scope: "frame",
        inputs: [],
        outputs: [],
        execute(ctx: ExecutionContext) {
            if (occ.sourceCount === 0 || readbackPending) return;

            const { device, encoder } = ctx;
            const bodyCount = physics.bodyEids.length;

            const dx = occ.listener[0] - refl.prevListener[0];
            const dy = occ.listener[1] - refl.prevListener[1];
            const dz = occ.listener[2] - refl.prevListener[2];
            if (dx * dx + dy * dy + dz * dz > 0.01) refl.needsReset = true;
            if (bodyCount !== refl.prevBodyCount) refl.needsReset = true;

            if (refl.fixedTick - refl.lastDispatchTick < 6 && !refl.needsReset) return;
            refl.lastDispatchTick = refl.fixedTick;

            if (!pipeline) {
                pipeline = device.createComputePipeline({
                    label: "acoustics-reflection",
                    layout: "auto",
                    compute: {
                        module: device.createShaderModule({ code: REFLECTION_SHADER }),
                    },
                });
            }

            if (!paramsBuffer) {
                paramsBuffer = device.createBuffer({
                    label: "reflection-params",
                    size: PARAMS_SIZE,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
            }

            if (!sourceBuffer) {
                sourceBuffer = device.createBuffer({
                    label: "reflection-sources",
                    size: MAX_SOURCES * 4 * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
            }

            if (!histogramBuffer) {
                histogramBuffer = device.createBuffer({
                    label: "reflection-histogram",
                    size: HISTOGRAM_SIZE,
                    usage:
                        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                });
            }

            if (!staging) {
                staging = device.createBuffer({
                    label: "acoustics-refl-staging",
                    size: HISTOGRAM_SIZE,
                    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                });
            }

            const matSize = Math.max(capacity() * 8 * 4, 16);
            if (!materialPropsBuffer || cachedMaterialSize < matSize) {
                materialPropsBuffer?.destroy();
                materialPropsBuffer = device.createBuffer({
                    label: "reflection-material-props",
                    size: matSize,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
                cachedMaterialSize = matSize;
                bg2 = null;
            }
            write(device.queue, materialPropsBuffer, 0, AcousticMaterialData, capacity());

            paramsF32[0] = occ.listener[0];
            paramsF32[1] = occ.listener[1];
            paramsF32[2] = occ.listener[2];
            paramsU32[3] = occ.sourceCount;
            paramsU32[4] = bodyCount;
            paramsU32[5] = refl.frameCounter++;
            device.queue.writeBuffer(paramsBuffer, 0, paramsUpload);

            device.queue.writeBuffer(
                sourceBuffer,
                0,
                occ.sources.buffer,
                occ.sources.byteOffset,
                occ.sourceCount * 4 * 4,
            );

            encoder.clearBuffer(histogramBuffer, 0, occ.sourceCount * NUM_BINS * 3 * 4);
            if (refl.needsReset) {
                refl.needsReset = false;
                refl.prevListener[0] = occ.listener[0];
                refl.prevListener[1] = occ.listener[1];
                refl.prevListener[2] = occ.listener[2];
                refl.prevBodyCount = bodyCount;
            }

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
                        { binding: 2, resource: { buffer: histogramBuffer } },
                    ],
                });
                bg1 = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: treeNodesBuffer } },
                        {
                            binding: 1,
                            resource: { buffer: physics.lbvh.lbvh.sortedIds },
                        },
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

            const pass = beginComputePass(encoder, ctx.timestampWrites?.("acoustics-reflection"));
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg0);
            pass.setBindGroup(1, bg1);
            pass.setBindGroup(2, bg2);
            pass.dispatchWorkgroups(Math.ceil(NUM_RAYS / 64));
            pass.end();

            const readbackSize = occ.sourceCount * NUM_BINS * 3 * 4;
            encoder.copyBufferToBuffer(histogramBuffer, 0, staging, 0, readbackSize);

            const count = occ.sourceCount;
            const gen = ++refl.generation;
            readbackPending = true;
            ctx.afterSubmit(async () => {
                try {
                    await staging!.mapAsync(GPUMapMode.READ);
                    if (refl.generation === gen) {
                        const len = count * NUM_BINS * 3;
                        const src = new Uint32Array(staging!.getMappedRange(), 0, len);
                        const dest = refl.histogram;
                        for (let i = 0; i < len; i++) {
                            dest[i] = src[i] * invScale;
                        }
                        refl.histogramReady = true;
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
