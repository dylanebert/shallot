import {
    hasProperties,
    instanceStructWGSL,
    instanceBindingWGSL,
    SKY_DIR_WGSL,
    compileSurfaceBlock,
    SURFACE_DATA_STRUCT_WGSL,
    SCENE_STRUCT_WGSL,
    SKY_STRUCT_WGSL,
    DATA_STRUCT_WGSL,
    OKLAB_WGSL,
    POINT_LIGHT_STRUCT_WGSL,
    type SurfaceData,
    SURFACE_HELPERS_WGSL,
} from "../../standard/render/core";
import {
    BVH_STRUCTS,
    TLAS_BLAS_STRUCTS,
    TLAS_BLAS_TRAVERSAL,
    BVH_UTILS_WGSL,
    BLAS_SHADOW_WGSL,
    ANALYTIC_SHADOW_WGSL,
    TLAS_BLAS_SHADOW,
} from "./bvh/traverse";
import { RAY_STRUCT_WGSL, HIT_RESULT_STRUCT_WGSL, RAY_EPSILON } from "./bvh/structs";
import { WAVEFRONT_RAY_STRUCT_WGSL, WAVEFRONT_HIT_STRUCT_WGSL, READ_RAY_WGSL } from "./buffers";

function rtInstanceWGSL(): string {
    return hasProperties() ? `${instanceStructWGSL()}\n${instanceBindingWGSL(6)}` : "";
}

const WF_SCENE_BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> data: array<Data>;
@group(0) @binding(2) var<uniform> sky: Sky;
@group(0) @binding(3) var<storage, read> matrices: array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read> pointLights: array<PointLightData>;
`;

const WF_DATA_UTILS = /* wgsl */ `
fn getData(eid: u32) -> Data {
    return data[eid];
}

fn getPrimitive(eid: u32) -> u32 {
    return (data[eid].flags >> 13u) & 7u;
}

fn getInstanceCount() -> u32 {
    return scene.instanceCount;
}

fn toObjectSpace(wp: vec3<f32>, eid: u32) -> vec3<f32> {
    let m = matrices[eid];
    let p = wp - m[3].xyz;
    return vec3(dot(p, m[0].xyz), dot(p, m[1].xyz), dot(p, m[2].xyz));
}

fn computeHitUV(objectPos: vec3<f32>, objectNormal: vec3<f32>, size: vec3<f32>) -> vec2<f32> {
    let p = objectPos / size;
    let a = abs(objectNormal);
    if (a.x > a.y && a.x > a.z) {
        let flip = select(1.0, -1.0, objectNormal.x > 0.0);
        return vec2(flip * p.z + 0.5, p.y + 0.5);
    } else if (a.y > a.z) {
        let flip = select(1.0, -1.0, objectNormal.y > 0.0);
        return vec2(p.x + 0.5, flip * p.z + 0.5);
    }
    let flip = select(1.0, -1.0, objectNormal.z < 0.0);
    return vec2(flip * p.x + 0.5, p.y + 0.5);
}
`;

export function compileRaygenShader(): string {
    return /* wgsl */ `
${SCENE_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> scene: Scene;

${WAVEFRONT_RAY_STRUCT_WGSL}
@group(1) @binding(0) var<storage, read_write> rayBufferOut: array<u32>;
@group(1) @binding(1) var<storage, read_write> counters: array<atomic<u32>>;

${SKY_DIR_WGSL}

struct PrimaryRay {
    origin: vec3<f32>,
    direction: vec3<f32>,
    skyDir: vec3<f32>,
}

fn generateRay(screenX: f32, screenY: f32) -> PrimaryRay {
    var result: PrimaryRay;
    result.skyDir = computeSkyDir(screenX, screenY);

    let width = scene.viewport.x;
    let height = scene.viewport.y;
    let ndcX = screenX * 2.0 - 1.0;
    let ndcY = 1.0 - screenY * 2.0;
    let aspect = width / height;

    let cameraWorld = scene.cameraWorld;
    let camPosX = cameraWorld[3][0];
    let camPosY = cameraWorld[3][1];
    let camPosZ = cameraWorld[3][2];

    if (scene.cameraMode > 0.5) {
        let r00 = cameraWorld[0][0]; let r10 = cameraWorld[0][1]; let r20 = cameraWorld[0][2];
        let r01 = cameraWorld[1][0]; let r11 = cameraWorld[1][1]; let r21 = cameraWorld[1][2];
        let r02 = cameraWorld[2][0]; let r12 = cameraWorld[2][1]; let r22 = cameraWorld[2][2];

        let halfHeight = scene.cameraSize;
        let halfWidth = halfHeight * aspect;
        let offsetX = ndcX * halfWidth;
        let offsetY = ndcY * halfHeight;
        let fwdX = -r02; let fwdY = -r12; let fwdZ = -r22;

        result.origin = vec3(
            camPosX + r00 * offsetX + r01 * offsetY + fwdX * scene.near,
            camPosY + r10 * offsetX + r11 * offsetY + fwdY * scene.near,
            camPosZ + r20 * offsetX + r21 * offsetY + fwdZ * scene.near
        );
        result.direction = vec3(fwdX, fwdY, fwdZ);
    } else {
        let dir = result.skyDir;
        result.origin = vec3(camPosX + dir.x * scene.near, camPosY + dir.y * scene.near, camPosZ + dir.z * scene.near);
        result.direction = dir;
    }

    return result;
}

fn writeRayDirect(idx: u32, ray: WavefrontRay) {
    let base = idx * RAY_STRIDE_U32;
    rayBufferOut[base] = bitcast<u32>(ray.origin.x);
    rayBufferOut[base + 1u] = bitcast<u32>(ray.origin.y);
    rayBufferOut[base + 2u] = bitcast<u32>(ray.origin.z);
    rayBufferOut[base + 3u] = ray.pixelIndex;
    rayBufferOut[base + 4u] = bitcast<u32>(ray.direction.x);
    rayBufferOut[base + 5u] = bitcast<u32>(ray.direction.y);
    rayBufferOut[base + 6u] = bitcast<u32>(ray.direction.z);
    rayBufferOut[base + 7u] = ray.flags;
    rayBufferOut[base + 8u] = bitcast<u32>(ray.throughput.x);
    rayBufferOut[base + 9u] = bitcast<u32>(ray.throughput.y);
    rayBufferOut[base + 10u] = bitcast<u32>(ray.throughput.z);
    rayBufferOut[base + 11u] = 0u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(scene.viewport.x);
    let height = u32(scene.viewport.y);
    let pixelCount = width * height;

    let idx = gid.x;
    if (idx >= pixelCount) { return; }

    if (idx == 0u) {
        atomicStore(&counters[0], pixelCount);
    }

    let x = idx % width;
    let y = idx / width;

    let screenX = (f32(x) + 0.5) / f32(width);
    let screenY = (f32(y) + 0.5) / f32(height);

    let primary = generateRay(screenX, screenY);

    var ray: WavefrontRay;
    ray.origin = primary.origin;
    ray.pixelIndex = packPixel(x, y);
    ray.direction = primary.direction;
    ray.flags = 0u;
    ray.throughput = vec3(1.0);

    writeRayDirect(idx, ray);
}
`;
}

const TRACE_COMMON_WGSL = /* wgsl */ `
${RAY_STRUCT_WGSL}
${HIT_RESULT_STRUCT_WGSL}
${SCENE_STRUCT_WGSL}
${DATA_STRUCT_WGSL}
${BVH_STRUCTS}
${TLAS_BLAS_STRUCTS}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> data: array<Data>;

${WAVEFRONT_RAY_STRUCT_WGSL}
${WAVEFRONT_HIT_STRUCT_WGSL}

@group(1) @binding(0) var<storage, read> rayBuffer: array<u32>;
@group(1) @binding(1) var<storage, read_write> hitBuffer: array<u32>;
@group(1) @binding(2) var<storage, read> counters: array<u32>;

@group(2) @binding(0) var<storage, read> tlasNodes: array<BVHNode>;
@group(2) @binding(1) var<storage, read> blasNodes: array<BLASNode>;
@group(2) @binding(2) var<storage, read> blasTriIds: array<u32>;
@group(2) @binding(3) var<storage, read> blasTriangles: array<BLASTriangle>;
@group(2) @binding(4) var<storage, read> entityBlasMeta: array<u32>;
@group(2) @binding(5) var<storage, read> instanceInverses: array<mat4x4<f32>>;

fn getData(eid: u32) -> Data {
    return data[eid];
}

fn getPrimitive(eid: u32) -> u32 {
    return (data[eid].flags >> 13u) & 7u;
}

fn getInstanceCount() -> u32 {
    return scene.instanceCount;
}

const EPSILON: f32 = ${RAY_EPSILON};
`;

const WRITE_HIT_WGSL = /* wgsl */ `
fn writeHitLocal(idx: u32, hit: WavefrontHit) {
    let base = idx * 8u;
    hitBuffer[base] = bitcast<u32>(hit.normal.x);
    hitBuffer[base + 1u] = bitcast<u32>(hit.normal.y);
    hitBuffer[base + 2u] = bitcast<u32>(hit.normal.z);
    hitBuffer[base + 3u] = bitcast<u32>(hit.t);
    hitBuffer[base + 4u] = hit.entityId;
    hitBuffer[base + 5u] = bitcast<u32>(hit.u);
    hitBuffer[base + 6u] = bitcast<u32>(hit.v);
    hitBuffer[base + 7u] = 0u;
}
`;

export function compileClosestHitShader(): string {
    return /* wgsl */ `
${TRACE_COMMON_WGSL}
${TLAS_BLAS_TRAVERSAL}
${READ_RAY_WGSL}
${WRITE_HIT_WGSL}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rayCount = counters[0];
    let idx = gid.x;
    if (idx >= rayCount) { return; }

    let wfRay = readRayLocal(idx);

    var ray: Ray;
    ray.origin = wfRay.origin;
    ray.direction = wfRay.direction;

    let hit = trace(ray);

    var wfHit: WavefrontHit;
    if (hit.hit) {
        wfHit.normal = hit.normal;
        wfHit.t = hit.t;
        wfHit.entityId = hit.entityId;
        wfHit.u = hit.u;
        wfHit.v = hit.v;
    } else {
        wfHit.t = -1.0;
        wfHit.entityId = 0u;
        wfHit.normal = vec3(0.0);
        wfHit.u = 0.0;
        wfHit.v = 0.0;
    }
    wfHit._pad = 0u;
    writeHitLocal(idx, wfHit);
}
`;
}

export function compileAnyHitShader(): string {
    return /* wgsl */ `
${TRACE_COMMON_WGSL}
${BVH_UTILS_WGSL}
${ANALYTIC_SHADOW_WGSL}
${BLAS_SHADOW_WGSL}
${TLAS_BLAS_SHADOW}
${READ_RAY_WGSL}
${WRITE_HIT_WGSL}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rayCount = counters[0];
    let idx = gid.x;
    if (idx >= rayCount) { return; }

    let wfRay = readRayLocal(idx);

    var ray: Ray;
    ray.origin = wfRay.origin;
    ray.direction = wfRay.direction;

    let tMax = unpackShadowTMax(wfRay.flags);
    let occluded = traceShadowAnyHit(ray, tMax);

    var shadowHit: WavefrontHit;
    shadowHit.t = select(0.0, -1.0, occluded);
    shadowHit.entityId = 0u;
    shadowHit.normal = vec3(0.0);
    shadowHit.u = 0.0;
    shadowHit.v = 0.0;
    shadowHit._pad = 0u;
    writeHitLocal(idx, shadowHit);
}
`;
}

export function compileShadeShader(surfaces: SurfaceData[]): string {
    const surfaceBlock = compileSurfaceBlock(surfaces);

    return /* wgsl */ `
${SURFACE_DATA_STRUCT_WGSL}
${SCENE_STRUCT_WGSL}
${SKY_STRUCT_WGSL}
${DATA_STRUCT_WGSL}
${POINT_LIGHT_STRUCT_WGSL}

${WF_SCENE_BINDINGS}
${rtInstanceWGSL()}

${WAVEFRONT_RAY_STRUCT_WGSL}
${WAVEFRONT_HIT_STRUCT_WGSL}

// group 1: 6 storage + group 0's 3 storage = 9 total (limit 10)
@group(1) @binding(0) var<storage, read> rayBuffer: array<u32>;
@group(1) @binding(1) var<storage, read> hitBuffer: array<u32>;
@group(1) @binding(2) var<storage, read_write> pixelState: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> rayBufferOut: array<u32>;
@group(1) @binding(4) var<storage, read_write> shadowRayBuffer: array<u32>;
@group(1) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;

var<private> wfPixelCount: u32;

fn initPixelCount() {
    wfPixelCount = u32(scene.viewport.x) * u32(scene.viewport.y);
}

fn accumOffset(pixelIdx: u32) -> u32 { return pixelIdx * 4u; }
fn firstHitOffset(pixelIdx: u32) -> u32 { return wfPixelCount * 4u + pixelIdx * 2u; }

fn atomicAddF32(offset: u32, val: f32) {
    if (val == 0.0) { return; }
    var old = atomicLoad(&pixelState[offset]);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old) + val);
        let result = atomicCompareExchangeWeak(&pixelState[offset], old, new_val);
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}
var<private> localAccum: vec4<f32>;
var<private> isBounceZero: bool;

fn addAccum(pixelIdx: u32, v: vec4<f32>) {
    if (isBounceZero) {
        localAccum += v;
    } else {
        let o = accumOffset(pixelIdx);
        atomicAddF32(o, v.x);
        atomicAddF32(o + 1u, v.y);
        atomicAddF32(o + 2u, v.z);
        atomicAddF32(o + 3u, v.w);
    }
}

fn flushAccum(pixelIdx: u32) {
    if (!isBounceZero) { return; }
    let o = accumOffset(pixelIdx);
    atomicStore(&pixelState[o], bitcast<u32>(localAccum.x));
    atomicStore(&pixelState[o + 1u], bitcast<u32>(localAccum.y));
    atomicStore(&pixelState[o + 2u], bitcast<u32>(localAccum.z));
    atomicStore(&pixelState[o + 3u], bitcast<u32>(localAccum.w));
}

fn writeFirstHit(pixelIdx: u32, v: vec2<u32>) {
    let o = firstHitOffset(pixelIdx);
    atomicStore(&pixelState[o], v.x); atomicStore(&pixelState[o+1u], v.y);
}

const COUNTER_INPUT: u32 = 0u;
const COUNTER_OUTPUT: u32 = 1u;
const COUNTER_SHADOW: u32 = 2u;
const DBG_RAY_OVERFLOW: u32 = 4u;
const DBG_SHADOW_OVERFLOW: u32 = 5u;
const DBG_SHADE_PIXEL_OOB: u32 = 8u;
const DBG_MAX_OUTPUT: u32 = 6u;
const DBG_MAX_SHADOW: u32 = 7u;

${WF_DATA_UTILS}
${SURFACE_HELPERS_WGSL}
${OKLAB_WGSL}
${READ_RAY_WGSL}

fn readHitLocal(idx: u32) -> WavefrontHit {
    let base = idx * 8u;
    var h: WavefrontHit;
    h.normal = vec3(bitcast<f32>(hitBuffer[base]), bitcast<f32>(hitBuffer[base + 1u]), bitcast<f32>(hitBuffer[base + 2u]));
    h.t = bitcast<f32>(hitBuffer[base + 3u]);
    h.entityId = hitBuffer[base + 4u];
    h.u = bitcast<f32>(hitBuffer[base + 5u]);
    h.v = bitcast<f32>(hitBuffer[base + 6u]);
    h._pad = 0u;
    return h;
}

fn writeRayOut(ray: WavefrontRay) {
    let maxBounceRays = wfPixelCount * 4u;
    let idx = atomicAdd(&counters[COUNTER_OUTPUT], 1u);
    if (idx >= maxBounceRays) {
        atomicAdd(&counters[DBG_RAY_OVERFLOW], 1u);
        return;
    }
    atomicMax(&counters[DBG_MAX_OUTPUT], idx + 1u);
    let base = idx * RAY_STRIDE_U32;
    rayBufferOut[base] = bitcast<u32>(ray.origin.x);
    rayBufferOut[base + 1u] = bitcast<u32>(ray.origin.y);
    rayBufferOut[base + 2u] = bitcast<u32>(ray.origin.z);
    rayBufferOut[base + 3u] = ray.pixelIndex;
    rayBufferOut[base + 4u] = bitcast<u32>(ray.direction.x);
    rayBufferOut[base + 5u] = bitcast<u32>(ray.direction.y);
    rayBufferOut[base + 6u] = bitcast<u32>(ray.direction.z);
    rayBufferOut[base + 7u] = ray.flags;
    rayBufferOut[base + 8u] = bitcast<u32>(ray.throughput.x);
    rayBufferOut[base + 9u] = bitcast<u32>(ray.throughput.y);
    rayBufferOut[base + 10u] = bitcast<u32>(ray.throughput.z);
    rayBufferOut[base + 11u] = 0u;
}

fn emitShadowRay(origin: vec3<f32>, direction: vec3<f32>, pixelIndex: u32, tMax: f32, contrib: vec3<f32>) {
    let maxShadowRays = wfPixelCount * 5u;
    let idx = atomicAdd(&counters[COUNTER_SHADOW], 1u);
    if (idx >= maxShadowRays) {
        atomicAdd(&counters[DBG_SHADOW_OVERFLOW], 1u);
        return;
    }
    atomicMax(&counters[DBG_MAX_SHADOW], idx + 1u);
    let base = idx * RAY_STRIDE_U32;
    shadowRayBuffer[base] = bitcast<u32>(origin.x);
    shadowRayBuffer[base + 1u] = bitcast<u32>(origin.y);
    shadowRayBuffer[base + 2u] = bitcast<u32>(origin.z);
    shadowRayBuffer[base + 3u] = pixelIndex;
    shadowRayBuffer[base + 4u] = bitcast<u32>(direction.x);
    shadowRayBuffer[base + 5u] = bitcast<u32>(direction.y);
    shadowRayBuffer[base + 6u] = bitcast<u32>(direction.z);
    shadowRayBuffer[base + 7u] = packShadowFlags(tMax);
    shadowRayBuffer[base + 8u] = bitcast<u32>(contrib.x);
    shadowRayBuffer[base + 9u] = bitcast<u32>(contrib.y);
    shadowRayBuffer[base + 10u] = bitcast<u32>(contrib.z);
    shadowRayBuffer[base + 11u] = 0u;
}

const PI: f32 = 3.14159265359;
const MIN_CONTRIBUTION: f32 = 0.02;
const REFLECTION_EPSILON: f32 = 0.001;
const TRANSPARENCY_EPSILON: f32 = 0.001;

const KIND_PRIMARY: u32 = 0u;
const KIND_REFLECT: u32 = 2u;

${surfaceBlock}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    initPixelCount();
    let count = min(atomicLoad(&counters[COUNTER_INPUT]), wfPixelCount * 4u);
    let idx = gid.x;
    if (idx >= count) { return; }

    let wfRay = readRayLocal(idx);
    let hit = readHitLocal(idx);
    let bounceCount = rayBounceCount(wfRay.flags);
    isBounceZero = bounceCount == 0u;
    localAccum = vec4(0.0);
    let px = unpackPixelX(wfRay.pixelIndex);
    let py = unpackPixelY(wfRay.pixelIndex);
    let width = u32(scene.viewport.x);
    let pixelIdx = py * width + px;

    if (pixelIdx >= wfPixelCount) {
        atomicAdd(&counters[DBG_SHADE_PIXEL_OOB], 1u);
        return;
    }

    var tp0 = wfRay.throughput;

    if (hit.t < 0.0) {
        let skyColor = sampleSky(wfRay.direction);
        addAccum(pixelIdx, vec4(skyColor * tp0, 0.0));
        flushAccum(pixelIdx);
        return;
    }

    if (max(tp0.x, max(tp0.y, tp0.z)) < MIN_CONTRIBUTION) {
        return;
    }

    let eid = hit.entityId;
    let d = getData(eid);
    let surfaceId = d.flags & 0xFFu;

    let rayOrigin = wfRay.origin + wfRay.direction * hit.t;

    var surface: SurfaceData;
    surface.worldPos = rayOrigin;
    surface.objectPos = toObjectSpace(rayOrigin, eid);
    surface.worldNormal = hit.normal;
    surface.objectNormal = normalize(vec3(
        dot(hit.normal, matrices[eid][0].xyz),
        dot(hit.normal, matrices[eid][1].xyz),
        dot(hit.normal, matrices[eid][2].xyz)
    ));
    surface.uv = computeHitUV(surface.objectPos, surface.objectNormal, vec3(d.sizeX, d.sizeY, d.sizeZ));
    surface.baseColor = d.baseColor.rgb;
    surface.emission = d.emission.rgb * d.emission.a;
    surface.roughness = d.pbr.x;
    surface.reflectivity = d.pbr.y;
    surface.opacity = d.baseColor.a;

    dispatchFragment(surfaceId, &surface, vec4(f32(px), f32(py), hit.t, 1.0), eid);

    let tp = tp0;
    let opacity = surface.opacity;

    if (bounceCount == 0u) {
        if (opacity >= 0.1) {
            writeFirstHit(pixelIdx, vec2(bitcast<u32>(hit.t), eid));
        }
    } else {
        let o = firstHitOffset(pixelIdx);
        atomicCompareExchangeWeak(&pixelState[o], 0u, bitcast<u32>(hit.t));
        atomicCompareExchangeWeak(&pixelState[o + 1u], 0u, eid);
    }
    let reflectivity = surface.reflectivity;
    let V = -wfRay.direction;

    let ambient = scene.ambientColor.rgb * scene.ambientColor.a;
    let ambientColor = surface.baseColor * ambient + surface.emission;

    addAccum(pixelIdx, vec4(ambientColor * tp * opacity, 0.0));

    let L = -scene.sunDirection.xyz;
    let NdotL = max(dot(surface.worldNormal, L), 0.0);
    let sunDiffuse = scene.sunColor.rgb * NdotL;
    let specTerm = blinnPhongSpecular(surface.worldNormal, L, V, surface.roughness);
    let sunSpecular = scene.sunColor.rgb * specTerm * NdotL * reflectivity;
    let directSun = surface.baseColor * sunDiffuse + sunSpecular;

    let shadowOrigin = surface.worldPos + surface.worldNormal * 0.02;

    let sunContrib = directSun * tp * opacity;
    if (NdotL > 0.0) {
        if (scene.shadowSamples > 0u) {
            emitShadowRay(shadowOrigin, L, wfRay.pixelIndex, 1000.0, sunContrib);
        } else {
            addAccum(pixelIdx, vec4(sunContrib, 0.0));
        }
    }

    let plCount = scene.pointLightCount;
    for (var pli = 0u; pli < plCount; pli++) {
        if (pli >= 4u) { break; }
        let light = pointLights[pli];
        let toLight = light.position - surface.worldPos;
        let dist = length(toLight);
        if (dist >= light.radius) { continue; }
        let plL = toLight / dist;
        let plNdotL = max(dot(surface.worldNormal, plL), 0.0);
        if (plNdotL <= 0.0) { continue; }
        let ratio = 1.0 - dist / light.radius;
        let attenuation = ratio * ratio;
        let plDiffuse = light.color * plNdotL * attenuation;
        let plSpec = blinnPhongSpecular(surface.worldNormal, plL, V, surface.roughness) * plNdotL * attenuation * reflectivity;
        let plContrib = (surface.baseColor * plDiffuse + light.color * plSpec) * tp * opacity;
        if (light.shadowIdx >= 0.0) {
            emitShadowRay(shadowOrigin, plL, wfRay.pixelIndex, dist, plContrib);
        } else {
            addAccum(pixelIdx, vec4(plContrib, 0.0));
        }
    }

    if (opacity < 1.0 && bounceCount < 4u) {
        let nextTp = tp * (1.0 - opacity) * surface.baseColor;
        let avgTp = (nextTp.x + nextTp.y + nextTp.z) / 3.0;
        if (avgTp >= MIN_CONTRIBUTION) {
            var bounceRay: WavefrontRay;
            bounceRay.origin = rayOrigin + wfRay.direction * TRANSPARENCY_EPSILON;
            bounceRay.direction = wfRay.direction;
            bounceRay.pixelIndex = wfRay.pixelIndex;
            bounceRay.flags = packRayFlags(KIND_PRIMARY, bounceCount + 1u);
            bounceRay.throughput = nextTp;
            writeRayOut(bounceRay);
        }
    }

    if (scene.reflectionEnabled > 0u && reflectivity > MIN_CONTRIBUTION && bounceCount < 4u) {
        let smoothness = 1.0 - surface.roughness;
        let roughnessAtten = smoothness * smoothness;
        let reflWeight = reflectivity * roughnessAtten;

        if (reflWeight >= MIN_CONTRIBUTION) {
            var reflRay: WavefrontRay;
            reflRay.origin = rayOrigin + surface.worldNormal * REFLECTION_EPSILON;
            reflRay.direction = reflect(wfRay.direction, surface.worldNormal);
            reflRay.pixelIndex = wfRay.pixelIndex;
            reflRay.flags = packRayFlags(KIND_REFLECT, bounceCount + 1u);
            reflRay.throughput = tp * reflWeight;
            writeRayOut(reflRay);
        }
    }

    flushAccum(pixelIdx);
}
`;
}

export function compileResolveShader(): string {
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read> pixelState: array<u32>;
@group(0) @binding(1) var output_scene: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var output_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var output_entityId: texture_storage_2d<r32uint, write>;

struct ResolveParams {
    width: u32,
    height: u32,
}
@group(0) @binding(4) var<uniform> resolveParams: ResolveParams;

${SCENE_STRUCT_WGSL}
@group(0) @binding(5) var<uniform> scene: Scene;

${SKY_STRUCT_WGSL}
@group(0) @binding(6) var<uniform> sky: Sky;

fn applyHaze(color: vec3<f32>, dist: f32) -> vec3<f32> {
    if (sky.hazeDensity <= 0.0) { return color; }
    let haze = 1.0 - exp(-sky.hazeDensity * dist);
    return mix(color, sky.hazeColor.rgb, haze);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixelCount = resolveParams.width * resolveParams.height;
    let idx = gid.x;
    if (idx >= pixelCount) { return; }

    let x = idx % resolveParams.width;
    let y = idx / resolveParams.width;
    let coord = vec2<i32>(i32(x), i32(y));

    let ao = idx * 4u;
    let color = vec4(bitcast<f32>(pixelState[ao]), bitcast<f32>(pixelState[ao+1u]), bitcast<f32>(pixelState[ao+2u]), bitcast<f32>(pixelState[ao+3u]));

    let fo = pixelCount * 4u + idx * 2u;
    let rawDepth = bitcast<f32>(pixelState[fo]);
    let entityId = pixelState[fo + 1u];

    var finalColor = clamp(color.rgb, vec3(0.0), vec3(10.0));
    if (rawDepth > 0.0 && rawDepth <= scene.far) {
        finalColor = applyHaze(finalColor, rawDepth);
    }

    var ndcDepth = 1.0;
    if (rawDepth <= scene.far) {
        if (scene.cameraMode > 0.5) {
            ndcDepth = (rawDepth - scene.near) / (scene.far - scene.near);
        } else {
            ndcDepth = (scene.far * (rawDepth - scene.near)) / (rawDepth * (scene.far - scene.near));
        }
        ndcDepth = clamp(ndcDepth, 0.0, 1.0);
    }

    textureStore(output_scene, coord, vec4(finalColor, 1.0));
    textureStore(output_depth, coord, vec4(ndcDepth, 0.0, 0.0, 0.0));
    textureStore(output_entityId, coord, vec4(entityId, 0u, 0u, 0u));
}
`;
}

export function compileSwapCounterShader(): string {
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> counters: array<u32>;
@group(0) @binding(1) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

@compute @workgroup_size(1)
fn main() {
    let raw = counters[2];
    let maxShadow = params.x * params.y * 5u;
    let count = min(raw, maxShadow);
    if (raw > maxShadow) {
        counters[11] = raw;
    }
    counters[0] = count;
    counters[2] = 0u;
    indirect[0] = (count + 255u) / 256u;
    indirect[1] = 1u;
    indirect[2] = 1u;
}
`;
}

export function compileSwapBounceCounterShader(): string {
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> counters: array<u32>;

struct Params { width: u32, height: u32 }
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> indirect: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let raw = counters[1];
    let limit = params.width * params.height * 4u;
    let count = min(raw, limit);
    if (raw > limit) {
        counters[10] = raw;
    }
    counters[0] = count;
    counters[1] = 0u;
    counters[2] = 0u;
    indirect[0] = (count + 255u) / 256u;
    indirect[1] = 1u;
    indirect[2] = 1u;
}
`;
}

export function compileApplyShadowShader(): string {
    return /* wgsl */ `
${WAVEFRONT_RAY_STRUCT_WGSL}
@group(0) @binding(0) var<storage, read> shadowHits: array<u32>;
@group(0) @binding(1) var<storage, read> shadowRays: array<u32>;
@group(0) @binding(2) var<storage, read_write> pixelState: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> counters: array<u32>;

struct Params {
    width: u32,
    height: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

fn atomicAddF32(offset: u32, val: f32) {
    if (val == 0.0) { return; }
    var old = atomicLoad(&pixelState[offset]);
    loop {
        let new_val = bitcast<u32>(bitcast<f32>(old) + val);
        let result = atomicCompareExchangeWeak(&pixelState[offset], old, new_val);
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let count = counters[0];
    let idx = gid.x;
    if (idx >= count) { return; }

    let hitBase = idx * 8u;
    let hitT = bitcast<f32>(shadowHits[hitBase + 3u]);
    let shadow = select(1.0, 0.0, hitT < 0.0);

    let rayBase = idx * RAY_STRIDE_U32;
    let pixelIndex = shadowRays[rayBase + 3u];
    let px = pixelIndex & 0xFFFFu;
    let py = pixelIndex >> 16u;
    let pixelIdx = py * params.width + px;

    let pc = params.width * params.height;
    if (pixelIdx >= pc) {
        return;
    }

    let contribX = bitcast<f32>(shadowRays[rayBase + 8u]);
    let contribY = bitcast<f32>(shadowRays[rayBase + 9u]);
    let contribZ = bitcast<f32>(shadowRays[rayBase + 10u]);

    let ao = pixelIdx * 4u;
    atomicAddF32(ao, contribX * shadow);
    atomicAddF32(ao + 1u, contribY * shadow);
    atomicAddF32(ao + 2u, contribZ * shadow);
}
`;
}

export function compileClearPixelStateShader(): string {
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> pixelState: array<u32>;

struct ClearParams {
    width: u32,
    height: u32,
}
@group(0) @binding(1) var<uniform> params: ClearParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let pc = params.width * params.height;
    if (idx >= pc) { return; }

    let ao = idx * 4u;
    pixelState[ao] = 0u;
    pixelState[ao + 1u] = 0u;
    pixelState[ao + 2u] = 0u;
    pixelState[ao + 3u] = 0u;

    let fo = pc * 4u + idx * 2u;
    pixelState[fo] = bitcast<u32>(1e30);
    pixelState[fo + 1u] = 0u;
}
`;
}
