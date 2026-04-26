import { beginComputePass, type ComputeNode, type ExecutionContext } from "../compute";
import { POINT_LIGHT_STRUCT_WGSL } from "../render/core";
import { Camera } from "../render";
import { WorldTransform } from "../transforms";
import { invert } from "../../engine";

const TILE_SIZE = 16;
const DEPTH_SLICES = 24;
const MAX_LIGHTS_PER_CLUSTER = 128;
const MAX_LIGHT_INDICES = 1024 * 1024;

const CLUSTER_PARAMS_SIZE = 128;

const CLUSTER_CULL_WGSL = /* wgsl */ `
${POINT_LIGHT_STRUCT_WGSL}

struct ClusterParams {
    viewMatrix: mat4x4<f32>,
    tilesX: u32,
    tilesY: u32,
    sliceCount: u32,
    lightCount: u32,
    near: f32,
    far: f32,
    logRatio: f32,
    bias: f32,
    tanHalfFov: f32,
    aspect: f32,
    cameraMode: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: ClusterParams;
@group(0) @binding(1) var<storage, read> pointLights: array<PointLightData>;
@group(0) @binding(2) var<storage, read_write> clusterGrid: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> lightIndices: array<atomic<u32>>;

fn clusterAABB(tileX: u32, tileY: u32, slice: u32) -> array<vec3<f32>, 2> {
    let tilesXf = f32(params.tilesX);
    let tilesYf = f32(params.tilesY);

    let minXNdc = f32(tileX) / tilesXf * 2.0 - 1.0;
    let maxXNdc = f32(tileX + 1u) / tilesXf * 2.0 - 1.0;
    let minYNdc = 1.0 - f32(tileY + 1u) / tilesYf * 2.0;
    let maxYNdc = 1.0 - f32(tileY) / tilesYf * 2.0;

    var nearZ: f32;
    var farZ: f32;

    if (params.cameraMode > 0.5) {
        nearZ = params.near + f32(slice) / f32(params.sliceCount) * (params.far - params.near);
        farZ = params.near + f32(slice + 1u) / f32(params.sliceCount) * (params.far - params.near);
    } else {
        nearZ = params.near * pow(params.far / params.near, f32(slice) / f32(params.sliceCount));
        farZ = params.near * pow(params.far / params.near, f32(slice + 1u) / f32(params.sliceCount));
    }

    var minPt: vec3<f32>;
    var maxPt: vec3<f32>;

    if (params.cameraMode > 0.5) {
        let halfW = params.tanHalfFov * params.aspect;
        let halfH = params.tanHalfFov;
        minPt = vec3(minXNdc * halfW, minYNdc * halfH, nearZ);
        maxPt = vec3(maxXNdc * halfW, maxYNdc * halfH, farZ);
    } else {
        let nearHalfH = params.tanHalfFov * nearZ;
        let nearHalfW = nearHalfH * params.aspect;
        let farHalfH = params.tanHalfFov * farZ;
        let farHalfW = farHalfH * params.aspect;

        minPt = vec3(
            min(minXNdc * nearHalfW, minXNdc * farHalfW),
            min(minYNdc * nearHalfH, minYNdc * farHalfH),
            nearZ
        );
        maxPt = vec3(
            max(maxXNdc * nearHalfW, maxXNdc * farHalfW),
            max(maxYNdc * nearHalfH, maxYNdc * farHalfH),
            farZ
        );
    }

    return array<vec3<f32>, 2>(minPt, maxPt);
}

fn sphereAABBIntersect(center: vec3<f32>, radius: f32, aabbMin: vec3<f32>, aabbMax: vec3<f32>) -> bool {
    let closest = clamp(center, aabbMin, aabbMax);
    let d = center - closest;
    return dot(d, d) <= radius * radius;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tilesX = params.tilesX;
    let tilesY = params.tilesY;
    let sliceCount = params.sliceCount;
    let clusterId = gid.x;
    let totalClusters = tilesX * tilesY * sliceCount;

    if (clusterId >= totalClusters) { return; }

    let slice = clusterId / (tilesX * tilesY);
    let rem = clusterId % (tilesX * tilesY);
    let tileY = rem / tilesX;
    let tileX = rem % tilesX;

    let aabb = clusterAABB(tileX, tileY, slice);
    let aabbMin = aabb[0];
    let aabbMax = aabb[1];

    var count = 0u;

    var localIndices: array<u32, ${MAX_LIGHTS_PER_CLUSTER}>;

    for (var i = 0u; i < params.lightCount; i++) {
        let light = pointLights[i];
        let worldPos = vec4(light.position, 1.0);
        let viewPos = params.viewMatrix * worldPos;
        let lightView = vec3(viewPos.x, viewPos.y, -viewPos.z);

        if (sphereAABBIntersect(lightView, light.radius, aabbMin, aabbMax)) {
            if (count < ${MAX_LIGHTS_PER_CLUSTER}u) {
                localIndices[count] = i;
                count++;
            }
        }
    }

    if (count == 0u) {
        clusterGrid[clusterId] = vec2(0u, 0u);
        return;
    }

    let globalOffset = atomicAdd(&lightIndices[0], count);
    clusterGrid[clusterId] = vec2(globalOffset, count);

    for (var i = 0u; i < count; i++) {
        atomicStore(&lightIndices[globalOffset + 1u + i], localIndices[i]);
    }
}
`;

export const CLUSTER_BINDINGS_WGSL = /* wgsl */ `
struct ClusterParams {
    viewMatrix: mat4x4<f32>,
    tilesX: u32,
    tilesY: u32,
    sliceCount: u32,
    lightCount: u32,
    near: f32,
    far: f32,
    logRatio: f32,
    bias: f32,
    tanHalfFov: f32,
    aspect: f32,
    cameraMode: f32,
    _pad: f32,
}

@group(2) @binding(0) var<uniform> clusterParams: ClusterParams;
@group(2) @binding(1) var<storage, read> clusterGrid: array<vec2<u32>>;
@group(2) @binding(2) var<storage, read> clusterLightIndices: array<u32>;
`;

export const CLUSTER_LOOKUP_WGSL = /* wgsl */ `
fn getClusterIndex(fragCoord: vec2<f32>, viewZ: f32) -> u32 {
    let tileX = u32(fragCoord.x) / ${TILE_SIZE}u;
    let tileY = u32(fragCoord.y) / ${TILE_SIZE}u;
    var slice: u32;
    if (clusterParams.cameraMode > 0.5) {
        slice = u32(clamp((viewZ - clusterParams.near) / (clusterParams.far - clusterParams.near) * f32(clusterParams.sliceCount), 0.0, f32(clusterParams.sliceCount - 1u)));
    } else {
        slice = u32(clamp(log2(viewZ / clusterParams.near) * clusterParams.logRatio, 0.0, f32(clusterParams.sliceCount - 1u)));
    }
    return slice * clusterParams.tilesX * clusterParams.tilesY + tileY * clusterParams.tilesX + tileX;
}
`;

export const CLUSTERED_POINT_LIGHT_CALC_WGSL = /* wgsl */ `
fn computePointLights(surface: SurfaceData, V: vec3<f32>, fragCoord: vec2<f32>, viewZ: f32) -> vec3<f32> {
    var result = vec3(0.0);
    let cluster = getClusterIndex(fragCoord, viewZ);
    let gridEntry = clusterGrid[cluster];
    let offset = gridEntry.x;
    let count = gridEntry.y;

    for (var j = 0u; j < count; j++) {
        let i = clusterLightIndices[offset + 1u + j];
        let light = pointLights[i];
        let toLight = light.position - surface.worldPos;
        let dist = length(toLight);
        if (dist >= light.radius || dist < 1e-4) { continue; }

        let L = toLight / dist;
        let NdotL = max(dot(surface.worldNormal, L), 0.0);
        if (NdotL <= 0.0) { continue; }

        let ratio = 1.0 - dist / light.radius;
        let attenuation = ratio * ratio;

        var shadow = 1.0;
        if (light.shadowIdx >= 0.0) {
            shadow = samplePointShadow(surface.worldPos, surface.worldNormal, u32(light.shadowIdx), light.position, light.radius);
        }

        result += evaluatePointLight(surface, light.color, L, V, NdotL, attenuation, shadow);
    }
    return result;
}
`;

export const CLUSTERED_POINT_LIGHT_NOSHADOW_CALC_WGSL = /* wgsl */ `
fn computePointLights(surface: SurfaceData, V: vec3<f32>, fragCoord: vec2<f32>, viewZ: f32) -> vec3<f32> {
    var result = vec3(0.0);
    let cluster = getClusterIndex(fragCoord, viewZ);
    let gridEntry = clusterGrid[cluster];
    let offset = gridEntry.x;
    let count = gridEntry.y;

    for (var j = 0u; j < count; j++) {
        let i = clusterLightIndices[offset + 1u + j];
        let light = pointLights[i];
        let toLight = light.position - surface.worldPos;
        let dist = length(toLight);
        if (dist >= light.radius || dist < 1e-4) { continue; }

        let L = toLight / dist;
        let NdotL = max(dot(surface.worldNormal, L), 0.0);
        if (NdotL <= 0.0) { continue; }

        let ratio = 1.0 - dist / light.radius;
        let attenuation = ratio * ratio;

        result += evaluatePointLight(surface, light.color, L, V, NdotL, attenuation, 1.0);
    }
    return result;
}
`;

const _viewMat = new Float32Array(16);
const paramsData = new ArrayBuffer(CLUSTER_PARAMS_SIZE);
const paramsF32 = new Float32Array(paramsData);
const paramsU32 = new Uint32Array(paramsData);
const zero = new Uint32Array(1);

export function createClusterCullNode(
    raster: {
        clusterParamsBuffer: GPUBuffer;
        clusterGridBuffer: GPUBuffer;
        lightIndexBuffer: GPUBuffer;
    },
    pointLightBuffer: GPUBuffer,
    getCamera: () => number,
    getLightCount: () => number,
    maxWidth = 3840,
    maxHeight = 2160,
): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let boundGrid: GPUBuffer | null = null;

    return {
        name: "cluster-cull",
        inputs: ["point-light-raster"],
        outputs: ["cluster-data"],

        async prepare(device: GPUDevice) {
            const module = device.createShaderModule({ code: CLUSTER_CULL_WGSL });
            pipeline = await device.createComputePipelineAsync({
                label: "cluster-cull",
                layout: "auto",
                compute: { module, entryPoint: "main" },
            });
        },

        execute(ctx: ExecutionContext) {
            if (!pipeline) return;

            if (raster.clusterGridBuffer !== boundGrid) {
                boundGrid = raster.clusterGridBuffer;
                bindGroup = ctx.device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: raster.clusterParamsBuffer } },
                        { binding: 1, resource: { buffer: pointLightBuffer } },
                        { binding: 2, resource: { buffer: raster.clusterGridBuffer } },
                        { binding: 3, resource: { buffer: raster.lightIndexBuffer } },
                    ],
                });
            }

            if (!bindGroup) return;

            const eid = getCamera();
            if (eid < 0) return;

            const lightCount = getLightCount();
            if (lightCount === 0) {
                ctx.encoder.clearBuffer(raster.clusterGridBuffer);
                ctx.device.queue.writeBuffer(raster.lightIndexBuffer, 0, zero);
                return;
            }

            const world = WorldTransform.data.subarray(eid * 16, eid * 16 + 16);
            const view = invert(world, _viewMat);

            const fov = Camera.fov[eid];
            const near = Camera.near[eid];
            const far = Camera.far[eid];
            const mode = Camera.mode[eid];
            const width = ctx.getTexture("color")?.width ?? 1920;
            const height = ctx.getTexture("color")?.height ?? 1080;
            const aspect = width / height;

            const tilesX = Math.min(Math.ceil(width / TILE_SIZE), Math.ceil(maxWidth / TILE_SIZE));
            const tilesY = Math.min(
                Math.ceil(height / TILE_SIZE),
                Math.ceil(maxHeight / TILE_SIZE),
            );
            const sliceCount = DEPTH_SLICES;
            const totalClusters = tilesX * tilesY * sliceCount;

            const fovRad = (fov * Math.PI) / 180;
            const tanHalfFov = Math.tan(fovRad / 2);

            const logFarNear = Math.log2(far / near);
            const logRatio = sliceCount / logFarNear;
            const bias = (-sliceCount * Math.log2(near)) / logFarNear;

            paramsF32.set(view, 0);
            paramsU32[16] = tilesX;
            paramsU32[17] = tilesY;
            paramsU32[18] = sliceCount;
            paramsU32[19] = lightCount;
            paramsF32[20] = near;
            paramsF32[21] = far;
            paramsF32[22] = logRatio;
            paramsF32[23] = bias;
            paramsF32[24] = tanHalfFov;
            paramsF32[25] = aspect;
            paramsF32[26] = mode;
            paramsF32[27] = 0;

            ctx.device.queue.writeBuffer(raster.clusterParamsBuffer, 0, paramsData);
            ctx.device.queue.writeBuffer(raster.lightIndexBuffer, 0, zero);

            const pass = beginComputePass(ctx.encoder, ctx.timestampWrites?.("cluster-cull"));
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(Math.ceil(totalClusters / 64));
            pass.end();
        },
    };
}

export function clusterBufferSizes(maxWidth: number, maxHeight: number) {
    const tilesX = Math.ceil(maxWidth / TILE_SIZE);
    const tilesY = Math.ceil(maxHeight / TILE_SIZE);
    const maxClusters = tilesX * tilesY * DEPTH_SLICES;
    return {
        paramsSize: CLUSTER_PARAMS_SIZE,
        gridSize: maxClusters * 8,
        indexSize: 4 + MAX_LIGHT_INDICES * 4,
    };
}
