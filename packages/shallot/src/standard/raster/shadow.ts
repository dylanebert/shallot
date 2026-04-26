import type { ComputeNode, ExecutionContext } from "../compute";
import { bindView, type BufferView, type GBuf } from "../compute";
import { capacity } from "../../engine";
import {
    perspective,
    multiply,
    invert,
    lookAtMatrix,
    orthographicBounds,
    extractFrustumCorners,
    invertMatrix,
} from "../../engine";
import {
    type SurfaceData,
    type ShapeAtlas,
    type Batching,
    SHADOW_STRUCT_WGSL,
    POINT_SHADOW_STRUCT_WGSL,
    SCENE_STRUCT_WGSL,
    DATA_STRUCT_WGSL,
    VERTEX_PULL_WGSL,
    hasProperties,
    instanceStructWGSL,
    instanceBindingWGSL,
    compileVertexVariant,
    compileVertexDispatch,
    drawBatches,
} from "../render/core";

const _cmProj = new Float32Array(16);
const _cmView = new Float32Array(16);
const _cmVp = new Float32Array(16);
const _cmInvVp = new Float32Array(16);
const _cmCorners = new Float32Array(24);
const _cmLightView = new Float32Array(16);
const _cmLightProj = new Float32Array(16);
const _cmCascadeVP = new Float32Array(16);

const SHADOW_CASCADE_COUNT = 4;
const SHADOW_ATLAS_SIZE = 2048;
const SHADOW_CASCADE_SIZE = SHADOW_ATLAS_SIZE / 2;
export const SHADOW_BUFFER_SIZE = 288;

export function computeCascadeSplits(
    near: number,
    far: number,
    cascadeCount: number,
    lambda = 0.75,
): Float32Array {
    const splits = new Float32Array(cascadeCount);
    const ratio = far / near;

    for (let i = 0; i < cascadeCount; i++) {
        const p = (i + 1) / cascadeCount;
        const log = near * Math.pow(ratio, p);
        const uniform = near + (far - near) * p;
        splits[i] = lambda * log + (1 - lambda) * uniform;
    }

    return splits;
}

interface CascadeData {
    viewProj: Float32Array;
    texelSize: number;
}

export function computeCascadeMatrix(
    cameraWorld: Float32Array,
    fov: number,
    aspect: number,
    nearSplit: number,
    farSplit: number,
    lightDir: [number, number, number],
    shadowMapSize: number,
): CascadeData {
    const proj = perspective(fov, aspect, nearSplit, farSplit, _cmProj);
    const view = invert(cameraWorld, _cmView);
    const viewProj = multiply(proj, view, _cmVp);
    const invViewProj = invertMatrix(viewProj, _cmInvVp);

    const corners = extractFrustumCorners(invViewProj, 0, 1, _cmCorners);

    let centerX = 0,
        centerY = 0,
        centerZ = 0;
    for (let i = 0; i < 8; i++) {
        centerX += corners[i * 3];
        centerY += corners[i * 3 + 1];
        centerZ += corners[i * 3 + 2];
    }
    centerX /= 8;
    centerY /= 8;
    centerZ /= 8;

    const [lightDirX, lightDirY, lightDirZ] = lightDir;
    const len = Math.sqrt(lightDirX * lightDirX + lightDirY * lightDirY + lightDirZ * lightDirZ);
    const normLightX = lightDirX / len;
    const normLightY = lightDirY / len;
    const normLightZ = lightDirZ / len;

    let maxRadius = 0;
    for (let i = 0; i < 8; i++) {
        const dx = corners[i * 3] - centerX;
        const dy = corners[i * 3 + 1] - centerY;
        const dz = corners[i * 3 + 2] - centerZ;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        maxRadius = Math.max(maxRadius, dist);
    }

    const shadowDistance = maxRadius * 2;
    const eyeX = centerX - normLightX * shadowDistance;
    const eyeY = centerY - normLightY * shadowDistance;
    const eyeZ = centerZ - normLightZ * shadowDistance;

    const lightView = lookAtMatrix(
        eyeX,
        eyeY,
        eyeZ,
        centerX,
        centerY,
        centerZ,
        0,
        1,
        0,
        _cmLightView,
    );

    let minX = Infinity,
        maxX = -Infinity;
    let minY = Infinity,
        maxY = -Infinity;
    let minZ = Infinity,
        maxZ = -Infinity;
    for (let i = 0; i < 8; i++) {
        const wx = corners[i * 3];
        const wy = corners[i * 3 + 1];
        const wz = corners[i * 3 + 2];
        const lx = lightView[0] * wx + lightView[4] * wy + lightView[8] * wz + lightView[12];
        const ly = lightView[1] * wx + lightView[5] * wy + lightView[9] * wz + lightView[13];
        const lz = lightView[2] * wx + lightView[6] * wy + lightView[10] * wz + lightView[14];
        minX = Math.min(minX, lx);
        maxX = Math.max(maxX, lx);
        minY = Math.min(minY, ly);
        maxY = Math.max(maxY, ly);
        minZ = Math.min(minZ, lz);
        maxZ = Math.max(maxZ, lz);
    }

    minZ -= shadowDistance;
    maxZ += shadowDistance;

    const texelSizeX = (maxX - minX) / shadowMapSize;
    const texelSizeY = (maxY - minY) / shadowMapSize;
    const texelSnap = Math.max(texelSizeX, texelSizeY);
    minX = Math.floor(minX / texelSnap) * texelSnap;
    maxX = Math.ceil(maxX / texelSnap) * texelSnap;
    minY = Math.floor(minY / texelSnap) * texelSnap;
    maxY = Math.ceil(maxY / texelSnap) * texelSnap;

    const lightProj = orthographicBounds(minX, maxX, minY, maxY, -maxZ, -minZ, _cmLightProj);
    const cascadeViewProj = multiply(lightProj, lightView, _cmCascadeVP);

    const texelNDC = 2 / shadowMapSize;
    cascadeViewProj[12] = Math.floor(cascadeViewProj[12] / texelNDC) * texelNDC;
    cascadeViewProj[13] = Math.floor(cascadeViewProj[13] / texelNDC) * texelNDC;

    return { viewProj: cascadeViewProj, texelSize: texelSnap };
}

const shadowStaging = new ArrayBuffer(SHADOW_BUFFER_SIZE);
const shadowF32 = new Float32Array(shadowStaging);

export function createShadowBuffer(device: GPUDevice): GPUBuffer {
    return device.createBuffer({
        label: "shadow",
        size: SHADOW_BUFFER_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
}

export function createShadowAtlas(device: GPUDevice): GPUTexture {
    return device.createTexture({
        label: "shadow-atlas",
        size: [SHADOW_ATLAS_SIZE, SHADOW_ATLAS_SIZE, 1],
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
}

export function createShadowUploadNode(
    shadowBuffer: GPUBuffer,
    getCameraData: () => {
        world: Float32Array;
        fov: number;
        near: number;
        far: number;
        width: number;
        height: number;
    } | null,
    getLightDir: () => [number, number, number],
    shadowsEnabled: () => boolean,
    getShadowDistance: () => number,
): ComputeNode {
    return {
        name: "shadow-cascade-upload",
        inputs: ["data"],
        outputs: ["shadow-cascades"],

        execute(ctx: ExecutionContext) {
            if (!shadowsEnabled()) return;

            const camera = getCameraData();
            if (!camera) return;

            const { world, fov, near, far, width, height } = camera;
            const aspect = width / height;
            const lightDir = getLightDir();

            const shadowDistance = getShadowDistance();
            const effectiveFar = Math.min(far, shadowDistance);

            const splits = computeCascadeSplits(near, effectiveFar, SHADOW_CASCADE_COUNT);
            const splitsOffset = SHADOW_CASCADE_COUNT * 16;
            const texelSizeOffset = splitsOffset + SHADOW_CASCADE_COUNT;

            let prevSplit = near;
            for (let i = 0; i < SHADOW_CASCADE_COUNT; i++) {
                const { viewProj, texelSize } = computeCascadeMatrix(
                    world,
                    fov,
                    aspect,
                    prevSplit,
                    splits[i],
                    lightDir,
                    SHADOW_CASCADE_SIZE,
                );
                shadowF32.set(viewProj, i * 16);
                shadowF32[splitsOffset + i] = splits[i];
                shadowF32[texelSizeOffset + i] = texelSize;
                prevSplit = splits[i];
            }

            ctx.device.queue.writeBuffer(shadowBuffer, 0, shadowStaging);
        },
    };
}

interface ShadowForwardGPU {
    pipeline: GPURenderPipeline;
    cascadeIndexBuffers: GPUBuffer[];
    cascadeBindGroups: GPUBindGroup[];
    depthView: GPUTextureView;
}

function shadowNeedsInstanceData(surfaces: SurfaceData[]): boolean {
    return surfaces.some(
        (s) => s.properties?.length && hasProperties() && s.vertex?.includes("inst."),
    );
}

function compileShadowDepthShader(surfaces: SurfaceData[]): string {
    const variants = surfaces.map((s, i) => compileVertexVariant(i, s)).join("\n");
    const dispatch = compileVertexDispatch(surfaces.length);
    const needsProps = shadowNeedsInstanceData(surfaces);

    return /* wgsl */ `
struct VertexInput {
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instance: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

${SHADOW_STRUCT_WGSL}
${DATA_STRUCT_WGSL}
${SCENE_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> shadow: Shadow;
@group(0) @binding(1) var<storage, read> entityIds: array<u32>;
@group(0) @binding(2) var<storage, read> matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> sizes: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> cascadeIndex: u32;
@group(0) @binding(5) var<storage, read> data: array<Data>;
@group(0) @binding(6) var<uniform> scene: Scene;

${needsProps ? instanceStructWGSL() : ""}
${needsProps ? instanceBindingWGSL(7) : ""}

@group(0) @binding(8) var<storage, read> shapes: array<u32>;
@group(0) @binding(9) var<storage, read> meshVertexData: array<f32>;
@group(0) @binding(10) var<storage, read> meshMeta: array<vec4<u32>>;

${VERTEX_PULL_WGSL}

const SURFACE_ID_MASK: u32 = 0xFFu;

fn getCascadeViewProj(cascade: u32) -> mat4x4<f32> {
    switch cascade {
        case 0u: { return shadow.cascade0ViewProj; }
        case 1u: { return shadow.cascade1ViewProj; }
        case 2u: { return shadow.cascade2ViewProj; }
        default: { return shadow.cascade3ViewProj; }
    }
}

${variants}
${dispatch}

@vertex
fn vs(input: VertexInput) -> VertexOutput {
    let eid = entityIds[input.instance];
    var output: VertexOutput;
    if (sizes[eid].w == 0.0) {
        output.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return output;
    }
    let d = data[eid];
    let surfaceId = d.flags & SURFACE_ID_MASK;
    let vtx = pullVertex(input.vertexIndex, eid);
    let result = dispatchVertexTransform(surfaceId, vtx.position, vtx.normal, vtx.uv, eid);
    let world = matrices[eid];
    let scaledPos = result.position * sizes[eid].xyz;
    let worldPos = (world * vec4<f32>(scaledPos, 1.0)).xyz;
    let viewProj = getCascadeViewProj(cascadeIndex);
    _ = scene.time;

    output.position = viewProj * vec4<f32>(worldPos, 1.0);
    return output;
}

@fragment
fn fs() {}
`;
}

interface ShadowPipeline {
    pipeline: GPURenderPipeline;
    cascadeIndexBuffers: GPUBuffer[];
    needsProps: boolean;
}

async function compileShadowPipeline(
    device: GPUDevice,
    surfaces: SurfaceData[],
): Promise<ShadowPipeline> {
    const code = compileShadowDepthShader(surfaces);
    const module = device.createShaderModule({ code });
    const pipeline = await device.createRenderPipelineAsync({
        label: "shadow-dir",
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [] },
        depthStencil: {
            format: "depth32float",
            depthWriteEnabled: true,
            depthCompare: "less",
        },
        primitive: { topology: "triangle-list", cullMode: "front" },
    });

    const cascadeIndexBuffers: GPUBuffer[] = [];
    for (let i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        const buf = device.createBuffer({
            label: `cascade-index-${i}`,
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, new Uint32Array([i]));
        cascadeIndexBuffers.push(buf);
    }

    return { pipeline, cascadeIndexBuffers, needsProps: shadowNeedsInstanceData(surfaces) };
}

function createShadowBindGroups(
    device: GPUDevice,
    sp: ShadowPipeline,
    render: {
        matrices: GBuf;
        sizes: BufferView;
        data: GBuf;
        scene: GPUBuffer;
        shapes: BufferView;
        meshAtlas: ShapeAtlas;
        batching: Batching;
        instanceDataBuffer: GBuf | null;
    },
    shadowBuffer: GPUBuffer,
    atlas: GPUTexture,
): ShadowForwardGPU {
    const layout = sp.pipeline.getBindGroupLayout(0);
    const cascadeBindGroups = sp.cascadeIndexBuffers.map((indexBuf) => {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: shadowBuffer } },
            { binding: 1, resource: { buffer: render.batching.entityIds.buffer } },
            { binding: 2, resource: { buffer: render.matrices.buffer } },
            bindView(3, render.sizes),
            { binding: 4, resource: { buffer: indexBuf } },
            { binding: 5, resource: { buffer: render.data.buffer } },
            { binding: 6, resource: { buffer: render.scene } },
        ];
        if (sp.needsProps) {
            const instBuf = render.instanceDataBuffer;
            if (instBuf) entries.push({ binding: 7, resource: { buffer: instBuf.buffer } });
        }
        entries.push(
            bindView(8, render.shapes),
            { binding: 9, resource: { buffer: render.meshAtlas.vertices } },
            { binding: 10, resource: { buffer: render.meshAtlas.meta } },
        );
        return device.createBindGroup({ layout, entries });
    });

    return {
        pipeline: sp.pipeline,
        cascadeIndexBuffers: sp.cascadeIndexBuffers,
        cascadeBindGroups,
        depthView: atlas.createView(),
    };
}

const _shadowDepthAtt: GPURenderPassDepthStencilAttachment = {
    view: null! as GPUTextureView,
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
};
const _shadowDesc: GPURenderPassDescriptor = {
    colorAttachments: [],
    depthStencilAttachment: _shadowDepthAtt,
};

export function createShadowForwardNode(
    render: {
        matrices: GBuf;
        sizes: BufferView;
        data: GBuf;
        scene: GPUBuffer;
        shapes: BufferView;
        meshAtlas: ShapeAtlas;
        meshVersion: number;
        batching: Batching;
        instanceDataBuffer: GBuf | null;
    },
    shadowBuffer: GPUBuffer,
    getAtlas: () => GPUTexture,
    shadowsEnabled: () => boolean,
    getSurfaces: () => SurfaceData[],
): ComputeNode[] {
    let sp: ShadowPipeline | null = null;
    let compiling = false;
    let gpu: ShadowForwardGPU | null = null;
    let cachedCapacity = capacity();
    let cachedMeshVer = render.meshVersion;

    const node: ComputeNode = {
        name: "shadow-render",
        inputs: ["shadow-cascades"],
        outputs: ["shadow-atlas"],

        execute(ctx: ExecutionContext) {
            if (!shadowsEnabled()) return;

            if (!sp && !compiling) {
                compiling = true;
                compileShadowPipeline(ctx.device, getSurfaces())
                    .then((result) => {
                        sp = result;
                    })
                    .catch(() => {})
                    .finally(() => {
                        compiling = false;
                    });
            }

            if (!sp) return;

            let dirty = !gpu;

            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                dirty = true;
            }

            if (render.meshVersion !== cachedMeshVer) {
                cachedMeshVer = render.meshVersion;
                dirty = true;
            }

            if (dirty) {
                gpu = createShadowBindGroups(ctx.device, sp, render, shadowBuffer, getAtlas());
            }

            const g = gpu!;
            _shadowDepthAtt.view = g.depthView;
            _shadowDesc.timestampWrites = ctx.timestampWrites?.("raster-shadow");
            const pass = ctx.encoder.beginRenderPass(_shadowDesc);
            pass.setPipeline(g.pipeline);
            pass.setIndexBuffer(render.meshAtlas.indices, "uint32");
            for (let cascade = 0; cascade < SHADOW_CASCADE_COUNT; cascade++) {
                const offsetX = (cascade % 2) * SHADOW_CASCADE_SIZE;
                const offsetY = Math.floor(cascade / 2) * SHADOW_CASCADE_SIZE;
                pass.setViewport(offsetX, offsetY, SHADOW_CASCADE_SIZE, SHADOW_CASCADE_SIZE, 0, 1);
                pass.setScissorRect(offsetX, offsetY, SHADOW_CASCADE_SIZE, SHADOW_CASCADE_SIZE);
                pass.setBindGroup(0, g.cascadeBindGroups[cascade]);
                drawBatches(
                    pass,
                    render.batching.indirect,
                    0,
                    render.batching.activeSlots,
                    render.batching.activeSlotCount,
                );
            }
            pass.end();
        },
    };

    return [node];
}

export const MAX_POINT_SHADOWS = 4;
export const POINT_SHADOW_FACE_SIZE = 512;
export const POINT_SHADOW_VP_COUNT = MAX_POINT_SHADOWS * 6;

export interface PointShadowState {
    atlas: GPUTexture;
    buffer: GPUBuffer;
}

export function ensurePointShadows(
    device: GPUDevice,
    state: { current: PointShadowState | null },
): PointShadowState {
    if (state.current) return state.current;
    state.current = {
        atlas: createPointShadowAtlas(device),
        buffer: createPointShadowBuffer(device),
    };
    return state.current;
}

const POINT_SHADOW_ATLAS_WIDTH = POINT_SHADOW_FACE_SIZE * 6;
const POINT_SHADOW_ATLAS_HEIGHT = POINT_SHADOW_FACE_SIZE * MAX_POINT_SHADOWS;
export const POINT_SHADOW_BUFFER_SIZE = POINT_SHADOW_VP_COUNT * 64 + MAX_POINT_SHADOWS * 16;

const CUBE_FACE_DIRS = [
    { dx: 1, dy: 0, dz: 0, ux: 0, uy: -1, uz: 0 },
    { dx: -1, dy: 0, dz: 0, ux: 0, uy: -1, uz: 0 },
    { dx: 0, dy: 1, dz: 0, ux: 0, uy: 0, uz: 1 },
    { dx: 0, dy: -1, dz: 0, ux: 0, uy: 0, uz: -1 },
    { dx: 0, dy: 0, dz: 1, ux: 0, uy: -1, uz: 0 },
    { dx: 0, dy: 0, dz: -1, ux: 0, uy: -1, uz: 0 },
];

export function createPointShadowAtlas(device: GPUDevice): GPUTexture {
    return device.createTexture({
        label: "point-shadow-atlas",
        size: [POINT_SHADOW_ATLAS_WIDTH, POINT_SHADOW_ATLAS_HEIGHT, 1],
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
}

export function createPointShadowBuffer(device: GPUDevice): GPUBuffer {
    return device.createBuffer({
        label: "point-shadow",
        size: POINT_SHADOW_BUFFER_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
}

const _psProj = new Float32Array(16);
const _psView = new Float32Array(16);
const _psVP = new Float32Array(16);

function computePointShadowFaceMatrix(
    px: number,
    py: number,
    pz: number,
    faceIdx: number,
    radius: number,
): Float32Array {
    const face = CUBE_FACE_DIRS[faceIdx];
    const view = lookAtMatrix(
        px,
        py,
        pz,
        px + face.dx,
        py + face.dy,
        pz + face.dz,
        face.ux,
        face.uy,
        face.uz,
        _psView,
    );
    const proj = perspective(90, 1, 0.1, radius, _psProj);
    return multiply(proj, view, _psVP);
}

const pointShadowData = new ArrayBuffer(POINT_SHADOW_BUFFER_SIZE);
const pointShadowF32 = new Float32Array(pointShadowData);

export function createPointShadowUploadNode(
    getPointLights: () => [Float32Array, number],
    getBuffer: () => GPUBuffer | null,
): ComputeNode {
    return {
        name: "point-shadow-upload",
        inputs: ["point-light-raster"],
        outputs: ["point-shadow-data"],

        execute(ctx: ExecutionContext) {
            const pointShadowBuffer = getBuffer();
            if (!pointShadowBuffer) return;
            const [lights, count] = getPointLights();
            pointShadowF32.fill(0);

            let shadowIdx = 0;
            for (let i = 0; i < count && shadowIdx < MAX_POINT_SHADOWS; i++) {
                const o = i * 8;
                if (lights[o + 7] < 0) continue;

                const px = lights[o],
                    py = lights[o + 1],
                    pz = lights[o + 2];
                const radius = lights[o + 3];

                for (let face = 0; face < 6; face++) {
                    const vp = computePointShadowFaceMatrix(px, py, pz, face, radius);
                    const vpOffset = (shadowIdx * 6 + face) * 16;
                    pointShadowF32.set(vp, vpOffset);
                }

                const metaOffset = POINT_SHADOW_VP_COUNT * 16 + shadowIdx * 4;
                pointShadowF32[metaOffset] = px;
                pointShadowF32[metaOffset + 1] = py;
                pointShadowF32[metaOffset + 2] = pz;
                pointShadowF32[metaOffset + 3] = radius;

                shadowIdx++;
            }

            ctx.device.queue.writeBuffer(pointShadowBuffer, 0, pointShadowData);
        },
    };
}

function compilePointShadowDepthShader(surfaces: SurfaceData[]): string {
    const variants = surfaces.map((s, i) => compileVertexVariant(i, s)).join("\n");
    const dispatch = compileVertexDispatch(surfaces.length);
    const needsProps = shadowNeedsInstanceData(surfaces);

    return /* wgsl */ `
struct VertexInput {
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instance: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

${POINT_SHADOW_STRUCT_WGSL}
${DATA_STRUCT_WGSL}
${SCENE_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> pointShadow: PointShadow;
@group(0) @binding(1) var<storage, read> entityIds: array<u32>;
@group(0) @binding(2) var<storage, read> matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> sizes: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> vpIndex: u32;
@group(0) @binding(5) var<storage, read> data: array<Data>;
@group(0) @binding(6) var<uniform> scene: Scene;

${needsProps ? instanceStructWGSL() : ""}
${needsProps ? instanceBindingWGSL(7) : ""}

@group(0) @binding(8) var<storage, read> shapes: array<u32>;
@group(0) @binding(9) var<storage, read> meshVertexData: array<f32>;
@group(0) @binding(10) var<storage, read> meshMeta: array<vec4<u32>>;

${VERTEX_PULL_WGSL}

const SURFACE_ID_MASK: u32 = 0xFFu;

${variants}
${dispatch}

@vertex
fn vs(input: VertexInput) -> VertexOutput {
    let eid = entityIds[input.instance];
    var output: VertexOutput;
    if (sizes[eid].w == 0.0) {
        output.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return output;
    }
    let d = data[eid];
    let surfaceId = d.flags & SURFACE_ID_MASK;
    let vtx = pullVertex(input.vertexIndex, eid);
    let result = dispatchVertexTransform(surfaceId, vtx.position, vtx.normal, vtx.uv, eid);
    let world = matrices[eid];
    let scaledPos = result.position * sizes[eid].xyz;
    let worldPos = (world * vec4<f32>(scaledPos, 1.0)).xyz;
    _ = scene.time;
    output.position = pointShadow.viewProj[vpIndex] * vec4<f32>(worldPos, 1.0);
    return output;
}

@fragment
fn fs() {}
`;
}

interface PointShadowForwardGPU {
    pipeline: GPURenderPipeline;
    vpIndexBuffers: GPUBuffer[];
    vpBindGroups: GPUBindGroup[];
    atlasView: GPUTextureView;
}

interface PointShadowPipeline {
    pipeline: GPURenderPipeline;
    vpIndexBuffers: GPUBuffer[];
    needsProps: boolean;
}

async function compilePointShadowPipeline(
    device: GPUDevice,
    surfaces: SurfaceData[],
): Promise<PointShadowPipeline> {
    const code = compilePointShadowDepthShader(surfaces);
    const module = device.createShaderModule({ code });
    const pipeline = await device.createRenderPipelineAsync({
        label: "shadow-point",
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [] },
        depthStencil: {
            format: "depth32float",
            depthWriteEnabled: true,
            depthCompare: "less",
        },
        primitive: { topology: "triangle-list", cullMode: "front" },
    });

    const vpIndexBuffers: GPUBuffer[] = [];
    for (let i = 0; i < POINT_SHADOW_VP_COUNT; i++) {
        const buf = device.createBuffer({
            label: `point-shadow-vp-${i}`,
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, new Uint32Array([i]));
        vpIndexBuffers.push(buf);
    }

    return { pipeline, vpIndexBuffers, needsProps: shadowNeedsInstanceData(surfaces) };
}

function createPointShadowBindGroups(
    device: GPUDevice,
    psp: PointShadowPipeline,
    render: {
        matrices: GBuf;
        sizes: BufferView;
        data: GBuf;
        scene: GPUBuffer;
        shapes: BufferView;
        meshAtlas: ShapeAtlas;
        batching: Batching;
        instanceDataBuffer: GBuf | null;
    },
    pointShadowBuffer: GPUBuffer,
    atlas: GPUTexture,
): PointShadowForwardGPU {
    const layout = psp.pipeline.getBindGroupLayout(0);
    const vpBindGroups = psp.vpIndexBuffers.map((indexBuf) => {
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: pointShadowBuffer } },
            { binding: 1, resource: { buffer: render.batching.entityIds.buffer } },
            { binding: 2, resource: { buffer: render.matrices.buffer } },
            bindView(3, render.sizes),
            { binding: 4, resource: { buffer: indexBuf } },
            { binding: 5, resource: { buffer: render.data.buffer } },
            { binding: 6, resource: { buffer: render.scene } },
        ];
        if (psp.needsProps) {
            const instBuf = render.instanceDataBuffer;
            if (instBuf) entries.push({ binding: 7, resource: { buffer: instBuf.buffer } });
        }
        entries.push(
            bindView(8, render.shapes),
            { binding: 9, resource: { buffer: render.meshAtlas.vertices } },
            { binding: 10, resource: { buffer: render.meshAtlas.meta } },
        );
        return device.createBindGroup({ layout, entries });
    });

    return {
        pipeline: psp.pipeline,
        vpIndexBuffers: psp.vpIndexBuffers,
        vpBindGroups,
        atlasView: atlas.createView(),
    };
}

const _pointShadowDepthAtt: GPURenderPassDepthStencilAttachment = {
    view: null! as GPUTextureView,
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
};
const _pointShadowDesc: GPURenderPassDescriptor = {
    colorAttachments: [],
    depthStencilAttachment: _pointShadowDepthAtt,
};

export function createPointShadowForwardNode(
    render: {
        matrices: GBuf;
        sizes: BufferView;
        data: GBuf;
        scene: GPUBuffer;
        shapes: BufferView;
        meshAtlas: ShapeAtlas;
        meshVersion: number;
        batching: Batching;
        instanceDataBuffer: GBuf | null;
    },
    getPointShadows: () => PointShadowState | null,
    getPointLights: () => [Float32Array, number],
    getSurfaces: () => SurfaceData[],
): ComputeNode {
    let psp: PointShadowPipeline | null = null;
    let compiling = false;
    let gpu: PointShadowForwardGPU | null = null;
    let cachedCapacity = capacity();
    let cachedMeshVer = render.meshVersion;

    return {
        name: "point-shadow-render",
        inputs: ["point-shadow-data", "batched"],
        outputs: ["point-shadow-atlas"],

        execute(ctx: ExecutionContext) {
            const ps = getPointShadows();
            if (!ps) return;

            if (!psp && !compiling) {
                compiling = true;
                compilePointShadowPipeline(ctx.device, getSurfaces())
                    .then((result) => {
                        psp = result;
                    })
                    .catch(() => {})
                    .finally(() => {
                        compiling = false;
                    });
            }

            if (!psp) return;

            let dirty = !gpu;

            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                dirty = true;
            }

            if (render.meshVersion !== cachedMeshVer) {
                cachedMeshVer = render.meshVersion;
                dirty = true;
            }

            if (dirty) {
                gpu = createPointShadowBindGroups(ctx.device, psp, render, ps.buffer, ps.atlas);
            }

            const [lights, count] = getPointLights();
            let shadowCount = 0;
            for (let i = 0; i < count; i++) {
                if (lights[i * 8 + 7] >= 0) shadowCount++;
            }
            if (shadowCount === 0) return;

            const g = gpu!;
            _pointShadowDepthAtt.view = g.atlasView;
            _pointShadowDesc.timestampWrites = ctx.timestampWrites?.("raster-point-shadow");
            const pass = ctx.encoder.beginRenderPass(_pointShadowDesc);
            pass.setPipeline(g.pipeline);
            pass.setIndexBuffer(render.meshAtlas.indices, "uint32");

            for (let lightIdx = 0; lightIdx < shadowCount; lightIdx++) {
                for (let face = 0; face < 6; face++) {
                    const vpIdx = lightIdx * 6 + face;
                    const offsetX = face * POINT_SHADOW_FACE_SIZE;
                    const offsetY = lightIdx * POINT_SHADOW_FACE_SIZE;
                    pass.setViewport(
                        offsetX,
                        offsetY,
                        POINT_SHADOW_FACE_SIZE,
                        POINT_SHADOW_FACE_SIZE,
                        0,
                        1,
                    );
                    pass.setScissorRect(
                        offsetX,
                        offsetY,
                        POINT_SHADOW_FACE_SIZE,
                        POINT_SHADOW_FACE_SIZE,
                    );
                    pass.setBindGroup(0, g.vpBindGroups[vpIdx]);
                    drawBatches(
                        pass,
                        render.batching.indirect,
                        0,
                        render.batching.activeSlots,
                        render.batching.activeSlotCount,
                    );
                }
            }
            pass.end();
        },
    };
}
