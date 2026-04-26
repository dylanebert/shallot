import type { ComputeNode, ExecutionContext } from "../compute";
import { bindView, type BufferView, type GBuf } from "../compute";
import { capacity } from "../../engine";
import {
    type SurfaceData,
    type ShapeAtlas,
    type Batching,
    type PipelineVariantConfig,
    hasProperties,
    instanceStructWGSL,
    instanceBindingWGSL,
    COLOR_FORMAT,
    SCENE_STRUCT_WGSL,
    SKY_STRUCT_WGSL,
    SHADOW_STRUCT_WGSL,
    POINT_LIGHT_STRUCT_WGSL,
    POINT_SHADOW_STRUCT_WGSL,
    HAZE_WGSL,
    SPECULAR_WGSL,
    POINT_LIGHT_EVAL_WGSL,
    NOISE_WGSL,
    SHADOW_SAMPLE_WGSL,
    POINT_SHADOW_SAMPLE_WGSL,
    SKY_DIR_WGSL,
    SKY_WGSL,
    REFLECTION_WGSL,
    drawBatches,
    MAX_BATCH_SLOTS,
    compileSurfaceBlock,
    OPACITY_GUARD_WGSL,
    WGSL_STRUCTS,
    WGSL_LIGHTING_CALC,
} from "../render/core";
import {
    CLUSTER_BINDINGS_WGSL,
    CLUSTER_LOOKUP_WGSL,
    CLUSTERED_POINT_LIGHT_CALC_WGSL,
    CLUSTERED_POINT_LIGHT_NOSHADOW_CALC_WGSL,
} from "./cluster";

const RASTER_LIGHTING_CONFIG: PipelineVariantConfig = {
    lighting: {
        params: "shadowFactor: f32, fragCoord: vec2<f32>, viewZ: f32",
        body: () => `${WGSL_LIGHTING_CALC}
    return litColor + computePointLights(surface, V, fragCoord, viewZ);`,
    },
};

export function compileRasterShader(
    surfaces: SurfaceData[],
    useShadows: boolean,
    entityId = true,
    transparent = false,
): string {
    const surfaceBlock = compileSurfaceBlock(surfaces, RASTER_LIGHTING_CONFIG);

    const shadowBindings = useShadows
        ? /* wgsl */ `
${SHADOW_STRUCT_WGSL}

@group(1) @binding(0) var<uniform> shadow: Shadow;
@group(1) @binding(1) var shadowMap: texture_depth_2d;

${POINT_SHADOW_STRUCT_WGSL}

@group(1) @binding(2) var<uniform> pointShadow: PointShadow;
@group(1) @binding(3) var pointShadowMap: texture_depth_2d;
@group(1) @binding(4) var shadowSampler: sampler_comparison;

${SHADOW_SAMPLE_WGSL}
${POINT_SHADOW_SAMPLE_WGSL}
${CLUSTERED_POINT_LIGHT_CALC_WGSL}
`
        : "";

    const shadowCompute = useShadows
        ? /* wgsl */ `
    let viewZ = -dot(scene.cameraWorld[2].xyz, surface.worldPos - scene.cameraWorld[3].xyz);
    let rawShadow = sampleShadow(surface.worldPos, viewZ, input.position.xy);
    let shadowFactor = mix(1.0, rawShadow, scene.shadowStrength);
`
        : /* wgsl */ `
    let viewZ = -dot(scene.cameraWorld[2].xyz, surface.worldPos - scene.cameraWorld[3].xyz);
    let shadowFactor = 1.0;
`;

    const fragmentOutputStruct = entityId
        ? `struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) entityId: u32,
}`
        : `struct FragmentOutput {
    @location(0) color: vec4<f32>,
}`;

    const entityIdOutput = entityId
        ? transparent
            ? "\n    output.entityId = 0u;"
            : "\n    output.entityId = input.entityId;"
        : "";
    const reflectedColorExpr = transparent
        ? "let reflectedColor = litColor * surface.opacity + reflectionColor(surface, V);"
        : "let reflectedColor = applyReflection(surface, V, litColor);";

    const pointLightNoShadow = useShadows ? "" : CLUSTERED_POINT_LIGHT_NOSHADOW_CALC_WGSL;

    return /* wgsl */ `
${WGSL_STRUCTS.replace(/struct FragmentOutput \{[^}]+\}/, fragmentOutputStruct)}
${SKY_STRUCT_WGSL}
${POINT_LIGHT_STRUCT_WGSL}

const SURFACE_ID_MASK: u32 = 0xFFu;

@group(0) @binding(5) var<uniform> sky: Sky;
@group(0) @binding(6) var<storage, read> pointLights: array<PointLightData>;

${hasProperties() ? instanceStructWGSL() : ""}
${hasProperties() ? instanceBindingWGSL(7) : ""}

${CLUSTER_BINDINGS_WGSL}
${CLUSTER_LOOKUP_WGSL}

${NOISE_WGSL}
${HAZE_WGSL}
${SPECULAR_WGSL}
${POINT_LIGHT_EVAL_WGSL}
${SKY_WGSL}
${REFLECTION_WGSL}
${shadowBindings}
${pointLightNoShadow}

${surfaceBlock}

@vertex
fn vs(input: VertexInput) -> VertexOutput {
    let eid = entityIds[input.instance];
    let world = matrices[eid];
    let d = data[eid];
    let surfaceId = d.flags & SURFACE_ID_MASK;
    let vtx = pullVertex(input.vertexIndex, eid);
    let position = vtx.position;
    let normal = vtx.normal;
    let result = dispatchVertexTransform(surfaceId, position, normal, vtx.uv, eid);
    let scaledPos = result.position * sizes[eid].xyz;
    let finalWorldPos = (world * vec4<f32>(scaledPos, 1.0)).xyz;
    let worldNormal = normalize((world * vec4<f32>(normal, 0.0)).xyz);

    var output: VertexOutput;
    output.position = scene.viewProj * vec4<f32>(finalWorldPos, 1.0);
    output.color = d.baseColor;
    output.worldNormal = worldNormal;
    output.entityId = eid;
    output.worldPos = finalWorldPos;
    output.objectPos = position * sizes[eid].xyz;
    output.objectNormal = normal;
    output.uv = result.uv;
    return output;
}

@fragment
fn fs(input: VertexOutput) -> FragmentOutput {
    let eid = input.entityId;
    let d = data[eid];
    let surfaceId = d.flags & SURFACE_ID_MASK;

    var surface: SurfaceData;
    surface.worldPos = input.worldPos;
    surface.objectPos = input.objectPos;
    surface.worldNormal = normalize(input.worldNormal);
    surface.objectNormal = normalize(input.objectNormal);
    surface.baseColor = input.color.rgb;
    surface.emission = d.emission.rgb * d.emission.a;
    surface.uv = input.uv;
    surface.roughness = d.pbr.x;
    surface.reflectivity = d.pbr.y;
    surface.opacity = input.color.a;

    dispatchFragment(surfaceId, &surface, input.position, eid);
    ${OPACITY_GUARD_WGSL}

${shadowCompute}
    let litColor = dispatchLighting(surfaceId, surface, shadowFactor, input.position.xy, viewZ);
    let V = normalize(scene.cameraWorld[3].xyz - surface.worldPos);
    let dist = length(input.worldPos - scene.cameraWorld[3].xyz);
    ${reflectedColorExpr}

    var output: FragmentOutput;
    output.color = vec4<f32>(applyHaze(reflectedColor, dist), surface.opacity);${entityIdOutput}
    _ = clusterParams.tilesX;
    return output;
}
`;
}

export function compileSkyShader(entityId = false): string {
    const outputStruct = entityId
        ? `struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) entityId: u32,
}`
        : "";

    const fragmentReturn = entityId
        ? `var output: FragmentOutput;
    output.color = vec4(color, 1.0);
    output.entityId = 0u;
    return output;`
        : `return vec4(color, 1.0);`;

    const fragmentOutput = entityId ? "FragmentOutput" : "@location(0) vec4<f32>";

    return /* wgsl */ `
${SCENE_STRUCT_WGSL}
${SKY_STRUCT_WGSL}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<uniform> sky: Sky;

${SKY_DIR_WGSL}
${NOISE_WGSL}
${SKY_WGSL}

${outputStruct}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );
    var output: VertexOutput;
    output.position = vec4(positions[vertexIndex], 0.0, 1.0);
    output.uv = (positions[vertexIndex] + 1.0) * 0.5;
    output.uv.y = 1.0 - output.uv.y;
    return output;
}

@fragment
fn fs(input: VertexOutput) -> ${fragmentOutput} {
    let dir = computeSkyDir(input.uv.x, input.uv.y);
    let color = sampleSky(dir);
    ${fragmentReturn}
}
`;
}

interface ForwardGPU {
    opaque: GPURenderPipeline;
    transparent: GPURenderPipeline;
    sky: GPURenderPipeline;
}

interface ForwardBindGroups {
    scene: GPUBindGroup;
    sceneTransparent: GPUBindGroup;
    shadow: GPUBindGroup;
    shadowTransparent: GPUBindGroup;
    cluster: GPUBindGroup;
    clusterTransparent: GPUBindGroup;
    sky: GPUBindGroup;
}

async function createRasterPipeline(
    device: GPUDevice,
    surfaces: SurfaceData[],
    colorFormat: GPUTextureFormat,
    useShadows: boolean,
): Promise<GPURenderPipeline> {
    const code = compileRasterShader(surfaces, useShadows);
    const module = device.createShaderModule({ code });

    return device.createRenderPipelineAsync({
        label: "forward",
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [{ format: colorFormat }, { format: "r32uint" }],
        },
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "less",
        },
        primitive: { topology: "triangle-list", cullMode: "back" },
    });
}

async function createTransparentPipeline(
    device: GPUDevice,
    surfaces: SurfaceData[],
    colorFormat: GPUTextureFormat,
    useShadows: boolean,
): Promise<GPURenderPipeline> {
    const code = compileRasterShader(surfaces, useShadows, true, true);
    const module = device.createShaderModule({ code });
    return device.createRenderPipelineAsync({
        label: "forward-transparent",
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [
                {
                    format: colorFormat,
                    blend: {
                        color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                    },
                },
                { format: "r32uint", writeMask: 0 },
            ],
        },
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: false,
            depthCompare: "less-equal",
        },
        primitive: { topology: "triangle-list", cullMode: "none" },
    });
}

async function createSkyPipeline(
    device: GPUDevice,
    colorFormat: GPUTextureFormat,
): Promise<GPURenderPipeline> {
    const code = compileSkyShader(true);
    const module = device.createShaderModule({ code });

    return device.createRenderPipelineAsync({
        label: "sky",
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [{ format: colorFormat }, { format: "r32uint" }],
        },
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: false,
            depthCompare: "always",
        },
        primitive: { topology: "triangle-list" },
    });
}

async function prepareForwardGPUBase(
    device: GPUDevice,
    surfaces: SurfaceData[],
): Promise<ForwardGPU> {
    const [opaque, sky, transparent] = await Promise.all([
        createRasterPipeline(device, surfaces, COLOR_FORMAT, true),
        createSkyPipeline(device, COLOR_FORMAT),
        createTransparentPipeline(device, surfaces, COLOR_FORMAT, true),
    ]);

    return { opaque, transparent, sky };
}

function createForwardBindGroups(
    device: GPUDevice,
    gpu: ForwardGPU,
    render: {
        scene: GPUBuffer;
        sky: GPUBuffer;
        data: GBuf;
        matrices: GBuf;
        sizes: BufferView;
        shapes: BufferView;
        meshAtlas: ShapeAtlas;
        batching: Batching;
        instanceDataBuffer: GBuf | null;
        pointLightBuffer: GPUBuffer;
    },
    raster: {
        shadowBuffer: GPUBuffer;
        shadowAtlas: GPUTexture;
        pointShadowBuffer: GPUBuffer;
        pointShadowAtlas: GPUTexture;
        clusterParamsBuffer: GPUBuffer;
        clusterGridBuffer: GPUBuffer;
        lightIndexBuffer: GPUBuffer;
    },
): ForwardBindGroups {
    const sceneEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: render.scene } },
        { binding: 1, resource: { buffer: render.batching.entityIds.buffer } },
        { binding: 2, resource: { buffer: render.matrices.buffer } },
        bindView(3, render.sizes),
        { binding: 4, resource: { buffer: render.data.buffer } },
        { binding: 5, resource: { buffer: render.sky } },
        { binding: 6, resource: { buffer: render.pointLightBuffer } },
    ];
    if (render.instanceDataBuffer) {
        sceneEntries.push({ binding: 7, resource: { buffer: render.instanceDataBuffer.buffer } });
    }
    sceneEntries.push(
        bindView(8, render.shapes),
        { binding: 9, resource: { buffer: render.meshAtlas.vertices } },
        { binding: 10, resource: { buffer: render.meshAtlas.meta } },
    );

    const bindScene = (pipeline: GPURenderPipeline) =>
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: sceneEntries });

    const comparisonSampler = device.createSampler({
        compare: "less",
        magFilter: "linear",
        minFilter: "linear",
    });
    const shadowEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: raster.shadowBuffer } },
        { binding: 1, resource: raster.shadowAtlas.createView() },
        { binding: 2, resource: { buffer: raster.pointShadowBuffer } },
        { binding: 3, resource: raster.pointShadowAtlas.createView() },
        { binding: 4, resource: comparisonSampler },
    ];

    const bindShadow = (pipeline: GPURenderPipeline) =>
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: shadowEntries });

    const clusterEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: raster.clusterParamsBuffer } },
        { binding: 1, resource: { buffer: raster.clusterGridBuffer } },
        { binding: 2, resource: { buffer: raster.lightIndexBuffer } },
    ];

    const bindCluster = (pipeline: GPURenderPipeline) =>
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(2), entries: clusterEntries });

    return {
        scene: bindScene(gpu.opaque),
        sceneTransparent: bindScene(gpu.transparent),
        shadow: bindShadow(gpu.opaque),
        shadowTransparent: bindShadow(gpu.transparent),
        cluster: bindCluster(gpu.opaque),
        clusterTransparent: bindCluster(gpu.transparent),
        sky: device.createBindGroup({
            layout: gpu.sky.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: render.scene } },
                { binding: 1, resource: { buffer: render.sky } },
            ],
        }),
    };
}

const _colorAtt0: GPURenderPassColorAttachment = {
    view: null! as GPUTextureView,
    clearValue: { r: 0, g: 0, b: 0, a: 1 },
    loadOp: "clear",
    storeOp: "store",
};
const _colorAtt1: GPURenderPassColorAttachment = {
    view: null! as GPUTextureView,
    clearValue: { r: 0, g: 0, b: 0, a: 0 },
    loadOp: "clear",
    storeOp: "store",
};
const _depth: GPURenderPassDepthStencilAttachment = {
    view: null! as GPUTextureView,
    depthClearValue: 1.0,
    depthLoadOp: "clear",
    depthStoreOp: "store",
};
const _forwardDesc: GPURenderPassDescriptor = {
    colorAttachments: [_colorAtt0, _colorAtt1],
    depthStencilAttachment: _depth,
};

function renderGeometry(
    ctx: ExecutionContext,
    gpu: ForwardGPU,
    bindGroups: ForwardBindGroups,
    meshIndices: GPUBuffer,
    batching: Batching,
    colorView: GPUTextureView,
    eidView: GPUTextureView,
    zView: GPUTextureView,
    clearColor: { r: number; g: number; b: number },
    hasSky: boolean,
    timestamps?: GPURenderPassTimestampWrites,
): void {
    _colorAtt0.view = colorView;
    (_colorAtt0.clearValue as GPUColorDict).r = clearColor.r;
    (_colorAtt0.clearValue as GPUColorDict).g = clearColor.g;
    (_colorAtt0.clearValue as GPUColorDict).b = clearColor.b;
    _colorAtt1.view = eidView;
    _depth.view = zView;
    _forwardDesc.timestampWrites = timestamps;
    const pass = ctx.encoder.beginRenderPass(_forwardDesc);

    if (hasSky) {
        pass.setPipeline(gpu.sky);
        pass.setBindGroup(0, bindGroups.sky);
        pass.draw(3);
    }

    pass.setPipeline(gpu.opaque);
    pass.setBindGroup(0, bindGroups.scene);
    pass.setBindGroup(1, bindGroups.shadow);
    pass.setBindGroup(2, bindGroups.cluster);
    pass.setIndexBuffer(meshIndices, "uint32");
    drawBatches(pass, batching.indirect, 0, batching.activeSlots, batching.activeSlotCount);

    pass.setPipeline(gpu.transparent);
    pass.setBindGroup(0, bindGroups.sceneTransparent);
    pass.setBindGroup(1, bindGroups.shadowTransparent);
    pass.setBindGroup(2, bindGroups.clusterTransparent);
    drawBatches(
        pass,
        batching.indirect,
        MAX_BATCH_SLOTS,
        batching.activeSlots,
        batching.activeSlotCount,
    );

    pass.end();
}

export function createRasterForwardNode(
    render: {
        scene: GPUBuffer;
        sky: GPUBuffer;
        data: GBuf;
        matrices: GBuf;
        sizes: BufferView;
        shapes: BufferView;
        meshAtlas: ShapeAtlas;
        meshVersion: number;
        batching: Batching;
        instanceDataBuffer: GBuf | null;
        pointLightBuffer: GPUBuffer;
    },
    raster: {
        shadowBuffer: GPUBuffer;
        shadowAtlas: GPUTexture;
        pointShadowBuffer: GPUBuffer;
        pointShadowAtlas: GPUTexture;
        clusterParamsBuffer: GPUBuffer;
        clusterGridBuffer: GPUBuffer;
        lightIndexBuffer: GPUBuffer;
        bindGroupsDirty: boolean;
    },
    getSurfaces: () => SurfaceData[],
    getClearColor: () => { r: number; g: number; b: number },
    getSky: () => boolean,
): ComputeNode {
    let gpu: ForwardGPU | null = null;
    let bindGroups: ForwardBindGroups | null = null;
    let boundInstBuf: GBuf | null = null;
    let bindGroupsDirty = false;
    let cachedCapacity = capacity();
    let cachedMeshVer = render.meshVersion;

    return {
        name: "forward",
        inputs: ["culled", "shadow-atlas", "point-shadow-atlas", "cluster-data"],
        outputs: ["color", "eid", "z"],

        async prepare(device: GPUDevice) {
            const surfaces = getSurfaces();
            gpu = await prepareForwardGPUBase(device, surfaces);
            boundInstBuf = render.instanceDataBuffer;
            bindGroups = createForwardBindGroups(device, gpu, render, raster);
        },

        execute(ctx: ExecutionContext) {
            if (!gpu || !bindGroups) return;
            if ((globalThis as any).__SKIP_FORWARD) return;

            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                bindGroupsDirty = true;
            }

            if (render.meshVersion !== cachedMeshVer) {
                cachedMeshVer = render.meshVersion;
                bindGroupsDirty = true;
            }

            if (raster.bindGroupsDirty) {
                raster.bindGroupsDirty = false;
                bindGroupsDirty = true;
            }

            const currentInstBuf = render.instanceDataBuffer;
            if (currentInstBuf !== boundInstBuf || bindGroupsDirty) {
                boundInstBuf = currentInstBuf;
                bindGroupsDirty = false;
                bindGroups = createForwardBindGroups(ctx.device, gpu, render, raster);
            }

            const colorView = ctx.getTextureView("color");
            const eidView = ctx.getTextureView("eid");
            const zView = ctx.getTextureView("z");
            if (!colorView || !eidView || !zView) return;

            const clearColor = getClearColor();
            const hasSky = getSky();

            renderGeometry(
                ctx,
                gpu,
                bindGroups,
                render.meshAtlas.indices,
                render.batching,
                colorView as GPUTextureView,
                eidView as GPUTextureView,
                zView as GPUTextureView,
                clearColor,
                hasSky,
                ctx.timestampWrites?.("raster-forward"),
            );
        },
    };
}
