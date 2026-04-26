import { traits, type Plugin, type State } from "../../engine";
import {
    beginComputePass,
    Compute,
    ComputePlugin,
    type ComputeNode,
    type ExecutionContext,
} from "../../standard/compute";
import { ActiveCamera, RenderPlugin } from "../../standard/render";
import { COLOR_FORMAT } from "../../standard/render/core";

export const Bloom = {
    intensity: [] as number[],
    threshold: [] as number[],
    radius: [] as number[],
};

traits(Bloom, {
    defaults: () => ({
        intensity: 0.2,
        threshold: 0.8,
        radius: 0.5,
    }),
});

const MIP_FORMAT: GPUTextureFormat = "rgba16float";
const MAX_MIPS = 6;
const SLOT_STRIDE = 256;
const UNIFORM_BUFFER_SIZE = 4096;

const bloomComputeShader = /* wgsl */ `
struct Params {
    texelSize: vec2f,
    param0: f32,
    param1: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn thresholdMain(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);

    let threshold = params.param0;
    let knee = params.param1;
    let color = clamp(textureSampleLevel(srcTexture, srcSampler, uv, 0.0).rgb, vec3f(0.0), vec3f(65504.0));
    let brightness = max(max(color.r, color.g), color.b);
    let soft = clamp((brightness - threshold + knee) / (2.0 * knee + 0.0001), 0.0, 1.0);

    textureStore(output, vec2i(gid.xy), vec4f(color * soft * soft, 1.0));
}

@compute @workgroup_size(16, 16)
fn downsampleMain(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let ts = params.texelSize;

    var color = textureSampleLevel(srcTexture, srcSampler, uv, 0.0).rgb * 4.0;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(-ts.x, -ts.y), 0.0).rgb;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(ts.x, -ts.y), 0.0).rgb;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(-ts.x, ts.y), 0.0).rgb;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(ts.x, ts.y), 0.0).rgb;

    textureStore(output, vec2i(gid.xy), vec4f(color / 8.0, 1.0));
}
`;

const upsampleComputeShader = /* wgsl */ `
struct Params {
    texelSize: vec2f,
    param0: f32,
    param1: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var existingTexture: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn upsampleMain(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let ts = params.texelSize * params.param0;

    var color = vec3f(0.0);
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(-ts.x, -ts.y), 0.0).rgb;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(0.0, -ts.y), 0.0).rgb * 2.0;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(ts.x, -ts.y), 0.0).rgb;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(-ts.x, 0.0), 0.0).rgb * 2.0;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(ts.x, 0.0), 0.0).rgb * 2.0;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(-ts.x, ts.y), 0.0).rgb;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(0.0, ts.y), 0.0).rgb * 2.0;
    color += textureSampleLevel(srcTexture, srcSampler, uv + vec2f(ts.x, ts.y), 0.0).rgb;

    let weight = params.param1;
    let existing = textureLoad(existingTexture, vec2i(gid.xy), 0).rgb;
    textureStore(output, vec2i(gid.xy), vec4f(existing + color / 12.0 * weight, 1.0));
}
`;

const compositeComputeShader = /* wgsl */ `
struct Params {
    intensity: f32,
}

@group(0) @binding(0) var sceneTexture: texture_2d<f32>;
@group(0) @binding(1) var bloomTexture: texture_2d<f32>;
@group(0) @binding(2) var srcSampler: sampler;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn compositeMain(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);

    let scene = textureSampleLevel(sceneTexture, srcSampler, uv, 0.0).rgb;
    let bloom = textureSampleLevel(bloomTexture, srcSampler, uv, 0.0).rgb;
    textureStore(output, vec2i(gid.xy), vec4f(scene + bloom * params.intensity, 1.0));
}
`;

interface Mip {
    texture: GPUTexture;
    view: GPUTextureView;
    width: number;
    height: number;
}

function createBloomNode(state: State): ComputeNode {
    let thresholdPipeline: GPUComputePipeline | null = null;
    let downsamplePipeline: GPUComputePipeline | null = null;
    let upsamplePipeline: GPUComputePipeline | null = null;
    let compositePipeline: GPUComputePipeline | null = null;

    let effectComputeLayout: GPUBindGroupLayout | null = null;
    let upsampleComputeLayout: GPUBindGroupLayout | null = null;
    let compositeComputeLayout: GPUBindGroupLayout | null = null;

    let sampler: GPUSampler | null = null;
    let uniformBuffer: GPUBuffer | null = null;
    const uniformStaging = new Float32Array(UNIFORM_BUFFER_SIZE / 4);

    let intermediate: GPUTexture | null = null;
    let intermediateView: GPUTextureView | null = null;
    let mips: Mip[] = [];
    let upmips: Mip[] = [];
    let cachedWidth = 0;
    let cachedHeight = 0;
    let cachedInputView: GPUTextureView | null = null;
    let thresholdBindGroup: GPUBindGroup | null = null;
    let compositeBindGroup: GPUBindGroup | null = null;
    let downBindGroups: GPUBindGroup[] = [];
    let upBindGroups: GPUBindGroup[] = [];

    function uploadParams(idx: number, a: number, b: number, c: number, d: number): void {
        const off = (idx * SLOT_STRIDE) / 4;
        uniformStaging[off] = a;
        uniformStaging[off + 1] = b;
        uniformStaging[off + 2] = c;
        uniformStaging[off + 3] = d;
    }

    function ensureIntermediate(device: GPUDevice, width: number, height: number): boolean {
        if (intermediate && width === cachedWidth && height === cachedHeight) return false;

        intermediate?.destroy();
        intermediate = device.createTexture({
            label: "bloom-color",
            size: { width, height },
            format: COLOR_FORMAT,
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });
        intermediateView = intermediate.createView();

        for (const m of mips) m.texture.destroy();
        for (const m of upmips) m.texture.destroy();
        mips = [];
        upmips = [];

        let w = Math.max(1, width >> 1);
        let h = Math.max(1, height >> 1);
        const count = Math.min(
            MAX_MIPS,
            Math.max(2, Math.floor(Math.log2(Math.min(width, height)))),
        );

        for (let i = 0; i < count; i++) {
            const texture = device.createTexture({
                label: `bloom-mip-${i}`,
                size: { width: w, height: h },
                format: MIP_FORMAT,
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
            });
            mips.push({ texture, view: texture.createView(), width: w, height: h });
            w = Math.max(1, w >> 1);
            h = Math.max(1, h >> 1);
        }

        for (let i = 0; i < count - 1; i++) {
            const texture = device.createTexture({
                label: `bloom-upmip-${i}`,
                size: { width: mips[i].width, height: mips[i].height },
                format: MIP_FORMAT,
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
            });
            upmips.push({
                texture,
                view: texture.createView(),
                width: mips[i].width,
                height: mips[i].height,
            });
        }

        downBindGroups = [];
        for (let i = 1; i < mips.length; i++) {
            downBindGroups.push(
                device.createBindGroup({
                    layout: effectComputeLayout!,
                    entries: [
                        { binding: 0, resource: mips[i - 1].view },
                        { binding: 1, resource: sampler! },
                        {
                            binding: 2,
                            resource: {
                                buffer: uniformBuffer!,
                                offset: (1 + i) * SLOT_STRIDE,
                                size: 16,
                            },
                        },
                        { binding: 3, resource: mips[i].view },
                    ],
                }),
            );
        }

        upBindGroups = [];
        for (let i = mips.length - 2; i >= 0; i--) {
            const uIdx = mips.length + (mips.length - 2 - i);
            const srcView = i === mips.length - 2 ? mips[mips.length - 1].view : upmips[i + 1].view;
            upBindGroups.push(
                device.createBindGroup({
                    layout: upsampleComputeLayout!,
                    entries: [
                        { binding: 0, resource: srcView },
                        { binding: 1, resource: sampler! },
                        {
                            binding: 2,
                            resource: {
                                buffer: uniformBuffer!,
                                offset: uIdx * SLOT_STRIDE,
                                size: 16,
                            },
                        },
                        { binding: 3, resource: upmips[i].view },
                        { binding: 4, resource: mips[i].view },
                    ],
                }),
            );
        }

        cachedWidth = width;
        cachedHeight = height;
        cachedInputView = null;
        thresholdBindGroup = null;
        compositeBindGroup = null;
        return true;
    }

    return {
        name: "bloom",
        inputs: ["color"],
        outputs: ["color"],

        async prepare(device: GPUDevice) {
            sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

            uniformBuffer = device.createBuffer({
                label: "bloom-uniforms",
                size: UNIFORM_BUFFER_SIZE,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            effectComputeLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: { sampleType: "float" },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        sampler: { type: "filtering" },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "uniform" },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: { format: MIP_FORMAT, access: "write-only" },
                    },
                ],
            });

            upsampleComputeLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: { sampleType: "float" },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        sampler: { type: "filtering" },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "uniform" },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: { format: MIP_FORMAT, access: "write-only" },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: { sampleType: "float" },
                    },
                ],
            });

            compositeComputeLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: { sampleType: "float" },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: { sampleType: "float" },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        sampler: { type: "filtering" },
                    },
                    {
                        binding: 3,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "uniform" },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: { format: COLOR_FORMAT, access: "write-only" },
                    },
                ],
            });

            const bloomModule = device.createShaderModule({ code: bloomComputeShader });
            const upsampleModule = device.createShaderModule({ code: upsampleComputeShader });
            const compositeModule = device.createShaderModule({ code: compositeComputeShader });

            const effectPipelineLayout = device.createPipelineLayout({
                bindGroupLayouts: [effectComputeLayout],
            });
            const upsamplePipelineLayout = device.createPipelineLayout({
                bindGroupLayouts: [upsampleComputeLayout],
            });
            const compositePipelineLayout = device.createPipelineLayout({
                bindGroupLayouts: [compositeComputeLayout],
            });

            [thresholdPipeline, downsamplePipeline, upsamplePipeline, compositePipeline] =
                await Promise.all([
                    device.createComputePipelineAsync({
                        label: "bloom-threshold",
                        layout: effectPipelineLayout,
                        compute: { module: bloomModule, entryPoint: "thresholdMain" },
                    }),
                    device.createComputePipelineAsync({
                        label: "bloom-downsample",
                        layout: effectPipelineLayout,
                        compute: { module: bloomModule, entryPoint: "downsampleMain" },
                    }),
                    device.createComputePipelineAsync({
                        label: "bloom-upsample",
                        layout: upsamplePipelineLayout,
                        compute: { module: upsampleModule, entryPoint: "upsampleMain" },
                    }),
                    device.createComputePipelineAsync({
                        label: "bloom-composite",
                        layout: compositePipelineLayout,
                        compute: { module: compositeModule, entryPoint: "compositeMain" },
                    }),
                ]);
        },

        execute(ctx: ExecutionContext) {
            if (
                !thresholdPipeline ||
                !downsamplePipeline ||
                !upsamplePipeline ||
                !compositePipeline
            )
                return;

            const camEid = ActiveCamera.from(state)?.eid ?? -1;
            if (camEid < 0 || !state.hasComponent(camEid, Bloom)) return;

            const intensity = Bloom.intensity[camEid];
            if (intensity <= 0) return;

            const threshold = Bloom.threshold[camEid];
            const radius = Bloom.radius[camEid];

            const inputView = ctx.getTextureView("color");
            const colorTex = ctx.getTexture("color");
            if (!inputView || !colorTex) return;

            const { device, encoder } = ctx;
            const width = colorTex.width;
            const height = colorTex.height;

            const allocated = ensureIntermediate(device, width, height);
            if (mips.length < 2) return;

            uploadParams(0, 1 / width, 1 / height, threshold, 0.15);
            for (let i = 1; i < mips.length; i++) {
                uploadParams(1 + i, 1 / mips[i - 1].width, 1 / mips[i - 1].height, 0, 0);
            }
            const mipWeight = 1 / mips.length;
            for (let i = mips.length - 2; i >= 0; i--) {
                const uIdx = mips.length + (mips.length - 2 - i);
                uploadParams(
                    uIdx,
                    1 / mips[i + 1].width,
                    1 / mips[i + 1].height,
                    radius,
                    mipWeight,
                );
            }
            const compositeIdx = mips.length * 2;
            uploadParams(compositeIdx, intensity, 0, 0, 0);

            device.queue.writeBuffer(uniformBuffer!, 0, uniformStaging.buffer);

            const inputChanged = inputView !== cachedInputView;

            if (inputChanged || allocated) {
                thresholdBindGroup = device.createBindGroup({
                    layout: effectComputeLayout!,
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: sampler! },
                        {
                            binding: 2,
                            resource: { buffer: uniformBuffer!, offset: 0, size: 16 },
                        },
                        { binding: 3, resource: mips[0].view },
                    ],
                });
                compositeBindGroup = device.createBindGroup({
                    layout: compositeComputeLayout!,
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: upmips[0].view },
                        { binding: 2, resource: sampler! },
                        {
                            binding: 3,
                            resource: {
                                buffer: uniformBuffer!,
                                offset: compositeIdx * SLOT_STRIDE,
                                size: 16,
                            },
                        },
                        { binding: 4, resource: intermediateView! },
                    ],
                });
                cachedInputView = inputView;
            }

            let pass = beginComputePass(encoder);
            pass.setPipeline(thresholdPipeline);
            pass.setBindGroup(0, thresholdBindGroup!);
            pass.dispatchWorkgroups(Math.ceil(mips[0].width / 16), Math.ceil(mips[0].height / 16));
            pass.end();

            for (let i = 0; i < downBindGroups.length; i++) {
                pass = beginComputePass(encoder);
                pass.setPipeline(downsamplePipeline);
                pass.setBindGroup(0, downBindGroups[i]);
                pass.dispatchWorkgroups(
                    Math.ceil(mips[i + 1].width / 16),
                    Math.ceil(mips[i + 1].height / 16),
                );
                pass.end();
            }

            for (let i = 0; i < upBindGroups.length; i++) {
                const targetIdx = mips.length - 2 - i;
                pass = beginComputePass(encoder);
                pass.setPipeline(upsamplePipeline);
                pass.setBindGroup(0, upBindGroups[i]);
                pass.dispatchWorkgroups(
                    Math.ceil(upmips[targetIdx].width / 16),
                    Math.ceil(upmips[targetIdx].height / 16),
                );
                pass.end();
            }

            pass = beginComputePass(
                encoder,
                ctx.timestampWrites?.("bloom") as GPUComputePassTimestampWrites | undefined,
            );
            pass.setPipeline(compositePipeline);
            pass.setBindGroup(0, compositeBindGroup!);
            pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
            pass.end();

            ctx.setTextureView("color", intermediateView!);
        },
    };
}

export const BloomPlugin: Plugin = {
    name: "Bloom",
    components: { Bloom },
    dependencies: [ComputePlugin, RenderPlugin],
    initialize(state) {
        const compute = Compute.from(state);
        if (!compute) return;
        compute.graph.add(createBloomNode(state));
    },
};
