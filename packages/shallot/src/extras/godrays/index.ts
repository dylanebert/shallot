import { traits, type Plugin, type State } from "../../engine";
import {
    beginComputePass,
    Compute,
    ComputePlugin,
    type ComputeNode,
    type ExecutionContext,
} from "../../standard/compute";
import { ActiveCamera, Render, RenderPlugin } from "../../standard/render";
import { COLOR_FORMAT, projectActiveSun } from "../../standard/render/core";

export const GodRays = {
    intensity: [] as number[],
    samples: [] as number[],
    decay: [] as number[],
    density: [] as number[],
};

traits(GodRays, {
    defaults: () => ({
        intensity: 0.2,
        samples: 32,
        decay: 0.97,
        density: 1.0,
    }),
});

const godRaysShader = /* wgsl */ `
struct Params {
    sunUV: vec2f,
    intensity: f32,
    samples: f32,
    decay: f32,
    density: f32,
    sunVisibility: f32,
    _pad: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var zTexture: texture_depth_2d;
@group(0) @binding(4) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);

    let scene = textureSampleLevel(srcTexture, srcSampler, uv, 0.0).rgb;
    let sunUV = params.sunUV;
    let sampleCount = i32(params.samples);
    let delta = (uv - sunUV) * params.density / f32(sampleCount);
    let zDims = textureDimensions(zTexture);

    var sampleUV = uv;
    var weight = 1.0;
    var accum = vec3f(0.0);

    for (var i = 0; i < sampleCount; i++) {
        sampleUV -= delta;
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) {
            break;
        }
        let texCoord = vec2u(u32(sampleUV.x * f32(zDims.x)), u32(sampleUV.y * f32(zDims.y)));
        let depth = textureLoad(zTexture, texCoord, 0);
        if (depth >= 0.999) {
            let sampleColor = textureSampleLevel(srcTexture, srcSampler, sampleUV, 0.0).rgb;
            accum += sampleColor * weight;
        }
        weight *= params.decay;
    }

    let result = scene + accum * params.intensity * params.sunVisibility / f32(sampleCount);
    textureStore(output, vec2i(gid.xy), vec4f(result, 1.0));
}
`;

function createGodRaysNode(state: State): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let layout: GPUBindGroupLayout | null = null;
    let sampler: GPUSampler | null = null;
    let uniformBuffer: GPUBuffer | null = null;
    const uniformStaging = new Float32Array(8);

    let intermediate: GPUTexture | null = null;
    let intermediateView: GPUTextureView | null = null;
    let cachedWidth = 0;
    let cachedHeight = 0;
    let cachedInputView: GPUTextureView | null = null;
    let cachedZView: GPUTextureView | null = null;
    let bindGroup: GPUBindGroup | null = null;

    return {
        name: "godrays",
        inputs: ["color", "z"],
        outputs: ["color"],

        async prepare(device: GPUDevice) {
            sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
            uniformBuffer = device.createBuffer({
                label: "godrays-uniforms",
                size: 48,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            layout = device.createBindGroupLayout({
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
                        texture: { sampleType: "depth" },
                    },
                    {
                        binding: 4,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: { format: COLOR_FORMAT, access: "write-only" },
                    },
                ],
            });
            const module = device.createShaderModule({ code: godRaysShader });
            pipeline = await device.createComputePipelineAsync({
                label: "godrays",
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module, entryPoint: "main" },
            });
        },

        execute(ctx: ExecutionContext) {
            if (!pipeline || !layout || !sampler || !uniformBuffer) return;

            const camEid = ActiveCamera.from(state)?.eid ?? -1;
            if (camEid < 0 || !state.hasComponent(camEid, GodRays)) return;

            const intensity = GodRays.intensity[camEid];
            if (intensity <= 0) return;

            const render = Render.from(state);
            if (!render) return;

            const inputView = ctx.getTextureView("color");
            const zView = ctx.getTextureView("z");
            const colorTex = ctx.getTexture("color");
            if (!inputView || !zView || !colorTex) return;

            const { device, encoder } = ctx;
            const width = colorTex.width;
            const height = colorTex.height;

            let allocated = false;
            if (!intermediate || width !== cachedWidth || height !== cachedHeight) {
                intermediate?.destroy();
                intermediate = device.createTexture({
                    label: "godrays-color",
                    size: { width, height },
                    format: COLOR_FORMAT,
                    usage:
                        GPUTextureUsage.STORAGE_BINDING |
                        GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.RENDER_ATTACHMENT,
                });
                intermediateView = intermediate.createView();
                cachedWidth = width;
                cachedHeight = height;
                cachedInputView = null;
                cachedZView = null;
                allocated = true;
            }

            const sun = projectActiveSun(state, render.viewProj);

            uniformStaging[0] = sun.u;
            uniformStaging[1] = sun.v;
            uniformStaging[2] = intensity;
            uniformStaging[3] = GodRays.samples[camEid];
            uniformStaging[4] = GodRays.decay[camEid];
            uniformStaging[5] = GodRays.density[camEid];
            uniformStaging[6] = sun.visibility;
            uniformStaging[7] = 0;
            device.queue.writeBuffer(uniformBuffer, 0, uniformStaging.buffer, 0, 32);

            if (allocated || inputView !== cachedInputView || zView !== cachedZView) {
                bindGroup = device.createBindGroup({
                    layout,
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: sampler },
                        { binding: 2, resource: { buffer: uniformBuffer } },
                        { binding: 3, resource: zView },
                        { binding: 4, resource: intermediateView! },
                    ],
                });
                cachedInputView = inputView;
                cachedZView = zView;
            }

            const pass = beginComputePass(
                encoder,
                ctx.timestampWrites?.("godrays") as GPUComputePassTimestampWrites | undefined,
            );
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup!);
            pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
            pass.end();

            ctx.setTextureView("color", intermediateView!);
        },
    };
}

export const GodRaysPlugin: Plugin = {
    name: "GodRays",
    components: { GodRays },
    dependencies: [ComputePlugin, RenderPlugin],
    initialize(state) {
        const compute = Compute.from(state);
        if (!compute) return;
        compute.graph.add(createGodRaysNode(state));
    },
};
