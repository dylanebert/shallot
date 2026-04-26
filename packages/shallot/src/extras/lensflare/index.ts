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

export const LensFlare = {
    intensity: [] as number[],
    ghosts: [] as number[],
    dispersal: [] as number[],
    haloRadius: [] as number[],
    chromatic: [] as number[],
    starburst: [] as number[],
};

traits(LensFlare, {
    defaults: () => ({
        intensity: 0.1,
        ghosts: 4,
        dispersal: 0.3,
        haloRadius: 0.6,
        chromatic: 0.01,
        starburst: 0.1,
    }),
});

const lensFlareShader = /* wgsl */ `
struct Params {
    sunUV: vec2f,
    intensity: f32,
    ghostCount: f32,
    dispersal: f32,
    haloRadius: f32,
    chromatic: f32,
    starburst: f32,
    sunVisibility: f32,
    aspectRatio: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;

fn threshold(color: vec3f) -> vec3f {
    let brightness = max(max(color.r, color.g), color.b);
    let t = smoothstep(0.5, 1.0, brightness);
    return color * t;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);

    let scene = textureSampleLevel(srcTexture, srcSampler, uv, 0.0).rgb;
    let sunUV = params.sunUV;
    let ghostDir = normalize(vec2f(0.5) - sunUV);
    let ghostCount = i32(params.ghostCount);

    var flare = vec3f(0.0);

    for (var i = 0; i < ghostCount; i++) {
        let offset = f32(i + 1) * params.dispersal;
        let sampleUV = sunUV + ghostDir * offset;
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) {
            continue;
        }
        let chromaticShift = params.chromatic * f32(i + 1);
        let r = threshold(textureSampleLevel(srcTexture, srcSampler, sampleUV + vec2f(chromaticShift, 0.0), 0.0).rgb).r;
        let g = threshold(textureSampleLevel(srcTexture, srcSampler, sampleUV, 0.0).rgb).g;
        let b = threshold(textureSampleLevel(srcTexture, srcSampler, sampleUV - vec2f(chromaticShift, 0.0), 0.0).rgb).b;
        let ghost = vec3f(r, g, b);
        let falloff = 1.0 - length(sampleUV - vec2f(0.5)) * 2.0;
        flare += ghost * max(0.0, falloff);
    }

    let toCenter = vec2f(0.5) - uv;
    var aspectCorrected = toCenter;
    aspectCorrected.x *= params.aspectRatio;
    let dist = length(aspectCorrected);
    let halo = smoothstep(params.haloRadius - 0.05, params.haloRadius, dist)
             * smoothstep(params.haloRadius + 0.05, params.haloRadius, dist);
    let haloUV = sunUV + ghostDir * params.haloRadius * 2.0;
    if (haloUV.x >= 0.0 && haloUV.x <= 1.0 && haloUV.y >= 0.0 && haloUV.y <= 1.0) {
        let haloColor = threshold(textureSampleLevel(srcTexture, srcSampler, haloUV, 0.0).rgb);
        flare += haloColor * halo * 0.5;
    }

    let sunToPixel = uv - sunUV;
    let sunDist = length(sunToPixel);
    let angle = atan2(sunToPixel.y, sunToPixel.x);
    let burst = (cos(angle * 16.0) * 0.5 + 0.5) * (cos(angle * 23.0 + 1.7) * 0.5 + 0.5);
    let burstFalloff = exp(-sunDist * 4.0);
    flare += vec3f(1.0, 0.95, 0.8) * burst * burstFalloff * params.starburst;

    let result = scene + flare * params.intensity * params.sunVisibility;
    textureStore(output, vec2i(gid.xy), vec4f(result, 1.0));
}
`;

function createLensFlareNode(state: State): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let layout: GPUBindGroupLayout | null = null;
    let sampler: GPUSampler | null = null;
    let uniformBuffer: GPUBuffer | null = null;
    const uniformStaging = new Float32Array(16);

    let intermediate: GPUTexture | null = null;
    let intermediateView: GPUTextureView | null = null;
    let cachedWidth = 0;
    let cachedHeight = 0;
    let cachedInputView: GPUTextureView | null = null;
    let bindGroup: GPUBindGroup | null = null;

    return {
        name: "lensflare",
        inputs: ["color"],
        outputs: ["color"],

        async prepare(device: GPUDevice) {
            sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
            uniformBuffer = device.createBuffer({
                label: "lensflare-uniforms",
                size: 64,
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
                        storageTexture: { format: COLOR_FORMAT, access: "write-only" },
                    },
                ],
            });
            const module = device.createShaderModule({ code: lensFlareShader });
            pipeline = await device.createComputePipelineAsync({
                label: "lensflare",
                layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
                compute: { module, entryPoint: "main" },
            });
        },

        execute(ctx: ExecutionContext) {
            if (!pipeline || !layout || !sampler || !uniformBuffer) return;

            const camEid = ActiveCamera.from(state)?.eid ?? -1;
            if (camEid < 0 || !state.hasComponent(camEid, LensFlare)) return;

            const intensity = LensFlare.intensity[camEid];
            if (intensity <= 0) return;

            const render = Render.from(state);
            if (!render) return;

            const inputView = ctx.getTextureView("color");
            const colorTex = ctx.getTexture("color");
            if (!inputView || !colorTex) return;

            const { device, encoder } = ctx;
            const width = colorTex.width;
            const height = colorTex.height;

            let allocated = false;
            if (!intermediate || width !== cachedWidth || height !== cachedHeight) {
                intermediate?.destroy();
                intermediate = device.createTexture({
                    label: "lensflare-color",
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
                allocated = true;
            }

            const sun = projectActiveSun(state, render.viewProj);

            uniformStaging[0] = sun.u;
            uniformStaging[1] = sun.v;
            uniformStaging[2] = intensity;
            uniformStaging[3] = LensFlare.ghosts[camEid];
            uniformStaging[4] = LensFlare.dispersal[camEid];
            uniformStaging[5] = LensFlare.haloRadius[camEid];
            uniformStaging[6] = LensFlare.chromatic[camEid];
            uniformStaging[7] = LensFlare.starburst[camEid];
            uniformStaging[8] = sun.visibility;
            uniformStaging[9] = width / height;
            uniformStaging[10] = 0;
            uniformStaging[11] = 0;
            device.queue.writeBuffer(uniformBuffer, 0, uniformStaging.buffer, 0, 64);

            if (allocated || inputView !== cachedInputView) {
                bindGroup = device.createBindGroup({
                    layout,
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: sampler },
                        { binding: 2, resource: { buffer: uniformBuffer } },
                        { binding: 3, resource: intermediateView! },
                    ],
                });
                cachedInputView = inputView;
            }

            const pass = beginComputePass(
                encoder,
                ctx.timestampWrites?.("lensflare") as GPUComputePassTimestampWrites | undefined,
            );
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup!);
            pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
            pass.end();

            ctx.setTextureView("color", intermediateView!);
        },
    };
}

export const LensFlarePlugin: Plugin = {
    name: "LensFlare",
    components: { LensFlare },
    dependencies: [ComputePlugin, RenderPlugin],
    initialize(state) {
        const compute = Compute.from(state);
        if (!compute) return;
        compute.graph.add(createLensFlareNode(state));
    },
};
