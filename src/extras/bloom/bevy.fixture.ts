import { readFileSync } from "node:fs";
import type { World } from "../../engine";

// Bevy b3bdd75be bloom.wesl: resolve WESL's FIRST_DOWNSAMPLE, UNIFORM_SCALE and
// USE_THRESHOLD conditionals by hand; functions guarded without braces may remain unused.
function shader(first: boolean): string {
    let code = readFileSync(new URL("./bevy/bloom.wesl", import.meta.url), "utf8");
    code = code.replace(/@if\(UNIFORM_SCALE\) \{([\s\S]*?)\} @else \{([\s\S]*?)\n\}/, "$1");
    code = code.replace(
        /@if\(FIRST_DOWNSAMPLE\) \{([\s\S]*?)\} @else \{([\s\S]*?)\n\}/,
        first ? "$1" : "$2",
    );
    code = code.replace(/@if\(FIRST_DOWNSAMPLE\)\n/g, "");
    code = code.replace("@if(USE_THRESHOLD)\n    sample = soft_threshold(sample);", "");
    return (
        readFileSync(new URL("./bevy/fullscreen_vertex_shader.wesl", import.meta.url), "utf8") +
        code
    );
}
export async function bevyPipelines(world: World) {
    const device = world.gpu.device;
    const layout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const make = (first: boolean, entryPoint: string, blend?: GPUBlendState) => {
        const module = device.createShaderModule({ code: shader(first) });
        return device.createRenderPipelineAsync({
            layout: pipelineLayout,
            vertex: { module, entryPoint: "fullscreen_vertex_shader" },
            fragment: { module, entryPoint, targets: [{ format: "rgba16float", blend }] },
        });
    };
    return {
        layout,
        first: await make(true, "downsample_first"),
        down: await make(false, "downsample"),
        up: await make(false, "upsample", {
            color: { srcFactor: "constant", dstFactor: "one-minus-constant", operation: "add" },
            alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
        }),
    };
}
export async function renderBevy(
    world: World,
    pipelines: Awaited<ReturnType<typeof bevyPipelines>>,
    input: GPUTexture,
    output: GPUTexture,
) {
    const device = world.gpu.device;
    // bloom/mod.rs at NATURAL: max_mip_dimension tall, ilog2(max).max(2)-1 mips.
    const count = 8;
    const texture = device.createTexture({
        size: [Math.round((input.width * 512) / input.height), 512],
        mipLevelCount: count,
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const views = Array.from({ length: count }, (_, mip) =>
        texture.createView({ baseMipLevel: mip, mipLevelCount: 1 }),
    );
    const buffer = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
        buffer,
        0,
        new Float32Array([0, 0, 0, 25000, 0, 0, 1, 1, 1, 1, input.width / input.height, 0]),
    );
    const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear" });
    const group = (view: GPUTextureView) =>
        device.createBindGroup({
            layout: pipelines.layout,
            entries: [
                { binding: 0, resource: view },
                { binding: 1, resource: sampler },
                { binding: 2, resource: { buffer } },
            ],
        });
    const encoder = device.createCommandEncoder();
    const render = (
        pipeline: GPURenderPipeline,
        input: GPUTextureView,
        output: GPUTextureView,
        blend?: number,
    ) => {
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                { view: output, loadOp: blend === undefined ? "clear" : "load", storeOp: "store" },
            ],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group(input));
        if (blend !== undefined) pass.setBlendConstant({ r: blend, g: blend, b: blend, a: 1 });
        pass.draw(3);
        pass.end();
    };
    for (let mip = 0; mip < count; mip++)
        render(
            mip === 0 ? pipelines.first : pipelines.down,
            mip === 0 ? input.createView() : views[mip - 1]!,
            views[mip]!,
        );
    for (let mip = count - 1; mip > 0; mip--) {
        // Independently transcribed compute_blend_factor at NATURAL (high-pass term is 1).
        const factor = 0.15 + (1 - (1 - mip / 7) ** (1 / (1 - 0.95))) * 0.7 * (1 - 0.15);
        render(pipelines.up, views[mip]!, views[mip - 1]!, factor);
    }
    encoder.copyTextureToTexture({ texture: input }, { texture: output }, [
        input.width,
        input.height,
    ]);
    render(pipelines.up, views[0]!, output.createView(), 0.15);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    texture.destroy();
    buffer.destroy();
}
