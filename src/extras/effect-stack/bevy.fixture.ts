import { readFileSync } from "node:fs";
import type { World } from "../../engine";
import { probeTexture } from "../../engine/runtime";
import { type Case, lookup } from "./parity.fixture";

// Bevy b3bdd75be: verbatim WESL modules linked by dropping resolved imports and
// qualifying the lens module's private constant, which otherwise collides with vignette's.
function shader(): string {
    return [
        "fullscreen_vertex_shader",
        "chromatic_aberration",
        "lens_distortion",
        "vignette",
        "post_process",
    ]
        .map((name) => {
            let source = readFileSync(
                new URL(`./bevy/${name}.wesl`, import.meta.url),
                "utf8",
            ).replace(/^import .*;\n/gm, "");
            if (name === "lens_distortion")
                source = source.replaceAll("VISUAL_THRESHOLD", "LENS_VISUAL_THRESHOLD");
            return source;
        })
        .join("\n");
}
export async function bevyPipeline(world: World): Promise<GPURenderPipeline> {
    const module = world.gpu.device.createShaderModule({ code: shader() });
    return world.gpu.device.createRenderPipelineAsync({
        label: "bevy-effect-stack",
        layout: "auto",
        vertex: { module, entryPoint: "fullscreen_vertex_shader" },
        fragment: { module, entryPoint: "fragment_main", targets: [{ format: "rgba16float" }] },
    });
}
export async function renderBevy(
    world: World,
    pipeline: GPURenderPipeline,
    c: Case,
    input: GPUTexture,
    output: GPUTexture,
): Promise<Uint16Array> {
    const device = world.gpu.device;
    // effect_stack/mod.rs: bindings 0..5, linear filtering, default clamp addressing.
    const sampler = device.createSampler({
        minFilter: "linear",
        magFilter: "linear",
        mipmapFilter: "linear",
    });
    // ShaderType layouts from chromatic_aberration.rs (16), vignette.rs (48),
    // lens_distortion.rs (32 bytes). All cases are inside the CPU upload clamps.
    const a = new Float32Array(4);
    new Uint32Array(a.buffer)[1] = c.aberration?.maxSamples ?? 0;
    a[0] = c.aberration?.intensity ?? 0;
    const v = new Float32Array(12);
    if (c.vignette) {
        const s = c.vignette;
        v.set([
            s.intensity,
            s.radius,
            s.smoothness,
            s.roundness,
            ...s.center,
            s.edgeCompensation,
            0,
            ((s.color >> 16) & 255) / 255,
            ((s.color >> 8) & 255) / 255,
            (s.color & 255) / 255,
            1,
        ]);
    }
    const l = new Float32Array(8);
    if (c.lens) {
        const s = c.lens;
        l.set([s.intensity, s.scale, ...s.multiplier, ...s.center, s.edgeCurvature, 0]);
    }
    const buffers = [a, v, l].map((data) => {
        const buffer = device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    });
    const lut = lookup(world, "custom" in c);
    const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: input.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: lut.createView() },
            ...buffers.map((buffer, i) => ({ binding: i + 3, resource: { buffer } })),
        ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: output.createView(), loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const reference = new Uint16Array((await probeTexture(world, output)).bytes as ArrayBuffer);
    for (const buffer of buffers) buffer.destroy();
    return reference;
}
