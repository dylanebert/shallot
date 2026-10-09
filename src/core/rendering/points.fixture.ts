import type { Plugin } from "../../engine";
import {
    Camera,
    CorePipelinePlugin,
    RenderContext,
    RenderPhases,
    SAMPLE_COUNT,
    type View,
} from "./index";

export const pointsState = {
    create: () => ({
        pipelines: new Map<string, GPURenderPipeline>(),
        views: new WeakMap<
            View,
            {
                format: GPUTextureFormat;
                samples: number;
                pipeline: GPURenderPipeline;
                buffer: GPUBuffer | null;
                group: GPUBindGroup | null;
            }
        >(),
        layout: null as GPUBindGroupLayout | null,
        pipelineLayout: null as GPUPipelineLayout | null,
        binding: { buffer: null as unknown as GPUBuffer },
        descriptor: {
            layout: null as unknown as GPUBindGroupLayout,
            entries: [],
        } as GPUBindGroupDescriptor,
    }),
};

export const PointsPlugin: Plugin = {
    gpu: {},
    name: "PointsFixture",
    dependencies: [CorePipelinePlugin],
    initialize(world) {
        const cached = world.resource(pointsState);
        const device = world.gpu.device;
        cached.layout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            ],
        });
        cached.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [cached.layout] });
        cached.descriptor.layout = cached.layout;
        cached.descriptor.entries = [{ binding: 0, resource: cached.binding }];
        world.resource(RenderPhases).push({
            opaque(world, eid, view, pass) {
                const render = world.resource(RenderContext);
                const samples = world.storage(Camera).antialias.get(eid) ? SAMPLE_COUNT : 1;
                const format = view.framebufferFormat ?? render.format;
                let record = cached.views.get(view);
                if (!record || record.format !== format || record.samples !== samples) {
                    const key = `${format}:${samples}`;
                    let pipeline = cached.pipelines.get(key);
                    if (!pipeline) {
                        // WGSL escape hatch: point-list has no mesh or surface contract.
                        const module = world.gpu.device.createShaderModule({
                            code: `
struct View { viewProj: mat4x4f }
@group(0) @binding(0) var<uniform> view: View;
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let points = array<vec3f, 4>(vec3f(-1, 0, 0), vec3f(0, 0, 0),
        vec3f(1, 0, 0), vec3f(0, 0.25, 2));
    return view.viewProj * vec4f(points[i], 1);
}
@fragment fn fragment() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }
`,
                        });
                        pipeline = world.gpu.device.createRenderPipeline({
                            layout: cached.pipelineLayout!,
                            vertex: { module, entryPoint: "vertex" },
                            fragment: { module, entryPoint: "fragment", targets: [{ format }] },
                            primitive: { topology: "point-list" },
                            multisample: { count: samples },
                            depthStencil: {
                                format: "depth32float",
                                depthWriteEnabled: true,
                                depthCompare: "greater-equal",
                            },
                        });
                        cached.pipelines.set(key, pipeline);
                    }
                    if (record) {
                        record.format = format;
                        record.samples = samples;
                        record.pipeline = pipeline;
                    } else {
                        record = { format, samples, pipeline, buffer: null, group: null };
                        cached.views.set(view, record);
                    }
                }
                const buffer = render.viewBuffers[view.slot];
                if (record.buffer !== buffer) {
                    cached.binding.buffer = buffer;
                    record.group = device.createBindGroup(cached.descriptor);
                    record.buffer = buffer;
                }
                pass.setPipeline(record.pipeline);
                pass.setBindGroup(0, record.group!);
                pass.draw(4);
            },
        });
    },
};
