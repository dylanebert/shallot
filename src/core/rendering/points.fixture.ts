import type { Plugin } from "../../engine";
import { Camera, CorePipelinePlugin, Render, RenderPhases, SAMPLE_COUNT } from "./index";

const state = {
    create: () => ({ pipelines: new Map<string, GPURenderPipeline>() }),
};

export const PointsPlugin: Plugin = {
    name: "PointsFixture",
    dependencies: [CorePipelinePlugin],
    initialize(world) {
        world.resource(RenderPhases).push({
            opaque(world, eid, view, pass) {
                const render = world.resource(Render);
                const samples = world.storage(Camera).antialias.get(eid) ? SAMPLE_COUNT : 1;
                const format = view.framebufferFormat ?? render.format;
                const key = `${format}:${samples}`;
                const pipelines = world.resource(state).pipelines;
                let pipeline = pipelines.get(key);
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
                        layout: "auto",
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
                    pipelines.set(key, pipeline);
                }
                pass.setPipeline(pipeline);
                pass.setBindGroup(
                    0,
                    world.gpu.device.createBindGroup({
                        layout: pipeline.getBindGroupLayout(0),
                        entries: [
                            { binding: 0, resource: { buffer: render.viewBuffers[view.slot] } },
                        ],
                    }),
                );
                pass.draw(4);
            },
        });
    },
};
