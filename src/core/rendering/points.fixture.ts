import type { Plugin, System } from "../../engine";
import { OverlaySystem, Render, Views } from "./index";

const state = {
    create: () => ({ pipelines: new Map<string, GPURenderPipeline>() }),
};

const PointsSystem: System = {
    group: "draw",
    after: [OverlaySystem],
    update(world) {
        const render = world.resource(Render);
        for (const view of world.resource(Views).values()) {
            if (!view.framebuffer) continue;
            const format = view.framebufferFormat ?? render.format;
            const key = `${format}:${!!view.depth}`;
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
                    ...(view.depth
                        ? {
                              depthStencil: {
                                  format: "depth32float" as const,
                                  depthWriteEnabled: false,
                                  depthCompare: "greater-equal" as const,
                              },
                          }
                        : {}),
                });
                pipelines.set(key, pipeline);
            }
            const pass = render.encoder!.beginRenderPass({
                colorAttachments: [{ view: view.framebuffer, loadOp: "load", storeOp: "store" }],
                ...(view.depth
                    ? { depthStencilAttachment: { view: view.depth, depthReadOnly: true } }
                    : {}),
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(
                0,
                world.gpu.device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [{ binding: 0, resource: { buffer: render.viewBuffers[view.slot] } }],
                }),
            );
            pass.draw(4);
            pass.end();
        }
    },
};

export const PointsPlugin: Plugin = { name: "PointsFixture", systems: [PointsSystem] };
