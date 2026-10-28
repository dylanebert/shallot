import { type Plugin, precompile, type Resource } from "@dylanebert/shallot";
import {
    fogReadBindGroup,
    fogReadLayout,
    readFog,
    readFogInScatter,
} from "@dylanebert/shallot/extras";
import {
    Camera,
    CorePipelinePlugin,
    initialViewSampleCounts,
    RenderContext,
    RenderPhases,
    SAMPLE_COUNT,
    type View,
} from "@dylanebert/shallot/rendering";
import tgpu from "typegpu";
import { RoomPlugin } from "./room";

export const MOTE_COUNT = 512;
export const Motes: Resource<{
    positions: GPUBuffer | null;
    visible: boolean;
    pipelines: Map<string, GPURenderPipeline>;
    groups: WeakMap<View, { buffer: GPUBuffer; group: GPUBindGroup }>;
}> = {
    create: () => ({ positions: null, visible: true, pipelines: new Map(), groups: new WeakMap() }),
};

// WGSL escape hatch: camera-facing dust quads do not use a mesh or surface contract.
const shader = `
const CAMERA_FADE_NEAR: f32 = 0.3;
const CAMERA_FADE_FAR: f32 = 0.8;
const MOTE_SIZE_DEPTH: f32 = 2.0;
struct View { clipFromWorld: mat4x4f }
@group(0) @binding(0) var<uniform> moteView: View;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
// Pure single-scattering albedo; coverage and camera fade are separate factors.
// Mineral-dust modeling assumption grounded in Saharan dust measurements at 532 nm,
// not household dust across the visible spectrum: Linke et al. (2006), doi:10.5194/acp-6-3315-2006.
const MOTE_ALBEDO = 0.98;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f, @location(1) cameraFade: f32, @location(2) world: vec3f }
@vertex fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Vertex {
    let corners = array<vec2f, 6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
    let p = positions[i];
    var clip = moteView.clipFromWorld * vec4f(p.xyz, 1);
    let size = 0.006 * clip.w / max(clip.w, MOTE_SIZE_DEPTH);
    clip.x += corners[v].x * size;
    clip.y += corners[v].y * size;
    let cameraFade = smoothstep(CAMERA_FADE_NEAR, CAMERA_FADE_FAR, clip.w);
    return Vertex(clip, corners[v], cameraFade, p.xyz);
}
@fragment fn fragment(v: Vertex) -> @location(0) vec4f {
    let alpha = max(0, 1 - dot(v.uv, v.uv));
    return vec4f(MOTE_ALBEDO * readFogInScatter(v.world) * v.cameraFade * alpha * readFog(v.world).a, 0);
}`;

export const MotesPlugin: Plugin = {
    name: "Motes",
    gpu: {},
    dependencies: [RoomPlugin, CorePipelinePlugin],
    initialize(world) {
        const s = world.resource(Motes);
        const device = world.gpu.device;
        s.positions = device.createBuffer({
            label: "dust:positions",
            size: MOTE_COUNT * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        world.own(s.positions);
        let seed = 0x12345678;
        const random = () => {
            seed ^= seed << 13;
            seed ^= seed >>> 17;
            seed ^= seed << 5;
            return (seed >>> 0) / 4294967296;
        };
        const positions = new Float32Array(MOTE_COUNT * 4);
        for (let i = 0; i < MOTE_COUNT; i++) {
            positions[i * 4] = random() * 8.4 - 4.2;
            positions[i * 4 + 1] = random() * 4.8 + 0.1;
            positions[i * 4 + 2] = random() * 23 - 11.5;
            positions[i * 4 + 3] = random();
        }
        device.queue.writeBuffer(s.positions, 0, positions);
        const pipelineFor = (format: GPUTextureFormat, samples: number) => {
            const key = `${format}:${samples}`;
            let pipeline = s.pipelines.get(key);
            if (!pipeline) {
                const module = device.createShaderModule({
                    label: "dust:billboards",
                    // The fragment read shares fog's fragment-visible layout; the quad carries its center.
                    code: tgpu.resolve({
                        externals: { readFog, readFogInScatter },
                        template: shader,
                    }),
                });
                pipeline = device.createRenderPipeline({
                    label: "dust:draw",
                    layout: device.createPipelineLayout({
                        bindGroupLayouts: [
                            device.createBindGroupLayout({
                                entries: [
                                    {
                                        binding: 0,
                                        visibility: GPUShaderStage.VERTEX,
                                        buffer: { type: "uniform" },
                                    },
                                    {
                                        binding: 1,
                                        visibility: GPUShaderStage.VERTEX,
                                        buffer: { type: "read-only-storage" },
                                    },
                                ],
                            }),
                            device.createBindGroupLayout({ entries: [] }),
                            device.createBindGroupLayout({ entries: [] }),
                            world.gpu.root.unwrap(fogReadLayout),
                        ],
                    }),
                    vertex: { module, entryPoint: "vertex" },
                    fragment: {
                        module,
                        entryPoint: "fragment",
                        targets: [
                            {
                                format,
                                blend: {
                                    color: {
                                        srcFactor: "one",
                                        dstFactor: "one",
                                        operation: "add",
                                    },
                                    alpha: {
                                        srcFactor: "zero",
                                        dstFactor: "one",
                                        operation: "add",
                                    },
                                },
                            },
                        ],
                    },
                    primitive: { topology: "triangle-list" },
                    multisample: { count: samples },
                    depthStencil: {
                        format: "depth32float",
                        depthWriteEnabled: false,
                        depthCompare: "greater-equal",
                    },
                });
                s.pipelines.set(key, pipeline);
            }
            return pipeline;
        };
        precompile(world, "motes", () =>
            [...initialViewSampleCounts(world)].map((samples) =>
                pipelineFor("rgba16float", samples),
            ),
        );
        world.resource(RenderPhases).push({
            transparent(_world, _eid, _view, phase) {
                if (s.visible) phase.add(0, 0, 0, 1.6, 0, 0);
            },
            prepareTransparent(world, eid, view) {
                const render = world.resource(RenderContext);
                const samples = world.storage(Camera).antialias.get(eid) ? SAMPLE_COUNT : 1;
                const pipeline = pipelineFor(view.framebufferFormat ?? render.format, samples);
                const buffer = render.viewBuffers[view.slot];
                if (s.groups.get(view)?.buffer !== buffer)
                    s.groups.set(view, {
                        buffer,
                        group: device.createBindGroup({
                            layout: pipeline.getBindGroupLayout(0),
                            entries: [
                                { binding: 0, resource: { buffer } },
                                { binding: 1, resource: { buffer: s.positions! } },
                            ],
                        }),
                    });
            },
            drawTransparent(world, eid, view, _phase, _start, _end, pass) {
                const render = world.resource(RenderContext);
                const samples = world.storage(Camera).antialias.get(eid) ? SAMPLE_COUNT : 1;
                pass.setPipeline(
                    s.pipelines.get(`${view.framebufferFormat ?? render.format}:${samples}`)!,
                );
                pass.setBindGroup(0, s.groups.get(view)!.group);
                pass.setBindGroup(3, fogReadBindGroup(world, eid));
                pass.draw(6, MOTE_COUNT);
            },
        });
    },
};
