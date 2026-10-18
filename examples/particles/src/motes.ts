import type { Plugin, Resource } from "@dylanebert/shallot";
import { GlobalTransform } from "@dylanebert/shallot";
import {
    fogReadBindGroup,
    fogReadLayout,
    readFog,
    readFogInScatter,
} from "@dylanebert/shallot/extras";
import {
    Camera,
    CorePipelinePlugin,
    MainPassSystem,
    RenderContext,
    RenderPhases,
    SAMPLE_COUNT,
    type View,
} from "@dylanebert/shallot/rendering";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Room, RoomPlugin } from "./room";

export const MOTE_COUNT = 8192;
const layout = tgpu.bindGroupLayout({
    positions: { storage: d.arrayOf(d.vec4f), access: "mutable" },
    frame: { uniform: d.struct({ player: d.vec4f, clock: d.vec4f }) },
});
const hash = tgpu.fn(
    [d.vec3f],
    d.f32,
)((p) => {
    "use gpu";
    return std.fract(std.sin(std.dot(p, d.vec3f(127.1, 311.7, 74.7))) * 43758.5453);
});
const noise = tgpu.fn(
    [d.vec3f],
    d.f32,
)((p) => {
    "use gpu";
    const cell = std.floor(p);
    const f = std.fract(p);
    const u = f.mul(f).mul(d.vec3f(3).sub(f.mul(2)));
    return std.mix(
        std.mix(
            std.mix(hash(cell), hash(cell.add(d.vec3f(1, 0, 0))), u.x),
            std.mix(hash(cell.add(d.vec3f(0, 1, 0))), hash(cell.add(d.vec3f(1, 1, 0))), u.x),
            u.y,
        ),
        std.mix(
            std.mix(hash(cell.add(d.vec3f(0, 0, 1))), hash(cell.add(d.vec3f(1, 0, 1))), u.x),
            std.mix(hash(cell.add(d.vec3f(0, 1, 1))), hash(cell.add(d.vec3f(1, 1, 1))), u.x),
            u.y,
        ),
        u.z,
    );
});
const advance = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const i = input.gid.x;
        if (i >= MOTE_COUNT) return;
        const p = layout.$.positions[i];
        const dt = layout.$.frame.clock.x;
        const t = layout.$.frame.clock.y;
        const delta = p.xyz.sub(layout.$.frame.player.xyz);
        const distance = std.length(delta);
        const force = std.max(0, 1 - distance / 1.1) * 3;
        const field = p.xyz.mul(1.5).add(d.vec3f(0, t * 0.15, 0));
        const drift = d
            .vec3f(
                noise(field),
                noise(field.add(d.vec3f(19, 31, 7))),
                noise(field.add(d.vec3f(47, 11, 23))),
            )
            .sub(d.vec3f(0.5))
            .mul(0.09);
        const next = p.xyz.add(drift.add(delta.div(std.max(distance, 0.001)).mul(force)).mul(dt));
        layout.$.positions[i] = d.vec4f(
            std.clamp(next, d.vec3f(-3.2, 0.1, -3.2), d.vec3f(3.2, 3.1, 3.2)),
            p.w,
        );
    })
    .$name("particles:advance");

export const Motes: Resource<{
    positions: GPUBuffer | null;
    frame: GPUBuffer | null;
    compute: GPUComputePipeline | null;
    computeGroup: GPUBindGroup | null;
    visible: boolean;
    input: Float32Array;
    pipelines: Map<string, GPURenderPipeline>;
    groups: WeakMap<View, { buffer: GPUBuffer; group: GPUBindGroup }>;
}> = {
    create: () => ({
        positions: null,
        frame: null,
        compute: null,
        computeGroup: null,
        visible: true,
        input: new Float32Array(8),
        pipelines: new Map(),
        groups: new WeakMap(),
    }),
};

// WGSL escape hatch: camera-facing dust quads do not use a mesh or surface contract.
const shader = `
const CAMERA_FADE_NEAR: f32 = 0.3;
const CAMERA_FADE_FAR: f32 = 0.8;
const MOTE_SIZE_DEPTH: f32 = 2.0;
struct View { viewProj: mat4x4f }
@group(0) @binding(0) var<uniform> moteView: View;
@group(0) @binding(1) var<storage, read> positions: array<vec4f>;
const MOTE_ALBEDO = 1.5;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f, @location(1) cameraFade: f32, @location(2) world: vec3f }
@vertex fn vertex(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Vertex {
    let corners = array<vec2f, 6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
    let p = positions[i];
    var clip = moteView.viewProj * vec4f(p.xyz, 1);
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
            label: "particles:positions",
            size: MOTE_COUNT * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        s.frame = device.createBuffer({
            label: "particles:frame",
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        world.own(s.positions);
        world.own(s.frame);
        let seed = 0x12345678;
        const random = () => {
            seed ^= seed << 13;
            seed ^= seed >>> 17;
            seed ^= seed << 5;
            return (seed >>> 0) / 4294967296;
        };
        const positions = new Float32Array(MOTE_COUNT * 4);
        for (let i = 0; i < MOTE_COUNT; i++) {
            positions[i * 4] = random() * 6.4 - 3.2;
            positions[i * 4 + 1] = random() * 3 + 0.1;
            positions[i * 4 + 2] = random() * 6.4 - 3.2;
            positions[i * 4 + 3] = random();
        }
        device.queue.writeBuffer(s.positions, 0, positions);
        s.compute = world.gpu.root.unwrap(
            world.gpu.root.createComputePipeline({ compute: advance }).$name("particles:compute"),
        );
        s.computeGroup = device.createBindGroup({
            layout: s.compute.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: s.positions } },
                { binding: 1, resource: { buffer: s.frame } },
            ],
        });
        world.resource(RenderPhases).push({
            transparent(_world, _eid, _view, phase) {
                if (s.visible) phase.add(0, 0, 0, 1.6, 0, 0);
            },
            prepareTransparent(world, eid, view) {
                const render = world.resource(RenderContext);
                const samples = world.storage(Camera).antialias.get(eid) ? SAMPLE_COUNT : 1;
                const format = view.framebufferFormat ?? render.format;
                const key = `${format}:${samples}`;
                let pipeline = s.pipelines.get(key);
                if (!pipeline) {
                    const module = device.createShaderModule({
                        label: "particles:billboards",
                        // The fragment read shares fog's fragment-visible layout; the quad carries its centre.
                        code: tgpu.resolve({
                            externals: { readFog, readFogInScatter },
                            template: shader,
                        }),
                    });
                    pipeline = device.createRenderPipeline({
                        label: "particles:draw",
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
    systems: [
        {
            name: "advance-motes",
            group: "draw",
            before: [MainPassSystem],
            update(world) {
                const s = world.resource(Motes);
                const player = world.resource(Room).player;
                const position = world.storage(GlobalTransform).translation.column;
                const offset = player * 4;
                for (let axis = 0; axis < 3; axis++) s.input[axis] = position[offset + axis];
                s.input[4] = world.time.deltaTime;
                s.input[5] = world.time.elapsed;
                world.gpu.device.queue.writeBuffer(s.frame!, 0, s.input);
                const pass = world.frameEncoder()!.beginComputePass({
                    label: "particles:advance",
                    timestampWrites: world.gpu.span?.("particles:advance"),
                });
                pass.setPipeline(s.compute!);
                pass.setBindGroup(0, s.computeGroup!);
                pass.dispatchWorkgroups(MOTE_COUNT / 64);
                pass.end();
            },
        },
    ],
};
