import {
    type State,
    type Plugin,
    type Config,
    Compute,
    ComputePlugin,
    ViewportPlugin,
    type ComputeNode,
    type ExecutionContext,
} from "@dylanebert/shallot";

const PARTICLE_COUNT = 4096;

const simulateShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> entityIds: array<u32>;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&entityIds)) { return; }

    let eid = entityIds[idx];
    var pos = positions[eid];
    let vel = velocities[eid];

    pos += vel * 0.016;

    if (pos.x > 1.0) { pos.x = -1.0; }
    if (pos.x < -1.0) { pos.x = 1.0; }
    if (pos.y > 1.0) { pos.y = -1.0; }
    if (pos.y < -1.0) { pos.y = 1.0; }

    positions[eid] = pos;
}
`;

const drawShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> entityIds: array<u32>;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> pixelSize: vec2<f32>;

var<private> quadOffsets: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5),
    vec2(-0.5, 0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let eid = entityIds[instance];
    let pos = positions[eid];
    let color = colors[eid];
    let offset = quadOffsets[vertexIndex] * pixelSize * 3.0;

    var output: VertexOutput;
    output.position = vec4<f32>(pos + offset, 0.0, 1.0);
    output.color = color;
    return output;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
`;

interface ParticleState {
    entityIds: GPUBuffer;
    positions: GPUBuffer;
    velocities: GPUBuffer;
    colors: GPUBuffer;
    count: number;
}

function createSimulateNode(state: ParticleState): ComputeNode {
    let pipeline: GPUComputePipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;

    return {
        name: "simulate",
        inputs: [],
        outputs: ["positions"],

        execute(ctx: ExecutionContext) {
            const { device, encoder } = ctx;

            if (!pipeline) {
                const module = device.createShaderModule({ code: simulateShader });
                pipeline = device.createComputePipeline({
                    layout: "auto",
                    compute: { module, entryPoint: "main" },
                });
                bindGroup = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: state.entityIds } },
                        { binding: 1, resource: { buffer: state.positions } },
                        { binding: 2, resource: { buffer: state.velocities } },
                    ],
                });
            }

            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup!);
            pass.dispatchWorkgroups(Math.ceil(state.count / 64));
            pass.end();
        },
    };
}

function createDrawNode(state: ParticleState): ComputeNode {
    let pipeline: GPURenderPipeline | null = null;
    let pixelSizeBuffer: GPUBuffer | null = null;
    let bindGroup: GPUBindGroup | null = null;

    return {
        name: "draw",
        inputs: ["positions"],
        outputs: ["framebuffer"],

        execute(ctx: ExecutionContext) {
            const { device, encoder, canvasView, format, context } = ctx;
            const canvas = context.canvas;
            const w = canvas.width;
            const h = canvas.height;

            if (!pipeline) {
                const module = device.createShaderModule({ code: drawShader });
                pipeline = device.createRenderPipeline({
                    layout: "auto",
                    vertex: { module, entryPoint: "vs" },
                    fragment: {
                        module,
                        entryPoint: "fs",
                        targets: [{ format: format as GPUTextureFormat }],
                    },
                    primitive: { topology: "triangle-list" },
                });
                pixelSizeBuffer = device.createBuffer({
                    size: 8,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                bindGroup = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: state.entityIds } },
                        { binding: 1, resource: { buffer: state.positions } },
                        { binding: 2, resource: { buffer: state.colors } },
                        { binding: 3, resource: { buffer: pixelSizeBuffer } },
                    ],
                });
            }

            device.queue.writeBuffer(pixelSizeBuffer!, 0, new Float32Array([2 / w, 2 / h]));

            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: canvasView,
                        clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });

            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup!);
            pass.draw(6, state.count);
            pass.end();
        },
    };
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
    const m = l - c / 2;

    let r = 0,
        g = 0,
        b = 0;
    if (h < 1 / 6) {
        r = c;
        g = x;
    } else if (h < 2 / 6) {
        r = x;
        g = c;
    } else if (h < 3 / 6) {
        g = c;
        b = x;
    } else if (h < 4 / 6) {
        g = x;
        b = c;
    } else if (h < 5 / 6) {
        r = x;
        b = c;
    } else {
        r = c;
        b = x;
    }

    return [r + m, g + m, b + m];
}

export const ParticlePlugin: Plugin = {
    name: "Particle",
    dependencies: [ComputePlugin],

    initialize(state: State) {
        const compute = Compute.from(state);
        if (!compute) return;

        const { device, graph } = compute;

        const entityIds = new Uint32Array(PARTICLE_COUNT);
        const positions = new Float32Array(PARTICLE_COUNT * 2);
        const velocities = new Float32Array(PARTICLE_COUNT * 2);
        const colors = new Float32Array(PARTICLE_COUNT * 4);

        for (let i = 0; i < PARTICLE_COUNT; i++) {
            entityIds[i] = i;

            positions[i * 2] = Math.random() * 2 - 1;
            positions[i * 2 + 1] = Math.random() * 2 - 1;

            const angle = Math.random() * Math.PI * 2;
            const speed = 0.1 + Math.random() * 0.3;
            velocities[i * 2] = Math.cos(angle) * speed;
            velocities[i * 2 + 1] = Math.sin(angle) * speed;

            const hue = i / PARTICLE_COUNT;
            const [r, g, b] = hslToRgb(hue, 0.8, 0.6);
            colors[i * 4] = r;
            colors[i * 4 + 1] = g;
            colors[i * 4 + 2] = b;
            colors[i * 4 + 3] = 1;
        }

        const entityIdBuffer = device.createBuffer({
            label: "entityIds",
            size: PARTICLE_COUNT * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        device.queue.writeBuffer(entityIdBuffer, 0, entityIds);

        const positionBuffer = device.createBuffer({
            size: positions.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(positionBuffer, 0, positions);

        const velocityBuffer = device.createBuffer({
            size: velocities.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(velocityBuffer, 0, velocities);

        const colorBuffer = device.createBuffer({
            size: colors.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(colorBuffer, 0, colors);

        const particleState: ParticleState = {
            entityIds: entityIdBuffer,
            positions: positionBuffer,
            velocities: velocityBuffer,
            colors: colorBuffer,
            count: PARTICLE_COUNT,
        };

        graph.add(createSimulateNode(particleState));
        graph.add(createDrawNode(particleState));
    },
};

export const config: Config = {
    plugins: [ComputePlugin, ViewportPlugin, ParticlePlugin],
    defaults: false,
};
