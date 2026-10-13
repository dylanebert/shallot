import { afterAll, expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import { compileGpuFile, createTestGpuApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { PointLightGpu } from "./lighting";
import {
    pointCastersSchema,
    pointShadowRef,
    SunShadow,
    sampleSunShadow,
    tileRectsSchema,
} from "./shade";
import { EDGE_TEXELS, MAX_POINT_CASTERS } from "./shadows";

setDefaultTimeout(CEILING.node);
const SIDE = 256;
const OFFSETS = Array.from({ length: 33 }, (_, i) => (i - 16) / 4);
const apps: Awaited<ReturnType<typeof createTestGpuApp>>[] = [];
const fixture = compileGpuFile(import.meta.path, async () => {
    const app = await createTestGpuApp({
        defaults: false,
        plugins: [{ name: "ShadowProfileDevice", gpu: {} }],
    });
    apps.push(app);
    return app;
});
afterAll(() => disposeGpuApps(apps));

// A binary depth edge at an integer texel boundary, evaluated with the comparison sampler's
// bilinear footprint. Both the GPU receiver and the independent CPU reference see this same map.
function hardware(u: number, edge: number): number {
    return Math.max(0, Math.min(1, edge + 0.5 - u * SIDE));
}
function castano(offset: number): number {
    const pixel = SIDE / 2 + offset;
    const base = Math.floor(pixel + 0.5);
    const s = pixel + 0.5 - base;
    const weights = [4 - 3 * s, 7, 1 + 3 * s];
    const taps = [(3 - 2 * s) / weights[0] - 2, (3 + s) / 7, s / weights[2] + 2];
    // The map is constant vertically: Bevy's three vertical weights sum to 12.
    return (
        taps.reduce(
            (sum, tap, i) => sum + weights[i] * hardware((base - 0.5 + tap) / SIDE, SIDE / 2),
            0,
        ) / 12
    );
}
const POSITIONS = [
    [0.125, -0.375],
    [-0.125, 0.375],
    [0.625, 0.125],
    [-0.375, -0.625],
    [-0.625, 0.625],
    [-0.875, -0.125],
    [0.375, 0.875],
    [0.875, -0.875],
];
const COEFFICIENTS = [
    0.157112, 0.157112, 0.138651, 0.130251, 0.114946, 0.114946, 0.107982, 0.079001,
];
const TAN_HALF = 1 + (2 * EDGE_TEXELS) / SIDE;
function pointGaussian(offset: number): number {
    const original = [(0.5 - (2 * offset) / SIDE) * TAN_HALF, 0.7, 1];
    const length = Math.hypot(...original);
    const texelWorld = (length * 2 * TAN_HALF) / SIDE;
    const biased = original.map((v) => v * (1 - 0.04 / length));
    biased[1] += 0.3 * Math.SQRT2 * texelWorld;
    const n = Math.hypot(...biased);
    const [x, y, z] = biased.map((v) => v / n);
    const a = -1 / (1 + z);
    const b = x * y * a;
    const bx = [1 + x * x * a, b, -x];
    const by = [b, 1 + y * y * a, -y];
    // Bevy shadows.wgsl: distance_to_light is the unbiased maximum absolute axis.
    const scale = 0.003 * Math.max(...original.map(Math.abs));
    return POSITIONS.reduce((sum, [u, v], i) => {
        const tap = biased.map((value, axis) => value + scale * (u * bx[axis] + v * by[axis]));
        // All these off-centre directions remain on +Z; its atlas projection is (-x, y, z).
        const uv = 0.5 - (tap[0] / (tap[2] * TAN_HALF)) * 0.5;
        return sum + COEFFICIENTS[i] * hardware(uv, SIDE / 4);
    }, 0);
}

async function profile(kind: "directional" | "spot" | "point"): Promise<Float32Array> {
    const device = fixture().world.gpu.device;
    const resources: (GPUBuffer | GPUTexture)[] = [];
    device.pushErrorScope("validation");
    try {
        const texture = device.createTexture({
            size: [SIDE, SIDE],
            format: "depth32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        resources.push(texture);
        const fill = device.createShaderModule({
            code: `
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
    return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @builtin(frag_depth) f32 {
    return select(0.8, 0.2, p.x < ${kind === "point" ? SIDE / 4 : SIDE / 2}.0);
}`,
        });
        const fillPipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: { module: fill, entryPoint: "vs" },
            fragment: { module: fill, entryPoint: "fs", targets: [] },
            depthStencil: {
                format: "depth32float",
                depthWriteEnabled: true,
                depthCompare: "always",
            },
        });
        const code = tgpu.resolve({
            names: "strict",
            externals: {
                sampleSunShadow,
                pointShadowOf: pointShadowRef(),
                SunShadow,
                PointCasters: pointCastersSchema(),
                TileRects: tileRectsSchema(MAX_POINT_CASTERS * 6),
                PointLightGpu,
            },
            template: `
@group(0) @binding(0) var shadowMap: texture_depth_2d;
@group(0) @binding(1) var pointAtlas: texture_depth_2d;
@group(0) @binding(2) var shadowSamp: sampler_comparison;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<private> sunShadow: SunShadow;
var<private> pointShadows: PointCasters;
var<private> tileRects: TileRects;
struct ProfileView { right: vec4f, up: vec4f, eye: vec4f }
var<private> view: ProfileView;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
    let offset = (f32(id.x) - 16.0) * 0.25;
    let uv = 0.5 + offset / ${SIDE}.0;
    let identity = mat4x4f(vec4f(1,0,0,0),vec4f(0,1,0,0),vec4f(0,0,1,0),vec4f(0,0,0,1));
    view.right = vec4f(1,0,0,0); view.up = vec4f(0,1,0,0);
    sunShadow.enabled = 1.0; sunShadow.count = 1.0;
    sunShadow.lightDirection = vec4f(0,0,1,0);
    sunShadow.cascades[0].lightViewProj = identity;
    sunShadow.cascades[0].rect = vec4f(0,0,1,1);
    sunShadow.cascades[0].far = 100.0;
    for (var face = 0u; face < 6u; face++) { tileRects.rects[face] = vec4f(0,0,1,1); }
    var light: PointLightGpu;
    light.color.a = 42.0;
    pointShadows.casters[0].pos = vec4f(0,0,0,42);
    pointShadows.casters[0].nf = vec4f(0.5,100,0.04,0.3);
    ${
        kind === "directional"
            ? "output[id.x] = sampleSunShadow(vec3f(uv * 2.0 - 1.0,0,0.5),vec3f(0,1,0),1u);"
            : kind === "spot"
              ? `
    pointShadows.casters[0].nf = vec4f(1,100,0,0);
    pointShadows.casters[0].spotA = vec4f(1,0,0,1);
    pointShadows.casters[0].spotB = vec4f(0,1,0,0);
    pointShadows.casters[0].spotC = vec4f(0,0,1,0);
    output[id.x] = pointShadowOf(light,vec3f(0,1,0),vec3f((uv*2.0-1.0)*2.0,0,2),1u);`
              : `
    output[id.x] = pointShadowOf(light,vec3f(0,1,0),vec3f((0.5-2.0*offset/${SIDE}.0)*${TAN_HALF},0.7,1),1u);`
    }
}`,
        });
        const module = device.createShaderModule({ code });
        const pipeline = device.createComputePipeline({
            layout: "auto",
            compute: { module, entryPoint: "main" },
        });
        const output = device.createBuffer({
            size: OFFSETS.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const readback = device.createBuffer({
            size: OFFSETS.length * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        resources.push(output, readback);
        // Auto layouts omit the unused texture in each receiver; inspect the kind rather than adding unused bindings.
        const entries: GPUBindGroupEntry[] = [
            { binding: kind === "directional" ? 0 : 1, resource: texture.createView() },
            {
                binding: 2,
                resource: device.createSampler({
                    compare: "greater-equal",
                    magFilter: "linear",
                    minFilter: "linear",
                }),
            },
            { binding: 3, resource: { buffer: output } },
        ];
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const encoder = device.createCommandEncoder();
        const render = encoder.beginRenderPass({
            colorAttachments: [],
            depthStencilAttachment: {
                view: texture.createView(),
                depthLoadOp: "clear",
                depthStoreOp: "store",
                depthClearValue: 0,
            },
        });
        render.setPipeline(fillPipeline);
        render.draw(3);
        render.end();
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, group);
        compute.dispatchWorkgroups(OFFSETS.length);
        compute.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, OFFSETS.length * 4);
        device.queue.submit([encoder.finish()]);
        const error = await device.popErrorScope();
        expect(error?.message).toBeUndefined();
        await readback.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(readback.getMappedRange()).slice();
        readback.unmap();
        return result;
    } finally {
        for (const resource of resources) resource.destroy();
    }
}

for (const kind of ["directional", "spot", "point"] as const) {
    test(`${kind} Gaussian edge profile matches Bevy's CPU kernel on a binary shadow map`, async () => {
        const actual = await profile(kind);
        const expected = OFFSETS.map(kind === "point" ? pointGaussian : castano);
        const maxError = Math.max(...expected.map((value, i) => Math.abs(value - actual[i])));
        console.log(`${kind} Bevy edge-profile maximum error: ${maxError}`);
        // WebGPU comparison filtering may quantize sub-texel weights to eight fractional bits.
        expect(maxError).toBeLessThan(1 / 256);
    });
}
