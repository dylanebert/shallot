import { expect, test } from "bun:test";
import { build } from "../../engine/app";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after 5000 ms`)), 5000);
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}
function median(values: number[]): number {
    return values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)]!;
}

// Paired prototypes isolate the changed payload: both use the same independent row permutations,
// pose and Part records, vertex arithmetic, survivor order and shadow combo. Not a scene benchmark.
test("measure vertex and compaction/regather costs of the approved instance payload", async () => {
    const app = await build({
        defaults: false,
        plugins: [{ name: "InstancePayloadProbe", features: ["timestamp-query"] }],
    });
    const state = app.state;
    const device = state.gpu.device;
    const owned: { destroy(): void }[] = [];
    const own = <T extends { destroy(): void }>(value: T): T => {
        owned.push(value);
        return value;
    };
    device.addEventListener("uncapturederror", (event) => {
        throw new Error(event.error.message);
    });
    try {
        console.info(
            `[instance-payload] adapter=${state.gpu.adapter.identity} class=${state.gpu.adapter.class}`,
        );
        expect(state.gpu.adapter.class).toBe("real");
        const queries = own(device.createQuerySet({ type: "timestamp", count: 2 }));
        const resolved = own(
            device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            }),
        );
        const readback = own(
            device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            }),
        );
        const target = own(
            device.createTexture({
                size: [1, 1],
                format: "rgba8unorm",
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            }),
        );
        const view = target.createView();
        const timestamps = {
            querySet: queries,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
        };
        async function sample(
            label: string,
            encode: (encoder: GPUCommandEncoder) => void,
        ): Promise<number> {
            device.pushErrorScope("validation");
            const encoder = device.createCommandEncoder({ label });
            encode(encoder);
            encoder.resolveQuerySet(queries, 0, 2, resolved, 0);
            encoder.copyBufferToBuffer(resolved, 0, readback, 0, 16);
            device.queue.submit([encoder.finish()]);
            // Pop immediately: validation failure must not be deferred until after mapping.
            const error = await bounded(`${label} validation`, device.popErrorScope());
            if (error) throw new Error(`${label}: ${error.message}`);
            await bounded(`${label} timestamps`, readback.mapAsync(GPUMapMode.READ));
            const times = new BigUint64Array(readback.getMappedRange());
            const ns = Number(times[1]! - times[0]!);
            readback.unmap();
            return ns;
        }
        for (const count of [1000, 10000, 100000]) {
            const buffers: GPUBuffer[] = [];
            function buffer(label: string, data: Uint32Array | Float32Array): GPUBuffer {
                const result = own(
                    device.createBuffer({
                        label,
                        size: data.byteLength,
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                    }),
                );
                device.queue.writeBuffer(result, 0, data);
                buffers.push(result);
                return result;
            }
            const ids = new Uint32Array(count);
            const payload = new Uint32Array(count * 4);
            const transformMap = new Uint32Array(count);
            const partMap = new Uint32Array(count);
            for (let i = 0; i < count; i++) {
                ids[i] = i;
                transformMap[i] = ((i * 7) % count) + 1;
                partMap[i] = ((i * 11) % count) + 1;
                payload.set([i, transformMap[i]! - 1, partMap[i]!, 3], i * 4);
            }
            buffer("old-eids", ids);
            buffer("new-payload", payload);
            buffer("transform-map", transformMap);
            buffer("part-map", partMap);
            const poses = new Float32Array(count * 12);
            const parts = new Float32Array(count * 12);
            poses.fill(0.5);
            parts.fill(0.25);
            buffer("poses", poses);
            buffer("parts", parts);
            buffer("regather-output", new Uint32Array(count * 4));
            const layout = device.createBindGroupLayout({
                entries: buffers.map((_, binding) => ({
                    binding,
                    visibility:
                        binding === 6
                            ? GPUShaderStage.COMPUTE
                            : GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
                    buffer: { type: binding === 6 ? "storage" : "read-only-storage" },
                })),
            });
            const group = device.createBindGroup({
                layout,
                entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })),
            });
            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
            const common = `
struct Record { a: vec4f, b: vec4f, c: vec4f };
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> payload: array<vec4u>;
@group(0) @binding(2) var<storage, read> transformMap: array<u32>;
@group(0) @binding(3) var<storage, read> partMap: array<u32>;
@group(0) @binding(4) var<storage, read> poses: array<Record>;
@group(0) @binding(5) var<storage, read> parts: array<Record>;
@group(0) @binding(6) var<storage, read_write> output: array<u32>;
`;
            const renderPipelines = [false, true].map((rows) =>
                device.createRenderPipeline({
                    layout: pipelineLayout,
                    vertex: {
                        module: device.createShaderModule({
                            code:
                                common +
                                `
@vertex fn main(@builtin(instance_index) i: u32, @builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
${rows ? "let p = payload[i]; let eid = p.x; let tr = p.y; let pr = p.z - 1u;" : "let eid = ids[i]; let tr = transformMap[eid] - 1u; let pr = partMap[eid] - 1u;"}
let xf = poses[tr]; let part = parts[pr];
let x = xf.a + xf.b + xf.c + part.a + part.b + part.c;
// Outside the viewport: raster work cannot mask vertex timing. Every field affects the position.
return vec4f(x.xyz + vec3f(f32(v % 3u), f32(eid % 3u), 10.0), x.w);
}`,
                        }),
                        entryPoint: "main",
                    },
                    primitive: { topology: "triangle-list" },
                    fragment: {
                        module: device.createShaderModule({
                            code: "@fragment fn main() -> @location(0) vec4f { return vec4f(1); }",
                        }),
                        entryPoint: "main",
                        targets: [{ format: "rgba8unorm" }],
                    },
                }),
            );
            const computePipelines = [
                "compact-old",
                "compact-new",
                "regather-old",
                "regather-new",
            ].map((kind) => {
                const code =
                    kind === "compact-old"
                        ? "output[i] = ids[i];"
                        : kind === "compact-new"
                          ? "let eid = ids[i]; let base = i * 4u; output[base] = eid; output[base+1u] = transformMap[eid]-1u; output[base+2u] = partMap[eid]; output[base+3u] = 0u;"
                          : kind === "regather-old"
                            ? "output[i] = ids[i] | (3u << 20u);"
                            : "let p = payload[i]; let base = i * 4u; output[base] = p.x; output[base+1u] = p.y; output[base+2u] = p.z; output[base+3u] = 3u;";
                return device.createComputePipeline({
                    layout: pipelineLayout,
                    compute: {
                        module: device.createShaderModule({
                            code:
                                common +
                                `@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) { let i = gid.x; if (i >= ${count}u) { return; } ${kind.startsWith("compact") ? "let eidCull = ids[i]; let xfCull = poses[transformMap[eidCull]-1u]; let partCull = parts[partMap[eidCull]-1u]; if (xfCull.a.x + partCull.a.x < 0.0) { return; }" : ""} ${code} }`,
                        }),
                        entryPoint: "main",
                    },
                });
            });
            const repetitions = 128;
            async function measure(
                label: string,
                encode: (e: GPUCommandEncoder) => void,
            ): Promise<number> {
                const values: number[] = [];
                for (let round = 0; round < 9; round++) {
                    const ns = await sample(label, encode);
                    if (round > 1) values.push(ns / repetitions / 1000);
                }
                return median(values);
            }
            const computeTimes: number[] = [];
            for (let index = 0; index < computePipelines.length; index++) {
                computeTimes.push(
                    await measure(`count=${count} compute=${index}`, (encoder) => {
                        const pass = encoder.beginComputePass({ timestampWrites: timestamps });
                        pass.setPipeline(computePipelines[index]!);
                        pass.setBindGroup(0, group);
                        for (let r = 0; r < repetitions; r++)
                            pass.dispatchWorkgroups(Math.ceil(count / 64));
                        pass.end();
                    }),
                );
            }
            console.info(
                `[instance-payload] count=${count} compact old/new us=${computeTimes[0]}/${computeTimes[1]} regather old/new us=${computeTimes[2]}/${computeTimes[3]}`,
            );
            for (const vertices of [6, 36, 240]) {
                const times: number[] = [];
                for (let index = 0; index < renderPipelines.length; index++) {
                    times.push(
                        await measure(
                            `count=${count} vertices=${vertices} path=${index}`,
                            (encoder) => {
                                const pass = encoder.beginRenderPass({
                                    colorAttachments: [
                                        { view, loadOp: "clear", storeOp: "discard" },
                                    ],
                                    timestampWrites: timestamps,
                                });
                                pass.setPipeline(renderPipelines[index]!);
                                pass.setBindGroup(0, group);
                                for (let r = 0; r < repetitions; r++) pass.draw(vertices, count);
                                pass.end();
                            },
                        ),
                    );
                }
                console.info(
                    `[instance-payload] count=${count} vertices=${vertices} vertex old/new us=${times[0]}/${times[1]} compact+regather+vertex old/new us=${times[0]! + computeTimes[0]! + computeTimes[2]!}/${times[1]! + computeTimes[1]! + computeTimes[3]!}`,
                );
            }
        }
    } finally {
        for (const resource of owned.reverse()) resource.destroy();
        app.dispose();
    }
}, 8000);
