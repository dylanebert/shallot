import {
    Compute,
    drainLog,
    type Plugin,
    probeBuffer,
    probeTexture,
    run,
    type ShaderArtifact,
    validateGpu,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { type Check, register, type Scenario } from "../gym";

// The scenario catches the deliberate scoped failure so its own artifact checks can run; any uncaptured
// leak still enters verify's page-error channel and fails the run outside these checks.
const LABEL = "gym-invalid-shader";
const INVALID_WGSL = /* wgsl */ `
@compute @workgroup_size(1)
fn main() {
    let value: u32 = missing_identifier;
}
`;

let checks: Check[] = [];

const ProbeParams = d
    .struct({ mode: d.u32, mutate: d.u32, cutoff: d.f32, pad: d.f32 })
    .$name("GymProbeParams");

const cutoffLayout = tgpu.bindGroupLayout({
    sample: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
    params: { uniform: ProbeParams, visibility: ["fragment"] },
});

function fullscreenVertex(depth: number) {
    return tgpu.vertexFn({
        in: { vi: d.builtin.vertexIndex },
        out: { pos: d.builtin.position },
    })((input) => {
        "use gpu";
        const uv = d.vec2f(d.f32((input.vi << 1) & 2), d.f32(input.vi & 2));
        return { pos: d.vec4f(uv.x * 2 - 1, 1 - uv.y * 2, d.f32(depth), 1) };
    });
}

const cutoffVs = fullscreenVertex(0.25).$name("gymProbeCutoffVs");
const receiverVs = fullscreenVertex(0.5).$name("gymProbeReceiverVs");

const cutoffFs = tgpu
    .fragmentFn({ in: { pos: d.builtin.position }, out: d.vec4f })((input) => {
        "use gpu";
        const sampled = std.textureLoad(cutoffLayout.$.sample, d.vec2i(0), 0).x;
        const spatialReject = input.pos.x < 1;
        const sampledReject = sampled < cutoffLayout.$.params.cutoff;
        const useSampled = cutoffLayout.$.params.mode === 1 && cutoffLayout.$.params.mutate === 0;
        const reject = std.select(spatialReject, sampledReject, useSampled);
        if (reject) std.discard();
        return d.vec4f(0, 1, 0, 1);
    })
    .$name("gymProbeCutoffFs");

const receiverFs = tgpu
    .fragmentFn({ out: d.vec4f })(() => {
        "use gpu";
        return d.vec4f(1, 0, 0, 1);
    })
    .$name("gymProbeReceiverFs");

const fragmentLogFs = tgpu
    .fragmentFn({ out: d.vec4f })(() => {
        "use gpu";
        console.log("fragment-probe", d.u32(23));
        return d.vec4f(0, 0, 1, 1);
    })
    .$name("gymProbeFragmentLogFs");

function pass(
    encoder: GPUCommandEncoder,
    label: string,
    color: GPUTexture,
    depth: GPUTexture,
    load: GPULoadOp,
    draw: (render: GPURenderPassEncoder) => void,
): void {
    encoder.pushDebugGroup(label);
    const render = encoder.beginRenderPass({
        label,
        colorAttachments: [
            {
                view: color.createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: load,
                storeOp: "store",
            },
        ],
        depthStencilAttachment: {
            view: depth.createView(),
            depthClearValue: 1,
            depthLoadOp: load,
            depthStoreOp: "store",
        },
    });
    draw(render);
    render.end();
    encoder.popDebugGroup();
}

interface CutoffResult {
    mode: "spatial" | "sampled";
    sample: number;
    cutoff: number;
    expectedCutoff: boolean[];
    depth: number[];
    depthAfterReceiver: number[];
    rgba: number[][];
}

async function cutoffProbe(mutate: boolean): Promise<CutoffResult[]> {
    const device = Compute.device;
    const sample = device.createTexture({
        label: "gpu-probe-sample",
        size: [1, 1],
        format: "rgba8unorm",
        usage:
            GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });
    device.queue.writeTexture({ texture: sample }, Uint8Array.of(64, 0, 0, 255), {}, [1, 1]);
    const sampleSnapshot = await probeTexture(device, sample, { label: "gpu-probe-input-sample" });
    const params = Compute.root
        .createBuffer(ProbeParams)
        .$usage("uniform")
        .$name("gpu-probe-cutoff-params");
    const group = Compute.root.createBindGroup(cutoffLayout, {
        sample: sample.createView(),
        params,
    });
    const cutoffPipeline = Compute.root
        .createRenderPipeline({
            vertex: cutoffVs,
            fragment: cutoffFs,
            targets: { format: "rgba8unorm" },
            depthStencil: {
                format: "depth32float",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        })
        .$name("gpu-probe-cutoff");
    const receiverPipeline = Compute.root
        .createRenderPipeline({
            vertex: receiverVs,
            fragment: receiverFs,
            targets: { format: "rgba8unorm" },
            depthStencil: {
                format: "depth32float",
                depthWriteEnabled: false,
                depthCompare: "less",
            },
        })
        .$name("gpu-probe-receiver");

    const inputSample = new Uint8Array(sampleSnapshot.bytes)[0] / 255;
    const cutoff = 0.5;
    const results: CutoffResult[] = [];
    for (const [index, mode] of (["spatial", "sampled"] as const).entries()) {
        params.write({ mode: index, mutate: mutate ? 1 : 0, cutoff, pad: 0 });
        const color = device.createTexture({
            label: `gpu-probe-${mode}-color`,
            size: [2, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        const depth = device.createTexture({
            label: `gpu-probe-${mode}-depth`,
            size: [2, 1],
            format: "depth32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        const depthSnapshot = await probeTexture(device, depth, {
            aspect: "depth-only",
            label: `gpu-probe-cutoff-${mode}`,
            encode(encoder) {
                pass(encoder, `gpu-probe:cutoff:${mode}`, color, depth, "clear", (render) => {
                    cutoffPipeline.with(group).with(render).draw(3);
                });
            },
        });
        const colorSnapshot = await probeTexture(device, color, {
            label: `gpu-probe-receiver-${mode}`,
            encode(encoder) {
                pass(encoder, `gpu-probe:receiver:${mode}`, color, depth, "load", (render) => {
                    receiverPipeline.with(render).draw(3);
                });
            },
        });
        const depthAfterReceiver = await probeTexture(device, depth, {
            aspect: "depth-only",
            label: `gpu-probe-depth-after-receiver-${mode}`,
        });
        const depthValues = [...new Float32Array(depthSnapshot.bytes)];
        const depthAfterReceiverValues = [...new Float32Array(depthAfterReceiver.bytes)];
        const colorValues = new Uint8Array(colorSnapshot.bytes);
        const rgba = [[...colorValues.slice(0, 4)], [...colorValues.slice(4, 8)]];
        results.push({
            mode,
            sample: inputSample,
            cutoff,
            expectedCutoff:
                mode === "sampled" ? [inputSample < cutoff, inputSample < cutoff] : [true, false],
            depth: depthValues,
            depthAfterReceiver: depthAfterReceiverValues,
            rgba,
        });
        color.destroy();
        depth.destroy();
    }
    params.destroy();
    sample.destroy();
    return results;
}

function cutoffCheck(result: CutoffResult): Check {
    const expectedDepth = result.expectedCutoff.map((cut) => (cut ? 1 : 0.25));
    const expectedRgba = result.expectedCutoff.map((cut) =>
        cut ? [255, 0, 0, 255] : [0, 255, 0, 255],
    );
    const depthOk = result.depth.every((value, index) => value === expectedDepth[index]);
    const preservedDepth = result.depthAfterReceiver.every(
        (value, index) => value === result.depth[index],
    );
    const colorOk = result.rgba.every((value, index) =>
        value.every((channel, channelIndex) => channel === expectedRgba[index][channelIndex]),
    );
    return {
        name: `${result.mode} cutoff resource chain`,
        pass: depthOk && preservedDepth && colorOk,
        detail: `sample ${result.sample.toFixed(6)}, cutoff ${result.cutoff}; ${result.expectedCutoff
            .map(
                (cut, index) =>
                    `texel ${index}: expected discard ${cut}, depth ${result.depth[index]}→${result.depthAfterReceiver[index]}, rgba [${result.rgba[index].join(", ")}]`,
            )
            .join(" | ")}`,
    };
}

async function boundaryChecks(): Promise<Check[]> {
    const device = Compute.device;
    const rawValues = new Uint32Array([0x01234567, 0x89abcdef, 17, 0xffffffff]);
    const raw = device.createBuffer({
        label: "gpu-probe-raw",
        size: rawValues.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(raw, 0, rawValues);
    const [rawFirst, rawSecond] = await Promise.all([
        probeBuffer(device, raw, { label: "gpu-probe-raw-first" }),
        probeBuffer(device, raw, { label: "gpu-probe-raw-second" }),
    ]);
    const rawA = new Uint32Array(rawFirst.bytes);
    const rawB = new Uint32Array(rawSecond.bytes);

    const TypedValues = d.arrayOf(d.u32, 4);
    const typed = Compute.root.createBuffer(TypedValues).$usage("storage").$name("gpu-probe-typed");
    typed.write([3, 5, 8, 13]);
    const typedValues = await typed.read();

    const color = device.createTexture({
        label: "gpu-probe-known-color",
        size: [2, 2],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const colorSnapshot = () =>
        probeTexture(device, color, {
            label: "gpu-probe-known-color",
            encode(encoder) {
                const render = encoder.beginRenderPass({
                    label: "gpu-probe-known-color",
                    colorAttachments: [
                        {
                            view: color.createView(),
                            clearValue: { r: 64 / 255, g: 128 / 255, b: 192 / 255, a: 1 },
                            loadOp: "clear",
                            storeOp: "store",
                        },
                    ],
                });
                render.end();
            },
        });
    const colorFirst = await colorSnapshot();
    const colorSecond = await colorSnapshot();
    const colorA = new Uint8Array(colorFirst.bytes);
    const colorB = new Uint8Array(colorSecond.bytes);

    const depth = device.createTexture({
        label: "gpu-probe-known-depth",
        size: [2, 2],
        format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depthSnapshot = () =>
        probeTexture(device, depth, {
            aspect: "depth-only",
            label: "gpu-probe-known-depth",
            encode(encoder) {
                const render = encoder.beginRenderPass({
                    label: "gpu-probe-known-depth",
                    colorAttachments: [],
                    depthStencilAttachment: {
                        view: depth.createView(),
                        depthClearValue: 0.375,
                        depthLoadOp: "clear",
                        depthStoreOp: "store",
                    },
                });
                render.end();
            },
        });
    const depthFirst = await depthSnapshot();
    const depthSecond = await depthSnapshot();
    const depthA = new Float32Array(depthFirst.bytes);
    const depthB = new Float32Array(depthSecond.bytes);

    raw.destroy();
    typed.destroy();
    color.destroy();
    depth.destroy();
    return [
        {
            name: "raw buffer one-shot",
            pass:
                rawA.every((value, index) => value === rawValues[index]) &&
                rawB.every((value, index) => value === rawValues[index]) &&
                rawFirst.bytes !== rawSecond.bytes,
            detail: `[${[...rawA].join(", ")}] repeated ${rawFirst.bytes !== rawSecond.bytes}`,
        },
        {
            name: "typed buffer read",
            pass: typedValues.every((value, index) => value === [3, 5, 8, 13][index]),
            detail: `[${typedValues.join(", ")}] via TypeGPU .read()`,
        },
        {
            name: "color texture one-shot",
            pass:
                colorA.every((value, index) => value === [64, 128, 192, 255][index % 4]) &&
                colorB.every((value, index) => value === colorA[index]) &&
                colorFirst.bytes !== colorSecond.bytes,
            detail: `rgba [${[...colorA.slice(0, 4)].join(", ")}], repeated ${colorFirst.bytes !== colorSecond.bytes}`,
        },
        {
            name: "depth texture one-shot",
            pass:
                depthA.every((value) => value === 0.375) &&
                depthB.every((value, index) => value === depthA[index]) &&
                depthFirst.bytes !== depthSecond.bytes,
            detail: `depth [${[...depthA].join(", ")}], repeated ${depthFirst.bytes !== depthSecond.bytes}`,
        },
    ];
}

async function logChecks(): Promise<Check[]> {
    const compute = Compute.root
        .createGuardedComputePipeline(() => {
            "use gpu";
            console.log("compute-probe", d.u32(17));
        })
        .$name("gpu-probe-compute-log");
    const computeLines = await drainLog(() => compute.dispatchThreads());

    const target = Compute.device.createTexture({
        label: "gpu-probe-fragment-replay",
        size: [1, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const fragment = Compute.root
        .createRenderPipeline({
            vertex: fullscreenVertex(0).$name("gymProbeFragmentLogVs"),
            fragment: fragmentLogFs,
            targets: { format: "rgba8unorm" },
        })
        .$name("gpu-probe-fragment-log");
    const external = Compute.device.createCommandEncoder({ label: "gpu-probe-fragment-external" });
    const externalPass = external.beginRenderPass({
        label: "gpu-probe-fragment-external",
        colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store" }],
    });
    fragment.with(externalPass).draw(3);
    externalPass.end();
    Compute.device.queue.submit([external.finish()]);
    const fragmentLines = await drainLog(() => {
        fragment
            .withColorAttachment({
                view: target.createView(),
                loadOp: "clear",
                storeOp: "store",
            })
            .draw(3);
    });
    target.destroy();
    return [
        {
            name: "TypeGPU compute log",
            pass:
                computeLines.length === 1 &&
                computeLines[0].includes("compute-probe") &&
                computeLines[0].includes("17"),
            detail: computeLines.join(" | ") || "no compute log",
        },
        {
            name: "TypeGPU fragment replay log",
            pass:
                fragmentLines.length === 2 &&
                fragmentLines.every(
                    (line) => line.includes("fragment-probe") && line.includes("23"),
                ),
            detail: fragmentLines.join(" | ") || "no fragment log",
        },
    ];
}

function diagnosticPlugin(mutate: boolean): Plugin {
    return {
        name: "GymGpuDiagnostic",
        async warm() {
            let diagnostic: unknown;
            let directMessages: readonly GPUCompilationMessage[] = [];
            try {
                await validateGpu(Compute.device, LABEL, async () => {
                    const module = Compute.device.createShaderModule({
                        label: LABEL,
                        code: INVALID_WGSL,
                    });
                    directMessages = (await module.getCompilationInfo()).messages;
                    await Compute.device.createComputePipelineAsync({
                        label: LABEL,
                        layout: "auto",
                        compute: { module, entryPoint: "main" },
                    });
                });
            } catch (error) {
                diagnostic = error;
            }

            const artifact = (
                globalThis as { __gpuDiagnostics?: { artifacts: ShaderArtifact[] } }
            ).__gpuDiagnostics?.artifacts.find((entry) => entry.label === LABEL);
            const structured = diagnostic as {
                name?: string;
                label?: string;
                errorClass?: string;
                message?: string;
            };
            const cutoff = await cutoffProbe(mutate);
            checks = [
                {
                    name: "scoped invalid shader",
                    pass:
                        structured.name === "GpuDiagnosticError" &&
                        structured.label === LABEL &&
                        typeof structured.errorClass === "string" &&
                        structured.errorClass.length > 0 &&
                        structured.message?.includes(LABEL) === true,
                    detail: `${structured.errorClass ?? "none"}: ${structured.message ?? "no diagnostic"}`,
                },
                {
                    name: "compiler rejected invalid WGSL",
                    pass: directMessages.some((message) => message.type === "error"),
                    detail: directMessages.map((message) => message.message).join(" | "),
                },
                {
                    name: "exact shader artifact",
                    pass:
                        artifact?.source === INVALID_WGSL &&
                        artifact.stage === "compute" &&
                        /^[0-9a-f]{16}$/.test(artifact.hash) &&
                        artifact.messages.some((message) => message.type === "error"),
                    detail: artifact
                        ? `${artifact.label} ${artifact.hash}, ${artifact.messages.length} message(s)`
                        : "artifact missing",
                },
                ...(await boundaryChecks()),
                ...(await logChecks()),
                ...cutoff.map(cutoffCheck),
            ];
        },
    };
}

const scenario: Scenario = {
    name: "gpu-diagnostic",
    noRender: true,
    params: [{ key: "mutate", type: "bool", default: false, rebuild: true }],
    async build(_canvas, params) {
        checks = [];
        const { state, dispose } = await run({
            defaults: false,
            plugins: [ProfilePlugin, diagnosticPlugin(params.mutate === true)],
        });
        return { state, dispose };
    },
    async assert() {
        return checks;
    },
};

register(scenario);
