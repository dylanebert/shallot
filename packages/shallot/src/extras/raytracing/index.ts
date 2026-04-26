import { traits, capacity, type Plugin, type State } from "../../engine";
import { Compute, beginComputePass } from "../../standard/compute";
import type { ComputeNode, ExecutionContext } from "../../standard/compute";
import { Camera, Shadows } from "../../standard/render";
import { hasProperties, type SurfaceData } from "../../standard/render/core";
import { surfaceRegistry } from "../../standard/render";
import type { GBuf } from "../../standard/compute";
import { Render, RenderPlugin } from "../../standard/render";
import { BVH, initializeBVH } from "./bvh";
import type { BLASAtlas } from "./bvh";
import {
    compileRaygenShader,
    compileClosestHitShader,
    compileAnyHitShader,
    compileShadeShader,
    compileResolveShader,
    compileClearPixelStateShader,
    compileSwapCounterShader,
    compileSwapBounceCounterShader,
    compileApplyShadowShader,
} from "./shaders";
import {
    createWavefrontBuffers,
    COUNTER_BUF_SIZE,
    MAX_SHADOW_RAYS_PER_PIXEL,
    maxPixelsForDevice,
    type WavefrontBuffers,
} from "./buffers";

export const Raytracing = {
    width: [] as number[],
    height: [] as number[],
};

interface WavefrontPipelines {
    raygen: GPUComputePipeline;
    trace: GPUComputePipeline;
    shadowTrace: GPUComputePipeline;
    clear: GPUComputePipeline;
    shade: GPUComputePipeline;
    swapCounter: GPUComputePipeline;
    swapBounce: GPUComputePipeline;
    applyShadow: GPUComputePipeline;
    resolve: GPUComputePipeline;
}

interface WavefrontGPU {
    alive: boolean;
    pipelines: WavefrontPipelines | null;
    buffers: WavefrontBuffers | null;
    paramsBuffer: GPUBuffer | null;
    cachedWidth: number;
    cachedHeight: number;
    raygenSceneBG: GPUBindGroup | null;
    raygenRayBG: GPUBindGroup | null;
    traceSceneBG: GPUBindGroup | null;
    traceRayBG: GPUBindGroup | null;
    traceBvhBG: GPUBindGroup | null;
    clearBG: GPUBindGroup | null;
    shadeBG0: GPUBindGroup | null;
    shadeBG1: GPUBindGroup | null;
    swapCounterBG: GPUBindGroup | null;
    swapBounceCounterBG: GPUBindGroup | null;
    shadowTraceSceneBG: GPUBindGroup | null;
    shadowTraceBG1: GPUBindGroup | null;
    shadowTraceBvhBG: GPUBindGroup | null;
    applyShadowBG: GPUBindGroup | null;
    traceRayBGAlt: GPUBindGroup | null;
    shadeBG1Alt: GPUBindGroup | null;
    resolveBG: GPUBindGroup | null;
    cachedBlasAtlas: BLASAtlas | null;
    cachedColorView: GPUTextureView | null;
    cachedDepthView: GPUTextureView | null;
    cachedEidView: GPUTextureView | null;
    cachedCapacity: number;
    debugStagingBuffer: GPUBuffer | null;
    debugReadbackPending: boolean;
}

function compilePipeline(
    device: GPUDevice,
    code: string,
    label?: string,
): Promise<GPUComputePipeline> {
    const module = device.createShaderModule({ code });
    return device.createComputePipelineAsync({
        label,
        layout: "auto",
        compute: { module, entryPoint: "main" },
    });
}

function triggerWavefrontCompile(
    gpu: WavefrontGPU,
    device: GPUDevice,
    getSurfaces: () => SurfaceData[],
): void {
    const surfaces = getSurfaces();

    Promise.all([
        compilePipeline(device, compileRaygenShader(), "rt-raygen"),
        compilePipeline(device, compileClosestHitShader(), "rt-trace"),
        compilePipeline(device, compileAnyHitShader(), "rt-shadow-trace"),
        compilePipeline(device, compileShadeShader(surfaces), "rt-shade"),
        compilePipeline(device, compileResolveShader(), "rt-resolve"),
        compilePipeline(device, compileClearPixelStateShader(), "rt-clear-pixel-state"),
        compilePipeline(device, compileSwapCounterShader(), "rt-swap-counter"),
        compilePipeline(device, compileSwapBounceCounterShader(), "rt-swap-bounce-counter"),
        compilePipeline(device, compileApplyShadowShader(), "rt-apply-shadow"),
    ])
        .then(
            ([
                raygen,
                trace,
                shadowTrace,
                shade,
                resolve,
                clear,
                swapCounter,
                swapBounce,
                applyShadow,
            ]) => {
                if (!gpu.alive) return;
                gpu.pipelines = {
                    raygen,
                    trace,
                    shadowTrace,
                    clear,
                    shade,
                    swapCounter,
                    swapBounce,
                    applyShadow,
                    resolve,
                };
            },
        )
        .catch((e) => {
            if (!gpu.alive) return;
            console.error("wavefront compile failed:", e);
        });
}

function ensureBuffers(
    gpu: WavefrontGPU,
    device: GPUDevice,
    width: number,
    height: number,
): WavefrontBuffers {
    if (gpu.buffers && gpu.cachedWidth === width && gpu.cachedHeight === height) {
        return gpu.buffers;
    }
    gpu.buffers = createWavefrontBuffers(device, width * height);

    const paramsData = new Uint32Array([width, height]);
    if (!gpu.paramsBuffer) {
        gpu.paramsBuffer = device.createBuffer({
            label: "wf-params",
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }
    device.queue.writeBuffer(gpu.paramsBuffer, 0, paramsData);

    gpu.cachedWidth = width;
    gpu.cachedHeight = height;
    invalidateBindGroups(gpu);
    return gpu.buffers;
}

function invalidateBindGroups(gpu: WavefrontGPU): void {
    gpu.raygenSceneBG = null;
    gpu.raygenRayBG = null;
    gpu.traceSceneBG = null;
    gpu.traceRayBG = null;
    gpu.traceBvhBG = null;
    gpu.clearBG = null;
    gpu.shadeBG0 = null;
    gpu.shadeBG1 = null;
    gpu.swapCounterBG = null;
    gpu.swapBounceCounterBG = null;
    gpu.shadowTraceSceneBG = null;
    gpu.shadowTraceBG1 = null;
    gpu.shadowTraceBvhBG = null;
    gpu.applyShadowBG = null;
    gpu.traceRayBGAlt = null;
    gpu.shadeBG1Alt = null;
    gpu.resolveBG = null;
}

function executeWavefront(
    gpu: WavefrontGPU,
    ctx: ExecutionContext,
    render: {
        scene: GPUBuffer;
        sky: GPUBuffer;
        data: GBuf;
        matrices: GBuf;
        instanceDataBuffer: GBuf | null;
    },
    bvh: { tlas: { bvhNodes: GBuf }; instanceInverses: GBuf; blasAtlas: BLASAtlas },
    pointLightBuffer: GPUBuffer,
    getRaytracing: () => boolean,
    getShadows: () => boolean,
): void {
    if (!getRaytracing()) return;
    if (!gpu.pipelines) return;

    const {
        raygen,
        trace,
        shadowTrace,
        clear,
        shade,
        swapCounter,
        swapBounce,
        applyShadow,
        resolve,
    } = gpu.pipelines;

    const { device, encoder } = ctx;
    const colorView = ctx.getTextureView("color");
    const depthView = ctx.getTextureView("depth");
    const eidView = ctx.getTextureView("eid");
    if (!colorView || !depthView || !eidView) return;
    const colorTexture = ctx.getTexture("color");
    if (!colorTexture) return;

    const width = colorTexture.width;
    const height = colorTexture.height;
    const pixelCount = width * height;

    const bufs = ensureBuffers(gpu, device, width, height);
    const blas = bvh.blasAtlas;

    const viewsChanged =
        colorView !== gpu.cachedColorView ||
        depthView !== gpu.cachedDepthView ||
        eidView !== gpu.cachedEidView;
    const blasChanged = blas !== gpu.cachedBlasAtlas;
    const capChanged = capacity() !== gpu.cachedCapacity;

    if (capChanged) {
        invalidateBindGroups(gpu);
        gpu.cachedCapacity = capacity();
    }

    if (!gpu.raygenSceneBG) {
        gpu.raygenSceneBG = device.createBindGroup({
            layout: raygen.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: render.scene } }],
        });
    }
    if (!gpu.raygenRayBG) {
        gpu.raygenRayBG = device.createBindGroup({
            layout: raygen.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: bufs.rays0.buffer } },
                { binding: 1, resource: { buffer: bufs.counters.buffer } },
            ],
        });
    }

    if (!gpu.traceSceneBG) {
        gpu.traceSceneBG = device.createBindGroup({
            layout: trace.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: render.scene } },
                { binding: 1, resource: { buffer: render.data.buffer } },
            ],
        });
    }
    if (!gpu.traceRayBG) {
        gpu.traceRayBG = device.createBindGroup({
            layout: trace.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: bufs.rays0.buffer } },
                { binding: 1, resource: { buffer: bufs.hits.buffer } },
                { binding: 2, resource: { buffer: bufs.counters.buffer } },
            ],
        });
    }
    if (blasChanged || !gpu.traceBvhBG) {
        gpu.traceBvhBG = device.createBindGroup({
            layout: trace.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: { buffer: bvh.tlas.bvhNodes.buffer } },
                { binding: 1, resource: { buffer: blas.nodesBuffer } },
                { binding: 2, resource: { buffer: blas.triIdsBuffer } },
                { binding: 3, resource: { buffer: blas.trianglesBuffer } },
                { binding: 4, resource: { buffer: blas.entityBlasMetaBuffer } },
                { binding: 5, resource: { buffer: bvh.instanceInverses.buffer } },
            ],
        });
        gpu.cachedBlasAtlas = blas;
    }

    const wg256 = Math.ceil(pixelCount / 256);

    // zero counters before raygen
    device.queue.writeBuffer(bufs.counters.buffer, 0, bufs.zeroCounters as BufferSource);

    // raygen
    const raygenPass = beginComputePass(encoder, ctx.timestampWrites?.("wf-raygen"));
    raygenPass.setPipeline(raygen);
    raygenPass.setBindGroup(0, gpu.raygenSceneBG!);
    raygenPass.setBindGroup(1, gpu.raygenRayBG!);
    raygenPass.dispatchWorkgroups(wg256);
    raygenPass.end();

    // trace
    const tracePass = beginComputePass(encoder, ctx.timestampWrites?.("wf-trace"));
    tracePass.setPipeline(trace);
    tracePass.setBindGroup(0, gpu.traceSceneBG!);
    tracePass.setBindGroup(1, gpu.traceRayBG!);
    tracePass.setBindGroup(2, gpu.traceBvhBG!);
    tracePass.dispatchWorkgroups(wg256);
    tracePass.end();

    // clear pixel state via GPU, reset output+shadow counters
    if (!gpu.clearBG) {
        gpu.clearBG = device.createBindGroup({
            layout: clear.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufs.pixelState.buffer } },
                { binding: 1, resource: { buffer: gpu.paramsBuffer! } },
            ],
        });
    }
    const clearPass = beginComputePass(encoder, ctx.timestampWrites?.("wf-clear"));
    clearPass.setPipeline(clear);
    clearPass.setBindGroup(0, gpu.clearBG!);
    clearPass.dispatchWorkgroups(wg256);
    clearPass.end();

    device.queue.writeBuffer(bufs.counters.buffer, 4, bufs.zeroCounters as BufferSource, 0, 2);

    if (!gpu.shadeBG0) {
        const entries = [
            { binding: 0, resource: { buffer: render.scene } },
            { binding: 1, resource: { buffer: render.data.buffer } },
            { binding: 2, resource: { buffer: render.sky } },
            { binding: 3, resource: { buffer: render.matrices.buffer } },
            { binding: 4, resource: { buffer: pointLightBuffer } },
        ];
        if (hasProperties() && render.instanceDataBuffer) {
            entries.push({
                binding: 6,
                resource: { buffer: render.instanceDataBuffer.buffer },
            });
        }
        gpu.shadeBG0 = device.createBindGroup({
            layout: shade.getBindGroupLayout(0),
            entries,
        });
    }
    if (!gpu.shadeBG1) {
        gpu.shadeBG1 = device.createBindGroup({
            layout: shade.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: bufs.rays0.buffer } },
                { binding: 1, resource: { buffer: bufs.hits.buffer } },
                { binding: 2, resource: { buffer: bufs.pixelState.buffer } },
                { binding: 3, resource: { buffer: bufs.rays1.buffer } },
                { binding: 4, resource: { buffer: bufs.shadowRays.buffer } },
                { binding: 5, resource: { buffer: bufs.counters.buffer } },
            ],
        });
    }
    if (!gpu.shadeBG1Alt) {
        gpu.shadeBG1Alt = device.createBindGroup({
            layout: shade.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: bufs.rays1.buffer } },
                { binding: 1, resource: { buffer: bufs.hits.buffer } },
                { binding: 2, resource: { buffer: bufs.pixelState.buffer } },
                { binding: 3, resource: { buffer: bufs.rays0.buffer } },
                { binding: 4, resource: { buffer: bufs.shadowRays.buffer } },
                { binding: 5, resource: { buffer: bufs.counters.buffer } },
            ],
        });
    }
    if (!gpu.traceRayBGAlt) {
        gpu.traceRayBGAlt = device.createBindGroup({
            layout: trace.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: bufs.rays1.buffer } },
                { binding: 1, resource: { buffer: bufs.hits.buffer } },
                { binding: 2, resource: { buffer: bufs.counters.buffer } },
            ],
        });
    }
    if (!gpu.swapCounterBG) {
        gpu.swapCounterBG = device.createBindGroup({
            layout: swapCounter.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufs.counters.buffer } },
                { binding: 1, resource: { buffer: bufs.indirect.buffer } },
                { binding: 2, resource: { buffer: gpu.paramsBuffer! } },
            ],
        });
    }
    if (!gpu.swapBounceCounterBG) {
        gpu.swapBounceCounterBG = device.createBindGroup({
            layout: swapBounce.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufs.counters.buffer } },
                { binding: 1, resource: { buffer: gpu.paramsBuffer! } },
                { binding: 2, resource: { buffer: bufs.indirect.buffer } },
            ],
        });
    }
    if (!gpu.shadowTraceSceneBG) {
        gpu.shadowTraceSceneBG = device.createBindGroup({
            layout: shadowTrace.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: render.scene } },
                { binding: 1, resource: { buffer: render.data.buffer } },
            ],
        });
    }
    if (!gpu.shadowTraceBG1) {
        gpu.shadowTraceBG1 = device.createBindGroup({
            layout: shadowTrace.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: bufs.shadowRays.buffer } },
                { binding: 1, resource: { buffer: bufs.shadowHits.buffer } },
                { binding: 2, resource: { buffer: bufs.counters.buffer } },
            ],
        });
    }
    if (blasChanged || !gpu.shadowTraceBvhBG) {
        gpu.shadowTraceBvhBG = device.createBindGroup({
            layout: shadowTrace.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: { buffer: bvh.tlas.bvhNodes.buffer } },
                { binding: 1, resource: { buffer: blas.nodesBuffer } },
                { binding: 2, resource: { buffer: blas.triIdsBuffer } },
                { binding: 3, resource: { buffer: blas.trianglesBuffer } },
                { binding: 4, resource: { buffer: blas.entityBlasMetaBuffer } },
                { binding: 5, resource: { buffer: bvh.instanceInverses.buffer } },
            ],
        });
    }
    if (!gpu.applyShadowBG) {
        gpu.applyShadowBG = device.createBindGroup({
            layout: applyShadow.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufs.shadowHits.buffer } },
                { binding: 1, resource: { buffer: bufs.shadowRays.buffer } },
                { binding: 2, resource: { buffer: bufs.pixelState.buffer } },
                { binding: 3, resource: { buffer: bufs.counters.buffer } },
                { binding: 4, resource: { buffer: gpu.paramsBuffer! } },
            ],
        });
    }

    const MaxBounces = 4;
    const hasShadows = getShadows();

    for (let bounce = 0; bounce < MaxBounces; bounce++) {
        const useB = bounce % 2 === 1;

        if (bounce > 0) {
            const swapBouncePass = beginComputePass(
                encoder,
                ctx.timestampWrites?.(`wf-swap-bounce-${bounce}`),
            );
            swapBouncePass.setPipeline(swapBounce);
            swapBouncePass.setBindGroup(0, gpu.swapBounceCounterBG!);
            swapBouncePass.dispatchWorkgroups(1);
            swapBouncePass.end();

            const bounceTracePass = beginComputePass(
                encoder,
                ctx.timestampWrites?.(`wf-trace-${bounce}`),
            );
            bounceTracePass.setPipeline(trace);
            bounceTracePass.setBindGroup(0, gpu.traceSceneBG!);
            bounceTracePass.setBindGroup(1, useB ? gpu.traceRayBGAlt! : gpu.traceRayBG!);
            bounceTracePass.setBindGroup(2, gpu.traceBvhBG!);
            bounceTracePass.dispatchWorkgroupsIndirect(bufs.indirect.buffer, 0);
            bounceTracePass.end();
        }

        // shade (hits + inline miss)
        const shadePass = beginComputePass(encoder, ctx.timestampWrites?.(`wf-shade-${bounce}`));
        shadePass.setPipeline(shade);
        shadePass.setBindGroup(0, gpu.shadeBG0!);
        shadePass.setBindGroup(1, useB ? gpu.shadeBG1Alt! : gpu.shadeBG1!);
        if (bounce === 0) {
            shadePass.dispatchWorkgroups(wg256);
        } else {
            shadePass.dispatchWorkgroupsIndirect(bufs.indirect.buffer, 0);
        }
        shadePass.end();

        if (hasShadows) {
            // swap shadow count → input, preserve bounce output count
            const swapPass = beginComputePass(
                encoder,
                ctx.timestampWrites?.(`wf-swap-shadow-${bounce}`),
            );
            swapPass.setPipeline(swapCounter);
            swapPass.setBindGroup(0, gpu.swapCounterBG!);
            swapPass.dispatchWorkgroups(1);
            swapPass.end();

            // shadow trace
            const shadowTracePass = beginComputePass(
                encoder,
                ctx.timestampWrites?.(`wf-shadow-trace-${bounce}`),
            );
            shadowTracePass.setPipeline(shadowTrace);
            shadowTracePass.setBindGroup(0, gpu.shadowTraceSceneBG!);
            shadowTracePass.setBindGroup(1, gpu.shadowTraceBG1!);
            shadowTracePass.setBindGroup(2, gpu.shadowTraceBvhBG!);
            shadowTracePass.dispatchWorkgroupsIndirect(bufs.indirect.buffer, 0);
            shadowTracePass.end();

            // apply shadow
            const applyShadowPass = beginComputePass(
                encoder,
                ctx.timestampWrites?.(`wf-apply-shadow-${bounce}`),
            );
            applyShadowPass.setPipeline(applyShadow);
            applyShadowPass.setBindGroup(0, gpu.applyShadowBG!);
            applyShadowPass.dispatchWorkgroupsIndirect(bufs.indirect.buffer, 0);
            applyShadowPass.end();
        }
    }

    // resolve
    if (viewsChanged || !gpu.resolveBG) {
        gpu.resolveBG = device.createBindGroup({
            layout: resolve.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufs.pixelState.buffer } },
                { binding: 1, resource: colorView },
                { binding: 2, resource: depthView },
                { binding: 3, resource: eidView },
                { binding: 4, resource: { buffer: gpu.paramsBuffer! } },
                { binding: 5, resource: { buffer: render.scene } },
                { binding: 6, resource: { buffer: render.sky } },
            ],
        });
        gpu.cachedColorView = colorView;
        gpu.cachedDepthView = depthView;
        gpu.cachedEidView = eidView;
    }

    const resolvePass = beginComputePass(encoder, ctx.timestampWrites?.("wf-resolve"));
    resolvePass.setPipeline(resolve);
    resolvePass.setBindGroup(0, gpu.resolveBG!);
    resolvePass.dispatchWorkgroups(wg256);
    resolvePass.end();

    if (!gpu.debugReadbackPending) {
        if (!gpu.debugStagingBuffer) {
            gpu.debugStagingBuffer = device.createBuffer({
                label: "wf-debug-staging",
                size: COUNTER_BUF_SIZE,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
        }
        encoder.copyBufferToBuffer(
            bufs.counters.buffer,
            0,
            gpu.debugStagingBuffer,
            0,
            COUNTER_BUF_SIZE,
        );
        gpu.debugReadbackPending = true;
        const staging = gpu.debugStagingBuffer;
        ctx.afterSubmit(() => {
            staging.mapAsync(GPUMapMode.READ, 0, COUNTER_BUF_SIZE).then(
                () => {
                    const data = new Uint32Array(staging.getMappedRange(0, COUNTER_BUF_SIZE));
                    const rayOverflow = data[4];
                    const shadowOverflow = data[5];
                    const maxOutput = data[6];
                    const maxShadow = data[7];
                    const shadePixelOOB = data[8];
                    const shadowPixelOOB = data[9];
                    const bounceClipped = data[10];
                    const shadowUnclamped = data[11];
                    const hasIssue =
                        rayOverflow > 0 ||
                        shadowOverflow > 0 ||
                        shadePixelOOB > 0 ||
                        shadowPixelOOB > 0 ||
                        bounceClipped > 0 ||
                        shadowUnclamped > 0;
                    if (hasIssue) {
                        console.warn(
                            `[RT] overflow: rays=${rayOverflow} shadows=${shadowOverflow} maxOut=${maxOutput}/${pixelCount * 4} maxShadow=${maxShadow}/${pixelCount * MAX_SHADOW_RAYS_PER_PIXEL}` +
                                ` shadeOOB=${shadePixelOOB} shadowOOB=${shadowPixelOOB}` +
                                ` bounceClipped=${bounceClipped} shadowUnclamped=${shadowUnclamped}`,
                        );
                    }
                    staging.unmap();
                    gpu.debugReadbackPending = false;
                },
                () => {
                    gpu.debugReadbackPending = false;
                },
            );
        });
    }
}

function createWavefrontNode(
    render: {
        scene: GPUBuffer;
        sky: GPUBuffer;
        data: GBuf;
        matrices: GBuf;
        instanceDataBuffer: GBuf | null;
    },
    bvh: { tlas: { bvhNodes: GBuf }; instanceInverses: GBuf; blasAtlas: BLASAtlas },
    pointLightBuffer: GPUBuffer,
    getSurfaces: () => SurfaceData[],
    getRaytracing: () => boolean,
    getShadows: () => boolean,
): { node: ComputeNode; dispose: () => void } {
    const gpu: WavefrontGPU = {
        alive: true,
        pipelines: null,
        buffers: null,
        clearBG: null,
        swapCounterBG: null,
        swapBounceCounterBG: null,
        shadowTraceSceneBG: null,
        shadowTraceBG1: null,
        shadowTraceBvhBG: null,
        applyShadowBG: null,
        traceRayBGAlt: null,
        shadeBG1Alt: null,
        paramsBuffer: null,
        cachedWidth: 0,
        cachedHeight: 0,
        raygenSceneBG: null,
        raygenRayBG: null,
        traceSceneBG: null,
        traceRayBG: null,
        traceBvhBG: null,
        shadeBG0: null,
        shadeBG1: null,
        resolveBG: null,
        cachedBlasAtlas: null,
        cachedColorView: null,
        cachedDepthView: null,
        cachedEidView: null,
        cachedCapacity: capacity(),
        debugStagingBuffer: null,
        debugReadbackPending: false,
    };

    const node: ComputeNode = {
        name: "rt-wavefront",
        inputs: ["tlas-bvh-nodes", "point-light-data"],
        outputs: ["color", "eid", "depth"],
        async prepare(device: GPUDevice) {
            triggerWavefrontCompile(gpu, device, getSurfaces);
        },
        execute(ctx: ExecutionContext) {
            executeWavefront(gpu, ctx, render, bvh, pointLightBuffer, getRaytracing, getShadows);
        },
    };

    return {
        node,
        dispose: () => {
            gpu.alive = false;
        },
    };
}

traits(Raytracing, {
    defaults: () => ({ width: 0, height: 480 }),
});

export const RaytracingPlugin: Plugin = {
    name: "Raytracing",
    systems: [],
    components: {
        Raytracing,
    },
    dependencies: [RenderPlugin],

    async initialize(state: State) {
        const compute = Compute.from(state);
        const render = Render.from(state);
        if (!compute || !render) return;

        render.needsDepth = true;

        const maxPixels = maxPixelsForDevice(compute.device);
        let warnedClamp = false;

        render.viewportCap = (cameraEid, w, h) => {
            if (!state.hasComponent(cameraEid, Raytracing)) return { w, h };

            const rw = Raytracing.width[cameraEid] || 0;
            const rh = Raytracing.height[cameraEid] || 480;

            let rtW: number;
            let rtH: number;
            if (rw > 0 && rh > 0) {
                rtW = rw;
                rtH = rh;
            } else if (rh > 0 && h > 0) {
                rtH = rh;
                rtW = Math.max(1, Math.round(rh * (w / h)));
            } else if (rw > 0 && w > 0) {
                rtW = rw;
                rtH = Math.max(1, Math.round(rw * (h / w)));
            } else {
                rtW = w;
                rtH = h;
            }

            if (rtW * rtH > maxPixels) {
                const scale = Math.sqrt(maxPixels / (rtW * rtH));
                rtW = Math.max(1, Math.floor(rtW * scale));
                rtH = Math.max(1, Math.floor(rtH * scale));
                if (!warnedClamp) {
                    console.warn(
                        `RT resolution clamped to ${rtW}x${rtH} (GPU limit: ${maxPixels} pixels)`,
                    );
                    warnedClamp = true;
                }
            }

            return { w: rtW, h: rtH };
        };

        await initializeBVH(state);
        const bvh = BVH.from(state);
        if (!bvh) return;

        const getRaytracing = () => {
            for (const eid of state.query([Camera])) {
                if (Camera.active[eid] && state.hasComponent(eid, Raytracing)) {
                    return true;
                }
            }
            return false;
        };
        bvh.activeChecks.push(getRaytracing);

        const getShadows = () => {
            for (const eid of state.query([Camera])) {
                if (Camera.active[eid] && state.hasComponent(eid, Shadows)) return true;
            }
            return false;
        };

        const sg = compute.graph.subGraph("raytracing");
        sg.check = (eid) => state.hasComponent(eid, Raytracing);

        const pointLightBuffer = render.pointLightBuffer;

        const wavefront = createWavefrontNode(
            render,
            bvh,
            pointLightBuffer,
            surfaceRegistry.all,
            getRaytracing,
            getShadows,
        );
        sg.add(wavefront.node);
        state.onDispose(wavefront.dispose);
    },
};
