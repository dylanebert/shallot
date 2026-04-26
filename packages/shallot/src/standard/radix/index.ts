import type { ComputeNode, ExecutionContext } from "../compute";
import { beginComputePass } from "../compute";

const WG_X = 16;
const WG_Y = 16;
const WG_SIZE = WG_X * WG_Y;

export const RADIX_WG_SIZE = WG_SIZE;
const ITEMS_PER_WG = 2 * WG_SIZE;
const MAX_PREFIX_LEVELS = 3;

const SORT_PARAMS_STRUCT = /* wgsl */ `struct SortParams { count: u32, wgCount: u32 }`;
const PREFIX_PARAMS_STRUCT = /* wgsl */ `struct PrefixParams { count: u32 }`;

const histogramShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(2) var<uniform> params: SortParams;

${SORT_PARAMS_STRUCT}

override BIT: u32;

var<workgroup> bins: array<atomic<u32>, 16>;

@compute @workgroup_size(${WG_X}, ${WG_Y}, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wdim: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let workgroup = wid.x + wid.y * wdim.x;
    let gid = workgroup * ${WG_SIZE}u + tid;

    if (tid < 16u) {
        atomicStore(&bins[tid], 0u);
    }
    workgroupBarrier();

    if (gid < params.count && workgroup < params.wgCount) {
        let digit = (input[gid] >> BIT) & 0xfu;
        atomicAdd(&bins[digit], 1u);
    }
    workgroupBarrier();

    if (tid < 16u) {
        histograms[tid * params.wgCount + workgroup] = atomicLoad(&bins[tid]);
    }
}
`;

const scatterShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read> inKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> outKeys: array<u32>;
@group(0) @binding(2) var<storage, read> histograms: array<u32>;
@group(0) @binding(3) var<storage, read> inVals: array<u32>;
@group(0) @binding(4) var<storage, read_write> outVals: array<u32>;
@group(0) @binding(5) var<uniform> params: SortParams;

${SORT_PARAMS_STRUCT}

override BIT: u32;

var<workgroup> digit_bits: array<atomic<u32>, 128>;

@compute @workgroup_size(${WG_X}, ${WG_Y}, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wdim: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let workgroup = wid.x + wid.y * wdim.x;
    let gid = workgroup * ${WG_SIZE}u + tid;

    if (tid < 128u) { atomicStore(&digit_bits[tid], 0u); }
    workgroupBarrier();

    var digit = 16u;
    if (gid < params.count && workgroup < params.wgCount) {
        digit = (inKeys[gid] >> BIT) & 0xfu;
    }

    if (digit < 16u) {
        atomicOr(&digit_bits[digit * 8u + (tid >> 5u)], 1u << (tid & 31u));
    }
    workgroupBarrier();

    if (digit >= 16u) { return; }

    let word = tid >> 5u;
    var rank = 0u;
    for (var w = 0u; w < word; w++) {
        rank += countOneBits(atomicLoad(&digit_bits[digit * 8u + w]));
    }
    rank += countOneBits(atomicLoad(&digit_bits[digit * 8u + word]) & ((1u << (tid & 31u)) - 1u));

    let dst = histograms[digit * params.wgCount + workgroup] + rank;
    outKeys[dst] = inKeys[gid];
    outVals[dst] = inVals[gid];
}
`;

const prefixSumShader = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(2) var<uniform> params: PrefixParams;

${PREFIX_PARAMS_STRUCT}

var<workgroup> temp: array<u32, ${ITEMS_PER_WG * 2}>;

@compute @workgroup_size(${WG_X}, ${WG_Y}, 1)
fn scan(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wdim: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let workgroup = wid.x + wid.y * wdim.x;
    let base = workgroup * ${WG_SIZE}u;
    let gid = base + tid;
    let eid = gid * 2;

    temp[tid * 2] = select(data[eid], 0u, eid >= params.count);
    temp[tid * 2 + 1] = select(data[eid + 1], 0u, eid + 1 >= params.count);

    var offset = 1u;
    for (var d = ${ITEMS_PER_WG}u >> 1; d > 0; d >>= 1) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (tid * 2 + 1) - 1;
            let bi = offset * (tid * 2 + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (tid == 0) {
        blockSums[workgroup] = temp[${ITEMS_PER_WG}u - 1];
        temp[${ITEMS_PER_WG}u - 1] = 0;
    }

    for (var d = 1u; d < ${ITEMS_PER_WG}u; d *= 2) {
        offset >>= 1;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (tid * 2 + 1) - 1;
            let bi = offset * (tid * 2 + 2) - 1;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    if (eid < params.count) { data[eid] = temp[tid * 2]; }
    if (eid + 1 < params.count) { data[eid + 1] = temp[tid * 2 + 1]; }
}

@compute @workgroup_size(${WG_X}, ${WG_Y}, 1)
fn addBlocks(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wdim: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let workgroup = wid.x + wid.y * wdim.x;
    let eid = (workgroup * ${WG_SIZE}u + tid) * 2;

    if (eid >= params.count) { return; }

    let sum = blockSums[workgroup];
    data[eid] += sum;
    if (eid + 1 < params.count) { data[eid + 1] += sum; }
}
`;

function dispatchSize(device: GPUDevice, count: number): [number, number] {
    const max = device.limits.maxComputeWorkgroupsPerDimension;
    if (count <= max) return [count, 1];
    const x = Math.ceil(Math.sqrt(count));
    return [x, Math.ceil(count / x)];
}

interface PrefixLevel {
    paramsBuffer: GPUBuffer;
    blockSums: GPUBuffer;
    bindGroup: GPUBindGroup;
    count: number;
    dispatch: [number, number];
}

interface PrefixSumState {
    scanPipeline: GPUComputePipeline;
    addBlocksPipeline: GPUComputePipeline;
    layout: GPUBindGroupLayout;
    levels: PrefixLevel[];
    device: GPUDevice;
}

function buildPrefixLevels(
    device: GPUDevice,
    layout: GPUBindGroupLayout,
    data: GPUBuffer,
    count: number,
): PrefixLevel[] {
    const levels: PrefixLevel[] = [];
    let currentData = data;
    let currentCount = count;

    for (let i = 0; i < MAX_PREFIX_LEVELS; i++) {
        const wgCount = Math.max(Math.ceil(currentCount / ITEMS_PER_WG), 1);
        const dispatch = dispatchSize(device, wgCount);

        const paramsBuffer = device.createBuffer({
            label: `prefix-params-${i}`,
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([currentCount]));

        const blockSums = device.createBuffer({
            label: `prefix-blockSums-${i}`,
            size: Math.max(wgCount * 4, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const bindGroup = device.createBindGroup({
            layout,
            entries: [
                { binding: 0, resource: { buffer: currentData } },
                { binding: 1, resource: { buffer: blockSums } },
                { binding: 2, resource: { buffer: paramsBuffer } },
            ],
        });

        levels.push({ paramsBuffer, blockSums, bindGroup, count: currentCount, dispatch });

        if (wgCount <= 1) break;
        currentData = blockSums;
        currentCount = wgCount;
    }

    return levels;
}

function disposePrefixLevels(levels: PrefixLevel[]): void {
    for (const level of levels) {
        level.paramsBuffer.destroy();
        level.blockSums.destroy();
    }
}

async function createPrefixSum(
    device: GPUDevice,
    data: GPUBuffer,
    count: number,
): Promise<PrefixSumState> {
    const module = device.createShaderModule({ code: prefixSumShader });

    const layout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });

    const [scanPipeline, addBlocksPipeline] = await Promise.all([
        device.createComputePipelineAsync({
            label: "prefix-scan",
            layout: pipelineLayout,
            compute: { module, entryPoint: "scan" },
        }),
        device.createComputePipelineAsync({
            label: "prefix-add",
            layout: pipelineLayout,
            compute: { module, entryPoint: "addBlocks" },
        }),
    ]);

    const levels = buildPrefixLevels(device, layout, data, count);

    return { scanPipeline, addBlocksPipeline, layout, levels, device };
}

function rebuildPrefixSum(state: PrefixSumState, data: GPUBuffer, count: number): void {
    disposePrefixLevels(state.levels);
    state.levels = buildPrefixLevels(state.device, state.layout, data, count);
}

function disposePrefixSum(state: PrefixSumState): void {
    disposePrefixLevels(state.levels);
}

function dispatchPrefixSum(state: PrefixSumState, pass: GPUComputePassEncoder): void {
    const { levels, scanPipeline, addBlocksPipeline } = state;

    for (const level of levels) {
        pass.setPipeline(scanPipeline);
        pass.setBindGroup(0, level.bindGroup);
        pass.dispatchWorkgroups(level.dispatch[0], level.dispatch[1], 1);
    }

    for (let i = levels.length - 2; i >= 0; i--) {
        pass.setPipeline(addBlocksPipeline);
        pass.setBindGroup(0, levels[i].bindGroup);
        pass.dispatchWorkgroups(levels[i].dispatch[0], levels[i].dispatch[1], 1);
    }
}

export {
    createPrefixSum,
    rebuildPrefixSum,
    dispatchPrefixSum,
    disposePrefixSum,
    type PrefixSumState,
};

interface RadixPass {
    histogram: { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup };
    scatter: { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup };
}

export interface RadixSortState {
    device: GPUDevice;
    histogramLayout: GPUBindGroupLayout;
    scatterLayout: GPUBindGroupLayout;
    histogramPipelines: GPUComputePipeline[];
    scatterPipelines: GPUComputePipeline[];
    paramsBuffer: GPUBuffer;
    tmpKeys: GPUBuffer;
    tmpVals: GPUBuffer;
    histograms: GPUBuffer;
    passes: RadixPass[];
    prefixSum: PrefixSumState;
    indirectBuffer: GPUBuffer;
    count: number;
}

export interface RadixSortConfig {
    keys: GPUBuffer;
    values: GPUBuffer;
    count: number;
}

function buildRadixBindGroups(
    device: GPUDevice,
    histogramLayout: GPUBindGroupLayout,
    scatterLayout: GPUBindGroupLayout,
    histogramPipelines: GPUComputePipeline[],
    scatterPipelines: GPUComputePipeline[],
    keys: GPUBuffer,
    values: GPUBuffer,
    tmpKeys: GPUBuffer,
    tmpVals: GPUBuffer,
    histograms: GPUBuffer,
    paramsBuffer: GPUBuffer,
): RadixPass[] {
    const passes: RadixPass[] = [];

    for (let i = 0; i < 8; i++) {
        const even = i % 2 === 0;
        const inK = even ? keys : tmpKeys;
        const inV = even ? values : tmpVals;
        const outK = even ? tmpKeys : keys;
        const outV = even ? tmpVals : values;

        passes.push({
            histogram: {
                pipeline: histogramPipelines[i],
                bindGroup: device.createBindGroup({
                    layout: histogramLayout,
                    entries: [
                        { binding: 0, resource: { buffer: inK } },
                        { binding: 1, resource: { buffer: histograms } },
                        { binding: 2, resource: { buffer: paramsBuffer } },
                    ],
                }),
            },
            scatter: {
                pipeline: scatterPipelines[i],
                bindGroup: device.createBindGroup({
                    layout: scatterLayout,
                    entries: [
                        { binding: 0, resource: { buffer: inK } },
                        { binding: 1, resource: { buffer: outK } },
                        { binding: 2, resource: { buffer: histograms } },
                        { binding: 3, resource: { buffer: inV } },
                        { binding: 4, resource: { buffer: outV } },
                        { binding: 5, resource: { buffer: paramsBuffer } },
                    ],
                }),
            },
        });
    }

    return passes;
}

export async function createRadixSort(
    device: GPUDevice,
    config: RadixSortConfig,
): Promise<RadixSortState> {
    const { keys, values, count } = config;
    const wgCount = Math.ceil(count / WG_SIZE);

    const indirectBuffer = device.createBuffer({
        label: "radix-sort-indirect",
        size: 12,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indirectBuffer, 0, new Uint32Array([wgCount, 1, 1]));

    const paramsBuffer = device.createBuffer({
        label: "radix-sort-params",
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([count, wgCount]));

    const tmpKeys = device.createBuffer({
        label: "radix-tmpKeys",
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const tmpVals = device.createBuffer({
        label: "radix-tmpVals",
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const histograms = device.createBuffer({
        label: "radix-histograms",
        size: 16 * wgCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const prefixSum = await createPrefixSum(device, histograms, 16 * wgCount);

    const histogramModule = device.createShaderModule({ code: histogramShader });
    const scatterModule = device.createShaderModule({ code: scatterShader });

    const histogramLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });

    const scatterLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });

    const histogramPipelines: GPUComputePipeline[] = [];
    const scatterPipelines: GPUComputePipeline[] = [];

    const pipelinePromises: Promise<void>[] = [];
    for (let bit = 0; bit < 32; bit += 4) {
        const idx = bit / 4;
        pipelinePromises.push(
            (async () => {
                const [hp, sp] = await Promise.all([
                    device.createComputePipelineAsync({
                        label: "radix-histogram",
                        layout: device.createPipelineLayout({
                            bindGroupLayouts: [histogramLayout],
                        }),
                        compute: {
                            module: histogramModule,
                            entryPoint: "main",
                            constants: { BIT: bit },
                        },
                    }),
                    device.createComputePipelineAsync({
                        label: "radix-scatter",
                        layout: device.createPipelineLayout({
                            bindGroupLayouts: [scatterLayout],
                        }),
                        compute: {
                            module: scatterModule,
                            entryPoint: "main",
                            constants: { BIT: bit },
                        },
                    }),
                ]);
                histogramPipelines[idx] = hp;
                scatterPipelines[idx] = sp;
            })(),
        );
    }
    await Promise.all(pipelinePromises);

    const passes = buildRadixBindGroups(
        device,
        histogramLayout,
        scatterLayout,
        histogramPipelines,
        scatterPipelines,
        keys,
        values,
        tmpKeys,
        tmpVals,
        histograms,
        paramsBuffer,
    );

    return {
        device,
        histogramLayout,
        scatterLayout,
        histogramPipelines,
        scatterPipelines,
        paramsBuffer,
        tmpKeys,
        tmpVals,
        histograms,
        passes,
        prefixSum,
        indirectBuffer,
        count,
    };
}

export function rebuildRadixSort(
    state: RadixSortState,
    keys: GPUBuffer,
    values: GPUBuffer,
    newCount: number,
): void {
    const { device } = state;
    const wgCount = Math.ceil(newCount / WG_SIZE);

    state.tmpKeys.destroy();
    state.tmpVals.destroy();
    state.histograms.destroy();

    state.count = newCount;

    device.queue.writeBuffer(state.paramsBuffer, 0, new Uint32Array([newCount, wgCount]));
    device.queue.writeBuffer(state.indirectBuffer, 0, new Uint32Array([wgCount, 1, 1]));

    state.tmpKeys = device.createBuffer({
        label: "radix-tmpKeys",
        size: newCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    state.tmpVals = device.createBuffer({
        label: "radix-tmpVals",
        size: newCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    state.histograms = device.createBuffer({
        label: "radix-histograms",
        size: 16 * wgCount * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    rebuildPrefixSum(state.prefixSum, state.histograms, 16 * wgCount);

    state.passes = buildRadixBindGroups(
        device,
        state.histogramLayout,
        state.scatterLayout,
        state.histogramPipelines,
        state.scatterPipelines,
        keys,
        values,
        state.tmpKeys,
        state.tmpVals,
        state.histograms,
        state.paramsBuffer,
    );
}

export function disposeRadixSort(state: RadixSortState): void {
    state.tmpKeys.destroy();
    state.tmpVals.destroy();
    state.histograms.destroy();
    state.paramsBuffer.destroy();
    state.indirectBuffer.destroy();
    disposePrefixSum(state.prefixSum);
}

export function dispatchRadixSort(state: RadixSortState, pass: GPUComputePassEncoder): void {
    for (const p of state.passes) {
        pass.setPipeline(p.histogram.pipeline);
        pass.setBindGroup(0, p.histogram.bindGroup);
        pass.dispatchWorkgroupsIndirect(state.indirectBuffer, 0);

        dispatchPrefixSum(state.prefixSum, pass);

        pass.setPipeline(p.scatter.pipeline);
        pass.setBindGroup(0, p.scatter.bindGroup);
        pass.dispatchWorkgroupsIndirect(state.indirectBuffer, 0);
    }
}

export function createRadixSortNode(config: RadixSortConfig): ComputeNode {
    let sort: RadixSortState | null = null;

    return {
        name: "radix-sort",
        inputs: [],
        outputs: [],

        async prepare(device: GPUDevice) {
            sort = await createRadixSort(device, config);
        },

        execute(ctx: ExecutionContext) {
            const pass = beginComputePass(ctx.encoder, ctx.timestampWrites?.("radix-sort"));
            dispatchRadixSort(sort!, pass);
            pass.end();
        },
    };
}
