// Gathering per-combo culled ranges into mesh-major runs lets an atlas draw each mesh once.
// View-independent producers instead duplicate their direct range across combos. Every instance
// retains its identity/transform payload and gains a combo index. Pipelines are shared per World;
// each atlas owns its gathered list, indirect args and metadata.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { World } from "../../engine";
import {
    dispatchWorkgroups,
    dispatchWorkgroups2,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";
import { DrawIndexedIndirect } from "./registry";

// One indirect record per casting mesh; instanceCount sums combo survivors and firstInstance
// addresses its gathered run. Derive the byte stride from DrawIndexedIndirect to keep writers aligned.
export const SHADOW_ARG_STRIDE = d.sizeOf(DrawIndexedIndirect);

interface RegatherState {
    aPipe: GPUComputePipeline | null;
    bPipe: GPUComputePipeline | null;
    aLayout: GPUBindGroupLayout | null;
    bLayout: GPUBindGroupLayout | null;
    pipelineDevice: GPUDevice | null;
    pipelineCapacity: number;
}

const regatherStateKey = { create: () => createRegatherState() };
const createRegatherState = (): RegatherState => ({
    aPipe: null,
    bPipe: null,
    aLayout: null,
    bLayout: null,
    pipelineDevice: null,
    pipelineCapacity: 0,
});

function regatherState(world: World): RegatherState {
    return world.resource(regatherStateKey);
}

export function initializeRegatherState(world: World): void {
    world.resource(regatherStateKey);
}

/** Pass A's actual compiled source, generated lazily for device-free indirect-record contract checks. */
export const regatherArgsWgsl = (): string =>
    tgpu.resolve({
        names: "strict",
        externals: { DrawIndexedIndirect },
        template: /* wgsl */ `
struct RgParams { draws: u32, combos: u32, pairCount: u32, phase: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: array<DrawIndexedIndirect>;
@group(0) @binding(1) var<storage, read_write> rgMeta: array<u32>;       // [combo slots (C) | draw pairs (D)]
@group(0) @binding(2) var<storage, read_write> shadowArgs: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<uniform> params: RgParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D = params.draws;
    let C = params.combos;
    let i = gid.x;
    if (i >= D) { return; }
    let scratch = C + D;
    let step = params.phase;
    if (step != 0u) {
        // Ping-pong inclusive scan: each dispatch observes only the preceding level.
        let lane = countTrailingZeros(step) % 2u;
        let src = scratch + lane * D;
        let dst = scratch + (1u - lane) * D;
        var total = rgMeta[src + i];
        if (i >= step) { total += rgMeta[src + i - step]; }
        rgMeta[dst + i] = total;
        shadowArgs[i].firstInstance = total - shadowArgs[i].instanceCount;
        return;
    }
    let pc = params.pairCount;
    let pair = rgMeta[C + i];
    var total = 0u;
    var src = pair;
    if (pc == 0u) {
        total = drawArgs[src].instanceCount * C;
    } else {
        src = rgMeta[0] * pc + pair;
        for (var c = 0u; c < C; c++) {
            total += drawArgs[rgMeta[c] * pc + pair].instanceCount;
        }
    }
    shadowArgs[i].indexCount = drawArgs[src].indexCount;
    shadowArgs[i].instanceCount = total;
    shadowArgs[i].firstIndex = drawArgs[src].firstIndex;
    shadowArgs[i].baseVertex = drawArgs[src].baseVertex;
    shadowArgs[i].firstInstance = 0u;
    rgMeta[scratch + i] = total;
}`,
    });

const regatherEidsWgsl = (): string =>
    tgpu.resolve({
        names: "strict",
        externals: { DrawIndexedIndirect },
        template: /* wgsl */ `
struct RgParams { draws: u32, combos: u32, pairCount: u32, capacity: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: array<DrawIndexedIndirect>;
@group(0) @binding(1) var<storage, read> packedEids: array<vec4u>;
@group(0) @binding(2) var<storage, read> shadowArgs: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<storage, read> rgMeta: array<u32>;
@group(0) @binding(4) var<storage, read_write> shadowEids: array<vec4u>;
@group(0) @binding(5) var<uniform> params: RgParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let C = params.combos;
    if (gid.x >= params.capacity) { return; }
    let t = gid.x + gid.y * params.capacity;
    let last = shadowArgs[params.draws - 1u];
    if (t >= last.firstInstance + last.instanceCount) { return; }
    // Upper bound skips empty mesh runs, including consecutive empty runs.
    var lo = 0u;
    var hi = params.draws;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (shadowArgs[mid].firstInstance <= t) { lo = mid + 1u; }
        else { hi = mid; }
    }
    let i = lo - 1u;
    let pc = params.pairCount;
    let pair = rgMeta[C + i];
    var k = t - shadowArgs[i].firstInstance;
    for (var c = 0u; c < C; c++) {
        let idx = select(rgMeta[c] * pc + pair, pair, pc == 0u);
        let count = drawArgs[idx].instanceCount;
        if (k < count) {
            let instance = packedEids[drawArgs[idx].firstInstance + k];
            shadowEids[t] = vec4u(instance.xyz, c);
            return;
        }
        k -= count;
    }
}`,
    });

/** compile the shared A/B re-gather pipelines once (idempotent): called from `prepareStandardRenderer`, folded into its
 * warm `Promise.all`. Every {@link Regather} instance in this World uses these layouts. */
export async function prepareRegather(
    world: World,
    device: GPUDevice,
    capacity: number,
): Promise<void> {
    if (
        regatherState(world).aPipe &&
        regatherState(world).pipelineDevice === device &&
        regatherState(world).pipelineCapacity === capacity
    )
        return;
    regatherState(world).aPipe = null;
    regatherState(world).bPipe = null;
    regatherState(world).aLayout = null;
    regatherState(world).bLayout = null;
    regatherState(world).pipelineDevice = device;
    regatherState(world).pipelineCapacity = capacity;
    regatherState(world).aLayout = device.createBindGroupLayout({
        label: "standard-regather-a",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });
    regatherState(world).bLayout = device.createBindGroupLayout({
        label: "standard-regather-b",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
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
    const aWgsl = regatherArgsWgsl();
    const bWgsl = regatherEidsWgsl();

    const [a, b] = await Promise.all([
        device.createComputePipelineAsync({
            label: "standard-regather-a",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [regatherState(world).aLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    label: "standard-regather-a",
                    code: aWgsl,
                }),
                entryPoint: "main",
            },
        }),
        device.createComputePipelineAsync({
            label: "standard-regather-b",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [regatherState(world).bLayout],
            }),
            compute: {
                module: device.createShaderModule({ label: "standard-regather-b", code: bWgsl }),
                entryPoint: "main",
            },
        }),
    ]);
    regatherState(world).aPipe = a;
    regatherState(world).bPipe = b;
}

/** one shadow atlas's re-gather instance: its own packed list + indirect args + meta, sharing the
 * World-owned A/B pipelines. The point atlas and the CSM cascade atlas each own one. */
export interface Regather {
    /** the re-gathered instance list (`eid, globalTransformRow, encodedMeshInstanceSlot, combo`), bound at the consumer
     * pipeline's `eids` lane. `null` until {@link Regather.ensure} allocates it (the first casting frame). */
    eids(): GPUBuffer | null;
    /** the indirect buffer the atlas render pass draws from: one DrawIndexedIndirect record per casting
     * mesh (Pass A fills it). `null` until {@link Regather.reserve} allocates it. */
    args(): GPUBuffer | null;
    /** lazily allocate the packed list (at least `maxCombos × capacity`, the provably-safe bound: each combo
     * view slot holds ≤ capacity culled eids); capacity doubles when it grows and never shrinks. Fires the
     * `onAlloc` callback registered via {@link Regather.reset} (standard rebuilds the bind groups that bind this lane). Idempotent once allocated. */
    ensure(maxCombos: number, capacity: number): void;
    /** preflight the largest batch before recording any run into an encoder. A run never reallocates this
     * shared output: earlier GPU commands in the same unsubmitted encoder must keep the buffer they captured. */
    reserve(maxDraws: number): void;
    /** upload the per-frame meta + run Pass A then Pass B on `cpass` (one compute pass, the intra-pass
     * dispatch ordering the MeshInstance pack relies on). `comboSlots` = the view slot each dense combo packed into
     * (the first `comboCount`); `drawPairs` = the source indirect-record indices (the first `drawCount`);
     * `pairCount` = the pack's pair stride, or zero for a
     * view-independent producer whose direct range is duplicated across combos. */
    run(
        cpass: GPUComputePassEncoder,
        drawArgs: GPUBuffer,
        packedEids: GPUBuffer,
        comboSlots: number[],
        comboCount: number,
        drawPairs: number[],
        drawCount: number,
        pairCount: number,
        runIndex?: number,
    ): void;
    /** (re)create the per-instance params buffer + clear the caches on a (re)build; `onAlloc` is the standard
     * side-effect run when `ensure` allocates the packed list (clear the bind-group cache + bump the gen). */
    reset(onAlloc: () => void): void;
    /** destroy every GPU buffer this instance owns (at plugin dispose). */
    dispose(): void;
}

/** create a shadow-atlas re-gather instance. `label` names its GPU buffers. The A/B pipelines must be
 * compiled once via {@link prepareRegather} before {@link Regather.runApp}. */
export function createRegather(world: World, label: string): Regather {
    let _eids: GPUBuffer | null = null;
    let _eidCapacity = 0;
    let _eidCombos = 0;
    let _args: GPUBuffer | null = null;
    let _argsCap = 0;
    let _meta: GPUBuffer[] = [];
    let _metaCap: number[] = [];
    let _metaStaging: Uint32Array[] = [];
    let _params: GPUBuffer[] = [];
    const _paramsStaging = new Uint32Array(4);
    let _aGroups: ({
        args: GPUBuffer;
        meta: GPUBuffer;
        drawArgs: GPUBuffer;
        group: GPUBindGroup;
    } | null)[] = [];
    let _bGroups: ({
        args: GPUBuffer;
        meta: GPUBuffer;
        eids: GPUBuffer;
        drawArgs: GPUBuffer;
        packed: GPUBuffer;
        group: GPUBindGroup;
    } | null)[] = [];
    let _onAlloc: () => void = () => {};

    function ensureArgs(world: World, count: number): void {
        if (_args && _argsCap >= count) return;
        _args?.destroy();
        _argsCap = Math.max(count, 8);
        _args = world.gpu.device.createBuffer({
            label: `standard-${label}-shadow-args`,
            size: _argsCap * SHADOW_ARG_STRIDE,
            usage:
                GPUBufferUsage.INDIRECT |
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
        });
        _aGroups.length = 0;
        _bGroups.length = 0;
    }

    function ensureMeta(world: World, n: number, runIndex: number): GPUBuffer {
        if (_meta[runIndex] && _metaCap[runIndex] >= n) return _meta[runIndex];
        _meta[runIndex]?.destroy();
        const cap = Math.max(n, 64);
        const buffer = world.gpu.device.createBuffer({
            label: `standard-${label}-regather-meta-${runIndex}`,
            size: cap * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        _meta[runIndex] = buffer;
        _metaCap[runIndex] = cap;
        _metaStaging[runIndex] = new Uint32Array(cap);
        _aGroups.length = 0;
        _bGroups.length = 0;
        return buffer;
    }

    function params(world: World, runIndex: number): GPUBuffer {
        let buffer = _params[runIndex];
        if (buffer) return buffer;
        buffer = world.gpu.device.createBuffer({
            label: `standard-${label}-regather-params-${runIndex}`,
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        _params[runIndex] = buffer;
        return buffer;
    }

    // Pass A bind group (drawArgs + meta → args). `drawArgs` is the MeshInstance pack's shared indirect buffer (read
    // from a casting Draw — the atlas stays producer-agnostic), which reallocs on pack growth
    function aGroup(
        world: World,
        drawArgs: GPUBuffer,
        meta: GPUBuffer,
        runIndex: number,
    ): GPUBindGroup {
        const cached = _aGroups[runIndex];
        if (
            cached &&
            cached.args === _args &&
            cached.meta === meta &&
            cached.drawArgs === drawArgs
        ) {
            return cached.group;
        }
        const group = world.gpu.device.createBindGroup({
            label: `standard-${label}-regather-a`,
            layout: regatherState(world).aLayout!,
            entries: [
                { binding: 0, resource: { buffer: drawArgs } },
                { binding: 1, resource: { buffer: meta } },
                { binding: 2, resource: { buffer: _args! } },
                { binding: 3, resource: { buffer: params(world, runIndex) } },
            ],
        });
        _aGroups[runIndex] = { args: _args!, meta, drawArgs, group };
        return group;
    }

    function bGroup(
        world: World,
        drawArgs: GPUBuffer,
        packed: GPUBuffer,
        meta: GPUBuffer,
        runIndex: number,
    ): GPUBindGroup {
        const cached = _bGroups[runIndex];
        if (
            cached &&
            cached.args === _args &&
            cached.meta === meta &&
            cached.eids === _eids &&
            cached.drawArgs === drawArgs &&
            cached.packed === packed
        ) {
            return cached.group;
        }
        const group = world.gpu.device.createBindGroup({
            label: `standard-${label}-regather-b`,
            layout: regatherState(world).bLayout!,
            entries: [
                { binding: 0, resource: { buffer: drawArgs } },
                { binding: 1, resource: { buffer: packed } },
                { binding: 2, resource: { buffer: _args! } },
                { binding: 3, resource: { buffer: meta } },
                { binding: 4, resource: { buffer: _eids! } },
                { binding: 5, resource: { buffer: params(world, runIndex) } },
            ],
        });
        _bGroups[runIndex] = {
            args: _args!,
            meta,
            eids: _eids!,
            drawArgs,
            packed,
            group,
        };
        return group;
    }

    return {
        eids: () => _eids,
        args: () => _args,
        ensure(maxCombos: number, capacity: number): void {
            if (_eids && _eidCapacity >= capacity && _eidCombos >= maxCombos) return;
            _eids?.destroy();
            // doubles like the eid columns, so a high-water mark rising each frame reallocates logarithmically
            if (capacity > _eidCapacity) _eidCapacity = Math.max(capacity, _eidCapacity * 2);
            _eidCombos = maxCombos;
            _eids = world.gpu.device.createBuffer({
                label: `standard-${label}-regather-eids`,
                size: maxCombos * _eidCapacity * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            _onAlloc();
        },
        reserve(maxDraws: number): void {
            ensureArgs(world, maxDraws);
        },
        run(
            cpass,
            drawArgs,
            packedEids,
            comboSlots,
            comboCount,
            drawPairs,
            drawCount,
            pairCount,
            runIndex = 0,
        ): void {
            const C = comboCount;
            const D = drawCount;
            if (!_args || _argsCap < D) {
                throw new Error(
                    `standard ${label} re-gather run has ${D} draws after a ${_argsCap}-draw reserve`,
                );
            }
            const meta = ensureMeta(world, C + 3 * D, runIndex);
            const staging = _metaStaging[runIndex];
            // meta = [combo slots (C) | draw pairs (D)]: the view slot each dense combo packed into (its
            // per-combo culled counts live in drawArgs there), and the (surface,mesh) pair each casting draw owns
            for (let c = 0; c < C; c++) staging[c] = comboSlots[c];
            for (let i = 0; i < D; i++) staging[C + i] = drawPairs[i];
            world.gpu.device.queue.writeBuffer(
                meta,
                0,
                staging as Uint32Array<ArrayBuffer>,
                0,
                C + D,
            );
            if (D === 0 || C === 0) return;
            _paramsStaging[0] = D;
            _paramsStaging[1] = C;
            _paramsStaging[2] = pairCount;
            // Each scan level needs its own uniform: queue writes precede the whole encoder.
            setComputePipeline(cpass, regatherState(world).aPipe!);
            let level = 0;
            for (let step = 0; step < D; step = step === 0 ? 1 : step * 2) {
                const index = runIndex * 33 + level++;
                _paramsStaging[3] = step;
                world.gpu.device.queue.writeBuffer(
                    params(world, index),
                    0,
                    _paramsStaging as Uint32Array<ArrayBuffer>,
                );
                setComputeBindGroup(cpass, 0, aGroup(world, drawArgs, meta, index));
                dispatchWorkgroups(cpass, Math.ceil(D / 64));
            }
            const index = runIndex * 33 + 32;
            _paramsStaging[3] = _eidCapacity;
            world.gpu.device.queue.writeBuffer(
                params(world, index),
                0,
                _paramsStaging as Uint32Array<ArrayBuffer>,
            );
            setComputePipeline(cpass, regatherState(world).bPipe!);
            setComputeBindGroup(cpass, 0, bGroup(world, drawArgs, packedEids, meta, index));
            dispatchWorkgroups2(cpass, Math.ceil(_eidCapacity / 64), C);
        },
        reset(onAlloc: () => void): void {
            _onAlloc = onAlloc;
            _eids?.destroy();
            _eids = null;
            _eidCapacity = 0;
            _eidCombos = 0;
            _args?.destroy();
            _args = null;
            _argsCap = 0;
            for (const buffer of _meta) buffer.destroy();
            _meta = [];
            _metaCap = [];
            _metaStaging = [];
            for (const buffer of _params) buffer?.destroy();
            _params = [];
            _aGroups = [];
            _bGroups = [];
        },
        dispose(): void {
            _eids?.destroy();
            _args?.destroy();
            for (const buffer of _meta) buffer.destroy();
            for (const buffer of _params) buffer?.destroy();
            _eids = null;
            _eidCapacity = 0;
            _eidCombos = 0;
            _args = null;
            _argsCap = 0;
            _meta = [];
            _metaCap = [];
            _metaStaging = [];
            _params = [];
            _aGroups = [];
            _bGroups = [];
        },
    };
}
