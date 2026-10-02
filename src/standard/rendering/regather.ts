// The shadow-atlas re-gather: concatenate the per-combo *culled* regions the MeshInstance pack wrote (slot-major
// `drawArgs` + the `packedEids` pool), or duplicate a view-independent producer's direct range, into one
// contiguous, mesh-major run per casting mesh + a per-instance combo index. Each shadow atlas (the
// point/spot tiles, the CSM cascade tiles) instantiates its own `Regather`; the two A/B compute pipelines
// are geometry-blind (they read slot-major counts + the instance pool alone, with no projection or mesh
// knowledge). Each World owns the shared pipeline pair; each atlas owns its output buffers. The re-gather is a *consumer* of the cull spine's output (`render` owns the spine that
// feeds it); it knows sear-private concepts (the packing convention below, the atlas record shape, the
// `eids`-lane swap), so it lives here, not in render (render stays renderer-agnostic).

import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { World } from "../../engine";
import { DrawIndexedIndirect } from "./registry";

// one DrawIndexedIndirect record per casting mesh, written by Pass A: instanceCount = Σ combo
// survivors, firstInstance = the mesh's base into the re-gathered list. Stride derived from the schema
// (a second hand-authored stride is layout drift waiting to happen).
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

/** Create this world's regather pipeline state during StandardRenderer initialization. */
export function initializeRegatherState(world: World): void {
    world.resource(regatherStateKey);
}

/** Pass A's exact compiled source, exposed lazily for the device-free indirect-record contract test. */
export const regatherArgsWgsl = (): string =>
    tgpu.resolve({
        names: "strict",
        externals: { DrawIndexedIndirect },
        template: /* wgsl */ `
struct RgParams { draws: u32, combos: u32, pairCount: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: array<DrawIndexedIndirect>;
@group(0) @binding(1) var<storage, read> rgMeta: array<u32>;       // [combo slots (C) | draw pairs (D)]
@group(0) @binding(2) var<storage, read_write> shadowArgs: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<uniform> params: RgParams;
@compute @workgroup_size(1)
fn main() {
    let D = params.draws;
    let C = params.combos;
    let pc = params.pairCount;
    let slot0 = rgMeta[0]; // any combo slot carries the static lanes (the pack seeds every slot)
    var base = 0u;       // running exclusive prefix over the per-mesh totals
    for (var i = 0u; i < D; i = i + 1u) {
        let pair = rgMeta[C + i];
        var total = 0u;
        var src = pair;
        if (pc == 0u) {
            total = drawArgs[src].instanceCount * C;
        } else {
            src = slot0 * pc + pair;
            for (var c = 0u; c < C; c = c + 1u) {
                total = total + drawArgs[rgMeta[c] * pc + pair].instanceCount;
            }
        }
        shadowArgs[i].indexCount = drawArgs[src].indexCount;
        shadowArgs[i].instanceCount = total;
        shadowArgs[i].firstIndex = drawArgs[src].firstIndex;
        shadowArgs[i].baseVertex = drawArgs[src].baseVertex;
        shadowArgs[i].firstInstance = base;
        base = base + total;
    }
}`,
    });

const regatherEidsWgsl = (): string =>
    tgpu.resolve({
        names: "strict",
        externals: { DrawIndexedIndirect },
        template: /* wgsl */ `
struct RgParams { draws: u32, combos: u32, pairCount: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: array<DrawIndexedIndirect>;
@group(0) @binding(1) var<storage, read> packedEids: array<vec4u>;
@group(0) @binding(2) var<storage, read> shadowArgs: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<storage, read> rgMeta: array<u32>;
@group(0) @binding(4) var<storage, read_write> shadowEids: array<vec4u>;
@group(0) @binding(5) var<uniform> params: RgParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let C = params.combos;
    let t = gid.x;
    if (t >= params.draws * C) { return; }
    let i = t / C;
    let c = t % C;
    let pc = params.pairCount;
    let pair = rgMeta[C + i];
    let idx = select(rgMeta[c] * pc + pair, pair, pc == 0u);
    let cnt = drawArgs[idx].instanceCount;
    if (cnt == 0u) { return; }
    let src = drawArgs[idx].firstInstance;      // base into packedEids for (combo c, mesh i)
    var off = 0u;                               // within-run offset: Σ earlier combos' counts for this mesh
    if (pc == 0u) {
        off = c * cnt;
    } else {
        for (var cc = 0u; cc < c; cc = cc + 1u) {
            off = off + drawArgs[rgMeta[cc] * pc + pair].instanceCount;
        }
    }
    let dst = shadowArgs[i].firstInstance + off; // the mesh's run base + the combo's within-run offset
    for (var k = 0u; k < cnt; k = k + 1u) {
        let instance = packedEids[src + k];
        shadowEids[dst + k] = vec4u(instance.xyz, c);
    }
}`,
    });

/** compile the shared A/B re-gather pipelines once (idempotent): called from `prepareSear`, folded into its
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
    // Pass A — one thread: for each casting mesh, sum its per-combo culled counts (the spine's drawArgs at
    // each combo slot), exclusive-prefix the totals into per-mesh run bases, and write one DrawIndexedIndirect
    // record (instanceCount = the sum, firstInstance = the base; the static indexCount/firstIndex from any
    // combo slot, which the pack seeds per slot). D + C are tiny, so a serial single thread is free
    regatherState(world).aLayout = device.createBindGroupLayout({
        label: "sear-regather-a",
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
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });
    // Pass B — one thread per (casting mesh, combo): copy that combo's culled eids from the spine's
    // packedEids region into the mesh's contiguous run at the combo's within-run offset (Σ earlier combos'
    // counts), setting the payload's combo lane. The serial inner copy is the per-(mesh, combo)
    // count; a per-instance dispatch is the deferred optimization if a mesh ever owns a large
    // per-combo count
    regatherState(world).bLayout = device.createBindGroupLayout({
        label: "sear-regather-b",
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
            label: "sear-regather-a",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [regatherState(world).aLayout],
            }),
            compute: {
                module: device.createShaderModule({
                    label: "sear-regather-a",
                    code: aWgsl,
                }),
                entryPoint: "main",
            },
        }),
        device.createComputePipelineAsync({
            label: "sear-regather-b",
            layout: device.createPipelineLayout({
                bindGroupLayouts: [regatherState(world).bLayout],
            }),
            compute: {
                module: device.createShaderModule({ label: "sear-regather-b", code: bWgsl }),
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
    /** lazily allocate the packed list (sized `maxCombos × capacity`, the provably-safe bound: each combo
     * view slot holds ≤ capacity culled eids). Fires the `onAlloc` callback registered via
     * {@link Regather.reset} (sear rebuilds the bind groups that bind this lane). Idempotent once allocated. */
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
    /** (re)create the per-instance params buffer + clear the caches on a (re)build; `onAlloc` is the sear
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

    // (re)allocate the per-mesh indirect args (one DrawIndexedIndirect record per casting draw); grows as the
    // casting-draw count rises, invalidating the bind groups on grow
    function ensureArgs(world: World, count: number): void {
        if (_args && _argsCap >= count) return;
        _args?.destroy();
        _argsCap = Math.max(count, 8);
        _args = world.gpu.device.createBuffer({
            label: `sear-${label}-shadow-args`,
            size: _argsCap * SHADOW_ARG_STRIDE,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        _aGroups.length = 0;
        _bGroups.length = 0;
    }

    // (re)allocate the meta buffer to hold `combos + draws` u32 (the combo slots then the draw pairs)
    function ensureMeta(world: World, n: number, runIndex: number): GPUBuffer {
        if (_meta[runIndex] && _metaCap[runIndex] >= n) return _meta[runIndex];
        _meta[runIndex]?.destroy();
        const cap = Math.max(n, 64);
        const buffer = world.gpu.device.createBuffer({
            label: `sear-${label}-regather-meta-${runIndex}`,
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
            label: `sear-${label}-regather-params-${runIndex}`,
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        _params[runIndex] = buffer;
        return buffer;
    }

    // Pass A bind group (drawArgs + meta → args). `drawArgs` is the MeshInstance pack's shared indirect buffer (read
    // from a casting Draw — sear stays part-agnostic), which reallocs on pack growth
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
            label: `sear-${label}-regather-a`,
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

    // Pass B bind group (drawArgs + packedEids + args + meta → eids)
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
            label: `sear-${label}-regather-b`,
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
            if (_eids && _eidCapacity === capacity && _eidCombos >= maxCombos) return;
            _eids?.destroy();
            _eidCapacity = capacity;
            _eidCombos = maxCombos;
            _eids = world.gpu.device.createBuffer({
                label: `sear-${label}-regather-eids`,
                size: maxCombos * capacity * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
                    `sear ${label} re-gather run has ${D} draws after a ${_argsCap}-draw reserve`,
                );
            }
            const meta = ensureMeta(world, C + D, runIndex);
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
            _paramsStaging[0] = D;
            _paramsStaging[1] = C;
            _paramsStaging[2] = pairCount;
            world.gpu.device.queue.writeBuffer(
                params(world, runIndex),
                0,
                _paramsStaging as Uint32Array<ArrayBuffer>,
            );
            // Pass A (per-mesh args, 1 thread) → Pass B (scatter, one thread per (mesh, combo)) in one pass —
            // the same intra-pass dispatch-ordering the MeshInstance pack relies on, so B sees A's args writes
            //. The atlas render then sees the compute output by in-encoder ordering
            cpass.setPipeline(regatherState(world).aPipe!);
            cpass.setBindGroup(0, aGroup(world, drawArgs, meta, runIndex));
            cpass.dispatchWorkgroups(1);
            cpass.setPipeline(regatherState(world).bPipe!);
            cpass.setBindGroup(0, bGroup(world, drawArgs, packedEids, meta, runIndex));
            cpass.dispatchWorkgroups(Math.ceil((D * C) / 64));
        },
        reset(onAlloc: () => void): void {
            _onAlloc = onAlloc;
            // the packed list + args + meta allocate lazily on the first casting frame; drop any a prior
            // World left behind so a fresh World rebuilds its own
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
            for (const buffer of _params) buffer.destroy();
            _params = [];
            _aGroups = [];
            _bGroups = [];
        },
        dispose(): void {
            _eids?.destroy();
            _args?.destroy();
            for (const buffer of _meta) buffer.destroy();
            for (const buffer of _params) buffer.destroy();
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
