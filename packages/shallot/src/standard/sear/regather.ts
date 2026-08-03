// The shadow-atlas re-gather: concatenate the per-combo *culled* regions the Part pack wrote (slot-major
// `drawArgs` + the `packedEids` pool), or duplicate a view-independent producer's direct range, into one
// contiguous, mesh-major run per casting mesh + a per-instance combo index. Each shadow atlas (the
// point/spot tiles, the CSM cascade tiles) instantiates its own `Regather`; the two A/B compute pipelines
// are geometry-blind (they read slot-major counts + the eid pool alone, with no projection or mesh
// knowledge), so they're module-scope singletons shared across every instance — one shader module, two
// buffer sets. The re-gather is a *consumer* of the cull spine's output (`render/core` owns the spine that
// feeds it); it knows sear-private concepts (the packing convention below, the atlas record shape, the
// `eids`-lane swap), so it lives here, not in render/core (the agnosticism inversion render.md forbids).

import tgpu from "typegpu";
import { Compute, capacity } from "../../engine";
import { DrawIndexedIndirect } from "../render/core";

// the re-gather packs each instance's (eid, dense combo index) into one u32 in the re-gathered list — eid in
// the low COMBO_SHIFT bits, the combo above. The list rides the surface's `eids` binding lane (the heaviest
// surfaces sit at the 10-storage ceiling — gpu.md — so the combo can't get its own binding). COMBO_SHIFT
// holds the whole eid range (`capacity`), leaving 32 − COMBO_SHIFT bits for the combo (≫ the combo caps).
// The packer (Pass B below) and the atlas VS that unpacks it (sear's point/cascade pipelines) share these.
export const COMBO_SHIFT = Math.ceil(Math.log2(capacity));
export const EID_MASK = (1 << COMBO_SHIFT) - 1;

// one 20-byte DrawIndexedIndirect record per casting mesh, written by Pass A: instanceCount = Σ combo
// survivors, firstInstance = the mesh's base into the re-gathered list
export const SHADOW_ARG_STRIDE = 20;

// the two A/B compute pipelines — module-scope singletons, compiled once by prepareRegather. The WGSL is
// geometry-blind (slot-major counts + the eid pool + meta), so both the point atlas and the cascade atlas
// share them; only the bound buffers differ per Regather instance.
let _aPipe: GPUComputePipeline | null = null;
let _bPipe: GPUComputePipeline | null = null;
let _aLayout: GPUBindGroupLayout | null = null;
let _bLayout: GPUBindGroupLayout | null = null;

let _aWgsl: string | null = null;
let _bWgsl: string | null = null;

/** Pass A's exact compiled source, exposed lazily for the device-free indirect-record contract test. */
export const regatherArgsWgsl = (): string =>
    (_aWgsl ??= tgpu.resolve({
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
    }));

const regatherEidsWgsl = (): string =>
    (_bWgsl ??= tgpu.resolve({
        names: "strict",
        externals: { DrawIndexedIndirect },
        template: /* wgsl */ `
struct RgParams { draws: u32, combos: u32, pairCount: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: array<DrawIndexedIndirect>;
@group(0) @binding(1) var<storage, read> packedEids: array<u32>;
@group(0) @binding(2) var<storage, read> shadowArgs: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<storage, read> rgMeta: array<u32>;
@group(0) @binding(4) var<storage, read_write> shadowEids: array<u32>;
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
        shadowEids[dst + k] = packedEids[src + k] | (c << ${COMBO_SHIFT}u);
    }
}`,
    }));

/** compile the shared A/B re-gather pipelines once (idempotent): called from `prepareSear`, folded into its
 * warm `Promise.all`. Every {@link Regather} instance binds against these singleton layouts. */
export async function prepareRegather(device: GPUDevice): Promise<void> {
    if (_aPipe) return;
    // Pass A — one thread: for each casting mesh, sum its per-combo culled counts (the spine's drawArgs at
    // each combo slot), exclusive-prefix the totals into per-mesh run bases, and write one DrawIndexedIndirect
    // record (instanceCount = the sum, firstInstance = the base; the static indexCount/firstIndex from any
    // combo slot, which the pack seeds per slot). D + C are tiny, so a serial single thread is free
    _aLayout = device.createBindGroupLayout({
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
    // counts), packing the dense combo index above the eid. The serial inner copy is the per-(mesh, combo)
    // count; a per-instance dispatch is the deferred optimization (gpu.md rule 8) if a mesh ever owns a large
    // per-combo count
    _bLayout = device.createBindGroupLayout({
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
            layout: device.createPipelineLayout({ bindGroupLayouts: [_aLayout] }),
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
            layout: device.createPipelineLayout({ bindGroupLayouts: [_bLayout] }),
            compute: {
                module: device.createShaderModule({ label: "sear-regather-b", code: bWgsl }),
                entryPoint: "main",
            },
        }),
    ]);
    _aPipe = a;
    _bPipe = b;
}

/** one shadow atlas's re-gather instance: its own packed list + indirect args + meta, sharing the
 * module-singleton A/B pipelines. The point atlas and the CSM cascade atlas each own one. */
export interface Regather {
    /** the re-gathered packed instance list (`(combo << COMBO_SHIFT) | eid`), bound at the consumer
     * pipeline's `eids` lane. `null` until {@link Regather.ensure} allocates it (the first casting frame). */
    eids(): GPUBuffer | null;
    /** the indirect buffer the atlas render pass draws from: one DrawIndexedIndirect record per casting
     * mesh (Pass A fills it). `null` until {@link Regather.reserve} allocates it. */
    args(): GPUBuffer | null;
    /** lazily allocate the packed list (sized `maxCombos × capacity`, the provably-safe bound: each combo
     * view slot holds ≤ capacity culled eids). Fires the `onAlloc` callback registered via
     * {@link Regather.reset} (sear rebuilds the bind groups that bind this lane). Idempotent once allocated. */
    ensure(maxCombos: number): void;
    /** preflight the largest batch before recording any run into an encoder. A run never reallocates this
     * shared output: earlier GPU commands in the same unsubmitted encoder must keep the buffer they captured. */
    reserve(maxDraws: number): void;
    /** upload the per-frame meta + run Pass A then Pass B on `cpass` (one compute pass, the intra-pass
     * dispatch ordering the Part pack relies on). `comboSlots` = the view slot each dense combo packed into;
     * `drawPairs` = the source indirect-record indices; `pairCount` = the pack's pair stride, or zero for a
     * view-independent producer whose direct range is duplicated across combos. */
    run(
        cpass: GPUComputePassEncoder,
        drawArgs: GPUBuffer,
        packedEids: GPUBuffer,
        comboSlots: number[],
        drawPairs: number[],
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
 * compiled once via {@link prepareRegather} before {@link Regather.run}. */
export function createRegather(label: string): Regather {
    let _eids: GPUBuffer | null = null;
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
    function ensureArgs(count: number): void {
        if (_args && _argsCap >= count) return;
        _args?.destroy();
        _argsCap = Math.max(count, 8);
        _args = Compute.device.createBuffer({
            label: `sear-${label}-shadow-args`,
            size: _argsCap * SHADOW_ARG_STRIDE,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        _aGroups.length = 0;
        _bGroups.length = 0;
    }

    // (re)allocate the meta buffer to hold `combos + draws` u32 (the combo slots then the draw pairs)
    function ensureMeta(n: number, runIndex: number): GPUBuffer {
        if (_meta[runIndex] && _metaCap[runIndex] >= n) return _meta[runIndex];
        _meta[runIndex]?.destroy();
        const cap = Math.max(n, 64);
        const buffer = Compute.device.createBuffer({
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

    function params(runIndex: number): GPUBuffer {
        let buffer = _params[runIndex];
        if (buffer) return buffer;
        buffer = Compute.device.createBuffer({
            label: `sear-${label}-regather-params-${runIndex}`,
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        _params[runIndex] = buffer;
        return buffer;
    }

    // Pass A bind group (drawArgs + meta → args). `drawArgs` is the Part pack's shared indirect buffer (read
    // from a casting Draw — sear stays part-agnostic), which reallocs on pack growth
    function aGroup(drawArgs: GPUBuffer, meta: GPUBuffer, runIndex: number): GPUBindGroup {
        const cached = _aGroups[runIndex];
        if (
            cached &&
            cached.args === _args &&
            cached.meta === meta &&
            cached.drawArgs === drawArgs
        ) {
            return cached.group;
        }
        const group = Compute.device.createBindGroup({
            label: `sear-${label}-regather-a`,
            layout: _aLayout!,
            entries: [
                { binding: 0, resource: { buffer: drawArgs } },
                { binding: 1, resource: { buffer: meta } },
                { binding: 2, resource: { buffer: _args! } },
                { binding: 3, resource: { buffer: params(runIndex) } },
            ],
        });
        _aGroups[runIndex] = { args: _args!, meta, drawArgs, group };
        return group;
    }

    // Pass B bind group (drawArgs + packedEids + args + meta → eids)
    function bGroup(
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
        const group = Compute.device.createBindGroup({
            label: `sear-${label}-regather-b`,
            layout: _bLayout!,
            entries: [
                { binding: 0, resource: { buffer: drawArgs } },
                { binding: 1, resource: { buffer: packed } },
                { binding: 2, resource: { buffer: _args! } },
                { binding: 3, resource: { buffer: meta } },
                { binding: 4, resource: { buffer: _eids! } },
                { binding: 5, resource: { buffer: params(runIndex) } },
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
        ensure(maxCombos: number): void {
            if (_eids) return;
            _eids = Compute.device.createBuffer({
                label: `sear-${label}-regather-eids`,
                size: maxCombos * capacity * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            _onAlloc();
        },
        reserve(maxDraws: number): void {
            ensureArgs(maxDraws);
        },
        run(cpass, drawArgs, packedEids, comboSlots, drawPairs, pairCount, runIndex = 0): void {
            const C = comboSlots.length;
            const D = drawPairs.length;
            if (!_args || _argsCap < D) {
                throw new Error(
                    `sear ${label} re-gather run has ${D} draws after a ${_argsCap}-draw reserve`,
                );
            }
            const meta = ensureMeta(C + D, runIndex);
            const staging = _metaStaging[runIndex];
            // meta = [combo slots (C) | draw pairs (D)]: the view slot each dense combo packed into (its
            // per-combo culled counts live in drawArgs there), and the (surface,mesh) pair each casting draw owns
            for (let c = 0; c < C; c++) staging[c] = comboSlots[c];
            for (let i = 0; i < D; i++) staging[C + i] = drawPairs[i];
            Compute.device.queue.writeBuffer(
                meta,
                0,
                staging as Uint32Array<ArrayBuffer>,
                0,
                C + D,
            );
            _paramsStaging[0] = D;
            _paramsStaging[1] = C;
            _paramsStaging[2] = pairCount;
            Compute.device.queue.writeBuffer(
                params(runIndex),
                0,
                _paramsStaging as Uint32Array<ArrayBuffer>,
            );
            // Pass A (per-mesh args, 1 thread) → Pass B (scatter, one thread per (mesh, combo)) in one pass —
            // the same intra-pass dispatch-ordering the Part pack relies on, so B sees A's args writes
            // (gpu.md "Cross a dispatch boundary"). The atlas render then sees the compute output by in-encoder ordering
            cpass.setPipeline(_aPipe!);
            cpass.setBindGroup(0, aGroup(drawArgs, meta, runIndex));
            cpass.dispatchWorkgroups(1);
            cpass.setPipeline(_bPipe!);
            cpass.setBindGroup(0, bGroup(drawArgs, packedEids, meta, runIndex));
            cpass.dispatchWorkgroups(Math.ceil((D * C) / 64));
        },
        reset(onAlloc: () => void): void {
            _onAlloc = onAlloc;
            // the packed list + args + meta allocate lazily on the first casting frame; drop any a prior
            // State left behind so a fresh State rebuilds its own
            _eids?.destroy();
            _eids = null;
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
