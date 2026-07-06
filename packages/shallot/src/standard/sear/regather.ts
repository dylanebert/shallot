// The shadow-atlas re-gather: concatenate the per-combo *culled* regions the Part pack wrote (slot-major
// `drawArgs` + the `packedEids` pool) into one contiguous, mesh-major run per casting mesh + a per-instance
// combo index, so the atlas renders in **one indirect draw per casting mesh** — the property the deleted
// amplify trick bought, now reading per-combo culled counts (no over-amplification). Each shadow atlas (the
// point/spot tiles, the CSM cascade tiles) instantiates its own `Regather`; the two A/B compute pipelines
// are geometry-blind (they read slot-major counts + the eid pool alone, with no projection or mesh
// knowledge), so they're module-scope singletons shared across every instance — one shader module, two
// buffer sets. The re-gather is a *consumer* of the cull spine's output (`render/core` owns the spine that
// feeds it); it knows sear-private concepts (the packing convention below, the atlas record shape, the
// `eids`-lane swap), so it lives here, not in render/core (the agnosticism inversion render.md forbids).

import { Compute, capacity } from "../../engine";

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

/** compile the shared A/B re-gather pipelines once (idempotent) — called from `prepareSear`, folded into its
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
    const aWgsl = /* wgsl */ `
struct RgParams { draws: u32, combos: u32, pairCount: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: array<u32>;
@group(0) @binding(1) var<storage, read> rgMeta: array<u32>;       // [combo slots (C) | draw pairs (D)]
@group(0) @binding(2) var<storage, read_write> shadowArgs: array<u32>;
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
        for (var c = 0u; c < C; c = c + 1u) {
            total = total + drawArgs[(rgMeta[c] * pc + pair) * 5u + 1u];
        }
        let rec = i * 5u;
        shadowArgs[rec + 0u] = drawArgs[(slot0 * pc + pair) * 5u + 0u]; // indexCount (static)
        shadowArgs[rec + 1u] = total;                                   // instanceCount
        shadowArgs[rec + 2u] = drawArgs[(slot0 * pc + pair) * 5u + 2u]; // firstIndex (static)
        shadowArgs[rec + 3u] = 0u;                                      // baseVertex
        shadowArgs[rec + 4u] = base;                                    // firstInstance (the mesh's run base)
        base = base + total;
    }
}`;

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
    const bWgsl = /* wgsl */ `
struct RgParams { draws: u32, combos: u32, pairCount: u32 }
@group(0) @binding(0) var<storage, read> drawArgs: array<u32>;
@group(0) @binding(1) var<storage, read> packedEids: array<u32>;
@group(0) @binding(2) var<storage, read> shadowArgs: array<u32>;
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
    let idx = (rgMeta[c] * pc + pair) * 5u;
    let cnt = drawArgs[idx + 1u];
    if (cnt == 0u) { return; }
    let src = drawArgs[idx + 4u];               // base into packedEids for (combo c, mesh i)
    var off = 0u;                               // within-run offset: Σ earlier combos' counts for this mesh
    for (var cc = 0u; cc < c; cc = cc + 1u) {
        off = off + drawArgs[(rgMeta[cc] * pc + pair) * 5u + 1u];
    }
    let dst = shadowArgs[i * 5u + 4u] + off;    // the mesh's run base + the combo's within-run offset
    for (var k = 0u; k < cnt; k = k + 1u) {
        shadowEids[dst + k] = packedEids[src + k] | (c << ${COMBO_SHIFT}u);
    }
}`;

    const [a, b] = await Promise.all([
        device.createComputePipelineAsync({
            label: "sear-regather-a",
            layout: device.createPipelineLayout({ bindGroupLayouts: [_aLayout] }),
            compute: {
                module: device.createShaderModule({ label: "sear-regather-a", code: aWgsl }),
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

/** one shadow atlas's re-gather instance — its own packed list + indirect args + meta, sharing the
 * module-singleton A/B pipelines. The point atlas and the CSM cascade atlas each own one. */
export interface Regather {
    /** the re-gathered packed instance list (`(combo << COMBO_SHIFT) | eid`), bound at the consumer
     * pipeline's `eids` lane. `null` until {@link Regather.ensure} allocates it (the first casting frame). */
    eids(): GPUBuffer | null;
    /** the indirect buffer the atlas render pass draws from — one DrawIndexedIndirect record per casting
     * mesh (Pass A fills it). `null` until {@link Regather.run} allocates it. */
    args(): GPUBuffer | null;
    /** lazily allocate the packed list (sized `maxCombos × capacity`, the provably-safe bound: each combo
     * view slot holds ≤ capacity culled eids). Fires the `onAlloc` callback registered via
     * {@link Regather.reset} (sear rebuilds the bind groups that bind this lane). Idempotent once allocated. */
    ensure(maxCombos: number): void;
    /** upload the per-frame meta + run Pass A then Pass B on `cpass` (one compute pass, the intra-pass
     * dispatch ordering the Part pack relies on). `comboSlots` = the view slot each dense combo packed into;
     * `drawPairs` = the (surface,mesh) pair each casting draw owns; `pairCount` = the pack's pair stride. */
    run(
        cpass: GPUComputePassEncoder,
        drawArgs: GPUBuffer,
        packedEids: GPUBuffer,
        comboSlots: number[],
        drawPairs: number[],
        pairCount: number,
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
    let _meta: GPUBuffer | null = null;
    let _metaCap = 0;
    let _metaStaging = new Uint32Array(0);
    let _params: GPUBuffer | null = null;
    const _paramsStaging = new Uint32Array(4);
    let _aGroup: {
        args: GPUBuffer;
        meta: GPUBuffer;
        drawArgs: GPUBuffer;
        group: GPUBindGroup;
    } | null = null;
    let _bGroup: {
        args: GPUBuffer;
        meta: GPUBuffer;
        eids: GPUBuffer;
        drawArgs: GPUBuffer;
        packed: GPUBuffer;
        group: GPUBindGroup;
    } | null = null;
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
        _aGroup = null;
        _bGroup = null;
    }

    // (re)allocate the meta buffer to hold `combos + draws` u32 (the combo slots then the draw pairs)
    function ensureMeta(n: number): void {
        if (_meta && _metaCap >= n) return;
        _meta?.destroy();
        _metaCap = Math.max(n, 64);
        _meta = Compute.device.createBuffer({
            label: `sear-${label}-regather-meta`,
            size: _metaCap * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        _metaStaging = new Uint32Array(_metaCap);
        _aGroup = null;
        _bGroup = null;
    }

    // Pass A bind group (drawArgs + meta → args). `drawArgs` is the Part pack's shared indirect buffer (read
    // from a casting Draw — sear stays part-agnostic), which reallocs on pack growth
    function aGroup(drawArgs: GPUBuffer): GPUBindGroup {
        if (
            _aGroup &&
            _aGroup.args === _args &&
            _aGroup.meta === _meta &&
            _aGroup.drawArgs === drawArgs
        ) {
            return _aGroup.group;
        }
        const group = Compute.device.createBindGroup({
            label: `sear-${label}-regather-a`,
            layout: _aLayout!,
            entries: [
                { binding: 0, resource: { buffer: drawArgs } },
                { binding: 1, resource: { buffer: _meta! } },
                { binding: 2, resource: { buffer: _args! } },
                { binding: 3, resource: { buffer: _params! } },
            ],
        });
        _aGroup = { args: _args!, meta: _meta!, drawArgs, group };
        return group;
    }

    // Pass B bind group (drawArgs + packedEids + args + meta → eids)
    function bGroup(drawArgs: GPUBuffer, packed: GPUBuffer): GPUBindGroup {
        if (
            _bGroup &&
            _bGroup.args === _args &&
            _bGroup.meta === _meta &&
            _bGroup.eids === _eids &&
            _bGroup.drawArgs === drawArgs &&
            _bGroup.packed === packed
        ) {
            return _bGroup.group;
        }
        const group = Compute.device.createBindGroup({
            label: `sear-${label}-regather-b`,
            layout: _bLayout!,
            entries: [
                { binding: 0, resource: { buffer: drawArgs } },
                { binding: 1, resource: { buffer: packed } },
                { binding: 2, resource: { buffer: _args! } },
                { binding: 3, resource: { buffer: _meta! } },
                { binding: 4, resource: { buffer: _eids! } },
                { binding: 5, resource: { buffer: _params! } },
            ],
        });
        _bGroup = { args: _args!, meta: _meta!, eids: _eids!, drawArgs, packed, group };
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
        run(cpass, drawArgs, packedEids, comboSlots, drawPairs, pairCount): void {
            const C = comboSlots.length;
            const D = drawPairs.length;
            ensureArgs(D);
            ensureMeta(C + D);
            // meta = [combo slots (C) | draw pairs (D)]: the view slot each dense combo packed into (its
            // per-combo culled counts live in drawArgs there), and the (surface,mesh) pair each casting draw owns
            for (let c = 0; c < C; c++) _metaStaging[c] = comboSlots[c];
            for (let i = 0; i < D; i++) _metaStaging[C + i] = drawPairs[i];
            Compute.device.queue.writeBuffer(
                _meta!,
                0,
                _metaStaging as Uint32Array<ArrayBuffer>,
                0,
                C + D,
            );
            _paramsStaging[0] = D;
            _paramsStaging[1] = C;
            _paramsStaging[2] = pairCount;
            Compute.device.queue.writeBuffer(
                _params!,
                0,
                _paramsStaging as Uint32Array<ArrayBuffer>,
            );
            // Pass A (per-mesh args, 1 thread) → Pass B (scatter, one thread per (mesh, combo)) in one pass —
            // the same intra-pass dispatch-ordering the Part pack relies on, so B sees A's args writes
            // (gpu.md "Cross a dispatch boundary"). The atlas render then sees the compute output by in-encoder ordering
            cpass.setPipeline(_aPipe!);
            cpass.setBindGroup(0, aGroup(drawArgs));
            cpass.dispatchWorkgroups(1);
            cpass.setPipeline(_bPipe!);
            cpass.setBindGroup(0, bGroup(drawArgs, packedEids));
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
            _meta?.destroy();
            _meta = null;
            _metaCap = 0;
            _metaStaging = new Uint32Array(0);
            _params?.destroy();
            _params = Compute.device.createBuffer({
                label: `sear-${label}-regather-params`,
                size: 16, // { draws, combos, pairCount, _pad }
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            _aGroup = null;
            _bGroup = null;
        },
        dispose(): void {
            _eids?.destroy();
            _args?.destroy();
            _meta?.destroy();
            _params?.destroy();
            _eids = null;
            _args = null;
            _argsCap = 0;
            _meta = null;
            _metaCap = 0;
            _metaStaging = new Uint32Array(0);
            _params = null;
            _aGroup = null;
            _bGroup = null;
        },
    };
}
