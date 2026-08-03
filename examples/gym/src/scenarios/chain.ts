import {
    Camera,
    CameraMode,
    Compute,
    InputPlugin,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    type Plugin,
    precompile,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    type System,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { Profile, ProfilePlugin } from "@dylanebert/shallot/extras";
import { BeginFrameSystem, Render } from "@dylanebert/shallot/render/core";
import type {
    StorageFlag,
    TgpuBindGroup,
    TgpuBuffer,
    TgpuComputePipeline,
    UniformFlag,
} from "typegpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";

// chain — the per-phase boundary-constant microbench the physics waste audit keys on (roadmap "Physics —
// the structure tax", Phase A). A "phase" is one serially-dependent GPU step: a read-modify-write of one
// storage word that the next phase must see. The scenario runs the same N-phase chain two ways each frame:
//
//   • chain:dispatch — one compute pass, N dispatches of a minimal kernel (`data[0] += 1`). The write→read
//     hazard on `data` orders them, so span/N is the cost of a DISPATCH-BOUNDARY phase: launch + drain +
//     the dependent storage round trip.
//   • chain:barrier — ONE single-workgroup dispatch looping the same N phases with `storageBarrier()`
//     between. span/N is the cost of an IN-KERNEL-BARRIER phase: the same dependent round trip, no launch.
//
// The two constants anchor the structure-tax model: dispatch ≈ barrier confirms the megakernel refutation
// (a phase's cost is its dependent memory round trip, not removable launch overhead — physics.md "Dispatch
// count"); their gap is the true launch overhead a phase-deleting optimization could recover. The bench
// (scripts/physics-bench.ts --audit) reads both off the `measured` payload, min-over-reps.

const DEFAULT_PHASES = 70; // ≈ the physics step's phase count at gameplay scale (sparse colors)

let dataMirror: Mirror | null = null;
let expectedFrames: ReadonlyMap<number, number> | null = null;
let phasesRun = 0; // the loop count the per-frame pass actually encoded (the assert divides by it)
let params: Params | null = null;

const ChainParams = d
    .struct({ phases: d.u32, pad0: d.u32, pad1: d.u32, pad2: d.u32 })
    .$name("ChainParams");
const ChainData = d.arrayOf(d.u32, 4);
const chainLayout = tgpu.bindGroupLayout({
    data: { storage: ChainData, access: "mutable" },
    cp: { uniform: ChainParams },
});

const phaseKernel = tgpu
    .computeFn({
        in: { lid: d.builtin.localInvocationIndex },
        workgroupSize: [64],
    })((input) => {
        "use gpu";
        if (input.lid === d.u32(0)) {
            chainLayout.$.data[d.u32(0)] = chainLayout.$.data[d.u32(0)] + d.u32(1);
        }
    })
    .$name("phase");

const barrierKernel = tgpu
    .computeFn({
        in: { lid: d.builtin.localInvocationIndex },
        workgroupSize: [64],
    })((input) => {
        "use gpu";
        for (let i = d.u32(0); i < chainLayout.$.cp.phases; i = i + d.u32(1)) {
            if (input.lid === d.u32(0)) {
                chainLayout.$.data[d.u32(0)] = chainLayout.$.data[d.u32(0)] + d.u32(1);
            }
            std.storageBarrier();
        }
    })
    .$name("barrierChain");

/** the emitted chain WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function chainWgsl(): string {
    return tgpu.resolve([phaseKernel, barrierKernel], { names: "strict" });
}

type ChainDataBuf = TgpuBuffer<typeof ChainData> & StorageFlag;
type ChainParamsBuf = TgpuBuffer<typeof ChainParams> & UniformFlag;
type ChainResources = {
    readonly dataRaw?: GPUBuffer;
    readonly params?: ChainParamsBuf;
    readonly mirror?: Mirror;
};
type ChainRun = {
    uploaded: number;
    expected: number;
    readonly frames: Map<number, number>;
};
type ChainOwner = ChainResources & {
    readonly dataRaw: GPUBuffer;
    readonly data: ChainDataBuf;
    readonly params: ChainParamsBuf;
    readonly mirror: Mirror;
    readonly dispatch: TgpuComputePipeline;
    readonly barrier: TgpuComputePipeline;
    readonly group: TgpuBindGroup<typeof chainLayout.entries>;
    readonly run: ChainRun;
};

const cleanedChainResources = new WeakSet<object>();

function cleanupChain(resources: ChainResources): void {
    if (cleanedChainResources.has(resources)) return;
    cleanedChainResources.add(resources);
    resources.mirror?.dispose();
    resources.dataRaw?.destroy();
    resources.params?.destroy();
    if (dataMirror === resources.mirror) dataMirror = null;
}

function forceChain(pipeline: TgpuComputePipeline, label: string): TgpuComputePipeline {
    const { device, root } = Compute;
    const owned: { dataRaw?: GPUBuffer; params?: ChainParamsBuf } = {};
    let submitted = false;
    const cleanup = () => cleanupChain(owned);
    try {
        owned.dataRaw = device.createBuffer({
            label: `${label}-data`,
            size: d.sizeOf(ChainData),
            usage: GPUBufferUsage.STORAGE,
        });
        const data = root.createBuffer(ChainData, owned.dataRaw).$usage("storage");
        owned.params = root.createBuffer(ChainParams).$usage("uniform");
        const group = root.createBindGroup(chainLayout, { data, cp: owned.params });
        const encoder = device.createCommandEncoder({ label });
        const pass = encoder.beginComputePass({ label });
        pipeline.with(group).with(pass).dispatchWorkgroups(0, 1, 1);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const cleanupAfterFence = device.queue.onSubmittedWorkDone().then(cleanup, cleanup);
        submitted = true;
        void cleanupAfterFence;
        return pipeline;
    } finally {
        if (!submitted) cleanup();
    }
}

/** invoke the two benchmark arms: N separate phase dispatches, then one barrier-chain dispatch.
 *  @internal */
export function runChainArms(phases: number, dispatch: () => void, barrier: () => void): void {
    for (let i = 0; i < phases; i++) dispatch();
    barrier();
}

function chainPlugin(): Plugin {
    let active: ChainOwner | null = null;

    const pass: System = {
        group: "draw",
        annotations: { mode: "always" },
        after: [BeginFrameSystem],
        update() {
            const owner = active;
            if (!owner || !Render.encoder) return;
            const n = Math.max(1, Math.round((params?.phases as number) ?? DEFAULT_PHASES));
            if (n !== owner.run.uploaded) {
                owner.params.write({ phases: n, pad0: 0, pad1: 0, pad2: 0 });
                owner.run.uploaded = n;
            }
            phasesRun = n;
            owner.run.frames.set(Compute.frame, owner.run.expected);
            owner.run.frames.delete(Compute.frame - 16);
            const dispatchPass = Render.encoder.beginComputePass({
                timestampWrites: Compute.span?.("chain:dispatch"),
            });
            const dispatch = owner.dispatch.with(owner.group).with(dispatchPass);
            runChainArms(
                n,
                () => dispatch.dispatchWorkgroups(1, 1, 1),
                () => {
                    dispatchPass.end();
                    const barrierPass = Render.encoder!.beginComputePass({
                        timestampWrites: Compute.span?.("chain:barrier"),
                    });
                    owner.barrier.with(owner.group).with(barrierPass).dispatchWorkgroups(1, 1, 1);
                    barrierPass.end();
                },
            );
            owner.run.expected = (owner.run.expected + n * 2) >>> 0;
        },
    };

    return {
        name: "GymChain",
        systems: [pass],
        dependencies: [RenderPlugin],
        async warm(state) {
            if (state.disposed) return;
            const prior = active;
            if (prior) {
                cleanupChain(prior);
                if (active === prior) active = null;
                if (expectedFrames === prior.run.frames) expectedFrames = null;
            }
            const device = Compute.device;
            const root = Compute.root;
            const draft: { dataRaw?: GPUBuffer; params?: ChainParamsBuf; mirror?: Mirror } = {};
            let owner: ChainOwner | undefined;
            try {
                draft.dataRaw = device.createBuffer({
                    label: "gym-chain-data",
                    size: d.sizeOf(ChainData),
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                });
                const data = root.createBuffer(ChainData, draft.dataRaw).$usage("storage");
                draft.params = root
                    .createBuffer(ChainParams)
                    .$usage("uniform")
                    .$name("gym-chain-params");
                const dispatch = root
                    .createComputePipeline({ compute: phaseKernel })
                    .$name("gym-chain-dispatch");
                const barrier = root
                    .createComputePipeline({ compute: barrierKernel })
                    .$name("gym-chain-barrier");
                const group = root.createBindGroup(chainLayout, { data, cp: draft.params });
                draft.mirror = mirror(draft.dataRaw);
                const run = { uploaded: -1, expected: 0, frames: new Map<number, number>() };
                owner = Object.freeze({
                    dataRaw: draft.dataRaw,
                    data,
                    params: draft.params,
                    mirror: draft.mirror,
                    dispatch,
                    barrier,
                    group,
                    run,
                });
                active = owner;
                dataMirror = owner.mirror;
                expectedFrames = owner.run.frames;
                state.onDispose(() => {
                    cleanupChain(owner!);
                    if (active === owner) active = null;
                    if (expectedFrames === owner!.run.frames) expectedFrames = null;
                });
                await precompile("gym-chain-dispatch", () =>
                    forceChain(owner!.dispatch, "gym-chain-dispatch"),
                );
                await precompile("gym-chain-barrier", () =>
                    forceChain(owner!.barrier, "gym-chain-barrier"),
                );
            } catch (cause) {
                cleanupChain(owner ?? draft);
                if (active === owner) active = null;
                if (owner && expectedFrames === owner.run.frames) expectedFrames = null;
                throw cause;
            }
        },
    };
}

const scenario: Scenario = {
    name: "chain",
    // a GPU-compute microbench: it runs two dependent compute chains and draws no framed scene, so the
    // canvas is legitimately blank. Opt out of the pixel gate — the chain's correctness is its assert.
    noRender: true,
    params: [
        {
            key: "phases",
            type: "number",
            default: DEFAULT_PHASES,
            min: 1,
            max: 1024,
            step: 1,
            label: "phases",
        },
    ],

    async build(_canvas, p: Params) {
        params = p;
        const { state, dispose } = await run({
            defaults: false,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                MirrorPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                SearPlugin,
                chainPlugin(),
            ],
        });

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);

        await frames(3);
        return { state, dispose };
    },

    // one correctness gate (the chains actually ran — the counter advanced) + the measured reporter the
    // bench reads: both spans + the per-phase constants in µs.
    async assert(): Promise<Check[]> {
        const checks: Check[] = [];
        if (dataMirror) {
            await settle(dataMirror);
            const snapshot = dataMirror.snapshot;
            const words = snapshot && new Uint32Array(snapshot.bytes);
            const count = words?.[0] ?? 0;
            const expected = snapshot ? expectedFrames?.get(snapshot.frame) : undefined;
            checks.push({
                name: "both chains exact recurrence",
                pass: expected !== undefined && count === expected,
                detail:
                    expected === undefined
                        ? `no host reference for Mirror frame ${snapshot?.frame ?? "none"}`
                        : `data[0] = ${count}, expected ${expected} at frame ${snapshot!.frame}`,
            });
        }
        const dispatch = Profile.gpu.get("chain:dispatch") ?? 0;
        const barrier = Profile.gpu.get("chain:barrier") ?? 0;
        const n = Math.max(1, phasesRun);
        const data: Record<string, number> = {
            "chain:dispatch": dispatch,
            "chain:barrier": barrier,
            phases: n,
            dispatchUs: (dispatch * 1000) / n,
            barrierUs: (barrier * 1000) / n,
        };
        const spansValid =
            Number.isFinite(dispatch) && dispatch > 0 && Number.isFinite(barrier) && barrier > 0;
        checks.push({
            name: "measured (chain spans)",
            pass: spansValid,
            detail: spansValid
                ? `${n} phases: dispatch ${((dispatch * 1000) / n).toFixed(2)} µs/phase, ` +
                  `barrier ${((barrier * 1000) / n).toFixed(2)} µs/phase`
                : `invalid chain spans: dispatch=${dispatch}, barrier=${barrier}`,
            data,
        });
        return checks;
    },

    live(): string {
        const d = Profile.gpu.get("chain:dispatch") ?? 0;
        const b = Profile.gpu.get("chain:barrier") ?? 0;
        const n = Math.max(1, phasesRun);
        return `chain ${n} phases · dispatch ${((d * 1000) / n).toFixed(2)} µs/phase · barrier ${((b * 1000) / n).toFixed(2)} µs/phase`;
    },
};

register(scenario);
