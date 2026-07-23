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
let phasesRun = 0; // the loop count the per-frame pass actually encoded (the assert divides by it)
let params: Params | null = null;

const KERNEL_WGSL = /* wgsl */ `
struct ChainParams { phases: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> cp: ChainParams;

// one phase: a dependent read-modify-write of one storage word. Lane 0 only — the chain measures the
// phase boundary, not parallel work.
@compute @workgroup_size(64)
fn phase(@builtin(local_invocation_index) lid: u32) {
    if (lid == 0u) { data[0] = data[0] + 1u; }
}

// the same chain as in-kernel barriers: a dynamic loop bound (stays rolled — DXC unrolls only constant
// bounds), storageBarrier() making each iteration's write visible to the next.
@compute @workgroup_size(64)
fn barrierChain(@builtin(local_invocation_index) lid: u32) {
    for (var i = 0u; i < cp.phases; i = i + 1u) {
        if (lid == 0u) { data[0] = data[0] + 1u; }
        storageBarrier();
    }
}
`;

function chainPlugin(): Plugin {
    let dispatchPipe: GPUComputePipeline | null = null;
    let barrierPipe: GPUComputePipeline | null = null;
    let bg: GPUBindGroup | null = null;
    let dataBuf: GPUBuffer | null = null;
    let ubo: GPUBuffer | null = null;
    let uploaded = -1;

    const pass: System = {
        group: "draw",
        annotations: { mode: "always" },
        after: [BeginFrameSystem],
        update() {
            if (!dispatchPipe || !barrierPipe || !bg || !ubo || !Render.encoder) return;
            const n = Math.max(1, Math.round((params?.phases as number) ?? DEFAULT_PHASES));
            if (n !== uploaded) {
                Compute.device.queue.writeBuffer(ubo, 0, new Uint32Array([n]));
                uploaded = n;
            }
            phasesRun = n;
            {
                const p = Render.encoder.beginComputePass({
                    timestampWrites: Compute.span?.("chain:dispatch"),
                });
                p.setPipeline(dispatchPipe);
                p.setBindGroup(0, bg);
                for (let i = 0; i < n; i++) p.dispatchWorkgroups(1);
                p.end();
            }
            {
                const p = Render.encoder.beginComputePass({
                    timestampWrites: Compute.span?.("chain:barrier"),
                });
                p.setPipeline(barrierPipe);
                p.setBindGroup(0, bg);
                p.dispatchWorkgroups(1);
                p.end();
            }
        },
    };

    return {
        name: "GymChain",
        systems: [pass],
        dependencies: [RenderPlugin],
        async warm() {
            const device = Compute.device;
            dataBuf = device.createBuffer({
                label: "gym-chain-data",
                size: 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            ubo = device.createBuffer({
                label: "gym-chain-params",
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const module = device.createShaderModule({ label: "gym-chain", code: KERNEL_WGSL });
            // explicit shared layout — the dispatch kernel doesn't read the uniform, so an "auto"
            // layout would omit binding 1 and the shared bind group would fail validation.
            const layout = device.createBindGroupLayout({
                label: "gym-chain",
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "storage" },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: { type: "uniform" },
                    },
                ],
            });
            const pipelineLayout = device.createPipelineLayout({
                label: "gym-chain",
                bindGroupLayouts: [layout],
            });
            [dispatchPipe, barrierPipe] = await Promise.all([
                device.createComputePipelineAsync({
                    label: "gym-chain-dispatch",
                    layout: pipelineLayout,
                    compute: { module, entryPoint: "phase" },
                }),
                device.createComputePipelineAsync({
                    label: "gym-chain-barrier",
                    layout: pipelineLayout,
                    compute: { module, entryPoint: "barrierChain" },
                }),
            ]);
            bg = device.createBindGroup({
                label: "gym-chain",
                layout,
                entries: [
                    { binding: 0, resource: { buffer: dataBuf } },
                    { binding: 1, resource: { buffer: ubo } },
                ],
            });
            dataMirror = mirror(dataBuf);
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
            const words = dataMirror.snapshot && new Uint32Array(dataMirror.snapshot.bytes);
            const count = words?.[0] ?? 0;
            checks.push({
                name: "chain advanced",
                pass: count > 0,
                detail: `data[0] = ${count} after both chains`,
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
        checks.push({
            name: "measured (chain spans)",
            pass: true, // a reporter — the bench gates on the payload
            detail:
                dispatch > 0
                    ? `${n} phases: dispatch ${((dispatch * 1000) / n).toFixed(2)} µs/phase, ` +
                      `barrier ${((barrier * 1000) / n).toFixed(2)} µs/phase`
                    : "no chain spans resolved",
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
