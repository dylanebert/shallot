import { Compute, type Plugin, precompile, type System } from "@dylanebert/shallot";
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

// The GPU-ALU burn knob: a fixed heavy compute kernel dispatched each frame on its own submit, looping
// `getLoad()` iterations per lane. The per-frame fence (Compute.sync) waits on it, so the run loop's dt
// grows and effective fps drops — the *induce* half of induce-and-measure, shared by the character
// latency probe (`scenarios/character.ts`, drop fps into the felt-lag regime) and the stress atom
// (`scenarios/stress.ts`, saturate the compute axis). `setLoad(0)` makes it a no-op.
//
// Unlike a one-off probe dispatch, this pass is a TIMED profiler span ("load"), so the stress atom can
// read its per-pass time/percentiles and prove the induced load lands on its own span and no other. The
// timestamp is free when ProfilePlugin is absent (`Compute.span?.()` → undefined → an untimed pass).

const LOAD_WG = 16384; // fixed workgroups (× 64 lanes ≈ 1M); the loop scales per-lane iterations
const LOAD_N = 4096; // scratch length; each lane writes buf[gid % N] so DXC can't drop the loop
// the loop bound is a DYNAMIC uniform (`cfg.iter`), not a literal — a literal bound risks a DXC unroll
// (gpu.md "DXC shader compilation"); dynamic stays rolled, and lets the load scale GPU time at runtime.
const LoadCfg = d
    .struct({ iter: d.u32, pad0: d.u32, pad1: d.u32, pad2: d.u32 })
    .$name("GymLoadCfg");
const LOAD_SCHEMA = d.arrayOf(d.f32, LOAD_N);
const loadLayout = tgpu.bindGroupLayout({
    buf: { storage: LOAD_SCHEMA, access: "mutable" },
    cfg: { uniform: LoadCfg },
});
const loadKernel = tgpu
    .computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [64],
    })((input) => {
        "use gpu";
        let x = d.f32(input.gid.x) * d.f32(0.0001) + d.f32(1.0);
        for (let i = d.u32(0); i < loadLayout.$.cfg.iter; i = i + d.u32(1)) {
            x = std.sin(x) * d.f32(1.0001) + std.cos(x * d.f32(1.3)) * d.f32(0.5) + d.f32(0.7);
        }
        loadLayout.$.buf[input.gid.x % d.u32(LOAD_N)] = x;
    })
    .$name("load");

/** the emitted load WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function loadWgsl(): string {
    return tgpu.resolve([loadKernel], { names: "strict" });
}

type LoadBuf = TgpuBuffer<typeof LOAD_SCHEMA> & StorageFlag;
type LoadCfgBuf = TgpuBuffer<typeof LoadCfg> & UniformFlag;

let pipeline: TgpuComputePipeline | null = null;
let bind: TgpuBindGroup<typeof loadLayout.entries> | null = null;
let buf: LoadBuf | null = null;
let cfgBuf: LoadCfgBuf | null = null;
let iters = 0;

/** set the per-lane iteration count — the GPU-time inflation level. ≤0 makes the pass a no-op. */
export function setLoad(n: number): void {
    iters = Math.max(0, n | 0);
}

/** the active per-lane iteration count. */
export function getLoad(): number {
    return iters;
}

export const LoadPlugin: Plugin = {
    name: "Load",
    async warm() {
        const device = Compute.device;
        if (!device) return;
        const root = Compute.root;
        buf = root.createBuffer(LOAD_SCHEMA).$usage("storage").$name("load");
        cfgBuf = root.createBuffer(LoadCfg).$usage("uniform").$name("load-cfg");
        pipeline = root.createComputePipeline({ compute: loadKernel }).$name("load");
        bind = root.createBindGroup(loadLayout, { buf, cfg: cfgBuf });
        const readyBind = bind;
        const readyPipeline = pipeline;
        await precompile("gym-load", () => {
            readyPipeline.with(readyBind).dispatchWorkgroups(0, 1, 1);
            return readyPipeline;
        });
    },
    systems: [
        {
            name: "load",
            group: "draw",
            annotations: { mode: "always" },
            update() {
                const device = Compute.device;
                if (!device || !pipeline || !bind || !cfgBuf || iters <= 0) return;
                cfgBuf.write({ iter: iters, pad0: 0, pad1: 0, pad2: 0 });
                const enc = device.createCommandEncoder({ label: "load" });
                const pass = enc.beginComputePass({
                    label: "load",
                    timestampWrites: Compute.span?.("load"),
                });
                pipeline.with(bind).with(pass).dispatchWorkgroups(LOAD_WG, 1, 1);
                pass.end();
                device.queue.submit([enc.finish()]);
            },
        } satisfies System,
    ],
};

/** release the GPU resources + reset the load to 0 — call from a scenario's `dispose`. */
export function disposeLoad(): void {
    iters = 0;
    buf?.destroy();
    buf = null;
    cfgBuf?.destroy();
    cfgBuf = null;
    pipeline = null;
    bind = null;
}
