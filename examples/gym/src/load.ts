import { Compute, type Plugin, type System } from "@dylanebert/shallot";

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
const LOAD_WGSL = `
struct Cfg { iter: u32 };
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> cfg: Cfg;
@compute @workgroup_size(64)
fn load(@builtin(global_invocation_id) gid: vec3<u32>) {
    var x = f32(gid.x) * 0.0001 + 1.0;
    for (var i = 0u; i < cfg.iter; i = i + 1u) {
        x = sin(x) * 1.0001 + cos(x * 1.3) * 0.5 + 0.7;
    }
    buf[gid.x % ${LOAD_N}u] = x;
}`;

let pipeline: GPUComputePipeline | null = null;
let bind: GPUBindGroup | null = null;
let buf: GPUBuffer | null = null;
let cfgBuf: GPUBuffer | null = null;
const _cfg = new Uint32Array(1);
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
        buf = device.createBuffer({
            label: "load",
            size: LOAD_N * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        cfgBuf = device.createBuffer({
            label: "load-cfg",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        pipeline = await device.createComputePipelineAsync({
            label: "load",
            layout: "auto",
            compute: { module: device.createShaderModule({ code: LOAD_WGSL }), entryPoint: "load" },
        });
        bind = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: buf } },
                { binding: 1, resource: { buffer: cfgBuf } },
            ],
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
                _cfg[0] = iters;
                device.queue.writeBuffer(cfgBuf, 0, _cfg);
                const enc = device.createCommandEncoder({ label: "load" });
                const pass = enc.beginComputePass({
                    label: "load",
                    timestampWrites: Compute.span?.("load"),
                });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bind);
                pass.dispatchWorkgroups(LOAD_WG, 1, 1);
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
