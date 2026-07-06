import { Compute, type Plugin, type System } from "@dylanebert/shallot";

// The DRAM-bandwidth burn knob: a streaming read-modify-write over a large (> L2) buffer, `getBandwidth()`
// sweeps per frame on its own submit, each sweep moving 2× the buffer (read + write). Minimal ALU (one
// FMA per vec4), fully coalesced, working set > L2 — so it saturates *memory bandwidth*, not ALU or
// dispatch (gpu.md "Bandwidth ceiling check" / "the L2-cache caveat"). The sibling of the compute `load`
// knob (../load), the same induce-half shape: a per-frame submit the fence waits on (dt grows, fps drops),
// timed as its own "bandwidth" span so the stress atom proves the induced bytes land on their own span and
// no other. `setBandwidth(0)` makes it a no-op.
//
// The buffer is 1 GB — ≫ the 4090's 98 MB L2 (~10% resident), so every sweep re-streams from DRAM (a buffer
// near L2 size would sit in cache and measure latency, not DRAM bandwidth). The knob is the *sweep count*
// (the outer loop); each sweep is one full grid-stride pass + a per-sweep address rotation (below) so the
// passes can't be register-hoisted into an ALU loop. One dispatch, one timestamp pair.

const BW_BYTES = 1 << 30; // 1 GB ≫ the 4090's 98 MB L2 (~10% resident), so each sweep is genuinely DRAM-bound
const BW_VEC4 = BW_BYTES >> 4; // 67,108,864 vec4<f32> (one per lane-step)
const BW_WG = 32768; // workgroups (× 64 = 2,097,152 lanes); a grid-stride covers BW_VEC4 in 32 steps/lane
const BW_LANES = BW_WG * 64;
const BW_SHIFT = 1048573; // a prime per-sweep rotation; coprime with 2^26 so sweeps don't realign
// The sweep count is a DYNAMIC uniform (`cfg.sweeps`) — it scales the moved bytes at runtime. CRITICAL: each
// sweep touches a ROTATED permutation of the buffer (`idx = (i + s·SHIFT) mod N`), so the access address
// depends on the dynamic sweep index `s`. Without the rotation the compiler hoists `buf[i]` into a register
// across the sweep loop (the read + write are the same loop-invariant address) — turning the read-modify-
// write into a register-resident ALU loop that measures COMPUTE, not bandwidth (caught 2026-06-18: the naive
// form reported a physically-impossible 4 TB/s on a 1 TB/s-DRAM 4090). The rotation is a permutation, so
// every sweep still streams all N·16 B > L2 from DRAM. `% N` is a mask (N = 2^26); the wrap is one subtract.
const BW_WGSL = `
struct Cfg { sweeps: u32 };
@group(0) @binding(0) var<storage, read_write> buf: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> cfg: Cfg;
@compute @workgroup_size(64)
fn bandwidth(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var s = 0u; s < cfg.sweeps; s = s + 1u) {
        let off = (s * ${BW_SHIFT}u) % ${BW_VEC4}u;
        var i = gid.x;
        loop {
            if (i >= ${BW_VEC4}u) { break; }
            var idx = i + off;
            if (idx >= ${BW_VEC4}u) { idx = idx - ${BW_VEC4}u; }
            buf[idx] = buf[idx] * 1.0001 + vec4<f32>(0.5);
            i = i + ${BW_LANES}u;
        }
    }
}`;

let pipeline: GPUComputePipeline | null = null;
let bind: GPUBindGroup | null = null;
let buf: GPUBuffer | null = null;
let cfgBuf: GPUBuffer | null = null;
const _cfg = new Uint32Array(1);
let sweeps = 0;

/** bytes moved per sweep (read + write of the whole buffer) — the per-sweep bandwidth unit. */
export const BANDWIDTH_SWEEP_BYTES = BW_BYTES * 2;

/** set the per-frame sweep count — the bandwidth (bytes/frame) inflation level. ≤0 makes the pass a no-op. */
export function setBandwidth(n: number): void {
    sweeps = Math.max(0, n | 0);
}

/** the active per-frame sweep count. */
export function getBandwidth(): number {
    return sweeps;
}

export const BandwidthPlugin: Plugin = {
    name: "Bandwidth",
    async warm() {
        const device = Compute.device;
        if (!device) return;
        buf = device.createBuffer({
            label: "bandwidth",
            size: BW_BYTES,
            usage: GPUBufferUsage.STORAGE,
        });
        cfgBuf = device.createBuffer({
            label: "bandwidth-cfg",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        pipeline = await device.createComputePipelineAsync({
            label: "bandwidth",
            layout: "auto",
            compute: {
                module: device.createShaderModule({ code: BW_WGSL }),
                entryPoint: "bandwidth",
            },
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
            name: "bandwidth",
            group: "draw",
            annotations: { mode: "always" },
            update() {
                const device = Compute.device;
                if (!device || !pipeline || !bind || !cfgBuf || sweeps <= 0) return;
                _cfg[0] = sweeps;
                device.queue.writeBuffer(cfgBuf, 0, _cfg);
                const enc = device.createCommandEncoder({ label: "bandwidth" });
                const pass = enc.beginComputePass({
                    label: "bandwidth",
                    timestampWrites: Compute.span?.("bandwidth"),
                });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bind);
                pass.dispatchWorkgroups(BW_WG, 1, 1);
                pass.end();
                device.queue.submit([enc.finish()]);
            },
        } satisfies System,
    ],
};

/** release the GPU resources + reset the sweep count to 0 — call from a scenario's `dispose`. */
export function disposeBandwidth(): void {
    sweeps = 0;
    buf?.destroy();
    buf = null;
    cfgBuf?.destroy();
    cfgBuf = null;
    pipeline = null;
    bind = null;
}
