import {
    AmbientLight,
    Body,
    Camera,
    CameraMode,
    Color,
    Compute,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    precompile,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    ShapeKind,
    SlabPlugin,
    type State,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { AvbdPlugin } from "@dylanebert/shallot/avbd";
import {
    Avbd,
    BODY_VEC4,
    collideWgsl,
    MAX_CONTACTS,
    PENALTY_MIN,
    packHulls,
} from "@dylanebert/shallot/avbd/core";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { Hulls } from "@dylanebert/shallot/physics/core";
import { Meshes } from "@dylanebert/shallot/render/core";
import { precompileScope } from "@dylanebert/shallot/runtime";
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
import { boxHull, coneHull, tetHull } from "../../../../packages/shallot/tests/avbd/hull";
import { add, type Quat, rotate, type Vec3 } from "../../../../packages/shallot/tests/avbd/math";
import {
    body,
    capsule,
    hull,
    type Body as OracleBody,
    sphere,
} from "../../../../packages/shallot/tests/avbd/rigid";
import { narrowphase } from "../../../../packages/shallot/tests/avbd/rounded";
import gold from "../../../../packages/shallot/tests/avbd/sat-gold-vectors.json";
import {
    collideBoxBox,
    collideHull,
    collideRounded,
    collideRoundedPolytope,
    polyMake,
    SatResult,
} from "../../../../packages/shallot/tests/avbd/tgsl";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";

// sat — the SAT gate on the real GPU. Runs the production `collideWgsl()` over the 14 C++ gold configs
// (sat-gold-vectors.json, the same spec sat.test.ts and the software-adapter cross-check use), reads
// back contact count + basis + feature key + manifold arms, diffs byte-exact (f32 tol 1e-4, ~5× margin
// derived from accumulated f32 rounding).
//
// The two passes:
//   "main-kernel"     — production-shape: workgroup_size(64), one thread per config, 14 active lanes.
//                       matches the narrowphase pass at standard/avbd/step.ts:325.
//   "ref-kernel"      — workgroup_size(32), one active lane, hardcoded face-y-overlap. The reference
//                       configuration the SAT kernel is correct under.
//
// This gate caught a Metal-3-only miscompile: under multi-lane SIMD execution the full collideBoxBox's
// peak function-private footprint spilled, and Metal miscompiled the per-lane offset of the spilled
// SatResult, collapsing every face manifold to count=1 (single-lane was fine; the ref-kernel proves
// the SAT math correct). Fixed by cutting Poly to its exact 8-vertex bound (collide.ts MAX_POLY_VERTS),
// dropping the footprint below the spill threshold. metal-3 now passes; nvidia/lovelace was already
// green and the fix only reduces an over-allocation, so it stays green. Mechanism + refuted
// candidates: shallot gpu.md "WebGPU-specific traps".

interface GoldContact {
    feature: number;
    rA: number[];
    rB: number[];
}
interface GoldConfig {
    name: string;
    a: { size: number[]; pos: number[]; quat: number[]; vel: number[] };
    b: { size: number[]; pos: number[]; quat: number[]; vel: number[] };
    numContacts: number;
    basis: number[] | null;
    contacts: GoldContact[];
}

const TOL = 1e-4;
// the C++ harness solver dt (gold-sat builds a Solver, dt = 1/60); the velocity sweep reads
// dRel = (velA − velB)·dt, so the kernel must thread the gold's velocities at this dt to match (sat.test.ts).
const DT = 1 / 60;

const at = <T extends d.BaseData>(schema: T, field: (value: d.Infer<T>) => unknown): number =>
    d.memoryLayoutOf(schema, field).offset / 4;

const SatCfg = d
    .struct({
        posA: d.vec4f,
        quatA: d.vec4f,
        sizeA: d.vec4f,
        posB: d.vec4f,
        quatB: d.vec4f,
        sizeB: d.vec4f,
        dRel: d.vec4f,
    })
    .$name("SatCfg");
const ResultContact = d.struct({
    feature: d.f32,
    rA: d.arrayOf(d.f32, 3),
    rB: d.arrayOf(d.f32, 3),
});
const SatReadback = d.struct({
    count: d.f32,
    basis: d.arrayOf(d.f32, 9),
    contacts: d.arrayOf(ResultContact, MAX_CONTACTS),
});
const HullReadback = d.struct({
    count: d.f32,
    normal: d.arrayOf(d.f32, 3),
    contacts: d.arrayOf(ResultContact, MAX_CONTACTS),
});
const SAT_CFG_FLOATS = d.sizeOf(SatCfg) / 4;
const SAT_CFG_AT = {
    posA: at(SatCfg, (c) => c.posA),
    quatA: at(SatCfg, (c) => c.quatA),
    sizeA: at(SatCfg, (c) => c.sizeA),
    posB: at(SatCfg, (c) => c.posB),
    quatB: at(SatCfg, (c) => c.quatB),
    sizeB: at(SatCfg, (c) => c.sizeB),
    dRel: at(SatCfg, (c) => c.dRel),
};
const RESULT_CONTACT_FLOATS = d.sizeOf(ResultContact) / 4;
const RESULT_CONTACT_AT = {
    feature: at(ResultContact, (contact) => contact.feature),
    rA: at(ResultContact, (contact) => contact.rA),
    rB: at(ResultContact, (contact) => contact.rB),
};
const SAT_READBACK_FLOATS = d.sizeOf(SatReadback) / 4;
const SAT_READBACK_AT = {
    count: at(SatReadback, (result) => result.count),
    basis: at(SatReadback, (result) => result.basis),
    contacts: at(SatReadback, (result) => result.contacts),
};
const HULL_READBACK_FLOATS = d.sizeOf(HullReadback) / 4;
const HULL_READBACK_AT = {
    count: at(HullReadback, (result) => result.count),
    normal: at(HullReadback, (result) => result.normal),
    contacts: at(HullReadback, (result) => result.contacts),
};
const SatParams = d
    .struct({ count: d.u32, pad0: d.u32, pad1: d.u32, pad2: d.u32 })
    .$name("SatParams");
const SatConfigs = d.arrayOf(SatCfg);
const SatOutput = d.arrayOf(d.f32);
const satLayout = tgpu.bindGroupLayout({
    cfgs: { storage: SatConfigs, access: "readonly" },
    out: { storage: SatOutput, access: "mutable" },
    params: { uniform: SatParams },
});

const boxSat = tgpu
    .fn(
        [d.vec3f, d.vec4f, d.vec3f, d.vec3f, d.vec4f, d.vec3f, d.vec3f],
        SatResult,
    )((posA, quatA, sizeA, posB, quatB, sizeB, dRel) => {
        "use gpu";
        return collideBoxBox(posA, quatA, sizeA, posB, quatB, sizeB, dRel);
    })
    .$name("boxSat");

const satKernel = tgpu
    .computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [64],
    })((input) => {
        "use gpu";
        const i = input.gid.x;
        if (i >= satLayout.$.params.count) return;
        const c = SatCfg(satLayout.$.cfgs[i]);
        const r = boxSat(
            c.posA.xyz,
            c.quatA,
            c.sizeA.xyz,
            c.posB.xyz,
            c.quatB,
            c.sizeB.xyz,
            c.dRel.xyz,
        );
        const base = i * d.u32(SAT_READBACK_FLOATS);
        satLayout.$.out[base + d.u32(SAT_READBACK_AT.count)] = std.bitcastU32toF32(r.count);
        const basis = base + d.u32(SAT_READBACK_AT.basis);
        satLayout.$.out[basis] = r.basis.r0.x;
        satLayout.$.out[basis + d.u32(1)] = r.basis.r0.y;
        satLayout.$.out[basis + d.u32(2)] = r.basis.r0.z;
        satLayout.$.out[basis + d.u32(3)] = r.basis.r1.x;
        satLayout.$.out[basis + d.u32(4)] = r.basis.r1.y;
        satLayout.$.out[basis + d.u32(5)] = r.basis.r1.z;
        satLayout.$.out[basis + d.u32(6)] = r.basis.r2.x;
        satLayout.$.out[basis + d.u32(7)] = r.basis.r2.y;
        satLayout.$.out[basis + d.u32(8)] = r.basis.r2.z;
        for (let k = d.u32(0); k < MAX_CONTACTS; k = k + d.u32(1)) {
            const contact =
                base + d.u32(SAT_READBACK_AT.contacts) + k * d.u32(RESULT_CONTACT_FLOATS);
            satLayout.$.out[contact + d.u32(RESULT_CONTACT_AT.feature)] = std.bitcastU32toF32(
                r.feat[k],
            );
            const rA = contact + d.u32(RESULT_CONTACT_AT.rA);
            satLayout.$.out[rA] = r.rA[k].x;
            satLayout.$.out[rA + d.u32(1)] = r.rA[k].y;
            satLayout.$.out[rA + d.u32(2)] = r.rA[k].z;
            const rB = contact + d.u32(RESULT_CONTACT_AT.rB);
            satLayout.$.out[rB] = r.rB[k].x;
            satLayout.$.out[rB + d.u32(1)] = r.rB[k].y;
            satLayout.$.out[rB + d.u32(2)] = r.rB[k].z;
        }
    })
    .$name("satMain");

// Reference kernel — workgroup_size(32) (matches Apple's SIMD width) with a single active lane on the
// face-y-overlap config (A unit cube at origin, B unit cube at y=0.97, expected count=4). A single-lane
// control: it was already bit-correct on apple/metal-3 while the multi-lane main-kernel diverged, which
// pinned the (now-fixed) Phase 4.5 spill to the multi-lane execution shape rather than the SAT math.
const REF_OUT_LEN = 1 + MAX_CONTACTS * 4; // count + MAX × (feat, rA.xyz)
const refKernelWgsl = () => `${collideWgsl()}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) { return; }
    let r = collideBoxBox(
        vec3<f32>(0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.0, 0.97, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0));
    out[0u] = bitcast<f32>(r.count);
    for (var k = 0u; k < ${MAX_CONTACTS}u; k = k + 1u) {
        let o = 1u + k * 4u;
        out[o + 0u] = bitcast<f32>(r.feat[k]);
        out[o + 1u] = r.rA[k].x;
        out[o + 2u] = r.rA[k].y;
        out[o + 3u] = r.rA[k].z;
    }
}`;

interface GpuResult {
    count: number;
    basis: number[];
    contacts: { feature: number; rA: number[]; rB: number[] }[];
}

/** pack the C++ box configurations into the production seven-vec4 storage record. @internal */
export function packBoxConfigs(configs: readonly GoldConfig[]): Float32Array<ArrayBuffer> {
    const data = new Float32Array(configs.length * SAT_CFG_FLOATS);
    for (let i = 0; i < configs.length; i++) {
        const c = configs[i];
        const o = i * SAT_CFG_FLOATS;
        data.set(c.a.pos, o + SAT_CFG_AT.posA);
        data.set(c.a.quat, o + SAT_CFG_AT.quatA);
        data.set(c.a.size, o + SAT_CFG_AT.sizeA);
        data.set(c.b.pos, o + SAT_CFG_AT.posB);
        data.set(c.b.quat, o + SAT_CFG_AT.quatB);
        data.set(c.b.size, o + SAT_CFG_AT.sizeB);
        for (let k = 0; k < 3; k++) data[o + SAT_CFG_AT.dRel + k] = (c.a.vel[k] - c.b.vel[k]) * DT;
    }
    return data;
}

/** run the exact production TGSL leaf on the CPU for the C++ differential. @internal */
export function boxResult(config: GoldConfig): GpuResult {
    const r = boxSat(
        d.vec3f(config.a.pos[0], config.a.pos[1], config.a.pos[2]),
        d.vec4f(config.a.quat[0], config.a.quat[1], config.a.quat[2], config.a.quat[3]),
        d.vec3f(config.a.size[0], config.a.size[1], config.a.size[2]),
        d.vec3f(config.b.pos[0], config.b.pos[1], config.b.pos[2]),
        d.vec4f(config.b.quat[0], config.b.quat[1], config.b.quat[2], config.b.quat[3]),
        d.vec3f(config.b.size[0], config.b.size[1], config.b.size[2]),
        d.vec3f(
            (config.a.vel[0] - config.b.vel[0]) * DT,
            (config.a.vel[1] - config.b.vel[1]) * DT,
            (config.a.vel[2] - config.b.vel[2]) * DT,
        ),
    );
    const contacts: GpuResult["contacts"] = [];
    for (let k = 0; k < r.count; k++) {
        contacts.push({
            feature: r.feat[k],
            rA: [r.rA[k].x, r.rA[k].y, r.rA[k].z],
            rB: [r.rB[k].x, r.rB[k].y, r.rB[k].z],
        });
    }
    return {
        count: r.count,
        basis: [
            r.basis.r0.x,
            r.basis.r0.y,
            r.basis.r0.z,
            r.basis.r1.x,
            r.basis.r1.y,
            r.basis.r1.z,
            r.basis.r2.x,
            r.basis.r2.y,
            r.basis.r2.z,
        ],
        contacts,
    };
}

/** workgroups needed for one 64-lane thread per input record. @internal */
export function workgroupsFor(count: number): number {
    return Math.ceil(count / 64);
}

type SatConfigsBuffer = TgpuBuffer<ReturnType<typeof SatConfigs>> & StorageFlag;
type SatOutputBuffer = TgpuBuffer<ReturnType<typeof SatOutput>> & StorageFlag;
type SatParamsBuffer = TgpuBuffer<typeof SatParams> & UniformFlag;
type SatGroup = TgpuBindGroup<typeof satLayout.entries>;

async function runSat(): Promise<GpuResult[]> {
    const configs = gold.configs as GoldConfig[];
    const n = configs.length;
    const { device, root } = Compute;
    const cfgData = packBoxConfigs(configs);
    const bytesOut = n * d.sizeOf(SatReadback);
    let cfgs: SatConfigsBuffer | undefined;
    let outRaw: GPUBuffer | undefined;
    let params: SatParamsBuffer | undefined;
    let read: GPUBuffer | undefined;
    let bytes: ArrayBuffer;
    try {
        cfgs = root.createBuffer(SatConfigs(n)).$usage("storage").$name("sat-configs");
        cfgs.write(cfgData.buffer);
        outRaw = device.createBuffer({
            label: "sat-out",
            size: bytesOut,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const out: SatOutputBuffer = root
            .createBuffer(SatOutput(n * SAT_READBACK_FLOATS), outRaw)
            .$usage("storage");
        params = root.createBuffer(SatParams).$usage("uniform").$name("sat-params");
        params.write({ count: n, pad0: 0, pad1: 0, pad2: 0 });
        read = device.createBuffer({
            label: "sat-read",
            size: bytesOut,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const group: SatGroup = root.createBindGroup(satLayout, { cfgs, out, params });
        const pipelines = await satPipelines(root, device);
        const encoder = device.createCommandEncoder({ label: "sat-main" });
        const pass = encoder.beginComputePass({ label: "sat-main" });
        pipelines.main.with(group).with(pass).dispatchWorkgroups(workgroupsFor(n));
        pass.end();
        encoder.copyBufferToBuffer(outRaw, 0, read, 0, bytesOut);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        bytes = read.getMappedRange().slice(0);
        read.unmap();
    } finally {
        cfgs?.destroy();
        outRaw?.destroy();
        params?.destroy();
        read?.destroy();
    }

    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const out: GpuResult[] = [];
    for (let ci = 0; ci < n; ci++) {
        const base = ci * SAT_READBACK_FLOATS;
        const count = u[base + SAT_READBACK_AT.count];
        const basis: number[] = [];
        for (let i = 0; i < 9; i++) basis.push(f[base + SAT_READBACK_AT.basis + i]);
        const contacts: GpuResult["contacts"] = [];
        for (let k = 0; k < count; k++) {
            const contact = base + SAT_READBACK_AT.contacts + k * RESULT_CONTACT_FLOATS;
            const rA = contact + RESULT_CONTACT_AT.rA;
            const rB = contact + RESULT_CONTACT_AT.rB;
            contacts.push({
                feature: u[contact + RESULT_CONTACT_AT.feature],
                rA: [f[rA], f[rA + 1], f[rA + 2]],
                rB: [f[rB], f[rB + 1], f[rB + 2]],
            });
        }
        out.push({ count, basis, contacts });
    }
    return out;
}

// run the wg(32)-single-lane reference kernel — proves the SAT body is correct on Metal in isolation.
async function runRef(): Promise<{ count: number; entries: { feat: number; rA: number[] }[] }> {
    const device = Compute.device;
    const outBuf = device.createBuffer({
        label: "sat-ref-out",
        size: REF_OUT_LEN * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuf = device.createBuffer({
        label: "sat-ref-read",
        size: REF_OUT_LEN * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const pipeline = await device.createComputePipelineAsync({
        label: "sat-ref",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "sat-ref-module", code: refKernelWgsl() }),
            entryPoint: "main",
        },
    });
    const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: outBuf } }],
    });
    const enc = device.createCommandEncoder({ label: "sat-ref-encoder" });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, REF_OUT_LEN * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const bytes = readBuf.getMappedRange().slice(0);
    readBuf.unmap();
    outBuf.destroy();
    readBuf.destroy();
    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const entries: { feat: number; rA: number[] }[] = [];
    for (let k = 0; k < MAX_CONTACTS; k++) {
        const o = 1 + k * 4;
        entries.push({ feat: u[o], rA: [f[o + 1], f[o + 2], f[o + 3]] });
    }
    return { count: u[0], entries };
}

// compare one config against its gold twin. Returns "" on agreement, else a one-line detail naming
// what diverged — count, feature key, basis component, or arm coordinate.
/** compare one production result with its C++ gold record. @internal */
export function diff(cfg: GoldConfig, got: GpuResult): { detail: string; err: number } {
    let err = 0;
    if (got.count !== cfg.numContacts) {
        const feats = got.contacts
            .map((c) => `0x${(c.feature >>> 0).toString(16).padStart(8, "0")}`)
            .join(",");
        return {
            detail: `count ${got.count} vs gold ${cfg.numContacts} (gpu features: ${feats || "—"})`,
            err: 0,
        };
    }
    if (cfg.numContacts === 0) return { detail: "", err: 0 };

    const gb = cfg.basis as number[];
    for (let i = 0; i < 9; i++) {
        const d = Math.abs(got.basis[i] - gb[i]);
        err = Math.max(err, d);
        if (d >= TOL) {
            return {
                detail: `basis[${i}] gpu ${got.basis[i].toExponential(3)} vs gold ${gb[i].toExponential(3)} (Δ ${d.toExponential(2)})`,
                err,
            };
        }
    }

    // match each gold contact to its GPU twin by feature key (bit-identical u32) — order-independent.
    const map = new Map<number, { rA: number[]; rB: number[] }>();
    for (const c of got.contacts) map.set(c.feature >>> 0, c);
    for (const want of cfg.contacts) {
        const g = map.get(want.feature >>> 0);
        if (!g) {
            const have = Array.from(map.keys())
                .map((k) => `0x${k.toString(16)}`)
                .join(",");
            return {
                detail: `missing feature 0x${(want.feature >>> 0).toString(16)} (gpu features: ${have})`,
                err,
            };
        }
        for (let i = 0; i < 3; i++) {
            const d = Math.abs(g.rA[i] - want.rA[i]);
            err = Math.max(err, d);
            if (d >= TOL) {
                return {
                    detail: `feat 0x${(want.feature >>> 0).toString(16)} rA[${i}] gpu ${g.rA[i].toExponential(3)} vs gold ${want.rA[i].toExponential(3)} (Δ ${d.toExponential(2)})`,
                    err,
                };
            }
        }
        for (let i = 0; i < 3; i++) {
            const d = Math.abs(g.rB[i] - want.rB[i]);
            err = Math.max(err, d);
            if (d >= TOL) {
                return {
                    detail: `feat 0x${(want.feature >>> 0).toString(16)} rB[${i}] gpu ${g.rB[i].toExponential(3)} vs gold ${want.rB[i].toExponential(3)} (Δ ${d.toExponential(2)})`,
                    err,
                };
            }
        }
    }
    return { detail: "", err };
}

async function adapterInfo(): Promise<string> {
    const adapter = await navigator.gpu?.requestAdapter();
    const info = adapter?.info;
    if (!info) return "unknown";
    return [info.vendor, info.architecture, info.device, info.description]
        .filter(Boolean)
        .join(" / ");
}

// ── hull SAT gate (Phase 6.3) ────────────────────────────────────────────────────────────────────────
// The production hull dispatch (collideHull / collideRoundedPolytope WGSL) over the polytope families —
// box-as-hull (reproduces box-box), tet, cone, the box × hull mixed path, and rounded × hull (sphere/
// capsule) — diffed against the f64 oracle `narrowphase` (tests/avbd/rounded.ts, which dispatches the same
// matrix). Geometry is matched order-independently: count + B→A normal + each oracle contact's nearest GPU
// contact by feature key, then compares its world anchors. Symmetric reductions use the feature key as
// their deterministic final tie-break, so a precision-dependent alternate manifold is a real mismatch.
// The hull geometry is packed by the production `packHulls` and bound as `hullData`.

const HULL_TOL = 2e-3; // f32 vs f64 over the SAT projection/clip across many verts — looser than the box 1e-4
const Z90H: Quat = [0, 0, Math.SQRT1_2, Math.SQRT1_2]; // +Y capsule axis → horizontal (−X)
const TET_ROT: Quat = [0, 0.24740396, 0, 0.96891242]; // the bullet gold tet-tet-rotated quat

let hullSeq = 0;
const reg = (geom: ReturnType<typeof boxHull>): number =>
    Hulls.register({ name: `sat-${hullSeq++}`, ...geom });
// a hull body + its registered id (the GPU references geometry by id; the oracle body carries it directly)
const bh = (
    geom: ReturnType<typeof boxHull>,
    mass: number,
    pos: Vec3,
    quat: Quat = [0, 0, 0, 1],
    vel: Vec3 = [0, 0, 0],
): { body: OracleBody; id: number } => ({
    body: hull(geom, mass, 0.5, pos, vel, quat),
    id: reg(geom),
});

interface HCfg {
    name: string;
    a: OracleBody;
    b: OracleBody;
    ha: number; // hullId of a (0 if a is not a hull)
    hb: number;
    dRel: Vec3;
}

function hullConfigs(): HCfg[] {
    const cfgs: HCfg[] = [];
    const add2 = (
        name: string,
        a: { body: OracleBody; id: number } | OracleBody,
        b: { body: OracleBody; id: number } | OracleBody,
        dRel: Vec3 = [0, 0, 0],
    ) => {
        const ab = "body" in a ? a : { body: a, id: 0 };
        const bb = "body" in b ? b : { body: b, id: 0 };
        cfgs.push({ name, a: ab.body, b: bb.body, ha: ab.id, hb: bb.id, dRel });
    };

    // box-as-hull (hull × hull, both box-hulls) — reproduces the box-box face manifold on the hull path
    add2(
        "box-hull face rest",
        bh(boxHull([1, 1, 1]), 1, [0, 0.97, 0]),
        bh(boxHull([6, 1, 6]), 0, [0, 0, 0]),
    );
    add2(
        "box-hull offset face",
        bh(boxHull([1, 1, 1]), 1, [0.3, 0.97, 0.2]),
        bh(boxHull([6, 1, 6]), 0, [0, 0, 0]),
    );
    add2(
        "box-hull side overlap (x)",
        bh(boxHull([1, 1, 1]), 1, [0.8, 0, 0]),
        bh(boxHull([1, 1, 1]), 0, [0, 0, 0]),
    );
    // mixed: a box-hull vs a plain box body (and the box-as-A ordering)
    add2(
        "box-hull A vs box B",
        bh(boxHull([1, 1, 1]), 1, [0, 0.97, 0]),
        body([6, 1, 6], 0, 0.5, [0, 0, 0]),
    );
    add2(
        "box A vs box-hull B",
        body([1, 1, 1], 1, 0.5, [0, 0.97, 0]),
        bh(boxHull([6, 1, 6]), 0, [0, 0, 0]),
    );
    // tet / cone — the bullet-gold overlapping poses (genuinely non-box SAT, slanted faces + edges)
    add2(
        "tet-cube overlap",
        bh(tetHull(0.5), 1, [0, 0, 0]),
        bh(boxHull([1, 1, 1]), 0, [0.5, 0, 0]),
    );
    add2("tet-tet overlap", bh(tetHull(0.5), 1, [0, 0, 0]), bh(tetHull(0.5), 0, [0.6, 0, 0]));
    add2(
        "cone8-cube overlap",
        bh(coneHull(0.4, 1, 8), 1, [0, 0.8, 0]),
        bh(boxHull([1, 1, 1]), 0, [0, 0, 0]),
    );
    add2(
        "tet-tet rotated",
        bh(tetHull(0.5), 1, [0, 0, 0]),
        bh(tetHull(0.5), 0, [0.4, 0.2, 0], TET_ROT),
    );
    add2(
        "cone8-tet overlap",
        bh(coneHull(0.3, 0.8, 8), 1, [0, 0.4, 0]),
        bh(tetHull(0.5), 0, [0, 0, 0]),
    );
    // rounded × hull — sphere closest-point + the capsule segment-clip, the new manifold the port unlocks
    add2(
        "sphere × box-hull face",
        sphere(0.5, 1, 0.5, [0, 1.4, 0]),
        bh(boxHull([2, 2, 2]), 0, [0, 0, 0]),
    );
    add2(
        "sphere × box-hull, box-as-A",
        bh(boxHull([2, 2, 2]), 0, [0, 0, 0]),
        sphere(0.5, 1, 0.5, [0, 1.4, 0]),
    );
    add2("sphere × tet", sphere(0.4, 1, 0.5, [0, 0.95, 0]), bh(tetHull(0.5), 0, [0, 0, 0]));
    add2(
        "capsule × box-hull flat",
        capsule(1, 0.4, 1, 0.5, [0, 1.4, 0], [0, 0, 0], Z90H),
        bh(boxHull([4, 2, 4]), 0, [0, 0, 0]),
    );
    add2(
        "capsule × box-hull overhang (mid-segment)",
        capsule(1.5, 0.4, 1, 0.5, [0, 0.9, 0], [0, 0, 0], Z90H),
        bh(boxHull([1, 1, 1]), 0, [0, 0, 0]),
    );
    return cfgs;
}

const HullCfg = d
    .struct({
        posA: d.vec4f,
        quatA: d.vec4f,
        sizeRadA: d.vec4f,
        posB: d.vec4f,
        quatB: d.vec4f,
        sizeRadB: d.vec4f,
        dRel: d.vec4f,
        shapes: d.vec4f,
    })
    .$name("HullCfg");
const HULL_CFG_FLOATS = d.sizeOf(HullCfg) / 4;
const HULL_CFG_AT = {
    posA: at(HullCfg, (c) => c.posA),
    quatA: at(HullCfg, (c) => c.quatA),
    sizeRadA: at(HullCfg, (c) => c.sizeRadA),
    posB: at(HullCfg, (c) => c.posB),
    quatB: at(HullCfg, (c) => c.quatB),
    sizeRadB: at(HullCfg, (c) => c.sizeRadB),
    dRel: at(HullCfg, (c) => c.dRel),
    shapes: at(HullCfg, (c) => c.shapes),
};
const HullConfigs = d.arrayOf(HullCfg);
const HullData = d.arrayOf(d.u32);
const hullLayout = tgpu.bindGroupLayout({
    cfgs: { storage: HullConfigs, access: "readonly" },
    out: { storage: SatOutput, access: "mutable" },
    params: { uniform: SatParams },
    hullData: { storage: HullData, access: "readonly" },
});

const hullKernel = tgpu
    .computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [64],
    })((input) => {
        "use gpu";
        const i = input.gid.x;
        if (i >= hullLayout.$.params.count) return;
        const c = HullCfg(hullLayout.$.cfgs[i]);
        const sA = d.u32(c.shapes.x);
        const sB = d.u32(c.shapes.y);
        const hA = d.u32(c.shapes.z);
        const hB = d.u32(c.shapes.w);
        const roundedA = sA === d.u32(1) || sA === d.u32(2);
        const roundedB = sB === d.u32(1) || sB === d.u32(2);
        let r = SatResult();
        if (roundedA && roundedB) {
            r = SatResult(
                collideRounded(
                    c.posA.xyz,
                    c.quatA,
                    c.sizeRadA.xyz,
                    c.sizeRadA.w,
                    c.posB.xyz,
                    c.quatB,
                    c.sizeRadB.xyz,
                    c.sizeRadB.w,
                    c.dRel.xyz,
                ),
            );
        } else {
            if (!(roundedA || roundedB)) {
                if (sA === d.u32(0) && sB === d.u32(0)) {
                    r = SatResult(
                        boxSat(
                            c.posA.xyz,
                            c.quatA,
                            c.sizeRadA.xyz,
                            c.posB.xyz,
                            c.quatB,
                            c.sizeRadB.xyz,
                            c.dRel.xyz,
                        ),
                    );
                } else {
                    r = SatResult(
                        collideHull(
                            polyMake(sA, c.posA.xyz, c.quatA, c.sizeRadA.xyz, hA),
                            polyMake(sB, c.posB.xyz, c.quatB, c.sizeRadB.xyz, hB),
                            c.dRel.xyz,
                        ),
                    );
                }
            } else {
                r = SatResult(
                    collideRoundedPolytope(
                        c.posA.xyz,
                        c.quatA,
                        c.sizeRadA.xyz,
                        c.sizeRadA.w,
                        sA,
                        hA,
                        c.posB.xyz,
                        c.quatB,
                        c.sizeRadB.xyz,
                        c.sizeRadB.w,
                        sB,
                        hB,
                        c.dRel.xyz,
                    ),
                );
            }
        }
        const base = i * d.u32(HULL_READBACK_FLOATS);
        const hullZero = std.bitcastU32toF32(hullLayout.$.hullData[d.u32(0)] & d.u32(0));
        hullLayout.$.out[base + d.u32(HULL_READBACK_AT.count)] = std.bitcastU32toF32(r.count);
        const normal = base + d.u32(HULL_READBACK_AT.normal);
        hullLayout.$.out[normal] = r.basis.r0.x + hullZero;
        hullLayout.$.out[normal + d.u32(1)] = r.basis.r0.y;
        hullLayout.$.out[normal + d.u32(2)] = r.basis.r0.z;
        for (let k = d.u32(0); k < MAX_CONTACTS; k = k + d.u32(1)) {
            const contact =
                base + d.u32(HULL_READBACK_AT.contacts) + k * d.u32(RESULT_CONTACT_FLOATS);
            hullLayout.$.out[contact + d.u32(RESULT_CONTACT_AT.feature)] = std.bitcastU32toF32(
                r.feat[k],
            );
            const rA = contact + d.u32(RESULT_CONTACT_AT.rA);
            hullLayout.$.out[rA] = r.rA[k].x;
            hullLayout.$.out[rA + d.u32(1)] = r.rA[k].y;
            hullLayout.$.out[rA + d.u32(2)] = r.rA[k].z;
            const rB = contact + d.u32(RESULT_CONTACT_AT.rB);
            hullLayout.$.out[rB] = r.rB[k].x;
            hullLayout.$.out[rB + d.u32(1)] = r.rB[k].y;
            hullLayout.$.out[rB + d.u32(2)] = r.rB[k].z;
        }
    })
    .$name("satHull");

/** the emitted typed SAT entries and certified collide graph. @internal */
export function satEntryWgsl(): string {
    return [satKernel, hullKernel]
        .map((kernel) => tgpu.resolve([kernel], { names: "strict" }))
        .join("\n");
}

interface SatPipelines {
    readonly main: TgpuComputePipeline;
    readonly hull: TgpuComputePipeline;
}

function forceSatMain(
    root: typeof Compute.root,
    device: GPUDevice,
    pipeline: TgpuComputePipeline,
): TgpuComputePipeline {
    const owned: { destroy(): void }[] = [];
    let submitted = false;
    const cleanup = () => {
        for (const resource of owned) resource.destroy();
    };
    try {
        const cfgs = root.createBuffer(SatConfigs(1)).$usage("storage");
        const out = root.createBuffer(SatOutput(SAT_READBACK_FLOATS)).$usage("storage");
        const params = root.createBuffer(SatParams).$usage("uniform");
        owned.push(cfgs, out, params);
        const group: SatGroup = root.createBindGroup(satLayout, { cfgs, out, params });
        const encoder = device.createCommandEncoder({ label: "sat-main-force" });
        const pass = encoder.beginComputePass({ label: "sat-main-force" });
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

function forceSatHull(
    root: typeof Compute.root,
    device: GPUDevice,
    pipeline: TgpuComputePipeline,
): TgpuComputePipeline {
    const owned: { destroy(): void }[] = [];
    let submitted = false;
    const cleanup = () => {
        for (const resource of owned) resource.destroy();
    };
    try {
        const cfgs = root.createBuffer(HullConfigs(1)).$usage("storage");
        const out = root.createBuffer(SatOutput(HULL_READBACK_FLOATS)).$usage("storage");
        const params = root.createBuffer(SatParams).$usage("uniform");
        const hullData = root.createBuffer(HullData(1)).$usage("storage");
        owned.push(cfgs, out, params, hullData);
        const group: HullGroup = root.createBindGroup(hullLayout, {
            cfgs,
            out,
            params,
            hullData,
        });
        const encoder = device.createCommandEncoder({ label: "sat-hull-force" });
        const pass = encoder.beginComputePass({ label: "sat-hull-force" });
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

/** one retryable single-flight pipeline record per adopted TypeGPU root for post-build callers.
 *  `initialize` must resolve only after compilation completes; a queued warm-time {@link precompile}
 *  registration resolves before the build drain and does not satisfy this contract. @internal */
export function createLateSatPipelineCache<Root extends object, Context, Pipelines>(
    create: (root: Root, context: Context) => Pipelines,
    initialize: (root: Root, context: Context, pipelines: Pipelines) => Promise<void>,
): (root: Root, context: Context) => Promise<Pipelines> {
    const cache = new WeakMap<Root, { pipelines: Pipelines; ready: Promise<void> }>();
    return async (root, context) => {
        let entry = cache.get(root);
        if (!entry) {
            const pipelines = create(root, context);
            const draft = { pipelines, ready: Promise.resolve() };
            entry = draft;
            cache.set(root, entry);
            try {
                draft.ready = initialize(root, context, pipelines).catch((cause) => {
                    if (cache.get(root) === draft) cache.delete(root);
                    throw cause;
                });
            } catch (cause) {
                if (cache.get(root) === draft) cache.delete(root);
                throw cause;
            }
        }
        await entry.ready;
        return entry.pipelines;
    };
}

const cachedSatPipelines = createLateSatPipelineCache(
    (root: typeof Compute.root) =>
        Object.freeze({
            main: root.createComputePipeline({ compute: satKernel }).$name("sat-main"),
            hull: root.createComputePipeline({ compute: hullKernel }).$name("sat-hull"),
        }),
    async (root, device: GPUDevice, pipelines) => {
        const scope = precompileScope("gym-sat");
        await precompile(`${scope}-main`, () => forceSatMain(root, device, pipelines.main));
        await precompile(`${scope}-hull`, () => forceSatHull(root, device, pipelines.hull));
    },
);

function satPipelines(root: typeof Compute.root, device: GPUDevice): Promise<SatPipelines> {
    // scenario assertions run after build's precompileAll drain, so these late registrations return the
    // real validation + completion promises rather than warm-time queue acknowledgements.
    return cachedSatPipelines(root, device);
}

interface GpuHull {
    count: number;
    normal: Vec3;
    contacts: { feature: number; rA: Vec3; rB: Vec3 }[];
}

/** pack the f64 hull-oracle cases into the production eight-vec4 storage record. @internal */
export function packHullConfigs(cfgs: readonly HCfg[]): Float32Array<ArrayBuffer> {
    const data = new Float32Array(cfgs.length * HULL_CFG_FLOATS);
    for (let i = 0; i < cfgs.length; i++) {
        const c = cfgs[i];
        const o = i * HULL_CFG_FLOATS;
        data.set(c.a.posLin, o + HULL_CFG_AT.posA);
        data.set(c.a.posAng, o + HULL_CFG_AT.quatA);
        data.set(
            [c.a.size[0], c.a.size[1], c.a.size[2], c.a.roundRadius],
            o + HULL_CFG_AT.sizeRadA,
        );
        data.set(c.b.posLin, o + HULL_CFG_AT.posB);
        data.set(c.b.posAng, o + HULL_CFG_AT.quatB);
        data.set(
            [c.b.size[0], c.b.size[1], c.b.size[2], c.b.roundRadius],
            o + HULL_CFG_AT.sizeRadB,
        );
        data.set(c.dRel, o + HULL_CFG_AT.dRel);
        data.set([c.a.shape, c.b.shape, c.ha, c.hb], o + HULL_CFG_AT.shapes);
    }
    return data;
}

type HullConfigsBuffer = TgpuBuffer<ReturnType<typeof HullConfigs>> & StorageFlag;
type HullDataBuffer = TgpuBuffer<ReturnType<typeof HullData>> & StorageFlag;
type HullGroup = TgpuBindGroup<typeof hullLayout.entries>;

async function runHullKernel(cfgs: HCfg[]): Promise<GpuHull[]> {
    const n = cfgs.length;
    const { device, root } = Compute;
    const cfgData = packHullConfigs(cfgs);
    const hullData = packHulls();
    const bytesOut = n * d.sizeOf(HullReadback);
    let configs: HullConfigsBuffer | undefined;
    let outRaw: GPUBuffer | undefined;
    let params: SatParamsBuffer | undefined;
    let hulls: HullDataBuffer | undefined;
    let read: GPUBuffer | undefined;
    let bytes: ArrayBuffer;
    try {
        configs = root.createBuffer(HullConfigs(n)).$usage("storage").$name("sat-hull-configs");
        configs.write(cfgData.buffer);
        outRaw = device.createBuffer({
            label: "sat-hull-out",
            size: bytesOut,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const out: SatOutputBuffer = root
            .createBuffer(SatOutput(n * HULL_READBACK_FLOATS), outRaw)
            .$usage("storage");
        params = root.createBuffer(SatParams).$usage("uniform").$name("sat-hull-params");
        params.write({ count: n, pad0: 0, pad1: 0, pad2: 0 });
        hulls = root
            .createBuffer(HullData(hullData.length))
            .$usage("storage")
            .$name("sat-hull-data");
        hulls.write(hullData.buffer as ArrayBuffer);
        read = device.createBuffer({
            label: "sat-hull-read",
            size: bytesOut,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const group: HullGroup = root.createBindGroup(hullLayout, {
            cfgs: configs,
            out,
            params,
            hullData: hulls,
        });
        const pipelines = await satPipelines(root, device);
        const encoder = device.createCommandEncoder({ label: "sat-hull" });
        const pass = encoder.beginComputePass({ label: "sat-hull" });
        pipelines.hull.with(group).with(pass).dispatchWorkgroups(workgroupsFor(n));
        pass.end();
        encoder.copyBufferToBuffer(outRaw, 0, read, 0, bytesOut);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        bytes = read.getMappedRange().slice(0);
        read.unmap();
    } finally {
        configs?.destroy();
        outRaw?.destroy();
        params?.destroy();
        hulls?.destroy();
        read?.destroy();
    }

    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const out: GpuHull[] = [];
    for (let i = 0; i < n; i++) {
        const b = i * HULL_READBACK_FLOATS;
        const count = u[b + HULL_READBACK_AT.count];
        const contacts: GpuHull["contacts"] = [];
        for (let k = 0; k < count; k++) {
            const contact = b + HULL_READBACK_AT.contacts + k * RESULT_CONTACT_FLOATS;
            const rA = contact + RESULT_CONTACT_AT.rA;
            const rB = contact + RESULT_CONTACT_AT.rB;
            contacts.push({
                feature: u[contact + RESULT_CONTACT_AT.feature],
                rA: [f[rA], f[rA + 1], f[rA + 2]],
                rB: [f[rB], f[rB + 1], f[rB + 2]],
            });
        }
        const normal = b + HULL_READBACK_AT.normal;
        out.push({ count, normal: [f[normal], f[normal + 1], f[normal + 2]], contacts });
    }
    return out;
}

// diff a GPU hull result against the oracle narrowphase: count, the B→A normal, then each contact by its
// persistent feature key and world anchors. Order is irrelevant; feature identity is not.
function diffHull(cfg: HCfg, got: GpuHull): { detail: string; err: number } {
    const { contacts, basis } = narrowphase(cfg.a, cfg.b, cfg.dRel);
    if (got.count !== contacts.length)
        return { detail: `count ${got.count} vs oracle ${contacts.length}`, err: 0 };
    if (contacts.length === 0) return { detail: "", err: 0 };

    let err = 0;
    for (let i = 0; i < 3; i++) {
        const d = Math.abs(got.normal[i] - basis[0][i]);
        err = Math.max(err, d);
        if (d >= HULL_TOL)
            return {
                detail: `normal[${i}] gpu ${got.normal[i].toExponential(3)} vs oracle ${basis[0][i].toExponential(3)}`,
                err,
            };
    }

    const xA = (b: OracleBody, r: Vec3): Vec3 => add(rotate(b.posAng, r), b.posLin);
    const xB = (b: OracleBody, r: Vec3): Vec3 => add(rotate(b.posAng, r), b.posLin);
    const oraclePts = contacts.map((c) => ({
        feature: c.feature >>> 0,
        a: xA(cfg.a, c.rA),
        b: xB(cfg.b, c.rB),
    }));
    const gpuPts = new Map(
        got.contacts.map((c) => [c.feature >>> 0, { a: xA(cfg.a, c.rA), b: xB(cfg.b, c.rB) }]),
    );
    for (const op of oraclePts) {
        const best = gpuPts.get(op.feature);
        if (!best) {
            const have = [...gpuPts.keys()].map((key) => `0x${key.toString(16)}`).join(",");
            return {
                detail: `missing feature 0x${op.feature.toString(16)} (gpu features: ${have})`,
                err,
            };
        }
        for (let i = 0; i < 3; i++) {
            const da = Math.abs(best.a[i] - op.a[i]);
            const db = Math.abs(best.b[i] - op.b[i]);
            err = Math.max(err, da, db);
            if (da >= HULL_TOL || db >= HULL_TOL)
                return {
                    detail: `anchor xA[${i}] Δ ${da.toExponential(2)} / xB[${i}] Δ ${db.toExponential(2)}`,
                    err,
                };
        }
    }
    return { detail: "", err };
}

// ── full-pipeline hull rests (Phase 6.3) ───────────────────────────────────────────────────────────────
// The kernel gate above validates the narrowphase math standalone; these settle convex-hull bodies through
// the WHOLE AvbdPlugin (seed reads hullId from halfExtents.w → uploads the packed Hulls → the collide
// dispatch routes box × hull + hull × hull → the solver), reading the rest pose back via Mirror. Box-hulls
// render as their AABB cube (collider == render mesh), so the rests are visible. The oracle (hull.test.ts)
// pins the expected heights: a unit box-hull rests at ground-top + half − COLLISION_MARGIN.
const MARGIN = 0.01;
const DROP_X = 0; // a box-hull dropped on a box ground (the box × hull mixed pipeline)
const STACK_X = 10; // two box-hulls stacking on a box ground
const HULLG_X = -10; // a box-hull resting on a box-hull ground (hull × hull)

function spawnBox(
    state: State,
    pos: Vec3,
    half: Vec3,
    mass: number,
    color: [number, number, number],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Box);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, mass);
    state.add(eid, Part);
    Part.mesh.set(eid, Meshes.id("cube") ?? 0);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

let hullRest = 0;
function spawnBoxHull(
    state: State,
    pos: Vec3,
    fullSize: Vec3,
    mass: number,
    color: [number, number, number],
): number {
    const id = Hulls.register({ name: `rest-${hullRest++}`, ...boxHull(fullSize) });
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Hull);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    // halfExtents.xyz = the hull's AABB half (mass + broadphase), .w = the registered hull id
    Body.halfExtents.set(eid, fullSize[0] / 2, fullSize[1] / 2, fullSize[2] / 2, id);
    Body.mass.set(eid, mass);
    state.add(eid, Part);
    Part.mesh.set(eid, Meshes.id("cube") ?? 0);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

let dropEid = -1;
let stackLowerEid = -1;
let stackUpperEid = -1;
let hullgEid = -1;
let restMirror: Mirror | null = null;

const scenario: Scenario = {
    name: "sat",
    params: [],

    async build(_canvas, _p: Params) {
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
                AvbdPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ],
        });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);

        // match the oracle's iters=10 (the plugin ships 6) so the rests are comparable to the f64 spec
        Avbd.step?.configure({
            dt: 1 / 60,
            gravity: -10,
            alpha: 0.99,
            penalty: PENALTY_MIN,
            betaLin: 1e4,
            betaAng: 100,
            gamma: 0.999,
            iterations: 10,
            maxColors: 8,
        });

        // a unit box-hull dropped on a static box ground (box × hull through the pipeline)
        spawnBox(state, [DROP_X, 0, 0], [4, 0.5, 4], 0, [0.3, 0.32, 0.36]);
        dropEid = spawnBoxHull(state, [DROP_X, 2, 0], [1, 1, 1], 1, [0.5, 0.7, 0.9]);

        // two box-hulls stacking on a box ground (hull × hull + box × hull)
        spawnBox(state, [STACK_X, 0, 0], [3, 0.5, 3], 0, [0.3, 0.32, 0.36]);
        stackLowerEid = spawnBoxHull(state, [STACK_X, 1.2, 0], [1, 1, 1], 1, [0.85, 0.6, 0.55]);
        stackUpperEid = spawnBoxHull(state, [STACK_X, 2.4, 0], [1, 1, 1], 1, [0.55, 0.6, 0.85]);

        // a box-hull resting on a static box-hull ground (hull × hull, both convex hulls). The ground is
        // [3,2,3] (top at y = 1.0), so the rest (1.49) is distinct from the box-ground drop above.
        spawnBoxHull(state, [HULLG_X, 0, 0], [3, 2, 3], 0, [0.3, 0.32, 0.36]);
        hullgEid = spawnBoxHull(state, [HULLG_X, 2, 0], [1, 1, 1], 1, [0.5, 0.75, 0.55]);

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.distance.set(cam, 20);

        await frames(3);
        if (Avbd.step) restMirror = mirror(Avbd.step.bodies);
        await frames(420); // settle every rest

        return {
            state,
            dispose() {
                restMirror?.dispose();
                restMirror = null;
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        const checks: Check[] = [];
        checks.push({ name: "hardware", pass: true, detail: await adapterInfo() });

        const results = await runSat();
        const configs = gold.configs as GoldConfig[];
        let maxErr = 0;
        for (let i = 0; i < configs.length; i++) {
            const cfg = configs[i];
            const { detail, err } = diff(cfg, results[i]);
            maxErr = Math.max(maxErr, err);
            checks.push({
                name: cfg.name,
                pass: detail === "",
                detail:
                    detail ||
                    `${results[i].count} contacts, all features + arms match (max err ${err.toExponential(2)})`,
            });
        }
        checks.push({
            name: "max-abs-error (TOL 1e-4)",
            pass: maxErr < TOL,
            detail: `${maxErr.toExponential(2)}`,
        });

        // wg(32) single-active-lane reference: the SAT body is correct on Metal under this shape. A
        // failure here would mean the SAT math itself is broken on this target; a pass while the main
        // 14-config kernel fails pins the divergence to multi-lane execution.
        const ref = await runRef();
        const wantFeats = [0x00010100, 0x00010101, 0x00010102, 0x00010103];
        const wantRA: [number, number, number][] = [
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ];
        let refOk = ref.count === 4;
        let refDelta = 0;
        for (let k = 0; k < 4 && refOk; k++) {
            if (ref.entries[k].feat !== wantFeats[k]) refOk = false;
            for (let i = 0; i < 3; i++) {
                const d = Math.abs(ref.entries[k].rA[i] - wantRA[k][i]);
                refDelta = Math.max(refDelta, d);
                if (d >= TOL) refOk = false;
            }
        }
        checks.push({
            name: "ref-kernel (workgroup_size 32, single active lane, face-y-overlap)",
            pass: refOk,
            detail: refOk
                ? `count=4, all features + rA match (max err ${refDelta.toExponential(2)})`
                : `count=${ref.count}; first contact feat 0x${(ref.entries[0].feat >>> 0).toString(16).padStart(8, "0")}`,
        });

        // ── hull SAT gate (Phase 6.3) — the production hull dispatch vs the f64 oracle narrowphase ──
        const hcfgs = hullConfigs();
        const hres = await runHullKernel(hcfgs);
        let hMaxErr = 0;
        for (let i = 0; i < hcfgs.length; i++) {
            const { detail, err } = diffHull(hcfgs[i], hres[i]);
            hMaxErr = Math.max(hMaxErr, err);
            checks.push({
                name: `hull: ${hcfgs[i].name}`,
                pass: detail === "",
                detail:
                    detail ||
                    `${hres[i].count} contact(s), normal + arms match (err ${err.toExponential(2)})`,
            });
        }
        checks.push({
            name: `hull kernel max-abs-error (TOL ${HULL_TOL})`,
            pass: hMaxErr < HULL_TOL,
            detail: hMaxErr.toExponential(2),
        });

        // ── full-pipeline hull rests — the step.ts wiring (seed hullId → hullData upload → dispatch → solve) ──
        if (!restMirror)
            return [...checks, { name: "hull rests", pass: false, detail: "no physics step" }];
        await settle(restMirror);
        const snap = restMirror.snapshot;
        if (!snap) return [...checks, { name: "hull rests", pass: false, detail: "no snapshot" }];
        const s = new Float32Array(snap.bytes);
        const cap = s.length / (BODY_VEC4 * 4);
        const posY = (eid: number): number => s[(0 * cap + eid) * 4 + 1];
        const speed = (eid: number, col: number): number => {
            const o = (col * cap + eid) * 4;
            return Math.hypot(s[o], s[o + 1], s[o + 2]);
        };
        const settled = (eid: number, restY: number, tol: number): boolean =>
            Number.isFinite(posY(eid)) &&
            Math.abs(posY(eid) - restY) < tol &&
            speed(eid, 6) < 5e-2 && // B_VELL
            speed(eid, 7) < 1e-1; // B_VELA
        const restDetail = (eid: number, restY: number): string =>
            `y ${posY(eid).toFixed(3)} (rest ${restY.toFixed(3)}), lin ${speed(eid, 6).toExponential(2)}, ang ${speed(eid, 7).toExponential(2)}`;

        // a unit box-hull rests at ground-top (0.5) + half (0.5) − margin (box × hull pipeline)
        const dropRest = 0.5 + 0.5 - MARGIN;
        checks.push({
            name: "hull: box-hull settles on a box ground (box × hull pipeline)",
            pass: settled(dropEid, dropRest, 0.03),
            detail: restDetail(dropEid, dropRest),
        });
        // two box-hulls stack: lower ≈ 1.0, upper ≈ 2.0 (hull × hull + box × hull)
        checks.push({
            name: "hull: lower box-hull in the stack",
            pass: settled(stackLowerEid, 1.0, 0.03),
            detail: restDetail(stackLowerEid, 1.0),
        });
        checks.push({
            // the upper box of a 2-stack settles a hair low (the margin compresses each contact ~mg/k);
            // the oracle rest is ~1.95–1.97, so the band is generous (f32 + a 420-frame settle).
            name: "hull: upper box-hull in the stack",
            pass: settled(stackUpperEid, 1.96, 0.06),
            detail: restDetail(stackUpperEid, 1.96),
        });
        // a box-hull resting on a box-hull ground (hull × hull, both convex): ground-top (1.0) + half − margin
        const hullgRest = 1.0 + 0.5 - MARGIN;
        checks.push({
            name: "hull: box-hull settles on a box-hull ground (hull × hull pipeline)",
            pass: settled(hullgEid, hullgRest, 0.03),
            detail: restDetail(hullgEid, hullgRest),
        });

        return checks;
    },

    live(): string {
        return [
            "sat",
            "real-GPU SAT vs C++ gold vectors",
            "(headless: bun bench --scenario sat)",
        ].join("\n");
    },
};

register(scenario);
