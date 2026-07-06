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
    PhysicsPlugin,
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
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import {
    BODY_VEC4,
    PENALTY_MIN,
    Physics,
    packHulls,
    registerHull,
} from "@dylanebert/shallot/physics/core";
import { Meshes } from "@dylanebert/shallot/render/core";
import {
    COLLIDE_WGSL,
    HULL_WGSL,
    MAX_CONTACTS,
} from "../../../../packages/shallot/src/standard/physics/collide";
import { boxHull, coneHull, tetHull } from "../../../../packages/shallot/tests/avbd/hull";
import {
    add,
    dot,
    type Quat,
    rotate,
    sub,
    type Vec3,
} from "../../../../packages/shallot/tests/avbd/math";
import {
    body,
    capsule,
    hull,
    type Body as OracleBody,
    sphere,
} from "../../../../packages/shallot/tests/avbd/rigid";
import { narrowphase } from "../../../../packages/shallot/tests/avbd/rounded";
import gold from "../../../../packages/shallot/tests/avbd/sat-gold-vectors.json";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";

// sat — the SAT gate on the real GPU. Runs the production `COLLIDE_WGSL` over the 14 C++ gold configs
// (sat-gold-vectors.json, the same spec sat.test.ts and the software-adapter cross-check use), reads
// back contact count + basis + feature key + manifold arms, diffs byte-exact (f32 tol 1e-4, ~5× margin
// derived from accumulated f32 rounding).
//
// The two passes:
//   "main-kernel"     — production-shape: workgroup_size(64), one thread per config, 14 active lanes.
//                       matches the narrowphase pass at standard/physics/step.ts:325.
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
    rA: [number, number, number];
    rB: [number, number, number];
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
const OUT_STRIDE = 1 + 9 + MAX_CONTACTS * 7; // count, basis(9), MAX × (feat, rA.xyz, rB.xyz)
const CFG_VEC4 = 7; // posA, quatA, sizeA, posB, quatB, sizeB, dRel

// The main SAT kernel — production shape (workgroup_size 64, one thread per config). Matches the
// standard/physics/step.ts narrowphase pass that calls collideBoxBox per broadphase candidate.
const KERNEL_WGSL = `${COLLIDE_WGSL}
struct Cfg { posA: vec4<f32>, quatA: vec4<f32>, sizeA: vec4<f32>, posB: vec4<f32>, quatB: vec4<f32>, sizeB: vec4<f32>, dRel: vec4<f32> };
@group(0) @binding(0) var<storage, read> cfgs: array<Cfg>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.x) { return; }
    let c = cfgs[i];
    let r = collideBoxBox(c.posA.xyz, c.quatA, c.sizeA.xyz, c.posB.xyz, c.quatB, c.sizeB.xyz, c.dRel.xyz);
    let base = i * params.y;
    out[base] = bitcast<f32>(r.count);
    out[base + 1u] = r.basis.r0.x; out[base + 2u] = r.basis.r0.y; out[base + 3u] = r.basis.r0.z;
    out[base + 4u] = r.basis.r1.x; out[base + 5u] = r.basis.r1.y; out[base + 6u] = r.basis.r1.z;
    out[base + 7u] = r.basis.r2.x; out[base + 8u] = r.basis.r2.y; out[base + 9u] = r.basis.r2.z;
    for (var k = 0u; k < ${MAX_CONTACTS}u; k = k + 1u) {
        let o = base + 10u + k * 7u;
        out[o] = bitcast<f32>(r.feat[k]);
        out[o + 1u] = r.rA[k].x; out[o + 2u] = r.rA[k].y; out[o + 3u] = r.rA[k].z;
        out[o + 4u] = r.rB[k].x; out[o + 5u] = r.rB[k].y; out[o + 6u] = r.rB[k].z;
    }
}`;

// Reference kernel — workgroup_size(32) (matches Apple's SIMD width) with a single active lane on the
// face-y-overlap config (A unit cube at origin, B unit cube at y=0.97, expected count=4). A single-lane
// control: it was already bit-correct on apple/metal-3 while the multi-lane main-kernel diverged, which
// pinned the (now-fixed) Phase 4.5 spill to the multi-lane execution shape rather than the SAT math.
const REF_OUT_LEN = 1 + MAX_CONTACTS * 4; // count + MAX × (feat, rA.xyz)
const REF_KERNEL_WGSL = `${COLLIDE_WGSL}
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

// run all 14 configs once through the production-shape kernel, decode the per-config SatResults.
async function runSat(): Promise<GpuResult[]> {
    const device = Compute.device;
    const configs = gold.configs as GoldConfig[];
    const n = configs.length;

    const cfgData = new Float32Array(n * CFG_VEC4 * 4);
    for (let i = 0; i < n; i++) {
        const c = configs[i];
        const o = i * CFG_VEC4 * 4;
        cfgData.set(c.a.pos, o + 0);
        cfgData.set(c.a.quat, o + 4);
        cfgData.set(c.a.size, o + 8);
        cfgData.set(c.b.pos, o + 12);
        cfgData.set(c.b.quat, o + 16);
        cfgData.set(c.b.size, o + 20);
        // velocity sweep input: dRel = (velA − velB)·dt (zero for the static configs)
        for (let k = 0; k < 3; k++) cfgData[o + 24 + k] = (c.a.vel[k] - c.b.vel[k]) * DT;
    }

    const cfgBuf = device.createBuffer({
        label: "sat-cfg",
        size: cfgData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outBuf = device.createBuffer({
        label: "sat-out",
        size: n * OUT_STRIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const paramBuf = device.createBuffer({
        label: "sat-params",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const readBuf = device.createBuffer({
        label: "sat-read",
        size: n * OUT_STRIDE * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(cfgBuf, 0, cfgData);
    device.queue.writeBuffer(paramBuf, 0, new Uint32Array([n, OUT_STRIDE, 0, 0]));

    const pipeline = await device.createComputePipelineAsync({
        label: "sat-main",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "sat-main-module", code: KERNEL_WGSL }),
            entryPoint: "main",
        },
    });
    const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cfgBuf } },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: paramBuf } },
        ],
    });

    const encoder = device.createCommandEncoder({ label: "sat-main-encoder" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, n * OUT_STRIDE * 4);
    device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const bytes = readBuf.getMappedRange().slice(0);
    readBuf.unmap();
    cfgBuf.destroy();
    outBuf.destroy();
    paramBuf.destroy();
    readBuf.destroy();

    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const out: GpuResult[] = [];
    for (let ci = 0; ci < n; ci++) {
        const base = ci * OUT_STRIDE;
        const count = u[base];
        const basis: number[] = [];
        for (let i = 0; i < 9; i++) basis.push(f[base + 1 + i]);
        const contacts: GpuResult["contacts"] = [];
        for (let k = 0; k < count; k++) {
            const o = base + 10 + k * 7;
            contacts.push({
                feature: u[o],
                rA: [f[o + 1], f[o + 2], f[o + 3]],
                rB: [f[o + 4], f[o + 5], f[o + 6]],
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
            module: device.createShaderModule({ label: "sat-ref-module", code: REF_KERNEL_WGSL }),
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
function diff(cfg: GoldConfig, got: GpuResult): { detail: string; err: number } {
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
// contact by world anchor (robust to a reduce-order / feature-key tie the SAT math doesn't pin). The hull
// geometry is packed by the production `packHulls` and bound as `hullData`.

const HULL_TOL = 2e-3; // f32 vs f64 over the SAT projection/clip across many verts — looser than the box 1e-4
const Z90H: Quat = [0, 0, Math.SQRT1_2, Math.SQRT1_2]; // +Y capsule axis → horizontal (−X)
const TET_ROT: Quat = [0, 0.24740396, 0, 0.96891242]; // the bullet gold tet-tet-rotated quat

let hullSeq = 0;
const reg = (geom: ReturnType<typeof boxHull>): number => registerHull(`sat-${hullSeq++}`, geom);
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

const H_OUT_STRIDE = 1 + 3 + MAX_CONTACTS * 7; // count, normal(3), MAX × (feat, rA.xyz, rB.xyz)
const H_CFG_STRIDE = 32; // 8 vec4: posA, quatA, sizeRadA, posB, quatB, sizeRadB, dRel, shapes(sA,sB,hA,hB)
const HULL_KERNEL_WGSL = `${COLLIDE_WGSL}${HULL_WGSL}
struct HCfg { posA: vec4<f32>, quatA: vec4<f32>, sizeRadA: vec4<f32>, posB: vec4<f32>, quatB: vec4<f32>, sizeRadB: vec4<f32>, dRel: vec4<f32>, shapes: vec4<f32> };
@group(0) @binding(0) var<storage, read> cfgs: array<HCfg>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@group(0) @binding(3) var<storage, read> hullData: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.x) { return; }
    let c = cfgs[i];
    let sA = u32(c.shapes.x); let sB = u32(c.shapes.y);
    let hA = u32(c.shapes.z); let hB = u32(c.shapes.w);
    let roundedA = (sA == 1u || sA == 2u); let roundedB = (sB == 1u || sB == 2u);
    // the production collide-pass dispatch (step.ts), validated here against the oracle narrowphase
    var r: SatResult;
    if (roundedA && roundedB) {
        r = collideRounded(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.sizeRadA.w, c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.sizeRadB.w, c.dRel.xyz);
    } else if (!roundedA && !roundedB) {
        if (sA == 0u && sB == 0u) {
            r = collideBoxBox(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.dRel.xyz);
        } else {
            r = collideHull(polyMake(sA, c.posA.xyz, c.quatA, c.sizeRadA.xyz, hA), polyMake(sB, c.posB.xyz, c.quatB, c.sizeRadB.xyz, hB), c.dRel.xyz);
        }
    } else {
        r = collideRoundedPolytope(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.sizeRadA.w, sA, hA, c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.sizeRadB.w, sB, hB, c.dRel.xyz);
    }
    let base = i * params.y;
    out[base] = bitcast<f32>(r.count);
    out[base + 1u] = r.basis.r0.x; out[base + 2u] = r.basis.r0.y; out[base + 3u] = r.basis.r0.z;
    for (var k = 0u; k < ${MAX_CONTACTS}u; k = k + 1u) {
        let o = base + 4u + k * 7u;
        out[o] = bitcast<f32>(r.feat[k]);
        out[o + 1u] = r.rA[k].x; out[o + 2u] = r.rA[k].y; out[o + 3u] = r.rA[k].z;
        out[o + 4u] = r.rB[k].x; out[o + 5u] = r.rB[k].y; out[o + 6u] = r.rB[k].z;
    }
}`;

interface GpuHull {
    count: number;
    normal: Vec3;
    contacts: { rA: Vec3; rB: Vec3 }[];
}

async function runHullKernel(cfgs: HCfg[]): Promise<GpuHull[]> {
    const device = Compute.device;
    const n = cfgs.length;
    const cfgData = new Float32Array(n * H_CFG_STRIDE);
    for (let i = 0; i < n; i++) {
        const c = cfgs[i];
        const o = i * H_CFG_STRIDE;
        cfgData.set([c.a.posLin[0], c.a.posLin[1], c.a.posLin[2], 0], o);
        cfgData.set(c.a.posAng, o + 4);
        cfgData.set([c.a.size[0], c.a.size[1], c.a.size[2], c.a.roundRadius], o + 8);
        cfgData.set([c.b.posLin[0], c.b.posLin[1], c.b.posLin[2], 0], o + 12);
        cfgData.set(c.b.posAng, o + 16);
        cfgData.set([c.b.size[0], c.b.size[1], c.b.size[2], c.b.roundRadius], o + 20);
        cfgData.set([c.dRel[0], c.dRel[1], c.dRel[2], 0], o + 24);
        cfgData.set([c.a.shape, c.b.shape, c.ha, c.hb], o + 28);
    }
    const hullData = packHulls();

    const cfgBuf = device.createBuffer({
        label: "hull-cfg",
        size: cfgData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outBuf = device.createBuffer({
        label: "hull-out",
        size: n * H_OUT_STRIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const paramBuf = device.createBuffer({
        label: "hull-params",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const hullBuf = device.createBuffer({
        label: "hull-data",
        size: hullData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const readBuf = device.createBuffer({
        label: "hull-read",
        size: n * H_OUT_STRIDE * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(cfgBuf, 0, cfgData);
    device.queue.writeBuffer(paramBuf, 0, new Uint32Array([n, H_OUT_STRIDE, 0, 0]));
    device.queue.writeBuffer(hullBuf, 0, hullData as Uint32Array<ArrayBuffer>);

    const pipeline = await device.createComputePipelineAsync({
        label: "hull-kernel",
        layout: "auto",
        compute: {
            module: device.createShaderModule({ label: "hull-module", code: HULL_KERNEL_WGSL }),
            entryPoint: "main",
        },
    });
    const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cfgBuf } },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: paramBuf } },
            { binding: 3, resource: { buffer: hullBuf } },
        ],
    });
    const enc = device.createCommandEncoder({ label: "hull-enc" });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, n * H_OUT_STRIDE * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const bytes = readBuf.getMappedRange().slice(0);
    readBuf.unmap();
    for (const b of [cfgBuf, outBuf, paramBuf, hullBuf, readBuf]) b.destroy();

    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const out: GpuHull[] = [];
    for (let i = 0; i < n; i++) {
        const b = i * H_OUT_STRIDE;
        const count = u[b];
        const contacts: GpuHull["contacts"] = [];
        for (let k = 0; k < count; k++) {
            const oo = b + 4 + k * 7;
            contacts.push({
                rA: [f[oo + 1], f[oo + 2], f[oo + 3]],
                rB: [f[oo + 4], f[oo + 5], f[oo + 6]],
            });
        }
        out.push({ count, normal: [f[b + 1], f[b + 2], f[b + 3]], contacts });
    }
    return out;
}

// diff a GPU hull result against the oracle narrowphase: count, the B→A normal, then each oracle contact's
// arms matched to the nearest GPU contact by world anchor (xA+xB distance) — order/feature-key independent.
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
    const oraclePts = contacts.map((c) => ({ a: xA(cfg.a, c.rA), b: xB(cfg.b, c.rB) }));
    const gpuPts = got.contacts.map((c) => ({ a: xA(cfg.a, c.rA), b: xB(cfg.b, c.rB) }));
    for (const op of oraclePts) {
        let best = gpuPts[0];
        let bestD = Infinity;
        for (const gp of gpuPts) {
            const dd =
                dot(sub(gp.a, op.a), sub(gp.a, op.a)) + dot(sub(gp.b, op.b), sub(gp.b, op.b));
            if (dd < bestD) {
                bestD = dd;
                best = gp;
            }
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
// the WHOLE PhysicsPlugin (seed reads hullId from halfExtents.w → uploads the packed Hulls → the collide
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
    const id = registerHull(`rest-${hullRest++}`, boxHull(fullSize));
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
                PhysicsPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ],
        });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);

        // match the oracle's iters=10 (the plugin ships 6) so the rests are comparable to the f64 spec
        Physics.step?.configure({
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
        if (Physics.step) restMirror = mirror(Physics.step.bodies);
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
