import { globals } from "bun-webgpu";
import { type Check, queryFlags, skipReason, verify } from "./verify";

// SMALL_N's module graph (avbd/step → physics/core → sear) references GPU enum constants at module
// scope, which exist in a browser but not under bare `bun run` — install bun-webgpu's constants-only
// globals (no adapter) before evaluating it, the same shim `bun test` preloads.
globals();
const { SMALL_N } = await import("../packages/shallot/src/standard/avbd/step");

// physics-bench — the standing perf + scaling-robustness surface for the AVBD solver. Drives the §6 gym
// physics scenarios headless, one isolated page per cell (fresh vite server + GPU), and reads each scenario's
// `measured` reporter (a structured `Check.data` payload — the per-step GPU spans, the dispatch count, the
// store size, the fall-through signals). Three outputs:
//
//   • the MATRIX — the `pile` scenario over count × ground × layout × boundary × collider → the per-step GPU
//     pass breakdown + dispatch count + memory, the repeatable surface the solve-perform work reads. The
//     spans are the `measured` per-step snapshot (Profile.gpu ≈ one physics step), NOT the window-mean
//     metrics.gpu.passes (inflated + non-monotonic by the drain/greedy-hold cadence). Characterize from per-step.
//   • the FALL-THROUGH guard — sweep `pile` count and assert no pile tunnels. Keys on the IMMEDIATE signals
//     (throughFloor = bodies below the surface over the footprint, staticDrop = a ground support evicted),
//     with minY as a lagging secondary. RED if a change reintroduces a fall-through.
//   • the SCENARIOS row — the `constraints` + `character` scenarios once each (single parametric scenes, no
//     count sweep): their per-step spans incl. the joint / character passes.
//
//   • the AUDIT (--audit, roadmap "Physics — the structure tax" Phase A) — replaces the three outputs
//     above with the waste-audit tax table: a pile count sweep at the SHIPPED iters (default 4),
//     min-over-`--reps` per span, against a derived per-pass theoretical floor (bytes/BW + flops/rate)
//     and the per-phase boundary constant measured by the gym `chain` microbench (N dependent empty
//     dispatches vs the same chain as in-kernel storageBarrier()s). Every pass gets: measured µs, phase
//     count, structure prediction (phases × boundary constant), floor µs, and the tax (measured − floor).
//
// Run: bun run scripts/physics-bench.ts
//      [--counts 512,2048] [--grounds 0,200] [--layers 4] [--gap 1.1] [--layouts grid,heap,pyramid]
//      [--boundaries flat,drum] [--shapes box,sphere,capsule,hull] [--joints false,true]
//      [--scale-counts 2048,8192,50000] [--warmup 60] [--frames 120] [--timeout 300000]
//      [--audit] [--audit-counts 64,256,1024,4096,16384,50000] [--reps 3] [--iters 4]
//      [--phases 70] [--bw 1008] [--tflops 82.6] [--ldsN 0] [--smallN 0]
// `--layouts` sweeps the pile layout (grid / heap / pyramid — different contact regimes); `--boundaries` the
// container (flat ground / drum pen — a drum holds a rounded pile so it settles). `--shapes` sweeps the
// collider (the narrowphase under test); `--joints true` chains the rows (the joint passes). A non-box /
// jointed cell is perf + no-tunnel — the box correctness suite gates only on box.

// the solver-core passes (the matrix's `step` subtotal). `phys:joint` is 0 unless the cell chains joints. The
// full per-step set (incl. bvh.* / aabb / coloring / pack) lands in PASS × COUNT below.
const CORE = [
    "phys:broadphase",
    "phys:csr",
    "phys:collide",
    "phys:joint",
    "phys:inertial",
    "phys:primal",
    "phys:dual",
    "phys:velocity",
    "phys:compose",
] as const;
// a contained pile rests at y≈1 and sinks to ~0.7 while unsettled; a fall-through explodes to y≪0. −1
// cleanly separates "contained" from "fell through the floor" (the lagging signal — staticDrop/throughFloor
// fire the frame it happens).
const FLOOR = -1;

interface Cell {
    count: number;
    ground: number;
    layers: number;
    gap: number;
    layout: string; // grid (dense) | heap (random-yaw drop) | pyramid (a stacked-box-pyramid that holds)
    boundary: string; // flat (the single ground) | drum (a static wall pen that holds a rounded pile)
    shape: string; // the collider: box | sphere | capsule | hull (the narrowphase under test)
    joints: boolean; // chain the pile rows with spherical joints (exercises the joint passes)
}

interface CellResult {
    cell: Cell;
    data: Record<string, number>;
    step: number; // the solver-core subtotal (ms)
    error?: string;
}

function arg(name: string, fallback: string): string {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}
const nums = (s: string): number[] =>
    s
        .split(",")
        .map((x) => Number(x.trim()))
        .filter((x) => Number.isFinite(x));
const strs = (s: string): string[] =>
    s
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);

const WARMUP = Number(arg("warmup", "60"));
const FRAMES = Number(arg("frames", "120"));
const TIMEOUT = Number(arg("timeout", "300000"));

// the pile scenario's query string for a cell. viz stays on (the scenario default): render overhead is tiny,
// the phys:* spans are isolated GPU timestamps so it doesn't perturb the measurement, and it's watchable live.
function query(cell: Cell): string {
    return [
        "scenario=pile",
        `count=${cell.count}`,
        `ground=${cell.ground}`,
        `layers=${cell.layers}`,
        `gap=${cell.gap}`,
        `layout=${cell.layout}`,
        `boundary=${cell.boundary}`,
        `shape=${cell.shape}`,
        `joints=${cell.joints}`,
    ].join("&");
}

// the `measured` reporter's structured payload — the per-step spans + dispatch/memory/fall-through signals.
const measured = (checks: Check[]): Record<string, number> | undefined =>
    checks.find((c) => c.name.startsWith("measured"))?.data;

async function runUrl(
    query: string,
): Promise<{ data: Record<string, number>; ok: boolean; error?: string; checks: Check[] }> {
    try {
        // one isolated `shallot verify` per cell (fresh browser + GPU + port, verify picks its own). The
        // scenario knobs ride `--query`; warmup/frames become gym params the harness run() coerces. --memory
        // samples the retained-leak slope (informational). hardware + memory land in the JSON if a caller wants.
        const res = await verify(
            "examples/gym",
            [
                ...queryFlags([...query.split("&"), `warmup=${WARMUP}`, `frames=${FRAMES}`]),
                "--memory",
                "--timeout",
                String(TIMEOUT),
            ],
            true, // quiet — the sweep table is the output, not per-cell JSON blobs
        );
        if (!res)
            return { data: {}, ok: false, error: "no JSON result from shallot verify", checks: [] };
        const checks = res.verdict?.checks ?? [];
        const data = measured(checks) ?? {};
        return {
            data,
            ok: res.pass,
            error: res.pass ? undefined : (res.error ?? res.errors?.[0] ?? "run not ok"),
            checks,
        };
    } catch (e) {
        return {
            data: {},
            ok: false,
            error: e instanceof Error ? e.message : String(e),
            checks: [],
        };
    }
}

async function runCell(cell: Cell): Promise<CellResult> {
    const { data, ok, error } = await runUrl(query(cell));
    const step = CORE.reduce((s, n) => s + (data[n] ?? 0), 0);
    return { cell, data, step, error: ok ? undefined : error };
}

// a pile fell through when a body sank below the surface over the footprint (throughFloor), a ground support
// was evicted (staticDrop), or — the lagging signal — minY dropped past the floor.
const fellThrough = (r: CellResult): boolean =>
    (r.data.throughFloor ?? 0) > 0 || (r.data.staticDrop ?? 0) > 0 || (r.data.minY ?? 0) < FLOOR;

const groundLabel = (g: number): string => (g === 0 ? "auto" : String(g));
const sceneLabel = (c: Cell): string => `${c.layout}/${c.boundary}`; // pile layout + container
const colliderLabel = (c: Cell): string => c.shape + (c.joints ? "+j" : ""); // the collider (+ joints)
const col = (s: string): string => s.padStart(9);
const ms = (x: number | undefined): string =>
    x === undefined ? "        —" : x.toFixed(3).padStart(9);
const mb = (bytes: number | undefined): string =>
    bytes === undefined ? "        —" : (bytes / (1 << 20)).toFixed(1).padStart(9);

// the full per-pass surface, rows=pass × cols=count — the format the cross-device bound characterization
// reads (overhead-bound is flat across count, BW/ALU scales). Covers every per-step span. Run a single
// shape+ground so the columns are an unambiguous count sweep.
function printPasses(rows: CellResult[]): void {
    const keys = new Set<string>();
    for (const r of rows) for (const k of Object.keys(r.data)) if (k.includes(":")) keys.add(k);
    const cols = [...keys].sort();
    const w = (s: string): string => s.padStart(13);
    console.log(
        "\nPASS × COUNT — per-step physics GPU span (ms) — overhead-bound is flat, BW/ALU scales",
    );
    console.log([w("pass"), ...rows.map((r) => w(String(r.cell.count)))].join(" "));
    for (const k of cols)
        console.log([w(k), ...rows.map((r) => w(ms(r.data[k]).trim()))].join(" "));
}

function printMatrix(rows: CellResult[]): void {
    console.log("\nMATRIX — per-step GPU pass breakdown (ms, solver core) + dispatch + memory");
    console.log(
        [
            col("scene"),
            col("collider"),
            col("count"),
            col("ground"),
            col("step"),
            ...CORE.map((p) => col(p.replace("phys:", ""))),
            col("colors"),
            col("MB"),
        ].join(" "),
    );
    for (const r of rows) {
        const line = [
            col(sceneLabel(r.cell)),
            col(colliderLabel(r.cell)),
            col(String(r.cell.count)),
            col(groundLabel(r.cell.ground)),
            ms(r.step),
            ...CORE.map((p) => ms(r.data[p])),
            col(String(r.data.dispatchedColors ?? "—")),
            mb(r.data.bytes),
        ].join(" ");
        console.log(line + (r.error ? `  ✗ ${r.error}` : ""));
    }
}

function printScale(rows: CellResult[]): void {
    console.log("\nFALL-THROUGH guard — count sweep");
    console.log(
        [
            col("scene"),
            col("count"),
            col("step"),
            col("minY"),
            col("through"),
            col("staticDrop"),
            col("verdict"),
        ].join(" "),
    );
    for (const r of rows) {
        const verdict = r.error ? `✗ ${r.error}` : fellThrough(r) ? "✗ FELL THROUGH" : "ok";
        console.log(
            [
                col(sceneLabel(r.cell)),
                col(String(r.cell.count)),
                ms(r.step),
                col(r.data.minY?.toFixed(2) ?? "—"),
                col(String(r.data.throughFloor ?? "—")),
                col(String(r.data.staticDrop ?? "—")),
                col(verdict),
            ].join(" "),
        );
    }
}

async function sweep(label: string, cells: Cell[]): Promise<CellResult[]> {
    const rows: CellResult[] = [];
    for (const cell of cells) {
        process.stdout.write(
            `  ${label} ${sceneLabel(cell)} ${colliderLabel(cell)} count=${cell.count} ground=${groundLabel(cell.ground)} … `,
        );
        const r = await runCell(cell);
        console.log(
            r.error
                ? `ERROR ${r.error}`
                : `step ${r.step.toFixed(3)} ms, minY ${r.data.minY?.toFixed(2) ?? "—"}`,
        );
        rows.push(r);
    }
    return rows;
}

// the constraints + character scenarios — constraints is a single parametric scene; character sweeps
// `filler` (inert far statics) so the solver core stays flat across world body count, and runs the `probe`
// cell — the latency regression gate that the CPU character sweep keeps position input→camera off the GPU
// readback (the probe self-calibrates a GPU-load knob, then gates pose latency == orientation under load).
async function sweepScenarios(): Promise<
    { name: string; data: Record<string, number>; step: number; error?: string }[]
> {
    const out: { name: string; data: Record<string, number>; step: number; error?: string }[] = [];
    const cells = [
        { name: "constraints", query: "scenario=constraints" },
        {
            // regime-cross: thresholds at the pre-spawn live count (63 boxes + ground = 64), so the
            // scenario's spawn-despawn flow crosses BOTH the smallN and ldsN gates up and back down —
            // the pile's regime-cross check is the gate (it fails loud if a flip misses).
            name: "regime-cross",
            query: `${query({
                count: 63,
                ground: 0,
                layers: 4,
                gap: 1.1,
                layout: "grid",
                boundary: "flat",
                shape: "box",
                joints: false,
            })}&smallN=64&ldsN=64`,
        },
        ...nums(arg("fillers", "0,256,4096,16384")).map((f) => ({
            name: `character f${f}`,
            query: `scenario=character&filler=${f}`,
        })),
        // the input→camera latency regression gate (roadmap "CPU character controller"): under a calibrated
        // GPU load, the position pose updates same-frame as orientation — no readback in the position path.
        { name: "character probe", query: "scenario=character&probe=1" },
    ];
    for (const cell of cells) {
        process.stdout.write(`  scenario ${cell.name} … `);
        const { data, ok, error, checks } = await runUrl(cell.query);
        const step = CORE.reduce((s, n) => s + (data[n] ?? 0), 0);
        // scenario rows are gate rows — a failing scenario check (e.g. regime-cross) reddens the bench
        const failed = checks.find((c) => !c.ok);
        const err = !ok
            ? error
            : failed
              ? `check failed: ${failed.name} — ${failed.detail}`
              : undefined;
        console.log(err ? `ERROR ${err}` : `step ${step.toFixed(3)} ms`);
        out.push({ name: cell.name, data, step, error: err });
    }
    return out;
}

function printScenarios(
    rows: { name: string; data: Record<string, number>; step: number; error?: string }[],
): void {
    console.log("\nSCENARIOS — per-step GPU solver-core breakdown (ms)");
    console.log(
        [
            col("scenario"),
            col("step"),
            ...CORE.map((p) => col(p.replace("phys:", ""))),
            col("MB"),
        ].join(" "),
    );
    for (const r of rows) {
        const line = [
            col(r.name),
            ms(r.step),
            ...CORE.map((p) => ms(r.data[p])),
            mb(r.data.bytes),
        ].join(" ");
        console.log(line + (r.error ? `  ✗ ${r.error}` : ""));
    }
}

// ── waste audit (--audit) — the Phase A structure-tax accounting ─────────────────────────────────────
//
// Per pass: measured span (min over --reps) vs a derived theoretical floor, plus a structure prediction
// from the phase count × the boundary constant the `chain` microbench measures. The floor is
// max(bytes/BW, flops/rate) with peak-DRAM BW — when a working set fits L2 the true floor is lower, so
// the printed tax is a LOWER bound on the waste. The byte model is derived from the kernels (per-lane
// column traffic; comments name the buffers counted) and is ±2×-class, not exact: it ranks passes and
// sizes the tax, it doesn't certify a kernel optimal.

interface AuditCtx {
    n: number; // live bodies (pile + ground + the spawn-despawn extra)
    slots: number; // pair slots = n × PAIRS_PER_BODY(8)
    contacts: number; // live contact records (counters[6], warmstarted ≈ all on a settled pile)
    colors: number; // dispatchedColors (the readback-bounded primal loop)
    iters: number;
    cap: number; // eid capacity (pile sizes it max(1024, count+16))
    // the broadphase regime (n ≤ SMALL_N — the cells run the default threshold): the BVH passes don't
    // run, the broadphase is the one-dispatch O(n²) tile scan, and the CSR + coloring tail is the C1.1
    // fused single-WG dispatch. The phase/byte models follow it so the struct prediction stays honest.
    small: boolean;
}

// bvh bounds-relaxation sweep count (bvh/build.ts: heightBound min(64, 30+log2(cap)), LEVELS=3)
const sweeps = (cap: number): number =>
    Math.ceil(Math.min(64, 30 + Math.ceil(Math.log2(Math.max(2, cap)))) / 3);
const lg = (n: number): number => Math.log2(Math.max(2, n));

// every span in record() order: phases = serially-dependent dispatches inside the span; bytes/flops =
// the derived main-memory traffic + arithmetic for the whole span. Contact records are CONTACT_VEC4=7
// vec4 = 112 B each; body columns 16 B each (BODY_VEC4=14); pairList entries vec2<u32> = 8 B.
const AUDIT_PASSES: {
    name: string;
    phases(c: AuditCtx): number;
    bytes(c: AuditCtx): number;
    flops(c: AuditCtx): number;
}[] = [
    {
        // pack count→scan→scatter (C1.3): membership read twice (count + scatter), dense eid map written;
        // the scan's lane 0 carries the fused clamp+publish
        name: "phys:pack",
        phases: () => 3,
        bytes: (c) => 8 * c.cap + 8 * c.n,
        flops: (c) => 2 * c.cap,
    },
    {
        // per body: eid 4 + pos/quat/half/round cols 64 read, prim AABB 32 written
        name: "phys:aabb",
        phases: () => 1,
        bytes: (c) => c.n * 100,
        flops: (c) => c.n * 80,
    },
    {
        // grid-stride reduce over prims (32 B each) + a 1-WG finalize. The small regime skips the BVH.
        name: "bvh:bounds",
        phases: (c) => (c.small ? 0 : 2),
        bytes: (c) => (c.small ? 0 : c.n * 32),
        flops: (c) => (c.small ? 0 : c.n * 10),
    },
    {
        // per prim: AABB 32 read, key 4 + payload 4 written. The small regime skips the BVH.
        name: "bvh:morton",
        phases: (c) => (c.small ? 0 : 1),
        bytes: (c) => (c.small ? 0 : c.n * 40),
        flops: (c) => (c.small ? 0 : c.n * 40),
    },
    {
        // Decoupled Fallback: prepare + init + globalHist + scan + 4 binning (sort.ts sortIndirect).
        // Traffic ≈ hist read 4 + init 16 + 4 × (key+payload read 8, write 8) per key. Small regime: skipped.
        name: "bvh:sort",
        phases: (c) => (c.small ? 0 : 8),
        bytes: (c) => (c.small ? 0 : c.n * 84),
        flops: (c) => (c.small ? 0 : c.n * 60),
    },
    {
        // prepare + leaf + topo + sweeps(cap) relaxation dispatches (build.ts). leaf 72 B/prim; topo's
        // binary range search reads ~2·log2(n) keys + writes 32; each sweep reads ~4 nodes + writes 1.
        // Small regime: skipped.
        name: "bvh:build",
        phases: (c) => (c.small ? 0 : 3 + sweeps(c.cap)),
        bytes: (c) => (c.small ? 0 : c.n * (72 + 8 * lg(c.n) + 32) + sweeps(c.cap) * c.n * 160),
        flops: (c) => (c.small ? 0 : c.n * 40 * lg(c.n) + sweeps(c.cap) * c.n * 30),
    },
    {
        // BVH regime — per body: pose cols 64 read + descent visits ~2·(K + log2 n) nodes × 32 B + pair
        // block 64 written. Small regime — the O(n²) tile scan: each of n/64 workgroups stages all n
        // prims (32 B) once, every lane tests n staged boxes (~10 flops each) + block 64 written.
        name: "phys:broadphase",
        phases: () => 1,
        bytes: (c) =>
            c.small ? (c.n * c.n * 32) / 64 + c.n * 128 : c.n * (64 + 64 * (8 + lg(c.n)) + 64),
        flops: (c) => (c.small ? c.n * c.n * 10 : c.n * (8 + lg(c.n)) * 60),
    },
    {
        // 4 class-gated pipelines over the slots (each reads the 8 B pairList entry to gate); per live
        // contact: prev record 112 read (warmstart match) + 112 written + amortized pose reads ~64
        name: "phys:collide",
        phases: () => 4,
        bytes: (c) => c.slots * 8 * 4 + c.contacts * 288,
        flops: (c) => c.contacts * 800,
    },
    {
        // BVH regime — count + scan + scatter (buildCsr): slot meta reads ×2 passes + per-pair csrList
        // entries. Small regime — the C1.1 fused single-WG tail (CSR_COLOR_SMALL_WGSL) does the same
        // traffic PLUS the greedy coloring's (reported here; phys:coloring reads 0).
        name: "phys:csr",
        phases: (c) => (c.small ? 1 : 3),
        bytes: (c) =>
            c.slots * 20 + c.n * 8 + c.contacts * 4 + (c.small ? c.n * 12 + c.contacts * 8 : 0),
        flops: (c) => c.slots * 4 + (c.small ? c.n * 30 : 0),
    },
    {
        // one greedy sweep (no joints in the audit cells): per body read neighbor colors via CSR, write 1.
        // Small regime: folded into the fused phys:csr dispatch above.
        name: "phys:coloring",
        phases: (c) => (c.small ? 0 : 1),
        bytes: (c) => (c.small ? 0 : c.n * 12 + c.contacts * 8),
        flops: (c) => (c.small ? 0 : c.n * 30),
    },
    {
        // per body: pos/quat/vel cols read, inertial target + x⁻ snapshot cols written
        name: "phys:inertial",
        phases: () => 1,
        bytes: (c) => c.n * 160,
        flops: (c) => c.n * 120,
    },
    {
        // iters × colors × (primal + commit). Per iteration each body solves once across the colors
        // (own pose 64 + CSR 8 + per touching contact: record λ/c0 112 + partner pose 32) and commits
        // (solveOut 64); every dispatch also reads 8 B eids per lane (colors × 2 × n lane overhead)
        name: "phys:primal",
        phases: (c) => c.iters * c.colors * 2,
        bytes: (c) => c.iters * (c.n * 136 + 2 * c.contacts * 144 + c.colors * 2 * c.n * 8),
        flops: (c) => c.iters * (c.n * 600 + 2 * c.contacts * 400),
    },
    {
        // one dispatch per iteration over the slots: 16 B meta gate + per live contact record 112 read,
        // λ/penalty ~48 written, endpoint poses ~64
        name: "phys:dual",
        phases: (c) => c.iters,
        bytes: (c) => c.iters * (c.slots * 16 + c.contacts * 224),
        flops: (c) => c.iters * c.contacts * 300,
    },
    {
        // BDF1 recovery: pose + x⁻ cols read, vel cols written
        name: "phys:velocity",
        phases: () => 1,
        bytes: (c) => c.n * 128,
        flops: (c) => c.n * 80,
    },
    {
        // pose + prev cols read, interpolated mat4 (64 B) written into the transforms firehose
        name: "phys:compose",
        phases: () => 1,
        bytes: (c) => c.n * 160,
        flops: (c) => c.n * 100,
    },
];

// min-over-reps per span (the per-step snapshot occasionally catches a 2-step stack); colors/contacts take the max (structural, not timing, so the stack can't inflate them).
async function auditCell(query: string, reps: number): Promise<Record<string, number> | null> {
    let out: Record<string, number> | null = null;
    for (let r = 0; r < reps; r++) {
        const { data, ok, error } = await runUrl(query);
        if (!ok) {
            console.log(`    rep ${r + 1} ERROR ${error}`);
            continue;
        }
        if (!out) {
            out = { ...data };
        } else {
            for (const k of Object.keys(data)) {
                out[k] = k.includes(":")
                    ? Math.min(out[k] ?? Number.POSITIVE_INFINITY, data[k])
                    : Math.max(out[k] ?? 0, data[k]);
            }
        }
        process.stdout.write(`rep${r + 1} `);
    }
    return out;
}

const us = (x: number): string => (x >= 100 ? x.toFixed(0) : x.toFixed(1)).padStart(9);

async function audit(): Promise<void> {
    const counts = nums(arg("audit-counts", "64,256,1024,4096,16384,50000"));
    const reps = Number(arg("reps", "3"));
    const iters = Number(arg("iters", "4")); // the SHIPPED value — the gameplay-budget framing
    const ldsN = arg("ldsN", ""); // "" = the shipped default; "0" = force the looped solve (C1.2 A/B)
    const smallN = arg("smallN", ""); // "" = the shipped default; "0" = force the BVH front-end (C1.0/C1.1 A/B)
    const phases = Number(arg("phases", "70"));
    const bw = Number(arg("bw", "1008")) * 1e9; // bytes/s (default: 4090 peak DRAM)
    const flopRate = Number(arg("tflops", "82.6")) * 1e12; // f32 FLOP/s (default: 4090 peak)

    console.log(
        `waste audit — counts {${counts.join(",")}} × ${reps} reps, iters=${iters}, ` +
            `chain phases=${phases}, floor at ${(bw / 1e9).toFixed(0)} GB/s + ${(flopRate / 1e12).toFixed(1)} TFLOP/s`,
    );

    // the boundary-constant microbenches: N dependent empty dispatches, and the same chain in-kernel
    process.stdout.write(`  chain phases=${phases} … `);
    const chain = await auditCell(`scenario=chain&phases=${phases}`, reps);
    console.log("");
    const tDispatch = chain?.dispatchUs ?? 0;
    const tBarrier = chain?.barrierUs ?? 0;
    console.log(
        `\nBOUNDARY CONSTANT — ${phases}-phase dependent chain (min over ${reps}):\n` +
            `  dispatch boundary ${tDispatch.toFixed(2)} µs/phase · in-kernel storageBarrier ` +
            `${tBarrier.toFixed(2)} µs/phase · launch overhead = ${(tDispatch - tBarrier).toFixed(2)} µs/phase`,
    );

    const cells: { count: number; data: Record<string, number> }[] = [];
    for (const count of counts) {
        process.stdout.write(`  audit count=${count} … `);
        const q = `${query({
            count,
            ground: 0,
            layers: 4,
            gap: 1.1,
            layout: "grid",
            boundary: "flat",
            shape: "box",
            joints: false,
        })}&iters=${iters}${ldsN === "" ? "" : `&ldsN=${ldsN}`}${smallN === "" ? "" : `&smallN=${smallN}`}`;
        const data = await auditCell(q, reps);
        console.log(data ? `step ok, contacts ${data.contacts ?? "—"}` : "ERROR (cell skipped)");
        if (data) cells.push({ count, data });
    }

    // the tax table, one block per count: every pass's measured span vs the structure prediction
    // (phases × the dispatch-boundary constant) and the derived floor. tax = measured − floor.
    for (const { count, data } of cells) {
        const n = data.bodies ?? count + 2; // pile + ground + the spawn-despawn extra
        const ctx: AuditCtx = {
            n,
            slots: n * 8,
            contacts: data.contacts ?? 4 * count,
            colors: Math.max(1, data.dispatchedColors ?? 8),
            iters,
            cap: Math.max(1024, count + 16),
            small: n <= SMALL_N,
        };
        console.log(
            `\nTAX — count ${count} (bodies ${ctx.n}, contacts ${ctx.contacts}, colors ${ctx.colors}, ` +
                `iters ${iters}) — µs per step`,
        );
        console.log(
            [
                "pass".padStart(16),
                "measured".padStart(9),
                "phases".padStart(7),
                "struct".padStart(9),
                "floor".padStart(9),
                "tax".padStart(9),
                "tax%".padStart(6),
            ].join(" "),
        );
        let mTot = 0;
        let pTot = 0;
        let fTot = 0;
        for (const p of AUDIT_PASSES) {
            const measured = (data[p.name] ?? 0) * 1000;
            const ph = p.phases(ctx);
            const floor = Math.max((p.bytes(ctx) / bw) * 1e6, (p.flops(ctx) / flopRate) * 1e6);
            const tax = Math.max(0, measured - floor);
            mTot += measured;
            pTot += ph;
            fTot += floor;
            console.log(
                [
                    p.name.padStart(16),
                    us(measured),
                    String(ph).padStart(7),
                    us(ph * tDispatch),
                    us(floor),
                    us(tax),
                    `${measured > 0 ? ((tax / measured) * 100).toFixed(0) : "—"}%`.padStart(6),
                ].join(" "),
            );
        }
        console.log(
            [
                "TOTAL".padStart(16),
                us(mTot),
                String(pTot).padStart(7),
                us(pTot * tDispatch),
                us(fTot),
                us(Math.max(0, mTot - fTot)),
                `${mTot > 0 ? (((mTot - fTot) / mTot) * 100).toFixed(0) : "—"}%`.padStart(6),
            ].join(" "),
        );
    }

    const green = cells.length === counts.length && tDispatch > 0;
    console.log(
        `\naudit: ${green ? "green (all cells + chain resolved)" : "has failures (see ✗)"}`,
    );
    if (!green) process.exit(1);
}

async function main(): Promise<void> {
    const skip = skipReason();
    if (skip) {
        console.log(`physics-bench needs native hardware (${skip}). Skipping.`);
        return;
    }
    if (process.argv.includes("--audit")) {
        await audit();
        return;
    }
    const counts = nums(arg("counts", "512,2048"));
    const grounds = nums(arg("grounds", "0,200"));
    const layers = Number(arg("layers", "4"));
    const gap = Number(arg("gap", "1.1"));
    const scaleCounts = nums(arg("scale-counts", "2048,8192,50000"));
    const layouts = strs(arg("layouts", "grid"));
    const boundaries = strs(arg("boundaries", "flat"));
    // the collider under test (the narrowphase): box (default) | sphere | capsule | hull. `--joints true`
    // chains the pile rows so the joint passes run. A non-box / jointed cell is a perf + no-tunnel cell — the
    // gym scenario gates the box correctness suite only on box (the tight rounded/hull/joint gate is the
    // sibling rounded/sat/constraints surfaces).
    const shapes = strs(arg("shapes", "box"));
    const joints = arg("joints", "false")
        .split(",")
        .map((s) => s.trim() === "true");

    const matrixCells: Cell[] = [];
    for (const count of counts)
        for (const ground of grounds)
            for (const layout of layouts)
                for (const boundary of boundaries)
                    for (const shape of shapes)
                        for (const jt of joints)
                            matrixCells.push({
                                count,
                                ground,
                                layers,
                                gap,
                                layout,
                                boundary,
                                shape,
                                joints: jt,
                            });
    // the fall-through guard stays box / grid / flat — it's the count-scaling pool-overflow probe (shape-agnostic);
    // the shape perf + the per-cell no-tunnel gate ride the matrix above.
    const scaleCells: Cell[] = [];
    for (const count of scaleCounts)
        scaleCells.push({
            count,
            ground: 0,
            layers,
            gap,
            layout: "grid",
            boundary: "flat",
            shape: "box",
            joints: false,
        });

    console.log(
        `physics-bench — matrix ${matrixCells.length} cells, scale guard ${scaleCells.length} cells, ` +
            `+ scenario rows (warmup ${WARMUP}, frames ${FRAMES})`,
    );

    const matrixRows = await sweep("matrix", matrixCells);
    printMatrix(matrixRows);
    printPasses(matrixRows);

    const scaleRows = await sweep("scale", scaleCells);
    printScale(scaleRows);

    const scenarioRows = await sweepScenarios();
    printScenarios(scenarioRows);

    const matrixGreen = matrixRows.length > 0 && matrixRows.every((r) => !r.error && r.step > 0);
    const guardPass = scaleRows.length > 0 && scaleRows.every((r) => !r.error && !fellThrough(r));
    // the probe cell is the latency gate, not a solver scene — it has no GPU solver step, so its verdict is
    // its checks (an `error` on a failed gate), exempt from the solver-cells' step>0 ran-sanity.
    const scenariosGreen = scenarioRows.every(
        (r) => !r.error && (r.step > 0 || r.name.includes("probe")),
    );

    console.log(
        `\nmatrix: ${matrixGreen ? "green (all cells ran + spans resolved)" : "has failures (see ✗)"}`,
    );
    console.log(
        `fall-through guard: ${guardPass ? "PASS (no pile fell through at any count)" : "FAIL — RED (a pile fell through; check pool sizing / overflow counters)"}`,
    );
    console.log(
        `scenarios: ${scenariosGreen ? "green (constraints + character ran)" : "has failures (see ✗)"}`,
    );

    if (!matrixGreen || !guardPass || !scenariosGreen) process.exit(1);
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});
