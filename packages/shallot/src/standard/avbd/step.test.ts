import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as m from "../../../tests/avbd/math";
import {
    body,
    flat,
    integerDiscipline,
    noIntegerDivision,
    pointerDiscipline,
} from "../../../tests/wgsl";
import { Compute, precompile, precompileAll, requestGPU } from "../../engine/runtime";
import {
    JOINT_GROUP,
    LDS_IO_GROUP,
    PhysicsStep,
    PRIMAL_COLOR_GROUP,
    SHARED_STORAGE,
    SOLVER_GROUP,
    Step,
    solverRoRo,
    solverVariants,
    stepWgsl,
} from "./step";

// Structural gates over the step's emitted WGSL — the device-free half of the port's differential (the
// real-GPU half is the gym `pile` / `constraints` / `character` single-step gates against the f64 oracle).
// Every pass is one `stepWgsl` entry: the exact text its pipeline compiles.

const passes = Object.entries(stepWgsl);

const defs = (wgsl: string): string[] =>
    [...wgsl.matchAll(/^(?:fn|struct|const)\s+([A-Za-z0-9_]+)/gm)].map((x) => x[1]);

test("every pass declares each function, struct and const once", () => {
    for (const [name, wgsl] of passes) {
        const seen = defs(wgsl());
        const dupes = seen.filter((x, i) => seen.indexOf(x) !== i);
        expect(dupes, `${name} declares a definition twice`).toEqual([]);
    }
});

// The chunk a shared definition lands in is resolution-order dependent (`step-forcing.test.ts`), so a
// pass can splice a chunk whose base declaration went to a sibling it doesn't splice — Tint then rejects
// the module for an unresolved identifier and the only signal is a bench run. This is that check,
// device-free: whatever a pass *uses* of the shared group, it must also declare.
test("every pass declares what it reads from the shared group", () => {
    for (const [name, wgsl] of passes) {
        const src = wgsl();
        for (const [use, decl] of [
            ["params.", "var<uniform> params:"],
            ["bodies[", "> bodies:"],
            ["pairContacts[", "> pairContacts:"],
        ]) {
            if (!src.includes(use)) continue;
            expect(src.includes(decl), `${name} reads ${use} with no declaration`).toBe(true);
        }
    }
});

test("every pass keeps its integer locals unsigned and divides no integers", () => {
    for (const [name, wgsl] of passes) {
        const src = wgsl();
        try {
            integerDiscipline(src);
            noIntegerDivision(src);
        } catch (cause) {
            throw new Error(`${name}: ${(cause as Error).message}`, { cause });
        }
    }
});

test("a kernel binds the shared solver group and never redeclares its three bindings", () => {
    for (const [name, wgsl] of passes) {
        const src = wgsl();
        const all = [...src.matchAll(/@group\((\d)\) @binding\(\d\) var<[^>]*> (\w+):/g)];
        // a pass that splices an accessor takes all three from the shared group; the pack passes splice
        // none (they address `bodies` by column through their own binding), so they own their declaration
        const shared = all.some((x) => x[1] === String(SOLVER_GROUP));
        for (const kind of ["bodies", "pairContacts", "params"]) {
            const at = all.filter((x) => x[2] === kind);
            expect(at.length, `${name} declares ${kind} ${at.length} times`).toBeLessThanOrEqual(1);
            if (!shared) continue;
            for (const x of at)
                expect(x[1], `${name}: ${kind} left the shared group`).toBe(String(SOLVER_GROUP));
        }
    }
});

// gpu.md: `maxStorageBuffersPerShaderStage` is 10 and Chrome fails pipeline creation with no diagnostic
// past it. The count now spans two groups — a kernel's own declarations plus the shared layout's two
// storage entries, which count whether or not the shader declares them — so it's checked here rather than
// left to a bench run (`jointRecords` adds to the count, and the four passes below sit exactly at the ceiling).
test("no pass exceeds the 10-storage-binding floor across both groups", () => {
    for (const [name, wgsl] of passes) {
        const src = wgsl();
        const declared = [...src.matchAll(/@group\((\d)\) @binding\(\d\) var<storage/g)];
        const own = declared.filter((x) => x[1] !== String(SOLVER_GROUP)).length;
        const shared = src.includes(`@group(${SOLVER_GROUP})`) ? SHARED_STORAGE : 0;
        expect(own + shared, `${name} binds ${own + shared} storage buffers`).toBeLessThanOrEqual(
            10,
        );
    }
});

// pointerDiscipline is name-scoped, not scope-aware, so it runs per ported chunk rather than over a whole
// pass: a `let n` in one spliced function false-positives against a `(&n)` in another (the collide pass
// does exactly that — `var n` in `capsulePoly` vs `let n = polyVertCount(p)` in `supportPoly`).
// The body/rest/pose/contact/boxExtent accessors thread no pointer (every accessor is by-value); the
// broadphase accumulator + the collide warmstart merge below are the first ported chunks that do.
test("no ported chunk takes a pointer to a let", () => {
    const V = solverVariants.roRo;
    for (const chunk of [V.mathWgsl, V.bodyRestWgsl, V.bodyWgsl, V.contactWgsl, V.boxExtentWgsl])
        pointerDiscipline(chunk());
    // the broadphase accumulator (nbr/nd2/count, threaded by `d.ref` + the widened-count escape) and the
    // collide warmstart merge (rA/rB threaded through `mergeWarmstart`) are the two ported chunks that DO
    // thread a pointer — the standing guard's first real cases. Scoped to each fn's own body (not the whole
    // resolved pass): a splice site pulls in the whole SAT dependency tree, whose own `let n`/`(&n)` pairs
    // (capsulePoly, already forced) would false-positive across scope.
    pointerDiscipline(body(stepWgsl.broadphase(), "fn broadCandidate("));
    pointerDiscipline(body(stepWgsl.broadphase(), "fn broadEmit("));
    pointerDiscipline(body(stepWgsl.collideBox(), "fn mergeWarmstart("));
});

// The broadphase descent + the small-N scan splice the SAME candidate/emit functions (`broadCandidate` /
// `broadEmit`), so the nearest-K prune, the ascending sort, and the INVALID-fill are one source of truth —
// the precondition for warmstart carrying across a regime flip. This pins that the two mains actually
// resolve to identical shared-function bodies (not two independently-drifted copies).
test("broadphase and broadphase-small splice the same candidate/emit bodies", () => {
    const descent = body(stepWgsl.broadphase(), "fn broadCandidate(");
    const scan = body(stepWgsl.broadphaseSmall(), "fn broadCandidate(");
    expect(descent).toBe(scan);
    expect(body(stepWgsl.broadphase(), "fn broadEmit(")).toBe(
        body(stepWgsl.broadphaseSmall(), "fn broadEmit("),
    );
});

test("a static body's broadphase block is cleared to INVALID with no descent", () => {
    expect(flat(body(stepWgsl.broadphase(), "fn broadphaseMain("))).toContain(
        "if ((bMass(i) <= 0f))",
    );
    // the small-N scan clears via the shared `broadEmit` (count stays 0 for a static lane, `act` false),
    // not a standalone early clear — its own gate is the `act` flag, not a static-body branch
    expect(flat(stepWgsl.broadphaseSmall())).toContain("let act = (inRange && (bMass(i) > 0f));");
});

// collideHull/collideRoundedPoly reach hullData only through the WGSL-bodied hull-geometry readers
// (hullRef/hVertL/…), which name it as a free identifier invisible to tgpu.resolve's dependency walk —
// with no forcing touch the readers' bodies still call `hullData[...]` in the emitted text, but nothing
// declares the binding, an undeclared-identifier compile error Tint would catch, not resolve. Pins the
// touch structurally (the same bug class step.test.ts already pins for joint-dual's counters binding).
test("collideHull and collideRoundedPoly declare the hullData binding their readers use", () => {
    for (const wgsl of [stepWgsl.collideHull(), stepWgsl.collideRoundedPoly()]) {
        expect(wgsl).toContain("var<storage, read> hullData: array<u32>;");
    }
});

test("each collide kernel gates on its own shape-pair class and only box clears dead slots", () => {
    expect(flat(stepWgsl.collideBox())).toContain("clearBlock(recBase);");
    // rounded/hull/rounded-poly return bare (no clearBlock) on an INVALID or separated pair — only a
    // 0-contact SAT result clears (every kernel reaching the SAT owns the pair for this frame)
    for (const [wgsl, name] of [
        [stepWgsl.collideRounded(), "collideRoundedMain"],
        [stepWgsl.collideHull(), "collideHullMain"],
        [stepWgsl.collideRoundedPoly(), "collideRoundedPolyMain"],
    ] as const) {
        const src = flat(body(wgsl, `fn ${name}(`));
        // exactly one clearBlock call in these three kernels: the 0-contact-SAT branch
        expect(src.match(/clearBlock\(recBase\);/g)?.length).toBe(1);
    }
});

test("the aabb prim is the oriented-box extent padded by the band and the velocity sweep", () => {
    const src = body(stepWgsl.aabb(), "fn aabbMain");
    // the pad is prim-only: band + |vel|·dt, and BOTH prim halves come off the same `e`
    expect(flat(src)).toContain(
        "let e = ((boxExtent(i) + vec3f(0.03999999910593033)) + (abs(bVelL(i)) * params.dt));",
    );
    expect(flat(src)).toContain("prims[(2u * slot)] = vec4f((p - e), 0f);");
    expect(flat(src)).toContain("prims[((2u * slot) + 1u)] = vec4f((p + e), 0f);");
    // one thread per dense slot, early-out past the live count
    expect(flat(src)).toContain("let i = eids[(1u + slot)];");
});

test("boxExtent is |R|·h inflated by the rounding radius", () => {
    const src = body(stepWgsl.aabb(), "fn boxExtent");
    expect(flat(src)).toContain(
        "return ((((ax0 * h.x) + (ax1 * h.y)) + (ax2 * h.z)) + vec3f(bRadius(i)));",
    );
    // the three axes are the rotated unit axes, absolute-valued
    for (const axis of ["vec3f(1, 0, 0)", "vec3f(0, 1, 0)", "vec3f(0, 0, 1)"])
        expect(flat(src)).toContain(`abs(qRotateW(q, ${axis}))`);
});

test("the body accessors read the columns the SoA layout assigns", () => {
    // every typed pass now emits only the accessors it actually calls (tgpu.resolve's call-graph walk),
    // so no single pipeline carries every column reader any more — the raw accessor-chunk getter (kept for
    // the chunk-forcing tests) is the one place that still splices the full body chunk unconditionally.
    const src = solverVariants.roRo.bodyWgsl();
    const col = (fn: string) => /bCol\((\d+)u,/.exec(body(src, `fn ${fn}(`))?.[1];
    expect(col("bPos")).toBe("0");
    expect(col("bQuat")).toBe("1");
    expect(col("bInertL")).toBe("2");
    expect(col("bInertQ")).toBe("3");
    expect(col("bInitL")).toBe("4");
    expect(col("bInitQ")).toBe("5");
    expect(col("bVelL")).toBe("6");
    expect(col("bVelA")).toBe("7");
    expect(col("bPrevV")).toBe("8");
    expect(col("bMass")).toBe("9");
    expect(col("bHalf")).toBe("10");
    expect(col("bFriction")).toBe("10");
    expect(col("bShape")).toBe("11");
    expect(col("bRadius")).toBe("11");
    expect(col("bHullId")).toBe("11");
    expect(flat(body(src, "fn bCol("))).toContain("bodies[((col * params.eidCap) + i)]");
});

test("the contact reader indexes the record SoA by column", () => {
    expect(flat(body(stepWgsl.dual(), "fn cc("))).toContain(
        "pairContacts[((col * params.recordCap) + rec)]",
    );
});

// The quat math is now one source for the shader and the CPU, so it differentials against the f64
// oracle's own quaternion helpers (tests/avbd/math.ts) with no device — the arithmetic the GPU runs,
// checked against the executable spec. f32 rounding is the only gap, so the tolerance is f32 epsilon.
const q0: [number, number, number, number] = [0.2, -0.5, 0.34, 0.771];
const q1: [number, number, number, number] = [-0.1, 0.42, 0.6, 0.671];
const v0: [number, number, number] = [1.5, -2.25, 0.75];
const close = (a: readonly number[], b: readonly number[]) => {
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i], 5);
};

test("the ported quat math matches the f64 oracle", () => {
    const V = solverVariants.roRo;
    close(V.qConjW(d.vec4f(...q0)), m.qconj(q0));
    close(V.qMulW(d.vec4f(...q0), d.vec4f(...q1)), m.qmul(q0, q1));
    close(V.qInvW(d.vec4f(...q0)), m.qinverse(q0));
    close(V.qSubW(d.vec4f(...q0), d.vec4f(...q1)), m.qsub(q0, q1));
    close(V.qRotateW(d.vec4f(...q0), d.vec3f(...v0)), m.rotate(q0, v0));
    close(V.qAddW(d.vec4f(...q0), d.vec3f(...v0)), m.qadd(q0, v0));
});

// `configure` / `setJoints` stage the uniform through an ArrayBuffer at derived offsets, so the schema is
// the source of truth on the CPU side too — reordering a field must move the write, not silently shift the
// WGSL struct out from under it (the layout-mismatch class the port exists to kill).
test("the Step schema pins the uniform layout the CPU writer stages against", () => {
    expect(d.sizeOf(Step)).toBe(64);
    const at = (f: keyof (typeof Step)["propTypes"]) => d.memoryLayoutOf(Step, (s) => s[f]).offset;
    expect(at("recordCap")).toBe(0);
    expect(at("dt")).toBe(16);
    expect(at("invDt2")).toBe(32);
    expect(at("jointCount")).toBe(48);
    expect(at("substeps")).toBe(52);
});

// commit / dual / joint-init / joint-dual / csr-color-small run as typed pipelines over
// the shared solver factory (contactForce/contactContrib/springContrib/jointContrib/solvePose/dualSlot/
// jointDualOne). No manual chunk()/ns splicing needed for these — each is a whole `tgpu.resolve([kernel])`
// call, so the generic passes-loop tests above already cover duplicate-defs/idiv/pointer/storage-floor;
// these pin the specific shapes below.

test("commit reads its color index from a per-color static uniform, not a dynamic offset", () => {
    const src = stepWgsl.commit();
    expect(src).toContain("struct ColorIdx {");
    expect(src).toContain("var<uniform> color: ColorIdx;");
    expect(flat(body(src, "fn commitMain("))).toContain("(colors[bid] != color.value)");
});

test("joint-init and joint-dual bind jointRecords in its own group, distinct from the shared solver group", () => {
    expect(JOINT_GROUP).not.toBe(SOLVER_GROUP);
    for (const wgsl of [stepWgsl.jointInit(), stepWgsl.jointDual()]) {
        const at = [
            ...wgsl.matchAll(/@group\((\d)\) @binding\(\d\) var<storage[^>]*> jointRecords:/g),
        ];
        expect(at.length).toBe(1);
        expect(at[0][1]).toBe(String(JOINT_GROUP));
    }
});

test("dual recomputes the same cone-clamped force the primal's contactContrib reads, then ramps the penalty", () => {
    const src = flat(body(stepWgsl.dual(), "fn dualMain("));
    expect(src).toContain("dualSlot(");
    const slot = flat(body(stepWgsl.dual(), "fn dualSlot("));
    // pre-clamp magnitude + bound gate the tangent ramp (avbd.md "friction ramp"), never the post-clamp force
    expect(slot).toContain("if ((fs <= bounds))");
    expect(slot).toContain("if ((cf.force.x < 0f))");
});

test("joint-dual's all-static gate mirrors joint-init's construction-time rejection", () => {
    const dualSrc = flat(body(stepWgsl.jointDual(), "fn jointDualOne("));
    const initSrc = flat(body(stepWgsl.jointInit(), "fn jointInitMain("));
    expect(dualSrc).toContain("if ((solverStatic(b) && (aWorld || solverStatic(a))))");
    expect(initSrc).toContain("(aStatic && (bMass(b) <= 0f))");
});

// joint-dual's only inputs are the shared roRo group + jointRecords, so with no forcing touch its
// group-0 `counters` layout is invisible to tgpu.resolve's reference walk and the emitted WGSL omits
// the binding entirely — a mismatch against the JS-side bind group that only Dawn's pipeline creation
// catches (real-device only). Pins the touch structurally instead.
test("joint-dual references its own-group counters binding (the forcing touch)", () => {
    expect(stepWgsl.jointDual()).toContain(
        "var<storage, read_write> counters: array<atomic<u32>>;",
    );
});

// The shared factory's contactContrib/springContrib/jointContrib/solvePose are the typed primal's (and
// solve-lds's) real pipeline consumer since 2b — this pins that the roRo-bound instance the CPU
// differential exercises also resolves cleanly standalone (the "author once" mandate).
test("the shared factory's primal-side functions resolve standalone (CPU-differential-testable)", () => {
    for (const fn of [
        solverRoRo.contactForce,
        solverRoRo.contactContrib,
        solverRoRo.springContrib,
        solverRoRo.jointContrib,
        solverRoRo.solvePose,
    ]) {
        expect(() => tgpu.resolve([fn] as never, { names: "strict" })).not.toThrow();
    }
});

test("the three access variants describe the same three bindings", () => {
    const keys = (v: keyof typeof solverVariants) => Object.keys(solverVariants[v].layout.entries);
    expect(keys("roRo")).toEqual(["params", "bodies", "pairContacts"]);
    expect(keys("roRw")).toEqual(keys("roRo"));
    expect(keys("rwRw")).toEqual(keys("roRo"));
    // and each emits its own access mode, which is why they cannot be one layout
    expect(stepWgsl.primal()).toContain("var<storage, read> bodies");
    expect(stepWgsl.dual()).toContain("var<storage, read_write> pairContacts");
    expect(stepWgsl.velocity()).toContain("var<storage, read_write> bodies");
});

// primal + solve-lds run as typed pipelines over the shared solver factory
// (contactMath/contribMath/dualMath/jointDualMath). The raw MAT3_WGSL/CONTACT_FORCE_WGSL/
// JOINT_REC_WGSL/SOLVE_MATH_WGSL/DUAL_MATH_WGSL/JOINT_DUAL_MATH_WGSL/STEP_CONSTS_WGSL chunks are gone —
// these pin the shapes that replaced them.

test("primal reads its color index from a per-color static uniform, not a dynamic offset", () => {
    const src = stepWgsl.primal();
    expect(src).toContain("struct ColorIdx {");
    expect(src).toContain("var<uniform> color: ColorIdx;");
    expect(flat(body(src, "fn primalMain("))).toContain("(colors[bid] != color.value)");
});

test("primal and solve-lds bind jointRecords in JOINT_GROUP, distinct from the shared solver group", () => {
    for (const wgsl of [stepWgsl.primal(), stepWgsl.solveLds()]) {
        const at = [
            ...wgsl.matchAll(/@group\((\d)\) @binding\(\d\) var<storage[^>]*> jointRecords:/g),
        ];
        expect(at.length).toBe(1);
        expect(at[0][1]).toBe(String(JOINT_GROUP));
    }
});

test("primal and solve-lds each declare a fourth own-I/O group, distinct from every other group", () => {
    expect(LDS_IO_GROUP).not.toBe(SOLVER_GROUP);
    expect(LDS_IO_GROUP).not.toBe(JOINT_GROUP);
    expect(PRIMAL_COLOR_GROUP).not.toBe(SOLVER_GROUP);
    expect(PRIMAL_COLOR_GROUP).not.toBe(JOINT_GROUP);
    expect(stepWgsl.primal()).toContain(`@group(${PRIMAL_COLOR_GROUP})`);
    expect(stepWgsl.solveLds()).toContain(`@group(${LDS_IO_GROUP})`);
});

test("solve-lds resolves the whole iters × colors block as one workgroup-resident kernel", () => {
    const src = flat(stepWgsl.solveLds());
    // the LDS pose arrays + the eid→slot map, replacing the raw denseMap/lpx/lpy/lpz/lq var<workgroup>s
    expect(src).toContain("var<workgroup> lpx: array<f32, 512");
    expect(src).toContain("var<workgroup> lq: array<vec4f, 512");
    expect(src).toContain("workgroupUniformLoad(");
    // one compute entry point, not the looped path's per-color dispatch pair
    expect([...src.matchAll(/@compute/g)]).toHaveLength(1);
});

// The shared chunks (Mat3/CForce/Contrib/Sol/NewPose) are authored exactly once and closed over per
// reader set (contactMath/contribMath/dualMath/jointDualMath) — every pass that calls into them must emit
// byte-identical struct text, never an independent raw copy that could drift from the typed one (the
// defect class the interim two-copy state at 2a was sanctioned to avoid for exactly one sub-stage).
test("the shared solver structs emit byte-identical text in every pass that uses them", () => {
    const allThree = {
        dual: stepWgsl.dual(),
        primal: stepWgsl.primal(),
        solveLds: stepWgsl.solveLds(),
    };
    // dual calls only contactForce (via dualSlot) — Contrib/Sol/NewPose belong to contribMath, which only
    // primal + solve-lds reach (solvePose)
    const primalAndLds = { primal: stepWgsl.primal(), solveLds: stepWgsl.solveLds() };
    for (const [name, passes] of [
        ["Mat3", allThree],
        ["CForce", allThree],
        ["Contrib", primalAndLds],
        ["Sol", primalAndLds],
        ["NewPose", primalAndLds],
    ] as const) {
        const seen: Record<string, string> = {};
        for (const [pass, src] of Object.entries(passes)) {
            const found = new RegExp(`struct ${name} \\{[^}]*\\}`).exec(src);
            expect(found, `${name} missing from ${pass}`).toBeTruthy();
            seen[pass] = found![0];
        }
        const bodies = new Set(Object.values(seen));
        expect(bodies.size, `${name} drifted: ${JSON.stringify(seen)}`).toBe(1);
    }
});

// One app stands up more than one PhysicsStep — the gym `pile` scenario builds two, `constraints` three —
// and the precompile queue rejects a duplicate label. So the labels are per-instance: the first keeps the
// bare `phys-*` names (stable profiler rows for the single-world apps), each later one takes a numbered
// scope. `phys-compose` registers lazily on the first `compose()`, long after construction, so it must read
// the same stored scope rather than minting a second one.
describe("per-instance precompile labels", () => {
    // no adapter (testing.md): the step is built against a recording stub, which is enough to reach the
    // precompile registrations — nothing here dispatches.
    const stub = (): GPUDevice =>
        ({
            features: new Set(["subgroups"]),
            limits: { maxStorageBufferBindingSize: 1 << 30 },
            queue: { writeBuffer() {} },
            createBuffer: (desc: GPUBufferDescriptor) => ({ ...desc, destroy() {} }),
            createBindGroupLayout: (desc: unknown) => desc,
            createBindGroup: (desc: unknown) => desc,
            createPipelineLayout: (desc: unknown) => desc,
            createShaderModule: (desc: unknown) => desc,
            createComputePipeline: (desc: unknown) => desc,
            createComputePipelineAsync: async (desc: unknown) => desc,
        }) as unknown as GPUDevice;

    test("a second PhysicsStep on one device takes a scoped label instead of colliding", async () => {
        const saved = { ...Compute };
        try {
            const device = stub();
            await requestGPU(device);

            await PhysicsStep.create(device, 64, 64);
            expect(() => precompile("phys-aabb", () => true)).toThrow(/duplicate/);

            await PhysicsStep.create(device, 64, 64);
            expect(() => precompile("phys-2-aabb", () => true)).toThrow(/duplicate/);
            expect(() => precompile("phys-2-collide-box", () => true)).toThrow(/duplicate/);
            expect(() => precompile("phys-2-joint-dual", () => true)).toThrow(/duplicate/);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("the prepared compose label rides the same instance scope", async () => {
        const saved = { ...Compute };
        try {
            const device = stub();
            await requestGPU(device);
            const first = await PhysicsStep.create(device, 64, 64);
            const second = await PhysicsStep.create(device, 64, 64);

            // The stub buffer carries no schema for the indirect dispatch to read. Swallow that dispatch,
            // never a duplicate label: the collision this guards against surfaces at preparation.
            const composeOnce = async (step: PhysicsStep) => {
                const transforms = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });
                const encoder = {
                    beginComputePass: () => ({ end() {} }),
                } as unknown as GPUCommandEncoder;
                await step.prepareCompose(transforms);
                try {
                    step.compose(encoder, transforms);
                } catch (err) {
                    if (String(err).includes("duplicate precompile")) throw err;
                }
            };

            await composeOnce(first);
            expect(() => precompile("phys-compose", () => true)).toThrow(/duplicate/);
            await composeOnce(second);
            expect(() => precompile("phys-2-compose", () => true)).toThrow(/duplicate/);
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("late compose preparation settles validation before the pipeline can be used", async () => {
        const saved = { ...Compute };
        let rejectFence!: (error: Error) => void;
        const fence = new Promise<void>((_, reject) => {
            rejectFence = reject;
        });
        let lateFence: Promise<void> | undefined;
        try {
            const device = {
                ...stub(),
                queue: {
                    writeBuffer() {},
                    onSubmittedWorkDone: () => lateFence ?? Promise.resolve(),
                },
                pushErrorScope() {},
                popErrorScope: async () => null,
            } as unknown as GPUDevice;
            await requestGPU(device);
            const bound = {
                $name() {
                    return this;
                },
                with() {
                    return this;
                },
                dispatchWorkgroups() {
                    return this;
                },
                dispatchWorkgroupsIndirect() {
                    return this;
                },
            };
            Object.assign(Compute, {
                root: {
                    createComputePipeline: () => bound,
                    createBindGroup: () => ({}),
                    unwrap: (value: unknown) => value,
                },
            });
            const step = await PhysicsStep.create(device, 64, 64);
            await precompileAll();
            lateFence = fence;

            const transforms = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE });
            const preparing = step.prepareCompose(transforms);
            const encoder = {
                beginComputePass: () => {
                    throw new Error("compose encoded before validation settled");
                },
            } as unknown as GPUCommandEncoder;
            expect(() => step.compose(encoder, transforms)).toThrow(
                "await prepareCompose(transforms)",
            );

            rejectFence(new Error("late compose fence failed"));
            const failure = await preparing.catch((error: unknown) => error);
            expect(failure).toMatchObject({
                name: "GpuDiagnosticError",
                label: "phys-compose",
            });
            expect(String(failure)).toContain("late compose fence failed");
            expect(() => step.compose(encoder, transforms)).toThrow("late compose fence failed");
        } finally {
            Object.assign(Compute, saved);
        }
    });

    test("the public factory propagates a late precompile fence failure", async () => {
        const saved = { ...Compute };
        let physicsAllocated = false;
        let physicsFences = 0;
        let rejectFirst!: (error: Error) => void;
        let releaseRest!: () => void;
        const firstFence = new Promise<void>((_, reject) => {
            rejectFirst = reject;
        });
        const restFence = new Promise<void>((resolve) => {
            releaseRest = resolve;
        });
        try {
            const base = stub();
            const createBuffer = base.createBuffer.bind(base);
            const device = {
                ...base,
                queue: {
                    writeBuffer() {},
                    onSubmittedWorkDone() {
                        if (!physicsAllocated) return Promise.resolve();
                        physicsFences++;
                        return physicsFences === 1 ? firstFence : restFence;
                    },
                },
                pushErrorScope() {},
                popErrorScope: async () => null,
                createBuffer(descriptor: GPUBufferDescriptor) {
                    if (descriptor.label === "phys-bodies") physicsAllocated = true;
                    return createBuffer(descriptor);
                },
            } as unknown as GPUDevice;
            await requestGPU(device);
            await precompileAll();
            const bound = {
                $name() {
                    return this;
                },
                with() {
                    return this;
                },
                dispatchWorkgroups() {
                    return this;
                },
            };
            Object.assign(Compute, {
                root: {
                    createComputePipeline: () => bound,
                    createBindGroup: () => ({}),
                    unwrap: (value: unknown) => value,
                },
            });

            const creation = PhysicsStep.create(device, 64, 64);
            while (!physicsAllocated) await Promise.resolve();
            let returned = false;
            void creation.then(
                () => {
                    returned = true;
                },
                () => {
                    returned = true;
                },
            );
            await Promise.resolve();
            await Promise.resolve();
            expect(returned).toBe(false);

            rejectFirst(new Error("late physics fence failed"));
            while (physicsFences < 2) await Promise.resolve();
            await Promise.resolve();
            expect(returned).toBe(false);

            releaseRest();
            await expect(creation).rejects.toMatchObject({
                name: "GpuDiagnosticError",
                label: "phys-aabb",
                message: expect.stringContaining("late physics fence failed"),
            });
        } finally {
            Object.assign(Compute, saved);
        }
    });
});
