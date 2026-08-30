import { join } from "node:path";
import tgpu, { isTgpuFn } from "typegpu";
import { TEST_TIER_SUFFIXES } from "./test-tiers";
import { integerDiscipline, noDivision, noIntegerDivision, pointerDiscipline } from "./wgsl";

export const STANDARDS_POPULATION_GOLDEN = 101;

const SRC_DIR = join(import.meta.dir, "../src");

export interface ResolvedKernel {
    file: string;
    name: string;
    wgsl: string;
}

export interface KernelExport {
    file: string;
    name: string;
    value: unknown;
}

export async function sourceModules(): Promise<string[]> {
    const out: string[] = [];
    for await (const path of new Bun.Glob("**/*.ts").scan({ cwd: SRC_DIR })) {
        if (/\.d\.ts$/.test(path) || TEST_TIER_SUFFIXES.test(path) || /\.fixture\.ts$/.test(path)) {
            continue;
        }
        out.push(path);
    }
    return out.sort();
}

/** The shared filesystem/import/identity-dedup enumerator for both standards verdicts. */
export async function kernelExports(): Promise<Map<string, KernelExport>> {
    const byIdentity = new Map<unknown, KernelExport[]>();
    for (const file of await sourceModules()) {
        const mod = (await import(join(SRC_DIR, file))) as Record<string, unknown>;
        for (const [name, value] of Object.entries(mod)) {
            if (!isTgpuFn(value)) continue;
            const group = byIdentity.get(value);
            if (group) group.push({ file, name, value });
            else byIdentity.set(value, [{ file, name, value }]);
        }
    }

    const byName = new Map<string, KernelExport>();
    for (const group of byIdentity.values()) {
        const nonBarrel = group.filter(
            (entry) => !entry.file.endsWith("/index.ts") && entry.file !== "index.ts",
        );
        const candidates = nonBarrel.length > 0 ? nonBarrel : group;
        candidates.sort((a, b) => a.file.length - b.file.length || a.file.localeCompare(b.file));
        const entry = candidates[0];
        const prior = byName.get(entry.name);
        if (prior) {
            throw new Error(
                `kernel name collision: "${entry.name}" in ${prior.file} and ${entry.file}`,
            );
        }
        byName.set(entry.name, entry);
    }
    return byName;
}

/** Resolves the complete identity-deduplicated population of exported `tgpu.fn` kernels. */
export async function resolvedKernels(): Promise<ResolvedKernel[]> {
    return [...(await kernelExports()).values()]
        .sort((a, b) => a.name.localeCompare(b.name))
        .map(({ file, name, value }) => ({
            file,
            name,
            wgsl: tgpu.resolve([value] as never, { names: "strict" }),
        }));
}

// The declared registry + pure checker half of the TGSL-corpus meta-test. Shape copied from `examples/gym/src/scenarios/timeouts.ts` + `coverage.ts`: plain committed
// data, a pure checker asserted both directions, red-provable against fixtures. `standards.test.ts` is
// the one real-filesystem/real-import/real-resolve seam — it walks `packages/shallot/src`, dynamic-imports
// every module, duck-detects live TGSL kernels, resolves each to WGSL, and runs the checks below against
// the real text, producing a `Population` this file's `checkStandards` consumes. This file imports
// `tests/wgsl.ts`'s existing checkers (the four discipline checks it already ships and every ported-kernel
// test file already calls) rather than reimplementing their regexes a second time — "one source of truth"
// outweighs a literal zero-runtime-import reading here, since the alternative is a second,
// driftable copy of the same regex logic.

export type DisciplineCheck =
    | "noIntegerDivision"
    | "integerDiscipline"
    | "pointerDiscipline"
    | "noDivision";

const CHECKERS: Record<DisciplineCheck, (src: string) => void> = {
    noIntegerDivision,
    integerDiscipline,
    pointerDiscipline,
    noDivision,
};

/** the checks every live kernel's resolved WGSL is held to by default. `noDivision` is deliberately NOT
 *  here — it is the stronger form for "a kernel whose every index is a shift, mask or multiply-add"
 *  (`wgsl.ts`'s own JSDoc); applied corpus-wide it would red every kernel doing a legitimate float
 *  division (a normalize, a reciprocal). A kernel opts into it via its registry entry's `also`. */
export const DEFAULT_CHECKS: readonly DisciplineCheck[] = [
    "noIntegerDivision",
    "integerDiscipline",
    "pointerDiscipline",
];

/** runs one discipline check against a kernel's resolved WGSL; returns the violation message, or `null`
 *  when the check holds. `tests/wgsl.ts`'s checkers assert via `expect(...)`, throwing on failure — this
 *  is the one seam that turns that throw into a value, so {@link checkStandards} stays pure over data. */
export function violation(check: DisciplineCheck, wgsl: string): string | null {
    try {
        CHECKERS[check](wgsl);
        return null;
    } catch (e) {
        return e instanceof Error ? e.message : String(e);
    }
}

export interface KernelEntry {
    /** checks beyond {@link DEFAULT_CHECKS} this kernel is also held to. */
    also?: readonly DisciplineCheck[];
    /** per-check exemptions. Each reason claims the load-bearing property that actually exempts the
     *  kernel, never a structural shape (`testing.md`'s exemption-reason law). Keyed (kernel, check) —
     *  never by kernel alone, so an exemption for one check can't shadow the three sound ones
     *  (Locked decision, "Key an exemption by (subject, quantity)"). */
    exempt?: Partial<Record<DisciplineCheck, string>>;
}

export type StandardsRegistry = Record<string, KernelEntry>;

/** the per-kernel exemptions and check opt-ins the corpus meta-test reads. Each reason claims the
 *  load-bearing property that makes the flagged pattern correct, grounded in the kernel's resolved WGSL,
 *  never a structural shape (`testing.md`'s exemption-reason law). A kernel whose violation is a real
 *  defect is fixed at the source instead of landing a row here. */
export const STANDARDS_REGISTRY: StandardsRegistry = {
    decodePos: {
        exempt: {
            noIntegerDivision:
                "the divisor is the unorm16 quantization denominator 65535, a compile-time constant: " +
                "the resolved WGSL is `f32((w1 & 65535u)) / 65535f` — the numerator is already an f32 " +
                "before the `/`, so this is a float dequantize with no integer-truncation semantics, " +
                "not the mixed-operand shape the check exists to catch.",
        },
    },
    sampleStars: {
        exempt: {
            integerDiscipline:
                "the 3×3 neighbor-cell loop's `dx`/`dy` range over [-1, 1] to reach every adjacent hash " +
                "cell — a neighbor offset is genuinely signed, and there is no unsigned encoding of " +
                "'one cell in either direction'. Magnitude never exceeds ±1, nowhere near the bit-31 " +
                "signed-overflow the check exists to prevent.",
        },
    },
    sampleSky: {
        exempt: {
            integerDiscipline:
                "sampleSky calls sampleStars, and tgpu.resolve inlines the callee's body into sampleSky's " +
                "resolved WGSL — this is the same signed 3×3 neighbor loop as sampleStars:integerDiscipline, " +
                "not a second violation, so the exemption is identical.",
        },
    },
    packQuatSmallest3: {
        exempt: {
            integerDiscipline:
                "the smallest-3 quat pack quantizes each retained component to a signed 10-bit " +
                "two's-complement lane (`i32(clamp(round(abc.x * scale), -511.0, 511.0))`) before " +
                "`u32(s0) & 0x3FFu` truncates it into the packed bitfield — the retained components can " +
                "be negative after the largest-component sign-fix, and the wire format is two's-complement " +
                "(`unpackQuatSmallest3` sign-extends via `bitcast<i32>`), so the intermediate has to be " +
                "signed to match the decoder — a bias encoding would avoid the cast but is a format change.",
        },
    },
};

export type FindingKind =
    | "stale-exemption-key"
    | "missing-exemption-reason"
    | "exempt-shadows-passing"
    | "unexempted-violation";

export interface Finding {
    kind: FindingKind;
    detail: string;
}

/** kernel name -> the checks (from {@link DEFAULT_CHECKS} ∪ its registry entry's `also`) that actually
 *  fail against its real resolved WGSL, computed by the real-import/resolve seam (`standards.test.ts`).
 *  This file never resolves a kernel or touches the filesystem — the population is handed in. */
export type Population = Record<string, readonly DisciplineCheck[]>;

/** the both-directions check over `{population, registry}`, pure — no filesystem, no dynamic import, no
 *  WGSL resolution. Mirrors `coverage.ts`'s `checkGateEntries`/`checkCompleteness` split: a declared
 *  registry asserted against a real population the caller supplies.
 *
 *  Four finding kinds: a registry key naming a kernel absent from the population (`stale-exemption-key`,
 *  a rename/removal orphaned the exemption); an exemption with no reason (`missing-exemption-reason`);
 *  an exemption for a (kernel, check) pair whose check doesn't actually fail — a dead exemption hiding
 *  a check that would otherwise run clean (`exempt-shadows-passing`); and a live kernel violating an
 *  applicable check with no matching exemption (`unexempted-violation`) — the corpus-wide direction
 *  `standards.test.ts`'s "every live kernel's discipline violations are fixed or exempted" test asserts is empty. */
export function checkStandards(population: Population, registry: StandardsRegistry): Finding[] {
    const findings: Finding[] = [];

    for (const [name, entry] of Object.entries(registry)) {
        if (!Object.hasOwn(population, name)) {
            findings.push({ kind: "stale-exemption-key", detail: name });
            continue;
        }
        const violations = population[name];
        for (const [check, reason] of Object.entries(entry.exempt ?? {}) as [
            DisciplineCheck,
            string,
        ][]) {
            if (!reason || reason.trim() === "") {
                findings.push({ kind: "missing-exemption-reason", detail: `${name}:${check}` });
            }
            if (!violations.includes(check)) {
                findings.push({ kind: "exempt-shadows-passing", detail: `${name}:${check}` });
            }
        }
    }

    for (const [name, violations] of Object.entries(population)) {
        const exempt = new Set(Object.keys(registry[name]?.exempt ?? {}));
        for (const check of violations) {
            if (!exempt.has(check)) {
                findings.push({ kind: "unexempted-violation", detail: `${name}:${check}` });
            }
        }
    }

    return findings;
}

// ── Part 2: the differential registry ────────────────────────────────────────────────────────
//
// CPU-callability, the TypeGPU port's headline win (testing.md "A CPU-callable kernel is the
// logic-truth surface for the same authored function the GPU pipeline resolves"), was recorded only
// in deleted spec prose. This is the declared registry that replaces it: per kernel, either a named
// differential test (a real `.test.ts` file that calls the kernel on the CPU against a reference) or
// a reason it genuinely can't have one — naming the mechanism, not a category (`testing.md`'s
// exemption-reason law applies here too). Same shape as Part 1: plain data, a pure checker, asserted
// both directions, red-provable with no filesystem.

/** a named CPU differential: the `.test.ts` file (repo-relative, from the shallot repo root — the
 *  same convention `examples/gym/src/scenarios/timeouts.ts`'s `covers` globs use) that calls this
 *  kernel on the CPU, and the exported symbol it calls. Two fields because "a file exists" alone
 *  proves nothing — a differential test file that stopped calling its kernel (a rename, a refactor
 *  that dropped the direct call) would still be "the named file exists" and silently stop meaning
 *  anything; naming the symbol makes that a mechanically checkable claim. `standards.test.ts`'s
 *  real-filesystem half runs two checks together, uniformly over every row, bare or aliased:
 *  {@link callsSymbol} (call-position syntax — not satisfiable by an unused import) and
 *  {@link importsSymbol} (a real import binding for that name from this file, not a string literal,
 *  a comment with parens, or a same-named local shadow definition). Neither alone closes the property;
 *  together they prove the file both imports and calls the named kernel. The residual limit: they
 *  prove a call happened, not that its result is asserted against a reference. */
export interface DifferentialTest {
    file: string;
    symbol: string;
    /** the local name the file actually calls, where it imports the kernel under an alias. Without
     *  it the arm reads both ways wrong on the same file: `rounded.oracle.ts` imports the kernel as
     *  `tgslCollideRounded` (so the bare name never appears at a call site) while defining its own
     *  f64 reference *named* `collideRounded` (so a bare-name grep passes against the wrong
     *  function). Declaring the alias makes the grep name the call the row is claiming. */
    alias?: string;
}

/** a kernel has exactly one of three: a named CPU differential; a reason it *can't* have one — the
 *  mechanism that makes CPU execution meaningless for it (a raw-WGSL leaf with no CPU arm, an atomic
 *  op with no CPU semantics, a pointer into GPU-only address space), never a category ("GPU-only
 *  kernel" restates the row instead of grounding it); or a `gap`, meaning nothing forbids a
 *  differential and none has been written.
 *
 *  The third arm is the point, not a concession: making the gap enumerable is what the registry
 *  buys. Writing the missing differentials is deliberately not part of that, so without it a
 *  pure-math kernel with no test caller would be forced into a false `reason` — a claimed mechanism
 *  nothing re-checks, which is exactly what `testing.md`'s exemption-reason law exists to stop. The
 *  gap count is frozen as a golden in `standards.test.ts`, so the cheap arm can't quietly absorb a
 *  new kernel: taking it moves a number in the same commit. */
export type DifferentialEntry = { test: DifferentialTest } | { reason: string } | { gap: string };

export type DifferentialRegistry = Record<string, DifferentialEntry>;

/** the differential registry's representative slice. This hand-authored slice exists to exercise every arm of
 *  {@link checkDifferentials} against real kernels, not to be complete: two genuine named-differential
 *  rows and one real instance of each can't-have mechanism.
 *
 *  - `octEncodeNormal`: `octEncode === octEncodeNormal` (`engine/utils/encode.test.ts`) calls the
 *    kernel directly on the CPU against the legacy scalar codec, 4096 random unit vectors plus the
 *    cardinal/degenerate cases.
 *  - `collideHull`: the packed-hull TGSL graph test (`standard/avbd/collide.test.ts`) calls the
 *    kernel directly on the CPU against the f64 hull SAT oracle (`tests/avbd/hull.ts`).
 *  - `packQuatSmallest3` / `unpackQuatSmallest3`: raw-WGSL leaves — their own JSDoc
 *    (`engine/utils/encode.ts`) states why: smallest-3 dynamically indexes a vector (`q[largest]`)
 *    and switches on the result, neither of which TGSL expresses, so the body stays WGSL text with no
 *    CPU arm to call.
 *  - `uniformLoad`: a raw-WGSL leaf over `ptr<workgroup, u32>` — its JSDoc (`engine/utils/tgsl.ts`)
 *    states the mechanism: "a workgroup pointer has no CPU meaning."
 *  - `compareExchange`: a raw-WGSL leaf over `ptr<storage, atomic<u32>, read_write>` —
 *    `atomicCompareExchangeWeak` is a device-memory compare-and-swap with no CPU-side semantics to
 *    reproduce; its JSDoc (`engine/utils/tgsl.ts`) states "GPU-only."
 */
export const DIFFERENTIAL_REGISTRY: DifferentialRegistry = {
    octEncodeNormal: {
        test: {
            file: "packages/shallot/src/engine/utils/encode.test.ts",
            symbol: "octEncodeNormal",
        },
    },
    collideHull: {
        test: { file: "packages/shallot/src/standard/avbd/collide.test.ts", symbol: "collideHull" },
    },
    packQuatSmallest3: {
        reason:
            "raw-WGSL leaf: smallest-3 dynamically indexes a vector (q[largest]) and switches on the " +
            "result, neither of which TGSL expresses, so the body is authored as a WGSL string with no " +
            "CPU arm — see the kernel's own JSDoc, engine/utils/encode.ts.",
    },
    unpackQuatSmallest3: {
        reason:
            "raw-WGSL leaf: the inverse of packQuatSmallest3's dynamic-index/switch pack, same " +
            "mechanism, no CPU arm — see the kernel's own JSDoc, engine/utils/encode.ts.",
    },
    uniformLoad: {
        reason:
            "raw-WGSL leaf over ptr<workgroup, u32>: a workgroup pointer has no CPU meaning (the kernel's " +
            "own JSDoc, engine/utils/tgsl.ts) — workgroupUniformLoad is a control-barrier primitive with " +
            "no CPU-side analogue to compare against.",
    },
    compareExchange: {
        reason:
            "raw-WGSL leaf over ptr<storage, atomic<u32>, read_write>: atomicCompareExchangeWeak is a " +
            "device-memory compare-and-swap with no CPU-side semantics to reproduce — GPU-only per the " +
            "kernel's own JSDoc, engine/utils/tgsl.ts.",
    },
    applyGrade: {
        test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "applyGrade" },
    },
    applySaturation: {
        test: {
            file: "packages/shallot/src/standard/glaze/glaze.test.ts",
            symbol: "applySaturation",
        },
    },
    applyVignette: {
        test: {
            file: "packages/shallot/src/standard/glaze/glaze.test.ts",
            symbol: "applyVignette",
        },
    },
    bayer4: {
        test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "bayer4" },
    },
    bitcastF32toU32: {
        test: { file: "packages/shallot/src/engine/utils/tgsl.test.ts", symbol: "bitcastF32toU32" },
    },
    brdf: { test: { file: "packages/shallot/src/standard/sear/shade.test.ts", symbol: "brdf" } },
    brdfSphere: {
        test: { file: "packages/shallot/src/standard/sear/shade.test.ts", symbol: "brdfSphere" },
    },
    bvhAnyHit: {
        reason: "transitive caller of a raw-WGSL leaf: bvhAnyHit calls bvhSlab, which calls the WGSL-bodied bvhNodeMin/bvhNodeMax/bvhLeft/bvhRight readers that read a `nodes` global the consumer declares by name (traverse.ts:114-142) — no CPU spelling exists for that binding. bvhAnyHit also reads its own tgpu.privateVar restart-trail/short-stack state (traverse.ts:183-188), a GPU-only address space with no CPU semantics.",
    },
    bvhClosestHit: {
        reason: "transitive caller of the same raw-WGSL node-reader leaves as bvhAnyHit (bvhNodeMin/bvhNodeMax/bvhLeft/bvhRight, reading the consumer-declared `nodes` global with no CPU spelling), plus its own tgpu.privateVar far-child stack (bvhStack) — GPU-only address space.",
    },
    bvhRoot: {
        test: { file: "packages/shallot/src/standard/bvh/traverse.test.ts", symbol: "bvhRoot" },
    },
    clusterCell: {
        test: {
            file: "packages/shallot/src/standard/render/cluster.test.ts",
            symbol: "clusterCell",
        },
    },
    clusterOf: {
        gap: "a CPU differential would call clusterOf's pure vec/branch math against a hand-computed froxel index for a given fragCoord/near/far/cluster config; none written",
    },
    collideBoxBox: {
        test: { file: "packages/shallot/tests/avbd/sat.oracle.ts", symbol: "collideBoxBox" },
    },
    collideRounded: {
        test: {
            file: "packages/shallot/tests/avbd/rounded.oracle.ts",
            symbol: "collideRounded",
            alias: "tgslCollideRounded",
        },
    },
    collideRoundedPolytope: {
        gap: "a CPU differential would run collideRoundedPolytope against the closed-form capsule/sphere-vs-polytope reference the way rounded.oracle.ts already does for collideRounded; none written. Its siblings collideBoxBox and collideRounded both carry oracle-tier differentials, so nothing about the shape forbids one here.",
    },
    decodePos: {
        gap: "a CPU differential would call decodePos on a quantized (w0, w1, MeshQuant) triple and compare against the analytic dequantize (q.posOffset + t*q.posScale for each axis); none written",
    },
    decodeUv: {
        gap: "a CPU differential would call decodeUv on a quantized w3 word and a MeshQuant and compare against the analytic uv dequantize; none written",
    },
    distanceAttenuation: {
        test: {
            file: "packages/shallot/src/standard/render/lighting.test.ts",
            symbol: "distanceAttenuation",
        },
    },
    distributionGGX: {
        gap: "a CPU differential would compare distributionGGX(ndh, a) against the closed-form GGX/Trowbridge-Reitz density for a hand-picked (ndh, a) pair; none written",
    },
    ditherPosterizeL: {
        test: {
            file: "packages/shallot/src/standard/glaze/glaze.test.ts",
            symbol: "ditherPosterizeL",
        },
    },
    encodePos: {
        gap: "a CPU differential would call encodePos on a world position + MeshQuant and compare against decodePos's inverse (round-trip within the unorm16 lattice's 1 LSB); none written",
    },
    encodeUv: {
        gap: "a CPU differential would call encodeUv on a uv + MeshQuant and compare against decodeUv's inverse; none written",
    },
    fogComposite: {
        test: { file: "packages/shallot/src/standard/fog/march.test.ts", symbol: "fogComposite" },
    },
    fogDensity: {
        test: { file: "packages/shallot/src/standard/fog/march.test.ts", symbol: "fogDensity" },
    },
    fogTransmittance: {
        test: {
            file: "packages/shallot/src/standard/fog/march.test.ts",
            symbol: "fogTransmittance",
        },
    },
    fresnelSchlick: {
        test: {
            file: "packages/shallot/src/standard/sear/shade.test.ts",
            symbol: "fresnelSchlick",
        },
    },
    halfLambert: {
        test: { file: "packages/shallot/src/standard/sear/shade.test.ts", symbol: "halfLambert" },
    },
    henyeyGreenstein: {
        test: {
            file: "packages/shallot/src/standard/fog/march.test.ts",
            symbol: "henyeyGreenstein",
        },
    },
    idiv: { test: { file: "packages/shallot/src/engine/utils/tgsl.test.ts", symbol: "idiv" } },
    ign: {
        gap: "a CPU differential would call ign(vec2f) on a handful of pixel coordinates and compare against the hand-computed interleaved-gradient-noise fraction; none written",
    },
    inScatterContribution: {
        test: {
            file: "packages/shallot/src/standard/fog/march.test.ts",
            symbol: "inScatterContribution",
        },
    },
    interleaveBits32: {
        test: {
            file: "packages/shallot/src/standard/bvh/morton.test.ts",
            symbol: "interleaveBits32",
        },
    },
    lightFactor: {
        gap: "a CPU differential would call lightFactor(normal) with a fixed lighting/pointLights population and assert against a hand-summed ambient+sun+point value; none written",
    },
    linearToSrgb: {
        gap: "a CPU differential would compare linearToSrgb against the IEC 61966-2-1 piecewise reference (the 12.92x low branch, the 1.055*x^(1/2.4)-0.055 high branch, split at 0.0031308); none written",
    },
    linearToSrgb1: {
        gap: "a CPU differential would call linearToSrgb1 on a swept [0,1] domain and compare against the IEC 61966-2-1 sRGB transfer reference; none written",
    },
    lineFs: {
        reason: "calls std.fwidth, a screen-space derivative intrinsic that requires quad-based fragment execution — no CPU semantics for a per-pixel derivative.",
    },
    lineQuad: {
        test: { file: "packages/shallot/src/extras/lines/lines.test.ts", symbol: "lineQuad" },
    },
    lineVs: {
        reason: "reads bound GPU resources through the surface's layout slots: lineLayout.$.lineSegments (a storage array binding) and engineLayout.$.view.viewProj (a uniform binding) — nothing on the CPU to bind to those slots.",
    },
    lit: {
        gap: "a CPU differential would call lit(baseColor, normal) and assert it equals baseColor * a hand-computed lightFactor; none written",
    },
    litPbr: {
        gap: "a CPU differential would call litPbr(Pbr, normal, world) against a hand-computed Cook-Torrance value (or against brdf at radius 0) and assert agreement; none written",
    },
    meshIdOf: {
        test: { file: "packages/shallot/src/engine/utils/encode.test.ts", symbol: "meshIdOf" },
    },
    mortonCode: {
        test: { file: "packages/shallot/src/standard/bvh/morton.test.ts", symbol: "mortonCode" },
    },
    octDecodeNormal: {
        test: {
            file: "packages/shallot/src/engine/utils/encode.test.ts",
            symbol: "octDecodeNormal",
        },
    },
    oklabL: {
        test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "oklabL" },
    },
    orderU32: {
        test: { file: "packages/shallot/src/standard/bvh/bounds.test.ts", symbol: "orderU32" },
    },
    packHdrColor: {
        gap: "a CPU differential would call packHdrColor on a swept rgb domain and compare each channel's r11/g11/b10 bit lanes against a hand-computed f16-pack-then-mantissa-drop reference; none written",
    },
    packLdrColor: {
        test: { file: "packages/shallot/src/engine/utils/encode.test.ts", symbol: "packLdrColor" },
    },
    packQuatSnorm16x4: {
        test: {
            file: "packages/shallot/src/engine/utils/encode.test.ts",
            symbol: "packQuatSnorm16x4",
        },
    },
    packSnorm2x16: {
        test: { file: "packages/shallot/src/engine/utils/tgsl.test.ts", symbol: "packSnorm2x16" },
    },
    packUnorm2x16: {
        test: { file: "packages/shallot/src/engine/utils/tgsl.test.ts", symbol: "packUnorm2x16" },
    },
    packUnorm4x8: {
        test: { file: "packages/shallot/src/engine/utils/tgsl.test.ts", symbol: "packUnorm4x8" },
    },
    pointFaceOf: {
        test: { file: "packages/shallot/src/standard/sear/shadows.test.ts", symbol: "pointFaceOf" },
    },
    pointFactor: {
        gap: "a CPU differential would call pointFactor(normal) with a populated lightGrid/lightIndices/pointLights and assert against a hand-summed clustered point contribution; none written",
    },
    pointReceiver: {
        test: {
            file: "packages/shallot/src/standard/sear/shadows.test.ts",
            symbol: "pointReceiver",
        },
    },
    pointShadowStub: {
        gap: "a CPU differential would call pointShadowStub(light, normal, fragWorld) for arbitrary inputs and assert it always returns 1; none written",
    },
    polyMake: {
        test: { file: "packages/shallot/src/standard/avbd/collide.test.ts", symbol: "polyMake" },
    },
    reconstructWorld: {
        test: {
            file: "packages/shallot/src/standard/fog/march.test.ts",
            symbol: "reconstructWorld",
        },
    },
    sampleSky: {
        test: { file: "packages/shallot/src/extras/sky/sky.test.ts", symbol: "sampleSky" },
    },
    sampleStars: {
        gap: 'a CPU differential would call sampleStars(dir, intensity, amount) directly and compare the hash-grid star color against a hand-computed reference cell/brightness; none written — sky.test.ts only greps the resolved WGSL for "fn sampleStars(", a structural check, not a call.',
    },
    sampleSunShadow: {
        reason: "raw-WGSL leaf reading shadowMap/shadowSamp/sunShadow as consumer-declared globals (texture + sampler + shadow-atlas uniform), so there is no CPU arm to call.",
    },
    screenCorner: {
        test: { file: "packages/shallot/src/extras/sprite/sprite.test.ts", symbol: "screenCorner" },
    },
    sdfToSignedDistance: {
        test: {
            file: "packages/shallot/src/extras/text/text.test.ts",
            symbol: "sdfToSignedDistance",
        },
    },
    spotFactor: {
        test: {
            file: "packages/shallot/src/standard/render/lighting.test.ts",
            symbol: "spotFactor",
        },
    },
    srgbToLinear1: {
        gap: "a CPU differential would call srgbToLinear1 on a swept [0,1] domain and compare against the IEC 61966-2-1 sRGB-to-linear transfer reference; none written",
    },
    sunInScatter: {
        test: { file: "packages/shallot/src/standard/fog/march.test.ts", symbol: "sunInScatter" },
    },
    textSrgbToLinear: {
        test: { file: "packages/shallot/src/extras/text/text.test.ts", symbol: "textSrgbToLinear" },
    },
    tgslCanary: {
        gap: "a CPU differential would compare tgslCanary(x) against the plain x+1 it computes; none written — every use resolves it to WGSL text (tgpu.resolve) to prove the build transform ran, never calls it as a function",
    },
    tmAces: {
        test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "tmAces" },
    },
    tmAgx: { test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "tmAgx" } },
    tmAgxContrast: {
        gap: "a CPU differential would call tmAgxContrast on a log-encoded probe and compare the degree-6 polynomial against a hand-computed value; none written",
    },
    tmLuma: {
        test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "tmLuma" },
    },
    tmNeutral: {
        test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "tmNeutral" },
    },
    tmReinhard: {
        test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "tmReinhard" },
    },
    tmReinhardLuminance: {
        test: {
            file: "packages/shallot/src/standard/glaze/glaze.test.ts",
            symbol: "tmReinhardLuminance",
        },
    },
    tmRgbToYcbcr: {
        gap: "a CPU differential would call tmRgbToYcbcr on a known RGB triple and compare against the Rec. 709 YCbCr matrix computed by hand; none written",
    },
    tmRrtOdtFit: {
        gap: "a CPU differential would call tmRrtOdtFit on an ACES-space probe and compare the rational polynomial fit against a hand-computed reference; none written",
    },
    tmSbCurve: {
        gap: "a CPU differential would call tmSbCurve on a scalar luma and compare against 1 - exp(-v) computed directly; none written",
    },
    tmSbCurve3: {
        gap: "a CPU differential would call tmSbCurve3 on a color and compare against tmSbCurve applied per channel; none written",
    },
    tmSomewhatBoring: {
        test: {
            file: "packages/shallot/src/standard/glaze/glaze.test.ts",
            symbol: "tmSomewhatBoring",
        },
    },
    tonemap: {
        test: { file: "packages/shallot/src/standard/glaze/glaze.test.ts", symbol: "tonemap" },
    },
    unorderU32: {
        test: { file: "packages/shallot/src/standard/bvh/bounds.test.ts", symbol: "unorderU32" },
    },
    unpackHdrColor: {
        gap: "a CPU differential would call unpackHdrColor(p) and compare against a hand-decoded r11g11b10ufloat unpack (bit-slice the u32, run each 11/10-bit field through unpack2x16float); none written",
    },
    unpackLdrColor: {
        gap: "a CPU differential would call unpackLdrColor(p) and compare against a hand-unpacked pack4x8unorm plus the sRGB-to-linear transfer curve; none written",
    },
    unpackQuatSnorm16x4: {
        test: {
            file: "packages/shallot/src/engine/utils/encode.test.ts",
            symbol: "unpackQuatSnorm16x4",
        },
    },
    unpackSnorm2x16: {
        test: { file: "packages/shallot/src/engine/utils/tgsl.test.ts", symbol: "unpackSnorm2x16" },
    },
    unpackUnorm2x16: {
        test: { file: "packages/shallot/src/engine/utils/tgsl.test.ts", symbol: "unpackUnorm2x16" },
    },
    viewDepth: {
        test: { file: "packages/shallot/src/standard/sear/shade.test.ts", symbol: "viewDepth" },
    },
    visible: {
        reason: "reads its transform, mesh-bounds, cull-params, and cull-volumes exclusively through cullLayout.$ — a tgpu bind-group-layout accessor bound to GPU storage/uniform buffers — so the frustum test has no CPU-side data to run against; there is nothing to sample without a bound GPU resource.",
    },
    visSmithGGX: {
        gap: "a CPU differential would compare visSmithGGX(ndl, ndv, a) against the closed-form Smith height-correlated visibility term for a hand-picked (ndl, ndv, a) triple; none written",
    },
    worldCorner: {
        test: { file: "packages/shallot/src/extras/sprite/sprite.test.ts", symbol: "worldCorner" },
    },
    xformMat: {
        gap: "a CPU differential would call xformMat(x) and compare its columns against a hand-composed T*R*S matrix (or against xformPoint/xformNormal's own column extraction); none written",
    },
    xformNormal: {
        gap: "a CPU differential would call xformNormal(x, n) and compare against a hand-computed inverse-transpose R*S⁻¹ applied to n, including the zero-scale-axis degenerate case; none written",
    },
    xformPoint: {
        gap: "a CPU differential would call xformPoint(x, p) and compare against a hand-composed T*R*S applied to p; none written",
    },
    xformQuat: {
        test: { file: "packages/shallot/src/engine/utils/encode.test.ts", symbol: "xformQuat" },
    },
    yLockedCorner: {
        test: {
            file: "packages/shallot/src/extras/sprite/sprite.test.ts",
            symbol: "yLockedCorner",
        },
    },
};

export type DifferentialFindingKind =
    | "stale-differential-key"
    | "missing-differential-reason"
    | "missing-differential-gap-note"
    | "missing-differential-test-file"
    | "missing-differential-test-symbol"
    | "kernel-without-differential-entry";

export interface DifferentialFinding {
    kind: DifferentialFindingKind;
    detail: string;
}

/** does `symbol` appear in call-position syntax somewhere in `text`? Half of the registry's
 *  real-filesystem arm, pure so it pins both directions without a file. Call syntax rules out a bare
 *  mention — an unused import, a name with no trailing `(` — but **not** a call-shaped mention that
 *  never calls the real kernel: a `toContain("fn sampleStars(")` string literal, a `// symbol(...)`
 *  comment, or a same-named local `function symbol(` definition all satisfy this regex while calling
 *  nothing real. {@link importsSymbol} closes that residual — a row is validated by both together, not
 *  this one alone. */
export function callsSymbol(text: string, symbol: string): boolean {
    return new RegExp(`\\b${symbol}\\s*\\(`).test(text);
}

/** does `text` contain a real ES import specifier binding local name `local` to the export named
 *  `exported`? Parses every `import { ... } from "...";` clause's specifier list (multi-line safe,
 *  since named imports commonly wrap) rather than grepping the whole file for the name, so it can't be
 *  satisfied by the name showing up outside an import clause — a string literal, a comment, or a
 *  same-named local declaration, the three ways {@link callsSymbol} alone is satisfiable without a
 *  real call. For a bare (non-aliased) registry row, `exported === local === symbol`; for an aliased
 *  row, `local` is the declared `alias`.
 *
 *  The residual it does *not* close: it matches the specifier name, never the `from` clause, so a
 *  binding of the right name from the wrong module satisfies it. `rounded.oracle.ts` is the live
 *  instance — it imports the real kernel aliased (`collideRounded as tgslCollideRounded`, from
 *  `src/standard/avbd/collide`) and separately imports its own f64 reference under the bare name from
 *  `./rounded`. That row is honest only because it declares the `alias`; a bare row sharing a file
 *  with an unrelated same-named import would pass both checks against the wrong function. Closing it
 *  needs the kernel's defining module carried per row, which the registry doesn't hold. */
export function importsSymbol(text: string, exported: string, local: string): boolean {
    const importClause = /import\s*\{([^}]*)\}\s*from\s*["'][^"']*["']/g;
    for (const match of text.matchAll(importClause)) {
        const specifiers = match[1]
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean);
        for (const spec of specifiers) {
            const aliased = spec.match(/^(?:type\s+)?([\w$]+)\s+as\s+([\w$]+)$/);
            if (aliased) {
                if (aliased[1] === exported && aliased[2] === local) return true;
            } else {
                const bare = spec.replace(/^type\s+/, "");
                if (bare === exported && exported === local) return true;
            }
        }
    }
    return false;
}

/** the kernels declaring a `gap` — no CPU differential written, none forbidden. Sorted, so
 *  `standards.test.ts` can freeze the list itself rather than a count: a count golden goes green on
 *  a swap (one gap closed, one new kernel taking the arm), and the whole point of the arm is that
 *  each occupant is named. */
export function gapKernels(registry: DifferentialRegistry): string[] {
    return Object.entries(registry)
        .filter(([, entry]) => "gap" in entry)
        .map(([name]) => name)
        .sort();
}

/** the both-directions check over `{kernels, registry}`, pure — no filesystem, no dynamic import. The
 *  real-filesystem half ("the named file exists, imports its kernel, and calls it") lives in
 *  `standards.test.ts` beside the population walk, per the same split `checkStandards` uses.
 *
 *  Six finding kinds: a registry key naming a kernel absent from the population
 *  (`stale-differential-key`); a can't-have entry with an empty reason
 *  (`missing-differential-reason`); a gap entry with an empty note
 *  (`missing-differential-gap-note`); a test entry with an empty file or symbol
 *  (`missing-differential-test-file` / `-symbol`); and a live kernel with no registry row at all
 *  (`kernel-without-differential-entry`) — the corpus-wide completeness direction. */
export function checkDifferentials(
    kernels: readonly string[],
    registry: DifferentialRegistry,
): DifferentialFinding[] {
    const findings: DifferentialFinding[] = [];
    const population = new Set(kernels);

    for (const [name, entry] of Object.entries(registry)) {
        if (!population.has(name)) {
            findings.push({ kind: "stale-differential-key", detail: name });
            continue;
        }
        if ("reason" in entry) {
            if (!entry.reason || entry.reason.trim() === "") {
                findings.push({ kind: "missing-differential-reason", detail: name });
            }
        } else if ("gap" in entry) {
            if (!entry.gap || entry.gap.trim() === "") {
                findings.push({ kind: "missing-differential-gap-note", detail: name });
            }
        } else {
            if (!entry.test.file || entry.test.file.trim() === "") {
                findings.push({ kind: "missing-differential-test-file", detail: name });
            }
            if (!entry.test.symbol || entry.test.symbol.trim() === "") {
                findings.push({ kind: "missing-differential-test-symbol", detail: name });
            }
        }
    }

    for (const name of population) {
        if (!Object.hasOwn(registry, name)) {
            findings.push({ kind: "kernel-without-differential-entry", detail: name });
        }
    }

    return findings;
}
