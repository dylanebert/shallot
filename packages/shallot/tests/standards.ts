import { integerDiscipline, noDivision, noIntegerDivision, pointerDiscipline } from "./wgsl";

// The declared registry + pure checker half of the TGSL-corpus meta-test (spec shallot-tgsl-standards,
// stage 2). Shape copied from `examples/gym/src/scenarios/timeouts.ts` + `coverage.ts`: plain committed
// data, a pure checker asserted both directions, red-provable against fixtures. `standards.test.ts` is
// the one real-filesystem/real-import/real-resolve seam — it walks `packages/shallot/src`, dynamic-imports
// every module, duck-detects live TGSL kernels, resolves each to WGSL, and runs the checks below against
// the real text, producing a `Population` this file's `checkStandards` consumes. This file imports
// `tests/wgsl.ts`'s existing checkers (the four discipline checks it already ships and every ported-kernel
// test file already calls) rather than reimplementing their regexes a second time — "one source of truth"
// (coding.md) outweighs a literal zero-runtime-import reading here, since the alternative is a second,
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
 *  Four finding kinds: a registry key naming a kernel no longer in the population (`stale-exemption-key`,
 *  a rename/removal orphaned the exemption); an exemption with no reason (`missing-exemption-reason`);
 *  an exemption for a (kernel, check) pair whose check doesn't actually fail — a dead exemption hiding
 *  a check that would otherwise run clean (`exempt-shadows-passing`); and a live kernel violating an
 *  applicable check with no matching exemption (`unexempted-violation`) — the corpus-wide direction stage 3
 *  triages, left as a `test.todo` in `standards.test.ts` until every current violation is fixed or exempted. */
export function checkStandards(population: Population, registry: StandardsRegistry): Finding[] {
    const findings: Finding[] = [];

    for (const [name, entry] of Object.entries(registry)) {
        if (!(name in population)) {
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

// ── Part 2: the differential registry (stage 4a) ──────────────────────────────────────────────
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
 *  anything; naming the symbol makes that a mechanically checkable claim
 *  ({@link standards.test.ts}'s real-filesystem half greps the file for `symbol(` — call syntax, not a
 *  bare mention, so an unused import or a symbol named only in a comment can't satisfy the row. The
 *  residual limit: it proves the file calls the kernel, not that it asserts against a reference. */
export interface DifferentialTest {
    file: string;
    symbol: string;
}

/** a kernel either has a named CPU differential, or a reason it can't: the mechanism that makes CPU
 *  execution meaningless for it (a raw-WGSL leaf with no CPU arm, an atomic op with no CPU semantics,
 *  a pointer into GPU-only address space), never a category ("GPU-only kernel" restates the row
 *  instead of grounding it). */
export type DifferentialEntry = { test: DifferentialTest } | { reason: string };

export type DifferentialRegistry = Record<string, DifferentialEntry>;

/** the differential registry's representative slice (stage 4a). Filling every one of the 101 kernels'
 *  rows is stage 4b's mechanical queue — this hand-authored slice exists to exercise every arm of
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
};

export type DifferentialFindingKind =
    | "stale-differential-key"
    | "missing-differential-reason"
    | "missing-differential-test-file"
    | "missing-differential-test-symbol"
    | "kernel-without-differential-entry";

export interface DifferentialFinding {
    kind: DifferentialFindingKind;
    detail: string;
}

/** the both-directions check over `{kernels, registry}`, pure — no filesystem, no dynamic import. The
 *  real-filesystem half ("the named file exists and references its kernel symbol") lives in
 *  `standards.test.ts` beside the population walk, per the same split `checkStandards` uses.
 *
 *  Five finding kinds: a registry key naming a kernel no longer in the population
 *  (`stale-differential-key`); a can't-have entry with an empty reason
 *  (`missing-differential-reason`); a test entry with an empty file or symbol
 *  (`missing-differential-test-file` / `-symbol`); and a live kernel with no registry row at all
 *  (`kernel-without-differential-entry`) — the corpus-wide completeness direction stage 4b's
 *  mechanical queue fills in, left as a `test.todo` in `standards.test.ts` until every kernel has a
 *  row. */
/** does `text` call `symbol`, as opposed to merely mentioning it? The predicate behind the registry's
 *  real-filesystem arm, pure so it pins both directions without a file. Call syntax is the bar: an
 *  unused import a refactor left behind, or a name appearing only in a comment, satisfies a bare word
 *  match while calling nothing — and this is the arm every filled row is validated against. It proves
 *  the file calls the kernel, not that it asserts the result against a reference; that residual is
 *  what the hand-authored row prose carries. */
export function callsSymbol(text: string, symbol: string): boolean {
    return new RegExp(`\\b${symbol}\\s*\\(`).test(text);
}

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
