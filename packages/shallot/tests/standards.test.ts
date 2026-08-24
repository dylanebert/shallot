import { describe, expect, test } from "bun:test";
import { join, resolve } from "node:path";
import tgpu, { isTgpuFn } from "typegpu";
import {
    callsSymbol,
    checkDifferentials,
    checkStandards,
    DEFAULT_CHECKS,
    DIFFERENTIAL_REGISTRY,
    type DifferentialFinding,
    type DisciplineCheck,
    type Finding,
    gapKernels,
    importsSymbol,
    type Population,
    STANDARDS_REGISTRY,
    violation,
} from "./standards";
import { TEST_TIER_SUFFIXES } from "./test-tiers";

// The real-filesystem/real-import/real-resolve seam over the pure `checkStandards` (`./standards.ts`):
// walk `packages/shallot/src`, dynamic-import every module, duck-detect live TGSL kernels with a
// tighter predicate than a bare duck-test (which also catches data schemas like `MeshQuant`) —
// TypeGPU's own `isTgpuFn`, identity-dedup, resolve each kernel to WGSL, and
// run the discipline checks against the real text — producing the `Population` the pure checker consumes.

const SRC_DIR = join(import.meta.dir, "../src");

/** every `.ts` module under `src`, repo-relative path, excluding the test-tier suffixes
 *  (`testing.md`'s tier convention, via the shared `test-tiers.ts` constant) plus `.fixture.ts`,
 *  which is not a tier `testing.md` names — it's a test-data file (the tumble step fixture), excluded
 *  from the kernel walk alongside the tiers, never a second hand-written tier list — and `.d.ts`
 *  declaration files, which have no runtime module body (dynamic-importing one throws on whatever
 *  ambient global it declares, e.g. `draco_wasm_wrapper.d.ts` -> `DracoDecoderModule is not defined`),
 *  which is a false import failure, not a real one. */
async function sourceModules(): Promise<string[]> {
    const out: string[] = [];
    for await (const p of new Bun.Glob("**/*.ts").scan({ cwd: SRC_DIR })) {
        if (/\.d\.ts$/.test(p)) continue;
        if (TEST_TIER_SUFFIXES.test(p)) continue;
        // `.fixture.ts` is not a tier `testing.md` names — it's a test-data file, excluded from the
        // kernel walk as roster-plus-named-extras, never a second hand-written tier list.
        if (/\.fixture\.ts$/.test(p)) continue;
        out.push(p);
    }
    return out.sort();
}

/** a TGSL kernel exported from `src`: the tighter predicate is TypeGPU's own `isTgpuFn` (marks the
 *  `resourceType === "function"` shell `tgpu.fn(...)` produces) rather than the population probe's duck
 *  test (which also matched `d.struct(...)` data schemas, e.g. `MeshQuant` — a schema, not a kernel: it
 *  carries no resolvable body, and running discipline checks against its resolved form is meaningless).
 *  `isTgpuFn` returns false for a data schema (`d.struct` sets no `resourceType: "function"`) and for a
 *  compute/vertex/fragment entry point (`isTgpuComputeFn` etc., a different `resourceType`) — the
 *  population is exactly the `tgpu.fn` CPU-callable seams `wgsl.ts`'s own header describes ("the
 *  device-free seam every ported kernel exposes as a `*Wgsl()` function"). */
interface Export {
    file: string;
    name: string;
    value: unknown;
}

async function walkExports(files: readonly string[]): Promise<Export[]> {
    const out: Export[] = [];
    for (const file of files) {
        const mod = (await import(join(SRC_DIR, file))) as Record<string, unknown>;
        for (const [name, value] of Object.entries(mod)) {
            if (isTgpuFn(value)) out.push({ file, name, value });
        }
    }
    return out;
}

/** identity-dedup: 53 of the 154 exported symbols found (measured 2026-08-06, this predicate) are the
 *  same object re-exported through a barrel — keying by `module#name` would multiply every row and let a
 *  barrel rename silently orphan an exemption (Locked decision). Canonical name per identity: the
 *  shortest path among candidates that isn't a barrel (`index.ts`) — the barrel is definitionally a
 *  re-export, so the non-barrel candidate is closer to the definition site; ties broken by shortest path,
 *  then lexicographic, for determinism. */
function canonicalize(exports: readonly Export[]): Map<unknown, Export> {
    const byIdentity = new Map<unknown, Export[]>();
    for (const e of exports) {
        const group = byIdentity.get(e.value);
        if (group) group.push(e);
        else byIdentity.set(e.value, [e]);
    }
    const canonical = new Map<unknown, Export>();
    for (const [identity, group] of byIdentity) {
        const nonBarrel = group.filter(
            (e) => !e.file.endsWith("/index.ts") && e.file !== "index.ts",
        );
        const candidates = nonBarrel.length > 0 ? nonBarrel : group;
        candidates.sort((a, b) => a.file.length - b.file.length || a.file.localeCompare(b.file));
        canonical.set(identity, candidates[0]);
    }
    return canonical;
}

/** kernel name -> checks that actually fail against its real resolved WGSL. `tgpu.resolve` on a single
 *  kernel standalone is the seam the Locked decision measured at 255/255 success — a resolve failure here
 *  would be a real defect, not something to swallow, so it's asserted, not caught. */
async function population(kernels: ReadonlyMap<string, Export>): Promise<Population> {
    const out: Population = {};
    for (const [name, kernel] of kernels) {
        const wgsl = tgpu.resolve([kernel.value] as never, { names: "strict" });
        const checks = [...DEFAULT_CHECKS, ...(STANDARDS_REGISTRY[name]?.also ?? [])];
        const failing = checks.filter((c) => violation(c, wgsl) !== null);
        out[name] = failing;
    }
    return out;
}

async function namedKernels(): Promise<Map<string, Export>> {
    const files = await sourceModules();
    const exports = await walkExports(files);
    const canonical = canonicalize(exports);
    const byName = new Map<string, Export>();
    for (const e of canonical.values()) {
        // the registry keys on the bare export name, so two distinct-identity kernels sharing one name
        // would silently overwrite — one kernel leaving the population with nothing red. 0 collisions
        // measured 2026-08-06; this throws the day that changes rather than shrinking the corpus.
        const prior = byName.get(e.name);
        if (prior) {
            throw new Error(`kernel name collision: "${e.name}" in ${prior.file} and ${e.file}`);
        }
        byName.set(e.name, e);
    }
    return byName;
}

describe("TGSL corpus standards", () => {
    test("the enumerator resolves the live population with no import or resolve failures", async () => {
        const kernels = await namedKernels();
        // exact by mechanism (a deterministic walk over committed source), so it's frozen like a byte
        // or pipeline-count golden rather than floored — a loose bound passes a walk that silently
        // drops half the corpus, which is the hole this meta-test exists to close. Bump it deliberately
        // in the same commit that adds or removes a kernel.
        expect(kernels.size).toBe(101);
        for (const [name, kernel] of kernels) {
            expect(
                () => tgpu.resolve([kernel.value] as never, { names: "strict" }),
                name,
            ).not.toThrow();
        }
    });

    test("no registry key is stale, no exemption reason is missing, no exemption shadows a passing check", async () => {
        const kernels = await namedKernels();
        const pop = await population(kernels);
        const findings = checkStandards(pop, STANDARDS_REGISTRY);
        const structural = findings.filter((f) => f.kind !== "unexempted-violation");
        expect(structural as Finding[]).toEqual([]);
    });

    // the corpus-wide violation direction: every live kernel's discipline reds are either fixed at the
    // source or carry a (kernel, check) exemption whose reason claims the load-bearing property
    // (testing.md, STANDARDS_REGISTRY). A new kernel violating a check reds here with no opt-in.
    test("every live kernel's discipline violations are fixed or exempted", async () => {
        const kernels = await namedKernels();
        const pop = await population(kernels);
        const unexempted = checkStandards(pop, STANDARDS_REGISTRY).filter(
            (f) => f.kind === "unexempted-violation",
        );
        expect(unexempted).toEqual([]);
    });
});

// the differential registry's real-filesystem half: "the named file exists and references
// its kernel symbol" needs the real filesystem, so it lives here beside the population walk, not in
// the pure `standards.ts` (which stays filesystem-free per checkDifferentials's own contract).
/** the kernels declaring `gap` — no CPU differential written, none forbidden. Frozen by name, not by
 *  count, so closing one gap while a new kernel takes the arm still reds (`gapKernels`'s own note).
 *  Writing these differentials is deliberately not part of the registry; making the list exist is.
 *  Shrinking it is the follow-on work. */
const GAP_GOLDEN = [
    "clusterOf",
    "collideRoundedPolytope",
    "decodePos",
    "decodeUv",
    "distributionGGX",
    "encodePos",
    "encodeUv",
    "ign",
    "lightFactor",
    "linearToSrgb",
    "linearToSrgb1",
    "lit",
    "litPbr",
    "packHdrColor",
    "pointFactor",
    "pointShadowStub",
    "sampleStars",
    "srgbToLinear1",
    "tgslCanary",
    "tmAgxContrast",
    "tmRgbToYcbcr",
    "tmRrtOdtFit",
    "tmSbCurve",
    "tmSbCurve3",
    "unpackHdrColor",
    "unpackLdrColor",
    "visSmithGGX",
    "xformMat",
    "xformNormal",
    "xformPoint",
];

describe("differential registry (real filesystem)", () => {
    const root = resolve(import.meta.dir, "../../..");

    test("no differential registry key is stale, no can't-have reason is missing, no test entry is missing a file or symbol", async () => {
        const kernels = await namedKernels();
        const findings = checkDifferentials([...kernels.keys()], DIFFERENTIAL_REGISTRY);
        const structural = findings.filter((f) => f.kind !== "kernel-without-differential-entry");
        expect(structural as DifferentialFinding[]).toEqual([]);
    });

    test("every declared differential test file exists, imports its kernel, and calls it", async () => {
        for (const [name, entry] of Object.entries(DIFFERENTIAL_REGISTRY)) {
            if (!("test" in entry)) continue;
            const path = join(root, entry.test.file);
            const file = Bun.file(path);
            const exists = await file.exists();
            expect(exists, `${entry.test.file} (for ${name}) does not exist`).toBe(true);
            if (!exists) continue;
            const text = await file.text();
            const called = entry.test.alias ?? entry.test.symbol;
            expect(
                callsSymbol(text, called),
                `${entry.test.file} never calls "${called}" (for ${name})`,
            ).toBe(true);
            // run over every row uniformly, bare or aliased — callsSymbol alone is satisfiable by a
            // string literal, a comment, or a same-named local shadow (`testing.md`'s exemption-reason
            // law's sibling limit); a real import binding is what rules those out.
            expect(
                importsSymbol(text, entry.test.symbol, called),
                `${entry.test.file} does not import ${entry.test.symbol}${entry.test.alias ? ` as ${entry.test.alias}` : ""} (for ${name})`,
            ).toBe(true);
        }
    });

    // the corpus-wide completeness direction: every live kernel declares a differential test, a
    // can't-have mechanism, or a gap. A kernel added tomorrow reds here until it answers.
    test("every live kernel has a declared differential entry", async () => {
        const kernels = await namedKernels();
        const findings = checkDifferentials([...kernels.keys()], DIFFERENTIAL_REGISTRY);
        const missing = findings.filter((f) => f.kind === "kernel-without-differential-entry");
        expect(missing).toEqual([]);
    });

    // the gap arm is the cheap one to reach for, so its occupants are frozen by name rather than by
    // count: a count golden goes green on a swap (one gap closed, one new kernel quietly taking the
    // arm), and naming them is what makes the port's untested surface enumerable — the defect this
    // spec exists to close. Writing these differentials is deliberately out of scope here; each row
    // says what one would compare against. Editing the list is the deliberate act, in the same
    // commit as the kernel or test that moved it.
    test("the declared differential gaps are exactly the frozen list", () => {
        expect(gapKernels(DIFFERENTIAL_REGISTRY)).toEqual(GAP_GOLDEN);
    });
});

/** pure-checker fixture proofs: each finding kind, red-provable with no filesystem. */
describe("checkStandards (fixture-only)", () => {
    test("stale-exemption-key: an exemption for a kernel no longer in the population", () => {
        const findings = checkStandards(
            { liveKernel: [] },
            { renamedKernel: { exempt: { integerDiscipline: "renamed from liveKernel" } } },
        );
        expect(findings).toEqual([{ kind: "stale-exemption-key", detail: "renamedKernel" }]);
    });

    test("missing-exemption-reason: an exemption with an empty reason", () => {
        const findings = checkStandards(
            { k: ["integerDiscipline"] },
            { k: { exempt: { integerDiscipline: "" } } },
        );
        expect(findings).toContainEqual({
            kind: "missing-exemption-reason",
            detail: "k:integerDiscipline",
        });
    });

    test("exempt-shadows-passing: an exemption for a check that doesn't actually fail", () => {
        const findings = checkStandards(
            { k: [] },
            { k: { exempt: { integerDiscipline: "a real reason" } } },
        );
        expect(findings).toEqual([
            { kind: "exempt-shadows-passing", detail: "k:integerDiscipline" },
        ]);
    });

    test("unexempted-violation: a live kernel fails a check with no exemption", () => {
        const findings = checkStandards({ k: ["pointerDiscipline"] }, {});
        expect(findings).toEqual([{ kind: "unexempted-violation", detail: "k:pointerDiscipline" }]);
    });

    test("an exemption clears its own (kernel, check) violation and only that pair", () => {
        const findings = checkStandards(
            { k: ["pointerDiscipline", "integerDiscipline"] },
            { k: { exempt: { pointerDiscipline: "real reason" } } },
        );
        expect(findings).toEqual([{ kind: "unexempted-violation", detail: "k:integerDiscipline" }]);
    });

    test("checkStandards is empty over an empty population and an empty registry", () => {
        expect(checkStandards({}, {})).toEqual([]);
    });

    test("a registry key shaped like an Object.prototype property still reads as stale, not a crash", () => {
        const findings = checkStandards(
            {},
            { constructor: { exempt: { integerDiscipline: "renamed away" } } },
        );
        expect(findings).toEqual([{ kind: "stale-exemption-key", detail: "constructor" }]);
    });
});

/** `violation()` proofs: each of the four discipline checks, red-provable against a mutated kernel, at
 *  the resolved-WGSL granularity the meta-test itself uses (Validation: "No check is satisfiable without
 *  its property"). */
describe("violation() per discipline check", () => {
    const bad: Record<DisciplineCheck, string> = {
        noIntegerDivision: "fn f() -> f32 { return f32(a) / f32(b); }",
        integerDiscipline: "fn f() -> u32 { return i32(3); }",
        pointerDiscipline: "fn f() -> f32 { let n = 3.0; let m = (&n); return m; }",
        noDivision: "fn f() -> f32 { return a / b; }",
    };
    const good: Record<DisciplineCheck, string> = {
        noIntegerDivision: "fn f() -> f32 { return a * b; }",
        integerDiscipline: "fn f() -> u32 { return a + b; }",
        pointerDiscipline: "fn f() -> f32 { var n = 3.0; let m = (&n); return m; }",
        noDivision: "fn f() -> f32 { return a * idivWgsl(x, y); }",
    };

    for (const check of Object.keys(bad) as DisciplineCheck[]) {
        test(`${check} reds on a mutated kernel`, () => {
            expect(violation(check, bad[check])).not.toBeNull();
        });
        test(`${check} holds on a clean kernel`, () => {
            expect(violation(check, good[check])).toBeNull();
        });
    }
});

/** pure-checker fixture proofs for {@link checkDifferentials}, each finding kind red-provable with no
 *  filesystem — mirrors `checkStandards (fixture-only)` above. */
describe("checkDifferentials (fixture-only)", () => {
    test("stale-differential-key: a registry entry for a kernel no longer in the population", () => {
        const findings = checkDifferentials(["liveKernel"], {
            renamedKernel: { reason: "renamed from liveKernel" },
        });
        expect(findings).toContainEqual({
            kind: "stale-differential-key",
            detail: "renamedKernel",
        });
    });

    test("missing-differential-reason: a can't-have entry with an empty reason", () => {
        const findings = checkDifferentials(["k"], { k: { reason: "" } });
        expect(findings).toContainEqual({ kind: "missing-differential-reason", detail: "k" });
    });

    test("missing-differential-test-file: a test entry with an empty file", () => {
        const findings = checkDifferentials(["k"], {
            k: { test: { file: "", symbol: "k" } },
        });
        expect(findings).toContainEqual({ kind: "missing-differential-test-file", detail: "k" });
    });

    test("missing-differential-test-symbol: a test entry with an empty symbol", () => {
        const findings = checkDifferentials(["k"], {
            k: { test: { file: "some/file.test.ts", symbol: "" } },
        });
        expect(findings).toContainEqual({ kind: "missing-differential-test-symbol", detail: "k" });
    });

    test("kernel-without-differential-entry: a live kernel with no registry row", () => {
        const findings = checkDifferentials(["k"], {});
        expect(findings).toEqual([{ kind: "kernel-without-differential-entry", detail: "k" }]);
    });

    test("a declared test entry clears its kernel and adds no finding", () => {
        const findings = checkDifferentials(["k"], {
            k: { test: { file: "some/file.test.ts", symbol: "k" } },
        });
        expect(findings).toEqual([]);
    });

    test("a declared can't-have reason clears its kernel and adds no finding", () => {
        const findings = checkDifferentials(["k"], { k: { reason: "atomics, GPU-only" } });
        expect(findings).toEqual([]);
    });

    test("missing-differential-gap-note: a gap entry with an empty note", () => {
        const findings = checkDifferentials(["k"], { k: { gap: "" } });
        expect(findings).toContainEqual({ kind: "missing-differential-gap-note", detail: "k" });
    });

    test("a declared gap clears its kernel and adds no finding", () => {
        const findings = checkDifferentials(["k"], {
            k: { gap: "nothing forbids one; unwritten" },
        });
        expect(findings).toEqual([]);
    });

    test("gapKernels names only the gap arm, sorted", () => {
        expect(
            gapKernels({
                zeta: { gap: "unwritten" },
                alpha: { gap: "unwritten" },
                beta: { reason: "raw-WGSL leaf" },
                gamma: { test: { file: "f.test.ts", symbol: "gamma" } },
            }),
        ).toEqual(["alpha", "zeta"]);
    });

    test("checkDifferentials is empty over an empty population and an empty registry", () => {
        expect(checkDifferentials([], {})).toEqual([]);
    });

    test("a kernel named like an Object.prototype key still reads as missing its row", () => {
        expect(checkDifferentials(["constructor", "toString"], {})).toEqual([
            { kind: "kernel-without-differential-entry", detail: "constructor" },
            { kind: "kernel-without-differential-entry", detail: "toString" },
        ]);
    });
});

/** {@link callsSymbol} is the property the real-filesystem arm rests on, and the one every stage-4b row
 *  is validated against — so both directions are pinned here rather than probed once by hand. */
describe("callsSymbol", () => {
    test("a direct call satisfies it, with or without a space", () => {
        expect(callsSymbol("expect(octEncodeNormal(v)).toBe(x);", "octEncodeNormal")).toBe(true);
        expect(callsSymbol("const r = collideHull (a, b);", "collideHull")).toBe(true);
    });

    test("an unused import left by a refactor does not", () => {
        expect(callsSymbol('import { decodePos } from "./encode";\nfoo();', "decodePos")).toBe(
            false,
        );
    });

    test("a mention in a comment with no call-position parens does not", () => {
        expect(callsSymbol("// TODO: differential-test clusterCell\nother();", "clusterCell")).toBe(
            false,
        );
    });

    test("a longer name sharing the prefix does not satisfy the shorter one", () => {
        expect(callsSymbol("packQuatSmallest3(q);", "packQuat")).toBe(false);
    });

    // the residual `callsSymbol` alone can't close — call syntax in the text, not a real call —
    // documented (not fixed) here because closing it is `importsSymbol`'s job, pinned below.
    test("a comment with call-position parens satisfies it — the residual importsSymbol closes", () => {
        expect(callsSymbol("// fn sampleStars(dir, intensity, amount)", "sampleStars")).toBe(true);
    });

    test("a `toContain` string literal satisfies it too", () => {
        expect(callsSymbol('expect(wgsl).toContain("fn sampleStars(");', "sampleStars")).toBe(true);
    });

    test("a same-named local function definition satisfies it too", () => {
        expect(callsSymbol("function collideRounded(a, b) { return a; }", "collideRounded")).toBe(
            true,
        );
    });
});

/** {@link importsSymbol} is what actually closes {@link callsSymbol}'s residual for the differential
 *  registry's real-filesystem arm (`standards.test.ts`'s "every declared differential test file..."):
 *  a real import binding proves the call-syntax match callsSymbol found is a genuine call on the real
 *  kernel, not a string literal, a comment, or a same-named local shadow. Applied uniformly over every
 *  `{ test }` row, bare or aliased — not only the aliased ones. */
describe("importsSymbol", () => {
    test("a bare named import satisfies it under the symbol's own name", () => {
        expect(
            importsSymbol(
                'import { octEncodeNormal } from "./encode";\noctEncodeNormal(v);',
                "octEncodeNormal",
                "octEncodeNormal",
            ),
        ).toBe(true);
    });

    test("an aliased import satisfies it only under the declared alias", () => {
        const text =
            'import { collideRounded as tgslCollideRounded } from "../src";\ntgslCollideRounded(a, b);';
        expect(importsSymbol(text, "collideRounded", "tgslCollideRounded")).toBe(true);
        expect(importsSymbol(text, "collideRounded", "collideRounded")).toBe(false);
    });

    test("a multi-line named-import block is parsed, not just a single line", () => {
        const text =
            'import {\n    octDecodeNormal,\n    octEncodeNormal,\n} from "./encode";\noctEncodeNormal(v);';
        expect(importsSymbol(text, "octEncodeNormal", "octEncodeNormal")).toBe(true);
    });

    test("sky.test.ts's toContain string literal does not satisfy it — no import statement at all", () => {
        expect(
            importsSymbol(
                'expect(wgsl).toContain("fn sampleStars(");',
                "sampleStars",
                "sampleStars",
            ),
        ).toBe(false);
    });

    test("a same-named local function definition does not satisfy it", () => {
        const text =
            'import { collideRounded as tgslCollideRounded } from "../src";\n' +
            "function collideRounded(a, b) { return a; }\n" +
            "collideRounded(a, b);";
        expect(importsSymbol(text, "collideRounded", "collideRounded")).toBe(false);
    });

    // the residual: the specifier name is matched, the `from` clause never is. This is
    // rounded.oracle.ts's real shape — the kernel imported aliased, a same-named f64 reference
    // imported bare from another module — and the bare arm passes. The row is honest only because it
    // declares the alias; pinned here so the limit is visible rather than assumed closed.
    test("a same-named import from an unrelated module satisfies it — the module path is unchecked", () => {
        const text =
            'import { collideRounded as tgslCollideRounded } from "../../src/standard/avbd/collide";\n' +
            'import { collideRounded } from "./rounded";\n' +
            "collideRounded(a, b);";
        expect(importsSymbol(text, "collideRounded", "collideRounded")).toBe(true);
    });

    test("a mention in a comment does not satisfy it", () => {
        expect(importsSymbol("// collideRounded(a, b)", "collideRounded", "collideRounded")).toBe(
            false,
        );
    });
});
