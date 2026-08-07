import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import tgpu, { isTgpuFn } from "typegpu";
import {
    checkStandards,
    DEFAULT_CHECKS,
    type DisciplineCheck,
    type Finding,
    type Population,
    STANDARDS_REGISTRY,
    violation,
} from "./standards";

// The real-filesystem/real-import/real-resolve seam over the pure `checkStandards` (`./standards.ts`):
// walk `packages/shallot/src`, dynamic-import every module, duck-detect live TGSL kernels with a
// tighter predicate than the population probe's (`shallot-tgsl-standards` Locked decision: the probe's
// duck-test also caught data schemas like `MeshQuant`), identity-dedup, resolve each kernel to WGSL, and
// run the discipline checks against the real text — producing the `Population` the pure checker consumes.

const SRC_DIR = join(import.meta.dir, "../src");

/** every `.ts` module under `src`, repo-relative path, excluding non-source suffixes (`testing.md`'s tier
 *  convention: `.test.ts` / `.fixture.ts` / `.lab.ts`) and `.d.ts` declaration files — a `.d.ts` has no
 *  runtime module body, so dynamic-importing one throws on whatever ambient global it declares
 *  (`draco_wasm_wrapper.d.ts` -> `DracoDecoderModule is not defined`), which is a false import failure,
 *  not a real one. */
async function sourceModules(): Promise<string[]> {
    const out: string[] = [];
    for await (const p of new Bun.Glob("**/*.ts").scan({ cwd: SRC_DIR })) {
        if (/\.d\.ts$/.test(p)) continue;
        if (/\.(test|fixture|lab)\.ts$/.test(p)) continue;
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

describe("TGSL corpus standards (shallot-tgsl-standards)", () => {
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
