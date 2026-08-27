// The both-directions coverage check over `SCENARIO_GATES` + `GATE_EXEMPTIONS`
// (a declared registry asserted both directions). `checkGateEntries` is pure over plain data + a glob
// matcher, so `coverage.test.ts` can
// red-prove it against fixtures with no filesystem or scenario import; `gpuModulePopulation` is the one
// real-filesystem seam, and `scenarioNames()` (imported for its side effect via `./index`) is the one
// real-registration seam.
import { GATE_EXEMPTIONS, SCENARIO_GATES, type ScenarioGate } from "./timeouts";

/** the GPU-side module population: every `.ts` file under the paths `AGENTS.md`'s `gpu.md` rule already
 *  declares as GPU-facing engine code (`engine/runtime`, `engine/utils/encode.ts`, the render/sear/part/slab
 *  pipeline, and `extras` — `gpu.md`'s own `paths:` frontmatter), plus two additions it doesn't already
 *  list: `standard/bvh` (the acceleration-structure pipeline `gpu.md`'s own body cites repeatedly as
 *  canonical GPU code — `bounds.ts`, `build.ts`, `sort.ts`; that gap in `gpu.md`'s frontmatter has since been
 *  fixed, so the two now agree here) and `standard/avbd` (the GPU physics swap-in, `avbd.md`'s own `paths:`,
 *  deliberately not `gpu.md`'s). Reusing the codebase's own committed boundaries for "GPU-side" rather
 *  than inventing a second one (one source of truth). Deliberately excludes tumble physics: it
 *  runs on the CPU wasm kernel and is bit-exact-gated by `bun test` + the committed fixtures (`tumble.md`),
 *  so its truth already lives in a tier this check isn't responsible for; it excludes ECS/scene/document
 *  core for the same reason (`bun test` unit coverage). */
export const GPU_MODULE_GLOBS: readonly string[] = [
    "packages/shallot/src/engine/runtime/**/*.ts",
    "packages/shallot/src/engine/utils/encode.ts",
    "packages/shallot/src/standard/render/**/*.ts",
    "packages/shallot/src/standard/sear/**/*.ts",
    "packages/shallot/src/standard/part/**/*.ts",
    "packages/shallot/src/standard/slab/**/*.ts",
    "packages/shallot/src/standard/bvh/**/*.ts",
    "packages/shallot/src/standard/avbd/**/*.ts",
    // `gpu.md` writes these eight as one brace glob; `globToRegExp` has no brace support, so they
    // enumerate here. Same set — `checkExtrasClassification` is what holds the two together.
    "packages/shallot/src/extras/gltf/**/*.ts",
    "packages/shallot/src/extras/lines/**/*.ts",
    "packages/shallot/src/extras/outline/**/*.ts",
    "packages/shallot/src/extras/profile/**/*.ts",
    "packages/shallot/src/extras/skin/**/*.ts",
    "packages/shallot/src/extras/sky/**/*.ts",
    "packages/shallot/src/extras/sprite/**/*.ts",
    "packages/shallot/src/extras/text/**/*.ts",
];

/** the `extras/` modules deliberately outside the GPU population, declared with the property that puts them
 *  there — no device, buffer, or WGSL surface. `GPU_MODULE_GLOBS` enumerates the GPU-facing extras
 *  directories one by one (mirroring `gpu.md`'s frontmatter, which enumerates the GPU-facing extras
 *  directories explicitly), so a
 *  *new* extras module would fall out of the population silently rather than reding as uncovered. This table
 *  is what keeps that from happening: every immediate subdirectory of `src/extras/` is either matched by the
 *  globs or listed here. */
export const NON_GPU_EXTRAS: Record<string, string> = {
    orbit: "CPU camera math + a DOM overlay (mountOverlay) — no device, buffer, or WGSL surface",
    tween: "pure numeric timing/easing atoms driving ECS fields — no device, buffer, or WGSL surface",
};

/** stage 3b's done-signal: flip to `true` once `SCENARIO_GATES` + `GATE_EXEMPTIONS` cover every scenario
 *  and every GPU-side module, per the completeness directions below. `coverage.test.ts` skips the two
 *  completeness assertions against real data while this is `false`; turning them on and green is the
 *  measure of "3b is done", not an optional follow-up. */
export const COMPLETENESS_ENFORCED = true;

export type FindingKind =
    | "glob-matches-nothing"
    | "unregistered-table-key"
    | "missing-exemption-reason"
    | "exempt-shadows-covered"
    | "scenario-missing-entry"
    | "module-not-covered"
    | "extras-unclassified";

export interface Finding {
    kind: FindingKind;
    detail: string;
}

/** converts a `dir/**\/*.ts`-style glob to a RegExp. `*` matches within one path segment; `**` matches
 *  across segments (including zero). No brace/character-class support — the gate table's globs don't need
 *  it, and adding it speculatively would be untested surface. */
export function globToRegExp(glob: string): RegExp {
    let pattern = "";
    for (let i = 0; i < glob.length; i++) {
        const c = glob[i];
        if (c === "*" && glob[i + 1] === "*") {
            pattern += ".*";
            i++;
            if (glob[i + 1] === "/") i++;
        } else if (c === "*") {
            pattern += "[^/]*";
        } else if ("\\^$.|?+()[]{}".includes(c)) {
            pattern += `\\${c}`;
        } else {
            pattern += c;
        }
    }
    return new RegExp(`^${pattern}$`);
}

/** the per-entry directions, live against real data from 3a: every `covers` glob resolves to at least one
 *  real module, every table key names a registered scenario, every exemption carries a non-empty reason,
 *  and no exemption shadows a module a scenario already covers. That last direction is what a false or
 *  redundant exemption reason can't be checked by itself — an exemption's *reason* is prose, unverifiable,
 *  but an exemption whose target a `covers` glob already matches is a real, mechanical contradiction: the
 *  module isn't actually uncovered, so the exemption is either false or dead weight. Pure over the table +
 *  exemptions + the real scenario/module lists, so a fixture proves each direction with no filesystem or
 *  scenario import. */
export function checkGateEntries(
    table: Record<string, ScenarioGate>,
    exemptions: Record<string, string>,
    scenarioNames: readonly string[],
    modulePopulation: readonly string[],
): Finding[] {
    const findings: Finding[] = [];
    const registered = new Set(scenarioNames);
    const coveredRes: RegExp[] = [];

    for (const [name, gate] of Object.entries(table)) {
        if (!registered.has(name)) {
            findings.push({ kind: "unregistered-table-key", detail: name });
        }
        for (const glob of gate.covers ?? []) {
            const re = globToRegExp(glob);
            coveredRes.push(re);
            if (!modulePopulation.some((m) => re.test(m))) {
                findings.push({ kind: "glob-matches-nothing", detail: `${name}: ${glob}` });
            }
        }
    }

    for (const [glob, reason] of Object.entries(exemptions)) {
        if (reason.trim() === "") {
            findings.push({ kind: "missing-exemption-reason", detail: glob });
        }
        const exemptRe = globToRegExp(glob);
        const shadowed = modulePopulation.filter(
            (m) => exemptRe.test(m) && coveredRes.some((re) => re.test(m)),
        );
        for (const m of shadowed) {
            findings.push({ kind: "exempt-shadows-covered", detail: `${glob} (${m})` });
        }
    }

    return findings;
}

/** the two completeness directions: every registered scenario has a table entry, and covered ∪ exempt
 *  equals the whole module population. Implemented and fixture-tested from 3a (per the spec's Live log
 *  resolution), now asserted against the real table via `COMPLETENESS_ENFORCED` — the gate table covers
 *  every registered scenario and every GPU-side module. */
export function checkCompleteness(
    table: Record<string, ScenarioGate>,
    exemptions: Record<string, string>,
    scenarioNames: readonly string[],
    modulePopulation: readonly string[],
): Finding[] {
    const findings: Finding[] = [];

    for (const name of scenarioNames) {
        if (!(name in table)) {
            findings.push({ kind: "scenario-missing-entry", detail: name });
        }
    }

    const coveredRes = Object.values(table).flatMap((g) => (g.covers ?? []).map(globToRegExp));
    const exemptRes = Object.keys(exemptions).map(globToRegExp);
    for (const m of modulePopulation) {
        const covered = coveredRes.some((re) => re.test(m));
        const exempt = exemptRes.some((re) => re.test(m));
        if (!covered && !exempt) {
            findings.push({ kind: "module-not-covered", detail: m });
        }
    }

    return findings;
}

/** walks the real filesystem under `root` (the shallot repo root) and returns every path (relative to
 *  `root`, forward-slashed) matching {@link GPU_MODULE_GLOBS}, excluding non-source suffixes
 *  (`.test.ts` / `.fixture.ts` / `.lab.ts`) the way `testing.md`'s tier convention already draws that
 *  line. */
export async function gpuModulePopulation(root: string): Promise<string[]> {
    const globs = GPU_MODULE_GLOBS.map(globToRegExp);
    const out: string[] = [];
    for await (const path of new Bun.Glob("packages/shallot/src/**/*.ts").scan({ cwd: root })) {
        if (/\.(test|fixture|lab)\.ts$/.test(path)) continue;
        if (globs.some((re) => re.test(path))) out.push(path);
    }
    return out;
}

/** every immediate subdirectory of `src/extras/` is either GPU-facing (it contributed at least one module to
 *  the population) or declared non-GPU in {@link NON_GPU_EXTRAS} with a reason — exactly one of the two, so
 *  both a new unclassified module and a stale declaration red. Pure over the population plus the two
 *  directory lists, so a fixture proves both directions. */
export function checkExtrasClassification(
    dirs: readonly string[],
    modulePopulation: readonly string[],
    nonGpu: Record<string, string>,
): Finding[] {
    const findings: Finding[] = [];
    for (const dir of dirs) {
        const prefix = `packages/shallot/src/extras/${dir}/`;
        const gpu = modulePopulation.some((m) => m.startsWith(prefix));
        const declared = (nonGpu[dir] ?? "").trim() !== "";
        if (gpu === declared) findings.push({ kind: "extras-unclassified", detail: dir });
    }
    return findings;
}

/** the second real-filesystem seam: the immediate subdirectory names under `src/extras/`. */
export async function extrasDirs(root: string): Promise<string[]> {
    const dirs = new Set<string>();
    for await (const path of new Bun.Glob("packages/shallot/src/extras/*/**").scan({ cwd: root })) {
        const dir = path.split("/")[4];
        if (dir) dirs.add(dir);
    }
    return [...dirs].sort();
}

export { GATE_EXEMPTIONS, SCENARIO_GATES };
