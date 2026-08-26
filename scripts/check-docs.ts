import { Glob } from "bun";
import { resolve } from "path";
import { template } from "../packages/create-shallot/index";
import { TEST_TIER_SUFFIX_NAMES } from "../packages/shallot/tests/test-tiers";

// Command docs standardize on `bunx shallot <cmd>`: bare `shallot` only resolves when the CLI is
// globally linked, while `bunx` resolves the local install everywhere — repo and consumer project
// alike. This guards against a bare `shallot <cmd>` command line creeping back into a fenced code
// block or a chained shell command. Prose that *names* the CLI surface ("the `shallot dev` server")
// is unaffected — it's never anchored at a line/chain start.

const root = resolve(import.meta.dir, "..");

// The doc set is what git tracks, not what the filesystem holds. A `**/*.md` scan reads whatever a
// particular checkout happens to have on disk: `examples/gym/dist/` after any build (448 files),
// and the glTF sample corpus through the `gym/public/gltf-samples` symlink wherever that corpus is
// checked out (451 more) — third-party and generated files we neither own nor should gate on, and present or
// absent depending on what the last command did. Asking git makes the scope identical in every
// checkout, which is the property that matters here: a check whose coverage depends on local state
// is how a stale tree reads green, and this release already paid for that lesson once.
//
// This set is the shared roster for every arm below — bare-command, pin, and citation — derived
// once from `git ls-files` rather than hand-listed per arm. The hand list this replaced omitted
// tracked docs (MIGRATION.md, CHANGELOG.md, evals NOTES) that sibling arms already scanned, so a
// bare `shallot <cmd>` creeping into one of those would have read green silently.
const tracked = Bun.spawnSync(["git", "ls-files", "-z", "*.md"], { cwd: root });
if (!tracked.success) {
    console.error(
        "✗ `git ls-files` failed — check-docs needs a git checkout to scope its doc set.",
    );
    process.exit(1);
}
const docs = tracked.stdout.toString().split("\0").filter(Boolean);
if (docs.length === 0) {
    console.error(
        "✗ `git ls-files '*.md'` matched nothing — the doc scan would be vacuously green.",
    );
    process.exit(1);
}

// The bare-command scan's exclusion list — each entry states its reason. The scan matches only
// inside fenced code blocks at a line/chain start, so a doc with no fenced commands simply
// produces no violations. No tracked .md is excluded: every tracked doc is a potential command-doc
// site, and one that carries no fenced block is harmless to scan. Mutation proof: adding a tracked
// .md with a fenced `shallot dev` line reds this arm (witnessed 2026-08-25, exit 1 — the
// `git ls-files` derivation catches a new doc the hand list would have missed).
const BARE_COMMAND_EXCLUSIONS: string[] = [];

const SUBCOMMAND = "(dev|build|run|verify|recipe)";
// A bare command-line-shaped `shallot <cmd>`: anchored at the start of a fenced code line, or
// right after a `&&` chain — never mid-prose, never preceded by `bunx `.
const BARE_COMMAND_RE = new RegExp(`(^|&&)\\s*shallot\\s+${SUBCOMMAND}\\b`);

type Violation = { file: string; line: number; text: string };

async function scan(file: string): Promise<Violation[]> {
    const violations: Violation[] = [];
    const lines = (await Bun.file(resolve(root, file)).text()).split("\n");
    let inFence = false;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.trim().startsWith("```")) {
            inFence = !inFence;
            continue;
        }
        if (!inFence) continue;
        if (BARE_COMMAND_RE.test(line.trim())) {
            violations.push({ file, line: i + 1, text: line.trim() });
        }
    }
    return violations;
}

const scanTargets = docs.filter((f) => !BARE_COMMAND_EXCLUSIONS.includes(f));
const violations = (await Promise.all(scanTargets.map(scan))).flat();

if (violations.length > 0) {
    console.error(`✗ ${violations.length} bare \`shallot <cmd>\` command line(s):\n`);
    for (const v of violations) {
        console.error(`  ${v.file}:${v.line}`);
        console.error(`    ${v.text}`);
    }
    console.error(
        "\nRunnable command lines standardize on `bunx shallot <cmd>` — bare `shallot` only " +
            "resolves when globally linked. Prose that names the CLI surface (not a line the " +
            "reader runs) is unaffected; only fenced/chained command lines trip this check.",
    );
    process.exit(1);
}

// The docs are a pin site too. The manifest-pin arm below enumerates every git-tracked
// `package.json` declaring a tracked package, and a bump that hits all of them still leaves the
// install block a reader actually runs pinned to the old minor — which is what shipped: 0.9.2's
// tree carried a `~0.12.0` peer while README.md and MIGRATION.md both still said
// `typegpu@~0.11.9`, a documented install that resolves to a peer conflict or a duplicate
// TypeGPU identity that dies at pipeline warm.
//
// Scope is a fenced `bun add` line: a command the reader runs, never prose. That distinction is
// load-bearing — CHANGELOG.md's 0.9.0 entry names `typegpu@~0.11.9` as a historical fact about what
// that release shipped with, and a blanket version sweep would be wrong to move it (same law as
// MIGRATION.md's dated prose, `check-versions.ts`).
const PIN_SOURCES: Record<string, { manifest: string; field: string }> = {
    typegpu: { manifest: "packages/shallot/package.json", field: "peerDependencies" },
    "unplugin-typegpu": { manifest: "packages/shallot/package.json", field: "dependencies" },
    "eslint-plugin-typegpu": { manifest: "package.json", field: "devDependencies" },
    typescript: { manifest: "package.json", field: "devDependencies" },
};

const declared: Record<string, string> = {};
for (const [name, { manifest, field }] of Object.entries(PIN_SOURCES)) {
    const json = await Bun.file(resolve(root, manifest)).json();
    const range = json[field]?.[name];
    if (typeof range !== "string") {
        console.error(
            `✗ ${manifest} declares no ${field}.${name} — update check-docs.ts's PIN_SOURCES.`,
        );
        process.exit(1);
    }
    declared[name] = range;
}

// `name@range`, where name is one of the tracked packages. A bare `unplugin-typegpu` with no `@` is
// unpinned prose-in-a-command and has nothing to disagree with, so it doesn't match. The
// `(?<![-\w])` lookbehind prevents a pin name from matching as a substring of a longer package
// name — `typescript` inside `@babel/plugin-syntax-typescript@^7.28.5` is not the `typescript` pin.
const PIN_RE = new RegExp(`(?<![-\\w])(${Object.keys(declared).join("|")})@(\\S+)`, "g");

type PinDrift = { file: string; line: number; name: string; found: string; want: string };
const drift: PinDrift[] = [];
let scanned = 0;

for (const match of docs) {
    scanned++;
    const lines = (await Bun.file(resolve(root, match)).text()).split("\n");
    let inFence = false;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.trim().startsWith("```")) {
            inFence = !inFence;
            continue;
        }
        if (!inFence || !line.trim().startsWith("bun add")) continue;
        for (const [, name, found] of line.matchAll(PIN_RE)) {
            if (found !== declared[name]) {
                drift.push({ file: match, line: i + 1, name, found, want: declared[name] });
            }
        }
    }
}

// The scaffold is a pin site too, and the source text isn't the artifact a `bun create shallot`
// user gets — `template()` is. Reading its *emitted* `package.json` (not grepping the source
// literal) pins the same object an installer actually resolves against. Mutation proof: bumping
// the `typescript` literal in `packages/create-shallot/index.ts` from `^7.0.2` to `^7.0.3` reds
// this arm (witnessed 2026-08-25, exit 1 — `typescript@^7.0.3 — the manifest declares ^7.0.2`).
const scaffoldPkg = JSON.parse(template("check-docs-probe")["package.json"]) as {
    dependencies?: Record<string, string>;
    devDependencies?: Record<string, string>;
};
const SCAFFOLD_PINS: { name: string; field: "dependencies" | "devDependencies" }[] = [
    { name: "typegpu", field: "dependencies" },
    { name: "unplugin-typegpu", field: "devDependencies" },
    { name: "typescript", field: "devDependencies" },
];
for (const { name, field } of SCAFFOLD_PINS) {
    const found = scaffoldPkg[field]?.[name];
    if (found !== declared[name]) {
        drift.push({
            file: "packages/create-shallot/index.ts (template() package.json)",
            line: 0,
            name,
            found: found ?? "(missing)",
            want: declared[name],
        });
    }
}

// The install-test fixtures are a third pin site: the install gate green-lights whatever version
// they carry, so a fixture stuck on the old minor certifies the drifted install as working. No
// hand-list of line numbers — scan every line naming `typegpu` with a version token attached
// (real code, not the prose comments that also mention the version for context) and classify each
// one, so a fixture bump landing outside today's known sites can't go unnoticed. `typegpu2` is a
// deliberate second physical copy (`identityFlow`'s duplicate-identity proof) whose version must
// still track the engine peer; `PM_RED_COPY_VERSION` is the one deliberate exclusion — that
// fixture needs a differing version on purpose. Mutation proof: bumping the first `typegpu` pin
// in `scripts/install-test.ts` from `~0.12.0` to `~0.12.1` reds this arm (witnessed 2026-08-25,
// exit 1 — `scripts/install-test.ts:365: typegpu@~0.12.1 — the manifest declares ~0.12.0`).
const FIXTURE_FILE = "scripts/install-test.ts";
const FIXTURE_EXCLUSION = "PM_RED_COPY_VERSION";
// A version token: an optional range prefix (`~`, `^`, or bare) followed by semver, or the
// `${PM_RED_COPY_VERSION}` interpolation — every pin *shape* the fixtures could carry, not just
// today's tilde-only forms (the regex must not be one surface form short: a `^0.12.0` or bare
// `0.12.0` pin must still be seen, not silently skipped).
const FIXTURE_VERSION_TOKEN = String.raw`(?:\$\{${FIXTURE_EXCLUSION}\}|[~^]?\d+\.\d+\.\d+)`;
const FIXTURE_ANY_VERSION_RE = new RegExp(FIXTURE_VERSION_TOKEN);
// The classifier: a recognized key (longest-first, so `typegpu2` doesn't get eaten by the bare
// `typegpu` alternative) followed by `: value`, where value is a plain string, or a template
// literal carrying the `npm:typegpu@` alias form.
const FIXTURE_PIN_RE = new RegExp(
    `(unplugin-typegpu|typegpu2|typegpu)"?\\s*:\\s*\`?"?(?:npm:typegpu@)?(${FIXTURE_VERSION_TOKEN})`,
    "g",
);

const fixtureText = await Bun.file(resolve(root, FIXTURE_FILE)).text();
const fixtureLines = fixtureText.split("\n");
const fixtureUnclassified: { line: number; text: string }[] = [];
let fixtureMatched = 0;

for (let i = 0; i < fixtureLines.length; i++) {
    const line = fixtureLines[i];
    const trimmed = line.trim();
    // Prose, not a fixture pin — a `//` line comment or a `/** … */` block-comment continuation
    // (the leading `*`). Both mention `typegpu`+version for context, never as a real pin.
    if (trimmed.startsWith("//") || trimmed.startsWith("*")) continue;
    if (!/typegpu/i.test(line)) continue;

    const pins = [...line.matchAll(FIXTURE_PIN_RE)];
    if (pins.length === 0) {
        if (!FIXTURE_ANY_VERSION_RE.test(line)) continue; // a `typegpu` mention, no version token
        fixtureUnclassified.push({ line: i + 1, text: trimmed });
        continue;
    }
    fixtureMatched++;
    for (const [, name, found] of pins) {
        if (found === `\${${FIXTURE_EXCLUSION}}`) continue; // the stated exclusion, differs on purpose
        // `typegpu2` is the alias copy — it must track the real `typegpu` peer's declared version.
        const want = name === "typegpu2" ? declared.typegpu : declared[name];
        if (found !== want) {
            drift.push({ file: FIXTURE_FILE, line: i + 1, name, found, want });
        }
    }
}

if (fixtureMatched === 0) {
    console.error(
        `✗ ${FIXTURE_FILE} scan matched no \`typegpu\` + version line — the fixture-pin arm ` +
            "would be vacuously green.",
    );
    process.exit(1);
}

if (fixtureUnclassified.length > 0) {
    console.error(
        `✗ ${fixtureUnclassified.length} line(s) in ${FIXTURE_FILE} carry \`typegpu\` and a ` +
            `version token in a form the fixture-pin arm can't classify:\n`,
    );
    for (const u of fixtureUnclassified) {
        console.error(`  ${FIXTURE_FILE}:${u.line}`);
        console.error(`    ${u.text}`);
    }
    console.error(
        "\nEvery `typegpu` + version line is either a checked pin (typegpu / typegpu2 / " +
            `unplugin-typegpu) or the stated \`${FIXTURE_EXCLUSION}\` exclusion — teach check-docs.ts ` +
            "the new form, or fix the line.",
    );
    process.exit(1);
}

// The manifests are a pin site too, and the roster is what git tracks, not what a hand list
// names — the same law as the doc set above. A `package.json` that declares
// `typegpu`/`unplugin-typegpu`/`eslint-plugin-typegpu` at a range the canonical manifest
// doesn't pin is the same drift the doc and fixture arms catch: an example project carrying
// the old minor nests its own copy and the two copies' branded internals disagree. This arm
// enumerates every git-tracked `package.json` and reds when a declared range disagrees with
// the manifest-declared pin. The canonical sources in `PIN_SOURCES` are included and
// trivially match themselves; the arm's value is the long tail of example/showcase/flows
// manifests a hand list would miss.
const DEP_FIELDS = [
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
] as const;

const manifestTracked = Bun.spawnSync(
    ["git", "ls-files", "-z", "package.json", "**/package.json"],
    { cwd: root },
);
if (!manifestTracked.success) {
    console.error(
        "✗ `git ls-files` failed — check-docs needs a git checkout to scope its manifest set.",
    );
    process.exit(1);
}
const manifestFiles = manifestTracked.stdout.toString().split("\0").filter(Boolean);
if (manifestFiles.length === 0) {
    console.error(
        "✗ `git ls-files 'package.json' '**/package.json'` matched nothing — the manifest-pin arm would be vacuously green.",
    );
    process.exit(1);
}

let manifestPkgCount = 0;
for (const file of manifestFiles) {
    const json = await Bun.file(resolve(root, file)).json();
    let declaredInThis = false;
    for (const field of DEP_FIELDS) {
        const deps = json[field];
        if (!deps || typeof deps !== "object") continue;
        for (const [name, range] of Object.entries(deps)) {
            if (!(name in declared)) continue;
            declaredInThis = true;
            if (typeof range !== "string") continue;
            if (range !== declared[name]) {
                drift.push({ file, line: 0, name, found: range, want: declared[name] });
            }
        }
    }
    if (declaredInThis) manifestPkgCount++;
}

if (manifestPkgCount === 0) {
    console.error(
        "✗ no tracked `package.json` declares a tracked package — the manifest-pin arm would be vacuously green.",
    );
    process.exit(1);
}

if (drift.length > 0) {
    console.error(`✗ ${drift.length} pin(s) disagree with the manifests:\n`);
    for (const d of drift) {
        // A drift row names the site it was read from and nothing else — three of the four arms
        // have no line to cite (a scaffold-emitted manifest, a workspace manifest), and labelling
        // those `(emitted)` reported a provenance the row didn't have.
        console.error(`  ${d.line ? `${d.file}:${d.line}` : d.file}`);
        console.error(`    ${d.name}@${d.found} — the manifest declares ${d.want}`);
    }
    console.error(
        "\nA documented install, a scaffold-emitted manifest, an install-test fixture, or a " +
            "workspace manifest must resolve against the shipped manifests. Bump the pin with the " +
            "manifest, in the same commit.",
    );
    process.exit(1);
}

// The entry-doc chain a reader (or an agent's context loader) actually walks is root-to-leaf, not
// a single file: `AGENTS.md` plus whichever leaf directory's own `AGENTS.md` it's working under.
// `style.md`'s budget was a remembered manual `wc -c` — enforced here per chain, since a bump that
// keeps every individual file under budget can still blow the chain a reader loads (measured
// 2026-08-16: the packages/shallot chain sat 3 B under 32768).
const ENTRY_DOC_BUDGET = 32768;
const ENTRY_DOC_CHAINS: string[][] = [
    ["AGENTS.md", "packages/shallot/AGENTS.md"],
    ["AGENTS.md", "examples/AGENTS.md"],
];

const chainOverages: { chain: string[]; bytes: number }[] = [];
for (const chain of ENTRY_DOC_CHAINS) {
    let bytes = 0;
    for (const file of chain) {
        bytes += Bun.file(resolve(root, file)).size;
    }
    if (bytes > ENTRY_DOC_BUDGET) {
        chainOverages.push({ chain, bytes });
    }
}

if (chainOverages.length > 0) {
    console.error(
        `✗ ${chainOverages.length} entry-doc chain(s) over the ${ENTRY_DOC_BUDGET} B budget:\n`,
    );
    for (const o of chainOverages) {
        console.error(`  ${o.chain.join(" + ")}: ${o.bytes} B`);
    }
    console.error(
        "\nAn agent's context loader reads root-to-leaf; past the budget the deepest file silently " +
            "drops and its whole contract vanishes. Fold detail into a path-scoped rule instead.",
    );
    process.exit(1);
}

// ── Arm (a): cross-citation resolution (case-insensitive) ──────────────────────────────────────
//
// A cross-citation is `<rule>.md "phrase"` — a rule filename followed by a double-quoted phrase —
// naming a passage in another rule file, optionally continued with ` / "phrase"` for additional
// phrases from the same rule (e.g. `render.md "Point-light shadows" / "Sun shadows"`). Each phrase
// must resolve in the named file, compared CASE-INSENSITIVELY: a case-sensitive first pass
// false-positived on avbd.md:44 citing gpu.md "reuse over add", which resolves against gpu.md:40's
// "**Reuse over add.**". The case rule is a recorded finding, not an implementation detail. The arm
// is green on this tree: the one true positive it once flagged — testing.md citing physics.md
// "the oracle is not the suspect", a phrase that lives only at avbd.md:13 — was corrected in S3.
// A continuation phrase (` / "phrase"`) belongs
// to the same rule citation, so each must be checked — not just the first: if the second phrase
// vanished from the named rule the arm would stay green if only the first were checked.

const RULE_NAMES: string[] = [];
for await (const match of new Glob("*.md").scan({ cwd: resolve(root, ".claude/rules") })) {
    RULE_NAMES.push(match.replace(/\.md$/, ""));
}

// cache rule file contents (lowercased for case-insensitive search)
const ruleFileCache = new Map<string, string>();
async function ruleFileText(name: string): Promise<string> {
    const path = `.claude/rules/${name}.md`;
    if (!ruleFileCache.has(path)) {
        ruleFileCache.set(path, (await Bun.file(resolve(root, path)).text()).toLowerCase());
    }
    return ruleFileCache.get(path)!;
}

// Match the full citation span: `rule.md "phrase"` plus any trailing ` / "phrase"` continuations
// that belong to the same rule citation. The continuation group is non-capturing (a repeated
// capture group would only keep the last match), so all phrases are extracted from the full match
// text via PHRASE_RE.
const CITATION_RE = new RegExp(
    `\\b(${RULE_NAMES.map((n) => n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\.md "([^"]+)"(?:\\s*/\\s*"([^"]+)")*`,
    "g",
);
const PHRASE_RE = /"([^"]+)"/g;

type CitationViolation = { file: string; line: number; rule: string; phrase: string };
const citationViolations: CitationViolation[] = [];
let citationCount = 0;

for (const match of docs) {
    const lines = (await Bun.file(resolve(root, match)).text()).split("\n");
    for (let i = 0; i < lines.length; i++) {
        for (const m of lines[i].matchAll(CITATION_RE)) {
            const rule = m[1];
            const text = await ruleFileText(rule);
            // extract every quoted phrase from the full citation span (head + continuations)
            for (const [, phrase] of m[0].matchAll(PHRASE_RE)) {
                citationCount++;
                if (!text.includes(phrase.toLowerCase())) {
                    citationViolations.push({
                        file: match,
                        line: i + 1,
                        rule,
                        phrase,
                    });
                }
            }
        }
    }
}

if (citationCount === 0) {
    console.error(
        '✗ cross-citation arm matched no `<rule>.md "phrase"` citation — the arm would be vacuously green.',
    );
    process.exit(1);
}

if (citationViolations.length > 0) {
    console.error(
        `✗ ${citationViolations.length} cross-citation(s) that don't resolve in the named file (case-insensitive):\n`,
    );
    for (const v of citationViolations) {
        console.error(`  ${v.file}:${v.line}: ${v.rule}.md "${v.phrase}"`);
    }
    console.error(
        '\nA `<rule>.md "phrase"` cross-citation must resolve as a case-insensitive substring in the ' +
            "named rule file. A case-sensitive comparison false-positives on phrases whose casing differs " +
            '(e.g. gpu.md "reuse over add" resolves against "**Reuse over add.**"), so the comparison ' +
            "is case-insensitive by design.",
    );
    process.exit(1);
}

// ── Arm (b): showcase index completeness (both directions) ──────────────────────────────────────
//
// Every `examples/showcase/*/` dir with at least one tracked file must have an index line in
// `examples/AGENTS.md`, and every index line must name a dir with tracked content. Both directions
// are reported — a stale index line is as much a defect as a missing one.
//
// The population is derived from the TRACKED set (git ls-files), not the filesystem: a dir whose
// tracked content was deleted by a sibling lane but whose untracked dist/node_modules/test-results/
// remain would be a false positive under a filesystem walk — the arm would flag a dir without an
// index line that has no tracked content to index. Asking git makes the scope identical in every
// checkout, which is the property that matters (same law as the doc scan above).

const SHOWCASE_DIR = "examples/showcase";
const showcaseTracked = Bun.spawnSync(["git", "ls-files", "-z", SHOWCASE_DIR], {
    cwd: root,
});
if (!showcaseTracked.success) {
    console.error(
        "✗ `git ls-files` failed — the showcase arm needs a git checkout to scope its dir set.",
    );
    process.exit(1);
}
const showcaseDirs = new Set<string>();
for (const path of showcaseTracked.stdout.toString().split("\0").filter(Boolean)) {
    const rel = path.slice(SHOWCASE_DIR.length + 1);
    const parts = rel.split("/");
    if (parts.length < 2) continue; // directly under showcase/, not a subdir (e.g. .gitkeep)
    showcaseDirs.add(parts[0]);
}
if (showcaseDirs.size === 0) {
    console.error(
        "✗ `git ls-files examples/showcase/` matched no subdir — the showcase arm would be vacuously green.",
    );
    process.exit(1);
}

const agentsMd = await Bun.file(resolve(root, "examples/AGENTS.md")).text();
const showcaseIndexRe = /`showcase\/([^`/]+)\//g;
const indexedShowcases = new Set<string>();
for (const [, name] of agentsMd.matchAll(showcaseIndexRe)) {
    indexedShowcases.add(name);
}

const missingIndex = [...showcaseDirs].filter((d) => !indexedShowcases.has(d)).sort();
const staleIndex = [...indexedShowcases].filter((d) => !showcaseDirs.has(d)).sort();

if (missingIndex.length > 0 || staleIndex.length > 0) {
    const parts: string[] = [];
    if (missingIndex.length > 0) {
        parts.push(
            `showcase dir(s) without an index line in examples/AGENTS.md: ${missingIndex.join(", ")}`,
        );
    }
    if (staleIndex.length > 0) {
        parts.push(
            `index line(s) in examples/AGENTS.md naming no showcase dir: ${staleIndex.join(", ")}`,
        );
    }
    console.error(
        `✗ showcase index mismatch (${parts.length} direction(s)):\n` +
            parts.map((p) => `  ${p}`).join("\n"),
    );
    console.error(
        "\nEvery `examples/showcase/*/` dir must have an index line in `examples/AGENTS.md`, and " +
            "every index line must name a real dir. Both directions are checked.",
    );
    process.exit(1);
}

// ── Arm (c): evals task-index completeness (both directions) ───────────────────────────────────
//
// Every `evals/tasks/<task>/` dir with at least one tracked file must have a row in the task
// table in `evals/README.md`, and every table row must name a dir with tracked content. Both
// directions are reported — a stale table row is as much a defect as a missing one. Mirrors the
// showcase index arm (Arm b) in shape: same `git ls-files` derivation, same both-directions
// report, same empty-population guard.
//
// The population is derived from the TRACKED set (git ls-files), not the filesystem — same law as
// the showcase arm and the doc scan above. Mutation proof: adding a tracked `evals/tasks/<new>/`
// dir with no table row reds this arm (witnessed 2026-08-25, exit 1); adding a table row naming
// no dir also reds. Both directions witnessed.

const EVALS_TASKS_DIR = "evals/tasks";
const evalsTasksTracked = Bun.spawnSync(["git", "ls-files", "-z", EVALS_TASKS_DIR], {
    cwd: root,
});
if (!evalsTasksTracked.success) {
    console.error(
        "✗ `git ls-files` failed — the evals task-index arm needs a git checkout to scope its dir set.",
    );
    process.exit(1);
}
const evalsTaskDirs = new Set<string>();
for (const path of evalsTasksTracked.stdout.toString().split("\0").filter(Boolean)) {
    const rel = path.slice(EVALS_TASKS_DIR.length + 1);
    const parts = rel.split("/");
    if (parts.length < 2) continue; // directly under tasks/, not a subdir
    evalsTaskDirs.add(parts[0]);
}
if (evalsTaskDirs.size === 0) {
    console.error(
        "✗ `git ls-files evals/tasks/` matched no subdir — the evals task-index arm would be vacuously green.",
    );
    process.exit(1);
}

const evalsReadme = await Bun.file(resolve(root, "evals/README.md")).text();
// A table row's first column is a backtick-quoted task name: `| \`task-name\` | ...`
const evalsTaskRowRe = /^\| `([^`]+)` \|/gm;
const indexedEvalsTasks = new Set<string>();
for (const [, name] of evalsReadme.matchAll(evalsTaskRowRe)) {
    indexedEvalsTasks.add(name);
}

const missingTaskIndex = [...evalsTaskDirs].filter((d) => !indexedEvalsTasks.has(d)).sort();
const staleTaskIndex = [...indexedEvalsTasks].filter((d) => !evalsTaskDirs.has(d)).sort();

if (missingTaskIndex.length > 0 || staleTaskIndex.length > 0) {
    const parts: string[] = [];
    if (missingTaskIndex.length > 0) {
        parts.push(
            `evals task dir(s) without a table row in evals/README.md: ${missingTaskIndex.join(", ")}`,
        );
    }
    if (staleTaskIndex.length > 0) {
        parts.push(
            `table row(s) in evals/README.md naming no evals task dir: ${staleTaskIndex.join(", ")}`,
        );
    }
    console.error(
        `✗ evals task-index mismatch (${parts.length} direction(s)):\n` +
            parts.map((p) => `  ${p}`).join("\n"),
    );
    console.error(
        "\nEvery `evals/tasks/<task>/` dir must have a row in the task table in `evals/README.md`, and " +
            "every table row must name a real task dir. Both directions are checked.",
    );
    process.exit(1);
}

// ── Arm (d): tier-suffix roster — one constant, derived consumers, asserted against testing.md ──────
//
// The test-tier suffix roster is ONE exported constant (`packages/shallot/tests/test-tiers.ts`).
// This arm asserts (1) the roster matches `testing.md`'s tier-section bullet ledes — the
// enumeration `testing.md` itself makes — (2) the section heading agrees with its own body, and
// (3) no file in the repo carries a literal tier-suffix roster of its own — a line enumerating 3+
// of the roster's suffix names either as bare words with regex alternation (`|`) or as an array literal of
// quoted `.suffix.ts` strings. The consumer set is DERIVED, not enumerated: the arm scans every
// tracked file itself, so a new file restating the roster — in either shape — is caught without
// updating a hand-list. A fix that leaves two hand-written lists in agreement fails this criterion.

const testingMd = await Bun.file(resolve(root, ".claude/rules/testing.md")).text();
const testingLines = testingMd.split("\n");

// find the tier section heading (## `.test.ts` vs ...)
let tierHeadingIdx = -1;
for (let i = 0; i < testingLines.length; i++) {
    if (/^## `\.test\.ts` vs /.test(testingLines[i])) {
        tierHeadingIdx = i;
        break;
    }
}
if (tierHeadingIdx === -1) {
    console.error(
        "✗ tier-suffix roster arm: could not find the `## `.test.ts` vs ...` heading in testing.md.",
    );
    process.exit(1);
}

// collect the section's lines until the next `## ` heading
const tierSectionLines: string[] = [];
for (let i = tierHeadingIdx + 1; i < testingLines.length; i++) {
    if (/^## /.test(testingLines[i])) break;
    tierSectionLines.push(testingLines[i]);
}

// extract suffix names from bullet ledes: `- **`.suffix.ts`**`
const bulletSuffixRe = /^- \*\*`\.(\w+)\.ts`\*\*/;
const bulletLedeSuffixes: string[] = [];
for (const line of tierSectionLines) {
    const m = bulletSuffixRe.exec(line);
    if (m) bulletLedeSuffixes.push(m[1]);
}

// extract suffix names from the heading: `` `.suffix.ts` ``
const headingSuffixRe = /`\.(\w+)\.ts`/g;
const headingSuffixes: string[] = [];
for (const m of testingLines[tierHeadingIdx].matchAll(headingSuffixRe)) {
    headingSuffixes.push(m[1]);
}

const rosterSuffixes = [...TEST_TIER_SUFFIX_NAMES];

const rosterFindings: string[] = [];
if (bulletLedeSuffixes.length === 0) {
    rosterFindings.push(
        "testing.md's tier section has no `- **`.suffix.ts`**` bullet ledes — the arm would be vacuously green.",
    );
}
if (rosterSuffixes.join(",") !== bulletLedeSuffixes.join(",")) {
    rosterFindings.push(
        `the shared roster [${rosterSuffixes.join(", ")}] does not match testing.md's bullet ledes [${bulletLedeSuffixes.join(", ")}].`,
    );
}
if (headingSuffixes.join(",") !== bulletLedeSuffixes.join(",")) {
    rosterFindings.push(
        `testing.md's tier-section heading [${headingSuffixes.join(", ")}] does not agree with its own bullet ledes [${bulletLedeSuffixes.join(", ")}].`,
    );
}

// Derive the consumer set: scan every tracked file in the repo for a literal tier-suffix roster —
// a line enumerating 3+ of the roster's suffix names either as bare words with regex alternation (`|`) or
// as an array literal of quoted `.suffix.ts` strings (e.g. `[".oracle.ts", ".probes.ts", ...]`).
// Any file that carries such a roster is restating it rather than reading the shared constant; the
// arm finds such files itself, so a new file restating the roster — in either shape — is caught
// without updating a hand-list.
//
// Two exclusions, stated explicitly:
// 1. `packages/shallot/tests/test-tiers.ts` — the roster's own definition module; it MUST contain the
//    suffix names (it is where the constant lives).
// 2. `scripts/check-docs.ts` — this arm's own text; a self-referential gate matches its own
//    description of what it checks (per `.claude/rules/specs.md`'s self-reference principle).
const ROSTER_EXCLUSIONS = new Set([
    "packages/shallot/tests/test-tiers.ts",
    "scripts/check-docs.ts",
]);
const allTrackedFiles = Bun.spawnSync(["git", "ls-files", "-z"], { cwd: root });
if (!allTrackedFiles.success) {
    console.error(
        "✗ `git ls-files` failed — the roster arm needs a git checkout to scope its file set.",
    );
    process.exit(1);
}
const suffixWords = [...TEST_TIER_SUFFIX_NAMES];
// An array-literal restatement (`[".oracle.ts", ".probes.ts", ".tier.ts", ".lab.ts"]`) is the other
// shape a hand-written roster takes, alongside regex alternation — both are caught below.
const arrayLiteralRe = /\[\s*(?:["'`]\.\w+\.ts["'`]\s*,\s*){2,}["'`]\.\w+\.ts["'`]\s*\]/;
for (const file of allTrackedFiles.stdout.toString().split("\0").filter(Boolean)) {
    if (ROSTER_EXCLUSIONS.has(file)) continue;
    const source = await Bun.file(resolve(root, file)).text();
    for (const [i, line] of source.split("\n").entries()) {
        const hits = suffixWords.filter((n) => new RegExp(`\\b${n}\\b`).test(line)).length;
        const shape = line.includes("|")
            ? "regex alternation"
            : arrayLiteralRe.test(line)
              ? "an array literal"
              : null;
        if (hits >= 3 && shape !== null) {
            rosterFindings.push(
                `${file}:${i + 1} carries a literal tier-suffix roster (a line enumerating ${hits} of the roster's suffix names as ${shape}) — the roster must be derived from the shared test-tiers.ts constant, not restated.`,
            );
            break; // one finding per file is enough
        }
    }
}

if (rosterFindings.length > 0) {
    console.error(`✗ tier-suffix roster arm: ${rosterFindings.length} finding(s):\n`);
    for (const f of rosterFindings) {
        console.error(`  ${f}`);
    }
    console.error(
        "\nThe test-tier suffix roster must be one exported constant with two consumers, asserted " +
            "against testing.md's own enumeration. A fix that leaves two hand-written lists in " +
            "agreement fails this criterion.",
    );
    process.exit(1);
}

// ── Arm (e): citation resolution — formatting-invariant identifier population ──────────────
//
// Every token in `.claude/rules/**` outside a fenced code block matching an identifier *shape* —
// camelCase, PascalCase, snake/SCREAMING_SNAKE, lowercase-with-digits, or a backticked `*.ts`
// path — **backticked or bare** — must resolve against the tree or a committed roster. The
// population predicate is formatting-invariant: a token is caught whether it's in backticks or
// bare in prose, so removing backticks (round 3's escape) no longer removes a citation from the
// arm's population. Round 3 removed backticks from 10 tokens to escape the arm; round 4 reverts
// those markup edits and widens the predicate to catch bare tokens too.
//
// Resolution is a one-pass token index over `*.ts`/`*.rs`/`*.wgsl` (excluding `node_modules`,
// `scripts/check-docs.ts`, `scripts/detect-stale-claims.ts`), NOT `git grep --fixed-strings`:
// substring matching reads 8 sites green off longer tokens (e.g. `spotInner` matches
// `spotInnerF`, `hullSat` matches `hullSatWgsl`, `InFragmentStage` matches
// `maxStorageBuffersInFragmentStage`). The token index tokenizes source files into individual
// identifier words and does exact set-membership — `spotInner` only resolves if `spotInner`
// appears as a standalone token, not as a substring of `spotInnerF`.
//
// The allowlist is roster classes (WGSL-builtin, WebGPU-IDL, foreign-namespace vendored symbol
// lists, x86/ISA mnemonic — each a committed file under `scripts/`) plus a per-site residue.
// Each entry is asserted TWO WAYS: (1) the mention is really present in that file, (2) the
// symbol/path is genuinely absent from the tree AND all rosters. The attribution leg is gone —
// round 3's attribution token was a proxy that laundered exemptions passed and real exemptions
// failed (10 entries failed the attribution leg and were de-backtickked rather than adjudicated).
// The roster replaces attribution: a foreign-namespace symbol resolves against a committed
// roster, not against an attribution token on the citing line.
//
// Witnessed red (mutation proofs, each exit code captured to a file):
//   (i)  Seed: `deadIdentifierCitation` seeded into `.claude/rules/gpu.md` → exit 1
//         (captured to /tmp/asc-mutation-seed.txt)
//   (ii) De-backtick: backticks removed from a live citation → still red (bare token caught)
//         (captured to /tmp/asc-mutation-debacktick.txt)
//   (iii) Launder: a dead shallot symbol added to a foreign-namespace roster → exit 1
//         (the two-way assertion catches it: present in file, absent from tree, but resolves
//         against the roster — the exemption is over because the symbol IS in the roster)
//         (captured to /tmp/asc-mutation-launder.txt)
//   (iv) Substring: `git grep --fixed-strings` reads a substring match green; the token index
//         does not — seeding `spotInner` (substring of `spotInnerF`) reds with the token index
//         but greens with `git grep --fixed-strings`
//         (captured to /tmp/asc-mutation-substring.txt)

import { FOREIGN_NAMESPACES, WEBGPU_IDL, WGSL_BUILTINS, X86_ISA } from "./rosters";

// ── Population: scan .claude/rules/**/*.md for identifier-shaped tokens ────────────────────
//
// The population is the set of tracked .md files under .claude/rules/, derived from `git ls-files`
// (same law as the doc scan above — the scope is what git tracks, not what the filesystem holds).

const trackedFiles = allTrackedFiles.stdout.toString().split("\0").filter(Boolean);
const trackedSet = new Set(trackedFiles);

const rulesTracked = Bun.spawnSync(["git", "-C", root, "ls-files", "-z", "*.md"], { cwd: root });
if (!rulesTracked.success) {
    console.error(
        "✗ `git ls-files` failed — the citation-resolution arm needs a git checkout to scope its rule set.",
    );
    process.exit(1);
}
const ruleFiles = rulesTracked.stdout
    .toString()
    .split("\0")
    .filter(Boolean)
    .filter((f) => f.startsWith(".claude/rules/"));
if (ruleFiles.length === 0) {
    console.error(
        "✗ `git ls-files '*.md'` matched nothing under .claude/rules/ — the citation-resolution arm would be vacuously green.",
    );
    process.exit(1);
}

// ── One-pass token index over *.ts / *.rs / *.wgsl ────────────────────────────────────────
//
// Build a Set<string> of every identifier token in every tracked source file. Resolution is
// exact set-membership, not `git grep --fixed-strings` (substring matching). Excludes
// `node_modules`, `scripts/check-docs.ts`, and `scripts/detect-stale-claims.ts` (their comments
// mention the symbols they check, which would false-resolve dead citations).

const indexSourceFiles = trackedFiles.filter(
    (f) =>
        (f.endsWith(".ts") || f.endsWith(".rs") || f.endsWith(".wgsl")) &&
        !f.includes("node_modules") &&
        f !== "scripts/check-docs.ts" &&
        f !== "scripts/detect-stale-claims.ts" &&
        f !== "scripts/rosters.ts",
);

const tokenIndex = new Set<string>();
const INDEX_TOKEN_RE = /[A-Za-z_][A-Za-z0-9_]*/g;
for (const f of indexSourceFiles) {
    const text = await Bun.file(resolve(root, f)).text();
    for (const m of text.matchAll(INDEX_TOKEN_RE)) {
        tokenIndex.add(m[0]);
    }
}
if (tokenIndex.size === 0) {
    console.error(
        "✗ token index is empty — no tracked *.ts/*.rs/*.wgsl files found (excluding node_modules and scripts).",
    );
    process.exit(1);
}

// ── Combined roster set ────────────────────────────────────────────────────────────────────
//
// Merge all rosters into a single set for O(1) lookup. Each roster is asserted non-empty below.

const allRosters: { name: string; roster: ReadonlySet<string> }[] = [
    { name: "WGSL_BUILTINS", roster: WGSL_BUILTINS },
    { name: "WEBGPU_IDL", roster: WEBGPU_IDL },
    { name: "X86_ISA", roster: X86_ISA },
    ...Object.entries(FOREIGN_NAMESPACES).map(([name, roster]) => ({
        name: `FOREIGN_NAMESPACES.${name}`,
        roster,
    })),
];

// Assert each roster non-empty — a roster that loses its last entry would make the arm vacuously
// green for that class.
for (const { name, roster } of allRosters) {
    if (roster.size === 0) {
        console.error(
            `✗ roster ${name} is empty — a citation-resolution arm with an empty roster is vacuously green for that class.`,
        );
        process.exit(1);
    }
}

const combinedRoster = new Set<string>();
for (const { roster } of allRosters) {
    for (const sym of roster) combinedRoster.add(sym);
}

// ── Identifier shape predicates ────────────────────────────────────────────────────────────
//
// camelCase: starts lowercase, has at least one uppercase, no underscore.
// PascalCase: starts uppercase, has a [a-z][A-Z] transition (compound, not an acronym), no underscore.
// snake/SCREAMING_SNAKE: contains underscore, at least 2 chars, has an alphanumeric char.
// lowercase-with-digits: all lowercase + digits, at least one of each, no underscore.
// Hex-shaped tokens (all [0-9a-f], at least one digit and one letter) are excluded by predicate
// (the bare git sha at gpu.md:268). Named false positives (AoSoA, iGPUs, EndFrame) are excluded
// by an explicit set.

function isCamelCase(w: string): boolean {
    return /^[a-z]/.test(w) && /[A-Z]/.test(w) && !w.includes("_");
}
function isPascalCase(w: string): boolean {
    return /^[A-Z]/.test(w) && /[a-z][A-Z]/.test(w) && !w.includes("_");
}
function isSnakeOrScreaming(w: string): boolean {
    return (
        w.includes("_") &&
        /^[A-Za-z_][A-Za-z0-9_]*$/.test(w) &&
        /[A-Za-z0-9]/.test(w) &&
        w.length >= 2
    );
}
function isLowercaseWithDigits(w: string): boolean {
    return /^[a-z][a-z0-9]*$/.test(w) && /[0-9]/.test(w) && !w.includes("_");
}
function isHex(w: string): boolean {
    return /^[0-9a-f]+$/.test(w) && /[0-9]/.test(w) && /[a-f]/.test(w);
}

const SHAPE_FALSE_POSITIVES = new Set(["AoSoA", "iGPUs", "EndFrame"]);

function matchesShape(w: string): boolean {
    if (w.length < 2) return false;
    if (SHAPE_FALSE_POSITIVES.has(w)) return false;
    if (isHex(w)) return false;
    return isCamelCase(w) || isPascalCase(w) || isSnakeOrScreaming(w) || isLowercaseWithDigits(w);
}

// ── Candidate extraction ───────────────────────────────────────────────────────────────────
//
// Two kinds of candidates:
//   1. Backtick-cited `*.ts` paths — a string between backticks ending in `.ts` that is a real
//      file path (not a suffix pattern `.test.ts`, not a glob `*.test.ts`, not a command).
//   2. Identifier-shaped tokens — every word matching an identifier shape, **backticked or bare**.
//      The bare extraction strips URLs (tokens inside URLs are not citations) but does NOT strip
//      backtick spans: a token inside a complex backtick expression (e.g. `TimestampWrites`
//      inside `Compute.span?: ... | undefined`) is still a token in the prose and must be caught.
//      A backtick span that IS a single identifier is already caught by the backtick extraction;
//      the bare extraction deduplicates by {file, line, ref}.

type CitationCandidate = {
    file: string;
    line: number;
    ref: string;
    kind: "ts-path" | "identifier";
};

const TS_PATH_RE = /`([^`]*\.ts)`/g;
const IDENTIFIER_RE = /`([A-Za-z_][A-Za-z0-9_]*(?:\(\))?)`/g;
const BARE_TOKEN_RE = /[A-Za-z_][A-Za-z0-9_]*/g;

const citationCandidates: CitationCandidate[] = [];
const seenCandidates = new Set<string>();

function addCandidate(file: string, line: number, ref: string, kind: "ts-path" | "identifier") {
    const key = `${file}:${line}:${ref}`;
    if (seenCandidates.has(key)) return;
    seenCandidates.add(key);
    citationCandidates.push({ file, line, ref, kind });
}

for (const file of ruleFiles) {
    const fullPath = resolve(root, file);
    const text = await Bun.file(fullPath).text();
    const lines = text.split("\n");
    let inFence = false;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.trim().startsWith("```")) {
            inFence = !inFence;
            continue;
        }
        if (inFence) continue;

        // Backtick-cited .ts paths
        for (const m of line.matchAll(TS_PATH_RE)) {
            const ref = m[1];
            if (ref.startsWith(".") || ref.includes("*") || ref.includes(" ") || ref.includes("{"))
                continue;
            addCandidate(file, i + 1, ref, "ts-path");
        }

        // Backtick-cited identifiers
        for (const m of line.matchAll(IDENTIFIER_RE)) {
            const ref = m[1].replace(/\(\)$/, "");
            if (ref.endsWith(".ts")) continue;
            if (matchesShape(ref)) addCandidate(file, i + 1, ref, "identifier");
        }

        // Bare identifier-shaped tokens (strip URLs, not backtick spans)
        const stripped = line.replace(/https?:\/\/[^\s)]*/g, " ");
        for (const m of stripped.matchAll(BARE_TOKEN_RE)) {
            const ref = m[0];
            if (matchesShape(ref)) addCandidate(file, i + 1, ref, "identifier");
        }
    }
}

if (citationCandidates.length === 0) {
    console.error(
        "✗ citation-resolution arm matched no identifier-shaped token or *.ts path — the arm would be vacuously green.",
    );
    process.exit(1);
}

// ── Resolution ──────────────────────────────────────────────────────────────────────────────
//
// For .ts paths: try as-is, with `packages/shallot/src/`, with `packages/shallot/`, then suffix
// match against the tracked set.
// For identifiers: exact set-membership in the token index (NOT substring matching).
// For both: if unresolved against the tree, check against the combined roster.

function tsPathResolves(path: string): boolean {
    const tries = [path, `packages/shallot/src/${path}`, `packages/shallot/${path}`];
    for (const t of tries) {
        if (trackedSet.has(t)) return true;
    }
    const suffix = `/${path}`;
    for (const f of trackedSet) {
        if (f.endsWith(suffix)) return true;
    }
    return false;
}

function resolvesAnywhere(ref: string, kind: string): boolean {
    if (kind === "ts-path") {
        if (tsPathResolves(ref)) return true;
        return combinedRoster.has(ref);
    }
    if (tokenIndex.has(ref)) return true;
    return combinedRoster.has(ref);
}

// ── Per-site residue ──────────────────────────────────────────────────────────────────────
//
// The per-site residue is for tokens that don't resolve against the tree or any roster but are
// deliberately exempt: anti-pattern examples, retirement notices, tool/framework references, and
// prose terms that match an identifier shape but are not code citations (bench metrics, formula
// variables, GPU vendor names, tool names, shorthand references inside complex backtick
// expressions, etc.). Each entry is asserted TWO WAYS: (1) the mention is really present in
// that file, (2) the symbol/path is genuinely absent from the tree AND all rosters.

type ResidueEntry = { file: string; ref: string; reason: string };

const PER_SITE_RESIDUE: ResidueEntry[] = [
    // Anti-pattern examples (illustrative names, not real code)
    {
        file: ".claude/rules/style.md",
        ref: "createMeshGeometryFromVertices",
        reason: "anti-pattern example",
    },
    { file: ".claude/rules/style.md", ref: "prepareX", reason: "anti-pattern example" },
    { file: ".claude/rules/style.md", ref: "buildY", reason: "anti-pattern example" },
    { file: ".claude/rules/style.md", ref: "applyZ", reason: "anti-pattern example" },
    { file: ".claude/rules/ecs.md", ref: "lastState", reason: "anti-pattern example" },
    { file: ".claude/rules/ecs.md", ref: "resetIfNewState", reason: "anti-pattern example" },
    { file: ".claude/rules/ecs.md", ref: "lastCamera", reason: "anti-pattern example" },
    { file: ".claude/rules/exports.md", ref: "readBuffer", reason: "anti-pattern example" },
    // Retirement notices (deliberately names a gone symbol)
    { file: ".claude/rules/audio.md", ref: "AudioCommand", reason: "retirement notice" },
    { file: ".claude/rules/ecs.md", ref: "detectVecN", reason: "retirement notice" },
    { file: ".claude/rules/exports.md", ref: "FogLight", reason: "retirement notice" },
    { file: ".claude/rules/exports.md", ref: "BVH_TRAVERSE_WGSL", reason: "retirement notice" },
    { file: ".claude/rules/tumble.md", ref: "ColumnState", reason: "retirement notice" },
    { file: ".claude/rules/gpu.md", ref: "VertexOutput", reason: "retirement notice" },
    // Tool/framework references (TypeGPU, Bun, GitHub Actions internals)
    {
        file: ".claude/rules/testing.md",
        ref: "InFragmentStage",
        reason: "tool/framework reference (prose shorthand for maxStorageBuffersInFragmentStage)",
    },
    { file: ".claude/rules/testing.md", ref: "shaderModules", reason: "tool/framework reference" },
    {
        file: ".claude/rules/testing.md",
        ref: "__TYPEGPU_AUTONAME__",
        reason: "tool/framework reference",
    },
    { file: ".claude/rules/testing.md", ref: "DependencyLoop", reason: "tool/framework reference" },
    {
        file: ".claude/rules/testing.md",
        ref: "disabled_manually",
        reason: "tool/framework reference",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "list_typegpu_exports",
        reason: "tool/framework reference",
    },
    { file: ".claude/rules/exports.md", ref: "sideEffects", reason: "tool/framework reference" },
    // Prose terms: bench metrics, formula variables, GPU terms, tool names, shorthand refs
    {
        file: ".claude/rules/audio.md",
        ref: "gain_effect",
        reason: "prose term (audio parameter name), not a code citation",
    },
    {
        file: ".claude/rules/audio.md",
        ref: "direct_effect",
        reason: "prose term (audio parameter name), not a code citation",
    },
    {
        file: ".claude/rules/avbd.md",
        ref: "vAng",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/avbd.md",
        ref: "collidePass",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/avbd.md",
        ref: "ownerEid",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/avbd.md",
        ref: "l12",
        reason: "prose term (BVH level shorthand), not a code citation",
    },
    {
        file: ".claude/rules/avbd.md",
        ref: "l4",
        reason: "prose term (BVH level shorthand), not a code citation",
    },
    {
        file: ".claude/rules/avbd.md",
        ref: "COMPACT_PAIRS_PER_BODY",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/examples.md",
        ref: "_measure",
        reason: "prose term (part of a .ts path citation, not a standalone symbol)",
    },
    {
        file: ".claude/rules/exports.md",
        ref: "_WGSL",
        reason: "prose term (part of a *_WGSL glob pattern, not a standalone symbol)",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "_WGSL",
        reason: "prose term (part of a *_WGSL glob pattern, not a standalone symbol)",
    },
    {
        file: ".claude/rules/render.md",
        ref: "_WGSL",
        reason: "prose term (part of a *_WGSL glob pattern, not a standalone symbol)",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "csrOffset",
        reason: "prose term (storage layout field shorthand), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "inputCols",
        reason: "prose term (storage layout field shorthand), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "entityCount",
        reason: "prose term (variable name in prose), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "sortedIds",
        reason: "prose term (variable name in prose), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "leafAABBs",
        reason: "prose term (variable name in prose), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "entityIds",
        reason: "prose term (variable name in prose), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "bytes_moved",
        reason: "prose term (bench metric name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "peak_BW",
        reason: "prose term (bench metric name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "schema_stride",
        reason: "prose term (bench metric name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "snorm8",
        reason: "prose term (type concept name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "NdotL",
        reason: "prose term (lighting term), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "C_init_",
        reason: "prose term (formula variable), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "C_init_n",
        reason: "prose term (formula variable), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "tangentBasis",
        reason: "prose term (variable name in prose), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "f16x2",
        reason: "prose term (type shorthand), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "localId",
        reason: "prose term (compute variable name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "warp_size",
        reason: "prose term (GPU hardware term), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "theoretical_min",
        reason: "prose term (bench metric name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "working_set",
        reason: "prose term (bench metric name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "L2_size",
        reason: "prose term (GPU hardware term), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "PowerVR",
        reason: "prose term (GPU vendor name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "RenderDoc",
        reason: "prose term (tool name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "webgpu_inspector",
        reason: "prose term (tool name), not a code citation",
    },
    {
        file: ".claude/rules/gpu.md",
        ref: "wg32",
        reason: "prose term (workgroup size shorthand), not a code citation",
    },
    {
        file: ".claude/rules/physics.md",
        ref: "sweptPos",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/render.md",
        ref: "syncMeshes",
        reason: "prose term (variable name in prose), not a code citation",
    },
    {
        file: ".claude/rules/render.md",
        ref: "colorPipelines",
        reason: "prose term (variable name in prose), not a code citation",
    },
    {
        file: ".claude/rules/render.md",
        ref: "fsPrepass",
        reason: "prose term (variable name in prose), not a code citation",
    },
    {
        file: ".claude/rules/testing.md",
        ref: "_REQUIRED",
        reason: "prose term inside a backtick env-var expression, not a standalone citation",
    },
    {
        file: ".claude/rules/tumble.md",
        ref: "boneObjNow_",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/tumble.md",
        ref: "boneObjBind_",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/tumble.md",
        ref: "relNowPos",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/tumble.md",
        ref: "relNowQuat",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
    {
        file: ".claude/rules/tumble.md",
        ref: "memory64",
        reason: "prose term (WASM feature name), not a code citation",
    },
    {
        file: ".claude/rules/tumble.md",
        ref: "tlsPtr",
        reason: "prose term inside a backtick code expression, not a standalone citation",
    },
];

// Build a flat lookup map from the per-site residue
const residueAllowlist = new Map<string, Set<string>>();
for (const { file, ref } of PER_SITE_RESIDUE) {
    if (!residueAllowlist.has(file)) residueAllowlist.set(file, new Set());
    residueAllowlist.get(file)!.add(ref);
}

type StaleCitation = {
    file: string;
    line: number;
    ref: string;
    kind: string;
    reason: string;
};

const staleCitations: StaleCitation[] = [];

// Assert each per-site residue entry two ways: mention present, target absent from tree + rosters.
for (const entry of PER_SITE_RESIDUE) {
    const file = entry.file;
    const ref = entry.ref;
    const filePath = resolve(root, file);
    let fileText: string;
    try {
        fileText = await Bun.file(filePath).text();
    } catch {
        staleCitations.push({
            file,
            line: 0,
            ref: "(residue)",
            kind: "residue",
            reason: `residue names ${file} but the file does not exist`,
        });
        continue;
    }
    // Direction 1: the mention is really present in the file
    if (!fileText.includes(ref)) {
        staleCitations.push({
            file,
            line: 0,
            ref,
            kind: "residue",
            reason: `residue entry \`${ref}\` is not present in ${file} — the mention was removed`,
        });
    }
    // Direction 2: the symbol/path is genuinely absent from the tree AND all rosters
    if (resolvesAnywhere(ref, ref.endsWith(".ts") ? "ts-path" : "identifier")) {
        staleCitations.push({
            file,
            line: 0,
            ref,
            kind: "residue",
            reason: `residue entry \`${ref}\` resolves against the tree or a roster — the exemption is over`,
        });
    }
}

// Check each candidate
for (const c of citationCandidates) {
    if (resolvesAnywhere(c.ref, c.kind)) continue; // live — no violation
    // Check if it's in the per-site residue for this file
    const allowlist = residueAllowlist.get(c.file);
    if (allowlist?.has(c.ref)) continue; // deliberately exempt — no violation
    // Stale citation
    staleCitations.push({
        file: c.file,
        line: c.line,
        ref: c.ref,
        kind: c.kind,
        reason: `stale ${c.kind} \`${c.ref}\` does not resolve against the tree or any roster`,
    });
}

if (staleCitations.length > 0) {
    console.error(
        `✗ citation resolution: ${staleCitations.length} stale citation(s) or residue failure(s):\n`,
    );
    for (const v of staleCitations) {
        console.error(`  ${v.file}${v.line ? `:${v.line}` : ""}: ${v.reason}`);
    }
    console.error(
        "\nEvery identifier-shaped token in `.claude/rules/**` (backticked or bare, outside " +
            "fenced code blocks) must resolve against the tree or a committed roster. A token " +
            "that no source file or roster contains is a stale claim. Deliberately-exempt " +
            "mentions (anti-pattern examples, retirement notices, tool/framework references, prose " +
            "terms) are admitted via the per-site residue, asserted two ways: the mention is " +
            "really present, and the symbol is genuinely absent from the tree and all rosters.",
    );
    process.exit(1);
}

const totalRosterEntries = allRosters.reduce((n, { roster }) => n + roster.size, 0);
console.log(
    `✓ doc commands clean (${scanTargets.length} file(s)), ` +
        `install/scaffold/fixture/manifest pins match the manifests (${scanned} doc(s), ${fixtureMatched} fixture line(s), ${manifestPkgCount} manifest(s)), ` +
        `entry-doc chains under budget (${ENTRY_DOC_CHAINS.length} chain(s)), ` +
        `cross-citations resolve (${citationCount} citation(s)), ` +
        `showcase index complete (${showcaseDirs.size} dir(s)), ` +
        `evals task-index complete (${evalsTaskDirs.size} task(s)), ` +
        `tier roster asserted (${rosterSuffixes.length} suffix(es)), ` +
        `citation resolution clean (${citationCandidates.length} citation(s) from ${ruleFiles.length} rule file(s), ` +
        `${allRosters.length} roster(s) with ${totalRosterEntries} entr(y/ies), ` +
        `${PER_SITE_RESIDUE.length} per-site residue entr(y/ies), ` +
        `token index ${tokenIndex.size} token(s) from ${indexSourceFiles.length} source file(s))`,
);
