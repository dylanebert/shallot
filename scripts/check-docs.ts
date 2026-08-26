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

// ── Arm (e): citation resolution — backtick-cited *.ts paths and identifier-shaped symbols ──────
//
// Every backtick-cited `*.ts` path AND every identifier-shaped backtick citation in
// `.claude/rules/**` must resolve against the tree — a cited path that no file matches or a cited
// symbol that no `.ts` file contains is a stale claim in a permanent file. The candidate set is
// drawn from the RULES files (scanning for backtick-cited patterns), NEVER from the live referents
// — candidates drawn from survivors are blind to exactly the dead referents this arm exists to
// find (checks.md: "never draw a resolution arm's candidates from the live referents"). A
// declared allowlist admits deliberately-exempt mentions, grouped into classes each carrying its
// reason and attribution token(s) (a foreign-namespace reference, an anti-pattern example, a
// retirement notice), and every allowlist entry is asserted THREE WAYS: (1) the mention is really
// present in that file, (2) the symbol/path is genuinely absent from the tree, and (3) the class's
// attribution token occurs on the citing line or in the citing line's paragraph — an entry that
// fails any direction reds (checks.md). The third leg (attribution) makes an exemption's safety a
// structural claim rather than the author's say-so: a dead shallot symbol laundered into a
// foreign-namespace class reads 0 on its class's token, so the arm catches it without a human
// re-read. The arm accumulates violations into the exit code rather than printing FAIL and
// exiting 0 (checks.md: "Nobody reads its return value").
//
// The widened detector that found the initial population lives at scripts/detect-stale-claims.ts
// (a one-off audit tool, not a gate — this arm is the gate).
//
// Witnessed red (mutation proofs):
//   (i)  Seeding `deadIdentifierCitation` into `.claude/rules/gpu.md` — a dead non-WGSL identifier,
//         the exact class round 1's arm could not see — reds: `bun run scripts/check-docs.ts`
//         exits 1 with `✗ citation resolution: 1 stale citation(s)`.
//   (ii) Breaking the allowlist entry for `GpuImage` in ecs.md: removing the mention from the
//         file reds (exit 1 — `allowlist entry \`GpuImage\` is not present in .claude/rules/ecs.md`);
//         adding a live `.ts` file containing the symbol reds (exit 1 — `allowlist entry
//         \`GpuImage\` resolves against the tree`).
//   (iii) Seeding `nonexistent/dead.ts` into `.claude/rules/gpu.md` — the round-1 class (a dead
//         `*.ts` path) — still reds: exit 1 with `✗ citation resolution: 1 stale citation(s)`.
//   (iv) Attribution leg: adding an allowlist entry whose class token is absent from its citing
//         line's paragraph reds — e.g. adding `{ file: ".claude/rules/testing.md", ref: "someDeadSym" }`
//         to the Bevy class (attribution ["Bevy"]) when the citing line's paragraph says no "Bevy":
//         exit 1 with `allowlist entry \`someDeadSym\` in class "... Bevy reference ..." fails the
//         attribution leg — none of [Bevy] occurs on the citing line or in its paragraph`
//         (witnessed 2026-08-26, `bun run scripts/check-docs.ts`, exit 1).
//
// Classes this arm CANNOT see:
//   - Prose-name references to removed concepts that don't use a backtick-cited identifier (a
//     paraphrase of a gone function's purpose, not its name) — the arm matches identifiers, not
//     prose. A coordinator grep keyed on prose names covers this class.
//   - References to removed concepts in non-`.ts` files (`.rs`, `.json`, `.toml`) — the arm's
//     symbol resolution greps only `*.ts`. A `.rs` comment naming a gone Rust function is invisible.
//   - Dotted method-call forms (`doc.begin`, `state.has`) — these contain a dot and don't match
//     the identifier regex. They are live by inspection (the object is live, the method is live).

// ── Population: scan .claude/rules/**/*.md for backtick-cited candidates ────────────────
//
// The population is the set of tracked .md files under .claude/rules/, derived from `git ls-files`
// (same law as the doc scan above — the scope is what git tracks, not what the filesystem holds).
// A `**/*.md` walk would read whatever a checkout happens to have on disk.

const rulesTracked = Bun.spawnSync(["git", "ls-files", "-z", "*.md"], {
    cwd: resolve(root, ".claude/rules"),
});
if (!rulesTracked.success) {
    console.error(
        "✗ `git ls-files` failed — the citation-resolution arm needs a git checkout to scope its rule set.",
    );
    process.exit(1);
}
const ruleFiles = rulesTracked.stdout.toString().split("\0").filter(Boolean);
if (ruleFiles.length === 0) {
    console.error(
        "✗ `git ls-files '*.md'` under .claude/rules/ matched nothing — the citation-resolution arm would be vacuously green.",
    );
    process.exit(1);
}

// ── Candidate extraction ─────────────────────────────────────────────────────────────────
//
// A backtick-cited `*.ts` path: a string between backticks ending in `.ts` that is a real file
// path — NOT a suffix pattern (`.test.ts`, `.oracle.ts` — starts with `.`), NOT a glob (`*.test.ts`
// — contains `*`), NOT a command (`bun test ./path.ts` — contains spaces).
//
// An identifier-shaped backtick citation: a string between backticks consisting solely of
// identifier characters (`[A-Za-z_][A-Za-z0-9_]*`), optionally followed by `()` for function-call
// forms. This subsumes the former `*_WGSL`/`*Wgsl`-specific shape and catches every dead
// identifier the old arm missed (e.g. `computeVelocities`, `onsync`, `NUM_RAYS`). Dotted forms
// (`doc.begin`, `state.has`) and paths (`engine/utils/encode.ts`) contain non-identifier
// characters and are excluded by construction.

type CitationCandidate = {
    file: string;
    line: number;
    ref: string;
    kind: "ts-path" | "identifier";
};

const TS_PATH_RE = /`([^`]*\.ts)`/g;
const IDENTIFIER_RE = /`([A-Za-z_][A-Za-z0-9_]*(?:\(\))?)`/g;

const citationCandidates: CitationCandidate[] = [];

for (const file of ruleFiles) {
    const fullPath = resolve(root, ".claude/rules", file);
    const text = await Bun.file(fullPath).text();
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        // .ts paths
        for (const m of line.matchAll(TS_PATH_RE)) {
            const ref = m[1];
            // Exclude suffix patterns (`.test.ts`), globs (`*.test.ts`), commands (contain spaces),
            // and brace expansions (`{encode,tgsl}.ts` — shorthand for multiple files, not a path)
            if (ref.startsWith(".") || ref.includes("*") || ref.includes(" ") || ref.includes("{"))
                continue;
            citationCandidates.push({
                file: `.claude/rules/${file}`,
                line: i + 1,
                ref,
                kind: "ts-path",
            });
        }
        // Identifier-shaped symbols (subsumes *_WGSL/*Wgsl)
        for (const m of line.matchAll(IDENTIFIER_RE)) {
            const ref = m[1];
            // Skip if this was already captured as a .ts path (the identifier regex doesn't match
            // strings containing dots, so this is a safety net, not a real overlap)
            if (ref.endsWith(".ts")) continue;
            citationCandidates.push({
                file: `.claude/rules/${file}`,
                line: i + 1,
                ref,
                kind: "identifier",
            });
        }
    }
}

if (citationCandidates.length === 0) {
    console.error(
        "✗ citation-resolution arm matched no backtick-cited *.ts path or identifier — the arm would be vacuously green.",
    );
    process.exit(1);
}

// ── Resolution: does each candidate resolve against the tree? ────────────────────────────
//
// For .ts paths: the path is relative to an unspecified root — the rules use paths relative to
// `packages/shallot/src/` (e.g., `engine/utils/encode.ts`), to `packages/shallot/` (e.g.,
// `tests/standards.ts`), or to the shallot root (e.g., `scripts/check-docs.ts`). Try each prefix,
// then fall back to a suffix match against the full tracked set (any tracked file ending in
// `/${path}`) — a path like `src/smoke.ts` in a recipe context resolves via the suffix match.
//
// For symbols: `git grep --fixed-strings` in `.ts` files (excluding node_modules) — if the
// symbol appears anywhere in a `.ts` file, it resolves. A symbol that appears only in comments
// still resolves (the arm checks presence, not whether it's a live export — the allowlist handles
// deliberately-retired mentions that happen to survive in comments elsewhere).

const allFilesTracked = Bun.spawnSync(["git", "ls-files", "-z"], { cwd: root });
if (!allFilesTracked.success) {
    console.error(
        "✗ `git ls-files` failed — the citation-resolution arm needs a git checkout to scope its file set.",
    );
    process.exit(1);
}
const trackedSet = new Set(allFilesTracked.stdout.toString().split("\0").filter(Boolean));

function tsPathResolves(path: string): boolean {
    // Try as-is, with packages/shallot/src/, with packages/shallot/
    const tries = [path, `packages/shallot/src/${path}`, `packages/shallot/${path}`];
    for (const t of tries) {
        if (trackedSet.has(t)) return true;
    }
    // Suffix match: any tracked file ending in `/${path}` (handles context-relative paths)
    const suffix = `/${path}`;
    for (const f of trackedSet) {
        if (f.endsWith(suffix)) return true;
    }
    return false;
}

function symbolResolves(symbol: string): boolean {
    // Strip trailing () for function-call forms
    const name = symbol.replace(/\(\)$/, "");
    // Search both .ts and .rs files — the rules cite TypeScript and Rust symbols
    const git = Bun.spawnSync(
        ["git", "-C", root, "grep", "-l", "--fixed-strings", name, "--", "*.ts", "*.rs"],
        { stdout: "pipe", stderr: "pipe" },
    );
    if (!git.success) return false;
    const files = git.stdout.toString().trim().split("\n").filter(Boolean);
    // Exclude node_modules and the arm/detector's own source (their comments mention the symbols they check)
    return files.some(
        (f) =>
            !f.includes("node_modules") &&
            f !== "scripts/check-docs.ts" &&
            f !== "scripts/detect-stale-claims.ts",
    );
}

// ── Allowlist: declared classes of exempt mentions, asserted three ways ───────────────────
//
// The allowlist is a small number of DECLARED CLASSES, each carrying its reason and attribution
// token(s), not a flat list of individual spellings (checks.md: "a declared allow-list both
// readers share, never a tighter regex"). The classes cover the legitimate exempt population:
//
//   1. Foreign-namespace citations the rules deliberately make — Bevy, Jolt, Box3D/C, webphysics,
//      Bullet, PlayCanvas references that are structural analogues, not shallot code.
//   2. Anti-pattern example names — illustrative identifiers that are not real code.
//   3. Standard API references — WGSL built-ins or WebGPU/Vulkan/CUDA API names.
//   4. Tool/framework references — TypeGPU, Bun, or GitHub Actions internals.
//   5. Explicit retirement notices — the mention is deliberate and the target is *meant* to be
//      absent (a gone symbol named in a retirement sentence).
//
// Every entry is asserted THREE WAYS: (1) the mention is really present in that file, (2) the
// symbol/path is genuinely absent from the tree, and (3) the class's attribution token occurs on
// the citing line or in the citing line's paragraph. An entry that fails any direction reds — a
// missing mention means the allowlist is stale, a present-in-tree symbol means the exemption is
// over (the symbol came back), and a missing attribution token means the entry is a candidate
// laundered exemption (a dead shallot symbol smuggled into a foreign-namespace class).

type AllowlistClass = {
    reason: string;
    // The attribution token(s) the class claims — each entry's citing line or its paragraph must
    // contain at least one (case-insensitive, word-boundary). This makes an exemption's safety a
    // structural claim rather than the author's say-so: a dead shallot symbol laundered into a
    // foreign-namespace class reads 0 on its class's token (e.g. compileSurfaceBlock read 0 on
    // "typegpu"/"bun"/"github"), so the arm catches it without a human re-read (checks.md).
    attribution: string[];
    entries: { file: string; ref: string }[];
};

const CITATION_ALLOWLIST_CLASSES: AllowlistClass[] = [
    {
        reason: "foreign-namespace: Bevy reference (structural reference, not shallot code)",
        attribution: ["Bevy"],
        entries: [
            { file: ".claude/rules/ecs.md", ref: "GpuImage" },
            { file: ".claude/rules/ecs.md", ref: "RenderApp" },
            { file: ".claude/rules/ecs.md", ref: "add_slot_edge" },
            { file: ".claude/rules/ecs.md", ref: "ChangeDetection" },
            { file: ".claude/rules/ecs.md", ref: "apply_deferred" },
            { file: ".claude/rules/ecs.md", ref: "SystemParam" },
        ],
    },
    {
        reason: "foreign-namespace: Jolt reference (structural reference, not shallot code)",
        attribution: ["Jolt"],
        entries: [
            { file: ".claude/rules/physics.md", ref: "SolveConstraints" },
            { file: ".claude/rules/physics.md", ref: "WalkStairs" },
        ],
    },
    {
        reason: "foreign-namespace: Box3D/C reference (structural reference, not shallot code)",
        attribution: ["box3d"],
        entries: [
            { file: ".claude/rules/tumble.md", ref: "b3Shape_SetSphere" },
            { file: ".claude/rules/tumble.md", ref: "SetCapsule" },
            { file: ".claude/rules/tumble.md", ref: "b3Shape_SetFilter" },
        ],
    },
    {
        reason: "foreign-namespace: webphysics reference (author's workspace layout, not shallot code)",
        attribution: ["webphysics"],
        entries: [
            { file: ".claude/rules/avbd.md", ref: "broadPhase.ts" },
            { file: ".claude/rules/avbd.md", ref: "reference/webphysics/.../avbdState.ts" },
            { file: ".claude/rules/avbd.md", ref: "contactSlop" },
            { file: ".claude/rules/avbd.md", ref: "dispatchBodyCount" },
        ],
    },
    {
        reason: "foreign-namespace: Bullet 3 reference (structural reference, not shallot code)",
        attribution: ["Bullet"],
        entries: [{ file: ".claude/rules/avbd.md", ref: "BatchSolveKernelContact" }],
    },
    {
        reason: "anti-pattern example: illustrative name, not real code",
        attribution: ["not"],
        entries: [
            { file: ".claude/rules/style.md", ref: "createMeshGeometryFromVertices" },
            { file: ".claude/rules/style.md", ref: "prepareX" },
            { file: ".claude/rules/style.md", ref: "buildY" },
            { file: ".claude/rules/style.md", ref: "applyZ" },
            { file: ".claude/rules/ecs.md", ref: "lastState" },
            { file: ".claude/rules/ecs.md", ref: "resetIfNewState" },
            { file: ".claude/rules/ecs.md", ref: "lastCamera" },
            { file: ".claude/rules/exports.md", ref: "readBuffer" },
        ],
    },
    {
        reason: "standard API reference: WGSL built-in or WebGPU/Vulkan/CUDA API name, not shallot code",
        attribution: ["WGSL", "WebGPU", "Vulkan", "CUDA"],
        entries: [
            { file: ".claude/rules/gpu.md", ref: "pack4x8snorm" },
            { file: ".claude/rules/gpu.md", ref: "subgroupAdd" },
            { file: ".claude/rules/gpu.md", ref: "__threadfence()" },
            { file: ".claude/rules/gpu.md", ref: "GPURenderBundle" },
            { file: ".claude/rules/gpu.md", ref: "executeBundles" },
            { file: ".claude/rules/render.md", ref: "ExecuteIndirect" },
            { file: ".claude/rules/render.md", ref: "vkCmdDrawIndirectCount" },
        ],
    },
    {
        reason: "tool/framework reference: TypeGPU, Bun, or GitHub Actions internal, not shallot code",
        attribution: ["TypeGPU", "Bun", "GitHub"],
        entries: [
            { file: ".claude/rules/testing.md", ref: "disabled_manually" },
            { file: ".claude/rules/testing.md", ref: "__TYPEGPU_AUTONAME__" },
            { file: ".claude/rules/testing.md", ref: "shaderModules" },
            { file: ".claude/rules/testing.md", ref: "DependencyLoop" },
            { file: ".claude/rules/gpu.md", ref: "list_typegpu_exports" },
            { file: ".claude/rules/exports.md", ref: "sideEffects" },
        ],
    },
    {
        reason: "retirement notice: deliberately names a gone symbol",
        attribution: ["gone"],
        entries: [
            { file: ".claude/rules/exports.md", ref: "BVH_TRAVERSE_WGSL" },
            { file: ".claude/rules/exports.md", ref: "FogLight" },
        ],
    },
    {
        reason: "foreign-namespace: PlayCanvas reference (structural reference, not shallot code)",
        attribution: ["PlayCanvas"],
        entries: [{ file: ".claude/rules/render.md", ref: "LightTextureAtlas" }],
    },
];

// Build a flat lookup map from the classes for the candidate check
const citationAllowlist = new Map<string, Set<string>>();
for (const cls of CITATION_ALLOWLIST_CLASSES) {
    for (const { file, ref } of cls.entries) {
        if (!citationAllowlist.has(file)) citationAllowlist.set(file, new Set());
        citationAllowlist.get(file)!.add(ref);
    }
}

type StaleCitation = {
    file: string;
    line: number;
    ref: string;
    kind: string;
    reason: string;
};

const staleCitations: StaleCitation[] = [];

// Helper: get the paragraph text (contiguous non-blank lines) containing a 1-indexed line number,
// plus the immediately preceding non-blank line(s) if separated by exactly one blank line and
// the preceding block is a short heading/intro (< 200 chars). This catches section headings like
// "**Take from Bevy:**" that introduce a list — the heading carries the attribution token even
// when the individual bullet does not.
function getParagraphText(lines: string[], lineno: number): string {
    const idx = lineno - 1;
    let start = idx;
    while (start > 0 && lines[start - 1].trim()) start--;
    let end = idx;
    while (end < lines.length - 1 && lines[end + 1].trim()) end++;
    // Check for a preceding heading/intro block separated by one blank line
    if (start > 1 && !lines[start - 1].trim()) {
        let prev = start - 2;
        if (prev >= 0 && lines[prev].trim()) {
            let prevStart = prev;
            while (prevStart > 0 && lines[prevStart - 1].trim()) prevStart--;
            const headingText = lines.slice(prevStart, prev + 1).join("");
            if (headingText.length < 200) {
                return lines.slice(prevStart, end + 1).join("\n");
            }
        }
    }
    return lines.slice(start, end + 1).join("\n");
}

// First, assert every allowlist entry three ways: mention present, target absent, attribution
// token present on the citing line or in its paragraph.
for (const cls of CITATION_ALLOWLIST_CLASSES) {
    for (const { file, ref } of cls.entries) {
        const filePath = resolve(root, file);
        let fileText: string;
        try {
            fileText = await Bun.file(filePath).text();
        } catch {
            staleCitations.push({
                file,
                line: 0,
                ref: "(allowlist)",
                kind: "allowlist",
                reason: `allowlist names ${file} but the file does not exist`,
            });
            continue;
        }
        // Direction 1: the mention is really present in the file
        if (!fileText.includes(ref)) {
            staleCitations.push({
                file,
                line: 0,
                ref,
                kind: "allowlist",
                reason: `allowlist entry \`${ref}\` is not present in ${file} — the mention was removed`,
            });
        }
        // Direction 2: the symbol/path is genuinely absent from the tree
        const resolves = ref.endsWith(".ts") ? tsPathResolves(ref) : symbolResolves(ref);
        if (resolves) {
            staleCitations.push({
                file,
                line: 0,
                ref,
                kind: "allowlist",
                reason: `allowlist entry \`${ref}\` resolves against the tree — the exemption is over (the symbol/path came back)`,
            });
        }
        // Direction 3 (attribution leg): the class's attribution token occurs on the citing line
        // or in the citing line's paragraph. A dead shallot symbol laundered into a foreign-
        // namespace class reads 0 here — the citing line carries no foreign-namespace attribution.
        const fileLines = fileText.split("\n");
        let citingLine = -1;
        for (let i = 0; i < fileLines.length; i++) {
            if (fileLines[i].includes(`\`${ref}\``)) {
                citingLine = i + 1;
                break;
            }
        }
        if (citingLine === -1) continue; // already flagged by direction 1
        const paraText = getParagraphText(fileLines, citingLine).toLowerCase();
        const hasAttribution = cls.attribution.some((tok) => {
            const re = new RegExp(`\\b${tok.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
            return re.test(paraText);
        });
        if (!hasAttribution) {
            staleCitations.push({
                file,
                line: citingLine,
                ref,
                kind: "allowlist",
                reason: `allowlist entry \`${ref}\` in class "${cls.reason}" fails the attribution leg — none of [${cls.attribution.join(", ")}] occurs on the citing line or in its paragraph`,
            });
        }
    }
}

// Then, check each candidate
for (const c of citationCandidates) {
    const resolves = c.kind === "ts-path" ? tsPathResolves(c.ref) : symbolResolves(c.ref);
    if (resolves) continue; // live — no violation
    // Check if it's allowlisted for this file
    const allowlist = citationAllowlist.get(c.file);
    if (allowlist?.has(c.ref)) continue; // deliberately exempt — no violation
    // Stale citation
    staleCitations.push({
        file: c.file,
        line: c.line,
        ref: c.ref,
        kind: c.kind,
        reason: `stale ${c.kind} \`${c.ref}\` does not resolve against the tree`,
    });
}

if (staleCitations.length > 0) {
    console.error(
        `✗ citation resolution: ${staleCitations.length} stale citation(s) or allowlist failure(s):\n`,
    );
    for (const v of staleCitations) {
        console.error(`  ${v.file}${v.line ? `:${v.line}` : ""}: ${v.reason}`);
    }
    console.error(
        "\nEvery backtick-cited `*.ts` path and identifier-shaped backtick citation in " +
            "`.claude/rules/**` must resolve against the tree. A cited path that no file matches " +
            "or a cited symbol that no `.ts` file contains is a stale claim. Deliberately-exempt " +
            "mentions (foreign-namespace references, anti-pattern examples, retirement notices) " +
            "are admitted via the declared allowlist, asserted three ways: the mention is really " +
            "present, the symbol/path is genuinely absent, and the class's attribution token " +
            "occurs on the citing line or in its paragraph.",
    );
    process.exit(1);
}

console.log(
    `✓ doc commands clean (${scanTargets.length} file(s)), ` +
        `install/scaffold/fixture/manifest pins match the manifests (${scanned} doc(s), ${fixtureMatched} fixture line(s), ${manifestPkgCount} manifest(s)), ` +
        `entry-doc chains under budget (${ENTRY_DOC_CHAINS.length} chain(s)), ` +
        `cross-citations resolve (${citationCount} citation(s)), ` +
        `showcase index complete (${showcaseDirs.size} dir(s)), ` +
        `evals task-index complete (${evalsTaskDirs.size} task(s)), ` +
        `tier roster asserted (${rosterSuffixes.length} suffix(es)), ` +
        `citation resolution clean (${citationCandidates.length} citation(s), ` +
        `${CITATION_ALLOWLIST_CLASSES.length} allowlist class(es), ` +
        `${CITATION_ALLOWLIST_CLASSES.reduce((n, c) => n + c.entries.length, 0)} allowlist entr(y/ies))`,
);
