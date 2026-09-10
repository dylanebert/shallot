import { existsSync, lstatSync, readFileSync, realpathSync, statSync } from "node:fs";
import { Glob } from "bun";
import { dirname, relative, resolve } from "path";
import { TEST_TIER_SUFFIX_NAMES } from "./test-tiers";

async function readScripts(pkgPath: string): Promise<Record<string, string>> {
    const pkg = (await Bun.file(pkgPath).json()) as { scripts?: Record<string, string> };
    return pkg.scripts ?? {};
}

// Expand the root `workspaces` globs (bun's own resolution: a literal dir, or `<base>/*`) into
// every member's package.json path.
async function workspacePkgPaths(rootDir: string, patterns: string[]): Promise<string[]> {
    const paths: string[] = [];
    for (const pattern of patterns) {
        const starIdx = pattern.indexOf("*");
        if (starIdx === -1) {
            paths.push(resolve(rootDir, pattern, "package.json"));
            continue;
        }
        const base = pattern.slice(0, starIdx).replace(/\/$/, "");
        const glob = new Glob(pattern.slice(base.length + 1));
        for await (const match of glob.scan({ cwd: resolve(rootDir, base), onlyFiles: false })) {
            paths.push(resolve(rootDir, base, match, "package.json"));
        }
    }
    return paths.filter((p) => existsSync(p));
}

// Every declared script resolves to an existing file/dir, or delegates to a script its target
// declares. `bunx` segments and bare `bun test` filters are external and unchecked.
async function checkExists(pkgPaths: string[]): Promise<{ detail: string }[]> {
    const violations: { detail: string }[] = [];
    for (const pkgPath of pkgPaths) {
        const dir = dirname(pkgPath);
        for (const [name, cmd] of Object.entries(await readScripts(pkgPath))) {
            for (const segment of cmd.split("&&")) {
                if (/\bbunx\s+/.test(segment)) continue;
                const m = segment.match(/\bbun\s+(?:(run|test)\s+)?(?:--cwd\s+(\S+)\s+)?(\S+)/);
                if (!m) continue;
                const [, verb, cwdArg, token] = m;
                if (token.startsWith("-")) continue;
                const base = cwdArg ? resolve(dir, cwdArg) : dir;
                if (token.includes("/") || token.includes(".")) {
                    const target = resolve(base, token);
                    if (!existsSync(target))
                        violations.push({
                            detail: `${pkgPath} ${name}: target "${token}" does not exist (resolved ${target})`,
                        });
                    continue;
                }
                if (verb === "test") continue;
                const targetPkgPath = resolve(base, "package.json");
                if (!existsSync(targetPkgPath)) continue;
                if (!(token in (await readScripts(targetPkgPath))))
                    violations.push({
                        detail: `${pkgPath} ${name}: delegates to script "${token}" not declared in ${targetPkgPath}`,
                    });
            }
        }
    }
    return violations;
}

/** Engine `files` entries written at build or pack time: tooling bundles and audio wasm. */
const PRODUCED = ["dist", "crates/audio/pkg"];

/** Every declared bin and positive files entry must exist or have a pack producer. */
async function checkRealization(root: string): Promise<string[]> {
    const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    const manifests = [
        resolve(root, "package.json"),
        ...(await workspacePkgPaths(root, pkg.workspaces ?? [])),
    ];
    const errors: string[] = [];
    for (const manifest of manifests) {
        const dir = dirname(manifest);
        const value = JSON.parse(readFileSync(manifest, "utf8"));
        const bins: string[] =
            typeof value.bin === "string" ? [value.bin] : Object.values(value.bin ?? {});
        const files: string[] = value.files ?? [];
        const projected = (target: string, kind: "bin" | "files"): boolean =>
            dir === resolve(root) && kind === "files" && PRODUCED.includes(target);
        for (const [kind, targets] of [
            ["bin", bins],
            ["files", files.filter((file) => !file.startsWith("!"))],
        ] as const) {
            for (const declared of targets) {
                const target = declared.replace(/^\.\//, "");
                const path = resolve(dir, target);
                const present =
                    kind === "bin"
                        ? existsSync(path) && statSync(path).isFile()
                        : existsSync(path) ||
                          [...new Glob(target).scanSync({ cwd: dir })].length > 0;
                if (
                    (path !== dir && !path.startsWith(`${dir}/`)) ||
                    (!present && !projected(target, kind))
                )
                    errors.push(
                        `${relative(root, manifest)} ${kind}: ${declared} is missing and has no pack-time projection`,
                    );
            }
        }
    }
    return errors;
}

// Consumer commands use the installed bin; repository commands must resolve in this tree,
// without a global link or bunx downloading an unrelated registry version.

const root = resolve(import.meta.dir, "..");
const commandErrors = await checkRealization(root);
const entry = (await Bun.file(resolve(root, "AGENTS.md")).text())
    .split("## Commands\n")[1]
    ?.split("\n## ")[0];
if (!entry) commandErrors.push("AGENTS.md: missing Commands block");
const scripts = (await Bun.file(resolve(root, "package.json")).json()).scripts;
let inCommandFence = false;
let commandCount = 0;
for (const line of (entry ?? "").split("\n")) {
    if (line.startsWith("```")) {
        inCommandFence = !inCommandFence;
        continue;
    }
    if (!inCommandFence) continue;
    for (const segment of line.split("#")[0].split(/&&|;/)) {
        const command = segment.trim();
        if (!command || command.startsWith("#")) continue;
        commandCount++;
        const match = /^(bunx|bun)\s+(?:run\s+)?([^\s]+)/.exec(command);
        const token = match?.[2];
        let reachable = false;
        if (match?.[1] === "bunx") {
            const bin = resolve(root, "node_modules/.bin", token!);
            const expected = resolve(root, "bin/shallot.ts");
            reachable =
                token === "shallot" &&
                existsSync(bin) &&
                existsSync(expected) &&
                realpathSync(bin) === realpathSync(expected);
        } else if (token) {
            reachable =
                Object.hasOwn(scripts, token) ||
                (token.includes("/") && existsSync(resolve(root, token)));
        }
        if (!reachable) commandErrors.push(`AGENTS.md: unreachable repository command: ${command}`);
    }
}
if (!commandCount) commandErrors.push("AGENTS.md: empty command population");
commandErrors.push(
    ...(await checkExists([resolve(root, "package.json")])).map((error) => error.detail),
);
if (commandErrors.length) {
    console.error(`✗ command resolution:\n${commandErrors.join("\n")}`);
    process.exit(1);
}
console.log(
    `✓ command resolution: ${commandCount} repository commands; declared bin/files realization`,
);

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
// tracked docs (MIGRATION.md, CHANGELOG.md) that sibling arms already scanned, so a
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

const SUBCOMMAND = "(dev|build|run|add|check)";
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
    typegpu: { manifest: "package.json", field: "peerDependencies" },
    "unplugin-typegpu": { manifest: "package.json", field: "dependencies" },
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

// The install-test fixtures are a second pin site: the install gate green-lights whatever version
// they carry, so a fixture stuck on the old minor certifies the drifted install as working. No
// hand-list of line numbers — scan every line naming `typegpu` with a version token attached
// (real code, not the prose comments that also mention the version for context) and classify each
// one, so a fixture bump landing outside today's known sites can't go unnoticed. `typegpu2` is a
// deliberate second physical copy (`identityFlow`'s duplicate-identity proof) whose version must
// still track the engine peer; `PM_RED_COPY_VERSION` is the one deliberate exclusion — that
// fixture needs a differing version on purpose. Mutation proof: bumping the first `typegpu` pin
// in `scripts/install-test.ts` from `~0.12.4` to `~0.12.5` reds this arm (witnessed 2026-09-01,
// exit 1 — `scripts/install-test.ts:375: typegpu@~0.12.5 — the manifest declares ~0.12.4`).
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
// `typegpu`/`unplugin-typegpu` at a range the canonical manifest
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
// a single file: `AGENTS.md` (which `CLAUDE.md` imports) plus whichever leaf
// directory's own `AGENTS.md` it's working under.
// The budget is enforced per chain, since a bump that
// keeps every individual file under budget can still blow the chain a reader loads (measured
// 2026-08-16: the published-package chain sat 3 B under 32768).
const ENTRY_DOC_BUDGET = 32768;
const ENTRY_DOC_CHAINS: string[][] = [["AGENTS.md"], ["AGENTS.md", "examples/AGENTS.md"]];

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

// ── Arm (d): tier-suffix roster — one constant, derived consumers ─────────────────────────
// Prose explains tier obligations without duplicating the constant as a heading/bullet roster.
const rosterFindings: string[] = [];

// Derive the consumer set: scan every tracked file in the repo for a literal tier-suffix roster —
// a line enumerating 3+ of the roster's suffix names either as bare words with regex alternation (`|`) or
// as an array literal of quoted `.suffix.ts` strings (e.g. `[".oracle.ts", ".probes.ts", ...]`).
// Any file that carries such a roster is restating it rather than reading the shared constant; the
// arm finds such files itself, so a new file restating the roster — in either shape — is caught
// without updating a hand-list.
//
// Two exclusions, stated explicitly:
// 1. `scripts/test-tiers.ts` — the roster's own definition module; it MUST contain the
//    suffix names (it is where the constant lives).
// 2. `scripts/check-docs.ts` — this arm's own text; a self-referential gate matches its own
//    description of what it checks (per `.claude/rules/specs.md`'s self-reference principle).
const ROSTER_EXCLUSIONS = new Set(["scripts/test-tiers.ts", "scripts/check-docs.ts"]);
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
    let source: string;
    try {
        source = await Bun.file(resolve(root, file)).text();
    } catch {
        continue; // file deleted but not yet committed — skip
    }
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
        "\nDerive test-tier suffix rosters from the shared test-tiers.ts constant, never restate them.",
    );
    process.exit(1);
}

const trackedFiles = allTrackedFiles.stdout.toString().split("\0").filter(Boolean);
// ── Arm (f): pointer-validity — dead *.md path citations in comments ──────────────────
//
// A comment in a .ts file citing a *.md path that resolves to nothing in-repo reds.
// style.md:45 — "A comment anchored to something outside this repo rots invisibly." A
// *.md path citation is the greppable surface form of that anchor: `roads-interactive.md`,
// `ecs.md`, `gpu.md:110`. The arm scans every tracked .ts file's comment lines for *.md
// path citations and checks each basename against the set of .md files tracked in the
// shallot repo. A citation that
// resolves to nothing is a dead anchor — it reads as authoritative for years.
//
// The discrimination follows the fixture-pin scan's shape (the `fixtureUnclassified`
// classifier at the FIXTURE_FILE scan): match a pattern, check if it resolves, red if
// not. The fixture-pin scan splits real pins from prose mentions by skipping comment
// lines and matching a pin regex; this arm does the inverse — it scans ONLY comment
// lines (where dead anchors live) and matches a *.md path regex. Non-comment lines
// (string literals, code) are not scanned: a .md path in a string literal is not a
// citation, it's a test description or a path.
//
// False positives preserved (asserted by presence, not just spared):
// - physics's `// Stage N:` algorithm-step labels (body.ts ×6, tree.ts ×3) — they name
//   the ported algorithm's own stages, not a workflow anchor (style.md:43). No *.md
//   path → not matched.
// - AASHTO derivation cites in flatten.ts (×2) — cite an external standard, not a
//   *.md path. Not matched.
// - English "used to <verb>" — no *.md path. Not matched.
//
// Witnessed red (pre-sweep tree, 2026-08-26): 46 dead *.md path citations in comments
// (43 `roads-interactive.md`, 1 `shallot-boot-noise.md`, 2
// `shallot-demo-slow-frame-attribution.md`) → exit 1.

// A citation resolves only if its basename is tracked in this repo, so the gate
// holds for a standalone reader without access to a private containing repository.
const resolvedMdBasenames = new Set<string>();
for (const path of docs) {
    resolvedMdBasenames.add(path.split("/").pop()!);
}

// A *.md path citation in a comment: a word ending in .md, not preceded by a word
// character or hyphen (so `unplugin-typegpu@…` does not match — its .md is not a path
// citation). The `(?<![-\w])` lookbehind is the same boundary law as PIN_RE's. The
// .md extension is the anchor — a bare `roads-interactive` without .md is a stage-ID
// anchor, not a path citation, and is swept by S2 rather than gated here.
const MD_PATH_RE = /(?<![-\w])([a-zA-Z][a-zA-Z0-9_-]*\.md)\b/g;

// Scan every tracked .ts file's comment lines for *.md path citations. Excludes
// `scripts/check-docs.ts` — its own comments describe the citation patterns the arm
// matches (e.g. `rule.md "phrase"`), which are not real citations.
const POINTER_EXCLUSION = new Set(["scripts/check-docs.ts"]);
type DeadPointer = { file: string; line: number; ref: string };
const deadPointers: DeadPointer[] = [];
let pointerCitationCount = 0;

for (const file of trackedFiles) {
    if (!file.endsWith(".ts") || POINTER_EXCLUSION.has(file)) continue;
    let source: string;
    try {
        source = await Bun.file(resolve(root, file)).text();
    } catch {
        continue; // file deleted but not yet committed — skip
    }
    const lines = source.split("\n");
    for (let i = 0; i < lines.length; i++) {
        const trimmed = lines[i].trim();
        // Comment lines only — `//` line comments, `*` block-comment continuations,
        // `/*` block-comment openers. Non-comment lines are not scanned.
        if (!trimmed.startsWith("//") && !trimmed.startsWith("*") && !trimmed.startsWith("/*"))
            continue;
        for (const [, ref] of lines[i].matchAll(MD_PATH_RE)) {
            pointerCitationCount++;
            if (!resolvedMdBasenames.has(ref)) {
                deadPointers.push({ file, line: i + 1, ref });
            }
        }
    }
}

if (pointerCitationCount === 0) {
    console.error(
        "✗ pointer-validity arm matched no `*.md` path citation in any comment — the arm would be vacuously green.",
    );
    process.exit(1);
}

if (deadPointers.length > 0) {
    console.error(
        `✗ pointer-validity: ${deadPointers.length} comment(s) cite a *.md path that resolves to nothing in-repo:\n`,
    );
    for (const p of deadPointers) {
        console.error(`  ${p.file}:${p.line}: ${p.ref}`);
    }
    console.error(
        "\nA comment citing a *.md path that resolves to nothing in-repo is a dead anchor " +
            "(style.md:45 — a comment anchored to something outside this repo rots " +
            "invisibly). Rewrite the comment as the invariant that holds today, or delete " +
            "it. This gate prevents re-accretion; it does not sweep what is already there.",
    );
    process.exit(1);
}

// Command composition is lexical documentation validation, not shell equivalence or NLP.
// Scan tracked docs for root-test command comments enumerating paths after "over", "in", or "paths:".
const rootScripts = (await Bun.file(resolve(root, "package.json")).json()).scripts as Record<
    string,
    string
>;
const testCommand = /(?:^| && )bun test\s+([^&]+)$/.exec(rootScripts.test ?? "");
if (!testCommand) {
    console.error("✗ command composition: missing test arguments");
    process.exit(1);
}
const testPaths = testCommand[1]
    .trim()
    .split(/\s+/)
    .filter((arg) => !arg.startsWith("--"));
if (testPaths.some((path) => path.startsWith("-") || /[;&|]/.test(path))) {
    console.error("✗ command composition: unsupported root test syntax; update the reader");
    process.exit(1);
}
const compositionFindings: string[] = [];
let testCones = 0;
for (const file of docs) {
    const text = await Bun.file(resolve(root, file)).text();
    for (const [index, line] of text.split("\n").entries()) {
        const comment = /^\s*bun run test\s+#\s*(.*)$/.exec(line)?.[1];
        if (!comment) continue;
        const cone = /\b(?:over|in|paths:)\s+([^()]+)/.exec(comment)?.[1];
        if (!cone) continue;
        testCones++;
        const paths = cone
            .trim()
            .replace(/`/g, "")
            .split(/[,\s]+/);
        const missing = testPaths.filter((path) => !paths.includes(path));
        const extra = paths.filter((path) => !testPaths.includes(path));
        if (missing.length || extra.length || new Set(paths).size !== paths.length) {
            compositionFindings.push(
                `${file}:${index + 1}: stale root test cone: missing [${missing.join(", ")}], extra [${extra.join(", ")}]`,
            );
        }
    }
}
if (compositionFindings.length) {
    console.error(`✗ command composition:\n${compositionFindings.join("\n")}`);
    process.exit(1);
}
console.log(
    `✓ command composition (${testPaths.length} manifest test paths, ${testCones} restated cones)`,
);

// One closed Git population; ignored files and other instruction names are outside this arm.
// Paragraphs are nonempty blank-line-delimited blocks, measured in Unicode characters.
const lowerInstructions = process.argv.slice(2);
if (lowerInstructions.length && lowerInstructions.join(" ") !== "--lower") {
    console.error("✗ instruction ratchet: expected no arguments or --lower");
    process.exit(1);
}
const instructionListing = Bun.spawnSync(["git", "ls-files", "-co", "--exclude-standard", "-z"], {
    cwd: root,
});
const instructionModes = Bun.spawnSync(["git", "ls-files", "--stage", "-z"], { cwd: root });
if (!instructionListing.success || !instructionModes.success) {
    console.error("✗ instruction ratchet: Git population unavailable");
    process.exit(1);
}
const instructionFiles = [...new Set(instructionListing.stdout.toString().split("\0"))]
    .filter((file) => /(?:^|\/)(?:AGENTS|CLAUDE)\.md$/.test(file))
    .sort();
const symlinkFiles = new Set(
    instructionModes.stdout
        .toString()
        .split("\0")
        .filter((entry) => entry.startsWith("120000 "))
        .map((entry) => entry.slice(entry.indexOf("\t") + 1)),
);
type InstructionSize = { bytes: number; paragraph: number };
type InstructionBudget = { total: number; files: Record<string, InstructionSize> };
const measured: InstructionBudget = { total: 0, files: {} };
const instructionFindings: string[] = [];
if (!instructionFiles.length) instructionFindings.push("empty population");
for (const file of instructionFiles) {
    try {
        if (symlinkFiles.has(file) || lstatSync(resolve(root, file)).isSymbolicLink()) {
            instructionFindings.push(`symlink member: ${file}`);
            continue;
        }
        const text = await Bun.file(resolve(root, file)).text();
        const bytes = Buffer.byteLength(text);
        const paragraph = Math.max(
            0,
            ...text.split(/\n\s*\n/).map((part) => [...part.trim()].length),
        );
        measured.files[file] = { bytes, paragraph };
        measured.total += bytes;
    } catch {
        instructionFindings.push(`unreadable member: ${file}`);
    }
}
const instructionBaseline = Bun.file(resolve(root, "scripts/instruction-budget.json"));
if (await instructionBaseline.exists()) {
    const budget = (await instructionBaseline.json()) as InstructionBudget;
    const ceiling = (value: number) => Number.isSafeInteger(value) && value >= 0;
    if (!ceiling(budget.total) || !budget.files || Array.isArray(budget.files)) {
        instructionFindings.push("invalid baseline");
    } else {
        for (const [file, size] of Object.entries(budget.files)) {
            if (!size || !ceiling(size.bytes) || !ceiling(size.paragraph)) {
                instructionFindings.push(`invalid ceiling: ${file}`);
            }
        }
        for (const [file, size] of Object.entries(measured.files)) {
            const cap = budget.files[file];
            if (!Object.hasOwn(budget.files, file))
                instructionFindings.push(`unlisted member: ${file}`);
            else {
                if (size.bytes > cap.bytes)
                    instructionFindings.push(`byte growth: ${file} ${size.bytes} > ${cap.bytes}`);
                if (size.paragraph > cap.paragraph)
                    instructionFindings.push(
                        `paragraph growth: ${file} ${size.paragraph} > ${cap.paragraph}`,
                    );
            }
        }
        if (measured.total > budget.total)
            instructionFindings.push(`corpus growth: ${measured.total} > ${budget.total}`);
    }
} else if (!lowerInstructions.length) {
    instructionFindings.push("missing baseline; seed explicitly with --lower");
}
if (instructionFindings.length) {
    console.error(`✗ instruction ratchet:\n${instructionFindings.join("\n")}`);
    process.exit(1);
}
if (lowerInstructions.length) {
    await Bun.write(instructionBaseline, `${JSON.stringify(measured, null, 4)}\n`);
}
console.log(
    `✓ instruction ratchet (${instructionFiles.length} files, ${measured.total} bytes; ${lowerInstructions.length ? "lowered" : "validate-only"})`,
);

console.log(
    `✓ doc commands clean (${scanTargets.length} file(s)), ` +
        `install/scaffold/fixture/manifest pins match the manifests (${scanned} doc(s), ${fixtureMatched} fixture line(s), ${manifestPkgCount} manifest(s)), ` +
        `entry-doc chains under budget (${ENTRY_DOC_CHAINS.length} chain(s)), ` +
        `tier restatements absent (${suffixWords.length} suffix(es)), ` +
        `pointer-validity clean (${pointerCitationCount} .md citation(s))`,
);
