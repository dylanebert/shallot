import { expect } from "bun:test";
import {
    cpSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    renameSync,
    rmSync,
    statSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import {
    collectPopulation,
    discoverTestFiles,
    readSurface,
    renderWorkflow,
    selectIntegrationRows,
    subjectTokens,
    writeWorkflow,
} from "@dylanebert/shallot/harness/surface";

const ROOT = resolve(import.meta.dir, "..");
const FIXTURES = resolve(ROOT, "scripts/fixtures/surface");

check(
    "physics Rust test targets stay partitioned",
    {
        claim: "the physics unit and gold Cargo rows partition every current Rust test target exactly once, keeping non-gold stages out of the C-reference gold population",
        size: "integration",
    },
    () => {
        const unitDeclaration = readFileSync(resolve(ROOT, "crates/physics/unit.test.ts"), "utf8");
        const goldDeclaration = readFileSync(resolve(ROOT, "crates/physics/gold.test.ts"), "utf8");
        expect(unitDeclaration).toMatch(
            /runCargoTest\("shallot-physics",\s*"--lib",\s*"--test",\s*"stages"\)/,
        );
        expect(unitDeclaration.match(/"--lib"/g)?.length).toBe(1);
        expect(goldDeclaration).not.toContain('"--lib"');
        const metadata = Bun.spawnSync(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            {
                cwd: ROOT,
                stdout: "pipe",
                stderr: "pipe",
            },
        );
        expect(metadata.exitCode).toBe(0);
        const physics = JSON.parse(metadata.stdout.toString()).packages.find(
            (pkg: { name: string }) => pkg.name === "shallot-physics",
        );
        const targets = physics.targets
            .filter((target: { kind: string[] }) => target.kind.includes("test"))
            .map((target: { name: string }) => target.name)
            .sort();
        const goldTargets = targets.filter((target: string) => target.endsWith("_gold"));
        const unitTargets = targets.filter((target: string) => !target.endsWith("_gold"));
        expect(unitTargets).toEqual(["stages"]);
        expect(goldTargets).toHaveLength(11);
        for (const target of targets) {
            const escaped = target.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
            const occurrences = [unitDeclaration, goldDeclaration].reduce(
                (count, declaration) =>
                    count +
                    (declaration.match(new RegExp(`"--test"\\s*,\\s*"${escaped}"`, "g"))?.length ??
                        0),
                0,
            );
            expect(occurrences).toBe(1);
            const owner = target.endsWith("_gold") ? goldDeclaration : unitDeclaration;
            expect(owner).toMatch(new RegExp(`"--test"\\s*,\\s*"${escaped}"`));
        }
    },
);

// Fixture check files are stored with a trailing `.fixture` so the real discovery and the real
// runner never see them; materializing strips it, giving the production readers a real tree.
function unfixture(dir: string): void {
    for (const entry of readdirSync(dir)) {
        const path = join(dir, entry);
        if (statSync(path).isDirectory()) {
            unfixture(path);
        } else if (entry.endsWith(".fixture")) {
            renameSync(path, path.slice(0, -".fixture".length));
        }
    }
}

function seed(name: string): string {
    const tree = mkdtempSync(join(tmpdir(), `shallot-surface-${name}-`));
    cpSync(resolve(FIXTURES, name), tree, { recursive: true });
    for (const dir of ["src", "scripts", "examples"])
        mkdirSync(join(tree, dir), { recursive: true });
    unfixture(tree);
    return tree;
}

function run(
    script: string,
    tree: string,
    ...args: string[]
): { code: number; out: string; err: string } {
    const proc = Bun.spawnSync(
        ["bun", resolve(ROOT, "scripts", script), "--list", ...args, "--root", tree],
        {
            cwd: ROOT,
        },
    );
    return {
        code: proc.exitCode ?? -1,
        out: proc.stdout.toString().trim(),
        err: proc.stderr.toString().trim(),
    };
}

function reader(name: string): { code: number; out: string; err: string } {
    const tree = seed(name);
    try {
        return run("check-surface.ts", tree);
    } finally {
        rmSync(tree, { recursive: true, force: true });
    }
}

function dependencyViolations(
    spec: string,
    packageName = "consumer",
    lock?: string,
    dependencyName = "@dylanebert/shallot-grid",
): string[] {
    const tree = mkdtempSync(join(tmpdir(), "shallot-surface-dependency-"));
    try {
        writeFileSync(
            join(tree, "package.json"),
            JSON.stringify({
                name: packageName,
                dependencies: { [dependencyName]: spec },
            }),
        );
        if (lock !== undefined) writeFileSync(join(tree, "bun.lock"), lock);
        return readSurface(tree);
    } finally {
        rmSync(tree, { recursive: true, force: true });
    }
}

check(
    "surface: link Shallot specs refuse",
    { claim: "the surface gate refuses link Shallot package specs" },
    () => {
        expect(dependencyViolations("link:../shallot").join("\\n")).toContain("link:");
    },
);

check(
    "surface: file Shallot specs refuse",
    { claim: "the surface gate refuses file Shallot package specs" },
    () => {
        expect(dependencyViolations("file:../shallot").join("\\n")).toContain("file:");
    },
);

check(
    "surface: git Shallot specs refuse",
    { claim: "the surface gate refuses git Shallot package specs" },
    () => {
        expect(
            dependencyViolations("git+https://github.com/dylanebert/shallot.git").join("\\n"),
        ).toContain("git");
    },
);

check(
    "surface: github Shallot specs refuse",
    { claim: "the surface gate refuses github Shallot package specs" },
    () => {
        expect(dependencyViolations("github:dylanebert/shallot#main").join("\\n")).toContain(
            "github:",
        );
    },
);

check(
    "surface: URL Shallot specs refuse",
    { claim: "the surface gate refuses URL Shallot package specs" },
    () => {
        expect(
            dependencyViolations("https://github.com/dylanebert/shallot-grid.git").join("\\n"),
        ).toContain("URL");
    },
);

check(
    "surface: self-link Shallot spec passes",
    { claim: "the surface gate permits a package's own self-link" },
    () => {
        expect(dependencyViolations("link:.", "@dylanebert/shallot-grid")).toEqual([]);
    },
);

check(
    "surface: full Git Shallot identity passes",
    { claim: "the surface gate permits a lock-recorded full Git commit for Shallot" },
    () => {
        const spec = "github:dylanebert/shallot#0123456789abcdef0123456789abcdef01234567";
        expect(dependencyViolations(spec, "consumer", `spec: ${spec}\\n`)).toEqual([]);
    },
);

check(
    "surface: moving Shallot identities refuse",
    { claim: "the surface gate refuses moving and short Git identities for Shallot" },
    () => {
        for (const spec of [
            "github:dylanebert/shallot#main",
            "github:dylanebert/shallot#0123456",
            "git+https://github.com/dylanebert/shallot.git",
        ]) {
            expect(dependencyViolations(spec).join("\\n")).toContain("full 40-hex Git commit");
        }
    },
);

check(
    "surface: mutable Shallot identities refuse",
    { claim: "the surface gate refuses saved local paths and mutable dist-tags for Shallot" },
    () => {
        expect(dependencyViolations("link:../shallot").join("\\n")).toContain("link");
        expect(dependencyViolations("file:../shallot").join("\\n")).toContain("file");
        for (const tag of ["latest", "next", "beta", "candidate", "custom-release"])
            expect(dependencyViolations(tag).join("\\n")).toContain("mutable dist-tag");
        expect(dependencyViolations("^0.10.0")).toEqual([]);
    },
);

check(
    "surface: artifact identities require evidence",
    { claim: "the surface gate requires lock integrity for remote Shallot tarballs" },
    () => {
        const url = "https://example.test/shallot-0.10.0.tgz";
        expect(dependencyViolations(url).join("\\n")).toContain("lock integrity");
        expect(
            dependencyViolations(url, "consumer", `${url}\\nsha512-abc123\\n`).join("\\n"),
        ).toEqual("");
    },
);

check(
    "surface: checked-in artifact identity requires provenance",
    {
        claim: "the surface gate accepts only a checked-in Shallot tarball with digest and source provenance",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-artifact-"));
        const vendor = join(tree, "vendor");
        mkdirSync(vendor);
        const tarball = join(vendor, "shallot.tgz");
        writeFileSync(
            join(tree, "package.json"),
            JSON.stringify({
                name: "consumer",
                dependencies: { "@dylanebert/shallot": "file:vendor/shallot.tgz" },
            }),
        );
        writeFileSync(tarball, "artifact");
        writeFileSync(`${tarball}.sha256`, `${"a".repeat(64)}  shallot.tgz\n`);
        try {
            expect(readSurface(tree).join("\n")).toContain(
                "checked-in Shallot tarball needs a full source-commit sidecar",
            );
            writeFileSync(`${tarball}.source-commit`, `${"b".repeat(40)}\n`);
            expect(readSurface(tree)).toEqual([]);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

function git(root: string, ...args: string[]): string {
    const proc = Bun.spawnSync(["git", ...args], { cwd: root, stdout: "pipe", stderr: "pipe" });
    expect(proc.exitCode).toBe(0);
    return proc.stdout.toString().trim();
}

function subjectTree(): {
    root: string;
    base: string;
    comment: string;
    audio: string;
    added: string;
} {
    const root = mkdtempSync(join(tmpdir(), "shallot-surface-refs-"));
    mkdirSync(join(root, "src"), { recursive: true });
    mkdirSync(join(root, "crates/audio/pkg"), { recursive: true });
    writeFileSync(join(root, "src/setup.ts"), "export const ready = true;\n");
    writeFileSync(join(root, "crates/audio/pkg/shallot_audio.js"), "audio\n");
    git(root, "init", "-q");
    git(root, "config", "user.email", "surface@example.test");
    git(root, "config", "user.name", "surface");
    git(root, "add", ".");
    git(root, "commit", "-qm", "base");
    const base = git(root, "rev-parse", "HEAD");
    writeFileSync(join(root, "src/setup.ts"), "export const ready = true; // comment only\n");
    git(root, "commit", "-qam", "comment");
    const comment = git(root, "rev-parse", "HEAD");
    writeFileSync(join(root, "crates/audio/pkg/shallot_audio.js"), "audio changed\n");
    git(root, "commit", "-qam", "audio");
    const audio = git(root, "rev-parse", "HEAD");
    writeFileSync(join(root, "src/new.ts"), "export const added = true;\n");
    git(root, "add", ".");
    git(root, "commit", "-qm", "add");
    const added = git(root, "rev-parse", "HEAD");
    return { root, base, comment, audio, added };
}

check(
    "selectors filter integration rows and refuse empty matches",
    {
        claim: "surface selectors select only matching integration rows and refuse an empty match",
        size: "integration",
    },
    () => {
        const tree = seed("selectors");
        try {
            const all = run("surface.ts", tree, "--integration", "--all");
            expect(all.code).toBe(0);
            expect(all.out).toContain("browser selector row");
            expect(all.out).toContain("display selector row");
            expect(all.out).not.toContain("unit selector exclusion");
            expect(all.out).not.toContain("oracle selector exclusion");

            const requires = run("surface.ts", tree, "--integration", "--requires", "chromium");
            expect(requires.code).toBe(0);
            expect(requires.out).toContain("browser selector row");
            expect(requires.out).not.toContain("display selector row");

            const subject = run("surface.ts", tree, "--integration", "--subject", "src/browser");
            expect(subject.code).toBe(0);
            expect(subject.out).toContain("browser selector row");
            expect(subject.out).not.toContain("display selector row");

            const composed = run(
                "surface.ts",
                tree,
                "--integration",
                "--requires",
                "chromium",
                "--subject",
                "src/browser",
            );
            expect(composed.code).toBe(0);
            expect(composed.out).toContain("browser selector row");
            expect(composed.out).not.toContain("display selector row");

            const conflict = run(
                "surface.ts",
                tree,
                "--integration",
                "--all",
                "--base",
                "base",
                "--diff",
                "diff",
            );
            expect(conflict.code).toBe(1);
            expect(conflict.err).toContain("selectors cannot be combined");

            for (const args of [["--all"], ["--requires", "gpu"], ["--subject", "missing"]]) {
                const emptyTree = seed("unit-only");
                try {
                    const empty = run("surface.ts", emptyTree, "--integration", ...args);
                    expect(empty.code).toBe(1);
                    expect(empty.err).toContain("selector matched no integration rows");
                } finally {
                    rmSync(emptyTree, { recursive: true, force: true });
                }
            }
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "--list prints exactly the declared population",
    {
        claim: "surface.ts --list prints one row per declared check in the tree, from files and manifests, and nothing else",
        size: "integration",
    },
    () => {
        const tree = seed("clean");
        try {
            const { code, out } = run("surface.ts", tree);
            expect(code).toBe(0);
            expect(out.split("\n")).toEqual([
                "claim             size         requires  subject  budget   file",
                "alpha holds       unit         -         -        250ms    src/alpha.test.ts",
                "alpha refuses     unit         -         -        250ms    src/alpha.test.ts",
                "beta builds       integration  -         -        20000ms  scripts/beta.test.ts",
                "demo recipe runs  integration  chromium  -        20000ms  examples/demo/check.test.ts",
                "4 checks (parsed 4; 0 quarantined)",
            ]);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
        const defaults = seed("defaults");
        try {
            const { code, out } = run("surface.ts", defaults);
            expect(code).toBe(0);
            expect(out).toContain("browser defaults");
            expect(out).toContain("integration");
            expect(out).toContain("chromium");
            expect(out).toContain("20000ms");
        } finally {
            rmSync(defaults, { recursive: true, force: true });
        }

        const malformed = mkdtempSync(join(tmpdir(), "shallot-surface-root-array-"));
        try {
            writeFileSync(join(malformed, "shallot.json"), '{"check":[]}');
            expect(readSurface(malformed).join("\\n")).toContain("check array must not be empty");
        } finally {
            rmSync(malformed, { recursive: true, force: true });
        }

        const empty = mkdtempSync(join(tmpdir(), "shallot-surface-empty-"));
        try {
            mkdirSync(join(empty, ".github/workflows"), { recursive: true });
            writeFileSync(join(empty, "package.json"), "{}");
            writeFileSync(join(empty, ".github/workflows/test-surface.yml"), "name: no-op\\n");
            expect(readSurface(empty).join("\\n")).toContain(
                "empty population must not have a generated workflow",
            );
            rmSync(join(empty, ".github/workflows/test-surface.yml"));
            expect(readSurface(empty)).toEqual([]);
        } finally {
            rmSync(empty, { recursive: true, force: true });
        }

        const portable = mkdtempSync(join(tmpdir(), "shallot-surface-portable-"));
        try {
            mkdirSync(join(portable, "src"), { recursive: true });
            writeFileSync(join(portable, "package.json"), "{}");
            writeFileSync(
                join(portable, "src/claim.test.ts"),
                'import { check } from "@dylanebert/shallot/harness/check";\ncheck("claim", { claim: "portable claim" }, () => {});\n',
            );
            writeWorkflow(portable);
            expect(readSurface(portable)).toEqual([]);
            writeFileSync(join(portable, ".github/workflows/test-surface.yml"), "drift\n");
            expect(readSurface(portable).join("\n")).toContain("generated workflow drift");
        } finally {
            rmSync(portable, { recursive: true, force: true });
        }

        expect(subjectTokens("const value = 1; // prose")).toEqual(
            subjectTokens("const value = 1; /* prose */"),
        );
        expect(subjectTokens("const value = 1;")).not.toEqual(subjectTokens("const value = 2;"));
    },
);

check(
    "root-law selection validates event ref trees and ignores comments, unrelated audio, and oracles",
    {
        claim: "surface selection compares complete valid pre/post subject token streams and excludes named oracle files from ordinary integration rows",
        size: "integration",
    },
    () => {
        const tree = subjectTree();
        const setup = {
            name: "setup",
            claim: "setup changes select",
            size: "integration" as const,
            requires: [],
            budget: 20000,
            file: "src/setup.test.ts",
            subjects: ["src/setup.ts"],
        };
        const oracle = {
            ...setup,
            claim: "oracle changes select",
            file: "tests/browser.oracle.ts",
            subjects: [],
        };
        const added = {
            ...setup,
            claim: "added path selects",
            file: "src/add.test.ts",
            subjects: ["src/new.ts"],
        };
        const unchangedDirectory = {
            ...setup,
            claim: "unchanged directory does not select",
            file: "src/audio.test.ts",
            subjects: ["crates/audio"],
        };
        const nestedEdit = {
            ...setup,
            claim: "nested file content edit selects",
            file: "src/audio-edit.test.ts",
            subjects: ["crates/audio"],
        };
        const population = {
            root: tree.root,
            rows: [setup, oracle, added, unchangedDirectory, nestedEdit],
            undeclared: [],
            invalid: [],
            files: [],
        };
        try {
            expect(selectIntegrationRows(population, tree.base, tree.comment)).toEqual([]);
            expect(
                selectIntegrationRows(
                    { ...population, rows: [unchangedDirectory] },
                    tree.base,
                    tree.comment,
                ),
            ).toEqual([]);
            expect(
                selectIntegrationRows(
                    { ...population, rows: [nestedEdit] },
                    tree.base,
                    tree.audio,
                ).map((row) => row.claim),
            ).toEqual(["nested file content edit selects"]);
            expect(
                selectIntegrationRows(population, tree.audio, tree.added).map((row) => row.claim),
            ).toEqual(["added path selects"]);
        } finally {
            rmSync(tree.root, { recursive: true, force: true });
        }
    },
);

check(
    "root manifests are complete authorities for static and launched populations",
    {
        claim: "root shallot.json check arrays admit exactly every visible entrypoint and keep named oracles out of ordinary launches",
        size: "integration",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-root-authority-"));
        mkdirSync(join(tree, "src"), { recursive: true });
        mkdirSync(join(tree, "tests"), { recursive: true });
        const checkModule = resolve(ROOT, "src/harness/check");
        writeFileSync(
            join(tree, "src/kept.test.ts"),
            `import { check } from ${JSON.stringify(checkModule)};\ncheck("kept", { claim: "kept" }, () => {});\n`,
        );
        writeFileSync(
            join(tree, "tests/named.oracle.ts"),
            `import { check } from ${JSON.stringify(checkModule)};\ncheck("named", { claim: "named" }, () => {});\n`,
        );
        writeFileSync(
            join(tree, "shallot.json"),
            JSON.stringify(
                {
                    check: [{ file: "src/kept.test.ts" }, { file: "tests/named.oracle.ts" }],
                },
                null,
                2,
            ),
        );
        try {
            let population = collectPopulation(tree);
            expect(population.invalid).toEqual([]);
            expect(population.undeclared).toEqual([]);
            expect(population.files).toEqual(["src/kept.test.ts", "tests/named.oracle.ts"]);
            expect(discoverTestFiles(tree)).toEqual(["src/kept.test.ts"]);
            expect(discoverTestFiles(tree, true)).toEqual([
                "src/kept.test.ts",
                "tests/named.oracle.ts",
            ]);

            writeFileSync(
                join(tree, "shallot.json"),
                JSON.stringify({ check: [{ file: "src/kept.test.ts" }] }),
            );
            population = collectPopulation(tree);
            expect(population.invalid).toContain(
                "unlisted check file: tests/named.oracle.ts; root shallot.json check is authoritative",
            );
            expect(population.files).toEqual(["src/kept.test.ts"]);

            writeFileSync(
                join(tree, "shallot.json"),
                JSON.stringify({
                    check: [
                        { file: "src/kept.test.ts" },
                        { file: "src/kept.test.ts" },
                        { file: "src/moved.test.ts" },
                    ],
                }),
            );
            population = collectPopulation(tree);
            expect(population.invalid).toEqual([
                "duplicate manifest entry: src/kept.test.ts",
                "manifest entry does not exist: src/moved.test.ts",
                "unlisted check file: tests/named.oracle.ts; root shallot.json check is authoritative",
            ]);

            writeFileSync(
                join(tree, "src/unlisted.test.ts"),
                `import { check } from ${JSON.stringify(checkModule)};\ncheck("unlisted", { claim: "unlisted" }, () => { throw new Error("UNLISTED_RAN"); });\n`,
            );
            writeFileSync(
                join(tree, "shallot.json"),
                JSON.stringify({ check: [{ file: "src/kept.test.ts" }] }),
            );
            const runner = Bun.spawnSync(
                ["bun", resolve(ROOT, "scripts/test-runner.ts"), "--root", tree],
                { cwd: ROOT, stdout: "pipe", stderr: "pipe" },
            );
            expect(runner.exitCode).toBe(1);
            expect(runner.stderr.toString()).toContain(
                "unlisted check file: tests/named.oracle.ts",
            );
            expect(runner.stderr.toString()).toContain("unlisted check file: src/unlisted.test.ts");
            expect(runner.stderr.toString()).not.toContain("UNLISTED_RAN");
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "workflow refs and requirement setup are portable and conditional",
    {
        claim: "workflow rendering runs once per pull-request revision and landed-main commit, cancels superseded revisions, materializes full history, derives event-correct refs, and installs only declared ordinary Chromium requirements",
        size: "integration",
    },
    () => {
        const ordinary = {
            root: "/tmp/project",
            rows: [
                {
                    name: "browser",
                    claim: "browser runs",
                    size: "integration" as const,
                    requires: ["chromium"],
                    budget: 20000,
                    file: "src/browser.test.ts",
                    subjects: [],
                },
            ],
            undeclared: [],
            invalid: [],
            files: [],
        };
        const rendered = renderWorkflow(ordinary);
        expect(rendered).toContain("push:\n    branches:\n      - main");
        expect(rendered).toContain("pull_request:\n    branches:\n      - main");
        expect(rendered).toContain(
            "group: test-surface-${{ github.event.pull_request.number || github.sha }}",
        );
        expect(rendered).toContain(
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
        );
        expect(rendered).not.toContain("on: [push, pull_request]");
        expect(rendered).toContain("actions/checkout@v7");
        expect(rendered).toContain("oven-sh/setup-bun@v2");
        expect(rendered).toContain("fetch-depth: 0");
        expect(rendered).toContain("github.event.pull_request.base.sha");
        expect(rendered).toContain("github.event.before");
        expect(rendered).toContain("github.event.repository.default_branch");
        expect(rendered).toContain("git merge-base");
        expect(rendered).toContain("bunx playwright install --with-deps chromium");
        expect(rendered).not.toContain("github.event.pull_request.base.sha || github.event.before");
        expect(rendered).not.toMatch(/origin\/main|Rust|GPU|display|deploy/);
        expect(rendered).not.toContain("actions/cache");

        const seats = {
            ...ordinary,
            rows: ["gpu", "display", "deploy"].map((requirement, index) => ({
                ...ordinary.rows[0],
                name: requirement,
                claim: `${requirement} refuses`,
                requires: [requirement],
                file: `src/${requirement}.test.ts`,
                subjects: [`src/${requirement}.ts`],
                budget: 20000,
                index,
            })),
        };
        expect(renderWorkflow(seats)).not.toContain("playwright install");
        expect(renderWorkflow(seats)).not.toContain("rust-toolchain");
        const cargo = {
            ...ordinary,
            rows: [{ ...ordinary.rows[0], claim: "cargo runs", requires: ["cargo"] }],
        };
        expect(renderWorkflow(cargo)).toContain("dtolnay/rust-toolchain@stable");
        expect(renderWorkflow(cargo)).toContain("actions/cache@v6");
        expect(renderWorkflow(cargo)).toContain("path: target");
        expect(renderWorkflow(cargo)).not.toContain("setup-node");
        const node = {
            ...ordinary,
            rows: [{ ...ordinary.rows[0], claim: "node runs", requires: ["node"] }],
        };
        expect(renderWorkflow(node)).toContain("actions/setup-node@v6");
        expect(renderWorkflow(node)).toContain("node-version-file: .node-version");
        expect(renderWorkflow(seats)).not.toContain("setup-node");
        const oracleOnly = {
            ...ordinary,
            rows: [{ ...ordinary.rows[0], file: "tests/browser.oracle.ts" }],
        };
        expect(renderWorkflow(oracleOnly)).toBe("");
        expect(
            renderWorkflow({
                ...ordinary,
                rows: [{ ...ordinary.rows[0], file: "tests/browser.oracle.ts" }, ordinary.rows[0]],
            }),
        ).toContain("playwright install");
    },
);

check(
    "integration runner rejects non-commit refs before reading subjects",
    {
        claim: "test-runner refuses zero and absent integration refs instead of treating missing files as empty trees",
        size: "integration",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-runner-"));
        mkdirSync(join(tree, "src"), { recursive: true });
        writeFileSync(
            join(tree, "src/check.test.ts"),
            'check("row", { claim: "runner row", size: "integration", subject: "src/setup.ts" }, () => {});\n',
        );
        git(tree, "init", "-q");
        git(tree, "config", "user.email", "surface@example.test");
        git(tree, "config", "user.name", "surface");
        git(tree, "add", ".");
        git(tree, "commit", "-qm", "base");
        const head = git(tree, "rev-parse", "HEAD");
        try {
            for (const base of [
                "0000000000000000000000000000000000000000",
                "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            ]) {
                const proc = Bun.spawnSync(
                    [
                        "bun",
                        resolve(ROOT, "scripts/test-runner.ts"),
                        "--root",
                        tree,
                        "--integration",
                        "--base",
                        base,
                        "--diff",
                        head,
                    ],
                    { cwd: ROOT, stdout: "pipe", stderr: "pipe" },
                );
                expect(proc.exitCode).toBe(1);
                expect(proc.stderr.toString()).toContain("existing commit objects");
            }
            expect(collectPopulation(tree).rows).toHaveLength(1);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "a named oracle runs as an exclusive file selection",
    {
        claim: "test-runner prefixes an exact named oracle claim so Bun loads only its oracle file",
        size: "integration",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-oracle-runner-"));
        mkdirSync(join(tree, "tests"), { recursive: true });
        const checkModule = resolve(ROOT, "src/harness/check");
        writeFileSync(
            join(tree, "tests/named.oracle.ts"),
            `import { check } from ${JSON.stringify(checkModule)};\ncheck("named oracle", { claim: "named oracle runs", size: "integration" }, () => { console.log("NAMED_ORACLE_RAN"); });\n`,
        );
        writeFileSync(
            join(tree, "shallot.json"),
            JSON.stringify({ check: [{ file: "tests/named.oracle.ts" }] }),
        );
        try {
            const proc = Bun.spawnSync(
                [
                    "bun",
                    resolve(ROOT, "scripts/test-runner.ts"),
                    "--root",
                    tree,
                    "--oracle",
                    "named oracle runs",
                ],
                { cwd: ROOT, stdout: "pipe", stderr: "pipe" },
            );
            expect(proc.exitCode).toBe(0);
            expect(proc.stdout.toString()).toContain("NAMED_ORACLE_RAN");
            expect(proc.stdout.toString()).toContain('"result":"pass"');
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "the installed bin owns exclusive named-oracle execution and listing",
    {
        claim: "the installed shallot bin runs and lists exactly one selected oracle, refusing unknown and composed requests",
        size: "integration",
        subject: [
            "bin/shallot.ts",
            "scripts/test-runner.ts",
            "scripts/surface.ts",
            "src/harness/surface.ts",
        ],
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-installed-oracle-"));
        const checkModule = resolve(ROOT, "src/harness/check");
        const selectedMarker = join(tree, "selected-oracle-ran");
        const poisonOracleMarker = join(tree, "poison-oracle-ran");
        const poisonUnitMarker = join(tree, "poison-unit-ran");
        const claim = "fixture selected oracle runs";
        const invoke = (...args: string[]) =>
            Bun.spawnSync(["bun", resolve(ROOT, "bin/shallot.ts"), ...args], {
                cwd: tree,
                stdout: "pipe",
                stderr: "pipe",
            });
        mkdirSync(join(tree, "src"), { recursive: true });
        mkdirSync(join(tree, "tests"), { recursive: true });
        writeFileSync(
            join(tree, "src/poison.test.ts"),
            `import { writeFileSync } from "node:fs";\nimport { check } from ${JSON.stringify(checkModule)};\ncheck("poison unit", { claim: "poison unit must not run" }, () => { writeFileSync(${JSON.stringify(poisonUnitMarker)}, "ran"); throw new Error("POISON_UNIT_RAN"); });\n`,
        );
        writeFileSync(
            join(tree, "tests/selected.oracle.ts"),
            `import { writeFileSync } from "node:fs";\nimport { check } from ${JSON.stringify(checkModule)};\ncheck("selected oracle", { claim: ${JSON.stringify(claim)}, size: "integration" }, () => { writeFileSync(${JSON.stringify(selectedMarker)}, "ran"); });\n`,
        );
        writeFileSync(
            join(tree, "tests/poison.oracle.ts"),
            `import { writeFileSync } from "node:fs";\nimport { check } from ${JSON.stringify(checkModule)};\ncheck("poison oracle", { claim: "poison oracle must not run", size: "integration" }, () => { writeFileSync(${JSON.stringify(poisonOracleMarker)}, "ran"); throw new Error("POISON_ORACLE_RAN"); });\n`,
        );
        writeFileSync(
            join(tree, "shallot.json"),
            JSON.stringify({
                check: [
                    { file: "src/poison.test.ts" },
                    { file: "tests/selected.oracle.ts" },
                    { file: "tests/poison.oracle.ts" },
                ],
            }),
        );
        try {
            const listed = invoke("list", "--oracle", claim);
            const listedOutput = listed.stdout.toString() + listed.stderr.toString();
            expect(listed.exitCode).toBe(0);
            expect(listedOutput).toContain(claim);
            expect(listedOutput).not.toContain("poison oracle must not run");
            expect(listedOutput).not.toContain("poison unit must not run");
            expect(listedOutput).toContain("1 checks (parsed 3; 0 quarantined)");

            const unknown = invoke("list", "--oracle", "unknown fixture oracle");
            expect(unknown.exitCode).toBe(1);
            expect(unknown.stderr.toString()).toContain(
                "named oracle not found: unknown fixture oracle",
            );

            const composed = invoke("test", "--oracle", claim, "--all");
            expect(composed.exitCode).toBe(1);
            expect(composed.stderr.toString()).toContain(
                "--oracle cannot be combined with selectors or integration mode",
            );

            const executed = invoke("test", "--oracle", claim);
            expect(executed.exitCode).toBe(0);
            expect(existsSync(selectedMarker)).toBe(true);
            expect(existsSync(poisonOracleMarker)).toBe(false);
            expect(existsSync(poisonUnitMarker)).toBe(false);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "an undeclared check file reds the reader",
    {
        claim: "check-surface.ts reds on a test-suffix file that registers no check() declaration",
        size: "integration",
    },
    () => {
        const { code, err } = reader("undeclared");
        expect(code).toBe(1);
        expect(err).toContain(
            "undeclared check file: src/naked.test.ts registers no check() declaration",
        );
    },
);

check(
    "a duplicate claim reds the reader",
    {
        claim: "check-surface.ts reds when two checks declare the same claim, naming both files",
        size: "integration",
    },
    () => {
        const { code, err } = reader("duplicate");
        expect(code).toBe(1);
        expect(err).toContain('duplicate claim: "same claim" declared in');
        expect(err).toContain("src/one.test.ts");
        expect(err).toContain("src/two.test.ts");
    },
);

check(
    "an over-budget unit declaration reds the reader",
    {
        claim: "check-surface.ts reds on a unit declaration whose budget is above the 250 ms ceiling",
        size: "integration",
    },
    () => {
        const { code, err } = reader("over-budget");
        expect(code).toBe(1);
        expect(err).toContain(
            'invalid declaration: src/slow.test.ts check("slow unit") budget 251ms is above the unit ceiling of 250ms',
        );
    },
);

check(
    "an orphan quarantine row reds the reader",
    {
        claim: "check-surface.ts reds when quarantine.json names a claim no check declares, and treats an absent file as zero rows",
        size: "integration",
    },
    () => {
        const orphan = reader("orphan");
        expect(orphan.code).toBe(1);
        expect(orphan.err).toContain(
            'orphan quarantine row: claim "claim nobody declares" names no check in the population',
        );
        const absent = reader("clean");
        expect(absent.code).toBe(0);
        expect(absent.err).toBe("");
    },
);

check(
    "an unapproved or orphan sanction row reds the reader",
    {
        claim: "check-surface.ts reds a sanction row whose approval is empty and one whose site names an absent file, and leaves an approved row on a present file alone",
        size: "integration",
    },
    () => {
        const { code, err } = reader("sanctions");
        expect(code).toBe(1);
        const sanctionLines = err
            .split("\n")
            .filter((line) => line.includes("sanction row"))
            .map((line) => line.trim());
        expect(sanctionLines).toEqual([
            'unapproved sanction row: site "src/kept.test.ts:2" awaits the person\'s approval',
            'orphan sanction row: site "src/gone.ts:3" names no file in the tree',
        ]);
        const absent = reader("clean");
        expect(absent.code).toBe(0);
        expect(absent.err).toBe("");
    },
);

check(
    "a non-literal declaration reds the reader",
    {
        claim: "check-surface.ts reds a check whose options use a spread, identifier or computed value, naming its file",
        size: "integration",
    },
    () => {
        const nonLiteral = reader("non-literal");
        expect(nonLiteral.code).toBe(1);
        expect(nonLiteral.err).toContain(
            'non-literal declaration: src/spread.test.ts check("spread declaration")',
        );
        expect(nonLiteral.err).toContain("src/identifier.test.ts");
        expect(nonLiteral.err).toContain("src/computed.test.ts");
    },
);

check(
    "an expired quarantine row reds the reader",
    {
        claim: "check-surface.ts reds a quarantine row whose ISO expiry is in the past",
        size: "integration",
    },
    () => {
        const expired = reader("expired");
        expect(expired.code).toBe(1);
        expect(expired.err).toContain('expired quarantine row: "expired claim" expired 2020-01-01');
    },
);

check(
    "a recipe manifest must name a file that declares a check",
    {
        claim: "check-surface.ts reds when a recipe manifest names a file with no check declaration",
        size: "integration",
    },
    () => {
        const missing = reader("manifest-no-check");
        expect(missing.code).toBe(1);
        expect(missing.err).toContain("undeclared check file: examples/no-check/check.test.ts");
    },
);

check(
    "the reader passes the shipped tree",
    {
        claim: "check-surface.ts is green on the engine's own tree, so the population is never an empty scan",
        size: "integration",
    },
    () => {
        const proc = Bun.spawnSync(["bun", resolve(ROOT, "scripts/check-surface.ts")], {
            cwd: ROOT,
        });
        expect(proc.stderr.toString().trim()).toBe("");
        expect(proc.exitCode).toBe(0);
        expect(proc.stdout.toString()).toMatch(/^\d+ declared checks/);
        expect(Number(proc.stdout.toString().split(" ")[0])).toBeGreaterThan(0);
    },
);
