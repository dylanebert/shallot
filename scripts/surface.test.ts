import { expect } from "bun:test";
import {
    cpSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
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
} from "@dylanebert/shallot/harness/surface";
import { unfixture } from "./unfixture";

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
        const unitTargets = targets.filter((target: string) => !target.endsWith("_gold"));
        expect(unitTargets).toEqual(["stages"]);
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
        claim: "surface.ts --list prints one row per declared check in the convention-discovered tree",
        size: "integration",
    },
    () => {
        const tree = seed("clean");
        try {
            const { code, out } = run("surface.ts", tree);
            expect(code).toBe(0);
            expect(out.split("\n").slice(0, -1)).toEqual([
                "claim             size         requires  subject  budget   file",
                "alpha holds       unit         -         -        250ms    src/alpha.test.ts",
                "alpha refuses     unit         -         -        250ms    src/alpha.test.ts",
                "beta builds       integration  -         -        20000ms  scripts/beta.test.ts",
                "demo recipe runs  integration  chromium  -        20000ms  examples/demo/check.test.ts",
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
            expect(readSurface(malformed)).toEqual([]);
        } finally {
            rmSync(malformed, { recursive: true, force: true });
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
    "discovery owns the complete population and ordinary launches exclude oracles",
    {
        claim: "the carrier reads every convention-named file once, requires each file's check declaration, ignores manifest admission, and keeps named oracles outside ordinary launches",
        size: "integration",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-surface-discovery-authority-"));
        mkdirSync(join(tree, "src"), { recursive: true });
        mkdirSync(join(tree, "tests"), { recursive: true });
        const checkModule = resolve(ROOT, "src/harness/check");
        writeFileSync(
            join(tree, "src/kept.test.ts"),
            `import { check } from ${JSON.stringify(checkModule)};\ncheck("kept", { claim: "kept" }, () => {});\n`,
        );
        writeFileSync(
            join(tree, "src/unlisted.test.ts"),
            `import { check } from ${JSON.stringify(checkModule)};\ncheck("unlisted", { claim: "unlisted" }, () => {});\n`,
        );
        writeFileSync(
            join(tree, "tests/named.oracle.ts"),
            `import { check } from ${JSON.stringify(checkModule)};\ncheck("named", { claim: "named" }, () => {});\n`,
        );
        writeFileSync(
            join(tree, "shallot.json"),
            JSON.stringify({ check: [{ file: "src/kept.test.ts" }, { file: "missing.test.ts" }] }),
        );
        try {
            const population = collectPopulation(tree);
            expect(population.invalid).toEqual([]);
            expect(population.undeclared).toEqual([]);
            expect(population.files).toEqual([
                "src/kept.test.ts",
                "src/unlisted.test.ts",
                "tests/named.oracle.ts",
            ]);
            expect(population.rows.map((row) => row.file)).toEqual([
                "src/kept.test.ts",
                "tests/named.oracle.ts",
                "src/unlisted.test.ts",
            ]);
            expect(discoverTestFiles(tree)).toEqual(["src/kept.test.ts", "src/unlisted.test.ts"]);
            expect(discoverTestFiles(tree, true)).toEqual(population.files);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);

check(
    "workflow requirement setup is conditional",
    {
        claim: "the rendered workflow installs setup only for declared ordinary requirements and renders nothing for an oracle-only population",
        size: "integration",
    },
    () => {
        const row = {
            name: "row",
            claim: "row runs",
            size: "integration" as const,
            budget: 20000,
            file: "src/row.test.ts",
            subjects: [],
        };
        const render = (requires: string[], file = row.file) =>
            renderWorkflow({
                root: "/tmp/project",
                rows: [{ ...row, requires, file }],
                undeclared: [],
                invalid: [],
                files: [],
            });
        expect(render(["chromium"])).toContain("playwright install --with-deps chromium");
        expect(render(["cargo"])).toContain("dtolnay/rust-toolchain@stable");
        expect(render(["node"])).toContain("actions/setup-node@v6");
        expect(render(["gpu"])).not.toMatch(/playwright install|rust-toolchain|setup-node/);
        expect(render(["chromium"], "tests/browser.oracle.ts")).toBe("");
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
            expect(listedOutput).toContain(
                "1 checks (parsed 3; 0 quarantined; 0 sanctioned, 0 unapproved; 0 red-circled, 0 unapproved)",
            );

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
        claim: "check-surface.ts reds when quarantine.json names a claim no check declares",
        size: "integration",
    },
    () => {
        const orphan = reader("orphan");
        expect(orphan.code).toBe(1);
        expect(orphan.err).toContain(
            'orphan quarantine row: claim "claim nobody declares" names no check in the population',
        );
    },
);

check(
    "unapproved, orphan and doubly declared sites red the reader",
    {
        claim: "check-surface.ts reds an unapproved row, a row whose site names an absent file, and a site declared as both a sanction and a red circle, in each of sanctions.json and red-circles.json, and leaves approved present rows alone",
        size: "integration",
    },
    () => {
        const { code, err } = reader("sanctions");
        expect(code).toBe(1);
        const lines = err
            .split("\n")
            .filter((line) => /sanction row|red-circle row|declared twice over/.test(line))
            .map((line) => line.trim());
        // Exact membership, so a rule that fires on the approved present rows, or one that stops firing,
        // is as visible as one that never fired at all.
        expect(lines.sort()).toEqual(
            [
                'unapproved sanction row: site "src/kept.test.ts:2" awaits the person\'s approval',
                'orphan sanction row: site "src/gone.ts:3" names no file in the tree',
                'unapproved red-circle row: site "src/kept.test.ts:6" awaits the person\'s approval',
                'orphan red-circle row: site "src/vanished.ts:7" names no file in the tree',
                'site declared twice over: site "src/kept.test.ts:4" is declared as a sanction and as a red-circle; a site classes once',
            ].sort(),
        );
        // An absent declaration file is zero rows and no violation: the `clean` tree carries none of
        // quarantine.json, sanctions.json or red-circles.json.
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
    "a convention-named file must declare a check",
    {
        claim: "check-surface.ts reds every convention-named file that registers no check declaration, regardless of its manifest",
        size: "integration",
    },
    () => {
        const missing = reader("manifest-no-check");
        expect(missing.code).toBe(1);
        expect(missing.err).toContain(
            "undeclared check file: examples/no-check/check.test.ts registers no check() declaration",
        );
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
