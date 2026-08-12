import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
// A meta-test over repo-root tooling, not the engine — same shape as cli-coverage.test.ts's
// registry walk, placed here so it rides the default `bun test` sweep instead of running under
// nobody's hand. `scripts/check-scripts.ts` stays at `scripts/`, beside its five siblings; only
// the test moves.
import {
    checkDocs,
    checkExists,
    checkReachable,
    run,
    workspacePkgPaths,
} from "../../../scripts/check-scripts";

// Fixture trees live under the OS tmpdir, never the repo — `--root`-style isolation so a
// mutation this suite needs (a phantom target, an undocumented script, an orphan file) never
// touches a tracked file. Each test gets its own dir; cleaned up after.

const roots: string[] = [];

function fixture(files: Record<string, string>): string {
    const dir = mkdtempSync(join(tmpdir(), "check-scripts-"));
    roots.push(dir);
    for (const [rel, content] of Object.entries(files)) {
        const full = join(dir, rel);
        mkdirSync(join(full, ".."), { recursive: true });
        writeFileSync(full, content);
    }
    return dir;
}

afterEach(() => {
    while (roots.length > 0) {
        const dir = roots.pop();
        if (dir) rmSync(dir, { recursive: true, force: true });
    }
});

describe("checkExists — direction 1", () => {
    test("flags a script whose target file is gone", async () => {
        const root = fixture({
            "package.json": JSON.stringify({
                workspaces: [],
                scripts: { phantom: "bun run scripts/ghost.ts" },
            }),
        });
        const violations = await checkExists([join(root, "package.json")]);
        expect(violations).toHaveLength(1);
        expect(violations[0].script).toBe("phantom");
        expect(violations[0].detail).toContain("does not exist");
    });

    test("passes a script whose target file exists", async () => {
        const root = fixture({
            "package.json": JSON.stringify({
                workspaces: [],
                scripts: { good: "bun run scripts/good.ts" },
            }),
            "scripts/good.ts": "console.log('good');",
        });
        const violations = await checkExists([join(root, "package.json")]);
        expect(violations).toHaveLength(0);
    });

    test("regression (finding 3): `bunx <npm-binary>` is a published binary, not a delegate", async () => {
        // Reviewer repro: "lint2": "bunx eslint ." false-red — no script named "eslint" exists,
        // and none should be expected to; bunx never resolves to a repo-local script.
        const root = fixture({
            "package.json": JSON.stringify({
                workspaces: [],
                scripts: { lint2: "bunx eslint ." },
            }),
        });
        const violations = await checkExists([join(root, "package.json")]);
        expect(violations).toHaveLength(0);
    });

    test("`bun run --cwd <dir> <name>` still requires the delegate script to exist", async () => {
        const root = fixture({
            "package.json": JSON.stringify({
                workspaces: [],
                scripts: { build: "bun run --cwd packages/x nope" },
            }),
            "packages/x/package.json": JSON.stringify({ scripts: { build: "bun run x.ts" } }),
        });
        const violations = await checkExists([join(root, "package.json")]);
        expect(violations).toHaveLength(1);
        expect(violations[0].detail).toContain('delegates to script "nope"');
    });
});

describe("checkDocs — direction 2", () => {
    test("flags a root script not cited anywhere", async () => {
        const root = fixture({ "AGENTS.md": "bun run good\n" });
        const violations = await checkDocs(root, { good: "x", undocumented: "y" });
        expect(violations).toHaveLength(1);
        expect(violations[0].script).toBe("undocumented");
    });

    test("passes a script cited as `bun run <name>`", async () => {
        const root = fixture({ "AGENTS.md": "bun run good\n" });
        const violations = await checkDocs(root, { good: "x" });
        expect(violations).toHaveLength(0);
    });

    test("regression (finding 2): a colon-namespaced citation doesn't also document its prefix", async () => {
        // Reviewer repro: citing only "foo:bar" must not satisfy a separate "foo" script — `\b`
        // treats `:` as a word boundary, so the old regex read "bun foo:bar" as citing "foo".
        const root = fixture({ "AGENTS.md": "bun run foo:bar\n" });
        const violations = await checkDocs(root, { foo: "x", "foo:bar": "y" });
        expect(violations.map((v) => v.script)).toEqual(["foo"]);
    });

    test("a namespaced script is satisfied by its own full-name citation", async () => {
        const root = fixture({ "AGENTS.md": "bun run test:install\n" });
        const violations = await checkDocs(root, { "test:install": "x" });
        expect(violations).toHaveLength(0);
    });
});

describe("checkReachable — direction 3", () => {
    test("flags a scripts/* file nothing reaches", async () => {
        const root = fixture({
            "package.json": JSON.stringify({ scripts: { good: "bun run scripts/good.ts" } }),
            "scripts/good.ts": "1;",
            "scripts/orphan.ts": "1;",
        });
        const violations = await checkReachable(root, { good: "bun run scripts/good.ts" });
        expect(violations.map((v) => v.script)).toEqual(["orphan.ts"]);
    });

    test("a script target is trivially reachable", async () => {
        const root = fixture({
            "package.json": JSON.stringify({ scripts: { good: "bun run scripts/good.ts" } }),
            "scripts/good.ts": "1;",
        });
        const violations = await checkReachable(root, { good: "bun run scripts/good.ts" });
        expect(violations).toHaveLength(0);
    });

    test("a by-path doc citation makes an unscripted file reachable", async () => {
        const root = fixture({
            "package.json": JSON.stringify({ scripts: {} }),
            "scripts/instrument.ts": "1;",
            "AGENTS.md": "Run `scripts/instrument.ts` by hand after a perf change.\n",
        });
        const violations = await checkReachable(root, {});
        expect(violations).toHaveLength(0);
    });

    test("regression (finding 1): a bare-stem mention is not a path citation", async () => {
        // Reviewer repro: `See "orphan-test-file" for context` — a quoted mention of the stem
        // with no path separator anywhere near it — must not count as reaching the file.
        const root = fixture({
            "package.json": JSON.stringify({ scripts: { good: "bun run scripts/good.ts" } }),
            "scripts/good.ts": "1;",
            "scripts/orphan-test-file.ts": "1;",
            "AGENTS.md":
                'bun run good\nSee "orphan-test-file" for context (not a real citation, just a mention)\n',
        });
        const violations = await checkReachable(root, { good: "bun run scripts/good.ts" });
        expect(violations.map((v) => v.script)).toEqual(["orphan-test-file.ts"]);
    });

    test("a relative import specifier (`./stem`, no extension) counts as a path citation", async () => {
        const root = fixture({
            "package.json": JSON.stringify({ scripts: { good: "bun run scripts/good.ts" } }),
            "scripts/good.ts": 'import { helper } from "./helper";\nhelper();',
            "scripts/helper.ts": "export function helper() {}",
        });
        const violations = await checkReachable(root, { good: "bun run scripts/good.ts" });
        expect(violations).toHaveLength(0);
    });

    test("a `.test.ts` file is exempt — bun test's own glob is its reachability mechanism", async () => {
        const root = fixture({
            "package.json": JSON.stringify({ scripts: { good: "bun run scripts/good.ts" } }),
            "scripts/good.ts": "1;",
            "scripts/good.test.ts": "1;",
        });
        const violations = await checkReachable(root, { good: "bun run scripts/good.ts" });
        expect(violations).toHaveLength(0);
    });
});

describe("workspacePkgPaths", () => {
    test("expands a `<base>/*` glob and a literal workspace entry", async () => {
        const root = fixture({
            "packages/a/package.json": "{}",
            "packages/b/package.json": "{}",
            "examples/gym/package.json": "{}",
        });
        const paths = await workspacePkgPaths(root, ["packages/*", "examples/gym"]);
        expect(paths.sort()).toEqual(
            [
                join(root, "packages/a/package.json"),
                join(root, "packages/b/package.json"),
                join(root, "examples/gym/package.json"),
            ].sort(),
        );
    });
});

describe("run — end to end", () => {
    test("a clean fixture reports zero violations across all three directions", async () => {
        const root = fixture({
            "package.json": JSON.stringify({
                workspaces: [],
                scripts: { good: "bun run scripts/good.ts" },
            }),
            "scripts/good.ts": "1;",
            "AGENTS.md": "bun run good\n",
        });
        const { existsViolations, docViolations, reachViolations } = await run(root);
        expect(existsViolations).toHaveLength(0);
        expect(docViolations).toHaveLength(0);
        expect(reachViolations).toHaveLength(0);
    });
});
