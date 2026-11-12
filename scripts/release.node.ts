import { expect, setDefaultTimeout, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { removeTestDirectory } from "./test-directory";

setDefaultTimeout(TIMEOUT.node);
type Step = { name?: string; if: string; run: string; env: Record<string, string>; uses?: string };
const workflow = Bun.YAML.parse(
    readFileSync(resolve(import.meta.dir, "../.github/workflows/release.yml"), "utf8"),
) as {
    permissions: Record<string, string>;
    jobs: Record<string, { permissions?: Record<string, string>; steps: Step[] }>;
};
const step = workflow.jobs.checksums.steps.find((s) => s.name === "Attach archives to release")!;

for (const event of ["push", "workflow_dispatch"]) {
    for (const ref of ["refs/tags/v0.10.4", "refs/heads/main"]) {
        test(`${event} ${ref}: only pushed release tags publish`, () => {
            const expression = step.if
                .replaceAll("github.event_name", JSON.stringify(event))
                .replaceAll("github.ref", JSON.stringify(ref));
            const admitted = new Function("startsWith", `return (${expression})`)(
                (value: string, prefix: string) => value.startsWith(prefix),
            );
            expect(admitted).toBe(event === "push" && ref.startsWith("refs/tags/v"));
        });
    }
}

function run(tag: string, copy: string, exists = false) {
    const root = mkdtempSync(join(tmpdir(), "shallot-release-"));
    try {
        mkdirSync(join(root, "bin"));
        mkdirSync(join(root, "archives"));
        writeFileSync(join(root, "archives/shallot-native-fixture.tar.gz"), "archive");
        writeFileSync(join(root, "SHA256SUMS"), "checksum");
        writeFileSync(join(root, "changelog"), copy);
        writeFileSync(join(root, "calls"), "");
        writeFileSync(
            join(root, "bin/gh"),
            `#!/bin/bash
set -eu
printf '%s\\n' "$*" >> calls
case "$1 $2" in
 'release view') exit ${exists ? 0 : 1} ;;
 'api repos/fixture/release/contents/CHANGELOG.md')
   [[ "$*" == *"ref=${tag}"* ]] || exit 90
   [[ "$*" == *"application/vnd.github.raw+json"* ]] || exit 91
   /bin/cat changelog ;;
 'release create')
   while [ "$#" -gt 0 ]; do
     if [ "$1" = --notes-file ]; then /bin/cp "$2" body; break; fi
     shift
   done ;;
 'release upload') ;;
 *) exit 92 ;;
esac
`,
        );
        chmodSync(join(root, "bin/gh"), 0o755);
        const source = step.run.replaceAll(`\${{ github.ref_name }}`, tag);
        const result = Bun.spawnSync(["/bin/bash", "-e", "-c", source], {
            cwd: root,
            env: {
                PATH: `${root}/bin:/usr/bin:/bin`,
                HOME: root,
                GH_CONFIG_DIR: root,
                XDG_CONFIG_HOME: root,
                GH_TOKEN: "fixture-only",
                GH_REPO: "fixture/release",
                RELEASE_TAG: tag,
            },
            stdout: "pipe",
            stderr: "pipe",
        });
        return {
            code: result.exitCode,
            calls: readFileSync(join(root, "calls"), "utf8"),
            body: Bun.file(join(root, "body")).size ? readFileSync(join(root, "body"), "utf8") : "",
        };
    } finally {
        removeTestDirectory(root);
    }
}

for (const tag of ["v0.10.4", "v0.10.4-next.1"]) {
    test(`${tag}: creation uses only matching tagged copy and preserves classification`, () => {
        const result = run(
            tag,
            `# Changelog\n\n## other\n\nwrong\n\n## ${tag.slice(1)}\n\n### highlights\n\n- selected\n\n## older\n\nwrong\n`,
        );
        expect(result.code).toBe(0);
        expect(result.body.trim()).toBe("### highlights\n\n- selected");
        expect(result.calls.includes("--prerelease")).toBe(tag.includes("-"));
        expect(result.calls).toContain("--verify-tag");
        expect(result.calls).not.toContain("--generate-notes");
    });
}
for (const copy of ["## 0.10.3\nwrong\n", "## 0.10.4\n \n\t\n## older\nwrong\n"]) {
    test(`missing or blank entry refuses writes: ${JSON.stringify(copy)}`, () => {
        const result = run("v0.10.4", copy);
        expect(result.code).not.toBe(0);
        expect(result.calls).not.toMatch(/release (create|upload|edit)/);
    });
}
test("existing release uploads only assets without reading or replacing notes", () => {
    const result = run("v0.10.4", "", true);
    expect(result.code).toBe(0);
    expect(result.calls.trim().split("\n")).toEqual([
        "release view v0.10.4",
        "release upload v0.10.4 --clobber archives/shallot-native-fixture.tar.gz SHA256SUMS",
    ]);
});
test("release permissions remain scoped and checksums needs no checkout", () => {
    expect(workflow.permissions).toEqual({ contents: "read" });
    expect(workflow.jobs.checksums.permissions).toEqual({ contents: "write" });
    expect(workflow.jobs.build.permissions).toBeUndefined();
    expect(
        workflow.jobs.checksums.steps.some((s: { uses?: string }) =>
            s.uses?.startsWith("actions/checkout"),
        ),
    ).toBe(false);
    expect(step.env.GH_REPO).toBe(`\${{ github.repository }}`);
});
