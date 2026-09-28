import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readProjectPolicy } from "./policy";

function dependencyViolations(
    spec: string,
    packageName = "consumer",
    lock?: string,
    dependencyName = "@dylanebert/shallot-grid",
): string[] {
    const tree = mkdtempSync(join(tmpdir(), "shallot-project-policy-dependency-"));
    try {
        writeFileSync(
            join(tree, "package.json"),
            JSON.stringify({
                name: packageName,
                dependencies: { [dependencyName]: spec },
            }),
        );
        if (lock !== undefined) writeFileSync(join(tree, "bun.lock"), lock);
        return readProjectPolicy(tree);
    } finally {
        rmSync(tree, { recursive: true, force: true });
    }
}

test("project policy rejects private engine imports and physics-world escapes", () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-project-policy-recipe-"));
        const recipe = join(tree, "examples/demo");
        mkdirSync(join(recipe, "src"), { recursive: true });
        writeFileSync(join(recipe, "shallot.json"), JSON.stringify({ kind: "recipe" }));
        writeFileSync(
            join(recipe, "src/game.ts"),
            'import { thing } from "@dylanebert/shallot/src/engine";\nPhysics.world();\n',
        );
        try {
            const violations = readProjectPolicy(tree).join("\n");
            expect(violations).toContain(
                "recipe source uses a deep engine import: examples/demo/src/game.ts",
            );
            expect(violations).toContain(
                "recipe source uses Physics.world/physicsWorld: examples/demo/src/game.ts",
            );
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    });

test("project policy permits a lock-recorded full Git commit for Shallot", () => {
        const spec = "github:dylanebert/shallot#0123456789abcdef0123456789abcdef01234567";
        expect(dependencyViolations(spec, "consumer", `spec: ${spec}\\n`)).toEqual([]);
    });

test("project policy refuses moving and short Git identities for Shallot", () => {
        for (const spec of [
            "github:dylanebert/shallot#main",
            "github:dylanebert/shallot#0123456",
            "git+https://github.com/dylanebert/shallot.git",
        ]) {
            expect(dependencyViolations(spec).join("\\n")).toContain("full 40-hex Git commit");
        }
    });

test("project policy refuses saved local paths and mutable dist-tags for Shallot", () => {
        expect(dependencyViolations("link:../shallot").join("\\n")).toContain("link");
        expect(dependencyViolations("file:../shallot").join("\\n")).toContain("file");
        for (const tag of ["latest", "next", "beta", "candidate", "custom-release"])
            expect(dependencyViolations(tag).join("\\n")).toContain("mutable dist-tag");
        expect(dependencyViolations("^0.10.0")).toEqual([]);
    });

test("project policy requires lock integrity for remote Shallot tarballs", () => {
        const url = "https://example.test/shallot-0.10.0.tgz";
        expect(dependencyViolations(url).join("\\n")).toContain("lock integrity");
        expect(
            dependencyViolations(url, "consumer", `${url}\\nsha512-abc123\\n`).join("\\n"),
        ).toEqual("");
    });

test("project policy accepts only a checked-in Shallot tarball with digest and source provenance", () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-project-policy-artifact-"));
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
            expect(readProjectPolicy(tree).join("\n")).toContain(
                "checked-in Shallot tarball needs a full source-commit sidecar",
            );
            writeFileSync(`${tarball}.source-commit`, `${"b".repeat(40)}\n`);
            expect(readProjectPolicy(tree)).toEqual([]);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    });
