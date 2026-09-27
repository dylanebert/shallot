import { expect } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { check } from "@dylanebert/shallot/harness/check";
import { collectPopulation } from "@dylanebert/shallot/harness/surface";
import { readProjectPolicy } from "./policy";

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
        return readProjectPolicy(tree);
    } finally {
        rmSync(tree, { recursive: true, force: true });
    }
}

check(
    "project policy stays outside check discovery",
    {
        claim: "check discovery parses declarations without applying recipe source policy, while project tooling still refuses private engine imports and physics-world escapes",
    },
    () => {
        const tree = mkdtempSync(join(tmpdir(), "shallot-project-policy-recipe-"));
        const recipe = join(tree, "examples/demo");
        mkdirSync(join(recipe, "src"), { recursive: true });
        writeFileSync(join(recipe, "shallot.json"), JSON.stringify({ kind: "recipe" }));
        writeFileSync(
            join(recipe, "src/game.ts"),
            'import { thing } from "@dylanebert/shallot/src/engine";\nPhysics.world();\n',
        );
        try {
            const population = collectPopulation(tree);
            expect(population.invalid).toEqual([]);
            expect(readProjectPolicy(tree).join("\\n")).toContain(
                "recipe source uses a deep engine import: examples/demo/src/game.ts",
            );
            expect(readProjectPolicy(tree).join("\\n")).toContain(
                "recipe source uses Physics.world/physicsWorld: examples/demo/src/game.ts",
            );
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
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
            expect(readProjectPolicy(tree).join("\n")).toContain(
                "checked-in Shallot tarball needs a full source-commit sidecar",
            );
            writeFileSync(`${tarball}.source-commit`, `${"b".repeat(40)}\n`);
            expect(readProjectPolicy(tree)).toEqual([]);
        } finally {
            rmSync(tree, { recursive: true, force: true });
        }
    },
);
