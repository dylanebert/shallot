import { expect, test } from "bun:test";
import {
    cpSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

const ROOT = resolve(import.meta.dir, "..");

function run(command: string[], cwd: string, label: string): string {
    const result = Bun.spawnSync(command, { cwd, stdout: "pipe", stderr: "pipe" });
    const output = `${result.stdout.toString()}${result.stderr.toString()}`;
    if (result.exitCode !== 0) throw new Error(`${label} failed:\n${output}`);
    return output;
}

test("the Bun preload aliases linked-engine TypeGPU imports to the consumer peer", async () => {
    const engineManifest = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
    const typegpuSource = join(ROOT, "node_modules/typegpu");
    const typegpuManifest = JSON.parse(readFileSync(join(typegpuSource, "package.json"), "utf8"));
    const scratch = mkdtempSync(join(tmpdir(), "shallot-package-shape-"));
    const nodeModules = join(scratch, "node_modules");
    const consumerTypegpu = join(nodeModules, "typegpu");
    const shallotLink = join(nodeModules, "@dylanebert/shallot");

    try {
        mkdirSync(join(nodeModules, "@dylanebert"), { recursive: true });
        mkdirSync(consumerTypegpu, { recursive: true });
        // Match bun link's node_modules symlink without touching the global link registry.
        symlinkSync(ROOT, shallotLink, "dir");
        cpSync(typegpuSource, consumerTypegpu, { recursive: true });
        for (const name of Object.keys(typegpuManifest.dependencies ?? {})) {
            const target = resolve(ROOT, "node_modules", ...name.split("/"));
            if (!existsSync(target)) throw new Error(`missing TypeGPU dependency ${name}`);
            const link = resolve(nodeModules, ...name.split("/"));
            mkdirSync(dirname(link), { recursive: true });
            if (!existsSync(link)) symlinkSync(target, link, "dir");
        }

        const fixturePackage = {
            name: "shallot-package-shape-link-fixture",
            private: true,
            type: "module",
            dependencies: { "@dylanebert/shallot": "^0.9.5" },
            devDependencies: { typegpu: engineManifest.peerDependencies.typegpu },
        };
        await Bun.write(
            join(scratch, "package.json"),
            `${JSON.stringify(fixturePackage, null, 2)}\n`,
        );
        await Bun.write(
            join(scratch, "bunfig.toml"),
            '[test]\npreload = ["@dylanebert/shallot/bun"]\n',
        );
        await Bun.write(
            join(scratch, "linked.test.ts"),
            `import { expect, test } from "bun:test";\nimport tgpu from "typegpu";\nimport { checkTgsl } from "@dylanebert/shallot/runtime";\n\ntest("consumer and linked engine share one TypeGPU instance", () => {\n    expect(typeof tgpu.fn).toBe("function");\n    expect(() => checkTgsl()).not.toThrow();\n});\n`,
        );

        const consumerPath = Bun.resolveSync("typegpu", scratch);
        const producerPath = Bun.resolveSync("typegpu", ROOT);
        console.log(`TypeGPU resolution: consumer=${consumerPath}`);
        console.log(`TypeGPU natural linked-engine resolution=${producerPath}`);
        expect(consumerPath).not.toBe(producerPath);

        const output = run(["bun", "test"], scratch, "testing linked TypeGPU identity");
        expect(output).toContain("consumer and linked engine share one TypeGPU instance");
        expect(output).toContain("1 pass");
        expect(output).not.toContain("Found duplicate TypeGPU version");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
});

test("release creation passes --prerelease only for hyphenated tags", () => {
    const workflow = readFileSync(join(ROOT, ".github/workflows/release.yml"), "utf8");
    const argsBlock = workflow.match(
        /release_args=\(--verify-tag --generate-notes\)\n\s+if \[\[ "\$RELEASE_TAG" == \*-\* \]\]; then\n\s+release_args\+=\(--prerelease\)\n\s+fi/,
    )?.[0];
    if (!argsBlock) throw new Error("release.yml has no recognized prerelease argument logic");
    expect(workflow).toContain(`gh release create "$RELEASE_TAG" "\${release_args[@]}" \\\n`);

    for (const [tag, prerelease] of [
        ["v0.10.0-next.1", true],
        ["v0.10.0", false],
    ] as const) {
        const result = Bun.spawnSync(
            ["bash", "-c", `${argsBlock}\nprintf '%s\\n' "\${release_args[@]}"`],
            {
                env: { ...process.env, RELEASE_TAG: tag },
                stdout: "pipe",
                stderr: "pipe",
            },
        );
        const output = `${result.stdout.toString()}${result.stderr.toString()}`;
        if (result.exitCode !== 0) throw new Error(`release args for ${tag} failed:\n${output}`);
        const args = result.stdout.toString().trim().split("\n");
        expect(args).toEqual(
            prerelease
                ? ["--verify-tag", "--generate-notes", "--prerelease"]
                : ["--verify-tag", "--generate-notes"],
        );
        console.log(`${tag}: ${args.join(" ")}`);
    }
});
