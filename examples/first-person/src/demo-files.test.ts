import { expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

test("Git ignores first-person build outputs while reporting source and lockfiles as untracked", () => {
    const scratch = mkdtempSync(join(tmpdir(), "shallot-example-ignore-"));
    const root = join(scratch, "project");
    const globalExcludes = join(scratch, "global-excludes");
    const globalConfig = join(scratch, "global.gitconfig");
    const template = join(scratch, "empty-template");
    mkdirSync(root);
    mkdirSync(template);
    writeFileSync(globalExcludes, "dist/\n");
    writeFileSync(globalConfig, `[core]\n\texcludesFile = ${globalExcludes}\n`);
    const env: NodeJS.ProcessEnv = {
        ...process.env,
        HOME: scratch,
        XDG_CONFIG_HOME: scratch,
        GIT_CONFIG_NOSYSTEM: "1",
        GIT_CONFIG_GLOBAL: globalConfig,
        GIT_CONFIG_COUNT: "0",
        GIT_CONFIG_PARAMETERS: "",
    };
    for (const key of [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ])
        delete env[key];
    for (const key of Object.keys(env)) if (key.startsWith("GIT_CONFIG_KEY_")) delete env[key];
    const git = (...args: string[]) =>
        execFileSync("git", ["-C", root, ...args], { encoding: "utf8", env });
    try {
        const ignore = readFileSync(resolve(import.meta.dir, "../.gitignore"), "utf8");
        writeFileSync(join(root, ".gitignore"), ignore);
        for (const dir of ["src", "dist", "build", ".artifacts"])
            mkdirSync(join(root, dir), { recursive: true });
        for (const file of [
            "src/demo.ts",
            "bun.lock",
            "dist/index.js",
            "build/app.js",
            ".artifacts/run",
        ])
            writeFileSync(join(root, file), "fixture\n");
        git("init", "-q", `--template=${template}`);

        const ignored = git(
            "check-ignore",
            "-v",
            "dist/index.js",
            "build/app.js",
            ".artifacts/run",
        );
        const status = git("status", "--short", "--untracked-files=all");
        expect(ignored).toContain("build/app.js");
        expect(ignored).toContain(".artifacts/run");
        const distDecision = ignored.split("\n").find((line) => line.endsWith("dist/index.js"));
        expect(distDecision).toContain(".gitignore:");
        writeFileSync(join(root, ".gitignore"), ignore.replace(/^dist\/\n/m, ""));
        expect(git("check-ignore", "-v", "dist/index.js")).toContain(globalExcludes);
        expect(status).toContain("?? bun.lock");
        expect(status).toContain("?? src/demo.ts");
        expect(status).not.toContain("dist/");
        expect(status).not.toContain("build/");
        expect(status).not.toContain(".artifacts/");
    } finally {
        rmSync(scratch, { recursive: true, force: true });
    }
}, 250);
