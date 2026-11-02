import { expect, setDefaultTimeout, test } from "bun:test";
import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { packedInstall } from "./packed-install-spawn";
import { removeTestDirectory } from "./test-directory";

setDefaultTimeout(TIMEOUT.node);

test("the installed bin runs in a hand-written Node project and without Node under Bun", () => {
    const root = resolve(import.meta.dirname, "..");
    const manifest = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
    const packageVersion = manifest.version;
    const scratch = mkdtempSync(join(tmpdir(), "shallot-packed-cli-"));
    try {
        const cliProject = join(scratch, "project");
        mkdirSync(cliProject);
        const pack = Bun.spawnSync(
            ["bun", "pm", "pack", "--ignore-scripts", "--destination", scratch, "--quiet"],
            { cwd: root, stdout: "pipe", stderr: "pipe" },
        );
        expect(pack.exitCode, pack.stderr.toString()).toBe(0);
        writeFileSync(
            join(cliProject, "package.json"),
            JSON.stringify({
                name: "hand-written-game",
                private: true,
                type: "module",
                dependencies: {
                    "@dylanebert/shallot": `file:../dylanebert-shallot-${packageVersion}.tgz`,
                    vite: manifest.devDependencies.vite,
                },
            }),
        );
        packedInstall(
            ["install", "--linker=isolated", "--no-progress"],
            cliProject,
            "installing hand-written project",
        );
        writeFileSync(
            join(cliProject, "index.html"),
            '<script type="module" src="/main.ts"></script>',
        );
        writeFileSync(
            join(cliProject, "main.ts"),
            'import { World } from "@dylanebert/shallot"; console.log(new World());',
        );
        writeFileSync(
            join(cliProject, "vite.config.ts"),
            'import { shallot } from "@dylanebert/shallot/vite"; export default { plugins: [shallot()] };',
        );
        const windows = process.platform === "win32";
        const executables = { node: "", bun: "" };
        for (const runtime of ["node", "bun"] as const) {
            const path = join(scratch, runtime);
            mkdirSync(path);
            executables[runtime] = join(path, `${runtime}${windows ? ".exe" : ""}`);
            symlinkSync(Bun.which(runtime)!, executables[runtime]);
            expect(Bun.which(runtime === "node" ? "bun" : "node", { PATH: path })).toBeNull();
        }
        const bin = windows
            ? join(cliProject, "node_modules/@dylanebert/shallot/dist/cli/shallot.js")
            : join(cliProject, "node_modules/.bin/shallot");
        const run = (runtime: "node" | "bun", args: string[], success = true) => {
            const path = join(scratch, runtime);
            const command =
                runtime === "bun"
                    ? [executables.bun, "run", bin, ...args]
                    : windows
                      ? [executables.node, bin, ...args]
                      : [bin, ...args];
            const result = Bun.spawnSync(command, {
                cwd: cliProject,
                env: { ...process.env, PATH: path },
                stdout: "pipe",
                stderr: "pipe",
            });
            const output = result.stdout.toString() + result.stderr.toString();
            if (success) expect(result.exitCode, output).toBe(0);
            else expect(result.exitCode, output).not.toBe(0);
            return output;
        };
        expect(run("node", ["--version"])).toContain(packageVersion);
        expect(run("node", ["--skill"])).toContain("shallot");
        run("node", ["migrate"]);
        run("node", ["create", join(scratch, "node-created")]);
        const host =
            process.platform === "darwin"
                ? "mac"
                : process.platform === "win32"
                  ? "windows"
                  : "linux";
        expect(run("node", ["preview", "--target", host], false)).toContain(
            `no ${host} build found at`,
        );
        expect(existsSync(join(scratch, "node-created/package.json"))).toBe(true);
        const build = Bun.spawnSync(
            [Bun.which("node")!, join(cliProject, "node_modules/vite/bin/vite.js"), "build"],
            {
                cwd: cliProject,
                env: { ...process.env, PATH: join(scratch, "node") },
                stdout: "pipe",
                stderr: "pipe",
            },
        );
        expect(build.exitCode, build.stderr.toString()).toBe(0);
        expect(run("bun", ["--version"])).toContain(packageVersion);
        run("bun", ["create", join(scratch, "bun-created")]);
        expect(existsSync(join(scratch, "bun-created/package.json"))).toBe(true);
    } finally {
        removeTestDirectory(scratch);
    }
});
