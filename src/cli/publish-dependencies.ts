import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { publishProgress } from "./publish-output";

export function installDev(projectDir: string, dependency: string): void {
    let directory = resolve(projectDir);
    let bunProject = false;
    while (true) {
        const locks = readdirSync(directory).filter((name) =>
            [
                "bun.lock",
                "bun.lockb",
                "package-lock.json",
                "npm-shrinkwrap.json",
                "yarn.lock",
                "pnpm-lock.yaml",
            ].includes(name),
        );
        if (locks.length) {
            bunProject = locks.includes("bun.lock");
            break;
        }
        const parent = dirname(directory);
        if (parent === directory) break;
        directory = parent;
    }
    if (!bunProject)
        throw new Error(`Add ${dependency} to the project's devDependencies, then rerun publish.`);
    publishProgress(`Adding ${dependency} to the project's devDependencies.`);
    const result = spawnSync("bun", ["add", "--dev", dependency], {
        cwd: projectDir,
        stdio: ["ignore", "pipe", "pipe"],
        encoding: "utf8",
    });
    if (result.error) throw result.error;
    if (result.status !== 0) {
        publishProgress(result.stdout, result.stderr);
        throw new Error(`Could not install ${dependency}.`);
    }
}
