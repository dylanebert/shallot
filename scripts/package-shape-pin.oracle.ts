import { afterAll, beforeAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";
import { normalizePath } from "../src/project/paths";
import { bunCommand } from "./bun-child";
import { spawnPackedInstall } from "./packed-install-spawn";

setDefaultTimeout(CEILING.node);
const ROOT = resolve(import.meta.dir, "..");
const TYPEGPU_RANGE = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")).peerDependencies
    .typegpu;
function run(command: string[], cwd: string, label: string): string {
    return spawnPackedInstall(
        command[0] === "bun" ? bunCommand(command.slice(1)) : command,
        cwd,
        label,
    );
}

for (const subject of ["pin", "overlay", "restore"] as const) {
    describe(`staged published project pin: ${subject}`, () => {
        const scratch = realpathSync.native(
            mkdtempSync(join(tmpdir(), "shallot-package-shape-stage-")),
        );
        const project = join(scratch, "project");
        const rootManifest = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
        const tarball = join(scratch, `dylanebert-shallot-${rootManifest.version}.tgz`);

        let manifestBefore: string;
        let lockBefore: string;
        let pinnedVersion: string;
        const installed = () =>
            JSON.parse(
                readFileSync(
                    join(project, "node_modules/@dylanebert/shallot/package.json"),
                    "utf8",
                ),
            ).version;

        afterAll(() => rmSync(scratch, { recursive: true, force: true }));
        beforeAll(() => {
            mkdirSync(project);
            run(
                ["bun", "pm", "pack", "--ignore-scripts", "--destination", scratch, "--quiet"],
                ROOT,
                "packing staged Shallot",
            );
            writeFileSync(
                join(project, "package.json"),
                `${JSON.stringify(
                    {
                        name: "shallot-package-shape-staged-fixture",
                        private: true,
                        type: "module",
                        dependencies: { "@dylanebert/shallot": "^0.9.5" },
                        devDependencies: { typegpu: TYPEGPU_RANGE },
                    },
                    null,
                    2,
                )}\n`,
            );
            if (subject !== "pin") {
                run(["bun", "install", "--no-progress"], project, "preparing the published pin");
                manifestBefore = readFileSync(join(project, "package.json"), "utf8");
                lockBefore = readFileSync(join(project, "bun.lock"), "utf8");
                pinnedVersion = installed();
                expect(Bun.semver.satisfies(pinnedVersion, "^0.9.5")).toBe(true);
            }
            if (subject === "restore") {
                run(
                    ["bun", "add", "--no-save", `./${normalizePath(relative(project, tarball))}`],
                    project,
                    "preparing the packed overlay",
                );
                expect(installed()).toBe(rootManifest.version);
                expect(installed()).not.toBe(pinnedVersion);
                expect(readFileSync(join(project, "package.json"), "utf8")).toBe(manifestBefore);
                expect(readFileSync(join(project, "bun.lock"), "utf8")).toBe(lockBefore);
            }
        }, CEILING.startup);

        if (subject === "pin")
            test("the published pin installs", () => {
                run(["bun", "install", "--no-progress"], project, "installing the published pin");
                manifestBefore = readFileSync(join(project, "package.json"), "utf8");
                lockBefore = readFileSync(join(project, "bun.lock"), "utf8");
                pinnedVersion = installed();
                expect(Bun.semver.satisfies(pinnedVersion, "^0.9.5")).toBe(true);
                console.log(`staged restore pin: @dylanebert/shallot@${pinnedVersion}`);
            });

        if (subject === "overlay")
            test("the packed overlay preserves the project pin and lock", () => {
                // A ./-relative tarball path, the same on every host.
                run(
                    ["bun", "add", "--no-save", `./${normalizePath(relative(project, tarball))}`],
                    project,
                    "overlaying the packed Shallot tarball",
                );
                expect(installed()).toBe(rootManifest.version);
                expect(readFileSync(join(project, "package.json"), "utf8")).toBe(manifestBefore);
                expect(readFileSync(join(project, "bun.lock"), "utf8")).toBe(lockBefore);
            });

        if (subject === "restore")
            test("bun install restores the published pin after the overlay", () => {
                run(["bun", "install", "--no-progress"], project, "restoring the published pin");
                expect(installed()).toBe(pinnedVersion);
                expect(readFileSync(join(project, "package.json"), "utf8")).toBe(manifestBefore);
                expect(readFileSync(join(project, "bun.lock"), "utf8")).toBe(lockBefore);
            });
    });
}

test(
    "a stranger's hud copy type-checks its Svelte components with online peer resolution",
    () => {
        const scratch = realpathSync.native(mkdtempSync(join(tmpdir(), "shallot-hud-pin-")));
        try {
            run(
                [
                    "bun",
                    "pm",
                    "pack",
                    "--ignore-scripts",
                    "--filename",
                    join(scratch, "packed.tgz"),
                ],
                ROOT,
                "packing hud engine",
            );
            const project = join(scratch, "hud");
            run(
                [
                    "bun",
                    "run",
                    join(ROOT, "bin/shallot.ts"),
                    "create",
                    project,
                    "--template",
                    "hud",
                ],
                ROOT,
                "creating hud copy",
            );
            const path = join(project, "package.json");
            const manifest = JSON.parse(readFileSync(path, "utf8"));
            manifest.dependencies["@dylanebert/shallot"] = "file:../packed.tgz";
            writeFileSync(path, JSON.stringify(manifest));
            run(["bun", "install"], project, "installing stranger hud copy online");
            expect(
                run(["bun", "run", "typecheck"], project, "typechecking stranger hud copy"),
            ).toContain("svelte-check found 0 errors");
        } finally {
            rmSync(scratch, { recursive: true, force: true });
        }
    },
    CEILING.startup,
);
