#!/usr/bin/env bun

import { existsSync, readFileSync, writeFileSync } from "fs";
import { join, resolve } from "path";

const projectDir = resolve(import.meta.dir, "..");
const name = process.argv[2] || "local-test";
const dir = resolve(name);

if (existsSync(dir)) {
    console.error(`"${name}" already exists`);
    process.exit(1);
}

const createShallot = resolve(projectDir, "packages/create-shallot/index.ts");
const scaffold = Bun.spawnSync(["bun", createShallot, name], {
    stdout: "inherit",
    stderr: "inherit",
});
if (scaffold.exitCode !== 0) {
    console.error(`scaffold failed (exit ${scaffold.exitCode})`);
    process.exit(1);
}

const pkgDir = resolve(projectDir, "packages/shallot");
Bun.spawnSync(["bun", "pm", "pack", "--destination", dir, "--quiet"], {
    cwd: pkgDir,
    stdout: "pipe",
    stderr: "inherit",
});

const tgz = new Bun.Glob("*.tgz").scanSync(dir).next().value;
if (!tgz) {
    console.error("pack failed");
    process.exit(1);
}

const pkgJsonPath = join(dir, "package.json");
const pkgJson = JSON.parse(readFileSync(pkgJsonPath, "utf-8"));
pkgJson.dependencies["@dylanebert/shallot"] = `./${tgz}`;
writeFileSync(pkgJsonPath, JSON.stringify(pkgJson, null, 2) + "\n");

Bun.spawnSync(["bun", "install"], { cwd: dir, stdout: "inherit", stderr: "inherit" });

console.log();
console.log(`cd ${name} && bunx shallot dev`);
