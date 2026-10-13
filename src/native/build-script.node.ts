import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";

setDefaultTimeout(CEILING.node);

const repo = resolve(import.meta.dir, "../..");
const crate = resolve(repo, "crates/native");
let temp: string;
let script: string;

beforeAll(() => {
    const started = performance.now();
    temp = mkdtempSync(resolve(tmpdir(), "shallot-native-build-script-"));
    script = resolve(temp, process.platform === "win32" ? "build-script.exe" : "build-script");
    execFileSync("rustc", ["--edition=2021", resolve(crate, "build.rs"), "-o", script], {
        cwd: repo,
        stdio: "pipe",
    });
    console.info(
        JSON.stringify({
            fixtureSetup: "standalone rustc build-script compile",
            beforeAllMs: Number((performance.now() - started).toFixed(2)),
        }),
    );
}, CEILING.startup);

afterAll(() => {
    rmSync(temp, { recursive: true, force: true });
});

function runBuild(label: string, vars: Record<string, string | undefined> = {}) {
    const outDir = resolve(temp, label);
    mkdirSync(outDir, { recursive: true });
    const env: Record<string, string | undefined> = {
        ...process.env,
        OUT_DIR: outDir,
        PROFILE: "release",
        CARGO_MANIFEST_DIR: crate,
        CARGO_CFG_TARGET_OS: "linux",
    };
    delete env.SHALLOT_ASSETS;
    delete env.SHALLOT_DIST;
    Object.assign(env, vars);
    const result = spawnSync(script, [], { cwd: crate, env, encoding: "utf8" });
    return {
        status: result.status,
        stdout: result.stdout,
        stderr: result.stderr,
        assets: existsSync(resolve(outDir, "assets.rs"))
            ? readFileSync(resolve(outDir, "assets.rs"), "utf8")
            : null,
    };
}

const warning =
    "cargo:warning=SHALLOT_DIST unset or empty — release binary embeds no web assets; it will serve the sibling dist/ on disk (Contents/Resources/dist on mac) instead";

test("declared disk release mode is quiet and generates an empty asset list", () => {
    const result = runBuild("disk-release", { SHALLOT_ASSETS: "disk" });
    expect(result.status).toBe(0);
    expect(result.stdout).toContain("cargo:rerun-if-env-changed=SHALLOT_ASSETS");
    expect(result.stdout).not.toContain("cargo:warning=");
    expect(result.assets).toBe("pub static ASSETS: &[(&str, &[u8])] = &[\n];\n");
});

test("an undeclared release with no assets retains the exact warning", () => {
    const result = runBuild("undeclared-release");
    expect(result.status).toBe(0);
    expect(result.stdout).toContain(warning);
    expect(result.assets).toBe("pub static ASSETS: &[(&str, &[u8])] = &[\n];\n");
});

test("source release embeds SHALLOT_DIST assets without the disk warning", () => {
    const dist = resolve(temp, "source-dist");
    mkdirSync(dist, { recursive: true });
    writeFileSync(resolve(dist, "index.html"), "fixture page");
    const result = runBuild("source-release", { SHALLOT_DIST: dist });
    expect(result.status).toBe(0);
    expect(result.stdout).not.toContain("cargo:warning=");
    expect(result.assets).toContain('("index.html", include_bytes!("');
});

test("debug with no assets does not warn", () => {
    const outDir = resolve(temp, "debug");
    mkdirSync(outDir, { recursive: true });
    const env: Record<string, string | undefined> = {
        ...process.env,
        OUT_DIR: outDir,
        PROFILE: "debug",
        CARGO_MANIFEST_DIR: crate,
        CARGO_CFG_TARGET_OS: "linux",
    };
    delete env.SHALLOT_ASSETS;
    delete env.SHALLOT_DIST;
    const result = spawnSync(script, [], { cwd: crate, env, encoding: "utf8" });
    expect(result.status).toBe(0);
    expect(result.stdout).not.toContain("cargo:warning=");
});

test("disk release mode refuses embedded source assets", () => {
    const dist = resolve(temp, "contradictory-dist");
    mkdirSync(dist, { recursive: true });
    writeFileSync(resolve(dist, "index.html"), "fixture page");
    const result = runBuild("contradictory-release", {
        SHALLOT_ASSETS: "disk",
        SHALLOT_DIST: dist,
    });
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain(
        "SHALLOT_ASSETS=disk cannot be used when SHALLOT_DIST contains assets",
    );
});
