import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";

import { removeTestDirectory } from "../../scripts/test-directory";

setDefaultTimeout(TIMEOUT.node);

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
}, TIMEOUT.startup);

afterAll(() => {
    removeTestDirectory(temp);
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
    delete env.SHALLOT_ICON_ICO;
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

test("Windows resource selection uses an absolute owner path and refuses failed overrides", () => {
    const tools = resolve(temp, "resource-tools");
    mkdirSync(tools);
    const source = resolve(tools, "rc.rs");
    writeFileSync(
        source,
        `fn main() {
        let args: Vec<String> = std::env::args().collect();
        let rc = std::fs::read_to_string(args.last().unwrap()).unwrap();
        std::fs::write(std::env::var("RC_FIXTURE_LOG").unwrap(), rc).unwrap();
        std::process::exit(std::env::var("RC_FIXTURE_STATUS").unwrap_or_else(|_| "0".into()).parse().unwrap());
    }`,
    );
    execFileSync("rustc", [source, "-o", resolve(tools, "rc.exe")], { stdio: "pipe" });
    const custom = resolve(temp, "consumer icon.ico");
    const bytes = Buffer.concat([
        readFileSync(resolve(crate, "icon.ico")),
        Buffer.from("consumer resource"),
    ]);
    writeFileSync(custom, bytes);
    const vars = {
        CARGO_CFG_TARGET_OS: "windows",
        PATH: tools,
        RC_FIXTURE_LOG: resolve(temp, "resource.log"),
        SHALLOT_ASSETS: "disk",
    };
    const defaultResult = runBuild("default-resource", { ...vars, SHALLOT_ICON_ICO: "" });
    expect(defaultResult.status).toBe(0);
    expect(JSON.parse(readFileSync(vars.RC_FIXTURE_LOG, "utf8").slice(7))).toBe(
        resolve(crate, "icon.ico"),
    );
    const replacement = runBuild("custom-resource", { ...vars, SHALLOT_ICON_ICO: custom });
    expect(replacement.status).toBe(0);
    expect(replacement.stdout).toContain(`cargo:rerun-if-changed=${custom}`);
    expect(JSON.parse(readFileSync(vars.RC_FIXTURE_LOG, "utf8").slice(7))).toBe(custom);
    expect(readFileSync(custom)).toEqual(bytes);
    const failed = runBuild("failed-resource", {
        ...vars,
        SHALLOT_ICON_ICO: custom,
        RC_FIXTURE_STATUS: "2",
    });
    expect(failed.status).not.toBe(0);
    expect(failed.stderr).toContain("cannot compile requested Windows ICO resource");
    expect(failed.stdout).not.toContain("cargo:rustc-link-arg=");
    writeFileSync(custom, "invalid ICO");
    const invalid = runBuild("invalid-resource", { ...vars, SHALLOT_ICON_ICO: custom });
    expect(invalid.status).not.toBe(0);
    expect(invalid.stderr).toContain("invalid Windows ICO resource");
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
