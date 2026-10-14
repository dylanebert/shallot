import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { CEILING } from "@dylanebert/shallot/testing";
import { cargoRuntimeDir } from "./provenance.js";

setDefaultTimeout(CEILING.node);

const repo = resolve(import.meta.dir, "../..");
let cargoRoot: string;
let firstRuntime: string;
let changedRuntime: string;
let shell: string;
let buildLog: string;
let firstBuild: string;

beforeAll(() => {
    const started = performance.now();
    cargoRoot = mkdtempSync(resolve(tmpdir(), "shallot-cargo-provenance-"));
    firstRuntime = runtime("runtime-first", "first runtime");
    changedRuntime = runtime("runtime-changed", "changed runtime");
    buildLog = resolve(cargoRoot, "dependency-build.log");
    const dependency = resolve(cargoRoot, "cef-dll-sys");
    shell = resolve(cargoRoot, "shell");

    mkdirSync(resolve(dependency, "src"), { recursive: true });
    writeFileSync(
        resolve(dependency, "Cargo.toml"),
        `[package]\nname="cef-dll-sys"\nversion="154.2.0"\nedition="2021"\nlinks="cef_dll_wrapper"\nbuild="build.rs"\n[lib]\npath="src/lib.rs"\n`,
    );
    writeFileSync(resolve(dependency, "src/lib.rs"), "pub fn fixture() {}\n");
    writeFileSync(
        resolve(dependency, "build.rs"),
        `use std::io::Write;\nfn main() {\n println!("cargo:rerun-if-env-changed=CEF_PATH");\n let cef=std::env::var("CEF_PATH").unwrap();\n println!("cargo::metadata=CEF_DIR={cef}");\n println!("cargo::metadata=CEF_API_VERSION=15400");\n let log=std::env::var("SHALLOT_CARGO_FIXTURE_LOG").unwrap();\n writeln!(std::fs::OpenOptions::new().create(true).append(true).open(log).unwrap(), "build-script").unwrap();\n}\n`,
    );

    mkdirSync(resolve(shell, "src"), { recursive: true });
    cpSync(resolve(repo, "crates/native/build.rs"), resolve(shell, "build.rs"));
    cpSync(resolve(repo, "crates/native/icon.ico"), resolve(shell, "icon.ico"));
    writeFileSync(
        resolve(shell, "Cargo.toml"),
        `[package]\nname="shallot-native"\nversion="0.10.0"\nedition="2021"\nbuild="build.rs"\n[dependencies]\ncef-dll-sys={path="../cef-dll-sys"}\n`,
    );
    writeFileSync(resolve(shell, "src/main.rs"), "fn main() {}\n");
    firstBuild = build(firstRuntime);
    console.info(
        JSON.stringify({
            fixtureSetup: "real Cargo provenance cold build",
            beforeAllMs: Number((performance.now() - started).toFixed(2)),
            startupCeilingMs: CEILING.startup,
        }),
    );
}, CEILING.startup);

afterAll(() => {
    rmSync(cargoRoot, { recursive: true, force: true });
});

function runtime(name: string, contents: string) {
    const path = resolve(cargoRoot, name);
    mkdirSync(path);
    writeFileSync(resolve(path, "libcef.so"), contents);
    return path;
}

function build(cefPath: string) {
    return execFileSync(
        "cargo",
        ["build", "--offline", "--message-format=json-render-diagnostics"],
        {
            cwd: shell,
            encoding: "utf8",
            env: {
                ...process.env,
                CEF_PATH: cefPath,
                SHALLOT_ICON_ICO: "",
                SHALLOT_CARGO_FIXTURE_LOG: buildLog,
            },
            stdio: ["ignore", "pipe", "pipe"],
        },
    );
}

test("Cargo forwards links metadata into native JSON on a warm cache hit", () => {
    const warm = build(firstRuntime);
    expect(cargoRuntimeDir(firstBuild, "libcef.so")).toBe(firstRuntime);
    expect(cargoRuntimeDir(warm, "libcef.so")).toBe(firstRuntime);
    expect(readFileSync(buildLog, "utf8")).toBe("build-script\n");
});

test("changing explicit CEF_PATH invalidates Cargo's cached dependency provenance", () => {
    const changed = build(changedRuntime);
    expect(cargoRuntimeDir(changed, "libcef.so")).toBe(changedRuntime);
    expect(readFileSync(buildLog, "utf8")).toBe("build-script\nbuild-script\n");
});
