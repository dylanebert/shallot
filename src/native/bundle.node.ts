import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { removeTestDirectory } from "../../scripts/test-directory";
import { normalizePath } from "../project/paths";
import { cargoRuntimeDir } from "./provenance.js";

setDefaultTimeout(TIMEOUT.node);

import { execFileSync } from "node:child_process";
import {
    chmodSync,
    cpSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, resolve } from "node:path";

const repo = resolve(import.meta.dir, "../..");
let packedPackageDir: string;
let packageArchive: string;
beforeAll(async () => {
    const started = performance.now();
    packedPackageDir = mkdtempSync(resolve(tmpdir(), "shallot-native-package-"));
    packageArchive = execFileSync(
        "bun",
        ["pm", "pack", "--ignore-scripts", "--quiet", "--destination", packedPackageDir],
        { cwd: repo, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] },
    ).trim();
    await new Bun.Archive(readFileSync(packageArchive)).extract(
        resolve(packedPackageDir, "unpacked"),
    );
    console.info(
        JSON.stringify({
            fixtureSetup: "native package pack/extract",
            beforeAllMs: Number((performance.now() - started).toFixed(2)),
        }),
    );
}, TIMEOUT.startup);

afterAll(() => {
    removeTestDirectory(packedPackageDir);
});

function fixture() {
    const root = mkdtempSync(resolve(tmpdir(), "shallot-native-consumer-"));
    const install = resolve(root, "node_modules/@dylanebert/shallot");
    const pkg = resolve(install, "package");
    const native = resolve(pkg, "src/native");
    const crate = resolve(pkg, "crates/native");
    const tools = resolve(root, "tools");
    const project = resolve(root, "game");
    const bashEnv = resolve(root, "bash-env.sh");
    mkdirSync(install, { recursive: true });
    cpSync(resolve(packedPackageDir, "unpacked/package"), pkg, { recursive: true });
    symlinkSync(
        resolve(repo, "node_modules"),
        resolve(pkg, "node_modules"),
        process.platform === "win32" ? "junction" : "dir",
    );
    mkdirSync(tools, { recursive: true });
    mkdirSync(resolve(project, "dist"), { recursive: true });
    writeFileSync(resolve(project, "dist/index.html"), "fixture page");

    const cargo = `#!/usr/bin/env node
const fs = require('node:fs'); const path = require('node:path');
const args = process.argv.slice(2); const target = args[args.indexOf('--target') + 1];
const release = args.includes('--release'); const profile = release ? 'release' : 'debug';
const targetDir = process.env.CARGO_TARGET_DIR;
const suffix = target.includes('windows') ? '.exe' : '';
const output = path.join(targetDir, target, profile);
fs.mkdirSync(output, { recursive: true });
const shell = path.join(output, 'shallot-native' + suffix);
fs.writeFileSync(shell, "#!/usr/bin/env node\\nif(process.env.FIXTURE_LAUNCH_LOG) require('node:fs').appendFileSync(process.env.FIXTURE_LAUNCH_LOG, JSON.stringify({ executable: process.argv[1], devUrl: process.env.SHALLOT_DEV_URL }) + String.fromCharCode(10));\\n");
fs.chmodSync(shell, 0o755);
if (target.includes('darwin')) fs.writeFileSync(path.join(output, 'shallot-helper'), 'helper');
if (process.env.CARGO_FIXTURE_LOG) fs.appendFileSync(process.env.CARGO_FIXTURE_LOG, 'cargo\\n');
if (process.env.ICON_FIXTURE_LOG) fs.appendFileSync(process.env.ICON_FIXTURE_LOG, JSON.stringify({ico: process.env.SHALLOT_ICON_ICO, icoBytes: process.env.SHALLOT_ICON_ICO ? fs.readFileSync(process.env.SHALLOT_ICON_ICO).toString('base64') : null, png: process.env.SHALLOT_DIST ? fs.readFileSync(path.join(process.env.SHALLOT_DIST, 'icon.png')).toString('base64') : null}) + '\\n');
if (process.env.ASSET_FIXTURE_LOG) fs.appendFileSync(process.env.ASSET_FIXTURE_LOG, JSON.stringify({ dist: process.env.SHALLOT_DIST, hasIndex: process.env.SHALLOT_DIST ? fs.existsSync(path.join(process.env.SHALLOT_DIST, 'index.html')) : false }) + '\\n');
const runtime = process.env.CEF_PATH || process.env.FIXTURE_RUNTIME;
const env = process.env.OMIT_CEF_PROVENANCE === '1' || !runtime ? [] : [['SHALLOT_CEF_DIR', runtime]];
console.log(JSON.stringify({ reason: 'build-script-executed', package_id: 'path+file:///fixture#shallot-native@0.10.0', env }));
`;
    writeTool(tools, "cargo", cargo);
    writeTool(tools, "cargo-xwin", cargo);
    writeTool(
        tools,
        "ditto",
        `#!/usr/bin/env node
const fs=require('node:fs');const cp=require('node:fs').cpSync; cp(process.argv[2],process.argv[3],{recursive:true});
`,
    );
    writeTool(
        tools,
        "sips",
        `#!/usr/bin/env node
const fs=require('node:fs');const a=process.argv.slice(2);const i=a.indexOf('--out');if(i>=0)fs.copyFileSync(a[i-1],a[i+1]);
if(process.env.MAC_ICON_LOG && a[0]==='-z') fs.writeFileSync(process.env.MAC_ICON_LOG,fs.readFileSync(a[3]));
`,
    );
    writeTool(
        tools,
        "iconutil",
        `#!/usr/bin/env node
const a=process.argv.slice(2);require('node:fs').writeFileSync(a[a.indexOf('-o')+1],'icon');
`,
    );
    for (const command of ["codesign", "strip"]) writeTool(tools, command, "process.exit(0);");
    const wslpath = `#!/usr/bin/env node
const a=process.argv.slice(2);const p=a[a.length-1];
if(a[0]==='-w') console.log('\\\\wsl.localhost\\\\Ubuntu\\\\fixture');
else if(p==='C:\\\\') console.log(process.env.FIXTURE_DRIVE_ROOT);
else if(p.includes('\\\\shallot\\\\winbuild')) console.log(process.env.FIXTURE_TARGET_DIR);
else if(p==='C:\\\\cef') console.log(process.env.FIXTURE_RUNTIME);
else console.log(process.env.FIXTURE_TARGET_DIR);
`;
    writeTool(tools, "wslpath", wslpath);
    const powershell = `
const fs=require('node:fs');const path=require('node:path');const a=process.argv.slice(2).join(' ');
if(a.includes('[Console]::Write($env:LOCALAPPDATA)')) { process.stdout.write('C:\\\\Users\\\\fixture\\\\AppData\\\\Local'); process.exit(0); }
const target='x86_64-pc-windows-msvc', profile=a.includes('--release')?'release':'debug';
const out=path.join(process.env.FIXTURE_TARGET_DIR,target,profile);fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'shallot-native.exe'),'shell');
const cef=process.env.FIXTURE_WIN_CEF || 'C:\\\\cef';
console.log(JSON.stringify({reason:'build-script-executed',package_id:'path+file:///fixture#shallot-native@0.10.0',env:[['SHALLOT_CEF_DIR',cef]]}));
`;
    if (process.platform !== "win32") writeTool(tools, "powershell.exe", powershell);
    if (process.platform === "win32") {
        writeFileSync(
            bashEnv,
            `ditto() { node "$(cygpath -w "$FIXTURE_TOOLS/ditto.js")" "$(cygpath -w "$1")" "$(cygpath -w "$2")"; }\nstrip() { node "$(cygpath -w "$FIXTURE_TOOLS/strip.js")"; }\n`,
        );
    }
    const driver = resolve(root, "run.ts");
    writeFileSync(
        driver,
        `import assert from 'node:assert/strict';
import { bundleNativeLinux, bundleNativeMac, bundleNativeWindows } from ${JSON.stringify(resolve(native, "index.ts"))};
const [kind, project, output, optsJson] = Bun.argv.slice(2); const opts=JSON.parse(optsJson);
try {
 if(kind==='linux') await bundleNativeLinux(project,output,opts);
 else if(kind==='windows') await bundleNativeWindows(project,output,opts);
 else if(kind==='mac') await bundleNativeMac(project,output,opts);
 else if(kind==='missing') await bundleNativeLinux(project,output,opts);
 else if(kind==='prebuilt-platform') {
   if(opts.target==='windows') await bundleNativeWindows(project,output,{release:true,portable:true});
   else if(opts.target==='mac') await bundleNativeMac(project,output,{release:true,portable:true});
   else await bundleNativeLinux(project,output,{release:true,portable:true});
   assert.equal(process.env.CEF_PATH,process.env.CALLER_CEF_PATH);
 }
 else if(kind==='prebuilt-sequence') {
   await bundleNativeLinux(project,output,{release:true,portable:true});
   assert.equal(process.env.CEF_PATH,process.env.CALLER_CEF_PATH);
   const cache=process.env.PREBUILT_CACHE; await import('node:fs').then(fs=>fs.rmSync(cache+'/.complete',{force:true}));
   globalThis.fetch=async()=>new Response('',{status:404});
   await bundleNativeLinux(project,output,{release:true,portable:true});
   assert.equal(process.env.CEF_PATH,process.env.CALLER_CEF_PATH);
 }
 else if(kind==='dev-sequence') {
   const { nativeOutDir } = await import(${JSON.stringify(resolve(native, "index.ts"))});
   const { bundleNativeLinux, bundleNativeMac, bundleNativeWindows } = await import(${JSON.stringify(resolve(native, "index.ts"))});
   const { buildProject } = await import(${JSON.stringify(resolve(pkg, "src/cli/build.ts"))});
   const { previewProject } = await import(${JSON.stringify(resolve(pkg, "src/cli/preview.ts"))});
   const platform = process.platform === 'win32' ? 'windows' : process.platform === 'darwin' ? 'mac' : 'linux';
   const portable = platform === 'linux';
   const standalone = nativeOutDir(project,platform,false,portable);
   const bundle = platform === 'windows' ? bundleNativeWindows : platform === 'mac' ? bundleNativeMac : bundleNativeLinux;
   await bundle(project,standalone,{portable});
   await buildProject(project,{target:platform,portable,dev:true});
   const standaloneDist = (await import('node:fs')).existsSync(platform === 'mac' ? standalone+'/game.app/Contents/Resources/dist/index.html' : standalone+'/dist/index.html');
   const devShell = nativeOutDir(project,platform,false,portable,true);
   const fs = await import('node:fs');
   const originalSpawn = Bun.spawn;
   Bun.spawn = (command, options) => {
     const args = Array.isArray(command) ? command : [command];
     const executable = args[0] === 'open' ? args[args.length - 1] : args[0];
     fs.appendFileSync(process.env.FIXTURE_LAUNCH_LOG, JSON.stringify({executable,devUrl:options?.env?.SHALLOT_DEV_URL}) + String.fromCharCode(10));
     return {exited:Promise.resolve(0)};
   };
   let previewStatus; let standalonePreviewStatus;
   try {
     previewStatus = await previewProject(project,{target:platform,portable,devUrl:'http://127.0.0.1:5173'});
     standalonePreviewStatus = await previewProject(project,{target:platform,portable});
   } finally { Bun.spawn = originalSpawn; }
   (await import('node:fs')).writeFileSync(process.env.DEV_SEQUENCE_LOG, JSON.stringify({platform,portable,standalone,standaloneDist,devShell,previewStatus,standalonePreviewStatus}));
 }
} catch(error) {
 if(kind==='missing') { assert.match(String(error),/no CEF runtime provenance/); assert.equal((await import('node:fs')).existsSync(output),false); }
 else throw error;
}
`,
    );
    return { root, pkg, crate, tools, project, driver, bashEnv };
}

function writeTool(dir: string, name: string, source: string) {
    if (process.platform === "win32") {
        writeFileSync(resolve(dir, `${name}.js`), source);
        writeFileSync(
            resolve(dir, `${name}.cmd`),
            `@echo off\r\n"${process.execPath}" "%~dp0${name}.js" %*\r\n`,
        );
        return;
    }
    const path = resolve(dir, name);
    const script = source.replace(/^#![^\n]*\n/, "");
    writeFileSync(path, `#!/usr/bin/env node\n${script}\n`);
    chmodSync(path, 0o755);
}

function bashPath(path: string) {
    if (process.platform !== "win32") return path;
    return normalizePath(path).replace(
        /^([A-Za-z]):/,
        (_, drive: string) => `/${drive.toLowerCase()}`,
    );
}

function shellToolEnvironment(f: ReturnType<typeof fixture>) {
    return process.platform === "win32"
        ? { BASH_ENV: bashPath(f.bashEnv), FIXTURE_TOOLS: bashPath(f.tools) }
        : {};
}

function runtimeDir(f: ReturnType<typeof fixture>, name: string, marker: string, content = name) {
    const path = resolve(f.root, "runtimes", name);
    mkdirSync(path, { recursive: true });
    if (marker.endsWith(".framework")) {
        const framework = resolve(path, marker);
        mkdirSync(framework, { recursive: true });
        writeFileSync(resolve(framework, "Chromium Embedded Framework"), content);
    } else writeFileSync(resolve(path, marker), content);
    return path;
}

function run(
    f: ReturnType<typeof fixture>,
    kind: string,
    output: string,
    opts: Record<string, unknown>,
    extraEnv: Record<string, string | undefined> = {},
) {
    const result = Bun.spawnSync(["bun", f.driver, kind, f.project, output, JSON.stringify(opts)], {
        cwd: repo,
        stdout: "pipe",
        stderr: "pipe",
        env: {
            ...process.env,
            PATH: [f.tools, process.env.PATH].filter(Boolean).join(delimiter),
            CEF_PATH: undefined,
            SHALLOT_DEV_URL: undefined,
            WSL_INTEROP: undefined,
            ...extraEnv,
        },
    });
    return {
        code: result.exitCode,
        stdout: new TextDecoder().decode(result.stdout),
        stderr: new TextDecoder().decode(result.stderr),
    };
}

function stagedAssetPath(output: string, platform: string): string {
    return platform === "mac"
        ? resolve(output, "game.app/Contents/Resources/dist/index.html")
        : resolve(output, "dist/index.html");
}

function seedAssetPrebuilt(
    f: ReturnType<typeof fixture>,
    platform: string,
    portable = false,
): Record<string, string> {
    const target = {
        linux: "x86_64-unknown-linux-gnu",
        mac: "aarch64-apple-darwin",
        windows: "x86_64-pc-windows-msvc",
    }[platform]!;
    const cacheRoot = resolve(f.root, "asset-cache");
    const version = JSON.parse(readFileSync(resolve(f.pkg, "package.json"), "utf8")).version;
    const cache = resolve(
        cacheRoot,
        "shallot/prebuilt",
        version,
        `${target}-${portable ? "portable" : "system"}`,
    );
    mkdirSync(cache, { recursive: true });
    writeFileSync(
        resolve(cache, platform === "windows" ? "shallot-native.exe" : "shallot-native"),
        "shell",
    );
    writeFileSync(resolve(cache, ".complete"), "");
    if (portable) {
        const cef = resolve(cache, "cef");
        mkdirSync(cef);
        writeFileSync(resolve(cef, "libcef.dll"), "cached runtime");
        return { XDG_CACHE_HOME: cacheRoot, LOCALAPPDATA: cacheRoot, FIXTURE_RUNTIME: cef };
    }
    return { XDG_CACHE_HOME: cacheRoot, LOCALAPPDATA: cacheRoot };
}

function assetBundle(
    platform: string,
    mode: "debug" | "source-release" | "prebuilt-release" | "dev",
    custom = false,
    staleDist = false,
) {
    const f = fixture();
    try {
        const output = resolve(f.root, `${platform}-${mode}`);
        const standard = readFileSync(resolve(repo, "assets/icon-1024.png"));
        const expected = custom ? Buffer.concat([standard, Buffer.from("consumer PNG")]) : standard;
        if (custom) {
            mkdirSync(resolve(f.project, "public"), { recursive: true });
            writeFileSync(resolve(f.project, "public/icon.png"), expected);
            writeFileSync(
                resolve(f.project, "dist/icon.png"),
                mode === "dev" ? "stale dist bytes" : expected,
            );
        }
        if (mode === "dev" && !staleDist)
            rmSync(resolve(f.project, "dist"), { recursive: true, force: true });
        if (mode === "source-release") {
            const pkg = JSON.parse(readFileSync(resolve(f.pkg, "package.json"), "utf8"));
            pkg.version = null;
            writeFileSync(resolve(f.pkg, "package.json"), JSON.stringify(pkg));
        }
        const extraEnv =
            mode === "source-release"
                ? { ASSET_FIXTURE_LOG: resolve(f.root, "asset.log") }
                : mode === "prebuilt-release"
                  ? seedAssetPrebuilt(f, platform)
                  : {};
        const result = run(
            f,
            platform,
            output,
            { release: mode.endsWith("release"), portable: false, dev: mode === "dev" },
            {
                ...extraEnv,
                ICON_FIXTURE_LOG: resolve(f.root, "icons.log"),
                MAC_ICON_LOG: resolve(f.root, "mac-icon.png"),
            },
        );
        expect(result.code, result.stderr).toBe(0);
        const staged = mode === "debug" || mode === "prebuilt-release";
        expect(existsSync(stagedAssetPath(output, platform))).toBe(staged);
        if (mode === "source-release") {
            const [record] = readFileSync(resolve(f.root, "asset.log"), "utf8").trim().split("\\n");
            expect(JSON.parse(record)).toEqual({
                dist: resolve(f.project, "dist"),
                hasIndex: true,
            });
        }
        if (mode === "dev") expect(existsSync(resolve(f.project, "dist"))).toBe(staleDist);
        if (mode !== "source-release") {
            const icon =
                platform === "mac"
                    ? resolve(output, "game.app/Contents/Resources/dist/icon.png")
                    : resolve(output, "dist/icon.png");
            expect(readFileSync(icon)).toEqual(expected);
        } else {
            const record = JSON.parse(readFileSync(resolve(f.root, "icons.log"), "utf8").trim());
            expect(Buffer.from(record.png, "base64")).toEqual(expected);
        }
        if (platform === "mac")
            expect(readFileSync(resolve(f.root, "mac-icon.png"))).toEqual(expected);
    } finally {
        removeTestDirectory(f.root);
    }
}

for (const platform of ["linux", "mac", "windows"]) {
    test(`native dev ${platform} ignores stale dist PNG`, () =>
        assetBundle(platform, "dev", true, true));
    for (const mode of ["debug", "source-release", "prebuilt-release", "dev"] as const) {
        test(`native asset delivery: ${platform} ${mode}`, () => assetBundle(platform, mode));
        test(`native custom PNG delivery: ${platform} ${mode}`, () =>
            assetBundle(platform, mode, true));
    }
}

for (const portable of [false, true]) {
    for (const selection of ["absent", "standard", "custom", "invalid"] as const) {
        test(`Windows ${portable ? "CEF" : "system"} executable resource ${selection} selects compatible prebuilt or owned source input`, () => {
            const f = fixture();
            try {
                const packaged = readFileSync(resolve(f.crate, "icon.ico"));
                const custom = Buffer.concat([packaged, Buffer.from("distinct consumer ICO")]);
                mkdirSync(resolve(f.project, "public"));
                const path = resolve(f.project, "public/icon.ico");
                if (selection !== "absent")
                    writeFileSync(
                        path,
                        selection === "custom"
                            ? custom
                            : selection === "invalid"
                              ? Buffer.from("bad ICO")
                              : packaged,
                    );
                const cacheEnv = seedAssetPrebuilt(f, "windows", portable);
                const env = {
                    ...cacheEnv,
                    CARGO_FIXTURE_LOG: resolve(f.root, "cargo.log"),
                    ICON_FIXTURE_LOG: resolve(f.root, "icons.log"),
                };
                const result = run(
                    f,
                    "windows",
                    resolve(f.root, "resource-output"),
                    { release: true, portable },
                    env,
                );
                if (selection === "invalid") {
                    expect(result.code).not.toBe(0);
                    expect(result.stderr).toContain("invalid Windows icon");
                } else {
                    expect(result.code, result.stderr).toBe(0);
                    if (selection === "custom") {
                        expect(result.stdout).toContain("public/icon.ico requires a source build");
                        const record = JSON.parse(
                            readFileSync(env.ICON_FIXTURE_LOG, "utf8").trim(),
                        );
                        expect(record.ico).toBe(path);
                        expect(Buffer.from(record.icoBytes, "base64")).toEqual(custom);
                        expect(readFileSync(path)).toEqual(custom);
                    } else expect(existsSync(env.CARGO_FIXTURE_LOG)).toBe(false);
                }
                expect(readFileSync(resolve(f.crate, "icon.ico"))).toEqual(packaged);
                const version = JSON.parse(
                    readFileSync(resolve(f.pkg, "package.json"), "utf8"),
                ).version;
                const cached = resolve(
                    cacheEnv.XDG_CACHE_HOME,
                    "shallot/prebuilt",
                    version,
                    `x86_64-pc-windows-msvc-${portable ? "portable" : "system"}`,
                    "shallot-native.exe",
                );
                expect(readFileSync(cached, "utf8")).toBe("shell");
            } finally {
                removeTestDirectory(f.root);
            }
        });
    }
}

for (const platform of ["linux", "mac", "windows"]) {
    test(`native standalone ${platform} build refuses without dist/index.html`, () => {
        const f = fixture();
        try {
            rmSync(resolve(f.project, "dist/index.html"));
            const output = resolve(f.root, `${platform}-missing-index`);
            const result = run(f, platform, output, { portable: false });
            expect(result.code).not.toBe(0);
            expect(result.stderr).toContain("vite build produced no dist/index.html");
            expect(existsSync(output)).toBe(false);
        } finally {
            removeTestDirectory(f.root);
        }
    });
}

test("native source release refuses without dist/index.html", () => {
    const f = fixture();
    try {
        rmSync(resolve(f.project, "dist/index.html"));
        const output = resolve(f.root, "linux-source-release-missing-index");
        const result = run(f, "linux", output, { release: true, portable: false });
        expect(result.code).not.toBe(0);
        expect(result.stderr).toContain("vite build produced no dist/index.html");
        expect(existsSync(output)).toBe(false);
    } finally {
        removeTestDirectory(f.root);
    }
});

test("native standalone build keeps the absent-directory refusal", () => {
    const f = fixture();
    try {
        rmSync(resolve(f.project, "dist"), { recursive: true, force: true });
        const output = resolve(f.root, "linux-missing-dist");
        const result = run(f, "linux", output, { portable: false });
        expect(result.code).not.toBe(0);
        expect(result.stderr).toContain("vite build produced no dist/ directory");
        expect(result.stderr).not.toContain("vite build produced no dist/index.html");
        expect(existsSync(output)).toBe(false);
    } finally {
        removeTestDirectory(f.root);
    }
});

test("native dev builds preserve the standalone bundle and preview launches the dev shell", () => {
    const f = fixture();
    try {
        const platform =
            process.platform === "win32"
                ? "windows"
                : process.platform === "darwin"
                  ? "mac"
                  : "linux";
        const portable = platform === "linux";
        const runtime = portable ? runtimeDir(f, "dev-sequence", "libcef.so") : undefined;
        const reportPath = resolve(f.root, "dev-sequence.json");
        const launchPath = resolve(f.root, "launch.jsonl");
        const output = resolve(f.root, "unused-output");
        const result = run(
            f,
            "dev-sequence",
            output,
            {},
            {
                FIXTURE_RUNTIME: runtime,
                DEV_SEQUENCE_LOG: reportPath,
                FIXTURE_LAUNCH_LOG: launchPath,
            },
        );
        expect(result.code, result.stderr).toBe(0);
        const report = JSON.parse(readFileSync(reportPath, "utf8"));
        expect(report.standaloneDist).toBe(true);
        expect(report.devShell).not.toBe(report.standalone);
        expect(report.previewStatus, result.stderr).toBe(0);
        expect(report.standalonePreviewStatus, result.stderr).toBe(0);
        const launches = readFileSync(launchPath, "utf8")
            .trim()
            .split("\n")
            .map((line) => JSON.parse(line));
        const binary = (dir: string) =>
            report.platform === "mac"
                ? resolve(dir, "game.app", "Contents", "MacOS", "game")
                : resolve(dir, `game${report.platform === "windows" ? ".exe" : ""}`);
        const standaloneLaunch =
            report.platform === "mac"
                ? resolve(report.standalone, "game.app")
                : binary(report.standalone);
        expect(launches).toHaveLength(2);
        expect(launches[0].executable).toBe(binary(report.devShell));
        expect(launches[0].devUrl).toBe("http://127.0.0.1:5173");
        expect(launches[1].executable).toBe(standaloneLaunch);
        expect(launches[1].devUrl).toBeUndefined();
    } finally {
        removeTestDirectory(f.root);
    }
});

function staleLinuxBundle(includeRelease: boolean) {
    const f = fixture();
    try {
        const target = "x86_64-unknown-linux-gnu";
        const selected = runtimeDir(f, "selected", "libcef.so");
        const old = runtimeDir(
            f,
            "old",
            "libcef.so",
            includeRelease ? "old release" : "old dependency",
        );
        const buildRoot = resolve(f.crate, "target", target);
        const staleBuild = includeRelease
            ? resolve(buildRoot, "release/build/cef-dll-sys-old/out/cef")
            : resolve(buildRoot, "debug/build/cef-dll-sys-dependency/out/cef");
        mkdirSync(staleBuild, { recursive: true });
        cpSync(resolve(old, "libcef.so"), resolve(staleBuild, "libcef.so"));
        const output = resolve(f.root, "linux-output");
        const result = run(f, "linux", output, { portable: true }, { FIXTURE_RUNTIME: selected });
        return {
            code: result.code,
            stderr: result.stderr,
            copied: existsSync(resolve(output, "cef/libcef.so"))
                ? readFileSync(resolve(output, "cef/libcef.so"), "utf8")
                : null,
        };
    } finally {
        removeTestDirectory(f.root);
    }
}

test("Linux bundler rejects an older release runtime beside the selected source build", () => {
    const result = staleLinuxBundle(true);
    expect(result.code, result.stderr).toBe(0);
    expect(result.copied).toBe("selected");
});

test("Linux bundler rejects an older dependency runtime in the selected profile", () => {
    const result = staleLinuxBundle(false);
    expect(result.code, result.stderr).toBe(0);
    expect(result.copied).toBe("selected");
});

test("Windows and Mac source bundle consumers copy their selected native markers", () => {
    const f = fixture();
    try {
        const windowsRuntime = runtimeDir(f, "windows", "libcef.dll");
        const windowsOut = resolve(f.root, "windows-output");
        const windows = run(
            f,
            "windows",
            windowsOut,
            { portable: true },
            { FIXTURE_RUNTIME: windowsRuntime },
        );
        expect(windows.code, windows.stderr).toBe(0);
        expect(readFileSync(resolve(windowsOut, "libcef.dll"), "utf8")).toBe("windows");

        const macRuntime = runtimeDir(f, "mac", "Chromium Embedded Framework.framework");
        const macOut = resolve(f.root, "mac-output");
        const mac = run(f, "mac", macOut, { portable: true }, { FIXTURE_RUNTIME: macRuntime });
        expect(mac.code, mac.stderr).toBe(0);
        expect(
            readFileSync(
                resolve(
                    macOut,
                    "game.app/Contents/Frameworks/Chromium Embedded Framework.framework/Chromium Embedded Framework",
                ),
                "utf8",
            ),
        ).toBe("mac");
    } finally {
        removeTestDirectory(f.root);
    }
});

test("explicit CEF_PATH is a Cargo build input; system mode needs no CEF; missing provenance refuses output", () => {
    const f = fixture();
    try {
        const configured = runtimeDir(f, "configured", "libcef.so");
        const output = resolve(f.root, "explicit-output");
        const explicit = run(f, "linux", output, { portable: true }, { CEF_PATH: configured });
        expect(explicit.code, explicit.stderr).toBe(0);
        expect(readFileSync(resolve(output, "cef/libcef.so"), "utf8")).toBe("configured");

        const systemOut = resolve(f.root, "system-output");
        const system = run(
            f,
            "linux",
            systemOut,
            { portable: false },
            { OMIT_CEF_PROVENANCE: "1" },
        );
        expect(system.code, system.stderr).toBe(0);
        expect(existsSync(resolve(systemOut, "cef"))).toBe(false);

        const missingOut = resolve(f.root, "missing-output");
        const missing = run(
            f,
            "missing",
            missingOut,
            { portable: true },
            { OMIT_CEF_PROVENANCE: "1" },
        );
        expect(missing.code, missing.stderr).toBe(0);
        expect(existsSync(missingOut)).toBe(false);
    } finally {
        removeTestDirectory(f.root);
    }
});

test("prebuilt hit and subsequent Cargo fallback preserve the caller's CEF_PATH", () => {
    const f = fixture();
    try {
        const target = "x86_64-unknown-linux-gnu";
        const cacheRoot = resolve(f.root, "cache");
        const version = JSON.parse(readFileSync(resolve(f.pkg, "package.json"), "utf8")).version;
        const cache = resolve(cacheRoot, "shallot/prebuilt", version, `${target}-portable`);
        const cachedRuntime = runtimeDir(f, "cached", "libcef.so");
        const callerRuntime = runtimeDir(f, "caller", "libcef.so");
        mkdirSync(resolve(cache, "cef"), { recursive: true });
        writeFileSync(resolve(cache, "shallot-native"), "prebuilt shell");
        cpSync(resolve(cachedRuntime, "libcef.so"), resolve(cache, "cef/libcef.so"));
        writeFileSync(resolve(cache, ".complete"), "");
        const output = resolve(f.root, "prebuilt-output");
        const result = run(
            f,
            "prebuilt-sequence",
            output,
            {},
            {
                XDG_CACHE_HOME: cacheRoot,
                LOCALAPPDATA: cacheRoot,
                CALLER_CEF_PATH: callerRuntime,
                CEF_PATH: callerRuntime,
                PREBUILT_CACHE: cache,
                CARGO_FIXTURE_LOG: resolve(f.root, "cargo.log"),
                FIXTURE_RUNTIME: callerRuntime,
            },
        );
        expect(result.code, result.stderr).toBe(0);
        expect(readFileSync(resolve(output, "cef/libcef.so"), "utf8")).toBe("caller");
        expect(readFileSync(resolve(f.root, "cargo.log"), "utf8")).toBe("cargo\n");
    } finally {
        removeTestDirectory(f.root);
    }
});

test("prebuilt Linux, Mac, and Windows shells bundle their archived runtime without environment leakage", () => {
    const f = fixture();
    try {
        const cacheRoot = resolve(f.root, "cache");
        const caller = runtimeDir(f, "caller-runtime", "libcef.so");
        const cases = [
            { kind: "linux", target: "x86_64-unknown-linux-gnu", marker: "libcef.so" },
            {
                kind: "mac",
                target: "aarch64-apple-darwin",
                marker: "Chromium Embedded Framework.framework",
            },
            { kind: "windows", target: "x86_64-pc-windows-msvc", marker: "libcef.dll" },
        ];
        for (const item of cases) {
            const version = JSON.parse(
                readFileSync(resolve(f.pkg, "package.json"), "utf8"),
            ).version;
            const cache = resolve(
                cacheRoot,
                "shallot/prebuilt",
                version,
                `${item.target}-portable`,
            );
            mkdirSync(cache, { recursive: true });
            const binary = item.target.includes("windows")
                ? "shallot-native.exe"
                : "shallot-native";
            writeFileSync(resolve(cache, binary), `prebuilt ${item.kind}`);
            if (item.kind === "mac") {
                const source = runtimeDir(f, "prebuilt-mac", item.marker, "prebuilt mac framework");
                cpSync(resolve(source, item.marker), resolve(cache, item.marker), {
                    recursive: true,
                });
                writeFileSync(resolve(cache, "shallot-helper"), "prebuilt helper");
            } else {
                const source = runtimeDir(
                    f,
                    `prebuilt-${item.kind}`,
                    item.marker,
                    `prebuilt ${item.kind} runtime`,
                );
                mkdirSync(resolve(cache, "cef"), { recursive: true });
                cpSync(resolve(source, item.marker), resolve(cache, "cef", item.marker));
            }
            writeFileSync(resolve(cache, ".complete"), "");
            const output = resolve(f.root, `${item.kind}-prebuilt-output`);
            const result = run(
                f,
                "prebuilt-platform",
                output,
                { target: item.kind },
                {
                    XDG_CACHE_HOME: cacheRoot,
                    LOCALAPPDATA: cacheRoot,
                    CEF_PATH: caller,
                    CALLER_CEF_PATH: caller,
                },
            );
            expect(result.code, result.stderr).toBe(0);
            if (item.kind === "mac") {
                expect(
                    readFileSync(
                        resolve(
                            output,
                            "game.app/Contents/Frameworks/Chromium Embedded Framework.framework/Chromium Embedded Framework",
                        ),
                        "utf8",
                    ),
                ).toBe("prebuilt mac framework");
            } else {
                const bundledMarker =
                    item.kind === "linux"
                        ? resolve(output, "cef", item.marker)
                        : resolve(output, item.marker);
                expect(readFileSync(bundledMarker, "utf8")).toBe(`prebuilt ${item.kind} runtime`);
            }
        }
    } finally {
        removeTestDirectory(f.root);
    }
});

test("WSL Windows bundler translates Cargo's runtime path before copying", () => {
    const f = fixture();
    try {
        const runtime = runtimeDir(f, "wsl", "libcef.dll", "translated runtime");
        if (process.platform === "win32") {
            // Native Windows CI has no WSL distro; it tests path translation while POSIX runs the consumer.
            const messages = JSON.stringify({
                reason: "build-script-executed",
                ["package_id"]: "path+file:///fixture#shallot-native@0.10.0",
                env: [["SHALLOT_CEF_DIR", "C:\\cef"]],
            });
            expect(
                cargoRuntimeDir(messages, "libcef.dll", (path) =>
                    path === "C:\\cef" ? runtime : path,
                ),
            ).toBe(runtime);
            return;
        }
        const output = resolve(f.root, "wsl-output");
        const result = run(
            f,
            "windows",
            output,
            { portable: true },
            {
                WSL_INTEROP: "fixture",
                FIXTURE_TARGET_DIR: resolve(f.root, "win-target"),
                FIXTURE_DRIVE_ROOT: f.root,
                FIXTURE_RUNTIME: runtime,
                FIXTURE_WIN_CEF: "C:\\cef",
            },
        );
        expect(result.code, result.stderr).toBe(0);
        expect(readFileSync(resolve(output, "libcef.dll"), "utf8")).toBe("translated runtime");
    } finally {
        removeTestDirectory(f.root);
    }
});

test("release archive assembly writes the selected Mac framework at the required archive path", async () => {
    const f = fixture();
    try {
        const target = "aarch64-apple-darwin";
        const crate = resolve(f.root, "release-crate");
        const build = resolve(crate, "target", target, "release");
        mkdirSync(build, { recursive: true });
        writeFileSync(resolve(build, "shallot-native"), "shell");
        writeFileSync(resolve(build, "shallot-helper"), "helper");
        const selected = runtimeDir(
            f,
            "release-selected",
            "Chromium Embedded Framework.framework",
            "selected framework",
        );
        const stale = resolve(crate, "target", target, "release/build/cef-dll-sys-stale/out/cef");
        const sameProfile = resolve(
            crate,
            "target",
            target,
            "release/build/cef-dll-sys-dependency/out/cef",
        );
        for (const path of [stale, sameProfile]) {
            const framework = resolve(path, "Chromium Embedded Framework.framework");
            mkdirSync(framework, { recursive: true });
            writeFileSync(
                resolve(framework, "Chromium Embedded Framework"),
                path === stale ? "stale release" : "stale dependency",
            );
        }
        const cargoJson = resolve(f.root, "cargo-messages.json");
        writeFileSync(
            cargoJson,
            JSON.stringify({
                reason: "build-script-executed",
                ["package_id"]: "path+file:///fixture#shallot-native@0.10.0",
                env: [["SHALLOT_CEF_DIR", normalizePath(selected)]],
            }),
        );
        const output = resolve(f.root, "release.tar.gz");
        const result = Bun.spawnSync(
            ["bash", bashPath(resolve(repo, "scripts/assemble-native-archive.sh"))],
            {
                cwd: repo,
                stdout: "pipe",
                stderr: "pipe",
                env: {
                    ...process.env,
                    ...shellToolEnvironment(f),
                    PATH: [f.tools, process.env.PATH].filter(Boolean).join(delimiter),
                    CRATE: bashPath(crate),
                    STAGE: bashPath(resolve(f.root, "stage")),
                    CARGO_MESSAGES: normalizePath(cargoJson),
                    TARGET: target,
                    MODE: "portable",
                    BINARY: "shallot-native",
                    ARCHIVE: bashPath(output),
                },
            },
        );
        expect(result.exitCode, new TextDecoder().decode(result.stderr)).toBe(0);
        const extracted = resolve(f.root, "archive");
        mkdirSync(extracted);
        await new Bun.Archive(readFileSync(output)).extract(extracted);
        expect(
            readFileSync(
                resolve(
                    extracted,
                    "Chromium Embedded Framework.framework/Chromium Embedded Framework",
                ),
                "utf8",
            ),
        ).toBe("selected framework");
        expect(
            existsSync(
                resolve(
                    extracted,
                    "Chromium Embedded Framework.framework/Chromium Embedded Framework.framework",
                ),
            ),
        ).toBe(false);
        expect(readFileSync(resolve(extracted, "shallot-native"), "utf8")).toBe("shell");

        for (const item of [
            { target: "x86_64-unknown-linux-gnu", binary: "shallot-native", marker: "libcef.so" },
            {
                target: "x86_64-pc-windows-msvc",
                binary: "shallot-native.exe",
                marker: "libcef.dll",
            },
        ]) {
            const platformCrate = resolve(f.root, item.target);
            const platformBuild = resolve(platformCrate, "target", item.target, "release");
            mkdirSync(platformBuild, { recursive: true });
            writeFileSync(resolve(platformBuild, item.binary), `shell ${item.target}`);
            const selectedRuntime = runtimeDir(
                f,
                `archive-${item.target}`,
                item.marker,
                `selected ${item.target}`,
            );
            const oldRuntime = resolve(
                platformCrate,
                "target",
                item.target,
                "release/build/cef-dll-sys-old/out/cef",
            );
            mkdirSync(oldRuntime, { recursive: true });
            writeFileSync(resolve(oldRuntime, item.marker), `stale ${item.target}`);
            writeFileSync(
                cargoJson,
                JSON.stringify({
                    reason: "build-script-executed",
                    ["package_id"]: "path+file:///fixture#shallot-native@0.10.0",
                    env: [["SHALLOT_CEF_DIR", normalizePath(selectedRuntime)]],
                }),
            );
            const platformArchive = resolve(f.root, `${item.target}.tar.gz`);
            const assembled = Bun.spawnSync(
                ["bash", bashPath(resolve(repo, "scripts/assemble-native-archive.sh"))],
                {
                    cwd: repo,
                    stdout: "pipe",
                    stderr: "pipe",
                    env: {
                        ...process.env,
                        ...shellToolEnvironment(f),
                        PATH: [f.tools, process.env.PATH].filter(Boolean).join(delimiter),
                        CRATE: bashPath(platformCrate),
                        STAGE: bashPath(resolve(f.root, `stage-${item.target}`)),
                        CARGO_MESSAGES: normalizePath(cargoJson),
                        TARGET: item.target,
                        MODE: "portable",
                        BINARY: item.binary,
                        ARCHIVE: bashPath(platformArchive),
                    },
                },
            );
            expect(assembled.exitCode, new TextDecoder().decode(assembled.stderr)).toBe(0);
            const platformExtracted = resolve(f.root, `archive-${item.target}`);
            mkdirSync(platformExtracted);
            await new Bun.Archive(readFileSync(platformArchive)).extract(platformExtracted);
            expect(readFileSync(resolve(platformExtracted, "cef", item.marker), "utf8")).toBe(
                `selected ${item.target}`,
            );
        }

        writeFileSync(output, "old archive");
        writeFileSync(cargoJson, "");
        const refused = Bun.spawnSync(
            ["bash", bashPath(resolve(repo, "scripts/assemble-native-archive.sh"))],
            {
                cwd: repo,
                stdout: "pipe",
                stderr: "pipe",
                env: {
                    ...process.env,
                    ...shellToolEnvironment(f),
                    PATH: [f.tools, process.env.PATH].filter(Boolean).join(delimiter),
                    CRATE: bashPath(crate),
                    STAGE: bashPath(resolve(f.root, "stage")),
                    CARGO_MESSAGES: normalizePath(cargoJson),
                    TARGET: target,
                    MODE: "portable",
                    BINARY: "shallot-native",
                    ARCHIVE: bashPath(output),
                },
            },
        );
        expect(refused.exitCode).not.toBe(0);
        expect(existsSync(output)).toBe(false);
    } finally {
        removeTestDirectory(f.root);
    }
});
