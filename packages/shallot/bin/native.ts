import { execSync } from "node:child_process";
import {
    chmodSync,
    cpSync,
    existsSync,
    mkdirSync,
    readdirSync,
    rmSync,
    statSync,
    writeFileSync,
} from "node:fs";
import { basename, resolve } from "node:path";
import { buildWeb } from "./build";

const RUST_CRATE = resolve(import.meta.dir, "../rust/window");
const DEFAULT_ICON = resolve(import.meta.dir, "../../../assets/icon-1024.png");
const WIN_TARGET = "x86_64-pc-windows-msvc";
const MAC_TARGET = "aarch64-apple-darwin";
const LINUX_TARGET = "x86_64-unknown-linux-gnu";
const CEF_FRAMEWORK = "Chromium Embedded Framework.framework";
// CEF derives helper bundle names from the main executable name + these suffixes; they must match
// exactly so the framework can find its subprocess executables.
const MAC_HELPER_SUFFIXES = [
    "Helper",
    "Helper (GPU)",
    "Helper (Renderer)",
    "Helper (Plugin)",
    "Helper (Alerts)",
];

// SwiftShader is Chromium's software-GL fallback. WebGPU runs on Vulkan/Metal/D3D12 directly, so a
// real-GPU target doesn't need it — but the Chromium compositor may, so dropping it risks a black
// window. Off by default until validated per platform; opt in with SHALLOT_DROP_SWIFTSHADER set.
const DROP_SWIFTSHADER = process.env.SHALLOT_DROP_SWIFTSHADER != null;

/**
 * a native build's output dir: `build/<platform>/<profile>-<mode>`. The mode segment keeps a portable
 * (CEF) and a system-webview build of the same project + profile in separate dirs (they'd otherwise
 * share a path and clobber).
 */
export function nativeOutDir(
    projectDir: string,
    platform: string,
    release: boolean,
    portable: boolean,
): string {
    const profile = release ? "release" : "debug";
    const mode = portable ? "portable" : "system";
    return resolve(projectDir, "build", platform, `${profile}-${mode}`);
}

// cargo's build output root for the crate. The windows-portable WSL build redirects to a local
// Windows dir (winBuildDir) instead; everything else uses this.
const CRATE_TARGET = resolve(RUST_CRATE, "target");

function cargoTarget(target: string, release: boolean, targetDir = CRATE_TARGET): string {
    const profile = release ? "release" : "debug";
    return resolve(
        targetDir,
        `${target}/${profile}/shallot-window${target.includes("windows") ? ".exe" : ""}`,
    );
}

// running under WSL — the Windows target then builds with the host's native MSVC toolchain through
// PowerShell, not cargo-xwin (see cargoBuild). Same probe the harness uses (harness/core/wsl.ts).
const isWSL = existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

// translate a WSL path to its Windows form (\\wsl.localhost\… UNC) so PowerShell can `cd` into it.
function winPath(p: string): string {
    return execSync(`wslpath -w "${p}"`, { encoding: "utf-8" }).trim();
}

// the windows-portable build can't write to the crate's target/ over the 9p UNC share — rust's
// incremental lock and the CEF cmake build fail there. It builds into this local Windows dir
// (under LOCALAPPDATA) instead: fast, lock-friendly, cached across builds. Returns the Windows path
// (for CARGO_TARGET_DIR) and its WSL /mnt view (for reading the artifacts back).
function winBuildDir(): { win: string; wsl: string } {
    const r = Bun.spawnSync(
        ["powershell.exe", "-NoProfile", "-Command", "[Console]::Write($env:LOCALAPPDATA)"],
        { stdout: "pipe" },
    );
    const local = new TextDecoder().decode(r.stdout).trim().replace(/\r/g, "");
    const win = `${local}\\shallot\\winbuild`;
    const wsl = execSync(`wslpath -u "${win}"`, { encoding: "utf-8" }).trim();
    return { win, wsl };
}

// vswhere ships at this fixed location on every VS install; documented by Microsoft.
const VSWHERE = "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe";

// PowerShell that enters the VS Developer environment (PATH + INCLUDE/LIB for cl.exe / link.exe /
// ATL) before running `cmd`, then cd's to the crate's UNC path to build. The dev-shell setup spawns
// cmd.exe internally, which can't run from a UNC working directory — so the powershell process is
// launched with a real Windows cwd (see the spawn `cwd` below) and only switches to the UNC path after.
function devShellBuild(flags: string[], distDir?: string, winTargetDir?: string): string {
    const setDist = distDir ? `$env:SHALLOT_DIST='${winPath(distDir)}'; ` : "";
    // build into a local Windows dir, not the crate's target/ on the 9p share (see winBuildDir).
    const setTarget = winTargetDir ? `$env:CARGO_TARGET_DIR='${winTargetDir}'; ` : "";
    // .cargo/config.toml pins the linker to lld-link for cargo-xwin's cross builds; the native MSVC
    // toolchain ships link.exe, not lld-link, so override back to it (env beats config in cargo).
    const linker = "$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER='link.exe'; ";
    return [
        `$p=(& '${VSWHERE}' -products * -all -latest -property installationPath | Select-Object -First 1);`,
        `if(-not $p){Write-Error 'Visual Studio with the C++ workload (incl. ATL) not found'; exit 1};`,
        `Import-Module (Join-Path $p 'Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll');`,
        `Enter-VsDevShell -VsInstallPath $p -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null;`,
        `Set-Location '${winPath(RUST_CRATE)}';`,
        `${linker}${setTarget}${setDist}cargo build ${flags.join(" ")}`,
    ].join(" ");
}

// SHALLOT_DIST bakes the web assets into the binary at compile time (release only) so the exe carries
// no appended overlay and never extracts itself at runtime. Debug serves from the sibling dist/ for
// fast iteration. `portable` selects the CEF backend (self-contained Chromium) over the default
// system webview; it drops the default `system` feature so the artifact never links wry. See build.rs.
function cargoBuild(
    target: string,
    release: boolean,
    portable: boolean,
    distDir?: string,
    winTargetDir?: string,
): void {
    const flags: string[] = ["--target", target];
    if (portable) flags.push("--no-default-features", "--features", "portable");
    if (release) flags.push("--release");

    // Windows portable (CEF) under WSL builds natively on the host via PowerShell — the same
    // WSL→Windows bridge the bench uses. cargo-xwin's clang-cl can't build CEF's libcef_dll_wrapper
    // (no ATL, and CEF's `/MP` trips clang-cl's `/WX`); real cl.exe handles both. The system build
    // stays on cargo-xwin (below) — it needs no C++/VS toolchain, so it works out of the box. cargo
    // runs in-place over the crate's UNC path (devShellBuild cd's there), so its target/ dir is the
    // one bundling reads back. The powershell process is launched with a real Windows cwd (the C:\
    // mount) so the dev-shell's internal cmd.exe doesn't choke on a UNC working directory.
    if (target.includes("msvc") && isWSL && portable) {
        const winCwd = execSync("wslpath -u 'C:\\'", { encoding: "utf-8" }).trim();
        const r = Bun.spawnSync(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                devShellBuild(flags, distDir, winTargetDir),
            ],
            { cwd: winCwd, stdout: "inherit", stderr: "inherit" },
        );
        if (r.exitCode !== 0) {
            console.error(
                "  windows portable build failed — the host needs the Rust MSVC toolchain plus Visual Studio (or Build Tools) with the C++ workload incl. ATL, which CEF's wrapper requires",
            );
            process.exit(1);
        }
        return;
    }

    // cross-compile a Windows target from a non-WSL host (cargo-xwin), or build a native target.
    const cmd = target.includes("msvc") ? "cargo xwin build" : "cargo build";
    const env = distDir ? { ...process.env, SHALLOT_DIST: distDir } : process.env;
    execSync([cmd, ...flags].join(" "), { cwd: RUST_CRATE, stdio: "inherit", env });
}

function ensureIcon(distDir: string): void {
    const distIcon = resolve(distDir, "icon.png");
    if (!existsSync(distIcon) && existsSync(DEFAULT_ICON)) {
        cpSync(DEFAULT_ICON, distIcon);
    }
}

// the downloaded CEF runtime lives in cef-dll-sys's build OUT_DIR (or CEF_PATH when set). Return the
// subdir under it holding `marker` (libcef.so / libcef.dll / the framework) for a target's build.
function findCefDir(target: string, marker: string, targetDir = CRATE_TARGET): string | null {
    const cefPath = process.env.CEF_PATH;
    if (cefPath && existsSync(resolve(cefPath, marker))) return cefPath;

    const profile = resolve(targetDir, target);
    for (const dir of ["release", "debug"]) {
        const buildDir = resolve(profile, dir, "build");
        if (!existsSync(buildDir)) continue;
        for (const entry of readdirSync(buildDir)) {
            if (!entry.startsWith("cef-dll-sys-")) continue;
            const out = resolve(buildDir, entry, "out");
            if (!existsSync(out)) continue;
            for (const sub of readdirSync(out)) {
                if (existsSync(resolve(out, sub, marker))) return resolve(out, sub);
            }
        }
    }
    return null;
}

// strip is the single biggest CEF size win — the Spotify builds ship libcef unstripped. Skips when the
// tool is absent (a non-stripped build still works, just larger). Mac framework binaries pass "-x -S"
// to keep external symbols so the dylib still loads + can be re-signed.
function tryStrip(file: string, args = ""): void {
    if (!existsSync(file)) return;
    try {
        execSync(`strip ${args} "${file}"`, { stdio: "pipe" });
    } catch {
        console.warn(`  strip unavailable — ${basename(file)} not size-minimized`);
    }
}

// ship only the active locale's pak (CefSettings.locale defaults to en-US); the other ~50 are dead
// weight. Officially supported — the locales dir is optional beyond the active locale.
function copyLocale(srcLocales: string, destLocales: string): void {
    const en = resolve(srcLocales, "en-US.pak");
    if (!existsSync(en)) return;
    mkdirSync(destLocales, { recursive: true });
    cpSync(en, resolve(destLocales, "en-US.pak"));
}

export async function bundleNativeWindows(
    projectDir: string,
    outputDir: string,
    opts: { release?: boolean; portable?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;
    const name = basename(projectDir);

    await buildWeb(projectDir);

    const distDir = resolve(projectDir, "dist");
    if (!existsSync(distDir)) {
        console.error("  vite build produced no dist/ directory");
        process.exit(1);
    }
    ensureIcon(distDir);

    // portable on WSL builds into a local Windows dir (off the 9p share); read its artifacts back
    // through the /mnt view. Every other case uses the in-tree target/.
    const winBuild = portable && isWSL ? winBuildDir() : null;
    const targetDir = winBuild ? winBuild.wsl : CRATE_TARGET;

    console.log(
        `  compiling ${portable ? "CEF" : "webview"} shell (${release ? "release" : "debug"})...`,
    );
    cargoBuild(WIN_TARGET, release, portable, release ? distDir : undefined, winBuild?.win);

    const exe = cargoTarget(WIN_TARGET, release, targetDir);
    if (!existsSync(exe)) {
        console.error(`  cargo build produced no exe at ${exe}`);
        process.exit(1);
    }

    if (existsSync(outputDir)) rmSync(outputDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });

    const outExe = resolve(outputDir, `${name}.exe`);
    cpSync(exe, outExe);

    // portable ships the Chromium runtime beside the exe (Windows resolves DLLs from the exe dir).
    if (portable) copyCefDlls(outputDir, targetDir);

    if (!release) {
        cpSync(distDir, resolve(outputDir, "dist"), { recursive: true });
    }

    const sizeMB = (statSync(outExe).size / 1024 / 1024).toFixed(1);
    console.log(`  ${name}.exe: ${sizeMB} MB`);
}

function prepareMacIcon(pngPath: string): string {
    const tmp = resolve(pngPath + ".mac.png");
    const inset = Math.round(1024 * 0.8);
    cpSync(pngPath, tmp);
    execSync(`sips -z ${inset} ${inset} "${tmp}"`, { stdio: "pipe" });
    execSync(`sips --padToHeightWidth 1024 1024 --padColor 1F1E1D "${tmp}"`, { stdio: "pipe" });
    return tmp;
}

function convertIconToIcns(pngPath: string, icnsPath: string): void {
    const macIcon = prepareMacIcon(pngPath);
    const iconsetDir = resolve(icnsPath + ".iconset");
    mkdirSync(iconsetDir, { recursive: true });

    const sizes = [16, 32, 128, 256, 512];
    for (const size of sizes) {
        execSync(
            `sips -z ${size} ${size} "${macIcon}" --out "${iconsetDir}/icon_${size}x${size}.png"`,
            {
                stdio: "pipe",
            },
        );
        const double = size * 2;
        execSync(
            `sips -z ${double} ${double} "${macIcon}" --out "${iconsetDir}/icon_${size}x${size}@2x.png"`,
            {
                stdio: "pipe",
            },
        );
    }

    execSync(`iconutil -c icns "${iconsetDir}" -o "${icnsPath}"`, { stdio: "pipe" });
    rmSync(iconsetDir, { recursive: true });
    rmSync(macIcon);
}

function macHelperBin(release: boolean): string {
    const profile = release ? "release" : "debug";
    return resolve(RUST_CRATE, `target/${MAC_TARGET}/${profile}/shallot-helper`);
}

function macInfoPlist(opts: {
    executable: string;
    bundleName: string;
    identifier: string;
    helper: boolean;
    icon: boolean;
}): string {
    const iconKey = opts.icon ? "\n    <key>CFBundleIconFile</key>\n    <string>app</string>" : "";
    // LSUIElement keeps helper processes out of the Dock and Cmd+Tab.
    const uiElement = opts.helper ? "\n    <key>LSUIElement</key>\n    <true/>" : "";
    return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>${opts.executable}</string>
    <key>CFBundleIdentifier</key>
    <string>${opts.identifier}</string>
    <key>CFBundleName</key>
    <string>${opts.bundleName}</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>${iconKey}
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>${uiElement}
</dict>
</plist>`;
}

// strip non-English locale paks from a copied CEF mac framework — covers both the per-locale `.lproj`
// layout and a flat `locales/` dir of paks, so it's a no-op if neither is present. Runs before
// codesign (which re-signs the trimmed tree).
function trimMacLocales(frameworkDir: string): void {
    const resources = resolve(frameworkDir, "Resources");
    if (!existsSync(resources)) return;
    for (const entry of readdirSync(resources)) {
        if (entry.endsWith(".lproj") && entry !== "en.lproj" && entry !== "en_US.lproj") {
            rmSync(resolve(resources, entry), { recursive: true, force: true });
        }
    }
    const locales = resolve(resources, "locales");
    if (existsSync(locales)) {
        for (const entry of readdirSync(locales)) {
            if (entry.endsWith(".pak") && entry !== "en-US.pak") {
                rmSync(resolve(locales, entry), { force: true });
            }
        }
    }
}

// macOS portable ships the Chromium runtime via CEF: the .app holds the framework plus five helper
// sub-apps (CEF runs renderer/GPU/etc. as separate processes), auto-discovered from the standard
// bundle layout. The default (system-webview) build is a plain .app over the wry binary — no framework,
// no helpers — that uses WKWebView. Both ad-hoc sign so they launch on Apple Silicon.
export async function bundleNativeMac(
    projectDir: string,
    outputDir: string,
    opts: { release?: boolean; portable?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;
    const name = basename(projectDir);

    await buildWeb(projectDir);

    const distDir = resolve(projectDir, "dist");
    if (!existsSync(distDir)) {
        console.error("  vite build produced no dist/ directory");
        process.exit(1);
    }
    ensureIcon(distDir);

    console.log(
        `  compiling ${portable ? "CEF" : "webview"} shell (${release ? "release" : "debug"})...`,
    );
    cargoBuild(MAC_TARGET, release, portable, release ? distDir : undefined);

    const bin = cargoTarget(MAC_TARGET, release);
    if (!existsSync(bin)) {
        console.error(`  cargo build produced no binary at ${bin}`);
        process.exit(1);
    }

    if (existsSync(outputDir)) rmSync(outputDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });

    const appDir = resolve(outputDir, `${name}.app`);
    const contentsDir = resolve(appDir, "Contents");
    const macosDir = resolve(contentsDir, "MacOS");
    const resourcesDir = resolve(contentsDir, "Resources");
    mkdirSync(macosDir, { recursive: true });
    mkdirSync(resourcesDir, { recursive: true });

    const identifier = `com.multiplekex.${name}`;
    writeFileSync(
        resolve(contentsDir, "Info.plist"),
        macInfoPlist({ executable: name, bundleName: name, identifier, helper: false, icon: true }),
    );

    const outBin = resolve(macosDir, name);
    cpSync(bin, outBin);
    chmodSync(outBin, 0o755);

    const distIcon = resolve(distDir, "icon.png");
    if (existsSync(distIcon)) {
        convertIconToIcns(distIcon, resolve(resourcesDir, "app.icns"));
    }

    if (portable) {
        const helperBin = macHelperBin(release);
        if (!existsSync(helperBin)) {
            console.error(`  cargo build produced no helper binary at ${helperBin}`);
            process.exit(1);
        }
        const cefDir = findCefDir(MAC_TARGET, CEF_FRAMEWORK);
        if (!cefDir) {
            console.error("  CEF framework not found. Set CEF_PATH or build first.");
            process.exit(1);
        }

        const frameworksDir = resolve(contentsDir, "Frameworks");
        mkdirSync(frameworksDir, { recursive: true });

        const fwOut = resolve(frameworksDir, CEF_FRAMEWORK);
        // ditto is the canonical tool for copying a macOS framework (preserves symlinks, perms, xattrs).
        execSync(`ditto "${resolve(cefDir, CEF_FRAMEWORK)}" "${fwOut}"`, { stdio: "pipe" });

        for (const suffix of MAC_HELPER_SUFFIXES) {
            const helperName = `${name} ${suffix}`;
            const helperContents = resolve(frameworksDir, `${helperName}.app`, "Contents");
            const helperMacos = resolve(helperContents, "MacOS");
            mkdirSync(helperMacos, { recursive: true });
            writeFileSync(
                resolve(helperContents, "Info.plist"),
                macInfoPlist({
                    executable: helperName,
                    bundleName: helperName,
                    identifier,
                    helper: true,
                    icon: false,
                }),
            );
            const outHelper = resolve(helperMacos, helperName);
            cpSync(helperBin, outHelper);
            chmodSync(outHelper, 0o755);
        }

        // size trims must run before codesign — stripping/pruning invalidates a signature.
        trimMacLocales(fwOut);
        if (release) tryStrip(resolve(fwOut, "Chromium Embedded Framework"), "-x -S");
    }

    // Debug stages dist/ on disk so asset edits show without a recompile (see rust/window/src/main.rs).
    // It goes in Resources/, not MacOS/ — codesign treats everything under MacOS/ as nested code and
    // rejects the bundle on the first non-Mach-O asset; Resources is where bundled data belongs.
    if (!release) {
        cpSync(distDir, resolve(resourcesDir, "dist"), { recursive: true });
    }

    // macOS requires valid signatures to launch on Apple Silicon; ad-hoc sign. Portable has a nested
    // framework + helper tree, so --deep signs inside-out; the plain default .app has none.
    const deep = portable ? "--deep " : "";
    execSync(`codesign --force ${deep}--sign - "${appDir}"`, { stdio: "pipe" });

    const appSize = (statSync(outBin).size / 1024 / 1024).toFixed(1);
    console.log(`  ${name}.app (${appSize} MB binary)`);
}

function copyCefLibs(outputDir: string, release: boolean): void {
    const cefSrc = findCefDir(LINUX_TARGET, "libcef.so");
    if (!cefSrc) {
        console.error("  CEF libs not found. Set CEF_PATH or build first.");
        process.exit(1);
    }

    const cefOut = resolve(outputDir, "cef");
    mkdirSync(cefOut, { recursive: true });

    const files = [
        "libcef.so",
        "libEGL.so",
        "libGLESv2.so",
        "libvulkan.so.1",
        "icudtl.dat",
        "v8_context_snapshot.bin",
        "chrome-sandbox",
        "chrome_100_percent.pak",
        "chrome_200_percent.pak",
        "resources.pak",
    ];
    if (!DROP_SWIFTSHADER) files.push("libvk_swiftshader.so");

    for (const file of files) {
        const src = resolve(cefSrc, file);
        if (existsSync(src)) cpSync(src, resolve(cefOut, file));
    }

    copyLocale(resolve(cefSrc, "locales"), resolve(cefOut, "locales"));

    if (!DROP_SWIFTSHADER) {
        const swiftshaderSrc = resolve(cefSrc, "swiftshader");
        if (existsSync(swiftshaderSrc)) {
            cpSync(swiftshaderSrc, resolve(cefOut, "swiftshader"), { recursive: true });
        }
    }

    if (release) tryStrip(resolve(cefOut, "libcef.so"));

    console.log(`  copied CEF libs from ${cefSrc}`);
}

function copyCefDlls(outputDir: string, targetDir = CRATE_TARGET): void {
    const cefSrc = findCefDir(WIN_TARGET, "libcef.dll", targetDir);
    if (!cefSrc) {
        console.error("  CEF libs not found. Set CEF_PATH or build first.");
        process.exit(1);
    }

    // Windows resolves DLLs beside the exe, so the runtime sits in the output dir directly (the window
    // binary's find_cef_dir checks the exe dir first). No strip step — libcef.dll ships without its PDB.
    const files = [
        "libcef.dll",
        "chrome_elf.dll",
        "d3dcompiler_47.dll",
        "dxcompiler.dll",
        "dxil.dll",
        "libEGL.dll",
        "libGLESv2.dll",
        "vulkan-1.dll",
        "icudtl.dat",
        "snapshot_blob.bin",
        "v8_context_snapshot.bin",
        "chrome_100_percent.pak",
        "chrome_200_percent.pak",
        "resources.pak",
    ];
    if (!DROP_SWIFTSHADER) files.push("vk_swiftshader.dll", "vk_swiftshader_icd.json");

    for (const file of files) {
        const src = resolve(cefSrc, file);
        if (existsSync(src)) cpSync(src, resolve(outputDir, file));
    }

    copyLocale(resolve(cefSrc, "locales"), resolve(outputDir, "locales"));

    console.log(`  copied CEF libs from ${cefSrc}`);
}

export async function bundleNativeLinux(
    projectDir: string,
    outputDir: string,
    opts: { release?: boolean; portable?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;
    const name = basename(projectDir);

    await buildWeb(projectDir);

    const distDir = resolve(projectDir, "dist");
    if (!existsSync(distDir)) {
        console.error("  vite build produced no dist/ directory");
        process.exit(1);
    }
    ensureIcon(distDir);

    console.log(
        `  compiling ${portable ? "CEF" : "webview"} shell (${release ? "release" : "debug"})...`,
    );
    cargoBuild(LINUX_TARGET, release, portable, release ? distDir : undefined);

    const bin = cargoTarget(LINUX_TARGET, release);
    if (!existsSync(bin)) {
        console.error(`  cargo build produced no binary at ${bin}`);
        process.exit(1);
    }

    if (existsSync(outputDir)) rmSync(outputDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });

    const outBin = resolve(outputDir, name);
    cpSync(bin, outBin);
    chmodSync(outBin, 0o755);

    // portable ships the Chromium runtime; the default depends on the host's WebKitGTK (no copy).
    if (portable) copyCefLibs(outputDir, release);

    if (!release) {
        cpSync(distDir, resolve(outputDir, "dist"), { recursive: true });
    }
}
