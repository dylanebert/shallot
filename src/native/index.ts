import { execSync } from "node:child_process";
import {
    chmodSync,
    cpSync,
    existsSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    rmSync,
    statSync,
    writeFileSync,
} from "node:fs";
import { basename, resolve } from "node:path";
import { tryPrebuilt } from "./prebuilt";
import { cargoRuntimeDir, requireRuntimeDir } from "./provenance.js";

const RUST_CRATE = resolve(import.meta.dir, "../../crates/native");
const DEFAULT_ICON = resolve(import.meta.dir, "../../assets/icon-1024.png");
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
// window. Off by default until validated per platform; opt in with SHALLOT_DROP_SWIFTSHADER set. Read
// per call, not once at module load, so a caller that sets the env var mid-process (a test, a script
// driving multiple builds) is honored rather than frozen at whatever it was on import.
export function dropSwiftshader(): boolean {
    return process.env.SHALLOT_DROP_SWIFTSHADER != null;
}

/**
 * A native build's output dir: `build/<platform>/<profile>-<mode>`, with `-dev` for dev shells.
 * Portable and system shells, and dev and standalone shells, occupy separate directories.
 */
export function nativeOutDir(
    projectDir: string,
    platform: string,
    release: boolean,
    portable: boolean,
    dev = false,
): string {
    const profile = release ? "release" : "debug";
    const mode = portable ? "portable" : "system";
    return resolve(projectDir, "build", platform, `${profile}-${mode}${dev ? "-dev" : ""}`);
}

// cargo's build output root for the crate. The windows-portable WSL build redirects to a local
// Windows dir (winBuildDir) instead; everything else uses this. cargoBuild pins CARGO_TARGET_DIR to
// it, so the path is the same whether the crate builds as a member of the source repo's workspace
// (whose default is the root target/) or standalone from the npm tarball.
const CRATE_TARGET = resolve(RUST_CRATE, "target");

export function cargoTarget(target: string, release: boolean, targetDir = CRATE_TARGET): string {
    const profile = release ? "release" : "debug";
    return resolve(
        targetDir,
        `${target}/${profile}/shallot-native${target.includes("windows") ? ".exe" : ""}`,
    );
}

// running under WSL — the Windows target then builds with the host's native MSVC toolchain through
// PowerShell, not cargo-xwin (see cargoBuild). This describes the *consumer's* machine, not any seat
// this repo is developed on: `shallot native` ships, and a user developing under the Windows Subsystem
// for Linux who builds a Windows portable target reaches the host toolchain through exactly this
// branch, so it stays.
const isWSL = process.env.WSL_INTEROP != null || existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");

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
function devShellBuild(
    flags: string[],
    distDir?: string,
    winTargetDir?: string,
    iconPath?: string,
): string {
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
        `${linker}${setTarget}${setDist}$env:SHALLOT_ICON_ICO='${iconPath ? winPath(iconPath).replaceAll("'", "''") : ""}'; cargo build ${flags.join(" ")}`,
    ].join(" ");
}

export type CargoInvocation =
    | { kind: "devshell"; flags: string[] }
    | { kind: "xwin"; flags: string[] }
    | { kind: "native"; flags: string[] };

/**
 * cargoBuild's portable/target/WSL decision table, pulled out pure: which cargo invocation a
 * (target, release, portable, isWSL) combination selects, and the flags it carries. `devshell` is
 * windows-portable under WSL (needs the host's native MSVC toolchain via PowerShell, see
 * devShellBuild); `xwin` is any other msvc target (cross-compiled with cargo-xwin); `native` is
 * everything else (`cargo build` in place).
 */
export function resolveCargoInvocation(
    target: string,
    release: boolean,
    portable: boolean,
    isWSL: boolean,
): CargoInvocation {
    const flags: string[] = ["--target", target];
    if (portable) flags.push("--no-default-features", "--features", "portable");
    if (release) flags.push("--release");

    const msvc = target.includes("msvc");
    if (msvc && isWSL && portable) return { kind: "devshell", flags };
    return { kind: msvc ? "xwin" : "native", flags };
}

// crates/native ships in the npm tarball (package.json `files` includes `crates/native`), so a missing
// crate dir means a corrupt install or a non-standard layout, not an unsupported path. Guard it
// before spawning cargo, since a raw ENOENT from `cwd: RUST_CRATE` below is an opaque failure.
export function missingCrateDiagnostic(crateDir: string): string | null {
    if (existsSync(crateDir)) return null;
    return `no crates/native crate found at ${crateDir}. The crate ships in the npm package, so this looks like a corrupt install or a non-standard layout. Reinstall @dylanebert/shallot, or build from the source repo.`;
}

function requireRustCrate(): void {
    const msg = missingCrateDiagnostic(RUST_CRATE);
    if (msg) {
        console.error(`  ${msg}`);
        process.exit(1);
    }
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
    iconPath?: string,
): string | null {
    requireRustCrate();
    const invocation = resolveCargoInvocation(target, release, portable, isWSL);

    // Windows portable (CEF) under WSL builds natively on the host via PowerShell — the same
    // WSL→Windows bridge the bench uses. cargo-xwin's clang-cl can't build CEF's libcef_dll_wrapper
    // (no ATL, and CEF's `/MP` trips clang-cl's `/WX`); real cl.exe handles both. The system build
    // stays on cargo-xwin (below) — it needs no C++/VS toolchain, so it works out of the box. cargo
    // runs in-place over the crate's UNC path (devShellBuild cd's there), so its target/ dir is the
    // one bundling reads back. The powershell process is launched with a real Windows cwd (the C:\
    // mount) so the dev-shell's internal cmd.exe doesn't choke on a UNC working directory.
    if (invocation.kind === "devshell") {
        const winCwd = execSync("wslpath -u 'C:\\'", { encoding: "utf-8" }).trim();
        const r = Bun.spawnSync(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                devShellBuild(
                    [...invocation.flags, "--message-format=json-render-diagnostics"],
                    distDir,
                    winTargetDir,
                    iconPath,
                ),
            ],
            { cwd: winCwd, stdout: "pipe", stderr: "inherit" },
        );
        if (r.exitCode !== 0) {
            console.error(
                "  windows portable build failed — the host needs the Rust MSVC toolchain plus Visual Studio (or Build Tools) with the C++ workload incl. ATL, which CEF's wrapper requires",
            );
            process.exit(1);
        }
        const output = new TextDecoder().decode(r.stdout);
        const marker = target.includes("windows")
            ? "libcef.dll"
            : target.includes("darwin")
              ? CEF_FRAMEWORK
              : "libcef.so";
        const translate =
            invocation.kind === "devshell"
                ? (path: string) => execSync(`wslpath -u "${path}"`, { encoding: "utf-8" }).trim()
                : undefined;
        return cargoRuntimeDir(output, marker, translate);
    }

    // cross-compile a Windows target from a non-WSL host (cargo-xwin), or build a native target.
    const cmd = invocation.kind === "xwin" ? "cargo xwin build" : "cargo build";
    const env = {
        ...process.env,
        CARGO_TARGET_DIR: CRATE_TARGET,
        SHALLOT_ICON_ICO: iconPath ?? "",
        ...(distDir ? { SHALLOT_DIST: distDir } : {}),
    };
    const output = execSync(
        [cmd, ...invocation.flags, "--message-format=json-render-diagnostics"].join(" "),
        {
            cwd: RUST_CRATE,
            encoding: "utf8",
            env,
            stdio: ["inherit", "pipe", "inherit"],
        },
    );
    const marker = target.includes("windows")
        ? "libcef.dll"
        : target.includes("darwin")
          ? CEF_FRAMEWORK
          : "libcef.so";
    return cargoRuntimeDir(output, marker);
}

function nativeIcon(projectDir: string, dev: boolean): string {
    const custom = resolve(projectDir, dev ? "public/icon.png" : "dist/icon.png");
    return existsSync(custom) ? custom : DEFAULT_ICON;
}

function stageDevIcon(projectDir: string, outputDir: string, dev: boolean): void {
    if (!dev) return;
    const dir = resolve(outputDir, "dist");
    mkdirSync(dir, { recursive: true });
    cpSync(nativeIcon(projectDir, true), resolve(dir, "icon.png"));
}

function ensureIcon(distDir: string): void {
    const distIcon = resolve(distDir, "icon.png");
    if (!existsSync(distIcon) && existsSync(DEFAULT_ICON)) {
        cpSync(DEFAULT_ICON, distIcon);
    }
}

function projectDist(projectDir: string, dev: boolean): string {
    const distDir = resolve(projectDir, "dist");
    if (!dev && !existsSync(distDir)) {
        console.error("  vite build produced no dist/ directory");
        process.exit(1);
    }
    if (!dev && !existsSync(resolve(distDir, "index.html"))) {
        console.error("  vite build produced no dist/index.html");
        process.exit(1);
    }
    if (!dev) ensureIcon(distDir);
    return distDir;
}

// The runtime path comes from this invocation's Cargo build-script message, not the target cache.

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
    opts: { release?: boolean; portable?: boolean; dev?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;
    const dev = opts.dev ?? false;
    const name = basename(projectDir);

    const distDir = projectDist(projectDir, dev);

    // portable on WSL builds into a local Windows dir (off the 9p share); read its artifacts back
    // through the /mnt view. Every other case uses the in-tree target/.
    console.log(
        `  compiling ${portable ? "CEF" : "webview"} shell (${release ? "release" : "debug"})...`,
    );

    // The prebuilt returns its runtime directory as plain data; a miss captures Cargo's selected result.
    const customIco = resolve(projectDir, "public/icon.ico");
    let iconPath = existsSync(customIco) ? customIco : undefined;
    if (iconPath) {
        const bytes = readFileSync(iconPath);
        if (
            bytes.length < 22 ||
            bytes.readUInt16LE(0) !== 0 ||
            bytes.readUInt16LE(2) !== 1 ||
            bytes.readUInt16LE(4) === 0
        ) {
            throw new Error(`invalid Windows icon: ${iconPath}; supply an ICO file`);
        }
        if (bytes.equals(readFileSync(resolve(RUST_CRATE, "icon.ico")))) iconPath = undefined;
        else
            console.log(
                "  public/icon.ico requires a source build (Rust + rc.exe or windres; target tools: shallot build --help)",
            );
    }
    const prebuilt = iconPath ? null : await tryPrebuilt(WIN_TARGET, release, portable);
    let exe: string;
    let cefDir: string | undefined;
    let targetDir = CRATE_TARGET;
    if (prebuilt) {
        exe = prebuilt.binaryPath;
        cefDir = prebuilt.cefDir;
    } else {
        // portable on WSL builds into a local Windows dir (off the 9p share); read its artifacts back
        // through the /mnt view. Every other case uses the in-tree target/.
        const winBuild = portable && isWSL ? winBuildDir() : null;
        targetDir = winBuild ? winBuild.wsl : CRATE_TARGET;
        cefDir =
            cargoBuild(
                WIN_TARGET,
                release,
                portable,
                release ? distDir : undefined,
                winBuild?.win,
                iconPath,
            ) ?? undefined;
        exe = cargoTarget(WIN_TARGET, release, targetDir);
    }
    if (portable) cefDir = requireRuntimeDir(cefDir, "libcef.dll");
    if (!existsSync(exe)) {
        console.error(`  cargo build produced no exe at ${exe}`);
        process.exit(1);
    }

    if (existsSync(outputDir)) rmSync(outputDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });

    const outExe = resolve(outputDir, `${name}.exe`);
    cpSync(exe, outExe);
    stageDevIcon(projectDir, outputDir, dev);

    // portable ships the Chromium runtime beside the exe (Windows resolves DLLs from the exe dir).
    if (portable) copyCefDlls(outputDir, cefDir!);

    if (!dev && (!release || prebuilt)) {
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

export function macHelperBin(release: boolean): string {
    const profile = release ? "release" : "debug";
    return resolve(RUST_CRATE, `target/${MAC_TARGET}/${profile}/shallot-helper`);
}

export function macInfoPlist(opts: {
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

// CefSettings.locale defaults to en-US (see copyLocale), so only the active locale's forms are kept.
// The two walks need different rules: Resources/ holds the per-locale `.lproj` bundle folders (both
// the regionless and region-qualified English spelling survive) alongside the required CEF resource
// paks (chrome_100_percent.pak, chrome_200_percent.pak, resources.pak) that ship unconditionally —
// so a `.pak` entry is never stale there. Resources/locales/ holds the flat per-locale `.pak` files,
// where only `en-US.pak` survives. Pulled out of trimMacLocales' two directory walks so the delete
// rule is testable without touching a filesystem.
export function isStaleLocaleEntry(entry: string, scope: "bundle" | "locales"): boolean {
    if (entry.endsWith(".lproj")) return entry !== "en.lproj" && entry !== "en_US.lproj";
    if (scope === "locales" && entry.endsWith(".pak")) return entry !== "en-US.pak";
    return false;
}

// strip non-English locale paks from a copied CEF mac framework — covers both the per-locale `.lproj`
// layout and a flat `locales/` dir of paks, so it's a no-op if neither is present. Runs before
// codesign (which re-signs the trimmed tree).
function trimMacLocales(frameworkDir: string): void {
    const resources = resolve(frameworkDir, "Resources");
    if (!existsSync(resources)) return;
    for (const entry of readdirSync(resources)) {
        if (isStaleLocaleEntry(entry, "bundle")) {
            rmSync(resolve(resources, entry), { recursive: true, force: true });
        }
    }
    const locales = resolve(resources, "locales");
    if (existsSync(locales)) {
        for (const entry of readdirSync(locales)) {
            if (isStaleLocaleEntry(entry, "locales")) {
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
    opts: { release?: boolean; portable?: boolean; dev?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;
    const dev = opts.dev ?? false;
    const name = basename(projectDir);

    const distDir = projectDist(projectDir, dev);

    console.log(
        `  compiling ${portable ? "CEF" : "webview"} shell (${release ? "release" : "debug"})...`,
    );

    // The prebuilt returns its runtime directory as plain data; a miss captures Cargo's selected result.
    const prebuilt = await tryPrebuilt(MAC_TARGET, release, portable);
    let bin: string;
    let cefDir = prebuilt?.cefDir;
    if (prebuilt) {
        bin = prebuilt.binaryPath;
    } else {
        cefDir =
            cargoBuild(MAC_TARGET, release, portable, release ? distDir : undefined) ?? undefined;
        bin = cargoTarget(MAC_TARGET, release);
    }
    const runtimeDir = portable ? requireRuntimeDir(cefDir, CEF_FRAMEWORK) : undefined;
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

    const identifier = `com.shallot.${name}`;
    writeFileSync(
        resolve(contentsDir, "Info.plist"),
        macInfoPlist({ executable: name, bundleName: name, identifier, helper: false, icon: true }),
    );

    const outBin = resolve(macosDir, name);
    cpSync(bin, outBin);
    chmodSync(outBin, 0o755);

    convertIconToIcns(nativeIcon(projectDir, dev), resolve(resourcesDir, "app.icns"));
    stageDevIcon(projectDir, resourcesDir, dev);

    if (portable) {
        const helperBin = prebuilt?.helperPath ?? macHelperBin(release);
        if (!existsSync(helperBin)) {
            console.error(`  cargo build produced no helper binary at ${helperBin}`);
            process.exit(1);
        }
        const frameworksDir = resolve(contentsDir, "Frameworks");
        mkdirSync(frameworksDir, { recursive: true });

        const fwOut = resolve(frameworksDir, CEF_FRAMEWORK);
        // ditto is the canonical tool for copying a macOS framework (preserves symlinks, perms, xattrs).
        execSync(`ditto "${resolve(runtimeDir!, CEF_FRAMEWORK)}" "${fwOut}"`, { stdio: "pipe" });

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

    // Disk assets live in Resources/, not MacOS/ — codesign treats everything under MacOS/ as nested
    // code and rejects the bundle on the first non-Mach-O asset; Resources is where bundled data belongs.
    if (!dev && (!release || prebuilt)) {
        cpSync(distDir, resolve(resourcesDir, "dist"), { recursive: true });
    }

    // macOS requires valid signatures to launch on Apple Silicon; ad-hoc sign. Portable has a nested
    // framework + helper tree, so --deep signs inside-out; the plain default .app has none.
    const deep = portable ? "--deep " : "";
    execSync(`codesign --force ${deep}--sign - "${appDir}"`, { stdio: "pipe" });

    const appSize = (statSync(outBin).size / 1024 / 1024).toFixed(1);
    console.log(`  ${name}.app (${appSize} MB binary)`);
}

function copyCefLibs(outputDir: string, cefSrc: string, release: boolean): void {
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
    if (!dropSwiftshader()) files.push("libvk_swiftshader.so");

    for (const file of files) {
        const src = resolve(cefSrc, file);
        if (existsSync(src)) cpSync(src, resolve(cefOut, file));
    }

    copyLocale(resolve(cefSrc, "locales"), resolve(cefOut, "locales"));

    if (!dropSwiftshader()) {
        const swiftshaderSrc = resolve(cefSrc, "swiftshader");
        if (existsSync(swiftshaderSrc)) {
            cpSync(swiftshaderSrc, resolve(cefOut, "swiftshader"), { recursive: true });
        }
    }

    if (release) tryStrip(resolve(cefOut, "libcef.so"));

    console.log(`  copied CEF libs from ${cefSrc}`);
}

function copyCefDlls(outputDir: string, cefSrc: string): void {
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
    if (!dropSwiftshader()) files.push("vk_swiftshader.dll", "vk_swiftshader_icd.json");

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
    opts: { release?: boolean; portable?: boolean; dev?: boolean },
): Promise<void> {
    const release = opts.release ?? false;
    const portable = opts.portable ?? false;
    const dev = opts.dev ?? false;
    const name = basename(projectDir);

    const distDir = projectDist(projectDir, dev);

    console.log(
        `  compiling ${portable ? "CEF" : "webview"} shell (${release ? "release" : "debug"})...`,
    );

    // The prebuilt returns its runtime directory as plain data; a miss captures Cargo's selected result.
    const prebuilt = await tryPrebuilt(LINUX_TARGET, release, portable);
    let bin: string;
    let cefDir = prebuilt?.cefDir;
    if (prebuilt) {
        bin = prebuilt.binaryPath;
    } else {
        cefDir =
            cargoBuild(LINUX_TARGET, release, portable, release ? distDir : undefined) ?? undefined;
        bin = cargoTarget(LINUX_TARGET, release);
    }
    if (portable) cefDir = requireRuntimeDir(cefDir, "libcef.so");
    if (!existsSync(bin)) {
        console.error(`  cargo build produced no binary at ${bin}`);
        process.exit(1);
    }

    if (existsSync(outputDir)) rmSync(outputDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });

    const outBin = resolve(outputDir, name);
    cpSync(bin, outBin);
    chmodSync(outBin, 0o755);
    stageDevIcon(projectDir, outputDir, dev);

    // portable ships the Chromium runtime; the default depends on the host's WebKitGTK (no copy).
    if (portable) copyCefLibs(outputDir, cefDir!, release);

    if (!dev && (!release || prebuilt)) {
        cpSync(distDir, resolve(outputDir, "dist"), { recursive: true });
    }
}
