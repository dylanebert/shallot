import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
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
import { homedir } from "node:os";
import { basename, isAbsolute, relative, resolve } from "node:path";
import { normalize } from "../src/project/manifest";
import { manifestPath } from "../src/project/vite";
import { buildWeb } from "./build";

const RUST_CRATE = resolve(import.meta.dir, "../rust/window");
const DEFAULT_ICON = resolve(import.meta.dir, "../assets/icon-1024.png");
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

/** the bundle identifier for a native build: the manifest's `identifier` field, or the default
 *  `com.shallot.<basename>` when omitted. */
export function bundleIdentifier(projectDir: string, name: string): string {
    try {
        const manifest = normalize(readFileSync(manifestPath(projectDir), "utf-8"));
        return manifest.identifier ?? `com.shallot.${name}`;
    } catch {
        return `com.shallot.${name}`;
    }
}

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

export function cargoTarget(target: string, release: boolean, targetDir = CRATE_TARGET): string {
    const profile = release ? "release" : "debug";
    return resolve(
        targetDir,
        `${target}/${profile}/shallot-window${target.includes("windows") ? ".exe" : ""}`,
    );
}

// running under WSL — the Windows target then builds with the host's native MSVC toolchain through
// PowerShell, not cargo-xwin (see cargoBuild). Same probe the eval harness uses (evals/harness/wsl.ts).
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

// --- Prebuilt shell resolution ---
//
// Prebuilts layer over the lazy path, never replace it. A release-profile shell published to GitHub
// Releases (tag v<version>) is downloaded + SHA256-verified + extracted into a cache dir; any miss
// (404, offline, checksum mismatch, source checkout) falls through to cargoBuild with a one-line
// note. Debug stays lazy — dev-iteration machines have the toolchain.
//
// SHA256SUMS ships from the same release as the archive, so the checksum defends transport
// corruption, not a compromised release (an attacker who can publish assets can publish a matching
// SHA256SUMS).

export type PrebuiltMode = "system" | "portable";

export type PrebuiltFetchResult =
    | "ok"
    | "not-found"
    | "offline"
    | "checksum-mismatch"
    | "source-checkout"
    | "extract-failed"
    | "cache-error";

export interface PrebuiltDecision {
    decision: "prebuilt" | "lazy";
    reason: string;
}

/**
 * Pure decision table for prebuilt vs lazy: given the version, target, mode, cache state, and fetch
 * result, decide whether to use a prebuilt shell or fall back to cargo. Pure — no network, no
 * filesystem — so the fallback arms (not-found, checksum-mismatch, offline) are testable without
 * touching either. The caller gathers the facts (check cache, attempt fetch) and feeds them in.
 */
export function resolvePrebuiltDecision(
    version: string | null,
    target: string,
    mode: PrebuiltMode,
    cacheHit: boolean,
    fetchResult: PrebuiltFetchResult | null,
): PrebuiltDecision {
    // target and mode are part of the input space the table maps over, but the prebuilt-vs-lazy
    // decision doesn't vary by them — a hit is a hit regardless of target/mode. They're kept in the
    // signature so the table's contract is explicit and callers don't need a separate lookup.
    void target;
    void mode;
    if (version === null)
        return {
            decision: "lazy",
            reason: "source checkout — prebuilts are keyed to published versions",
        };
    if (cacheHit) return { decision: "prebuilt", reason: "cache hit" };
    if (fetchResult === null) return { decision: "lazy", reason: "no fetch attempted" };
    switch (fetchResult) {
        case "ok":
            return { decision: "prebuilt", reason: "downloaded + SHA256 verified" };
        case "not-found":
            return { decision: "lazy", reason: "no matching release for this version" };
        case "offline":
            return { decision: "lazy", reason: "offline — could not reach GitHub" };
        case "checksum-mismatch":
            return { decision: "lazy", reason: "checksum mismatch — archive may be corrupted" };
        case "source-checkout":
            return {
                decision: "lazy",
                reason: "source checkout — no published release to download",
            };
        case "extract-failed":
            return {
                decision: "lazy",
                reason: "extract failed — archive may be corrupt or disk full",
            };
        case "cache-error":
            return { decision: "lazy", reason: "cache error — could not write to cache dir" };
    }
}

// Archive contract (pinned in the spec's Approach): one shallot-window-<target>-<mode>.tar.gz per
// matrix cell, plus a top-level SHA256SUMS asset covering all archives. URL derived from the version
// tag — no new version site for check-versions.ts.
const GITHUB_RELEASES = "https://github.com/dylanebert/shallot/releases/download";

export function prebuiltArchiveName(target: string, mode: PrebuiltMode): string {
    return `shallot-window-${target}-${mode}.tar.gz`;
}

export function prebuiltUrl(version: string, target: string, mode: PrebuiltMode): string {
    return `${GITHUB_RELEASES}/v${version}/${prebuiltArchiveName(target, mode)}`;
}

export function prebuiltSha256SumsUrl(version: string): string {
    return `${GITHUB_RELEASES}/v${version}/SHA256SUMS`;
}

// Cache: ~/.cache/shallot/prebuilt/<version>/<target>-<mode>/ (XDG_CACHE_HOME-aware; LOCALAPPDATA on
// Windows). Read per call so a test that sets env vars mid-process is honored.
function cacheBaseDir(): string {
    if (process.platform === "win32") {
        return process.env.LOCALAPPDATA ?? resolve(homedir(), "AppData", "Local");
    }
    return process.env.XDG_CACHE_HOME ?? resolve(homedir(), ".cache");
}

export function prebuiltCacheDir(version: string, target: string, mode: PrebuiltMode): string {
    return resolve(cacheBaseDir(), "shallot", "prebuilt", version, `${target}-${mode}`);
}

export function sha256Hex(buf: Buffer): string {
    return createHash("sha256").update(buf).digest("hex");
}

/** Parse a GNU-coreutils-style SHA256SUMS file for the hash matching `archiveName`. */
export function parseSha256Sums(sums: string, archiveName: string): string | null {
    for (const line of sums.split("\n")) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 2 && parts[1].replace(/^\*/, "") === archiveName) return parts[0];
    }
    return null;
}

/** Extract a .tar.gz archive into `destDir` (creates it if needed). Uses the system `tar`. */
export function extractTarGz(archivePath: string, destDir: string): void {
    mkdirSync(destDir, { recursive: true });
    // Containment guard (F4): reject archive entries that would land outside destDir — absolute
    // paths or `..` traversal. This defends a *malformed archive our own CI could produce*, not a
    // compromised release: under a compromised release the executed binary is the strictly greater
    // threat, and the archive is SHA256-verified against a same-origin SHA256SUMS (see the comment
    // at the comparison in tryPrebuilt) which blocks transport corruption but not a malicious
    // publisher. We list entries and refuse the set before extracting.
    const entries = execSync(`tar -tzf "${archivePath}"`, { encoding: "utf8" })
        .split("\n")
        .filter(Boolean);
    for (const entry of entries) {
        const cleaned = entry.replace(/^\.\//, "").replace(/\/$/, "");
        const target = resolve(destDir, cleaned);
        const rel = relative(destDir, target);
        if (rel.startsWith("..") || isAbsolute(rel)) {
            throw new Error(`archive entry escapes dest dir: ${entry}`);
        }
    }
    execSync(`tar -xzf "${archivePath}" -C "${destDir}"`, { stdio: "pipe" });
}

class PrebuiltFetchError extends Error {
    constructor(readonly kind: PrebuiltFetchResult) {
        super(kind);
    }
}

async function downloadToBuffer(url: string): Promise<Buffer> {
    let resp: Response;
    try {
        resp = await fetch(url);
    } catch {
        throw new PrebuiltFetchError("offline");
    }
    if (!resp.ok) {
        throw new PrebuiltFetchError(resp.status === 404 ? "not-found" : "offline");
    }
    return Buffer.from(await resp.arrayBuffer());
}

// Source checkout: running from a git clone, not an npm install. The package dir has no node_modules
// in its path, and there's likely no GitHub release for the dev version. Skip the fetch entirely.
function getPackageVersion(): string | null {
    if (!import.meta.dir.includes("node_modules")) return null;
    try {
        const pkg = JSON.parse(
            readFileSync(resolve(import.meta.dir, "..", "package.json"), "utf8"),
        );
        return pkg.version ?? null;
    } catch {
        return null;
    }
}

export interface PrebuiltResult {
    binaryPath: string;
    cefDir?: string;
    helperPath?: string;
}

function buildPrebuiltResult(
    cacheDir: string,
    binaryPath: string,
    target: string,
    portable: boolean,
): PrebuiltResult {
    const result: PrebuiltResult = { binaryPath };
    if (portable) {
        // Mac portable: the CEF framework sits at the cache dir top level (per the archive contract).
        // Linux/Windows portable: the CEF runtime is in a cef/ subdir.
        if (target.includes("darwin")) {
            result.cefDir = cacheDir;
            result.helperPath = resolve(cacheDir, "shallot-helper");
        } else {
            result.cefDir = resolve(cacheDir, "cef");
        }
    }
    return result;
}

/**
 * Try to resolve a prebuilt shell for (target, release, portable). Returns the binary path (and CEF
 * dir for portable) on a hit, null on any miss — the caller falls back to cargoBuild. Debug stays
 * lazy (returns null immediately). Every failure class prints a one-line note before returning null.
 * The optional `version` parameter defaults to the detected package version; tests pass it
 * explicitly to bypass the node_modules check.
 */
export async function tryPrebuilt(
    target: string,
    release: boolean,
    portable: boolean,
    version: string | null = getPackageVersion(),
): Promise<PrebuiltResult | null> {
    if (!release) return null;

    const mode: PrebuiltMode = portable ? "portable" : "system";

    if (version === null) {
        const d = resolvePrebuiltDecision(null, target, mode, false, null);
        console.log(`  prebuilt: ${d.reason}`);
        return null;
    }

    const cacheDir = prebuiltCacheDir(version, target, mode);
    const binaryName = target.includes("windows") ? "shallot-window.exe" : "shallot-window";
    const binaryPath = resolve(cacheDir, binaryName);
    // Sentinel written only after a fully successful extract, so a partial extraction (process
    // killed mid-write, disk full) is never read as a cache hit — the hit path checks this, not
    // just the binary's existence (F1).
    const completeMarker = resolve(cacheDir, ".complete");

    if (existsSync(completeMarker)) {
        const d = resolvePrebuiltDecision(version, target, mode, true, null);
        console.log(`  prebuilt: ${d.reason}`);
        return buildPrebuiltResult(cacheDir, binaryPath, target, portable);
    }

    const archiveName = prebuiltArchiveName(target, mode);
    const archiveUrl = prebuiltUrl(version, target, mode);
    const sumsUrl = prebuiltSha256SumsUrl(version);

    let fetchResult: PrebuiltFetchResult;
    try {
        const archiveBuf = await downloadToBuffer(archiveUrl);
        const sumsBuf = await downloadToBuffer(sumsUrl);
        const expectedHash = parseSha256Sums(sumsBuf.toString("utf8"), archiveName);
        if (expectedHash === null) {
            fetchResult = "not-found";
        } else if (sha256Hex(archiveBuf) !== expectedHash) {
            // SHA256SUMS ships from the same release as the archive, so this defends transport
            // corruption, not a compromised release (see the section-header comment above).
            fetchResult = "checksum-mismatch";
        } else {
            try {
                mkdirSync(cacheDir, { recursive: true });
                const tmpArchive = resolve(cacheDir, archiveName);
                writeFileSync(tmpArchive, archiveBuf);
                try {
                    extractTarGz(tmpArchive, cacheDir);
                } catch {
                    // Partial extraction must not poison the cache (F1): a half-extracted dir
                    // would read as a hit on the next run. Clean up so the miss is honest.
                    rmSync(cacheDir, { recursive: true, force: true });
                    throw new PrebuiltFetchError("extract-failed");
                }
                rmSync(tmpArchive, { force: true });
                if (existsSync(binaryPath)) {
                    writeFileSync(completeMarker, "");
                    fetchResult = "ok";
                } else {
                    rmSync(cacheDir, { recursive: true, force: true });
                    fetchResult = "not-found";
                }
            } catch (e) {
                if (e instanceof PrebuiltFetchError) throw e;
                // mkdir/writeFileSync failure: unwritable cache dir, not a network problem (F3).
                rmSync(cacheDir, { recursive: true, force: true });
                throw new PrebuiltFetchError("cache-error");
            }
        }
    } catch (e) {
        fetchResult = e instanceof PrebuiltFetchError ? e.kind : "offline";
    }

    const d = resolvePrebuiltDecision(version, target, mode, false, fetchResult);
    console.log(`  prebuilt: ${d.reason}`);

    if (d.decision === "prebuilt") {
        return buildPrebuiltResult(cacheDir, binaryPath, target, portable);
    }
    return null;
}

// rust/window ships in the npm tarball (package.json `files` includes `rust/window`), so a missing
// crate dir means a corrupt install or a non-standard layout, not an unsupported path. Guard it
// before spawning cargo, since a raw ENOENT from `cwd: RUST_CRATE` below is an opaque failure.
export function missingCrateDiagnostic(crateDir: string): string | null {
    if (existsSync(crateDir)) return null;
    return `no rust/window crate found at ${crateDir}. The crate ships in the npm package, so this looks like a corrupt install or a non-standard layout. Reinstall @dylanebert/shallot, or build from the source repo.`;
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
): void {
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
                devShellBuild(invocation.flags, distDir, winTargetDir),
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
    const cmd = invocation.kind === "xwin" ? "cargo xwin build" : "cargo build";
    const env = distDir ? { ...process.env, SHALLOT_DIST: distDir } : process.env;
    execSync([cmd, ...invocation.flags].join(" "), { cwd: RUST_CRATE, stdio: "inherit", env });
}

function ensureIcon(distDir: string): void {
    const distIcon = resolve(distDir, "icon.png");
    if (!existsSync(distIcon) && existsSync(DEFAULT_ICON)) {
        cpSync(DEFAULT_ICON, distIcon);
    }
}

// the downloaded CEF runtime lives in cef-dll-sys's build OUT_DIR (or CEF_PATH when set). Return the
// subdir under it holding `marker` (libcef.so / libcef.dll / the framework) for a target's build.
export function findCefDir(
    target: string,
    marker: string,
    targetDir = CRATE_TARGET,
): string | null {
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
    console.log(
        `  compiling ${portable ? "CEF" : "webview"} shell (${release ? "release" : "debug"})...`,
    );

    // Prebuilt hit skips cargo entirely; portable sets CEF_PATH at the existing findCefDir seam so
    // copyCefDlls below finds the runtime in the cache. Every miss falls through to cargoBuild.
    const prebuilt = await tryPrebuilt(WIN_TARGET, release, portable);
    let exe: string;
    let targetDir = CRATE_TARGET;
    if (prebuilt) {
        exe = prebuilt.binaryPath;
        if (prebuilt.cefDir) process.env.CEF_PATH = prebuilt.cefDir;
    } else {
        // portable on WSL builds into a local Windows dir (off the 9p share); read its artifacts back
        // through the /mnt view. Every other case uses the in-tree target/.
        const winBuild = portable && isWSL ? winBuildDir() : null;
        targetDir = winBuild ? winBuild.wsl : CRATE_TARGET;
        cargoBuild(WIN_TARGET, release, portable, release ? distDir : undefined, winBuild?.win);
        exe = cargoTarget(WIN_TARGET, release, targetDir);
    }
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

    // Prebuilt hit skips cargo entirely; portable sets CEF_PATH at the existing findCefDir seam so
    // the framework + helper are found in the cache. Every miss falls through to cargoBuild.
    const prebuilt = await tryPrebuilt(MAC_TARGET, release, portable);
    let bin: string;
    if (prebuilt) {
        bin = prebuilt.binaryPath;
        if (prebuilt.cefDir) process.env.CEF_PATH = prebuilt.cefDir;
    } else {
        cargoBuild(MAC_TARGET, release, portable, release ? distDir : undefined);
        bin = cargoTarget(MAC_TARGET, release);
    }
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

    const identifier = bundleIdentifier(projectDir, name);
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
        const helperBin = prebuilt?.helperPath ?? macHelperBin(release);
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

    // Prebuilt hit skips cargo entirely; portable sets CEF_PATH at the existing findCefDir seam so
    // copyCefLibs below finds the runtime in the cache. Every miss falls through to cargoBuild.
    const prebuilt = await tryPrebuilt(LINUX_TARGET, release, portable);
    let bin: string;
    if (prebuilt) {
        bin = prebuilt.binaryPath;
        if (prebuilt.cefDir) process.env.CEF_PATH = prebuilt.cefDir;
    } else {
        cargoBuild(LINUX_TARGET, release, portable, release ? distDir : undefined);
        bin = cargoTarget(LINUX_TARGET, release);
    }
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
