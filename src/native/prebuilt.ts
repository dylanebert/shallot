import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, relative, resolve } from "node:path";

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

// Archive contract (pinned in the spec's Approach): one shallot-native-<target>-<mode>.tar.gz per
// matrix cell, plus a top-level SHA256SUMS asset covering all archives. URL derived from the version
// tag — no new version site for check-versions.ts.
const GITHUB_RELEASES = "https://github.com/dylanebert/shallot/releases/download";

export function prebuiltArchiveName(target: string, mode: PrebuiltMode): string {
    return `shallot-native-${target}-${mode}.tar.gz`;
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
            readFileSync(resolve(import.meta.dir, "../..", "package.json"), "utf8"),
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
    const binaryName = target.includes("windows") ? "shallot-native.exe" : "shallot-native";
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
