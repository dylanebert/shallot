import { describe, expect, test } from "bun:test";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { gzipSync } from "node:zlib";

// js-yaml is a transitive dep without bundled types; createRequire avoids needing @types/js-yaml.
const require = createRequire(import.meta.url);
const yaml = require("js-yaml") as { load: (text: string) => unknown };

import {
    extractTarGz,
    type PrebuiltFetchResult,
    type PrebuiltMode,
    parseSha256Sums,
    prebuiltArchiveName,
    prebuiltCacheDir,
    prebuiltSha256SumsUrl,
    prebuiltUrl,
    resolvePrebuiltDecision,
    sha256Hex,
    tryPrebuilt,
} from "./native";

// --- Pure decision table ---

describe("resolvePrebuiltDecision", () => {
    const target = "x86_64-unknown-linux-gnu";
    const mode: PrebuiltMode = "system";

    test("source checkout (version null) -> lazy", () => {
        const d = resolvePrebuiltDecision(null, target, mode, false, null);
        expect(d.decision).toBe("lazy");
        expect(d.reason).toContain("source");
    });

    test("cache hit -> prebuilt", () => {
        const d = resolvePrebuiltDecision("1.0.0", target, mode, true, null);
        expect(d.decision).toBe("prebuilt");
        expect(d.reason).toContain("cache");
    });

    test("fetch ok -> prebuilt", () => {
        const d = resolvePrebuiltDecision("1.0.0", target, mode, false, "ok");
        expect(d.decision).toBe("prebuilt");
    });

    // RED-FIRST ARM (a): a version with no matching release chooses lazy.
    test("fetch not-found -> lazy", () => {
        const d = resolvePrebuiltDecision("0.0.0-norelease", target, mode, false, "not-found");
        expect(d.decision).toBe("lazy");
        expect(d.reason).toContain("no matching release");
    });

    // RED-FIRST ARM (b): a checksum mismatch chooses lazy.
    test("fetch checksum-mismatch -> lazy", () => {
        const d = resolvePrebuiltDecision("1.0.0", target, mode, false, "checksum-mismatch");
        expect(d.decision).toBe("lazy");
        expect(d.reason).toContain("checksum");
    });

    test("fetch offline -> lazy", () => {
        const d = resolvePrebuiltDecision("1.0.0", target, mode, false, "offline");
        expect(d.decision).toBe("lazy");
        expect(d.reason).toContain("offline");
    });

    test("no fetch attempted (cache miss, fetchResult null) -> lazy", () => {
        const d = resolvePrebuiltDecision("1.0.0", target, mode, false, null);
        expect(d.decision).toBe("lazy");
    });

    // Exhaustive over PrebuiltFetchResult. The Record<..., true> assignment fails at compile time
    // if the union gains a member, so the arm's name stays true to its assertion (checks.md).
    test("every fetch result maps to a decision with a non-empty reason", () => {
        const exhaustive: Record<PrebuiltFetchResult, true> = {
            ok: true,
            "not-found": true,
            offline: true,
            "checksum-mismatch": true,
            "source-checkout": true,
            "extract-failed": true,
            "cache-error": true,
        };
        const results = Object.keys(exhaustive) as PrebuiltFetchResult[];
        for (const fr of results) {
            const d = resolvePrebuiltDecision("1.0.0", target, mode, false, fr);
            expect(d.reason.length).toBeGreaterThan(0);
        }
    });
});

// --- Pure URL / archive / cache helpers ---

describe("prebuiltArchiveName", () => {
    test("system mode: shallot-window-<target>-system.tar.gz", () => {
        expect(prebuiltArchiveName("x86_64-unknown-linux-gnu", "system")).toBe(
            "shallot-window-x86_64-unknown-linux-gnu-system.tar.gz",
        );
    });

    test("portable mode: shallot-window-<target>-portable.tar.gz", () => {
        expect(prebuiltArchiveName("x86_64-pc-windows-msvc", "portable")).toBe(
            "shallot-window-x86_64-pc-windows-msvc-portable.tar.gz",
        );
    });
});

describe("prebuiltUrl", () => {
    test("derives the GitHub Releases download URL from version + target + mode", () => {
        expect(prebuiltUrl("0.9.2", "x86_64-unknown-linux-gnu", "system")).toBe(
            "https://github.com/dylanebert/shallot/releases/download/v0.9.2/shallot-window-x86_64-unknown-linux-gnu-system.tar.gz",
        );
    });
});

describe("prebuiltSha256SumsUrl", () => {
    test("derives the SHA256SUMS asset URL from version", () => {
        expect(prebuiltSha256SumsUrl("0.9.2")).toBe(
            "https://github.com/dylanebert/shallot/releases/download/v0.9.2/SHA256SUMS",
        );
    });
});

describe("prebuiltCacheDir", () => {
    test("XDG_CACHE_HOME overrides the default ~/.cache base", () => {
        const prev = process.env.XDG_CACHE_HOME;
        const tmp = mkdtempSync(join(tmpdir(), "shallot-xdg-"));
        process.env.XDG_CACHE_HOME = tmp;
        try {
            expect(prebuiltCacheDir("1.0.0", "x86_64-unknown-linux-gnu", "system")).toBe(
                resolve(tmp, "shallot", "prebuilt", "1.0.0", "x86_64-unknown-linux-gnu-system"),
            );
        } finally {
            if (prev === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prev;
            rmSync(tmp, { recursive: true, force: true });
        }
    });

    test("falls back to ~/.cache when XDG_CACHE_HOME is unset", () => {
        const prev = process.env.XDG_CACHE_HOME;
        delete process.env.XDG_CACHE_HOME;
        try {
            const dir = prebuiltCacheDir("1.0.0", "aarch64-apple-darwin", "portable");
            expect(dir).toContain("shallot");
            expect(dir).toContain("prebuilt");
            expect(dir).toContain("1.0.0");
            expect(dir).toContain("aarch64-apple-darwin-portable");
        } finally {
            if (prev === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prev;
        }
    });
});

// --- SHA256 + SHA256SUMS parsing ---

describe("sha256Hex", () => {
    test("matches node:crypto for a known input", () => {
        const buf = Buffer.from("hello");
        const expected = createHash("sha256").update(buf).digest("hex");
        expect(sha256Hex(buf)).toBe(expected);
    });
});

describe("parseSha256Sums", () => {
    test("finds the hash for the matching archive name", () => {
        const sums = `${"a".repeat(64)}  shallot-window-x86_64-unknown-linux-gnu-system.tar.gz\n${"b".repeat(64)}  shallot-window-x86_64-unknown-linux-gnu-portable.tar.gz\n`;
        expect(parseSha256Sums(sums, "shallot-window-x86_64-unknown-linux-gnu-system.tar.gz")).toBe(
            "a".repeat(64),
        );
        expect(
            parseSha256Sums(sums, "shallot-window-x86_64-unknown-linux-gnu-portable.tar.gz"),
        ).toBe("b".repeat(64));
    });

    test("returns null when the archive name is not in the sums", () => {
        const sums = `${"a".repeat(64)}  other-archive.tar.gz\n`;
        expect(
            parseSha256Sums(sums, "shallot-window-x86_64-unknown-linux-gnu-system.tar.gz"),
        ).toBeNull();
    });

    test("handles whitespace variations (multiple spaces, tabs)", () => {
        const sums = `${"a".repeat(64)}  \t shallot-window-x86_64-unknown-linux-gnu-system.tar.gz\n`;
        expect(parseSha256Sums(sums, "shallot-window-x86_64-unknown-linux-gnu-system.tar.gz")).toBe(
            "a".repeat(64),
        );
    });

    // R8: GNU `sha256sum -b` emits `<hash> *<name>` (binary mode). Without stripping the leading `*`,
    // the parser compares `*<name>` === `<name>` and silently returns null — the resolver falls back
    // to lazy compile forever, the exact expensive-class failure this unit exists to remove.
    test("handles binary-mode format (leading * before archive name)", () => {
        const sums = `${"c".repeat(64)} *shallot-window-x86_64-unknown-linux-gnu-system.tar.gz\n`;
        expect(parseSha256Sums(sums, "shallot-window-x86_64-unknown-linux-gnu-system.tar.gz")).toBe(
            "c".repeat(64),
        );
    });
});

// --- Extraction (local fixture archives, no network) ---

describe("extractTarGz", () => {
    test("extracts a fixture archive into the dest dir", () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-extract-"));
        try {
            // build a fixture: a dir with a fake binary + a cef/ subdir
            const srcDir = join(tmp, "src");
            mkdirSync(join(srcDir, "cef"), { recursive: true });
            writeFileSync(join(srcDir, "shallot-window"), "#!/bin/sh\necho hi\n");
            writeFileSync(join(srcDir, "cef", "libcef.so"), "fake cef");
            writeFileSync(join(srcDir, "README.txt"), "test archive");

            // pack into tar.gz
            const archivePath = join(tmp, "test.tar.gz");
            execSync(`tar -czf "${archivePath}" -C "${srcDir}" .`, { stdio: "pipe" });

            // extract
            const destDir = join(tmp, "dest");
            extractTarGz(archivePath, destDir);

            expect(existsSync(join(destDir, "shallot-window"))).toBe(true);
            expect(existsSync(join(destDir, "cef", "libcef.so"))).toBe(true);
            expect(existsSync(join(destDir, "README.txt"))).toBe(true);
        } finally {
            rmSync(tmp, { recursive: true, force: true });
        }
    });

    // F4: a fixture archive containing an escaping entry (../escape.txt) must be refused —
    // nothing written outside destDir.
    test("rejects an archive with a path-traversal entry", () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-f4-"));
        try {
            const archiveBuf = buildTarGz([{ name: "../escape.txt", content: "escaped" }]);
            const archivePath = join(tmp, "escape.tar.gz");
            writeFileSync(archivePath, archiveBuf);

            const destDir = join(tmp, "dest");
            expect(() => extractTarGz(archivePath, destDir)).toThrow();
            // Nothing written outside destDir: ../escape.txt from dest would land at tmp/escape.txt
            expect(existsSync(join(tmp, "escape.txt"))).toBe(false);
        } finally {
            rmSync(tmp, { recursive: true, force: true });
        }
    });
});

// --- Resolver-level tests (tryPrebuilt with fetch stubbed, no network) ---
//
// F2: the spec's red-first oracle names the resolver, not just the pure table. These arms call
// tryPrebuilt directly with globalThis.fetch stubbed and the cache root redirected through
// XDG_CACHE_HOME (the existing seam the spec pinned). No test touches the network; every stubbed
// global is restored in a finally.

const LINUX = "x86_64-unknown-linux-gnu";

describe("tryPrebuilt resolver", () => {
    // F1: a cache dir holding the shell binary but no CEF payload (no .complete sentinel) must
    // not read as a hit — the resolver chooses lazy (returns null) rather than handing back a
    // result whose CEF_PATH is incomplete.
    test("partial extraction (binary present, no sentinel, no CEF) -> lazy", async () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-f1-"));
        const prevXdg = process.env.XDG_CACHE_HOME;
        const prevFetch = globalThis.fetch;
        process.env.XDG_CACHE_HOME = tmp;
        globalThis.fetch = (() =>
            Promise.resolve(new Response("", { status: 404 }))) as unknown as typeof fetch;
        try {
            const cacheDir = prebuiltCacheDir("1.0.0", LINUX, "portable");
            mkdirSync(cacheDir, { recursive: true });
            writeFileSync(join(cacheDir, "shallot-window"), "#!/bin/sh\necho hi\n");
            // No .complete sentinel, no cef/ — a partial extraction.
            const result = await tryPrebuilt(LINUX, true, true, "1.0.0");
            expect(result).toBeNull();
        } finally {
            if (prevXdg === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prevXdg;
            globalThis.fetch = prevFetch;
            rmSync(tmp, { recursive: true, force: true });
        }
    });

    // F2 RED-FIRST ARM (a): resolver pointed at a version with no release chooses lazy.
    test("tryPrebuilt returns null on 404 (resolver-level)", async () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-f2-404-"));
        const prevXdg = process.env.XDG_CACHE_HOME;
        const prevFetch = globalThis.fetch;
        process.env.XDG_CACHE_HOME = tmp;
        globalThis.fetch = (() =>
            Promise.resolve(new Response("", { status: 404 }))) as unknown as typeof fetch;
        try {
            const result = await tryPrebuilt(LINUX, true, false, "1.0.0");
            expect(result).toBeNull();
        } finally {
            if (prevXdg === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prevXdg;
            globalThis.fetch = prevFetch;
            rmSync(tmp, { recursive: true, force: true });
        }
    });

    // F2 RED-FIRST ARM (b): a SHA256SUMS whose hash does not match the payload chooses lazy.
    test("tryPrebuilt returns null on checksum mismatch (resolver-level)", async () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-f2-mismatch-"));
        const prevXdg = process.env.XDG_CACHE_HOME;
        const prevFetch = globalThis.fetch;
        process.env.XDG_CACHE_HOME = tmp;
        const archiveName = prebuiltArchiveName(LINUX, "system");
        const sumsContent = `${"0".repeat(64)}  ${archiveName}\n`;
        globalThis.fetch = ((url: string) => {
            if (url.includes("SHA256SUMS")) {
                return Promise.resolve(new Response(sumsContent, { status: 200 }));
            }
            return Promise.resolve(new Response("fake archive content", { status: 200 }));
        }) as unknown as typeof fetch;
        try {
            const result = await tryPrebuilt(LINUX, true, false, "1.0.0");
            expect(result).toBeNull();
        } finally {
            if (prevXdg === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prevXdg;
            globalThis.fetch = prevFetch;
            rmSync(tmp, { recursive: true, force: true });
        }
    });

    // F3: an extraction failure (checksum-valid but unextractable archive) must not be reported
    // as "offline" — the note identifies the local failure class.
    test("extraction failure produces a non-offline note", async () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-f3-"));
        const prevXdg = process.env.XDG_CACHE_HOME;
        const prevFetch = globalThis.fetch;
        const prevLog = console.log;
        process.env.XDG_CACHE_HOME = tmp;

        const archiveContent = "corrupt archive data";
        const archiveBuf = Buffer.from(archiveContent);
        const archiveHash = sha256Hex(archiveBuf);
        const archiveName = prebuiltArchiveName(LINUX, "system");
        const sumsContent = `${archiveHash}  ${archiveName}\n`;

        globalThis.fetch = ((url: string) => {
            if (url.includes("SHA256SUMS")) {
                return Promise.resolve(new Response(sumsContent, { status: 200 }));
            }
            return Promise.resolve(new Response(archiveBuf, { status: 200 }));
        }) as unknown as typeof fetch;

        const logs: string[] = [];
        console.log = ((...args: unknown[]) => {
            logs.push(args.join(" "));
        }) as typeof console.log;

        try {
            const result = await tryPrebuilt(LINUX, true, false, "1.0.0");
            expect(result).toBeNull();
            const note = logs.find((l) => l.includes("prebuilt:")) ?? "";
            expect(note).not.toContain("offline");
            expect(note).toContain("extract");
        } finally {
            if (prevXdg === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prevXdg;
            globalThis.fetch = prevFetch;
            console.log = prevLog;
            rmSync(tmp, { recursive: true, force: true });
        }
    });

    // F5: a successful extract returns a non-null result with the binary path and a .complete
    // sentinel. All four existing arms assert toBeNull(); without a success arm, reverting the
    // success wiring to `return null` leaves the suite green — the exact regression this arm
    // catches. Reuses the extractTarGz fixture machinery (tar -czf from a src dir).
    test("successful system extract -> non-null result with binary path + sentinel", async () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-f5-sys-"));
        const prevXdg = process.env.XDG_CACHE_HOME;
        const prevFetch = globalThis.fetch;
        process.env.XDG_CACHE_HOME = tmp;
        try {
            const srcDir = join(tmp, "src");
            mkdirSync(srcDir, { recursive: true });
            writeFileSync(join(srcDir, "shallot-window"), "#!/bin/sh\necho hi\n");
            const archivePath = join(tmp, "test.tar.gz");
            execSync(`tar -czf "${archivePath}" -C "${srcDir}" .`, { stdio: "pipe" });
            const archiveBuf = readFileSync(archivePath);
            const archiveHash = sha256Hex(archiveBuf);
            const archiveName = prebuiltArchiveName(LINUX, "system");
            const sumsContent = `${archiveHash}  ${archiveName}\n`;

            globalThis.fetch = ((url: string) => {
                if (url.includes("SHA256SUMS")) {
                    return Promise.resolve(new Response(sumsContent, { status: 200 }));
                }
                return Promise.resolve(new Response(archiveBuf, { status: 200 }));
            }) as unknown as typeof fetch;

            const result = await tryPrebuilt(LINUX, true, false, "1.0.0");
            expect(result).not.toBeNull();
            const cacheDir = prebuiltCacheDir("1.0.0", LINUX, "system");
            expect(result!.binaryPath).toBe(resolve(cacheDir, "shallot-window"));
            expect(result!.cefDir).toBeUndefined();
            expect(existsSync(join(cacheDir, ".complete"))).toBe(true);
        } finally {
            if (prevXdg === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prevXdg;
            globalThis.fetch = prevFetch;
            rmSync(tmp, { recursive: true, force: true });
        }
    });

    // F5 (linux portable): cefDir points at the cef/ subdir, not the cache root.
    test("successful linux portable extract -> non-null result with binary/cefDir paths + sentinel", async () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-f5-lx-"));
        const prevXdg = process.env.XDG_CACHE_HOME;
        const prevFetch = globalThis.fetch;
        process.env.XDG_CACHE_HOME = tmp;
        try {
            const srcDir = join(tmp, "src");
            mkdirSync(join(srcDir, "cef"), { recursive: true });
            writeFileSync(join(srcDir, "shallot-window"), "#!/bin/sh\necho hi\n");
            writeFileSync(join(srcDir, "cef", "libcef.so"), "fake cef");
            const archivePath = join(tmp, "test.tar.gz");
            execSync(`tar -czf "${archivePath}" -C "${srcDir}" .`, { stdio: "pipe" });
            const archiveBuf = readFileSync(archivePath);
            const archiveHash = sha256Hex(archiveBuf);
            const archiveName = prebuiltArchiveName(LINUX, "portable");
            const sumsContent = `${archiveHash}  ${archiveName}\n`;

            globalThis.fetch = ((url: string) => {
                if (url.includes("SHA256SUMS")) {
                    return Promise.resolve(new Response(sumsContent, { status: 200 }));
                }
                return Promise.resolve(new Response(archiveBuf, { status: 200 }));
            }) as unknown as typeof fetch;

            const result = await tryPrebuilt(LINUX, true, true, "1.0.0");
            expect(result).not.toBeNull();
            const cacheDir = prebuiltCacheDir("1.0.0", LINUX, "portable");
            expect(result!.binaryPath).toBe(resolve(cacheDir, "shallot-window"));
            expect(result!.cefDir).toBe(resolve(cacheDir, "cef"));
            expect(existsSync(join(cacheDir, ".complete"))).toBe(true);
        } finally {
            if (prevXdg === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prevXdg;
            globalThis.fetch = prevFetch;
            rmSync(tmp, { recursive: true, force: true });
        }
    });

    // F5 (mac portable): cefDir is the cache root itself (CEF framework at top level), and
    // helperPath points at shallot-helper — the two path shapes the other arms don't reach.
    const Mac = "aarch64-apple-darwin";
    test("successful mac portable extract -> non-null result with binary/cefDir/helper paths + sentinel", async () => {
        const tmp = mkdtempSync(join(tmpdir(), "shallot-f5-mac-"));
        const prevXdg = process.env.XDG_CACHE_HOME;
        const prevFetch = globalThis.fetch;
        process.env.XDG_CACHE_HOME = tmp;
        try {
            const srcDir = join(tmp, "src");
            mkdirSync(srcDir, { recursive: true });
            writeFileSync(join(srcDir, "shallot-window"), "#!/bin/sh\necho hi\n");
            writeFileSync(join(srcDir, "shallot-helper"), "#!/bin/sh\necho helper\n");
            const archivePath = join(tmp, "test.tar.gz");
            execSync(`tar -czf "${archivePath}" -C "${srcDir}" .`, { stdio: "pipe" });
            const archiveBuf = readFileSync(archivePath);
            const archiveHash = sha256Hex(archiveBuf);
            const archiveName = prebuiltArchiveName(Mac, "portable");
            const sumsContent = `${archiveHash}  ${archiveName}\n`;

            globalThis.fetch = ((url: string) => {
                if (url.includes("SHA256SUMS")) {
                    return Promise.resolve(new Response(sumsContent, { status: 200 }));
                }
                return Promise.resolve(new Response(archiveBuf, { status: 200 }));
            }) as unknown as typeof fetch;

            const result = await tryPrebuilt(Mac, true, true, "1.0.0");
            expect(result).not.toBeNull();
            const cacheDir = prebuiltCacheDir("1.0.0", Mac, "portable");
            expect(result!.binaryPath).toBe(resolve(cacheDir, "shallot-window"));
            expect(result!.cefDir).toBe(cacheDir);
            expect(result!.helperPath).toBe(resolve(cacheDir, "shallot-helper"));
            expect(existsSync(join(cacheDir, ".complete"))).toBe(true);
        } finally {
            if (prevXdg === undefined) delete process.env.XDG_CACHE_HOME;
            else process.env.XDG_CACHE_HOME = prevXdg;
            globalThis.fetch = prevFetch;
            rmSync(tmp, { recursive: true, force: true });
        }
    });
});

// --- Helpers ---

/** Build a minimal USTAR .tar.gz from {name, content} entries. Used for the F4 path-traversal
 *  fixture so the test doesn't depend on tar --transform / python3 availability. */
function buildTarGz(entries: { name: string; content: string }[]): Buffer {
    const chunks: Buffer[] = [];
    for (const { name, content } of entries) {
        const contentBuf = Buffer.from(content);
        const header = Buffer.alloc(512);
        header.write(name, 0, "utf8");
        header.write("0000644\x00", 100, "utf8");
        header.write("0000000\x00", 108, "utf8");
        header.write("0000000\x00", 116, "utf8");
        header.write(contentBuf.length.toString(8).padStart(11, "0") + "\x00", 124, "utf8");
        header.write("00000000000\x00", 136, "utf8");
        // checksum field: 8 spaces (placeholder for checksum calculation)
        header.write("        ", 148, "utf8");
        header.write("0", 156, "utf8"); // typeflag: regular file
        header.write("ustar\x00", 257, "utf8");
        header.write("00", 263, "utf8");

        let sum = 0;
        for (let i = 0; i < 512; i++) sum += header[i];
        const checksumStr = sum.toString(8).padStart(6, "0") + "\x00 ";
        header.write(checksumStr, 148, "utf8");

        chunks.push(header);
        const paddedLen = Math.ceil(contentBuf.length / 512) * 512;
        const contentPadded = Buffer.alloc(paddedLen);
        contentBuf.copy(contentPadded);
        chunks.push(contentPadded);
    }
    chunks.push(Buffer.alloc(1024)); // two 512-byte zero blocks (end of archive)
    return gzipSync(Buffer.concat(chunks));
}

// --- Producer/consumer arms: release.yml is the producer for prebuiltArchiveName ---
//
// Stage B (shipped) exports prebuiltArchiveName/prebuiltUrl/prebuiltSha256SumsUrl as the consumer
// contract. Stage C's release.yml workflow is the *producer* — it must assemble exactly those archive
// names over the full 3×2 matrix, generate a SHA256SUMS covering all six, and gate the release upload
// on the tag trigger so a workflow_dispatch dry run can never create or mutate a release. These arms
// parse the YAML and assert that contract structurally; the CI dispatch itself is the coordinator's
// to run (network-fenced here).
//
// WHAT THESE ARMS CANNOT SEE (be honest about the gap):
//   • Inner archive layout — the arms verify the matrix names and that upload/download steps exist,
//     but not what's *inside* each tar.gz (binary present, CEF file-set correct, helper for mac).
//   • That the build actually produces the binary — the arms see the cargo build step exists, but
//     cannot verify it compiles or produces a working shallot-window.
//   • CEF file-set correctness — the assemble step's bash is not parsed; a wrong file list or a
//     missing findCefDir marker would pass these arms.
// The CI dry run (workflow_dispatch) is the gate for those — it exercises the real build, assemble,
// and upload pipeline end-to-end.

const WORKFLOW_PATH = resolve(import.meta.dir, "../../../.github/workflows/release.yml");

// The three rust triples the CLI actually expects (from native.ts WIN_TARGET/MAC_TARGET/LINUX_TARGET).
const RELEASE_TARGETS = [
    "x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
    "aarch64-apple-darwin",
] as const;
const RELEASE_MODES = ["system", "portable"] as const;

// The full 3×2 matrix of expected archive names — six, enumerated.
const EXPECTED_ARCHIVE_NAMES = new Set(
    RELEASE_TARGETS.flatMap((t) => RELEASE_MODES.map((m) => prebuiltArchiveName(t, m))),
);

function loadWorkflowYaml(): Record<string, unknown> {
    if (!existsSync(WORKFLOW_PATH)) throw new Error(`workflow not found at ${WORKFLOW_PATH}`);
    return yaml.load(readFileSync(WORKFLOW_PATH, "utf8")) as Record<string, unknown>;
}

describe("release.yml producer/consumer contract", () => {
    test("workflow file exists at .github/workflows/release.yml", () => {
        expect(existsSync(WORKFLOW_PATH)).toBe(true);
    });

    test("triggers: push on tag v* plus workflow_dispatch", () => {
        const wf = loadWorkflowYaml();
        const on = wf.on as Record<string, unknown>;
        expect(on).toBeDefined();
        // push must be tag-scoped to v*
        const push = on.push as Record<string, unknown> | undefined;
        expect(push).toBeDefined();
        const tags = push?.tags as unknown[] | undefined;
        expect(tags).toBeDefined();
        expect(tags).toContain("v*");
        // workflow_dispatch must be present (the dry-run path)
        expect(on.workflow_dispatch).toBeDefined();
    });

    // Arm 1: the set of archive names the workflow produces equals the full 3×2 matrix.
    // Six names enumerated from the matrix, not a regex that would pass on five.
    test("produces exactly the six archive names from the 3×2 matrix", () => {
        const wf = loadWorkflowYaml();
        const jobs = wf.jobs as Record<string, Record<string, unknown>>;
        const build = jobs.build;
        expect(build).toBeDefined();
        const matrix = (build.strategy as Record<string, unknown>).matrix as Record<
            string,
            unknown
        >;
        const include = matrix.include as Record<string, string>[];

        // Exactly six cells.
        expect(include.length).toBe(6);

        // Each cell must carry a target and mode.
        const produced = new Set<string>();
        for (const cell of include) {
            expect(cell.target).toBeDefined();
            expect(cell.mode).toBeDefined();
            produced.add(prebuiltArchiveName(cell.target, cell.mode as PrebuiltMode));
        }

        // Six unique names — no duplicates collapsing the matrix.
        expect(produced.size).toBe(6);

        // The set must equal the expected 3×2 enumeration exactly.
        expect(produced).toEqual(EXPECTED_ARCHIVE_NAMES);

        // Structural reachability: each cell's archive must be referenced by a real step, not
        // just present in the matrix text (the names live in the matrix, which is in the text —
        // a text-containment check is always true when the matrix is right). The build job must
        // have an upload-artifact step whose with.name or with.path references matrix.archive, and
        // the checksums job must have a download-artifact step whose with.pattern matches every
        // archive name. A workflow with a correct matrix but no upload step (or no collect/download
        // step) fails here.
        const buildSteps = jobs.build.steps as Record<string, unknown>[];
        const uploadStep = buildSteps.find((s) => String(s.uses ?? "").includes("upload-artifact"));
        expect(uploadStep).toBeDefined();
        const uploadWith = uploadStep?.with as Record<string, unknown> | undefined;
        expect(uploadWith).toBeDefined();
        const uploadName = String(uploadWith?.name ?? "");
        const uploadPath = String(uploadWith?.path ?? "");
        expect(uploadName.includes("matrix.archive") || uploadPath.includes("matrix.archive")).toBe(
            true,
        );

        const checksumsSteps = jobs.checksums.steps as Record<string, unknown>[];
        const downloadStep = checksumsSteps.find((s) =>
            String(s.uses ?? "").includes("download-artifact"),
        );
        expect(downloadStep).toBeDefined();
        const downloadWith = downloadStep?.with as Record<string, unknown> | undefined;
        expect(downloadWith).toBeDefined();
        const pattern = String(downloadWith?.pattern ?? "");
        expect(pattern).toBeTruthy();
        // Convert the glob to a regex and verify it matches every expected archive name.
        const globRegex = new RegExp(
            `^${pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*")}$`,
        );
        for (const name of EXPECTED_ARCHIVE_NAMES) {
            expect(globRegex.test(name)).toBe(true);
        }
    });

    // Arm 2: SHA256SUMS is generated and covers all six.
    test("SHA256SUMS is generated and covers all six archives", () => {
        const wf = loadWorkflowYaml();
        const jobs = wf.jobs as Record<string, Record<string, unknown>>;

        // A checksums job must exist and depend on the build job (so all six cells complete first).
        const checksums = jobs.checksums;
        expect(checksums).toBeDefined();
        const needs = checksums.needs as string | string[] | undefined;
        if (Array.isArray(needs)) {
            expect(needs).toContain("build");
        } else {
            expect(needs).toBe("build");
        }

        // It must download all build artifacts (the six archives).
        const steps = checksums.steps as Record<string, unknown>[];
        const downloadStep = steps.find((s) => String(s.uses ?? "").includes("download-artifact"));
        expect(downloadStep).toBeDefined();

        // It must generate SHA256SUMS via sha256sum over the archives.
        const sumsStep = steps.find(
            (s) =>
                (s.name as string | undefined)?.includes("SHA256SUMS") ||
                (typeof s.run === "string" && s.run.includes("SHA256SUMS")),
        );
        expect(sumsStep).toBeDefined();
        expect(typeof sumsStep?.run).toBe("string");
        expect(sumsStep?.run as string).toContain("sha256sum");

        // Structural: the sha256sum run command must reference a glob covering all six
        // archive names — not just a text-containment check that passes when the matrix is right.
        const sumsRun = String(sumsStep?.run ?? "");
        expect(sumsRun).toContain("sha256sum");
        // The sha256sum command must reference the archive glob (shallot-window-*.tar.gz).
        expect(sumsRun).toMatch(/shallot-window-\*\.tar\.gz/);
    });

    // Arm 3: the release-upload step is gated on the tag trigger (a dispatch run cannot reach it).
    test("release upload is gated on tag trigger (dispatch cannot reach it)", () => {
        const wf = loadWorkflowYaml();
        const jobs = wf.jobs as Record<string, Record<string, unknown>>;

        // Find the step that uploads to a GitHub release — now a `gh release upload` run step
        // (was softprops/action-gh-release; replaced with first-party gh to zero the third-party surface).
        const allSteps = Object.values(jobs).flatMap(
            (job) => (job.steps as Record<string, unknown>[] | undefined) ?? [],
        );
        const releaseStep = allSteps.find((s) => {
            const run = String(s.run ?? "");
            return run.includes("gh release upload");
        });
        expect(releaseStep).toBeDefined();

        // The step must have an if: condition that gates on the tag trigger.
        expect(releaseStep?.if).toBeDefined();
        const ifCond = String(releaseStep?.if);
        // Must gate on the literal tag ref prefix refs/tags/v — not a loose /tags|startsWith.*ref/i
        // pattern that would pass on wrong-but-similar gates (e.g. refs/heads/v* or refs/tags/nightly).
        expect(ifCond).toContain("refs/tags/v");
    });

    // Arm 4: the release step creates the GitHub Release object when the tag has none.
    // shallot's release procedure (testing.md § Release gate) ends at publish + push tag — it
    // never creates a Release object. So on the first tag push there is nothing to upload into,
    // and `gh release upload` alone would fail "release not found". The step must branch on
    // `gh release view` and create with --verify-tag (the tag is guaranteed present — the
    // workflow is tag-triggered) and --generate-notes, not a `||` chain that would swallow an
    // unrelated failure.
    test("release step creates the Release when the tag has none (create-if-missing)", () => {
        const wf = loadWorkflowYaml();
        const jobs = wf.jobs as Record<string, Record<string, unknown>>;
        const allSteps = Object.values(jobs).flatMap(
            (job) => (job.steps as Record<string, unknown>[] | undefined) ?? [],
        );
        // Finder stays specific to the release step: it must contain `gh release upload`
        // (the upload-into-existing path). A generic run: step would not match.
        const releaseStep = allSteps.find((s) => {
            const run = String(s.run ?? "");
            return run.includes("gh release upload");
        });
        expect(releaseStep).toBeDefined();
        const run = String(releaseStep?.run ?? "");
        // Must branch on `gh release view` (explicit branch, not a `||` chain).
        expect(run).toContain("gh release view");
        // Must create the Release with --verify-tag and --generate-notes.
        expect(run).toContain("gh release create");
        expect(run).toContain("--verify-tag");
        expect(run).toContain("--generate-notes");
    });

    // Arm 5: a re-run (Release already exists) uploads with --clobber so a partial upload
    // converges instead of erroring on the first already-present asset.
    test("release step uploads with --clobber for an existing Release (idempotent re-run)", () => {
        const wf = loadWorkflowYaml();
        const jobs = wf.jobs as Record<string, Record<string, unknown>>;
        const allSteps = Object.values(jobs).flatMap(
            (job) => (job.steps as Record<string, unknown>[] | undefined) ?? [],
        );
        const releaseStep = allSteps.find((s) => {
            const run = String(s.run ?? "");
            return run.includes("gh release upload");
        });
        expect(releaseStep).toBeDefined();
        const run = String(releaseStep?.run ?? "");
        expect(run).toContain("--clobber");
    });

    // Arm 6 (D1 fix): the workflow declares a workflow-level `defaults: run: shell: bash` so every
    // `run:` step executes in bash, not the Windows runner's default PowerShell. Without this, every
    // bash construct in the build/assemble bodies (`if [ ... ]`, `$(...)`, `for ... in`, etc.) dies
    // with a PowerShell ParserError on the windows-latest cells. The arm asserts the default exists
    // at the workflow top level and is `bash` — removing it or narrowing it (e.g. to `sh`, or moving
    // it to a single step) fails here. It also checks that no `run:` step overrides the default with
    // a non-bash shell, so a per-step `shell: powershell` cannot silently re-introduce the defect.
    test("workflow declares a bash shell default covering every run step (D1 fix)", () => {
        const wf = loadWorkflowYaml();

        // The workflow-level default must exist and set shell to bash.
        const defaults = wf.defaults as Record<string, unknown> | undefined;
        expect(defaults).toBeDefined();
        const runDefault = defaults?.run as Record<string, unknown> | undefined;
        expect(runDefault).toBeDefined();
        expect(runDefault?.shell).toBe("bash");

        // Every `run:` step across all jobs must either inherit the default (no explicit shell) or
        // explicitly set shell to bash — a step that overrides with a non-bash shell re-introduces
        // the defect on Windows. This catches a per-step `shell: powershell` that would slip past a
        // default-only check.
        const jobs = wf.jobs as Record<string, Record<string, unknown>>;
        const allSteps = Object.values(jobs).flatMap(
            (job) => (job.steps as Record<string, unknown>[] | undefined) ?? [],
        );
        const runSteps = allSteps.filter((s) => typeof s.run === "string");
        expect(runSteps.length).toBeGreaterThan(0);
        for (const step of runSteps) {
            const shell = step.shell as string | undefined;
            // If a step declares an explicit shell, it must be bash — not powershell, not sh.
            if (shell !== undefined) {
                expect(shell).toBe("bash");
            }
        }
    });

    // Arm 7 (D2/D3 fix): the ubuntu cells reach a dependency-install step, and the mac/windows cells
    // do not. The Linux system build (wry → WebKitGTK) needs glib/GTK dev headers (D2), and the
    // Linux portable build (CEF) needs X11 dev headers (D3). The install step must be gated to
    // `runner.os == 'Linux'` so mac and Windows — already green — do not grow an install step. The
    // arm asserts the step exists, its `if:` gates narrowly on Linux, and it installs the packages
    // that fix both failures (a webkit package for D2, an x11 package for D3). It also verifies no
    // other install step reaches mac or Windows by checking every apt-get step's `if:` condition.
    test("ubuntu cells reach a dependency-install step; mac/windows do not (D2/D3 fix)", () => {
        const wf = loadWorkflowYaml();
        const jobs = wf.jobs as Record<string, Record<string, unknown>>;
        const buildSteps = jobs.build.steps as Record<string, unknown>[];

        // Find the dependency-install step: a `run:` step whose body includes apt-get install.
        const installStep = buildSteps.find(
            (s) => typeof s.run === "string" && s.run.includes("apt-get install"),
        );
        expect(installStep).toBeDefined();

        // The step must be gated to Linux only — not unguarded, not gated to a broader condition
        // that would also fire on mac or Windows. The condition must name `runner.os` and `Linux`.
        const ifCond = String(installStep?.if ?? "");
        expect(ifCond).toContain("runner.os");
        expect(ifCond).toMatch(/Linux/);
        // Must NOT be gated to portable-only (the old step was `Linux && portable`, which left the
        // system build without glib/GTK headers — D2).
        expect(ifCond).not.toContain("portable");

        // The install must include a WebKitGTK dev package (D2: glib-2.0 not found by pkg-config) and
        // an X11 dev package (D3: -lX11 not linkable). Checking for the package names, not just the
        // step's existence, catches a step that installs only the old g++/pkg-config pair.
        const installRun = String(installStep?.run ?? "");
        expect(installRun).toMatch(/libwebkit2gtk-4\.[01]-dev/);
        expect(installRun).toMatch(/libx11-dev/);

        // No apt-get install step may fire on mac or Windows: every apt-get step's `if:` must gate
        // on Linux. A step with no `if:` (unguarded) would run on all runners — fail it.
        const aptSteps = buildSteps.filter(
            (s) => typeof s.run === "string" && s.run.includes("apt-get"),
        );
        for (const step of aptSteps) {
            const cond = String(step.if ?? "");
            expect(cond).toContain("runner.os");
            expect(cond).toMatch(/Linux/);
        }
    });

    // Arm 8 (S2 fix): any job with no actions/checkout step whose steps invoke `gh` must set
    // GH_REPO. `gh` with no repo context fails "fatal: not a git repository" (run 32543662094),
    // and that failure hits both branches of a release step — `gh release view` fails identically,
    // which is why the run took the else branch at all. GH_REPO lets `gh` resolve the repo from the
    // env and verify the tag against the remote API rather than a local clone, so no checkout is
    // needed in a job whose only inputs are downloaded artifacts. The property is stated over every
    // job in the workflow — a second gh-invoking job added later with no GH_REPO and no checkout
    // must trip it, not just the one named `checksums`.
    test("every gh-invoking job without a checkout sets GH_REPO", () => {
        const wf = loadWorkflowYaml();
        assertGhRepoOnNoCheckoutJobs(wf);
    });

    // Arm 8 fixture (criterion 2c): a hypothetical second gh-invoking job that sets neither
    // GH_REPO nor a checkout must trip the property. Taken against an in-test fixture string —
    // never by editing the real release.yml into a state to undo.
    test("a second gh-invoking job without GH_REPO or checkout reds the property", () => {
        const fixtureYaml = `
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
  checksums:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Attach archives to release
        env:
          GH_TOKEN: \${{ secrets.GITHUB_TOKEN }}
          GH_REPO: \${{ github.repository }}
        run: gh release upload v1.0.0 archives/*.tar.gz
  rogue:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Rogue gh step
        env:
          GH_TOKEN: \${{ secrets.GITHUB_TOKEN }}
        run: gh release create v1.0.0 --verify-tag
`;
        const wf = yaml.load(fixtureYaml) as Record<string, unknown>;
        expect(() => assertGhRepoOnNoCheckoutJobs(wf)).toThrow();
    });
});

/**
 * Property: every job in the workflow that (a) has no `actions/checkout` step and (b) has at
 * least one step whose `run` invokes `gh`, must set `GH_REPO` in either the job-level `env` or
 * the `env` of a `gh`-invoking step. Without `GH_REPO`, `gh` has no repo context and fails
 * "fatal: not a git repository" — the defect that prevented any shallot version from publishing
 * by CI (run 32543662094). Stated over all jobs, not one by name, so a second `gh`-invoking job
 * added later trips it.
 */
function assertGhRepoOnNoCheckoutJobs(wf: Record<string, unknown>): void {
    const jobs = wf.jobs as Record<string, Record<string, unknown>>;
    expect(jobs).toBeDefined();
    for (const [jobName, job] of Object.entries(jobs)) {
        const steps = (job.steps as Record<string, unknown>[] | undefined) ?? [];

        // Does this job have an actions/checkout step?
        const hasCheckout = steps.some((s) => String(s.uses ?? "").startsWith("actions/checkout"));
        if (hasCheckout) continue;

        // Does any step invoke `gh`?
        const ghSteps = steps.filter(
            (s) => typeof s.run === "string" && /\bgh\b/.test(s.run as string),
        );
        if (ghSteps.length === 0) continue;

        // The job must set GH_REPO — either at the job level or in a gh-invoking step's env.
        const jobEnv = (job.env as Record<string, unknown> | undefined) ?? {};
        const ghStepsWithRepo = ghSteps.filter((s) => {
            const stepEnv = (s.env as Record<string, unknown> | undefined) ?? {};
            return String(stepEnv.GH_REPO ?? "").trim() !== "";
        });
        const jobHasRepo = String(jobEnv.GH_REPO ?? "").trim() !== "";

        if (!jobHasRepo && ghStepsWithRepo.length === 0) {
            throw new Error(
                `job "${jobName}" invokes gh without a checkout step but sets no GH_REPO env`,
            );
        }
    }
}
