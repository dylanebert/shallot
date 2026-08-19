import { describe, expect, test } from "bun:test";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { gzipSync } from "node:zlib";
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

    test("every fetch result maps to a decision with a non-empty reason", () => {
        const results: PrebuiltFetchResult[] = [
            "ok",
            "not-found",
            "offline",
            "checksum-mismatch",
            "source-checkout",
        ];
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
