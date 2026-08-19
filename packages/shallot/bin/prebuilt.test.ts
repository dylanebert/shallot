import { describe, expect, test } from "bun:test";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
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
});
