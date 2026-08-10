import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
    cargoTarget,
    dropSwiftshader,
    findCefDir,
    isStaleLocaleEntry,
    macHelperBin,
    macInfoPlist,
    missingCrateDiagnostic,
    nativeOutDir,
    resolveCargoInvocation,
} from "./native";

describe("nativeOutDir", () => {
    test("builds <projectDir>/build/<platform>/<profile>-<mode>", () => {
        expect(nativeOutDir("/proj", "mac", true, true)).toBe(
            resolve("/proj", "build", "mac", "release-portable"),
        );
    });

    test("release/portable each independently flip their segment", () => {
        expect(nativeOutDir("/proj", "windows", false, false)).toBe(
            resolve("/proj", "build", "windows", "debug-system"),
        );
        expect(nativeOutDir("/proj", "windows", true, false)).toBe(
            resolve("/proj", "build", "windows", "release-system"),
        );
        expect(nativeOutDir("/proj", "windows", false, true)).toBe(
            resolve("/proj", "build", "windows", "debug-portable"),
        );
    });

    // the mode segment exists precisely so a portable and a system build of the same project+profile
    // don't share a path and clobber each other's output.
    test("portable and system builds of the same profile land in different dirs", () => {
        const system = nativeOutDir("/proj", "linux", true, false);
        const portable = nativeOutDir("/proj", "linux", true, true);
        expect(system).not.toBe(portable);
    });
});

describe("cargoTarget", () => {
    test("windows targets get an .exe suffix, others don't", () => {
        expect(cargoTarget("x86_64-pc-windows-msvc", true, "/t")).toBe(
            resolve("/t", "x86_64-pc-windows-msvc/release/shallot-window.exe"),
        );
        expect(cargoTarget("x86_64-unknown-linux-gnu", true, "/t")).toBe(
            resolve("/t", "x86_64-unknown-linux-gnu/release/shallot-window"),
        );
    });

    test("release selects the release profile dir, debug the debug dir", () => {
        expect(cargoTarget("aarch64-apple-darwin", false, "/t")).toBe(
            resolve("/t", "aarch64-apple-darwin/debug/shallot-window"),
        );
    });
});

describe("resolveCargoInvocation", () => {
    // the decision table cargoBuild used to inline: msvc + WSL + portable is the one combination that
    // needs the host's native MSVC toolchain via PowerShell (devShellBuild) rather than cargo/cargo-xwin.
    test("msvc target, on WSL, portable -> devshell", () => {
        const inv = resolveCargoInvocation("x86_64-pc-windows-msvc", false, true, true);
        expect(inv.kind).toBe("devshell");
    });

    test("msvc target, on WSL, system (not portable) -> xwin, not devshell", () => {
        const inv = resolveCargoInvocation("x86_64-pc-windows-msvc", false, false, true);
        expect(inv.kind).toBe("xwin");
    });

    test("msvc target, not on WSL, portable -> xwin (cross-compiled)", () => {
        const inv = resolveCargoInvocation("x86_64-pc-windows-msvc", false, true, false);
        expect(inv.kind).toBe("xwin");
    });

    test("non-msvc target is always native, regardless of WSL/portable", () => {
        expect(resolveCargoInvocation("x86_64-unknown-linux-gnu", false, true, true).kind).toBe(
            "native",
        );
        expect(resolveCargoInvocation("aarch64-apple-darwin", true, false, false).kind).toBe(
            "native",
        );
    });

    test("flags carry --target always, --release only when release, and the portable feature flags only when portable", () => {
        expect(
            resolveCargoInvocation("x86_64-unknown-linux-gnu", false, false, false).flags,
        ).toEqual(["--target", "x86_64-unknown-linux-gnu"]);
        expect(
            resolveCargoInvocation("x86_64-unknown-linux-gnu", true, false, false).flags,
        ).toEqual(["--target", "x86_64-unknown-linux-gnu", "--release"]);
        expect(
            resolveCargoInvocation("x86_64-unknown-linux-gnu", false, true, false).flags,
        ).toEqual([
            "--target",
            "x86_64-unknown-linux-gnu",
            "--no-default-features",
            "--features",
            "portable",
        ]);
    });
});

describe("missingCrateDiagnostic", () => {
    test("null when the crate dir exists", () => {
        const dir = mkdtempSync(join(tmpdir(), "shallot-crate-"));
        expect(missingCrateDiagnostic(dir)).toBeNull();
    });

    test("names the missing dir when it doesn't exist", () => {
        const dir = join(mkdtempSync(join(tmpdir(), "shallot-crate-")), "rust/window");
        const msg = missingCrateDiagnostic(dir);
        expect(msg).not.toBeNull();
        expect(msg).toContain(dir);
        expect(msg).toContain("native builds aren't available from an installed package");
    });
});

describe("isStaleLocaleEntry", () => {
    // CefSettings.locale defaults to en-US, so both spellings of the English .lproj bundle survive —
    // the regionless en.lproj and the region-qualified en_US.lproj.
    test("keeps both English .lproj spellings, strips every other .lproj, in the bundle scope", () => {
        expect(isStaleLocaleEntry("en.lproj", "bundle")).toBe(false);
        expect(isStaleLocaleEntry("en_US.lproj", "bundle")).toBe(false);
        expect(isStaleLocaleEntry("fr.lproj", "bundle")).toBe(true);
        expect(isStaleLocaleEntry("de_DE.lproj", "bundle")).toBe(true);
    });

    test("keeps en-US.pak, strips every other .pak, in the locales scope", () => {
        expect(isStaleLocaleEntry("en-US.pak", "locales")).toBe(false);
        expect(isStaleLocaleEntry("ja.pak", "locales")).toBe(true);
    });

    test("ignores entries that are neither shape", () => {
        expect(isStaleLocaleEntry("icudtl.dat", "bundle")).toBe(false);
        expect(isStaleLocaleEntry("README", "locales")).toBe(false);
    });

    // Resources/ never holds per-locale .pak files on mac (those live flat under Resources/locales/),
    // so the .pak arm has zero legitimate targets there — only the required CEF resource paks
    // (chrome_100_percent.pak, chrome_200_percent.pak, resources.pak) that native.ts declares mandatory
    // for the Linux/Windows CEF payloads. The bundle scope must never delete a .pak entry.
    test("never strips a required CEF resource pak in the bundle scope", () => {
        expect(isStaleLocaleEntry("chrome_100_percent.pak", "bundle")).toBe(false);
        expect(isStaleLocaleEntry("chrome_200_percent.pak", "bundle")).toBe(false);
        expect(isStaleLocaleEntry("resources.pak", "bundle")).toBe(false);
    });
});

describe("macHelperBin", () => {
    test("resolves under the crate's target dir for the given profile", () => {
        expect(macHelperBin(true)).toContain("target/aarch64-apple-darwin/release/shallot-helper");
        expect(macHelperBin(false)).toContain("target/aarch64-apple-darwin/debug/shallot-helper");
    });
});

describe("macInfoPlist", () => {
    const base = { executable: "demo", bundleName: "demo", identifier: "com.multiplekex.demo" };

    test("a helper bundle gets LSUIElement, a main app doesn't", () => {
        const helper = macInfoPlist({ ...base, helper: true, icon: false });
        const main = macInfoPlist({ ...base, helper: false, icon: false });
        expect(helper).toContain("<key>LSUIElement</key>\n    <true/>");
        expect(main).not.toContain("LSUIElement");
    });

    test("icon: true carries the CFBundleIconFile key, icon: false omits it", () => {
        const withIcon = macInfoPlist({ ...base, helper: false, icon: true });
        const withoutIcon = macInfoPlist({ ...base, helper: false, icon: false });
        expect(withIcon).toContain("<key>CFBundleIconFile</key>\n    <string>app</string>");
        expect(withoutIcon).not.toContain("CFBundleIconFile");
    });

    test("the two branches compose independently (helper + icon both set)", () => {
        const both = macInfoPlist({ ...base, helper: true, icon: true });
        expect(both).toContain("LSUIElement");
        expect(both).toContain("CFBundleIconFile");
    });
});

describe("findCefDir", () => {
    test("CEF_PATH short-circuits the search when the marker exists there", () => {
        const cefPath = mkdtempSync(join(tmpdir(), "shallot-cef-path-"));
        writeFileSync(join(cefPath, "libcef.so"), "");
        const prev = process.env.CEF_PATH;
        process.env.CEF_PATH = cefPath;
        try {
            expect(findCefDir("x86_64-unknown-linux-gnu", "libcef.so", "/nonexistent")).toBe(
                cefPath,
            );
        } finally {
            if (prev === undefined) delete process.env.CEF_PATH;
            else process.env.CEF_PATH = prev;
        }
    });

    test("CEF_PATH set but missing the marker falls through to the build-tree search", () => {
        const prev = process.env.CEF_PATH;
        process.env.CEF_PATH = mkdtempSync(join(tmpdir(), "shallot-cef-empty-"));
        try {
            const targetDir = mkdtempSync(join(tmpdir(), "shallot-cef-tree-"));
            const out = resolve(
                targetDir,
                "x86_64-unknown-linux-gnu/release/build/cef-dll-sys-abc123/out/cef-linux64",
            );
            mkdirSync(out, { recursive: true });
            writeFileSync(join(out, "libcef.so"), "");
            expect(findCefDir("x86_64-unknown-linux-gnu", "libcef.so", targetDir)).toBe(out);
        } finally {
            if (prev === undefined) delete process.env.CEF_PATH;
            else process.env.CEF_PATH = prev;
        }
    });

    test("searches release before debug, and returns null when neither has the marker", () => {
        const prev = process.env.CEF_PATH;
        delete process.env.CEF_PATH;
        try {
            const targetDir = mkdtempSync(join(tmpdir(), "shallot-cef-tree-"));
            const releaseOut = resolve(
                targetDir,
                "aarch64-apple-darwin/release/build/cef-dll-sys-1/out/cef-mac",
            );
            const debugOut = resolve(
                targetDir,
                "aarch64-apple-darwin/debug/build/cef-dll-sys-2/out/cef-mac",
            );
            mkdirSync(releaseOut, { recursive: true });
            mkdirSync(debugOut, { recursive: true });
            writeFileSync(join(releaseOut, "marker.txt"), "");
            writeFileSync(join(debugOut, "marker.txt"), "");
            expect(findCefDir("aarch64-apple-darwin", "marker.txt", targetDir)).toBe(releaseOut);

            const emptyTargetDir = mkdtempSync(join(tmpdir(), "shallot-cef-empty-tree-"));
            expect(findCefDir("aarch64-apple-darwin", "marker.txt", emptyTargetDir)).toBeNull();
        } finally {
            if (prev === undefined) delete process.env.CEF_PATH;
            else process.env.CEF_PATH = prev;
        }
    });
});

describe("dropSwiftshader", () => {
    // read per call, not frozen at module load — a caller (a test, a script driving multiple builds)
    // that sets the env var mid-process must be honored rather than seeing whatever it was on import.
    test("reflects the env var's current value, not its value at import time", () => {
        const prev = process.env.SHALLOT_DROP_SWIFTSHADER;
        try {
            delete process.env.SHALLOT_DROP_SWIFTSHADER;
            expect(dropSwiftshader()).toBe(false);
            process.env.SHALLOT_DROP_SWIFTSHADER = "1";
            expect(dropSwiftshader()).toBe(true);
            delete process.env.SHALLOT_DROP_SWIFTSHADER;
            expect(dropSwiftshader()).toBe(false);
        } finally {
            if (prev === undefined) delete process.env.SHALLOT_DROP_SWIFTSHADER;
            else process.env.SHALLOT_DROP_SWIFTSHADER = prev;
        }
    });
});
