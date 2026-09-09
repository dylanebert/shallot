import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { requiredFeatures, verdict, WEBVIEW_UNSUPPORTED } from "./features";

const dirs: string[] = [];
function temporary() {
    const dir = mkdtempSync(join(tmpdir(), "shallot-features-"));
    dirs.push(dir);
    return dir;
}
afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("verdict", () => {
    test("the same rule covers every backend row and required versus preferred gaps", () => {
        const rows = {
            unavailable: null,
            populated: ["timestamp-query", "subgroups"],
            complete: [],
        };
        try {
            Object.assign(WEBVIEW_UNSUPPORTED, rows);
            expect(Object.keys(WEBVIEW_UNSUPPORTED).sort()).toEqual([
                "complete",
                "linux",
                "mac",
                "populated",
                "unavailable",
                "windows",
            ]);
            for (const target of Object.keys(WEBVIEW_UNSUPPORTED)) {
                for (const required of [[], ["timestamp-query"], ["unrelated-feature"]]) {
                    expect(verdict(target, true, required)).toEqual([]);
                    const refused =
                        target === "linux" ||
                        target === "unavailable" ||
                        (target === "populated" && required.includes("timestamp-query"));
                    expect(verdict(target, false, required).length > 0).toBe(refused);
                }
            }
        } finally {
            for (const target of Object.keys(rows)) delete WEBVIEW_UNSUPPORTED[target];
        }
    });

    test("portable is always clean — CEF ships its own Chromium", () => {
        expect(verdict("linux", true, ["subgroups"])).toEqual([]);
        expect(verdict("mac", true, ["subgroups"])).toEqual([]);
        expect(verdict("windows", true, ["subgroups"])).toEqual([]);
    });

    test("linux default refuses — WebKitGTK has no usable WebGPU", () => {
        expect(verdict("linux", false, []).length).toBeGreaterThan(0);
        expect(verdict("linux", false, []).join(" ")).toContain("--portable");
    });

    test("mac default is clean — subgroups is preferred (LDS fallback), not a required gap", () => {
        // subgroups is a preferred feature, so it never reaches `required`; and mac WKWebView has no
        // standing required-feature gap, so even passing it in refuses nothing.
        expect(verdict("mac", false, ["subgroups"])).toEqual([]);
        expect(verdict("mac", false, [])).toEqual([]);
    });

    test("windows default is clean — WebView2 is full Chromium", () => {
        expect(verdict("windows", false, ["subgroups"])).toEqual([]);
        expect(verdict("windows", false, [])).toEqual([]);
    });
});

describe("requiredFeatures", () => {
    function project(manifest: object): string {
        const dir = temporary();
        writeFileSync(join(dir, "shallot.json"), JSON.stringify(manifest));
        return dir;
    }

    test("a physics project requires nothing beyond the base floor — subgroups is preferred", async () => {
        // the BVH broadphase prefers subgroups but falls back to LDS, so physics lists it as a
        // preferredFeature, never a required one. requiredFeatures surfaces only hard requirements.
        const dir = project({ plugins: { Physics: true } });
        expect(await requiredFeatures(dir)).toEqual([]);
    });

    test("a default (physics-free) project requires nothing beyond the base floor", async () => {
        const dir = project({ plugins: {} });
        expect(await requiredFeatures(dir)).toEqual([]);
    });

    test("a Profile project requires timestamp-query — it left the floor for the plugin", async () => {
        // the profiler is the one plugin with a hard requirement beyond the floor; a project that
        // doesn't enable it never requests the feature at all.
        const dir = project({ plugins: { Profile: true } });
        expect(await requiredFeatures(dir)).toEqual(["timestamp-query"]);
    });
});
