import { describe, expect, test } from "bun:test";
import {
    anchorNames,
    bareLines,
    formatBaseline,
    holeLines,
    leakLines,
    markerSubsystem,
    parseBaseline,
    prose,
    requiredSubsystems,
} from "./docs-check";

describe("bare-entry", () => {
    // bare = no rendered summary (no `<p class="ref-desc">`), whether a plain `<div class="ref-item">`
    // or an entry that expands to a field/option/parts table. a member renders `ref-item ref-method`
    // (no id), so it's excluded.
    const documented = `<details class="ref-entry" id="ref-mesh"><summary>...</summary><p class="ref-desc">builds a mesh</p></details>`;
    const member = `<div class="ref-item ref-method"><code>.query</code></div>`;
    const bare = `<div class="ref-item" id="ref-parse"><code>parse</code></div>`;
    // a component/plugin with a body table but no summary is still bare — the table isn't its description
    const tableOnly = `<details class="ref-entry ref-group" id="ref-Foo"><summary>...</summary><div class="ref-methods"><table class="ref-fields"></table></div></details>`;

    test("flags an export with no description", () => {
        expect(bareLines(bare, "engine/scene.md")).toEqual(["bare engine/scene.md parse"]);
    });

    test("flags a table-only entry with no summary", () => {
        expect(bareLines(tableOnly, "extras/foo.md")).toEqual(["bare extras/foo.md Foo"]);
    });

    test("ignores documented exports and members", () => {
        expect(bareLines(documented + member, "standard/render.md")).toEqual([]);
    });
});

describe("leak detection", () => {
    const html = `<details id="ref-whenLoaded">..</details><div id="ref-play">..</div>`;
    const names = anchorNames(html);

    test("flags a prose restatement of a reference entry", () => {
        const page = "call `whenLoaded(id)` before reading the buffer";
        expect(leakLines(prose(page), names, "standard/audio.md")).toEqual([
            "keep standard/audio.md whenLoaded",
        ]);
    });

    test("ignores fenced code and unmentioned entries", () => {
        // the name appears only inside a fenced example, which prose() strips
        const page = "intro\n```ts\nplay(sound)\n```\nno inline mention";
        expect(leakLines(prose(page), names, "standard/audio.md")).toEqual([]);
    });
});

describe("subsystem coverage", () => {
    const exports = {
        ".": "./src/index.ts",
        "./extras": "./src/extras/index.ts",
        "./glaze": "./src/standard/glaze/index.ts",
        "./physics/core": "./src/standard/physics/core.ts",
        "./bvh/core": "./src/standard/bvh/core.ts",
        "./src/*": "./src/*",
    };

    test("derives required subsystems from tier subpaths, skipping barrels and globs", () => {
        expect(requiredSubsystems(exports)).toEqual(new Set(["glaze", "physics", "bvh"]));
    });

    test("a marker covers its subsystem regardless of API/CORE shape", () => {
        expect(markerSubsystem("standard/bvh/core")).toBe("bvh");
        expect(markerSubsystem("render")).toBe("render");
    });

    test("flags a required subsystem with no covering marker", () => {
        const required = requiredSubsystems(exports);
        const covered = new Set(["bvh", "glaze"]); // physics page missing
        expect(holeLines(required, covered)).toEqual(["hole physics"]);
    });

    test("no holes when every subsystem is covered", () => {
        const required = requiredSubsystems(exports);
        expect(holeLines(required, new Set(["glaze", "physics", "bvh"]))).toEqual([]);
    });
});

describe("baseline diff", () => {
    test("roundtrips through format and parse, dropping comments", () => {
        const lines = ["bare engine/scene.md parse", "hole physics", "keep standard/audio.md play"];
        expect([...parseBaseline(formatBaseline(lines))].sort()).toEqual([...lines].sort());
    });

    test("parsed baseline reports membership a drift diff can query", () => {
        const baseline = parseBaseline(formatBaseline(["hole physics"]));
        expect(baseline.has("hole physics")).toBe(true);
        expect(baseline.has("bare engine/app.md run")).toBe(false);
    });
});
