import { describe, expect, test } from "bun:test";
import {
    anchorNames,
    bareLines,
    descText,
    docTargets,
    formatBaseline,
    holeLines,
    leakLines,
    linkLines,
    markerSubsystem,
    parseBaseline,
    prose,
    proseFindings,
    proseText,
    requiredSubsystems,
    sentenceCount,
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

describe("doc: link resolution", () => {
    const valid = new Set(["guide/quick-start", "engine/scene"]);

    test("resolves a target that names a known page slug", () => {
        const src = "see [scenes](doc:engine/scene#load) for the loader";
        expect(linkLines(src, valid, "guide/quick-start.md")).toEqual([]);
    });

    test("flags a target with no matching page", () => {
        const src = "[obby](doc:guide/make-a-game) walks through it";
        expect(linkLines(src, valid, "guide/quick-start.md")).toEqual([
            "docs/guide/quick-start.md: broken doc: link → guide/make-a-game",
        ]);
    });

    test("ignores a doc: syntax example inside a fence", () => {
        const src = "syntax:\n```md\n[x](doc:not/a/page)\n```\n";
        expect(docTargets(src)).toEqual([]);
    });
});

describe("prose gate", () => {
    test("flags an em dash but not one inside fenced code", () => {
        expect(proseFindings(proseText("the pass reads positions — only that"), "p.md")).toEqual([
            "prose p.md emdash",
        ]);
        expect(proseFindings(proseText("intro\n```ts\na — b\n```\ndone"), "p.md")).toEqual([]);
    });

    test("flags a banned word on a boundary, ignores it inside inline code", () => {
        expect(proseFindings(proseText("this can utilize the buffer"), "p.md")).toEqual([
            "prose p.md banned:utilize",
        ]);
        expect(proseFindings(proseText("call `utilize()` on it"), "p.md")).toEqual([]);
    });

    test("flags an editorial opener", () => {
        expect(proseFindings(proseText("In order to move it, set position"), "p.md")).toEqual([
            "prose p.md opener:in-order-to",
        ]);
    });

    test("reads rendered JSDoc summaries, code spans stripped", () => {
        const html = `<p class="ref-desc">a seamless <code>a—b</code> pass</p>`;
        expect(proseFindings(descText(html), "p.md")).toEqual(["prose p.md banned:seamless"]);
    });
});

describe("sentence budget", () => {
    test("counts terminators, not identifier or version dots", () => {
        expect(sentenceCount("Set `Orbit.sensitivity`. Ship v0.6.0 now. Done.")).toBe(3);
    });

    test("does not count e.g./i.e. as sentence ends", () => {
        expect(sentenceCount("Tune it, e.g. the damping, and go.")).toBe(1);
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
