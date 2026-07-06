import { describe, expect, test } from "bun:test";
import { buildIndex, docFor, parseBlocks, search } from "./docs";

// fixtures mimic docs/dist: frontmatter + render-ready HTML with tab markers + ref anchors
const ORBIT = `---
title: Orbit
description: orbit camera
source: extras/orbit
icon: orbit
order: 3
---

<!-- tabs -->
<!-- tab: Editor -->
<p>Third-person camera with orbit and fly.</p>
<!-- tab: Code -->
<p>Add <code>OrbitPlugin</code>.</p>
<!-- tab: Internals -->
<div class="ref-list"><div class="ref-item" id="ref-Orbit"><code>Orbit</code></div><div class="ref-item" id="ref-OrbitPlugin"><code>OrbitPlugin</code></div></div>
<!-- /tabs -->
`;

const LOADING = `---
title: Loading
order: 1
---

<!-- tabs -->
<!-- tab: Code -->
<!-- os -->
<!-- os: Windows -->
<p>windows path</p>
<!-- os: Mac/Linux -->
<p>unix path</p>
<!-- /os -->
<!-- /tabs -->
`;

const LIGHTING = `---
title: Lighting
source: standard/render
order: 2
---

<div class="ref-item" id="ref-DirectionalLight"><code>DirectionalLight</code></div>
`;

const HOME = `---
title: Home
---

<p>welcome</p>
`;

const corpus: Record<string, string> = {
    "/x/y/docs/dist/extras/orbit.md": ORBIT,
    "/x/y/docs/dist/standard/loading.md": LOADING,
    "/x/y/docs/dist/standard/lighting.md": LIGHTING,
    "/x/y/docs/dist/index.md": HOME,
};

describe("parseBlocks", () => {
    test("splits audience tabs and collects their names", () => {
        const { blocks, tabNames } = parseBlocks(ORBIT.replace(/^---[\s\S]*?---\n/, ""));
        expect(tabNames).toEqual(["Editor", "Code", "Internals"]);
        const group = blocks.find((b) => b.type === "tabs");
        expect(group?.type).toBe("tabs");
        if (group?.type === "tabs") {
            expect(group.tabs.map((t) => t.name)).toEqual(["Editor", "Code", "Internals"]);
            const ref = group.tabs[2].blocks[0];
            expect(ref.type).toBe("content");
            if (ref.type === "content") expect(ref.html).toContain('id="ref-Orbit"');
        }
    });

    test("expands inline os tabs inside an audience tab", () => {
        const { blocks } = parseBlocks(LOADING.replace(/^---[\s\S]*?---\n/, ""));
        const group = blocks.find((b) => b.type === "tabs");
        if (group?.type !== "tabs") throw new Error("expected tab group");
        const inline = group.tabs[0].blocks.find((b) => b.type === "inline");
        expect(inline?.type).toBe("inline");
        if (inline?.type === "inline")
            expect(inline.tabs.map((t) => t.name)).toEqual(["Windows", "Mac/Linux"]);
    });

    test("plain HTML with no markers is a single content block", () => {
        const { blocks, tabNames } = parseBlocks("<p>just prose</p>");
        expect(tabNames).toEqual([]);
        expect(blocks).toEqual([{ type: "content", html: "<p>just prose</p>" }]);
    });
});

describe("buildIndex", () => {
    const index = buildIndex(corpus);

    test("drops the index page and sorts by frontmatter order", () => {
        expect(index.pages.map((p) => p.slug)).toEqual([
            "standard/loading",
            "standard/lighting",
            "extras/orbit",
        ]);
    });

    test("orders categories guide → engine → standard → extras → editor, not by slug", () => {
        const ordered = buildIndex({
            "/x/y/docs/dist/engine/ecs.md": "---\ntitle: ECS\norder: 0\n---\n\n<p>ecs</p>\n",
            "/x/y/docs/dist/guide/quick-start.md":
                "---\ntitle: Quick Start\norder: 0\n---\n\n<p>start</p>\n",
            "/x/y/docs/dist/extras/orbit.md": "---\ntitle: Orbit\n---\n\n<p>orbit</p>\n",
            "/x/y/docs/dist/standard/render.md": "---\ntitle: Render\n---\n\n<p>render</p>\n",
        });
        // guide precedes engine despite both order 0 and slug "engine" < "guide"
        expect(ordered.pages.map((p) => p.group)).toEqual([
            "guide",
            "engine",
            "standard",
            "extras",
        ]);
    });

    test("reads frontmatter and derives group from the slug", () => {
        const orbit = index.pages.find((p) => p.slug === "extras/orbit")!;
        expect(orbit.title).toBe("Orbit");
        expect(orbit.source).toBe("extras/orbit");
        expect(orbit.icon).toBe("orbit");
        expect(orbit.group).toBe("extras");
    });

    test("collects reference symbols with page + anchor", () => {
        const orbit = index.symbols.filter((s) => s.slug === "extras/orbit").map((s) => s.name);
        expect(orbit).toEqual(["Orbit", "OrbitPlugin"]);
        expect(index.symbols.find((s) => s.name === "Orbit")?.anchor).toBe("ref-Orbit");
    });
});

describe("search", () => {
    const index = buildIndex(corpus);

    test("an exact name ranks first", () => {
        expect(search(index, "Orbit")[0].title).toBe("Orbit");
    });

    test("resolves a reference symbol by name", () => {
        const hit = search(index, "OrbitPlugin").find((h) => h.kind === "symbol");
        expect(hit?.slug).toBe("extras/orbit");
        expect(hit?.anchor).toBe("ref-OrbitPlugin");
    });

    test("matches body text, not just titles", () => {
        expect(search(index, "fly").some((h) => h.slug === "extras/orbit")).toBe(true);
    });

    test("empty query returns nothing", () => {
        expect(search(index, "   ")).toEqual([]);
    });
});

describe("docFor", () => {
    const index = buildIndex(corpus);

    test("maps a component to its reference symbol + anchor", () => {
        expect(docFor(index, "orbit")).toEqual({ slug: "extras/orbit", anchor: "ref-Orbit" });
    });

    test("PascalCases a kebab component name", () => {
        expect(docFor(index, "directional-light")).toEqual({
            slug: "standard/lighting",
            anchor: "ref-DirectionalLight",
        });
    });

    test("falls back to a slug match when no symbol documents it", () => {
        expect(docFor(index, "loading")).toEqual({ slug: "standard/loading" });
    });

    test("returns null for an undocumented component", () => {
        expect(docFor(index, "nonexistent")).toBeNull();
    });
});
