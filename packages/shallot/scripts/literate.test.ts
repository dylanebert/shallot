import { describe, expect, test } from "bun:test";
import { composePage, type DocBlock, type PageEntry, parseDocBlocks } from "./literate";

describe("parseDocBlocks", () => {
    test("reads kind + prose, stripping the comment prefix", () => {
        const src = ["// #doc:intro", "// the lead.", "//", "// a second paragraph."].join("\n");
        expect(parseDocBlocks(src, null)).toEqual([
            { kind: "intro", prose: "the lead.\n\na second paragraph.", example: null, page: null },
        ]);
    });

    test("a markdown heading inside a block is prose, not a terminator", () => {
        const src = ["// #doc:code", "// ### Pan Offset", "//", "// pan adds an offset."].join(
            "\n",
        );
        expect(parseDocBlocks(src, null)[0].prose).toBe("### Pan Offset\n\npan adds an offset.");
    });

    test("an explicit source: pairs a cross-file snippet", () => {
        const src = ["// #doc:code source:orbit/public/scenes/orbit.scene", "// ### Scene"].join(
            "\n",
        );
        expect(parseDocBlocks(src, "orbit/src/tune.ts")).toEqual([
            {
                kind: "code",
                prose: "### Scene",
                example: "orbit/public/scenes/orbit.scene",
                page: null,
            },
        ]);
    });

    test("a page: tag routes a block when a specimen feeds several pages", () => {
        const src = ["// #doc:code page:standard/surfaces", "// custom shading."].join("\n");
        expect(parseDocBlocks(src, null)[0].page).toBe("standard/surfaces");
    });

    test("source: and page: tags combine on one block, order-independent", () => {
        const src = ["// #doc:code source:kitchen/scenes/a.scene page:standard/rendering"].join(
            "\n",
        );
        expect(parseDocBlocks(src, "kitchen/src/x.ts")[0]).toEqual({
            kind: "code",
            prose: "",
            example: "kitchen/scenes/a.scene",
            page: "standard/rendering",
        });
    });

    test("a trailing // #region pairs the same-file region by the specimen path", () => {
        const src = ["// #doc:code", "// tune feel here:", "// #region tune", "const x = 1;"].join(
            "\n",
        );
        expect(parseDocBlocks(src, "orbit/src/tune.ts")[0]).toEqual({
            kind: "code",
            prose: "tune feel here:",
            example: "orbit/src/tune.ts#tune",
            page: null,
        });
    });

    test("implicit region pairing is off for non-specimen files (null path)", () => {
        const src = ["// #doc:dev", "// internals.", "// #region helper", "const x = 1;"].join(
            "\n",
        );
        expect(parseDocBlocks(src, null)[0].example).toBeNull();
    });

    test("a blank line before a region leaves the block prose-only", () => {
        const src = ["// #doc:code", "// just prose.", "", "// #region tune"].join("\n");
        expect(parseDocBlocks(src, "orbit/src/tune.ts")[0].example).toBeNull();
    });

    test("collects every block in file order across kinds", () => {
        const src = [
            "// #doc:intro",
            "// lead.",
            "",
            "// #doc:code",
            "// usage.",
            "",
            "// #doc:dev",
            "// internals.",
        ].join("\n");
        expect(parseDocBlocks(src, null).map((b) => b.kind)).toEqual(["intro", "code", "dev"]);
    });

    test("a non-comment line ends a block; later code is skipped", () => {
        const src = ["// #doc:code", "// usage.", "const x = 1;", "// a bare code comment."].join(
            "\n",
        );
        const blocks = parseDocBlocks(src, null);
        expect(blocks).toHaveLength(1);
        expect(blocks[0].prose).toBe("usage.");
    });
});

describe("composePage", () => {
    const block = (over: Partial<DocBlock>): DocBlock => ({
        kind: "code",
        prose: "",
        example: null,
        page: null,
        ...over,
    });
    const entry = (over: Partial<PageEntry>): PageEntry => ({
        slug: "extras/orbit",
        title: "Orbit",
        description: "orbit camera",
        icon: "orbit",
        source: "extras/orbit",
        specimen: "orbit",
        ...over,
    });

    test("one-to-one: single source emits the frontmatter source + one API marker", () => {
        const page = composePage(
            entry({}),
            [block({ kind: "intro", prose: "the lead." }), block({ prose: "usage." })],
            [],
        );
        expect(page).toContain("source: extras/orbit");
        expect(page).toContain("<!-- API:extras/orbit -->");
        expect(page).toContain("the lead.");
        expect(page).toContain("usage.");
        // no #doc:dev → single-audience page, no tab chrome
        expect(page).not.toContain("<!-- tabs -->");
    });

    test("specimen-only page (a guide): walkthrough alone, no source frontmatter, no reference table", () => {
        const page = composePage(
            entry({ slug: "guide/make-a-game", source: undefined, specimen: "obby" }),
            [block({ kind: "intro", prose: "build a game." }), block({ prose: "add a system." })],
            [],
        );
        expect(page).toContain("build a game.");
        expect(page).toContain("add a system.");
        expect(page).not.toMatch(/^source:/m);
        expect(page).not.toContain("<!-- API:");
        expect(page).not.toContain("<!-- CORE:");
        expect(page).not.toContain("<!-- tabs -->");
    });

    test("order rides frontmatter when set, absent otherwise — the nav sort key across entries", () => {
        expect(composePage(entry({ order: 3 }), [], [])).toContain("order: 3");
        expect(composePage(entry({}), [], [])).not.toMatch(/^order:/m);
    });

    test("multi-source: comma-joined API/CORE markers, single source frontmatter omitted", () => {
        const page = composePage(
            entry({ slug: "standard/rendering", source: ["standard/render", "standard/sear"] }),
            [block({ prose: "compose surfaces." })],
            [block({ kind: "dev", prose: "the froxel grid." })],
        );
        expect(page).not.toMatch(/^source:/m);
        expect(page).toContain("<!-- API:standard/render,standard/sear -->");
        // CORE keys by leaf segment, matching coreExports + the hand-authored pages
        expect(page).toContain("<!-- CORE:render,sear -->");
        expect(page).toContain("<!-- tabs -->");
        expect(page).toContain("the froxel grid.");
    });

    test("core decouples the CORE table from the author API sources", () => {
        // rendering: four author API modules, one extension story (render's /core)
        const page = composePage(
            entry({
                slug: "standard/rendering",
                source: ["standard/render", "standard/part", "standard/sear", "standard/glaze"],
                core: "standard/render",
            }),
            [block({ prose: "compose the scene." })],
            [block({ kind: "dev", prose: "the contract." })],
        );
        expect(page).toContain(
            "<!-- API:standard/render,standard/part,standard/sear,standard/glaze -->",
        );
        // CORE renders only render — NOT sear/part/glaze, even though sear is an API source
        expect(page).toContain("<!-- CORE:render -->");
        expect(page).not.toContain("CORE:sear");
        expect(page).toContain("<!-- tabs -->");
    });

    test("extender-only page: core with no author source is single-tab, CORE inline", () => {
        // surfaces: sear's /core, no author API of its own → no tab chrome
        const page = composePage(
            entry({ slug: "standard/surfaces", source: undefined, core: "standard/sear" }),
            [block({ prose: "author a surface." })],
            [block({ kind: "dev", prose: "the surface environment." })],
        );
        expect(page).not.toContain("<!-- tabs -->");
        expect(page).not.toMatch(/<!-- API:/);
        expect(page).not.toMatch(/^source:/m);
        expect(page).toContain("author a surface.");
        expect(page).toContain("the surface environment.");
        expect(page).toContain("<!-- CORE:sear -->");
    });

    test("page routing: a block routes only to its page slug; untagged blocks route everywhere", () => {
        const shared = [
            block({ prose: "shared intro.", kind: "intro" }),
            block({ prose: "rendering only.", page: "standard/rendering" }),
            block({ prose: "surfaces only.", page: "standard/surfaces" }),
        ];
        const rendering = composePage(entry({ slug: "standard/rendering" }), shared, []);
        expect(rendering).toContain("shared intro.");
        expect(rendering).toContain("rendering only.");
        expect(rendering).not.toContain("surfaces only.");

        const surfaces = composePage(entry({ slug: "standard/surfaces" }), shared, []);
        expect(surfaces).toContain("shared intro.");
        expect(surfaces).toContain("surfaces only.");
        expect(surfaces).not.toContain("rendering only.");
    });
});
