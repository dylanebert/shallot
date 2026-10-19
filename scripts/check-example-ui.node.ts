import { expect, setDefaultTimeout, test } from "bun:test";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { parse } from "@babel/parser";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { parseCss, parse as parseSvelte } from "svelte/compiler";
import { colorHex, PALETTES } from "../diagnostics/default-look/palette";

setDefaultTimeout(TIMEOUT.node);

async function sources(directory: string): Promise<string[]> {
    const result: string[] = [];
    for (const entry of await readdir(directory, { withFileTypes: true })) {
        if (["node_modules", "dist", "build", ".artifacts", "test-results"].includes(entry.name))
            continue;
        const path = join(directory, entry.name);
        if (entry.isDirectory()) result.push(...(await sources(path)));
        else if (
            /\.(css|html|svelte|[cm]?[jt]s)$/.test(entry.name) &&
            !/\.(test|node|e2e)\./.test(entry.name)
        )
            result.push(path);
    }
    return result;
}

function titleDeclaration(source: string, offset: number): boolean {
    const before = source.slice(0, offset);
    const tag = before.slice(before.lastIndexOf("<"));
    if (/^<h1\b[^>]*\bstyle\s*=/.test(tag)) return true;
    const open = before.lastIndexOf("{");
    const boundary = Math.max(before.lastIndexOf("}", open), before.lastIndexOf("{", open - 1));
    const selectors = before
        .slice(boundary + 1, open)
        .trim()
        .split(",");
    return (
        open > before.lastIndexOf("}") &&
        selectors.every((selector) => /(?:^|\s|>)h1\s*$/.test(selector))
    );
}

function styleSources(path: string, source: string): string[] {
    if (!/\.[cm]?[jt]s$/.test(path)) return [source];
    const strings: string[] = [];
    const ast = parse(source, { sourceType: "module", plugins: ["typescript"] });
    function visit(value: unknown): void {
        if (!value || typeof value !== "object") return;
        const node = value as Record<string, unknown>;
        if (node.type === "StringLiteral") strings.push(node.value as string);
        else if (node.type === "TemplateElement") strings.push((node.value as { raw: string }).raw);
        else
            for (const child of Object.values(node)) {
                if (Array.isArray(child)) for (const item of child) visit(item);
                else visit(child);
            }
    }
    visit(ast);
    return strings;
}

const carbon = PALETTES["aperture-carbon"];
// The ui tokens, the clear colour and every material's sRGB hex, the accents included.
const PALETTE_COLORS = new Set([
    carbon.ui.panel,
    carbon.ui.ink,
    carbon.ui.border,
    `#${carbon.clearColor.toString(16).padStart(6, "0")}`,
    ...Object.values(carbon.materials).map(colorHex),
]);
// The person's exception to aperture-carbon (strategy/shallot/examples.md, 2026-10-17).
const EXAMPLE_COLORS: Record<string, readonly string[]> = { voxel: ["#89b6e9"] };

const COLOR_FUNCTIONS = /\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color|color-mix)\(([^)]*)\)?/gi;
// A literal colour in hex, three, four, six or eight digits; any other length is not a colour.
const HEX = /#[0-9a-f]+\b/gi;
const KEYWORDS = new Set(["transparent", "currentcolor", "inherit", "none"]);
// A computed part of a cssText template or Svelte style attribute; it is not read.
const COMPUTED_NAME = "--computed";
const COMPUTED = `var(${COMPUTED_NAME})`;

function hexColor(value: string): string | undefined {
    const digits = value.slice(1).toLowerCase();
    if (digits.length === 6 || digits.length === 8) return `#${digits.slice(0, 6)}`;
    if (digits.length === 3 || digits.length === 4)
        return `#${[...digits.slice(0, 3)].map((digit) => digit + digit).join("")}`;
    return undefined;
}

/**
 * The CSS a source writes, as stylelint reads it: a stylesheet, a page's `<style>` text and style
 * attributes, and cssText strings. Scripts are never read; Svelte `style:` directives and DOM
 * `.style` properties set colours from script and are outside this check.
 */
type Css = { sheets: string[]; lists: string[] };

async function htmlCss(source: string): Promise<Css> {
    const css: Css = { sheets: [], lists: [] };
    let sheet = "";
    await new HTMLRewriter()
        .on("style", {
            text(chunk) {
                sheet += chunk.text;
                if (!chunk.lastInTextNode) return;
                css.sheets.push(sheet);
                sheet = "";
            },
        })
        .on("[style]", {
            element(element) {
                css.lists.push(element.getAttribute("style")!);
            },
        })
        .transform(new Response(source))
        .text();
    return css;
}

function svelteCss(source: string): Css {
    const ast = parseSvelte(source, { modern: true });
    const css: Css = { sheets: ast.css ? [ast.css.content.styles] : [], lists: [] };
    function visit(value: unknown): void {
        if (!value || typeof value !== "object") return;
        const node = value as Record<string, unknown>;
        if (node.type === "Attribute" && String(node.name).toLowerCase() === "style") {
            if (Array.isArray(node.value))
                css.lists.push(
                    node.value
                        .map((part: { type: string; data?: string }) =>
                            part.type === "Text" ? part.data : COMPUTED,
                        )
                        .join(""),
                );
            return;
        }
        for (const child of Object.values(node)) {
            if (Array.isArray(child)) for (const item of child) visit(item);
            else visit(child);
        }
    }
    visit(ast.fragment);
    return css;
}

function cssTextStrings(source: string): string[] {
    const lists: string[] = [];
    const ast = parse(source, { sourceType: "module", plugins: ["typescript"] });
    function visit(value: unknown): void {
        if (!value || typeof value !== "object") return;
        const node = value as Record<string, any>;
        if (
            node.type === "AssignmentExpression" &&
            node.left.type === "MemberExpression" &&
            (node.left.property.name ?? node.left.property.value) === "cssText"
        ) {
            if (node.right.type === "StringLiteral") lists.push(node.right.value);
            else if (node.right.type === "TemplateLiteral")
                lists.push(
                    node.right.quasis
                        .map((quasi: { value: { cooked: string } }) => quasi.value.cooked)
                        .join(COMPUTED),
                );
        }
        for (const child of Object.values(node)) {
            if (Array.isArray(child)) for (const item of child) visit(item);
            else visit(child);
        }
    }
    visit(ast);
    return lists;
}

async function cssOf(path: string, source: string): Promise<Css> {
    if (path.endsWith(".css")) return { sheets: [source], lists: [] };
    if (path.endsWith(".html")) return htmlCss(source);
    if (path.endsWith(".svelte")) return svelteCss(source);
    return { sheets: [], lists: cssTextStrings(source) };
}

type Declaration = { property: string; value: string };

function declarations(css: Css): { declarations: Declaration[]; failures: string[] } {
    const found: Declaration[] = [];
    const failures: string[] = [];
    function visit(value: unknown): void {
        if (!value || typeof value !== "object") return;
        const node = value as Record<string, unknown>;
        if (node.type === "Declaration") found.push(node as unknown as Declaration);
        for (const child of Object.values(node)) {
            if (Array.isArray(child)) for (const item of child) visit(item);
            else visit(child);
        }
    }
    for (const sheet of [...css.sheets, ...css.lists.map((list) => `x{${list}}`)]) {
        try {
            visit(parseCss(sheet));
        } catch (error) {
            failures.push(`css: ${(error as Error).message}`);
        }
    }
    return { declarations: found, failures };
}

type Scope = { example?: string; stepped?: boolean };

// CSS Syntax escapes: up to six hex digits and one optional whitespace, or any other character.
function decodeEscapes(value: string): string {
    return value.replace(/\\([0-9a-f]{1,6})[ \t\n\r\f]?|\\([\s\S])/gi, (_, hex, other) => {
        if (hex === undefined) return other;
        const code = Number.parseInt(hex, 16);
        const valid = code > 0 && code <= 0x10ffff && (code < 0xd800 || code > 0xdfff);
        return String.fromCodePoint(valid ? code : 0xfffd);
    });
}

/**
 * Colour literals written as CSS are palette values, with or without an alpha suffix, and opacity is
 * 0 or 1. A stepped test page's background sits under the canvas and is not UI.
 */
function colorViolations(css: Css, { example = "", stepped = false }: Scope = {}): string[] {
    const parsed = declarations(css);
    const failures = parsed.failures;
    for (const { property, value } of parsed.declarations) {
        const name = property.toLowerCase();
        // A partial opacity blends an element into an off-palette colour.
        if (name === "opacity") {
            if (value.trim() !== "0" && value.trim() !== "1")
                failures.push(`opacity: ${value.trim()}`);
            continue;
        }
        if (stepped && (name === "background" || name === "background-color")) continue;
        // Quoted strings and url() arguments are not colours; escapes decode before matching.
        const text = decodeEscapes(
            value.replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, ""),
        ).replace(/\burl\([^)]*\)/gi, "");
        for (const match of text.matchAll(HEX)) {
            const color = hexColor(match[0]);
            if (color && (PALETTE_COLORS.has(color) || EXAMPLE_COLORS[example]?.includes(color)))
                continue;
            failures.push(`color: ${match[0]}`);
        }
        // Palette colours are written in hex; a colour function is refused unless it is computed.
        for (const match of text.matchAll(COLOR_FUNCTIONS))
            if (!match[1]?.includes(COMPUTED_NAME)) failures.push(`color: ${match[0]}`);
        for (const word of text.match(/(?<![\w$#.-])[a-z]+(?![\w(-])/gi) ?? []) {
            const lower = word.toLowerCase();
            if (!KEYWORDS.has(lower) && Bun.color(lower, "hex") !== null)
                failures.push(`color: ${word}`);
        }
    }
    return failures;
}

async function colors(path: string, source: string, scope?: Scope): Promise<string[]> {
    return colorViolations(await cssOf(path, source), scope);
}

// Only a stepped test page's background is exempt; every other source under tests/ is held.
function steppedPage(path: string): boolean {
    return /^examples[\\/][^\\/]+[\\/]tests[\\/]stepped\.html$/.test(path);
}

function violations(source: string): string[] {
    const ui = carbon.ui;
    const failures: string[] = [];
    for (const match of source.matchAll(
        /\b(font(?:-family|-size|-weight)?|border(?:-top-left|-top-right|-bottom-left|-bottom-right)?-radius)\s*:\s*([^;}"'\n]+)/g,
    )) {
        const [, property, raw] = match;
        const value = raw!.trim();
        let valid = false;
        let size: string | undefined;
        if (property === "font-family") valid = value === ui.fontFamily;
        else if (property === "font-size") {
            size = value;
            valid = ui.fontSizes.includes(value);
        } else if (property === "font-weight") valid = value === ui.fontWeight;
        else if (property!.includes("radius")) valid = value === ui.radius || value === ui.circle;
        else {
            const font = value.match(/^(?:(\d+)\s+)?(\d+px)(?:\/[^\s]+)?\s+(.+)$/);
            size = font?.[2];
            valid =
                !!font &&
                (font[1] ?? "400") === ui.fontWeight &&
                ui.fontSizes.includes(font[2]!) &&
                font[3] === ui.fontFamily;
        }
        if (size === ui.fontSizes[1] && !titleDeclaration(source, match.index!)) valid = false;
        if (!valid) failures.push(`${property}: ${value}`);
    }
    // UI keeps CSS pixels (CONTRIBUTING.md, Examples).
    for (const match of source.matchAll(/\bzoom\s*:\s*([^;}"'\n]+)/g))
        failures.push(`zoom: ${match[1]!.trim()}`);
    return failures;
}

test("the UI scale reserves 24px for titles and rejects off-token CSS, inline declarations and zoom", () => {
    const ui = PALETTES["aperture-carbon"].ui;
    expect(ui.fontSizes).toEqual(["14px", "24px"]);
    expect(
        violations(
            `.panel { font: ${ui.fontWeight} ${ui.fontSizes[0]} ${ui.fontFamily}; border-radius: ${ui.radius}; }`,
        ),
    ).toEqual([]);
    expect(
        violations(
            '<span style="font-family: sans-serif; font-size: 18px; font-weight: 700; border-radius: 6px">',
        ),
    ).toHaveLength(4);
    expect(violations(`.reticle { border-radius: ${ui.circle}; }`)).toEqual([]);
    expect(violations(".panel { font: bold 48px serif; }")).toHaveLength(1);
    expect(violations("h1 { font-size: 24px; }")).toEqual([]);
    expect(violations('<h1 style="font-size: 24px">Name</h1>')).toEqual([]);
    expect(violations(".notice { font-size: 24px; }")).toHaveLength(1);
    expect(violations("h1, .notice { font-size: 24px; }")).toHaveLength(1);
    expect(violations('<p style="font-size: 24px">Status</p>')).toHaveLength(1);
    expect(violations("small { font-size: 12px; }")).toHaveLength(1);
    expect(violations(".card { zoom: max(1, calc(100vmin / 1080px)); }")).toEqual([
        "zoom: max(1, calc(100vmin / 1080px))",
    ]);
});

test("UI colours written as CSS are palette values or their alpha variants and opacity is 0 or 1; Svelte style: directives and DOM .style properties are not read", async () => {
    const css = (source: string, scope?: Scope) => colors("a.css", source, scope);
    expect(
        await css(
            `.panel { background: ${carbon.ui.panel}cc; color: ${carbon.ui.ink}; border: 1px solid ${carbon.ui.border}; }`,
        ),
    ).toEqual([]);
    expect(await css(".bar { background: #ddaa3f; } body { background: #191816; }")).toEqual([]);
    expect(await css(".help { color: #a69d8f; }")).toEqual(["color: #a69d8f"]);
    expect(await css(".panel { background: rgba(14, 17, 20, 0.72); }")).toEqual([
        "color: rgba(14, 17, 20, 0.72)",
    ]);
    expect(await css("body { background: #89b6e9; }", { example: "voxel" })).toEqual([]);
    expect(await css("body { background: #89b6e9; }", { example: "minimal" })).toEqual([
        "color: #89b6e9",
    ]);
    expect(await colors("a.ts", `el.style.cssText = \`color: rgb(\${channels})\`;`)).toEqual([]);
    expect(await css(".fade { opacity: 0; } .fade.shown { opacity: 1; }")).toEqual([]);
    expect(await css("button:disabled { opacity: 0.5; }")).toEqual(["opacity: 0.5"]);
    for (const color of [
        "red",
        "Tan",
        "lab(50% 80 60)",
        "RGB(255 0 0)",
        "hwb(0 0% 0%)",
        "oklab(0.5 0.1 0.1)",
        "lch(50% 30 40)",
        "color(srgb 1 0 0)",
        "color-mix(in srgb, #eee9df, #282622)",
        "#fff",
        "#ffff",
        "#12345",
    ])
        expect(await css(`.help { color: ${color}; }`)).toEqual([`color: ${color}`]);
    for (const [source, color] of [
        [".a { background-image: linear-gradient(red, #282622); }", "red"],
        [".a { mask: linear-gradient(black, transparent); }", "black"],
        [".a { scrollbar-color: gray transparent; }", "gray"],
        [".a { -webkit-text-fill-color: white; }", "white"],
    ])
        expect(await css(source!)).toEqual([`color: ${color}`]);
    expect(await css(".a { background-image: linear-gradient(red, blue); }")).toEqual([
        "color: red",
        "color: blue",
    ]);
    expect(
        await css(`.a { background-image: linear-gradient(${carbon.ui.panel}, transparent); }`),
    ).toEqual([]);
    expect(await css(".a { COLOR: red; Border: 1px solid TAN; }")).toEqual([
        "color: red",
        "color: TAN",
    ]);
    expect(await css(".a { COLOR: CurrentColor; BACKGROUND: TRANSPARENT; }")).toEqual([]);
    expect(await css(".a { OPACITY: 0.5; }")).toEqual(["opacity: 0.5"]);
    expect(await css(".a { color: #eee; background: #eee9; border-color: #EEE9DF; }")).toEqual([
        "color: #eee",
        "color: #eee9",
    ]);
    expect(await css(".a { color: currentColor; background: transparent; fill: none; }")).toEqual(
        [],
    );
    expect(await css(`.a { color: inherit; font: 400 14px ${carbon.ui.fontFamily}; }`)).toEqual([]);
    // Escapes decode before matching, and a string ends only at an unescaped quote.
    expect(await css(".a { color: r\\65 d; }")).toEqual(["color: red"]);
    expect(await css(".a { color: \\72\\65\\64; }")).toEqual(["color: red"]);
    expect(await css('.a { content: "\\"red"; color: #eee9df; }')).toEqual([]);
    expect(await css(".a { content: 'it\\'s red'; color: #eee9df; }")).toEqual([]);
    // A content string is not markup, and an id selector is not a colour.
    expect(
        await css('.a::before { content: "<b>"; color: tan; } #fab { color: #eee9df; }'),
    ).toEqual(["color: tan"]);
    expect(await colors("a.html", '<style>.a { content: "<b>"; color: tan; }</style>')).toEqual([
        "color: tan",
    ]);

    // Pages: <style> text and style attributes, quoted or not; scripts are never read.
    expect(
        await colors(
            "a.html",
            '<style>.a { color: tan; }</style><script>const s = { canvas: 1, c: "#a69d8f", f: "rgb(1, 2, 3)" };</script>',
        ),
    ).toEqual(["color: tan"]);
    expect(await colors("a.html", "<p style=color:red>x</p><p style='color: #fff'>y</p>")).toEqual([
        "color: red",
        "color: #fff",
    ]);
    expect(
        await colors(
            "a.svelte",
            '<script>const c = "#a69d8f";</script><p style=color:red>x</p>{#if a}<b style="color: tan; width: {w}px"></b>{/if}<i style:color="red"></i><style>.a { color: #fff; }</style>',
        ),
    ).toEqual(["color: #fff", "color: red", "color: tan"]);
    expect(await colors("a.ts", 'const c = "#a69d8f"; const css = ".a { color: red; }";')).toEqual(
        [],
    );
    expect(
        await colors("a.ts", 'el.style.cssText = "position:absolute;inset:0;color:red";'),
    ).toEqual(["color: red"]);
    // A message string is not a style string (examples/physics/tests/gold.ts).
    expect(
        await colors("a.ts", `throw new Error(\`step \${step}: invalid sample gold line\`);`),
    ).toEqual([]);

    const stepped = '<body style="background:#1c1917"><p style="color: #a69d8f">';
    expect(await colors("a.html", stepped, { stepped: true })).toEqual(["color: #a69d8f"]);
    expect(steppedPage(join("examples", "voxel", "tests", "stepped.html"))).toBe(true);
    expect(steppedPage(join("examples", "physics", "tests", "capture-page.ts"))).toBe(false);
    expect(steppedPage(join("examples", "hud", "src", "tests", "stepped.html"))).toBe(false);
    expect(
        await css("body { background: #1c1917; }", {
            stepped: steppedPage(join("examples", "physics", "tests", "overlay.html")),
        }),
    ).toEqual(["color: #1c1917"]);
});

test("world-space Text font fields are not CSS, while styles in script strings are checked", () => {
    const source =
        "const state: { font: number } = { font: 1 }; const text = { font: state.font }; const css = `.panel { font: bold 48px serif; }`;";
    expect(styleSources("scene.ts", source).flatMap((style) => violations(style))).toEqual([
        "font: bold 48px serif",
    ]);
});

for (const example of await readdir("examples")) {
    test(`${example} CSS and inline styles use the UI type, shape and colour tokens`, async () => {
        const failures: string[] = [];
        for (const path of await sources(join("examples", example))) {
            const source = await readFile(path, "utf8");
            failures.push(
                ...[
                    ...styleSources(path, source).flatMap((style) => violations(style)),
                    ...(await colors(path, source, { example, stepped: steppedPage(path) })),
                ].map((failure) => `${path}: ${failure}`),
            );
        }
        expect(failures).toEqual([]);
    });
}

test("embedded-scene host page uses the UI type, shape and colour tokens", async () => {
    const path = "examples/embedded-scene/src/style.css";
    const source = await readFile(path, "utf8");
    expect([...violations(source), ...(await colors(path, source))]).toEqual([]);
});
