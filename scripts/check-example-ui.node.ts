import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdir, mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative, resolve } from "node:path";
import { parse } from "@babel/parser";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { parseCss, parse as parseSvelte } from "svelte/compiler";
import { colorHex, EXAMPLE_CONTROLS, PALETTES } from "../diagnostics/default-look/palette";
import { removeTestDirectory } from "./test-directory";

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

function spacingViolations(css: Css): string[] {
    const parsed = declarations(css);
    const failures = parsed.failures;
    for (const { property, value } of parsed.declarations) {
        if (
            !/^(?:(?:margin|padding|inset)(?:-.+)?|(?:row-|column-)?gap|top|right|bottom|left|--.*(?:gap|pad|top|right|bottom|left).*)$/i.test(
                property,
            )
        )
            continue;
        for (const match of value.matchAll(/(-?\d*\.?\d+)(px|rem|em|vw|vh)\b/gi)) {
            const placement = /^(top|right|bottom|left|inset|--(?:top|right|bottom|left))$/.test(
                property,
            );
            const size = Math.abs(Number(match[1]));
            if (
                match[2] !== "px" ||
                (placement
                    ? size % carbon.ui.spacing.base !== 0
                    : ![0, carbon.ui.spacing.base, carbon.ui.spacing.group].includes(size))
            )
                failures.push(`${property}: ${value}`);
        }
    }
    return failures;
}

test("UI spacing uses only the compact set in sheets and inline declarations", async () => {
    expect(carbon.ui.spacing.base).toBe(4);
    expect(
        spacingViolations(
            await cssOf(
                "a.css",
                ".a { gap: 8px; padding: 4px 8px; margin: -4px auto; left: 50%; top: 24px; }",
            ),
        ),
    ).toEqual([]);
    for (const [path, source] of [
        ["a.css", ".a { gap: 12px; }"],
        ["a.html", '<b style="padding: 10px 8px">'],
        ["a.svelte", '<b style="margin-inline: 1rem"></b>'],
        ["a.ts", 'el.style.cssText = "row-gap: 5px";'],
    ])
        expect(spacingViolations(await cssOf(path!, source!))).toHaveLength(1);
});

function rowViolations(css: Css): string[] {
    const failures: string[] = [];
    for (const sheet of css.sheets) {
        function visit(value: unknown, row = false): void {
            if (!value || typeof value !== "object") return;
            const node = value as Record<string, any>;
            if (node.type === "Rule") {
                const selector = sheet.slice(node.prelude.start, node.prelude.end);
                row =
                    /(?:\bbutton\b|\.row\b|\.toggle\b|input\[type=["']?range)/.test(selector) &&
                    !/::|checkbox|\bnav\b/.test(selector);
            }
            if (
                row &&
                node.type === "Declaration" &&
                /^(?:min-)?height$/.test(node.property) &&
                /^\d+px$/.test(node.value.trim()) &&
                node.value.trim() !== `${carbon.ui.controls.row}px`
            )
                failures.push(`${node.property}: ${node.value}`);
            for (const child of Object.values(node)) {
                if (Array.isArray(child)) for (const item of child) visit(item, row);
                else visit(child, row);
            }
        }
        visit(parseCss(sheet));
    }
    return failures;
}

test("controls and list rows have one height", async () => {
    expect(
        rowViolations(await cssOf("a.css", ".row { height: 24px; } button { min-height: 24px; }")),
    ).toEqual([]);
    for (const css of ["button { height: 28px; }", ".row { min-height: 32px; }"])
        expect(rowViolations(await cssOf("a.css", css))).toHaveLength(1);
});

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

function sceneBackgrounds(css: Css, scope: Scope): Css {
    function mask(sheet: string): string {
        const edits: { start: number; end: number; text: string }[] = [];
        function visit(node: any): void {
            if (!node || typeof node !== "object") return;
            if (
                node.type === "Declaration" &&
                /^(background|background-color)$/i.test(node.property) &&
                (scope.stepped ||
                    (scope.example === "voxel" && node.value.toLowerCase() === "#89b6e9"))
            )
                edits.push({
                    start: node.start,
                    end: node.end,
                    text: `${node.property}: ${COMPUTED}`,
                });
            for (const child of Object.values(node)) {
                if (Array.isArray(child)) for (const item of child) visit(item);
                else if (child && typeof child === "object") visit(child);
            }
        }
        visit(parseCss(sheet));
        for (const edit of edits.sort((a, b) => b.start - a.start))
            sheet = sheet.slice(0, edit.start) + edit.text + sheet.slice(edit.end);
        return sheet;
    }
    return {
        sheets: css.sheets.map(mask),
        lists: css.lists.map((list) => mask(`x{${list}}`).slice(2, -1)),
    };
}

/** Local imports are read even when they leave src/; remote CSS cannot carry a standalone UI. */
async function exampleCss(root: string): Promise<Css> {
    const result: Css = { sheets: [], lists: [] };
    const seen = new Set<string>();
    async function load(path: string): Promise<void> {
        path = resolve(path);
        if (seen.has(path)) return;
        seen.add(path);
        const source = await readFile(path, "utf8");
        const css = sceneBackgrounds(await cssOf(path, source), {
            example: basename(root),
            stepped: steppedPage(relative(process.cwd(), path)),
        });
        result.sheets.push(
            ...css.sheets.map((sheet) =>
                sheet.replace(
                    /url\(\s*['"]?([^)'"\s]+\.woff2)['"]?\s*\)/g,
                    (whole, url: string) => {
                        if (/^(?:\/|[a-z]+:)/i.test(url)) return whole;
                        const asset = relative(
                            resolve(root, "public"),
                            resolve(dirname(path), url),
                        ).replaceAll("\\", "/");
                        return `url('/${asset}')`;
                    },
                ),
            ),
        );
        if (path.endsWith(join("src", "player.ts"))) {
            result.sheets.push(...css.lists.map((list) => `gameplay{${list}}`));
        } else result.lists.push(...css.lists);
        if (/\.[cm]?[jt]s$/.test(path)) {
            for (const text of styleSources(path, source)) {
                if (
                    /\{[^{}]*\b(?:font(?:-family|-size|-weight)?|border-radius|zoom)\s*:/.test(text)
                )
                    result.sheets.push(text);
            }
        }
        const imports = [
            ...css.sheets.join("\n").matchAll(/@import\s+(?:url\(\s*)?["']?([^\s"');]+)["']?/g),
        ];
        if (/\.[cm]?[jt]s$|\.svelte$/.test(path))
            imports.push(
                ...source.matchAll(/\bimport\s+(?:[^;]*?\sfrom\s*)?["']([^"']+\.css)["']/g),
            );
        for (const match of imports) {
            const name = match[1]!;
            if (/^(?:https?:|\/\/|data:)/.test(name)) throw new Error(`remote stylesheet: ${name}`);
            await load(
                name.startsWith("/") ? join(root, "public", name) : resolve(dirname(path), name),
            );
        }
    }
    for (const path of await sources(root)) await load(path);
    return result;
}

type StyleRule = { selector: string; media: string; declarations: Declaration[] };

function styleRules(css: Css): StyleRule[] {
    const rules: StyleRule[] = [];
    for (const sheet of [...css.sheets, ...css.lists.map((list) => `inline{${list}}`)]) {
        function visit(value: any, media = "", current?: StyleRule): void {
            if (!value || typeof value !== "object") return;
            if (value.type === "Atrule" && value.name === "media") media += value.prelude;
            if (value.type === "Rule") {
                current = {
                    selector: decodeEscapes(sheet.slice(value.prelude.start, value.prelude.end)),
                    media,
                    declarations: [],
                };
                rules.push(current);
            }
            if (value.type === "Declaration") current?.declarations.push(value);
            for (const child of Object.values(value)) {
                if (Array.isArray(child)) for (const item of child) visit(item, media, current);
                else if (child && typeof child === "object") visit(child, media, current);
            }
        }
        visit(parseCss(sheet));
    }
    return rules;
}

function selectors(text: string): string[] {
    const result: string[] = [];
    let depth = 0;
    let start = 0;
    let quote = "";
    for (let i = 0; i < text.length; i++) {
        const char = text[i]!;
        if (quote) {
            if (char === quote && text[i - 1] !== "\\") quote = "";
            continue;
        }
        if (char === "'" || char === '"') quote = char;
        else if (char === "(" || char === "[") depth++;
        else if (char === ")" || char === "]") depth--;
        else if (char === "," && depth === 0) {
            result.push(text.slice(start, i).trim());
            start = i + 1;
        }
    }
    result.push(text.slice(start).trim());
    return result;
}

function controlViolations(css: Css, light = false): string[] {
    const failures: string[] = [];
    let rules: StyleRule[];
    try {
        rules = styleRules(css).flatMap((rule) =>
            selectors(rule.selector).map((selector) => ({ ...rule, selector })),
        );
    } catch (error) {
        return [`css: ${error}`];
    }
    const palette = EXAMPLE_CONTROLS[light ? "light" : "dark"];
    const tokens: Record<string, string> = {
        ...EXAMPLE_CONTROLS.tokens,
        ...palette,
        "header-surface": palette.bg2,
    };
    const definitions = new Map<string, string[]>();
    for (const rule of rules)
        for (const { property, value } of rule.declarations) {
            if (!property.startsWith("--")) continue;
            const values = definitions.get(property) ?? [];
            values.push(value);
            definitions.set(property, values);
        }
    function expand(value: string, stack: string[] = []): string {
        return decodeEscapes(value).replace(
            /var\(\s*(--[\w-]+)\s*(?:,\s*([^()]*))?\)/g,
            (_, name, fallback) => {
                if (name === COMPUTED_NAME) return COMPUTED;
                if (stack.includes(name)) {
                    failures.push(`cyclic variable: ${name}`);
                    return "invalid";
                }
                const values = definitions.get(name);
                if (!values) {
                    if (fallback !== undefined) return expand(fallback, stack);
                    failures.push(`unknown variable: ${name}`);
                    return "invalid";
                }
                const expanded = values.map((v) => expand(v, [...stack, name]));
                if (new Set(expanded).size !== 1) failures.push(`ambiguous variable: ${name}`);
                return expanded[0]!;
            },
        );
    }
    const normalize = (s: string) => s.trim().replace(/\s+/g, " ").replace(/"/g, "'");
    const has = (value: string, expected: string) => normalize(value) === normalize(expected);
    const rowSelector = (s: string) =>
        /\bbutton\b|\bselect\b|\.row\b|\.toggle\b|\.section-title\b|\.panel-title\b|\.option\b|input\[type=['"]?range/.test(
            s,
        ) && !/::/.test(s);
    const heights = new Map<string, Set<string>>();
    const types = new Map<string, Declaration[]>();
    for (const rule of rules) {
        const { selector: s, media } = rule;
        const gameplay = /^(?:gameplay)$|\.reticle\b|\.reticle-dot\b|\[data-touch-stick\]/.test(s);
        const touch = /pointer:\s*coarse|max-width:\s*480px/.test(media);
        for (const selector of selectors(s)) {
            if (
                rowSelector(selector) &&
                !/:hover|:active|:focus|:disabled|:is|aria-/.test(selector) &&
                !heights.has(selector)
            )
                heights.set(selector, new Set());
            if (/\.panel-title\b|\.section-title\b|\bh[2-6]\b|\boutput\b|\.value\b/.test(selector))
                types.set(selector, [...(types.get(selector) ?? []), ...rule.declarations]);
        }
        if (/\d/.test(media.replace(/max-width:\s*480px/g, ""))) failures.push(`media: ${media}`);
        for (const { property: rawProperty, value: raw } of rule.declarations) {
            const p = decodeEscapes(rawProperty).toLowerCase();
            const v = expand(raw).trim();
            const fail = (role: string) => failures.push(`${role}: ${s} { ${p}: ${v} }`);
            if (p.startsWith("--")) {
                const expected = tokens[p.slice(2)];
                if (expected && !has(v, expected)) fail("token");
                // Unregistered aliases are still checked as literals, not an escape hatch.
            }
            const text = v
                .replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, "")
                .replace(/url\([^)]*\)/g, "");
            for (const match of text.matchAll(HEX))
                if (!Object.values(palette).includes(match[0].toLowerCase() as typeof palette.bg))
                    fail("chrome color");
            if (COLOR_FUNCTIONS.test(text) && !text.includes(COMPUTED_NAME)) fail("color syntax");
            COLOR_FUNCTIONS.lastIndex = 0;
            for (const word of text.match(/(?<![\w$#.-])[a-z]+(?![\w(-])/gi) ?? [])
                if (!KEYWORDS.has(word.toLowerCase()) && Bun.color(word, "hex") !== null)
                    fail("color syntax");
            if (p.startsWith("--")) {
                const values = new Set([...Object.values(tokens), "0", "1fr"]);
                for (const number of text.matchAll(
                    /(?<![\w#.-])\d*\.?\d+(?:px|rem|em|dvh|vh|vw|%|fr)?(?![\w.])/g,
                ))
                    if (!values.has(number[0])) fail("geometry token");
                continue;
            }
            if (p === "zoom" || /shadow/.test(p)) fail("surface");
            if (p === "opacity" && (!/^[01]$/.test(v) || /:disabled|aria-disabled/.test(s)))
                fail("opacity");
            if (/scale\(/.test(v)) fail("scale");
            const heading = /\.section-title\b|\bh[2-6]\b/.test(s);
            const title = /\.panel-title\b/.test(s);
            const output = /\boutput\b|\.value\b/.test(s);
            const family = title || heading || output ? tokens.mono! : tokens.sans!;
            const size = title ? tokens.title! : heading || output ? tokens.small! : tokens.body!;
            const weight = title ? tokens.strong! : tokens.regular!;
            if (
                p === "font" &&
                !(v === "inherit" && !title && !heading && !output) &&
                !has(v, `${weight} ${size}/${tokens.leading} ${family}`) &&
                !(/(?:^|[ >])h1$/.test(s) && has(v, `400 24px/20px ${tokens.mono}`))
            )
                fail("type role");
            if (p === "font-size" && v !== size && !(s === "h1" && v === "24px")) fail("type role");
            if (p === "font-family" && s !== "@font-face" && !has(v, family)) fail("type role");
            if (p === "font-weight" && v !== weight) fail("type role");
            if (p === "line-height" && v !== "20px") fail("type role");
            if (p === "letter-spacing" && (!heading || v !== ".08em")) fail("heading tracking");
            if (p === "text-transform" && v === "uppercase" && !heading) fail("heading case");
            if (!gameplay && /^(?:min-)?height$/.test(p) && rowSelector(s)) {
                if (v !== (touch ? "44px" : "32px")) fail("row");
                for (const selector of selectors(s)) {
                    const modes = heights.get(selector) ?? new Set();
                    if (!touch) modes.add("fine");
                    if (/pointer:\s*coarse/.test(media)) modes.add("coarse");
                    if (/max-width:\s*480px/.test(media)) modes.add("phone");
                    heights.set(selector, modes);
                }
            }
            if (!gameplay && /^(?:padding|margin|gap|row-gap|column-gap)(?:-|$)/.test(p)) {
                let expected = "0px";
                if (p.includes("gap"))
                    expected = /\.fields\b|\.categories\s*$/.test(s)
                        ? "16px"
                        : /\.section\b|\bnav\s*$|\.categories section/.test(s)
                          ? "4px"
                          : "8px";
                else if (p.startsWith("padding"))
                    expected = title
                        ? "12px 16px"
                        : heading
                          ? "0px 8px"
                          : /\.fields\b|\.scenes\b/.test(s)
                            ? "16px"
                            : /dependents/.test(s)
                              ? "8px"
                              : /button|select/.test(s)
                                ? "4px 8px"
                                : "0px";
                if (
                    !has(v, expected) &&
                    !/^0(?:px)?(?: 0(?:px)?)*$/.test(v) &&
                    !(
                        /slider-thumb/.test(s) &&
                        p === "margin-top" &&
                        has(v, "calc((4px - 16px) / 2)")
                    ) &&
                    !(heading && has(v, "0px 0px 8px"))
                )
                    fail("spacing role");
            }
            if (!gameplay && /^(?:top|right|bottom|left|inset)$/.test(p)) {
                const lengths = [...v.matchAll(/(\d+)px/g)].map((m) => m[1]);
                if (lengths.some((n) => n !== (touch ? "16" : "24")) && !/\.scenes\b/.test(s))
                    fail("inset role");
            }
            // State roles are independent of token membership.
            const disabled = /:disabled|aria-disabled/.test(s) && !/:not\(:disabled\)/.test(s);
            const hover = /:hover/.test(s);
            const pressed = /:active/.test(s);
            const selected = /aria-(?:pressed|current)=['"]?true|\.selected\b/.test(s);
            const check = /checkbox/.test(s);
            const thumb = /slider-thumb|range-thumb/.test(s);
            const track = /slider-runnable-track|range-track/.test(s);
            const focus = /:focus-visible/.test(s);
            const background = /^(background|background-color)$/.test(p);
            const border = /^(border(?:-color)?|border-(?:left|right|top|bottom)(?:-color)?)$/.test(
                p,
            );
            let role: keyof typeof palette | undefined;
            if (p === "color") role = disabled ? "muted" : heading ? "muted" : "ink";
            if (background && v !== "transparent")
                role = thumb
                    ? disabled
                        ? "dim"
                        : pressed
                          ? "gold"
                          : hover
                            ? "ink"
                            : "muted"
                    : track
                      ? "line"
                      : pressed
                        ? "line"
                        : check
                          ? /:checked/.test(s)
                              ? "dim"
                              : "bg"
                          : /controls|scenes|section-title/.test(s)
                            ? "bg2"
                            : "bg";
            if (border && v !== "0px" && v !== "0")
                role = selected
                    ? "gold"
                    : hover
                      ? check
                          ? "ink"
                          : "muted"
                      : check
                        ? "muted"
                        : "line";
            if (p === "outline" && focus) {
                if (!has(v, `2px solid ${palette.gold}`)) fail("focus role");
            }
            if (p === "outline-offset" && v !== "2px") fail("focus role");
            if (role && !v.includes(palette[role]) && !v.includes(COMPUTED_NAME))
                fail("state role");
            if (
                background &&
                v === "transparent" &&
                /button|select|checkbox|controls|section-title/.test(s)
            )
                fail("state role");
            if (
                (hover || pressed) &&
                selectors(s).some(
                    (selector) =>
                        /button|select|checkbox|thumb/.test(selector) &&
                        /:hover|:active/.test(selector) &&
                        !selector.includes(":not(:disabled)"),
                )
            )
                fail("disabled priority");
            // Gameplay stick, touch buttons and reticles retain their geometry outside the panel kit.
            if (gameplay) continue;
            if (/radius/.test(p) && !["0", "0px", "2px"].includes(v)) fail("corner role");
            if ((check || thumb || track) && /^(width|height)$/.test(p)) {
                const expected = check ? "16px" : thumb ? (p === "width" ? "8px" : "16px") : "4px";
                if (v !== expected) fail("affordance geometry");
            }
            // Every dimension has a named role; structural algebra is checked separately.
            const allowed = new Set([
                "0",
                "1",
                "2",
                "0px",
                "1fr",
                "100%",
                "50%",
                "100dvh",
                "60dvh",
            ]);
            let sizes: string[] = [];
            if (/radius/.test(p) || /^outline/.test(p)) sizes = ["2px"];
            else if (/border/.test(p)) sizes = ["1px"];
            else if (/width/.test(p))
                sizes = check
                    ? ["16px"]
                    : thumb
                      ? ["8px"]
                      : /button:not\(\.name\)/.test(s)
                        ? ["44px"]
                        : /\.name\b/.test(s)
                          ? ["360px", "64px", "16px"]
                          : /button/.test(s)
                            ? ["44px"]
                            : ["360px", "16px", "32px"];
            else if (/height/.test(p))
                sizes =
                    check || thumb
                        ? ["16px"]
                        : track
                          ? ["4px"]
                          : rowSelector(s)
                            ? [touch ? "44px" : "32px"]
                            : ["24px", "64px"];
            else if (p === "grid-template-columns")
                sizes = /\.slider\b/.test(s)
                    ? ["48px"]
                    : /dependents/.test(s)
                      ? ["144px", "8px", "1px"]
                      : ["144px"];
            else if (
                /font|line-height|letter-spacing|padding|margin|gap|top|right|bottom|left/.test(p)
            )
                continue;
            for (const m of text.matchAll(
                /(?<![\w#.-])\d*\.?\d+(?:px|rem|em|dvh|vh|vw|%|fr)?(?![\w.])/g,
            )) {
                if (!allowed.has(m[0]) && !sizes.includes(m[0])) fail("geometry role");
            }
        }
    }
    for (const rule of rules.filter((rule) => rule.selector.includes(":focus-visible"))) {
        if (
            !rule.declarations.some(
                (d) => d.property === "outline" && expand(d.value) === `2px solid ${palette.gold}`,
            ) ||
            !rule.declarations.some(
                (d) => d.property === "outline-offset" && expand(d.value) === "2px",
            )
        )
            failures.push(`focus outline/offset missing: ${rule.selector}`);
    }
    if (
        rules.some((rule) => /nav\s+\.name\b/.test(rule.selector)) &&
        !rules.some(
            (rule) =>
                /nav button:not\(\.name\)/.test(rule.selector) &&
                /max-width:\s*480px/.test(rule.media) &&
                rule.declarations.some(
                    (d) => d.property === "min-width" && expand(d.value) === "44px",
                ),
        )
    )
        failures.push("phone icon button width missing");
    for (const kind of ["button", "select", "checkbox", "range"]) {
        const owned = rules.filter((rule) => rule.selector.includes(kind));
        if (!owned.some((rule) => !/:|aria-/.test(rule.selector))) continue;
        for (const state of [":hover", ":active"]) {
            if (!owned.some((rule) => rule.selector.includes(state)))
                failures.push(`state ${state} missing: ${kind}`);
        }
        if (
            !rules.some(
                (rule) =>
                    rule.selector.includes(":focus-visible") &&
                    rule.declarations.some((d) => d.property === "outline"),
            )
        )
            failures.push(`focus missing: ${kind}`);
        if (
            !rules.some(
                (rule) =>
                    /:disabled/.test(rule.selector) &&
                    !/:not\(:disabled\)/.test(rule.selector) &&
                    rule.declarations.some((d) => d.property === "color"),
            )
        )
            failures.push(`disabled text missing: ${kind}`);
    }
    for (const [selector, declarations] of types) {
        if (/:hover|:active|:focus|:disabled/.test(selector)) continue;
        if (!declarations.some((d) => d.property === "font" || d.property === "font-size"))
            failures.push(`type role missing: ${selector}`);
        if (/\.section-title\b|\bh[2-6]\b/.test(selector)) {
            if (
                !declarations.some(
                    (d) => d.property === "letter-spacing" && expand(d.value) === ".08em",
                )
            )
                failures.push(`heading tracking missing: ${selector}`);
            if (
                !declarations.some(
                    (d) => d.property === "text-transform" && d.value === "uppercase",
                )
            )
                failures.push(`heading case missing: ${selector}`);
        }
    }
    for (const [selector, modes] of heights) {
        for (const mode of ["fine", "coarse", "phone"]) {
            const target = selector.replace(/:[\w-]+(?:\([^)]*\))?/g, "");
            const inherited = [...heights].some(
                ([base, coverage]) =>
                    coverage.has(mode) && target.endsWith(` ${base.split(/\s+/).at(-1)}`),
            );
            if (!modes.has(mode) && !inherited) failures.push(`row ${mode} missing: ${selector}`);
        }
    }
    return failures;
}

async function fontViolations(root: string, css: Css): Promise<string[]> {
    const failures: string[] = [];
    const sheet = css.sheets.join("\n");
    const faces = [...sheet.matchAll(/@font-face\s*\{([^}]+)\}/g)].map((m) => m[1]!);
    const families = new Set(["IBM Plex Sans", "JetBrains Mono"]);
    for (const match of sheet.matchAll(/font-family:\s*['"]([^'"]+)['"]/g)) families.add(match[1]!);
    for (const family of families) {
        if (!sheet.includes(family) && !css.lists.some((s) => s.includes(family))) continue;
        for (const weight of family === "JetBrains Mono" ? ["400", "700"] : ["400"]) {
            const face = faces.find(
                (f) => f.includes(family) && new RegExp(`font-weight:\\s*${weight}\\b`).test(f),
            );
            const url = face?.match(/url\(['"]?([^)'"\s]+)['"]?\)/)?.[1];
            if (!url?.endsWith(".woff2") || !url.startsWith("/")) {
                failures.push(`font face: ${family} ${weight}`);
                continue;
            }
            const path = resolve(root, "public", `.${url}`);
            try {
                if (relative(resolve(root, "public"), path).startsWith(".."))
                    throw new Error("outside public");
                const data = await readFile(path);
                if (data.toString("ascii", 0, 4) !== "wOF2") failures.push(`woff2: ${url}`);
                const license = await readFile(join(dirname(path), "OFL.txt"), "utf8");
                if (!/SIL OPEN FONT LICENSE/i.test(license)) failures.push(`OFL: ${url}`);
            } catch {
                failures.push(`shipped font/OFL: ${url}`);
            }
        }
    }
    return failures;
}

async function exampleControls(root: string, light = false): Promise<string[]> {
    try {
        const css = await exampleCss(root);
        return [...controlViolations(css, light), ...(await fontViolations(root, css))];
    } catch (error) {
        return [`source: ${error}`];
    }
}

const controlFixture = await readFile("scripts/fixtures/example-controls/controls.css", "utf8");

test("example-controls candidate holds colour, type, spacing, rows, states and geometry roles", () => {
    expect(controlViolations({ sheets: [controlFixture], lists: [] })).toEqual([]);
});

const controlMutations: [string, string, string][] = [
    ["1: off-site chrome", "--bg: #141210", "--bg: #191816"],
    ["1: scene colour is not chrome", "--ink: #f0e6d6", "--ink: #ddaa3f"],
    ["1: alpha surface", "--bg: #141210", "--bg: #141210cc"],
    ["1: escaped named colour", "--ink: #f0e6d6", "--ink: r\\65 d"],
    ["1: colour function", "--bg: #141210", "--bg: rgb(20 18 16)"],
    ["1: open headings", "--header-surface: var(--bg2)", "--header-surface: var(--bg)"],
    [
        "2: body size",
        "var(--regular) var(--body)/var(--leading) var(--sans)",
        "var(--regular) var(--small)/var(--leading) var(--sans)",
    ],
    ["2: title weight", "var(--strong) var(--title)", "var(--regular) var(--title)"],
    [
        "2: title family",
        "var(--strong) var(--title)/var(--leading) var(--mono)",
        "var(--strong) var(--title)/var(--leading) var(--sans)",
    ],
    ["2: heading tracking", "letter-spacing: var(--tracking)", "letter-spacing: .1em"],
    [
        "2: output family",
        "font-variant-numeric: tabular-nums",
        "font-family: var(--sans); font-variant-numeric: tabular-nums",
    ],
    [
        "2: title-only 24px",
        "font-variant-numeric: tabular-nums",
        "font-size: 24px; font-variant-numeric: tabular-nums",
    ],
    [
        "3: panel padding",
        "gap: var(--s4); padding: var(--s4)",
        "gap: var(--s4); padding: var(--s2)",
    ],
    [
        "3: independent sections",
        "gap: var(--s4); padding: var(--s4)",
        "gap: var(--s2); padding: var(--s4)",
    ],
    [
        "3: related rows",
        ".ec .section { display: grid; gap: var(--s1)",
        ".ec .section { display: grid; gap: var(--s4)",
    ],
    [
        "3: label-field gap",
        "align-items: center; gap: var(--s2)",
        "align-items: center; gap: var(--s1)",
    ],
    ["3: dependency indent", "padding-left: var(--s2)", "padding-left: var(--s4)"],
    ["3: title padding", "padding: var(--s3) var(--s4)", "padding: var(--s2) var(--s4)"],
    ["3: heading padding", "padding: var(--zero) var(--s2)", "padding: var(--zero) var(--s4)"],
    ["3: desktop inset", "top: var(--s5)", "top: var(--s4)"],
    ["3: phone inset", "top: calc(var(--s4)", "top: calc(var(--s5)"],
    ["4: fine row", "min-height: var(--row)", "min-height: var(--touch)"],
    ["4: coarse row", "min-height: var(--touch)", "min-height: var(--row)"],
    ["4: phone boundary", "max-width: 480px", "max-width: 479px"],
    ["4: missing coarse", "pointer: coarse", "pointer: fine"],
    ["4: no panel zoom", "box-sizing: border-box", "zoom: 1; box-sizing: border-box"],
    [
        "4: no panel scaling",
        "box-sizing: border-box",
        "transform: scale(1); box-sizing: border-box",
    ],
    [
        "8: hover border",
        "border-color: var(--muted); }\n.ec button:active",
        "border-color: var(--gold); }\n.ec button:active",
    ],
    [
        "8: pressed surface",
        "background: var(--line); }\n.ec button[aria-current",
        "background: var(--gold); }\n.ec button[aria-current",
    ],
    ["8: selected border", "{ border-color: var(--gold); }", "{ border-color: var(--muted); }"],
    [
        "8: checked fill",
        "background: var(--dim); border-color",
        "background: var(--gold); border-color",
    ],
    ["8: focus width", "outline: var(--radius)", "outline: var(--hair)"],
    ["8: focus offset", "outline-offset: var(--radius)", "outline-offset: var(--hair)"],
    [
        "8: disabled text",
        ".ec :disabled { color: var(--muted)",
        ".ec :disabled { color: var(--ink)",
    ],
    ["8: disabled opacity even 1", ".ec :disabled {", ".ec :disabled { opacity: 1;"],
    ["8: fractional opacity", ".ec { color:", ".ec { opacity: .5; color:"],
    ["8: disabled priority", "button:hover:not(:disabled)", "button:hover"],
    ["9: panel width", "width: var(--panel)", "width: 352px"],
    [
        "9: label allocation",
        "grid-template-columns: var(--label)",
        "grid-template-columns: var(--panel)",
    ],
    ["9: numeric column", "1fr) var(--value)", "1fr) var(--label)"],
    ["9: checkbox", "width: var(--check)", "width: var(--row)"],
    ["9: thumb", "width: var(--thumb)", "width: var(--check)"],
    ["9: track", "height: var(--track)", "height: var(--thumb)"],
    ["9: corners", "border-radius: var(--radius)", "border-radius: var(--s1)"],
    [
        "9: shadow",
        "box-sizing: border-box",
        "box-shadow: 0 0 2px var(--line); box-sizing: border-box",
    ],
    ["12: variable alias", "--bg: #141210", "--bg: var(--forbidden); --forbidden: #ff0000"],
    ["12: unknown variable", "color: var(--ink)", "color: var(--missing)"],
    ["12: cyclic variable", "--bg: #141210", "--bg: var(--bg)"],
    ["8: missing hover", "button:hover:not(:disabled)", "button.rest:not(:disabled)"],
    ["8: transparent panel", "background: var(--bg2)", "background: transparent"],
    ["4: wrong second breakpoint", "max-width: 480px", "max-width: 480px) and (min-width: 400px"],
    ["2: heading case missing", "text-transform: uppercase", "text-transform: none"],
    ["12: scoped variable", "--bg: #141210", "--bg: #141210; } .other { --bg: #ff0000"],
];
for (const [name, before, after] of controlMutations)
    test(`example-controls rejects ${name}`, () => {
        expect(controlFixture).toContain(before);
        expect(
            controlViolations({ sheets: [controlFixture.replace(before, after)], lists: [] })
                .length,
        ).toBeGreaterThan(0);
    });

test("variables may alias site tokens and inline labels retain the body type role", () => {
    expect(
        controlViolations({
            sheets: [
                controlFixture.replace("--bg: #141210", "--bg: var(--surface); --surface: #141210"),
            ],
            lists: ["font: 400 13px/20px 'IBM Plex Sans', system-ui, sans-serif; color: #f0e6d6"],
        }),
    ).toEqual([]);
    expect(
        controlViolations({ sheets: [controlFixture], lists: ["font-size: 12px"] }).length,
    ).toBeGreaterThan(0);
    expect(
        controlViolations({ sheets: [controlFixture], lists: ["color: r\\65 d"] }).length,
    ).toBeGreaterThan(0);
});

test("scene backgrounds keep the stepped-page and voxel exceptions, not an overlay colour escape", () => {
    const css = { sheets: ["body { background: #89b6e9; } .help { color: #f0e6d6; }"], lists: [] };
    expect(controlViolations(sceneBackgrounds(css, { example: "voxel" }))).toEqual([]);
    expect(controlViolations(sceneBackgrounds(css, { stepped: true }))).toEqual([]);
    expect(controlViolations(sceneBackgrounds(css, { example: "minimal" })).length).toBeGreaterThan(
        0,
    );
    expect(
        controlViolations(
            sceneBackgrounds(
                { sheets: [".help { color: #89b6e9; }"], lists: [] },
                { example: "voxel", stepped: true },
            ),
        ).length,
    ).toBeGreaterThan(0);
});

test("gameplay geometry stays outside the panel kit without exempting its type or colours", () => {
    const css =
        ".reticle { width: 12px; height: 12px; border-radius: 50%; color: #f0e6d6; } gameplay { width: 140px; height: 140px; border-radius: 50%; }";
    expect(controlViolations({ sheets: [css], lists: [] })).toEqual([]);
    for (const extra of [
        "gameplay { color: #ff0000; }",
        "gameplay { font-size: 14px; }",
        ".controls { border-radius: 50%; }",
    ])
        expect(controlViolations({ sheets: [css + extra], lists: [] }).length).toBeGreaterThan(0);
});

test("1: embedded host uses light counterparts without allowing scene materials", () => {
    let light = controlFixture;
    for (const key of Object.keys(EXAMPLE_CONTROLS.dark) as (keyof typeof EXAMPLE_CONTROLS.dark)[])
        light = light.replaceAll(EXAMPLE_CONTROLS.dark[key], EXAMPLE_CONTROLS.light[key]);
    expect(controlViolations({ sheets: [light], lists: [] }, true)).toEqual([]);
    expect(
        controlViolations({ sheets: [light.replace("#f7f3ec", "#141210")], lists: [] }, true)
            .length,
    ).toBeGreaterThan(0);
});

test("11/12: local script and CSS imports, font weights, woff2 and adjacent OFL are checked", async () => {
    const sandbox = await mkdtemp(join(tmpdir(), "shallot-controls-"));
    const root = join(sandbox, "example");
    const imported = join(sandbox, "imported.css");
    try {
        await mkdir(join(root, "src"), { recursive: true });
        await mkdir(join(root, "public/fonts"), { recursive: true });
        await writeFile(join(root, "src/main.ts"), 'import "../../controls.css";');
        await writeFile(join(sandbox, "controls.css"), "@import url(./imported.css);");
        await writeFile(imported, controlFixture);
        for (const name of ["plex", "mono", "mono-bold"])
            await writeFile(join(root, `public/fonts/${name}.woff2`), "wOF2fixture");
        await writeFile(join(root, "public/fonts/OFL.txt"), "SIL OPEN FONT LICENSE Version 1.1");
        expect(await exampleControls(root)).toEqual([]);
        for (const extra of [
            ".ec { color: red; }",
            ":root { --bad: #ff0000; } .ec { color: var(--bad); }",
            '@import "https://example.com/hidden.css";',
        ]) {
            await writeFile(imported, controlFixture + extra);
            expect((await exampleControls(root)).length).toBeGreaterThan(0);
        }
        await writeFile(imported, controlFixture);
        for (const [file, content] of [
            ["mono-bold.woff2", "not a font"],
            ["OFL.txt", "not a licence"],
        ]) {
            const path = join(root, "public/fonts", file!);
            const original = await readFile(path);
            await writeFile(path, content!);
            expect((await exampleControls(root)).length).toBeGreaterThan(0);
            await writeFile(path, original);
        }
        expect(
            (
                await fontViolations(root, {
                    sheets: [controlFixture.replace("font-weight: 700", "font-weight: 600")],
                    lists: [],
                })
            ).length,
        ).toBeGreaterThan(0);
        expect(
            (
                await fontViolations(root, {
                    sheets: [
                        controlFixture.replace(
                            "/fonts/plex.woff2",
                            "https://example.com/font.woff2",
                        ),
                    ],
                    lists: [],
                })
            ).length,
        ).toBeGreaterThan(0);
        expect(
            (
                await fontViolations(root, {
                    sheets: [controlFixture.replace("/fonts/plex.woff2", "/fonts/missing.woff2")],
                    lists: [],
                })
            ).length,
        ).toBeGreaterThan(0);
    } finally {
        await removeTestDirectory(sandbox);
    }
});

// Moving an example changes this list alone: new rules become a test, old rules leave.
const migrated = new Set<string>(["voxel"]);

for (const example of await readdir("examples")) {
    (migrated.has(example) ? test : test.todo)(`${example} follows example-controls`, async () => {
        expect(
            await exampleControls(join("examples", example), example === "embedded-scene"),
        ).toEqual([]);
    });
    if (migrated.has(example)) continue;
    test(`${example} CSS and inline styles use the UI type, shape and colour tokens`, async () => {
        const failures: string[] = [];
        for (const path of await sources(join("examples", example))) {
            const source = await readFile(path, "utf8");
            failures.push(
                ...[
                    ...styleSources(path, source).flatMap((style) => violations(style)),
                    ...(await colors(path, source, { example, stepped: steppedPage(path) })),
                    ...spacingViolations(await cssOf(path, source)),
                    ...rowViolations(await cssOf(path, source)),
                ].map((failure) => `${path}: ${failure}`),
            );
        }
        expect(failures).toEqual([]);
    });
}
