import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdir, mkdtemp, readdir, readFile, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative, resolve } from "node:path";
import { parse } from "@babel/parser";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { parseCss, parse as parseSvelte } from "svelte/compiler";
import { EXAMPLE_CONTROLS } from "../diagnostics/default-look/palette";
import { normalizePath } from "../src/project/paths";
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

const COLOR_FUNCTIONS = /\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color|color-mix)\(([^)]*)\)?/gi;
// A literal colour in hex, three, four, six or eight digits; any other length is not a colour.
const HEX = /#[0-9a-f]+\b/gi;
const KEYWORDS = new Set(["transparent", "currentcolor", "inherit", "none"]);
// A computed part of a cssText template or Svelte style attribute; it is not read.
const COMPUTED_NAME = "--computed";
const COMPUTED = `${COMPUTED_NAME}()`;

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
                if (element.tagName === "canvas")
                    css.sheets.push(`canvas{${element.getAttribute("style")!}}`);
                else css.lists.push(element.getAttribute("style")!);
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

function cssTextStrings(source: string, gameplay = false): string[] {
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
            const text =
                node.right.type === "StringLiteral"
                    ? node.right.value
                    : node.right.type === "TemplateLiteral"
                      ? node.right.quasis
                            .map((quasi: { value: { cooked: string } }) => quasi.value.cooked)
                            .join(COMPUTED)
                      : undefined;
            if (text !== undefined) {
                const name = node.left.object.object?.name;
                const selector =
                    name === "stick"
                        ? "[data-touch-stick]"
                        : name === "knob"
                          ? "[data-touch-stick] > div"
                          : "gameplay";
                lists.push(gameplay ? `${selector}{${text}}` : text);
            }
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

// Only the canvas background of a stepped test page is scene presentation.
function steppedPage(path: string): boolean {
    return /^examples[\\/][^\\]+[\\/]tests[\\/]stepped\.html$/.test(path);
}

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
            stepped: steppedPage(normalizePath(relative(process.cwd(), path))),
        });
        result.sheets.push(
            ...css.sheets.map((sheet) =>
                sheet.replace(
                    /url\(\s*['"]?([^)'"\s]+\.woff2)['"]?\s*\)/g,
                    (whole, url: string) => {
                        if (/^(?:\/|[a-z]+:)/i.test(url)) return whole;
                        const asset = normalizePath(
                            relative(resolve(root, "public"), resolve(dirname(path), url)),
                        );
                        return `url('/${asset}')`;
                    },
                ),
            ),
        );
        if (path.endsWith(join("src", "player.ts"))) {
            result.sheets.push(...cssTextStrings(source, true));
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
            ...css.sheets.join("\n").matchAll(/@import\s+(?:url\(\s*)?["']?([^\s"');]+)["']?/gi),
        ];
        if (/\.[cm]?[jt]s$|\.svelte$/.test(path))
            imports.push(
                ...source.matchAll(/\bimport\s+(?:[^;]*?\sfrom\s*)?["']([^"']+\.css)["']/g),
            );
        const names = imports.map((match) => match[1]!);
        if (path.endsWith(".html"))
            await new HTMLRewriter()
                .on("link[href]", {
                    element(element) {
                        if (
                            element
                                .getAttribute("rel")
                                ?.toLowerCase()
                                .split(/\s+/)
                                .includes("stylesheet")
                        )
                            names.push(element.getAttribute("href")!);
                    },
                })
                .transform(new Response(source))
                .text();
        for (const name of names) {
            if (/^(?:https?:|\/\/|data:)/.test(name)) throw new Error(`remote stylesheet: ${name}`);
            await load(
                name.startsWith("/src/")
                    ? join(root, name)
                    : name.startsWith("/")
                      ? join(root, "public", name)
                      : resolve(dirname(path), name),
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
            if (value.type === "Atrule" && value.name.toLowerCase() === "media")
                media += `${media ? " and " : ""}${value.prelude}`;
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

function mediaModes(media: string): string[] {
    if (!media) return ["fine"];
    const modes = selectors(media).map((query) =>
        query.match(
            /^(?:(?:screen|all)\s+and\s+)?\(\s*(pointer\s*:\s*(fine|coarse)|max-width\s*:\s*480px)\s*\)$/i,
        ),
    );
    return modes.some((mode) => !mode)
        ? []
        : modes.map((mode) => mode![2]?.toLowerCase() ?? "phone");
}

function boxValues(value: string): string[] {
    const [top, right = top, bottom = top, left = right] = value.split(/\s+/);
    return [top!, right!, bottom!, left!].map((v) => (v === "0" ? "0px" : v));
}

function spacingMatches(property: string, value: string, expected: string): boolean {
    if (/gap$/.test(property)) return value === expected;
    const sides: Record<string, number[]> = {
        top: [0],
        right: [1],
        bottom: [2],
        left: [3],
        block: [0, 2],
        inline: [1, 3],
        "block-start": [0],
        "block-end": [2],
        "inline-start": [3],
        "inline-end": [1],
    };
    const indices = sides[property.replace(/^(padding|margin)-/, "")] ?? [0, 1, 2, 3];
    if (value.split(/\s+/).length > indices.length) return false;
    const values = indices.length === 4 ? boxValues(value) : value.split(/\s+/);
    return indices.every((side, i) => (values[i] ?? values[0]) === boxValues(expected)[side]);
}

function targetsControl(selector: string, kind: string): boolean {
    const tag = kind === "checkbox" || kind === "range" ? "input" : kind;
    function covers(list: any): boolean {
        return list.children.some((branch: any) =>
            branch.children.at(-1).selectors.every((node: any) => {
                if (node.type === "TypeSelector") return node.name === tag || node.name === "*";
                if (node.type === "AttributeSelector")
                    return (
                        (node.name === "type" && node.matcher === "=" && node.value === kind) ||
                        (kind === "button" && node.name === "data-touch-action")
                    );
                if (node.type !== "PseudoClassSelector") return false;
                if (node.name === "focus-visible") return true;
                if (node.name === "is" || node.name === "where") return covers(node.args);
                // Disabled controls cannot focus; other restrictions do not cover the whole kind.
                return (
                    node.name === "not" &&
                    node.args.children.every(
                        (b: any) =>
                            b.children.length === 1 &&
                            b.children[0].selectors.length === 1 &&
                            b.children[0].selectors[0].name === "disabled",
                    )
                );
            }),
        );
    }
    return covers((parseCss(`${selector}{}`).children[0] as any).prelude);
}

type StateRole = [
    state: string,
    part: string,
    property: string,
    role: keyof typeof EXAMPLE_CONTROLS.dark,
];

function requiredStates(kind: string): StateRole[] {
    const states: StateRole[] = [["disabled", "", "color", "muted"]];
    if (kind === "range") {
        for (const engine of ["webkit", "moz"]) {
            states.push([
                "rest",
                engine === "webkit" ? "-webkit-slider-runnable-track" : "-moz-range-track",
                "background",
                "line",
            ]);
            for (const [state, role] of [
                ["rest", "muted"],
                ["hover", "ink"],
                ["active", "gold"],
                ["disabled", "dim"],
            ] as const)
                states.push([
                    state,
                    engine === "webkit" ? "-webkit-slider-thumb" : "-moz-range-thumb",
                    "background",
                    role,
                ]);
        }
    } else if (kind === "checkbox")
        states.push(
            ["rest", "", "background", "raised"],
            ["disabled", "", "background", "bg2"],
            ["rest", "", "border", "muted"],
            ["checked", "", "background", "dim"],
            ["checked", "after", "color", "ink"],
            ["hover", "", "border", "ink"],
            ["active", "", "background", "line"],
        );
    else {
        states.push(
            ["rest", "", "background", "raised"],
            ["disabled", "", "background", "bg2"],
            ["rest", "", "border", "line"],
            ["rest", "", "color", "ink"],
            ["hover", "", "border", "muted"],
            ["active", "", "background", "line"],
        );
        if (kind !== "select") states.push(["selected", "", "border", "gold"]);
    }
    return states;
}

function controlState(selector: string): string {
    const positive = selector.replace(/:not\([^)]*\)/g, "");
    const states = ["disabled", "active", "hover", "checked", "focus-visible"].filter((state) =>
        positive.includes(`:${state}`),
    );
    return states.length > 1
        ? "compound"
        : (states[0] ??
              (/aria-(?:pressed|current|selected)=['"]?true|\.selected\b/.test(positive)
                  ? "selected"
                  : "rest"));
}

function stateSubject(selector: string, kind: string): boolean {
    return targetsControl(
        selector
            .replace(/::[\w-]+/g, "")
            .replace(/:not\(:disabled\)|:(?:disabled|active|hover|checked)\b/g, (state) =>
                state.startsWith(":not") ? state : ":focus-visible",
            )
            .replace(
                /\[aria-(?:pressed|current|selected)=['"]?true['"]?\]|\.selected\b/g,
                ":focus-visible",
            )
            .replace(
                /(?:\b(?:button|div|li|span))?(?:\.option\b|\[role=['"]?option['"]?\])/g,
                "option",
            ),
        kind,
    );
}

function controlViolations(css: Css, light = false, embedded = false): string[] {
    const failures: string[] = [];
    let rules: StyleRule[];
    try {
        rules = styleRules(css).flatMap((rule) =>
            selectors(rule.selector).map((selector) => ({ ...rule, selector })),
        );
    } catch (error) {
        return [`css: ${error}`];
    }
    const hostPalette: Record<keyof typeof EXAMPLE_CONTROLS.light, string> & { raised?: string } =
        EXAMPLE_CONTROLS[light ? "light" : "dark"];
    const tokens: Record<string, string> = {
        ...EXAMPLE_CONTROLS.tokens,
        ...hostPalette,
        "header-surface": hostPalette.bg2,
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
        let text = value;
        for (;;) {
            const call = [...text.matchAll(/("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|var\(/gi)].find(
                (match) => !match[1],
            );
            if (!call) break;
            const start = call.index;
            let end = start + 4;
            let depth = 1;
            let comma = -1;
            let quote = "";
            for (; end < text.length && depth; end++) {
                const char = text[end]!;
                if (quote) {
                    if (char === quote && text[end - 1] !== "\\") quote = "";
                    continue;
                }
                if (char === "'" || char === '"') quote = char;
                else if (char === "(") depth++;
                else if (char === ")") depth--;
                else if (char === "," && depth === 1 && comma === -1) comma = end;
            }
            if (depth) {
                failures.push("unclosed variable");
                return "invalid";
            }
            const name = decodeEscapes(text.slice(start + 4, comma < 0 ? end - 1 : comma).trim());
            const fallback = comma < 0 ? undefined : text.slice(comma + 1, end - 1);
            const values = definitions.get(name);
            let replacement = "invalid";
            if (stack.includes(name)) failures.push(`cyclic variable: ${name}`);
            else if (values) {
                const expanded = values.map((v) => expand(v, [...stack, name]));
                if (new Set(expanded).size !== 1) failures.push(`ambiguous variable: ${name}`);
                replacement = expanded[0]!;
            } else if (fallback !== undefined) replacement = expand(fallback, stack);
            else failures.push(`unknown variable: ${name}`);
            text = text.slice(0, start) + replacement + text.slice(end);
        }
        return text;
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
        const page =
            /^(?::root|\*|html|body|html,body|h1|\.page|\.scene|\.frame|\.description|\.frame-error|\.bootstrap-panel|\.title-card|\.lockup|\.notice|\.error|\[data-shallot-bootstrap\](?:\[data-state="error"\])?|\[data-bootstrap-(?:label|track)\])$/.test(
                s,
            );
        const palette: Record<keyof typeof EXAMPLE_CONTROLS.light, string> & { raised?: string } =
            EXAMPLE_CONTROLS[
                light && (!embedded || /^(?::root|body|h1|\.page|\.description)$/.test(s))
                    ? "light"
                    : "dark"
            ];
        const stick = /^\[data-touch-stick\](?:\s*>\s*div)?$/.test(s);
        const gameplay =
            stick ||
            /^(?:gameplay|\[data-touch-stick\]|button\[data-touch-action\](?:\[[^\]]+\]|:[\w():-]+)*|\.reticle(?:-dot)?(?:\[hidden\])?|\.hint(?:\[hidden\])?|\.key-slot(?:\s+(?:>\s*)?(?:\*|kbd))?|\.hold-ring|\.track|\.progress)$/.test(
                s,
            );
        const passive = page || (gameplay && !/\bbutton\b/.test(s));
        const modes = mediaModes(media);
        const touch = modes.includes("coarse") || modes.includes("phone");
        if (
            !gameplay &&
            rowSelector(s) &&
            !/:hover|:active|:focus|:disabled|:is|aria-/.test(s) &&
            !heights.has(s)
        )
            heights.set(s, new Set());
        if (
            /^\.ec$|\.panel-title\b|\.section-title\b|\bh[2-6]\b|\boutput\b|\.value\b/.test(s) ||
            rule.declarations.some(
                (d) =>
                    /^(?:font(?:-family|-size|-weight)?|line-height)$/.test(d.property) &&
                    (d.value !== "inherit" || s === "body"),
            )
        )
            types.set(s, [...(types.get(s) ?? []), ...rule.declarations]);
        if (!modes.length) failures.push(`media: ${media}`);
        for (const { property: rawProperty, value: raw } of rule.declarations) {
            const p = decodeEscapes(rawProperty).toLowerCase();
            const v = expand(raw).trim();
            const fail = (role: string) => failures.push(`${role}: ${s} { ${p}: ${v} }`);
            if (p.startsWith("--")) {
                const expected = tokens[p.slice(2)];
                if (expected && !has(v, expected)) fail("token");
                // Unregistered aliases are still checked as literals, not an escape hatch.
            }
            const unquoted = (value: string) =>
                value
                    .replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, "")
                    .replace(/url\([^)]*\)/gi, "");
            const text = decodeEscapes(unquoted(v));
            const colors = `${text} ${decodeEscapes(unquoted(raw))}`;
            for (const match of colors.matchAll(HEX))
                if (
                    !Object.values(palette).includes(
                        (stick && /^#[0-9a-f]{8}$/i.test(match[0])
                            ? match[0].slice(0, 7)
                            : match[0]
                        ).toLowerCase() as typeof palette.bg,
                    )
                )
                    fail("chrome color");
            for (const match of colors.matchAll(COLOR_FUNCTIONS))
                if (!match[1]?.includes(`${COMPUTED_NAME}(`)) fail("color syntax");
            for (const word of colors.match(/(?<![\w$#.-])[a-z]+(?![\w(-])/gi) ?? [])
                if (!KEYWORDS.has(word.toLowerCase()) && Bun.color(word, "hex") !== null)
                    fail("color syntax");
            if (p.startsWith("--")) {
                const values = new Set([...Object.values(tokens), "0", "1fr"]);
                for (const number of text.matchAll(
                    /(?<![\w#.-])[+-]?\d*\.?\d+(?:e[+-]?\d+)?(?:[a-z]+|%)?(?![\w.])/gi,
                ))
                    if (!page && !values.has(number[0])) fail("geometry token");
                continue;
            }
            if (p === "zoom" || /shadow/.test(p)) fail("surface");
            if (
                (/image|mask/.test(p) && v !== "none") ||
                /(?:gradient|url|image|cross-fade|element|paint)\s*\(/i.test(v)
            )
                fail("image surface");
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
                !(s === "h1" && has(v, `400 24px/20px ${tokens.mono}`))
            )
                fail("type role");
            if (p === "font-size" && v !== size && !(s === "h1" && v === "24px")) fail("type role");
            if (p === "font-family" && !has(v, family)) fail("type role");
            if (p === "font-weight" && v !== weight) fail("type role");
            if (p === "line-height" && v !== "20px") fail("type role");
            if (p === "letter-spacing" && (!heading || v !== ".08em")) fail("heading tracking");
            if (p === "text-transform" && v === "uppercase" && !heading) fail("heading case");
            if (!gameplay && /^(?:min-)?height$/.test(p) && rowSelector(s)) {
                if (v !== (touch ? "44px" : "32px")) fail("row");
                const coverage = heights.get(s) ?? new Set();
                for (const mode of modes) coverage.add(mode);
                heights.set(s, coverage);
            }
            if (!passive && /^(?:padding|margin|gap|row-gap|column-gap)(?:-|$)/.test(p)) {
                let expected = "0px";
                if (p.includes("gap"))
                    expected = /\.fields\b|\.categories\s*$/.test(s)
                        ? "16px"
                        : /\.section\b|\bnav\s*$|\.categories section/.test(s)
                          ? "4px"
                          : "8px";
                else if (p.startsWith("margin") && heading && /categories/.test(s))
                    expected = "0px 0px 8px";
                else if (p.startsWith("padding"))
                    expected = title
                        ? "12px 16px"
                        : heading
                          ? "0px 8px"
                          : /\.fields\b|\.scenes\b/.test(s)
                            ? "16px"
                            : /dependents/.test(s)
                              ? "0px 0px 0px 8px"
                              : /button|select/.test(s)
                                ? "4px 8px"
                                : "0px";
                if (
                    !spacingMatches(p, v, expected) &&
                    !/^0(?:px)?(?: 0(?:px)?)*$/.test(v) &&
                    !(
                        /slider-thumb/.test(s) &&
                        p === "margin-top" &&
                        has(v, "calc((4px - 16px) / 2)")
                    )
                )
                    fail("spacing role");
            }
            if (!passive && /^(?:top|right|bottom|left|inset)$/.test(p)) {
                const lengths = [...v.matchAll(/([+-]?\d*\.?\d+)px/g)].map((m) => m[1]);
                if (lengths.some((n) => n !== (touch ? "16" : "24")) && !/\.scenes\b/.test(s))
                    fail("inset role");
            }
            // State roles are independent of token membership.
            const disabled = /:disabled|aria-disabled/.test(s) && !/:not\(:disabled\)/.test(s);
            const hover = /:hover/.test(s);
            const pressed = /:active/.test(s);
            const selected = /aria-(?:pressed|current|selected)=['"]?true|\.selected\b/.test(s);
            const check = /checkbox/.test(s);
            const thumb = /slider-thumb|range-thumb/.test(s);
            const track = /slider-runnable-track|range-track/.test(s);
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
                      : disabled
                        ? "bg2"
                        : pressed
                          ? "line"
                          : check
                            ? /:checked/.test(s)
                                ? "dim"
                                : "raised"
                            : /controls|scenes|section-title/.test(s)
                              ? "bg2"
                              : "raised";
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
            if (p === "outline-offset" && v !== "2px") fail("focus role");
            if (
                role &&
                !passive &&
                (!palette[role] ||
                    !has(
                        v,
                        border && !p.endsWith("color")
                            ? `1px solid ${palette[role]}`
                            : palette[role]!,
                    )) &&
                !v.includes(COMPUTED_NAME)
            )
                fail("state role");
            if (
                background &&
                v === "transparent" &&
                /button|select|checkbox|controls|section-title/.test(s)
            )
                fail("state role");
            if (
                (hover || pressed) &&
                /button|select|checkbox|thumb/.test(s) &&
                !s.includes(":not(:disabled)")
            )
                fail("disabled priority");
            // Gameplay stick, touch buttons and reticles retain their geometry outside the panel kit.
            if (passive || /(?:^|\s)canvas$/.test(s)) continue;
            if (
                /^(?:transform(?:-.+)?|translate|rotate|scale|perspective(?:-.+)?)$/.test(p) &&
                v !== "none" &&
                !(
                    p === "transform" &&
                    /\.picker\b|\.scenes\b|\.hud\b/.test(s) &&
                    ["translateX(calc(0px - 50%))", "translateX(-50%)"].some((center) =>
                        has(v, center),
                    )
                )
            )
                fail("transform role");
            if (p === "width" && /\.controls\b/.test(s) && v !== "360px") fail("panel width role");
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
                ...(p === "transform" ? ["-50%"] : []),
                "100dvh",
                "60dvh",
                ...(p === "order" && /\bkbd\b/.test(s) ? ["-1"] : []),
            ]);
            let sizes: string[] = [];
            if (/radius/.test(p) || /^outline/.test(p)) sizes = ["2px"];
            else if (/border/.test(p)) sizes = ["1px"];
            else if (p === "width")
                sizes = check
                    ? ["16px"]
                    : thumb
                      ? ["8px"]
                      : /\.controls\b|\.name\b/.test(s)
                        ? /\.name\b/.test(s)
                            ? touch
                                ? ["280px", "44px", "4px"]
                                : ["296px"]
                            : ["360px"]
                        : /\.save-hud button/.test(s)
                          ? ["144px"]
                          : [];
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
            else if (/^(?:min-|max-)?height$/.test(p))
                sizes =
                    check || thumb
                        ? ["16px"]
                        : track
                          ? ["4px"]
                          : rowSelector(s)
                            ? [touch ? "44px" : "32px"]
                            : ["24px", "64px", ...(/\.scenes\b/.test(s) && touch ? ["16px"] : [])];
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
                /(?<![\w#.-])[+-]?\d*\.?\d+(?:e[+-]?\d+)?(?:[a-z]+|%)?(?![\w.])/gi,
            )) {
                if (!allowed.has(m[0]) && !sizes.includes(m[0])) fail("geometry role");
            }
        }
    }
    const focusRules = rules.filter((rule) => rule.selector.includes(":focus-visible"));
    for (const rule of focusRules) {
        if (
            !rule.declarations.some(
                (d) =>
                    d.property === "outline" && expand(d.value) === `2px solid ${hostPalette.gold}`,
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
                mediaModes(rule.media).includes("phone") &&
                rule.declarations.some(
                    (d) => d.property === "min-width" && expand(d.value) === "44px",
                ),
        )
    )
        failures.push("phone icon button width missing");
    for (const kind of ["button", "select", "option", "checkbox", "range"]) {
        if (
            !rules.some((rule) => rule.selector.includes(kind) && stateSubject(rule.selector, kind))
        )
            continue;
        if (!focusRules.some((rule) => stateSubject(rule.selector, kind)))
            failures.push(`focus missing: ${kind}`);
        for (const [state, part, property, role] of requiredStates(kind)) {
            if (
                !rules.some(
                    (rule) =>
                        controlState(rule.selector) === state &&
                        (rule.selector.match(/::([\w-]+)/)?.[1] ?? "") === part &&
                        stateSubject(rule.selector, kind) &&
                        rule.declarations.some((d) => {
                            const p = d.property.toLowerCase();
                            return (
                                (p === property || p === `${property}-color`) &&
                                hostPalette[role] !== undefined &&
                                has(
                                    expand(d.value),
                                    p === "border"
                                        ? `1px solid ${hostPalette[role]}`
                                        : hostPalette[role]!,
                                )
                            );
                        }),
                )
            )
                failures.push(`state missing: ${kind} ${state} ${part} ${property}`);
        }
    }
    for (const [selector, declarations] of types) {
        if (/:hover|:active|:focus|:disabled/.test(selector)) continue;
        if (!declarations.some((d) => d.property === "font" && expand(d.value) !== "inherit")) {
            for (const property of ["font-family", "font-size", "font-weight", "line-height"])
                if (!declarations.some((d) => d.property === property))
                    failures.push(`type ${property} missing: ${selector}`);
        }
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

const licenseText = (text: string) => text.replace(/^-+$/gm, "").replace(/\s+/g, " ").trim();
const OFL_TEXT = licenseText(
    await readFile("scripts/fixtures/example-controls/OFL-1.1.txt", "utf8"),
);

async function fontViolations(root: string, css: Css): Promise<string[]> {
    const failures: string[] = [];
    const sheet = css.sheets.join("\n");
    const faces = [...sheet.matchAll(/@font-face\s*\{([^}]+)\}/gi)].map((m) => m[1]!);
    const families = new Set(["IBM Plex Sans", "JetBrains Mono"]);
    for (const match of sheet.matchAll(/font-family:\s*['"]([^'"]+)['"]/gi))
        families.add(match[1]!);
    for (const family of families) {
        if (!sheet.includes(family) && !css.lists.some((s) => s.includes(family))) continue;
        const weights = new Set(["400"]);
        for (const face of faces) {
            if (face.includes(family))
                weights.add(face.match(/font-weight:\s*(\d+)/i)?.[1] ?? "400");
        }
        // The type-role gate reserves Mono 700 for panel titles; other text uses 400.
        if (
            family === "JetBrains Mono" &&
            styleRules(css).some((r) => /\.panel-title\b/.test(r.selector))
        )
            weights.add("700");
        for (const weight of weights) {
            const face = faces.find(
                (f) =>
                    f.includes(family) && new RegExp(`font-weight:\\s*${weight}\\b`, "i").test(f),
            );
            const url = face?.match(/url\(['"]?([^)'"\s]+)['"]?\)/i)?.[1];
            if (!url?.endsWith(".woff2") || !url.startsWith("/")) {
                failures.push(`font face: ${family} ${weight}`);
                continue;
            }
            const path = resolve(root, "public", `.${url}`);
            try {
                if (normalizePath(relative(resolve(root, "public"), path)).startsWith(".."))
                    throw new Error("outside public");
                const data = await readFile(path);
                if (
                    data.length < 48 ||
                    data.toString("ascii", 0, 4) !== "wOF2" ||
                    data.readUInt32BE(8) !== data.length
                )
                    failures.push(`woff2: ${url}`);
                const license = await readFile(join(dirname(path), "OFL.txt"), "utf8");
                if (!licenseText(license).includes(OFL_TEXT)) failures.push(`OFL: ${url}`);
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
        return [...controlViolations(css, light, light), ...(await fontViolations(root, css))];
    } catch (error) {
        return [`source: ${error}`];
    }
}

const controlFixture = await readFile("scripts/fixtures/example-controls/controls.css", "utf8");

test("example-controls candidate holds colour, type, spacing, rows, states and geometry roles", () => {
    expect(controlViolations({ sheets: [controlFixture], lists: [] })).toEqual([]);
});

function deleteStateProperty(sheet: string, selector: string, property: string): string {
    for (const rule of parseCss(sheet).children as any[]) {
        if (rule.type !== "Rule" || sheet.slice(rule.prelude.start, rule.prelude.end) !== selector)
            continue;
        const declaration = rule.block.children.find((d: any) => d.property === property);
        if (declaration)
            return (
                sheet.slice(0, declaration.start) +
                sheet.slice(declaration.end + (sheet[declaration.end] === ";" ? 1 : 0))
            );
    }
    throw new Error(`fixture declaration missing: ${selector} ${property}`);
}

for (const rest of ["bg", "line"]) {
    test(`8: controls cannot rest on ${rest}`, () => {
        const css = controlFixture.replaceAll(
            "background: var(--raised)",
            `background: var(--${rest})`,
        );
        expect(
            controlViolations({ sheets: [css], lists: [] }).some((failure) =>
                failure.startsWith("state role:"),
            ),
        ).toBe(true);
    });
}

for (const surface of ["raised", "line", "bg"]) {
    test(`8: disabled fields and buttons cannot rest on ${surface}`, () => {
        const css = controlFixture.replaceAll(
            ":disabled { background: var(--bg2)",
            `:disabled { background: var(--${surface})`,
        );
        const failures = controlViolations({ sheets: [css], lists: [] });
        for (const kind of ["button", "select", "checkbox"])
            expect(failures).toContain(`state missing: ${kind} disabled  background`);
    });
}

const stateDeletions: [string, string, string][] = [
    ["button/select rest surface", ".ec button, .ec select", "background"],
    ["button/select disabled surface", ".ec button:disabled, .ec select:disabled", "background"],
    ["checkbox disabled surface", ".ec input[type='checkbox']:disabled", "background"],
    ["button/select rest border", ".ec button, .ec select", "border"],
    ["button/select rest text", ".ec button, .ec select", "color"],
    [
        "button/select hover",
        ".ec button:hover:not(:disabled), .ec select:hover:not(:disabled)",
        "border-color",
    ],
    [
        "button/select pressed",
        ".ec button:active:not(:disabled), .ec select:active:not(:disabled)",
        "background",
    ],
    [
        "button selected",
        ".ec button[aria-current='true'], .ec button[aria-pressed='true']",
        "border-color",
    ],
    ["checkbox unchecked surface", ".ec input[type='checkbox']", "background"],
    ["checkbox unchecked border", ".ec input[type='checkbox']", "border"],
    ["checkbox checked fill", ".ec input[type='checkbox']:checked", "background"],
    ["checkbox check mark", ".ec input[type='checkbox']:checked::after", "color"],
    ["checkbox hover", ".ec input[type='checkbox']:hover:not(:disabled)", "border-color"],
    ["checkbox pressed", ".ec input[type='checkbox']:active:not(:disabled)", "background"],
    ["disabled text", ".ec :disabled", "color"],
];
for (const engine of ["webkit", "moz"]) {
    const track = engine === "webkit" ? "-webkit-slider-runnable-track" : "-moz-range-track";
    const thumb = engine === "webkit" ? "-webkit-slider-thumb" : "-moz-range-thumb";
    stateDeletions.push([
        `${engine} range track`,
        `.ec input[type='range']::${track}`,
        "background",
    ]);
    for (const state of ["", ":hover:not(:disabled)", ":active:not(:disabled)", ":disabled"])
        stateDeletions.push([
            `${engine} range handle ${state || "rest"}`,
            `.ec input[type='range']${state}::${thumb}`,
            "background",
        ]);
}
for (const [name, selector, property] of stateDeletions)
    test(`8: deleting ${name} fails`, () => {
        expect(
            controlViolations({
                sheets: [deleteStateProperty(controlFixture, selector, property)],
                lists: [],
            }).some((failure) => failure.startsWith("state missing:")),
        ).toBe(true);
    });

test("8: deleting both disabled slider handles fails independently of disabled text", () => {
    let sheet = controlFixture;
    for (const part of ["-webkit-slider-thumb", "-moz-range-thumb"])
        sheet = deleteStateProperty(
            sheet,
            `.ec input[type='range']:disabled::${part}`,
            "background",
        );
    expect(
        controlViolations({ sheets: [sheet], lists: [] }).filter((failure) =>
            failure.startsWith("state missing: range disabled"),
        ),
    ).toHaveLength(2);
});

test("8: button-only disabled text cannot cover other control kinds", () => {
    const sheet = controlFixture.replace(".ec :disabled {", ".ec button:disabled {");
    const failures = controlViolations({ sheets: [sheet], lists: [] });
    for (const kind of ["select", "checkbox", "range"])
        expect(failures).toContain(`state missing: ${kind} disabled  color`);
});

test("8: styled options require rest, hover, pressed, selected and disabled text", () => {
    const option =
        ".ec option { background: var(--raised); border: var(--hair) solid var(--line); color: var(--ink); }\n.ec option:disabled { background: var(--bg2); }\n.ec option:hover:not(:disabled) { border-color: var(--muted); }\n.ec option:active:not(:disabled) { background: var(--line); }\n.ec option[aria-selected='true'] { border-color: var(--gold); }\n.ec option:focus-visible { outline: var(--radius) solid var(--gold); outline-offset: var(--radius); }";
    const sheet = controlFixture + option;
    expect(controlViolations({ sheets: [sheet], lists: [] })).toEqual([]);
    for (const [selector, property] of [
        [".ec option", "background"],
        [".ec option", "border"],
        [".ec option", "color"],
        [".ec option:hover:not(:disabled)", "border-color"],
        [".ec option:active:not(:disabled)", "background"],
        [".ec option[aria-selected='true']", "border-color"],
        [".ec :disabled", "color"],
    ])
        expect(
            controlViolations({
                sheets: [deleteStateProperty(sheet, selector!, property!)],
                lists: [],
            }).some((failure) => failure.startsWith("state missing: option")),
        ).toBe(true);
});

const controlMutations: [string, string, string][] = [
    [
        "1: gradient gold panel",
        "background: var(--bg2)",
        "background: linear-gradient(var(--bg2), var(--gold))",
    ],
    ["1: image panel", "background: var(--bg2)", "background: url('/surface.png') var(--bg2)"],
    [
        "1: background image",
        "background: var(--bg2)",
        "background: var(--bg2); background-image: url('/surface.png')",
    ],
    [
        "2: output inherits sans",
        ".ec output { font: var(--regular) var(--small)/var(--leading) var(--mono);",
        ".ec output { font-size: var(--small);",
    ],
    [
        "3: wrong heading padding sides",
        "padding: var(--zero) var(--s2)",
        "padding: var(--zero) var(--zero) var(--s2)",
    ],
    ["5: desktop picker name slot", "width: 296px", "width: 295px"],
    ["5: phone picker name cap", "min(280px,", "min(281px,"],
    ["3: wrong heading padding-left", "padding: var(--zero) var(--s2)", "padding-left: var(--s4)"],
    [
        "4: print coarse is not screen coverage",
        "@media (pointer: coarse)",
        "@media print and (pointer: coarse)",
    ],
    [
        "4: negated coarse is not coarse coverage",
        "@media (pointer: coarse)",
        "@media not (pointer: coarse)",
    ],
    [
        "4: coarse limited to phone is not coarse coverage",
        "@media (pointer: coarse)",
        "@media (pointer: coarse) and (max-width: 480px)",
    ],
    [
        "8: summary-only focus",
        ":is(button, input, select, summary):focus-visible",
        "summary:focus-visible",
    ],
    [
        "8: range-only focus omits checkboxes",
        ":is(button, input, select, summary):focus-visible",
        ":is(button, input[type='range'], select, summary):focus-visible",
    ],
    ["9: panel takes row token", "width: var(--panel); max-width", "width: var(--row); max-width"],
    [
        "9: unowned negative transform",
        "box-sizing: border-box",
        "transform: translateX(-17px); box-sizing: border-box",
    ],
    [
        "9: unowned positive transform",
        "box-sizing: border-box",
        "transform: translateX(16px); box-sizing: border-box",
    ],
    ["12: computed-name fallback", "color: var(--ink)", "color: var(--computed, #ff0000)"],
    ["12: nested fallback", "color: var(--ink)", "color: var(--missing, var(--other, #ff0000))"],
    ["12: dormant fallback literal", "color: var(--ink)", "color: var(--ink, #ff0000)"],
    ["12: dormant fallback function", "color: var(--ink)", "color: var(--ink, rgb(255 0 0))"],
    [
        "2: body inherits host family",
        "font: var(--regular) var(--body)/var(--leading) var(--sans)",
        "font-size: var(--body)",
    ],
    [
        "2: body inherits host scale",
        "font: var(--regular) var(--body)/var(--leading) var(--sans)",
        "font: inherit",
    ],
    ["9: unknown dimension unit", "width: var(--panel); max-width", "width: 1in; max-width"],
    ["9: independent rotation", "box-sizing: border-box", "rotate: 1turn; box-sizing: border-box"],
    ["9: independent scaling", "box-sizing: border-box", "scale: 2; box-sizing: border-box"],
    ["3: negative desktop inset", "top: var(--s5)", "top: -24px"],
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

test("audited screen queries, explicit type longhands and nested safe fallbacks pass", () => {
    for (const sheet of [
        controlFixture.replaceAll("@media", "@MEDIA").replaceAll("@font-face", "@FONT-FACE"),
        controlFixture + `.ec::before { content: "var(--not-a-variable), red"; }`,
        controlFixture + `.ec.hud { transform: translateX(-50%); }`,
        controlFixture + `.ec::before { content: 'it\\'s red'; }`,
        controlFixture.replace("(pointer: coarse)", "screen and (pointer: coarse)"),
        controlFixture.replaceAll(
            "translateX(calc(var(--zero) - var(--half)))",
            "translateX(-50%)",
        ),
        controlFixture.replace(
            "font: var(--regular) var(--small)/var(--leading) var(--mono); font-variant-numeric",
            "font-family: var(--mono); font-size: var(--small); font-weight: var(--regular); line-height: var(--leading); font-variant-numeric",
        ),
        controlFixture.replace(
            "padding: var(--zero) var(--s2)",
            "padding-top: 0px; padding-right: 8px; padding-bottom: 0px; padding-left: 8px",
        ),
        controlFixture.replace(
            "color: var(--ink)",
            "color: var(--missing, var(--other, var(--ink)))",
        ),
    ])
        expect(controlViolations({ sheets: [sheet], lists: [] })).toEqual([]);
    for (const extra of [
        ".ec.controls { transform: translateX(-17px); }",
        ".ec.hud { transform: translateX(-17px); }",
        ".ec { color: var(--computed, #ff0000); }",
    ])
        expect(
            controlViolations({ sheets: [controlFixture + extra], lists: [] }).length,
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
        ".reticle { width: 12px; height: 12px; border-radius: 50%; border: 1px solid #f0e6d6; } .key-slot kbd { min-width: 28px; height: 28px; } .progress { stroke: #f0e6d6; } gameplay { width: 140px; height: 140px; border-radius: 50%; } canvas { width: 640px; height: 360px; }";
    expect(controlViolations({ sheets: [css], lists: [] })).toEqual([]);
    for (const extra of [
        "gameplay { color: #ff0000; }",
        "gameplay { font-size: 14px; }",
        ".controls { border-radius: 50%; }",
        ".hint { color: #ff0000; }",
        ".key-slot kbd { font-size: 14px; }",
        "canvas { color: #ff0000; }",
        "button[data-touch-action] { width: 64px; height: 64px; border-radius: 50%; }", // Missing button states
        ".save-hud button { width: 132px; }",
        ".other { width: 144px; }",
        ".hint button { height: 64px; }",
        "canvas .controls { border-radius: 50%; }",
    ])
        expect(controlViolations({ sheets: [css + extra], lists: [] }).length).toBeGreaterThan(0);
});

test("gameplay stick alone accepts translucent site colours, including its authored knob", () => {
    const sheet =
        "[data-touch-stick] { background: #141210cc; } [data-touch-stick] > div { background: #f0e6d680; }";
    expect(controlViolations({ sheets: [sheet], lists: [] })).toEqual([]);
    expect(
        cssTextStrings(
            'stick.style.cssText = "background:#141210cc"; knob.style.cssText = "background:#f0e6d680";',
            true,
        ),
    ).toEqual([
        "[data-touch-stick]{background:#141210cc}",
        "[data-touch-stick] > div{background:#f0e6d680}",
    ]);
    for (const selector of [
        "button[data-touch-action]",
        ".ec",
        ".ec button",
        "[data-touch-stick] button",
    ])
        expect(
            controlViolations({
                sheets: [`${selector} { background: #141210cc; }`],
                lists: [],
            }).some((failure) => failure.startsWith("chrome color:")),
        ).toBe(true);
    expect(
        controlViolations({ sheets: ["[data-touch-stick] { background: #ff0000cc; }"], lists: [] }),
    ).not.toEqual([]);
});

test("passive page geometry does not excuse off-site colours, type, opacity or 24px panel text", () => {
    const sheet =
        "body { background: #141210; font: 400 13px/20px 'IBM Plex Sans', system-ui, sans-serif; } h1 { font: 400 24px/20px 'JetBrains Mono', ui-monospace, monospace; } .lockup { width: min(80%, 420px); }";
    expect(controlViolations({ sheets: [sheet], lists: [] })).toEqual([]);
    for (const extra of [
        ".notice { color: #ddaa3f; }",
        ".notice { color: r\\65 d; }",
        ".notice { background: #141210cc; }",
        ".notice { opacity: .5; }",
        ".notice { font-size: 24px; }",
        ".controls h1 { font: 400 24px/20px 'JetBrains Mono', ui-monospace, monospace; }",
        ".notice { zoom: 1; }",
        ".notice { color: rgb(240 230 214); }",
        ".notice { color: var(--missing); }",
    ])
        expect(controlViolations({ sheets: [sheet + extra], lists: [] }).length).toBeGreaterThan(0);
});

test("world-space Text fields are data, not CSS; CSS authored in script strings still runs the gate", () => {
    expect(
        styleSources("scene.ts", "const label = { font: 1, fontSize: 0.5 }; ").some((s) =>
            /font:/.test(s),
        ),
    ).toBe(false);
    const strings = styleSources("scene.ts", "const css = '.notice { font: bold 48px serif; }';");
    expect(controlViolations({ sheets: strings, lists: [] }).length).toBeGreaterThan(0);
});

test("touch action geometry does not excuse missing button states", async () => {
    const sheet = await readFile("examples/first-person/src/touch.css", "utf8");
    expect(controlViolations({ sheets: [sheet], lists: [] })).toEqual([]);
    for (const state of ["hover", "active", "focus-visible", "disabled"]) {
        const missing = sheet.replaceAll(
            `button[data-touch-action]:${state}`,
            `.not-an-action:${state}`,
        );
        expect(controlViolations({ sheets: [missing], lists: [] }).length).toBeGreaterThan(0);
    }
});

test("1: light counterparts have no control rest surface", () => {
    let light = controlFixture;
    for (const key of Object.keys(
        EXAMPLE_CONTROLS.light,
    ) as (keyof typeof EXAMPLE_CONTROLS.light)[])
        light = light.replaceAll(EXAMPLE_CONTROLS.dark[key], EXAMPLE_CONTROLS.light[key]);
    expect(controlViolations({ sheets: [light], lists: [] }, true).length).toBeGreaterThan(0);
    expect(
        controlViolations({ sheets: [light.replace("#f7f3ec", "#141210")], lists: [] }, true)
            .length,
    ).toBeGreaterThan(0);
});

test("embedded host and dark frame are checked independently", async () => {
    const sheet = await readFile("examples/embedded-scene/src/style.css", "utf8");
    expect(controlViolations({ sheets: [sheet], lists: [] }, true, true)).toEqual([]);
    for (const changed of [
        sheet.replace("--bg: #f7f3ec", "--bg: #141210"),
        sheet.replace("background: #1c1917", "background: #efe9df"),
        sheet.replace("--frame-error: #f0e6d6", "--frame-error: #2a231e"),
    ])
        expect(
            controlViolations({ sheets: [changed], lists: [] }, true, true).length,
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
        const font = Buffer.alloc(48);
        font.write("wOF2");
        font.writeUInt32BE(font.length, 8);
        for (const name of ["plex", "mono", "mono-bold"])
            await writeFile(join(root, `public/fonts/${name}.woff2`), font);
        await writeFile(join(root, "public/fonts/OFL.txt"), OFL_TEXT);
        expect(await exampleControls(root)).toEqual([]);
        for (const source of [
            '<link rel="stylesheet" href="../evil.css">',
            '<LINK REL="StyleSheet" HREF="../evil.css">',
            '<style>@IMPORT "../evil.css";</style>',
        ]) {
            await writeFile(join(sandbox, "evil.css"), ".ec { color: red; }");
            await writeFile(join(root, "index.html"), source);
            expect((await exampleControls(root)).length).toBeGreaterThan(0);
        }
        await writeFile(join(root, "index.html"), "");
        for (const extra of [
            ".ec { color: red; }",
            ":root { --bad: #ff0000; } .ec { color: var(--bad); }",
            '@import "https://example.com/hidden.css";',
            '@IMPORT "https://example.com/hidden.css";',
        ]) {
            await writeFile(imported, controlFixture + extra);
            expect((await exampleControls(root)).length).toBeGreaterThan(0);
        }
        await writeFile(imported, controlFixture);
        for (const file of ["plex.woff2", "OFL.txt"]) {
            const path = join(root, "public/fonts", file);
            const original = await readFile(path);
            await unlink(path);
            expect((await exampleControls(root)).length).toBeGreaterThan(0);
            await writeFile(path, original);
        }
        for (const [file, content] of [
            ["mono-bold.woff2", "not a font"],
            ["mono-bold.woff2", "wOF2"],
            ["mono-bold.woff2", font.subarray(0, 47)],
            ["mono-bold.woff2", Buffer.concat([font, Buffer.alloc(1)])],
            ["OFL.txt", "SIL OPEN FONT LICENSE"],
        ] as const) {
            const path = join(root, "public/fonts", file!);
            const original = await readFile(path);
            await writeFile(path, content!);
            expect((await exampleControls(root)).length).toBeGreaterThan(0);
            await writeFile(path, original);
        }
        const withoutBold = controlFixture.replace(
            /@font-face\s*\{[^}]*font-weight: 700[^}]*\}/g,
            "",
        );
        expect(await fontViolations(root, { sheets: [withoutBold], lists: [] })).toContain(
            "font face: JetBrains Mono 700",
        );
        const withoutTitle = withoutBold.replace(/[^{}]*\.panel-title[^{}]*\{[^}]*\}/g, "");
        expect(await fontViolations(root, { sheets: [withoutTitle], lists: [] })).toEqual([]);
        expect(
            await fontViolations(root, {
                sheets: [withoutTitle.replace(/@font-face\s*\{[^}]*JetBrains Mono[^}]*\}/g, "")],
                lists: [],
            }),
        ).toContain("font face: JetBrains Mono 400");
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

for (const example of await readdir("examples")) {
    test(`${example} follows example-controls`, async () => {
        expect(
            await exampleControls(join("examples", example), example === "embedded-scene"),
        ).toEqual([]);
    });
}
