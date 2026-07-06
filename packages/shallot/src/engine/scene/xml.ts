const INDENT = "    ";
const MAX_LINE = 100;

/** one parsed scene entity: an optional `id`, its component attributes, and a (always-empty) children list. */
export interface Node {
    id?: string;
    attrs: Attr[];
    /** always empty — kept for AST shape stability after flat-scene migration. */
    children: Node[];
    comments?: string[];
    blankBefore?: boolean;
}

/** one component attribute on a node: its kebab-case name and raw string value (`""` for a bare component). */
export interface Attr {
    name: string;
    value: string;
}

/** a scene parse or load failure, carrying a human-readable message. */
export interface ParseError {
    message: string;
}

const TAG_RE = /<!--([\s\S]*?)-->|<\s*(\/?)\s*(\w+)([^>]*)>/g;
const ATTR_RE = /([^\s=<>/]+)(?:\s*=\s*"([^"]*)")?/g;

/**
 * parses scene XML into a flat node tree, one `Node` per `<a>` element. Throws on malformed markup, an
 * unknown tag, or a nested `<a>` — scenes are flat, so cross-entity links use `@name` field refs.
 *
 * @example
 * const nodes = parse('<scene><a id="cam" camera orbit /></scene>');
 */
export function parse(xml: string): Node[] {
    if (/<[^>]*$/.test(xml)) {
        throw new Error("xml parse error: Unclosed tag at end of document");
    }

    const nodes: Node[] = [];
    let comments: string[] = [];
    let blank = false;
    let inEntity = false;
    let cursor = 0;

    TAG_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = TAG_RE.exec(xml)) !== null) {
        if (/\n\s*\n/.test(xml.slice(cursor, m.index))) blank = true;
        cursor = m.index + m[0].length;

        const [, comment, slash, name, body = ""] = m;

        if (comment !== undefined) {
            comments.push(comment.trim());
            continue;
        }

        const lower = name.toLowerCase();
        if (lower !== "scene" && lower !== "a") {
            throw new Error(`xml parse error: Unknown tag <${name}>`);
        }
        if (name !== lower) {
            throw new Error(`Invalid tag "${name}". Use lowercase <${lower}>`);
        }

        if (slash) {
            if (lower === "a") inEntity = false;
            continue;
        }

        if (lower === "scene") {
            comments = [];
            blank = false;
            continue;
        }

        if (inEntity) {
            throw new Error(
                "xml parse error: nested <a> elements are not supported. Scenes are flat; use @name field references for cross-entity links.",
            );
        }

        const node = parseEntity(body);
        if (comments.length > 0) node.comments = comments;
        if (blank) node.blankBefore = true;
        nodes.push(node);
        comments = [];
        blank = false;
        if (!body.trimEnd().endsWith("/")) inEntity = true;
    }

    return nodes;
}

function parseEntity(body: string): Node {
    const attrs: Attr[] = [];
    let id: string | undefined;
    ATTR_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = ATTR_RE.exec(body)) !== null) {
        const [, name, value = ""] = m;
        if (name === "id") id = value;
        else attrs.push({ name, value });
    }
    return { id, attrs, children: [] };
}

/** finds a node by its scene `id` in a parsed tree, or `undefined` if none matches. */
export function findNodeById(id: string, nodes: Node[]): Node | undefined {
    for (const node of nodes) {
        if (node.id === id) return node;
    }
}

/**
 * locates a node's parent in the tree. Scenes are flat, so a present node returns the passed `parent`
 * (`null` at the top level); a node absent from the tree returns `undefined`.
 */
export function findParent(
    target: Node,
    nodes: Node[],
    parent: Node | null = null,
): Node | null | undefined {
    for (const node of nodes) {
        if (node === target) return parent;
    }
    return undefined;
}

/**
 * renders a node tree back to formatted scene XML, the inverse of `parse`. Long entities wrap one
 * attribute per line; a `stringify(serialize(state))` round-trips a live scene to disk.
 *
 * @example
 * const xml = stringify(serialize(state));
 */
export function stringify(nodes: Node[]): string {
    const lines: string[] = ["<scene>"];
    for (let i = 0; i < nodes.length; i++) writeNode(nodes[i], lines, 1, i === 0);
    lines.push("</scene>");
    return lines.join("\n");
}

function writeNode(node: Node, lines: string[], depth: number, isFirst: boolean): void {
    const indent = INDENT.repeat(depth);

    if (node.blankBefore && !isFirst) lines.push("");
    if (node.comments) {
        for (const comment of node.comments) lines.push(`${indent}<!-- ${comment} -->`);
    }

    const parts = attrParts(node);
    const single = `${indent}<a${parts.join("")} />`;
    if (single.length <= MAX_LINE) {
        lines.push(single);
        return;
    }

    lines.push(`${indent}<a`);
    const attrIndent = INDENT.repeat(depth + 1);
    for (const part of parts) lines.push(`${attrIndent}${part.trim()}`);
    lines.push(`${indent}/>`);
}

function attrParts(node: Node): string[] {
    const parts: string[] = [];
    if (node.id) parts.push(` id="${escapeAttr(node.id)}"`);
    for (const { name, value } of node.attrs) {
        parts.push(value === "" ? ` ${name}` : ` ${name}="${escapeAttr(value)}"`);
    }
    return parts;
}

function escapeAttr(str: string): string {
    return str
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}
