const INDENT = "    ";
const MAX_LINE = 100;

export interface Node {
    id?: string;
    attrs: Attr[];
    children: Node[];
    comments?: string[];
    blankBefore?: boolean;
}

export interface Attr {
    name: string;
    value: string;
}

export interface ParseError {
    message: string;
    path?: string;
}

interface Token {
    type: "comment" | "open" | "close" | "blank";
    value: string;
    selfClosing?: boolean;
    attrs?: Record<string, string>;
    tagName?: string;
}

function tokenize(xml: string): Token[] {
    const tokens: Token[] = [];
    const regex = /<!--[\s\S]*?-->|<\/?\s*(\w+)[^>]*\/?>/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(xml)) !== null) {
        const before = xml.slice(lastIndex, match.index);
        if (/\n\s*\n/.test(before)) {
            tokens.push({ type: "blank", value: "" });
        }
        lastIndex = match.index + match[0].length;

        const tag = match[0];

        if (tag.startsWith("<!--")) {
            const content = tag.slice(4, -3).trim();
            tokens.push({ type: "comment", value: content });
        } else if (tag.startsWith("</")) {
            const tagName = tag.match(/<\/\s*(\w+)/)?.[1] ?? "";
            tokens.push({ type: "close", value: tag, tagName });
        } else {
            const selfClosing = tag.endsWith("/>");
            const tagMatch = tag.match(/<\s*(\w+)/);
            const tagName = tagMatch?.[1] ?? "";
            const attrs = parseTagAttrs(tag);
            tokens.push({ type: "open", value: tag, selfClosing, tagName, attrs });
        }
    }

    return tokens;
}

function parseTagAttrs(tag: string): Record<string, string> {
    const attrs: Record<string, string> = {};
    const attrRegex = /([^\s=<>/]+)(?:\s*=\s*"([^"]*)")?/g;
    const inner = tag.replace(/^<\s*\w+/, "").replace(/\/?>$/, "");
    let match: RegExpExecArray | null;

    while ((match = attrRegex.exec(inner)) !== null) {
        const name = match[1];
        const value = match[2] ?? "";
        attrs[name] = value;
    }

    return attrs;
}

export function parse(xml: string): Node[] {
    const unclosedMatch = xml.match(/<[^>]*$/);
    if (unclosedMatch) {
        throw new Error(`xml parse error: Unclosed tag at end of document`);
    }

    const tokens = tokenize(xml);

    for (const token of tokens) {
        if (token.type === "open" && token.tagName !== "scene" && token.tagName !== "a") {
            const tagName = token.tagName ?? "unknown";
            if (tagName.toLowerCase() === "a" || tagName.toLowerCase() === "scene") {
                continue;
            }
            throw new Error(`xml parse error: Unknown tag <${tagName}>`);
        }
    }

    const nodes: Node[] = [];
    const errors: ParseError[] = [];

    let i = 0;
    let pendingComments: string[] = [];
    let pendingBlank = false;

    while (i < tokens.length) {
        const token = tokens[i];

        if (token.type === "blank") {
            pendingBlank = true;
            i++;
            continue;
        }

        if (token.type === "comment") {
            pendingComments.push(token.value);
            i++;
            continue;
        }

        if (token.type === "open" && token.tagName === "scene") {
            pendingComments = [];
            pendingBlank = false;
            i++;
            continue;
        }

        if (token.type === "close" && token.tagName === "scene") {
            i++;
            continue;
        }

        if (token.type === "open" && token.tagName === "a") {
            const result = parseNodeFromTokens(tokens, i, errors);
            if (result.node) {
                result.node.comments = pendingComments.length > 0 ? pendingComments : undefined;
                result.node.blankBefore = pendingBlank || undefined;
                nodes.push(result.node);
            }
            pendingComments = [];
            pendingBlank = false;
            i = result.nextIndex;
            continue;
        }

        if (token.type === "open" && token.tagName?.toLowerCase() === "scene") {
            throw new Error(`Invalid tag "${token.tagName}". Use lowercase <scene>`);
        }

        if (
            token.type === "open" &&
            token.tagName?.toLowerCase() === "a" &&
            token.tagName !== "a"
        ) {
            throw new Error(`Invalid tag "${token.tagName}". Use lowercase <a>`);
        }

        i++;
    }

    if (errors.length > 0) {
        throw new Error(errors.map((e) => e.message).join("\n"));
    }

    return nodes;
}

function parseNodeFromTokens(
    tokens: Token[],
    startIndex: number,
    errors: ParseError[],
): { node: Node | null; nextIndex: number } {
    const token = tokens[startIndex];

    if (token.type !== "open" || token.tagName !== "a") {
        if (token.tagName?.toLowerCase() === "a") {
            errors.push({ message: `Invalid tag "${token.tagName}". Use lowercase <a>` });
        }
        return { node: null, nextIndex: startIndex + 1 };
    }

    const rawAttrs = token.attrs ?? {};
    const attrs: Attr[] = [];
    let nodeId: string | undefined;

    for (const [attrName, attrValue] of Object.entries(rawAttrs)) {
        if (attrName === "id") {
            nodeId = attrValue;
        } else {
            attrs.push({ name: attrName, value: attrValue });
        }
    }

    const children: Node[] = [];
    let i = startIndex + 1;

    if (!token.selfClosing) {
        let pendingComments: string[] = [];
        let pendingBlank = false;

        while (i < tokens.length) {
            const childToken = tokens[i];

            if (childToken.type === "blank") {
                pendingBlank = true;
                i++;
                continue;
            }

            if (childToken.type === "comment") {
                pendingComments.push(childToken.value);
                i++;
                continue;
            }

            if (childToken.type === "close" && childToken.tagName === "a") {
                i++;
                break;
            }

            if (childToken.type === "open" && childToken.tagName === "a") {
                const result = parseNodeFromTokens(tokens, i, errors);
                if (result.node) {
                    result.node.comments = pendingComments.length > 0 ? pendingComments : undefined;
                    result.node.blankBefore = pendingBlank || undefined;
                    children.push(result.node);
                }
                pendingComments = [];
                pendingBlank = false;
                i = result.nextIndex;
                continue;
            }

            i++;
        }
    }

    return {
        node: { id: nodeId, attrs, children },
        nextIndex: i,
    };
}

export function findParent(
    target: Node,
    nodes: Node[],
    parent: Node | null = null,
): Node | null | undefined {
    for (const node of nodes) {
        if (node === target) return parent;
        const found = findParent(target, node.children, node);
        if (found !== undefined) return found;
    }
    return undefined;
}

export function findNodeById(id: string, nodes: Node[]): Node | undefined {
    for (const node of nodes) {
        if (node.id === id) return node;
        const found = findNodeById(id, node.children);
        if (found) return found;
    }
}

export function serialize(nodes: Node[]): string {
    const lines: string[] = ["<scene>"];
    let isFirst = true;

    for (const node of nodes) {
        serializeNode(node, lines, 1, isFirst);
        isFirst = false;
    }

    lines.push("</scene>");
    return lines.join("\n");
}

function serializeNode(node: Node, lines: string[], depth: number, isFirst: boolean): void {
    const indent = INDENT.repeat(depth);

    if (node.blankBefore && !isFirst) {
        lines.push("");
    }

    if (node.comments) {
        for (const comment of node.comments) {
            lines.push(`${indent}<!-- ${comment} -->`);
        }
    }

    const attrParts = buildAttrParts(node);
    const singleLine = `${indent}<a${attrParts.join("")}${node.children.length === 0 ? " />" : ">"}`;

    if (singleLine.length <= MAX_LINE) {
        lines.push(singleLine);
    } else {
        lines.push(`${indent}<a`);
        const attrIndent = INDENT.repeat(depth + 1);
        for (const part of attrParts) {
            lines.push(`${attrIndent}${part.trim()}`);
        }
        lines.push(`${indent}${node.children.length === 0 ? "/>" : ">"}`);
    }

    if (node.children.length > 0) {
        let childIsFirst = true;
        for (const child of node.children) {
            serializeNode(child, lines, depth + 1, childIsFirst);
            childIsFirst = false;
        }
        lines.push(`${indent}</a>`);
    }
}

function buildAttrParts(node: Node): string[] {
    const parts: string[] = [];

    if (node.id) {
        parts.push(` id="${escapeAttr(node.id)}"`);
    }

    for (const attr of node.attrs) {
        if (attr.value === "") {
            parts.push(` ${attr.name}`);
        } else {
            parts.push(` ${attr.name}="${escapeAttr(attr.value)}"`);
        }
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
