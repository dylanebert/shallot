import { parse } from "@babel/parser";
import { gpuAllocationExpressions } from "../../src/engine/runtime/allocation-site";
import type { AllocationSite } from "./allocation";

const parserMetadata = new Set([
    "start",
    "end",
    "loc",
    "extra",
    "comments",
    "leadingComments",
    "trailingComments",
    "innerComments",
    "tokens",
    "errors",
]);
const admittedAllocationWrappers = new Map<string, { functionName: string }>(
    gpuAllocationExpressions.map((entry) => [entry.wrapperSite, { functionName: entry.fn.name }]),
);

function syntaxShape(node: any): unknown {
    if (Array.isArray(node)) return node.map(syntaxShape);
    if (!node || typeof node !== "object") return node;
    return Object.fromEntries(
        Object.entries(node)
            .filter(([key]) => ![
                "start",
                "end",
                "loc",
                "extra",
                "comments",
                "leadingComments",
                "trailingComments",
                "innerComments",
            ].includes(key))
            .map(([key, value]) => [key, syntaxShape(value)]),
    );
}

function mentionsArguments(node: any): boolean {
    if (Array.isArray(node)) return node.some(mentionsArguments);
    if (!node || typeof node !== "object") return false;
    if (node.type === "Identifier" && node.name === "arguments") return true;
    return Object.entries(node)
        .filter(([key]) => !parserMetadata.has(key))
        .some(([, value]) => mentionsArguments(value));
}

export function allocationWrapperFailure(
    sourceText: string,
    entry: (typeof gpuAllocationExpressions)[number],
): string | undefined {
    const functionName = entry.fn.name;
    let source;
    try {
        source = parse(sourceText, { sourceType: "module", plugins: ["typescript", "decorators-legacy"] });
    } catch {
        return `allocation exception wrapper ${functionName} source does not parse`;
    }
    const declarations: any[] = [];
    const visit = (node: any) => {
        if (!node || typeof node !== "object") return;
        if (Array.isArray(node)) {
            for (const child of node) visit(child);
            return;
        }
        if (node.type === "FunctionDeclaration" && node.id?.name === functionName) declarations.push(node);
        for (const [key, value] of Object.entries(node))
            if (!parserMetadata.has(key)) visit(value);
    };
    visit(source.program);
    if (declarations.length !== 1)
        return `allocation exception wrapper ${functionName} is missing or ambiguous`;
    const declaration = declarations[0]!;
    if (!declaration.id || !declaration.body)
        return `allocation exception wrapper ${functionName} has no body`;
    const namePosition = declaration.id.loc.end;
    const wrapperSite = `src/engine/runtime/allocation-site.ts:${namePosition.line}:${namePosition.column + 1}`;
    if (wrapperSite !== entry.wrapperSite)
        return `allocation exception wrapper ${functionName} moved from ${entry.wrapperSite} to ${wrapperSite}`;
    if (declaration.decorators?.length || declaration.params.some((parameter: any) => parameter.decorators?.length))
        return `allocation exception wrapper ${functionName} must not have decorators`;
    if (mentionsArguments(declaration.body))
        return `allocation exception wrapper ${functionName} must not use arguments`;
    let expectedDeclaration: any;
    try {
        const returnType = entry.returnType === undefined ? "" : `: ${entry.returnType}`;
        expectedDeclaration = parse(
            `function ${functionName}(${entry.parameters})${returnType} { ${entry.returns ? "return " : ""}${entry.expression}; }`,
            { sourceType: "module", plugins: ["typescript", "decorators-legacy"] },
        ).program.body[0];
    } catch {
        return `allocation exception wrapper ${functionName} has an invalid expected declaration`;
    }
    if (JSON.stringify(syntaxShape(declaration.params)) !== JSON.stringify(syntaxShape(expectedDeclaration?.params)))
        return `allocation exception wrapper ${functionName} must have exactly parameters (${entry.parameters})`;
    if (declaration.body.body.length !== 1)
        return `allocation exception wrapper ${functionName} must contain only ${entry.returns ? "return " : ""}${entry.expression}`;
    const statement = declaration.body.body[0];
    const actual = entry.returns
        ? statement.type === "ReturnStatement" ? statement.argument : undefined
        : statement.type === "ExpressionStatement" ? statement.expression : undefined;
    if (!actual || actual.type !== "CallExpression")
        return `allocation exception wrapper ${functionName} must contain only ${entry.returns ? "return " : ""}${entry.expression}`;
    let expected: any;
    try {
        expected = parse(entry.expression, { sourceType: "module", plugins: ["typescript", "decorators-legacy"] }).program.body[0];
    } catch {
        return `allocation exception wrapper ${functionName} has invalid expected expression`;
    }
    if (
        expected?.type !== "ExpressionStatement" ||
        expected.expression.type !== "CallExpression" ||
        JSON.stringify(syntaxShape(actual)) !== JSON.stringify(syntaxShape(expected.expression))
    )
        return `allocation exception wrapper ${functionName} must contain only ${entry.returns ? "return " : ""}${entry.expression}`;
    if (JSON.stringify(syntaxShape(declaration)) !== JSON.stringify(syntaxShape(expectedDeclaration)))
        return `allocation exception wrapper ${functionName} must be a plain declaration containing only ${entry.returns ? "return " : ""}${entry.expression}`;
    return undefined;
}

export function assertAllocationWrappers(sourceText: string): void {
    const failures = gpuAllocationExpressions
        .map((entry) => allocationWrapperFailure(sourceText, entry))
        .filter((failure): failure is string => failure !== undefined);
    if (failures.length > 0) throw new Error(failures.join("\n"));
}

export function isAdmittedAllocationSite(
    row: Pick<AllocationSite, "boundary" | "stack">,
): boolean {
    if (!row.boundary || !row.stack) return false;
    const expected = admittedAllocationWrappers.get(row.boundary);
    if (!expected) return false;
    let topJavaScript = -1;
    for (let i = row.stack.length - 1; i >= 0; i--) {
        if (!row.stack[i]!.native) {
            topJavaScript = i;
            break;
        }
    }
    const top = row.stack[topJavaScript];
    return (
        topJavaScript >= 0 &&
        top !== undefined &&
        top.functionName === expected.functionName &&
        top.site === `${expected.functionName} ${row.boundary}` &&
        top.boundary === row.boundary &&
        row.stack.slice(topJavaScript + 1).every((frame) => frame.native)
    );
}
