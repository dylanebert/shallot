import { resolve } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";

/** Categorical backstops reject hangs and runaway work, not slow timings. */
export function timeoutErrors(path: string, text: string): string[] {
    if (path.endsWith(".oracle.ts")) return [];
    const browser = path.endsWith("playwright.config.ts");
    const ceiling = browser
        ? 60_000
        : path.endsWith(".gpu.test.ts")
          ? 1000
          : path.endsWith(".node.ts")
            ? 20_000
            : 250;
    const file = parse(text, { sourceType: "module", plugins: ["typescript"] });
    const errors: string[] = [];
    const tests = new Set<string>();
    const setters = new Set<string>();
    const constants = new Map<string, any>();
    let defaultSet = false;
    let browserDefaults = 0;
    for (const statement of file.program.body) {
        if (statement.type === "ImportDeclaration" && statement.source.value === "bun:test") {
            for (const item of statement.specifiers) {
                if (item.type !== "ImportSpecifier") continue;
                const name =
                    item.imported.type === "Identifier" ? item.imported.name : item.imported.value;
                if (name === "test" || name === "it") tests.add(item.local.name);
                if (name === "setDefaultTimeout") setters.add(item.local.name);
            }
        }
        if (statement.type === "VariableDeclaration" && statement.kind === "const")
            for (const declaration of statement.declarations) {
                if (declaration.id.type === "Identifier" && declaration.init)
                    constants.set(declaration.id.name, declaration.init);
            }
    }
    function number(expression: any, seen = new Set<string>()): number | undefined {
        if (expression?.type === "NumericLiteral") return expression.value;
        if (expression?.type === "Identifier" && !seen.has(expression.name)) {
            seen.add(expression.name);
            return number(constants.get(expression.name), seen);
        }
        return undefined;
    }
    function fail(node: any, detail: string): void {
        errors.push(`${path}:${node.loc.start.line}: ${detail}`);
    }
    function root(expression: any): string | undefined {
        if (expression?.type === "Identifier") return expression.name;
        if (expression?.type === "MemberExpression") return root(expression.object);
        if (expression?.type === "CallExpression") return root(expression.callee);
        return undefined;
    }
    function visit(node: any): void {
        if (!node || typeof node !== "object") return;
        if (Array.isArray(node)) {
            for (const child of node) visit(child);
            return;
        }
        if (
            browser &&
            node.type === "ObjectProperty" &&
            (node.key.name ?? node.key.value) === "globalTimeout"
        ) {
            browserDefaults++;
            if (number(node.value) !== ceiling)
                fail(node, "browser globalTimeout must be the categorical 60000 ms backstop");
        }
        if (node.type === "CallExpression") {
            const name = root(node.callee);
            if (name && setters.has(name)) {
                const value = number(node.arguments[0]);
                if (value === undefined || value <= 0 || value > ceiling)
                    fail(node, `default timeout must be positive and at most ${ceiling} ms`);
            }
            if (name && tests.has(name) && node.arguments.length >= 3) {
                let timeout = node.arguments[2];
                if (timeout.type === "ObjectExpression") {
                    const property = timeout.properties.find(
                        (item: any) =>
                            item.type === "ObjectProperty" &&
                            (item.key.name ?? item.key.value) === "timeout",
                    );
                    timeout = property?.value;
                }
                if (timeout) {
                    const value = number(timeout);
                    if (value === undefined || value <= 0 || value > ceiling)
                        fail(
                            timeout,
                            `test timeout must be a positive constant at most ${ceiling} ms`,
                        );
                }
            }
        }
        for (const [key, child] of Object.entries(node))
            if (key !== "loc" && key !== "extra") visit(child);
    }
    visit(file.program);
    for (const statement of file.program.body) {
        if (
            statement.type === "ExpressionStatement" &&
            statement.expression.type === "CallExpression"
        ) {
            const call = statement.expression;
            if (setters.has(root(call.callee) ?? "") && number(call.arguments[0]) === ceiling)
                defaultSet = true;
        }
    }
    if (browser) {
        if (browserDefaults !== 1)
            errors.push(`${path}: exactly one globalTimeout: 60000 is required for this subject`);
    } else if (ceiling !== 250 && !defaultSet)
        errors.push(`${path}: setDefaultTimeout(${ceiling}) is required for this tier`);
    return errors;
}

if (import.meta.main) {
    const root = resolve(import.meta.dir, "..");
    const errors: string[] = [];
    for (const directory of ["src", "examples", "scripts", "diagnostics"]) {
        for (const path of new Glob("**/{*.test.ts,*.node.ts,playwright.config.ts}").scanSync(
            resolve(root, directory),
        )) {
            errors.push(
                ...timeoutErrors(
                    `${directory}/${path}`,
                    await Bun.file(resolve(root, directory, path)).text(),
                ),
            );
        }
    }
    if (errors.length) {
        console.error(errors.join("\n"));
        process.exit(1);
    }
}
