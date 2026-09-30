import { resolve } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";
import { CEILING } from "./test-tiers";

/** Check declarations, not elapsed time. Delay timers inside tests are not tier declarations. */
export function timeoutErrors(path: string, text: string): string[] {
    if (path.endsWith(".oracle.ts")) return [];
    const browser = path.endsWith("playwright.config.ts");
    const browserTest = path.endsWith(".e2e.ts");
    const sharedBrowser = path === "scripts/chromium.ts";
    const tier = path.endsWith(".gpu.test.ts")
        ? "gpu"
        : path.endsWith(".node.ts")
          ? "node"
          : "cheap";
    const ceiling = CEILING[tier];
    const file = parse(text, { sourceType: "module", plugins: ["typescript"] });
    const errors: string[] = [];
    const tests = new Set<string>();
    const setters = new Set<string>();
    const hooks = new Set<string>();
    const tiers = new Set<string>();
    const browserConfigs = new Set<string>();
    const serverConfigs = new Set<string>();
    const constants = new Map<string, any>();
    let defaults = 0;
    let correctDefault = false;
    let browserDefaults = 0;
    let serverDefaults = 0;
    for (const statement of file.program.body) {
        if (statement.type === "ImportDeclaration") {
            for (const item of statement.specifiers) {
                if (item.type !== "ImportSpecifier") continue;
                const name =
                    item.imported.type === "Identifier" ? item.imported.name : item.imported.value;
                const source = statement.source.value;
                if (/(?:^|\/)test-tiers(?:\.ts)?$/.test(source) && name === "CEILING")
                    tiers.add(item.local.name);
                if (/(?:^|\/)chromium(?:\.ts)?$/.test(source)) {
                    if (name === "BROWSER_CONFIG") browserConfigs.add(item.local.name);
                    if (name === "WEB_SERVER_CONFIG") serverConfigs.add(item.local.name);
                }
                if (["bun:test", "playwright/test", "@playwright/test"].includes(source)) {
                    if (["test", "it"].includes(name)) tests.add(item.local.name);
                    if (name === "setDefaultTimeout") setters.add(item.local.name);
                    if (["beforeAll", "beforeEach", "afterAll", "afterEach"].includes(name))
                        hooks.add(item.local.name);
                }
            }
        }
        if (statement.type === "VariableDeclaration" && statement.kind === "const") {
            for (const declaration of statement.declarations) {
                if (declaration.id.type === "Identifier" && declaration.init)
                    constants.set(declaration.id.name, declaration.init);
            }
        }
    }
    function budget(expression: any, seen = new Set<string>()): number | undefined {
        if (expression?.type === "MemberExpression" && tiers.has(expression.object.name)) {
            const key = expression.computed ? expression.property.value : expression.property.name;
            if (Object.hasOwn(CEILING, key)) return CEILING[key as keyof typeof CEILING];
        }
        if (expression?.type === "Identifier" && !seen.has(expression.name)) {
            seen.add(expression.name);
            return budget(constants.get(expression.name), seen);
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
    function checkBudget(node: any, maximum: number, repeated?: number): void {
        const value = budget(node);
        if (value === undefined)
            fail(
                node,
                "timeout must derive from CEILING in scripts/test-tiers.ts; literal timeout numbers are forbidden",
            );
        else if (value === repeated) fail(node, "test timeout repeats its file default; omit it");
        else if (value > maximum) fail(node, "timeout exceeds its tier backstop");
    }
    function visit(node: any): void {
        if (!node || typeof node !== "object") return;
        if (Array.isArray(node)) {
            for (const child of node) visit(child);
            return;
        }
        if (browser && node.type === "SpreadElement" && browserConfigs.has(node.argument.name))
            browserDefaults++;
        if (node.type === "ObjectProperty") {
            const key = node.key.name ?? node.key.value;
            if (!browser && !sharedBrowser && ["timeout", "globalTimeout"].includes(key))
                checkBudget(node.value, CEILING.startup);
            if (sharedBrowser && ["globalTimeout", "timeout"].includes(key)) {
                const expected = key === "globalTimeout" ? CEILING.browser : CEILING.startup;
                if (budget(node.value) !== expected)
                    fail(node, `shared ${key} must derive from CEILING`);
                if (key === "globalTimeout") browserDefaults++;
                else serverDefaults++;
            }
            if (browser) {
                if (
                    [
                        "globalTimeout",
                        "timeout",
                        "workers",
                        "fullyParallel",
                        "reporter",
                        "testMatch",
                        "reuseExistingServer",
                    ].includes(key)
                )
                    fail(node, `subject ${key} must come from scripts/chromium.ts`);
                if (
                    key === "webServer" &&
                    !node.value.properties?.some(
                        (property: any) =>
                            property.type === "SpreadElement" &&
                            serverConfigs.has(property.argument.name),
                    )
                )
                    fail(node, "webServer must spread WEB_SERVER_CONFIG from scripts/chromium.ts");
            }
        }
        if (node.type === "CallExpression") {
            const name = root(node.callee);
            if (name && setters.has(name)) {
                defaults++;
                checkBudget(node.arguments[0], ceiling);
                if (budget(node.arguments[0]) === ceiling) correctDefault = true;
            }
            if (name && tests.has(name)) {
                if (
                    browserTest &&
                    node.callee.type === "MemberExpression" &&
                    node.callee.property.name === "setTimeout"
                )
                    fail(node, "browser tests use Playwright's default test timeout");
                let timeout = node.arguments[2];
                if (timeout?.type === "ObjectExpression")
                    timeout = timeout.properties.find(
                        (item: any) =>
                            item.type === "ObjectProperty" &&
                            (item.key.name ?? item.key.value) === "timeout",
                    )?.value;
                if (
                    timeout &&
                    !["ArrowFunctionExpression", "FunctionExpression"].includes(timeout.type)
                ) {
                    if (browserTest)
                        fail(timeout, "browser tests use Playwright's default test timeout");
                    else checkBudget(timeout, ceiling, ceiling);
                }
            }
            if (name && hooks.has(name) && node.arguments[1])
                checkBudget(node.arguments[1], CEILING.startup);
        }
        for (const [key, child] of Object.entries(node))
            if (key !== "loc" && key !== "extra") visit(child);
    }
    visit(file.program);
    if (browser && browserDefaults !== 1)
        errors.push(`${path}: spread BROWSER_CONFIG from scripts/chromium.ts exactly once`);
    if (sharedBrowser && (browserDefaults !== 1 || serverDefaults !== 1))
        errors.push(`${path}: define the shared browser and startup backstops exactly once`);
    if (
        !browser &&
        !browserTest &&
        !sharedBrowser &&
        tier !== "cheap" &&
        (defaults !== 1 || !correctDefault)
    )
        errors.push(`${path}: one setDefaultTimeout(CEILING.${tier}) header is required`);
    return [...new Set(errors)];
}

export function packageTimeoutErrors(text: string): string[] {
    const command = JSON.parse(text).scripts?.test ?? "";
    const values = [...command.matchAll(/--timeout(?:=|\s+)(\d+)/g)].map((match) =>
        Number(match[1]),
    );
    return values.length === 1 && values[0] === CEILING.cheap
        ? []
        : ["package.json: test --timeout must equal CEILING.cheap"];
}

if (import.meta.main) {
    const root = resolve(import.meta.dir, "..");
    const errors = packageTimeoutErrors(await Bun.file(resolve(root, "package.json")).text());
    for (const directory of ["src", "examples", "scripts", "diagnostics"]) {
        for (const path of new Glob(
            "**/{*.test.ts,*.node.ts,*.e2e.ts,playwright.config.ts}",
        ).scanSync(resolve(root, directory))) {
            errors.push(
                ...timeoutErrors(
                    `${directory}/${path}`,
                    await Bun.file(resolve(root, directory, path)).text(),
                ),
            );
        }
    }
    errors.push(
        ...timeoutErrors(
            "scripts/chromium.ts",
            await Bun.file(resolve(root, "scripts/chromium.ts")).text(),
        ),
    );
    if (errors.length) {
        console.error(errors.join("\n"));
        process.exit(1);
    }
}
