import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";
import { WEB_SERVER_CONFIG } from "./chromium";

const root = resolve(import.meta.dir, "..");
const configs = [
    ...new Glob("examples/**/playwright.config.ts").scanSync({ cwd: root }),
    ...new Glob("scripts/**/playwright.config.ts").scanSync({ cwd: root }),
].sort();

function walk(node: unknown, visit: (node: Record<string, any>) => void): void {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) {
        for (const child of node) walk(child, visit);
        return;
    }
    const object = node as Record<string, any>;
    if (typeof object.type === "string") visit(object);
    for (const [key, value] of Object.entries(object)) {
        if (key !== "loc" && key !== "start" && key !== "end" && key !== "extra")
            walk(value, visit);
    }
}

function propertyName(node: Record<string, any>): string | undefined {
    if (node.type === "Identifier") return node.name;
    if (node.type === "StringLiteral") return node.value;
}

const violations: string[] = [];
let servers = 0;
if (WEB_SERVER_CONFIG.reuseExistingServer !== false)
    violations.push("scripts/chromium.ts: WEB_SERVER_CONFIG must refuse existing servers");
for (const config of configs) {
    const source = readFileSync(resolve(root, config), "utf8");
    const ast = parse(source, { sourceType: "module", plugins: ["typescript"] });
    const shared = new Set<string>();
    for (const statement of ast.program.body) {
        if (
            statement.type !== "ImportDeclaration" ||
            !/(?:^|\/)chromium(?:\.ts)?$/.test(statement.source.value)
        )
            continue;
        for (const item of statement.specifiers) {
            if (
                item.type === "ImportSpecifier" &&
                (item.imported.type === "Identifier" ? item.imported.name : item.imported.value) ===
                    "WEB_SERVER_CONFIG"
            )
                shared.add(item.local.name);
        }
    }
    const webServers: Record<string, any>[] = [];
    walk(ast.program, (node) => {
        if (node.type === "ObjectProperty" && propertyName(node.key) === "webServer")
            webServers.push(node.value);
    });
    for (const webServer of webServers) {
        servers++;
        if (webServer.type !== "ObjectExpression") {
            violations.push(
                `${config}: webServer must be a literal object with reuseExistingServer: false`,
            );
            continue;
        }
        const reuse = webServer.properties.filter(
            (property: Record<string, any>) =>
                property.type === "ObjectProperty" &&
                propertyName(property.key) === "reuseExistingServer",
        );
        if (
            reuse.length === 0 &&
            webServer.properties.some(
                (property: Record<string, any>) =>
                    property.type === "SpreadElement" && shared.has(property.argument.name),
            )
        )
            continue;
        if (
            reuse.length !== 1 ||
            reuse[0].value.type !== "BooleanLiteral" ||
            reuse[0].value.value !== false
        ) {
            violations.push(
                `${config}: webServer must inherit WEB_SERVER_CONFIG or refuse existing servers`,
            );
        }
    }
}

if (violations.length) {
    console.error(`✗ browser server ownership:\n${violations.join("\n")}`);
    process.exit(1);
}
console.log(`✓ browser server ownership: ${servers} webServer configs refuse existing servers`);
