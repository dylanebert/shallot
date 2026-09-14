import { readFileSync } from "node:fs";
import { relative, resolve } from "node:path";
import { parse } from "@babel/parser";
import { Glob } from "bun";
import {
    deviceTierContext,
    renderPackedContext,
    renderTierTable,
    replaceGeneratedTable,
} from "./generate/device-tiers";

interface PluginDeclaration {
    readonly name: string;
    readonly device: "required" | "optional" | undefined;
}

const DEVICE_READ = /\bCompute\.(?:device|root|buffers)\b/;
const PLUGIN_DECLARATION = /Plugin$/;

function propertyName(node: any): string | undefined {
    if (node?.computed) return undefined;
    return node?.key?.name ?? node?.key?.value;
}

function walk(node: unknown, visit: (node: any) => void): void {
    if (node === null || typeof node !== "object") return;
    visit(node);
    if (Array.isArray(node)) {
        for (const child of node) walk(child, visit);
        return;
    }
    for (const [key, value] of Object.entries(node)) {
        if (key === "loc" || key === "start" || key === "end") continue;
        walk(value, visit);
    }
}

function deviceValue(object: any): "required" | "optional" | undefined {
    if (object?.type !== "ObjectExpression") return undefined;
    const value = object.properties
        .filter((property: any) => property.type === "ObjectProperty")
        .find((property: any) => propertyName(property) === "device")?.value;
    return value?.type === "StringLiteral" &&
        (value.value === "required" || value.value === "optional")
        ? value.value
        : undefined;
}

function declarations(source: string, file: string): PluginDeclaration[] {
    let ast: any;
    try {
        ast = parse(source, { sourceType: "module", plugins: ["typescript"] });
    } catch (error) {
        throw new Error(`${file}: cannot parse: ${(error as Error).message}`);
    }
    const factories = new Map<string, any>();
    walk(ast.program, (node) => {
        if (node.type === "FunctionDeclaration" && typeof node.id?.name === "string")
            factories.set(node.id.name, node);
    });
    const found: PluginDeclaration[] = [];
    walk(ast.program, (node) => {
        if (node.type !== "VariableDeclarator") return;
        const name = node.id?.name;
        if (typeof name !== "string" || !PLUGIN_DECLARATION.test(name)) return;
        let value = deviceValue(node.init);
        if (value === undefined && node.init?.type === "CallExpression") {
            const factory = factories.get(node.init.callee?.name);
            let returned: any;
            walk(factory, (candidate) => {
                if (candidate.type === "ReturnStatement") returned ??= candidate.argument;
            });
            value = deviceValue(returned);
        }
        found.push({ name, device: value });
    });
    return found;
}

function moduleRoot(root: string, file: string): string | undefined {
    const relativeFile = relative(root, file).split("\\").join("/");
    const match = /^(src\/(?:standard|extras)\/[^/]+)/.exec(relativeFile);
    return match ? resolve(root, match[1]) : undefined;
}

/** Lexical structure gate: Compute reads must sit inside a module with a declared plugin tier. */
export function readDeviceTierViolations(root: string): string[] {
    const violations: string[] = [];
    const modules = new Map<string, PluginDeclaration[]>();
    const files = [
        ...new Glob("src/standard/**/*.ts").scanSync(root),
        ...new Glob("src/extras/**/*.ts").scanSync(root),
    ]
        .map((file) => resolve(root, file))
        .sort();
    for (const file of files) {
        const source = readFileSync(file, "utf8");
        if (!DEVICE_READ.test(source)) continue;
        const owner = moduleRoot(root, file);
        if (owner === undefined) continue;
        let plugins = modules.get(owner);
        if (plugins === undefined) {
            plugins = [];
            for (const candidate of new Glob("**/*.ts").scanSync(owner)) {
                if (candidate.endsWith(".d.ts")) continue;
                const candidatePath = resolve(owner, candidate);
                const source = readFileSync(candidatePath, "utf8");
                if (!source.includes("Plugin")) continue;
                plugins.push(...declarations(source, relative(root, candidatePath)));
            }
            modules.set(owner, plugins);
        }
        const declared = plugins.filter((plugin) => plugin.device !== undefined);
        if (declared.length === 0) {
            violations.push(
                `${relative(root, file)}: references Compute.device/root/buffers but its plugin has no device declaration`,
            );
        }
    }
    return violations;
}

export async function checkDeviceTiers(root: string): Promise<string[]> {
    const violations = readDeviceTierViolations(root);
    const context = await deviceTierContext();
    const expectedTable = renderTierTable(context);
    const contributing = readFileSync(resolve(root, "CONTRIBUTING.md"), "utf8");
    if (replaceGeneratedTable(contributing, expectedTable, "CONTRIBUTING.md") !== contributing)
        violations.push("CONTRIBUTING.md: generated device tier table is drifted");

    const packedPath = resolve(root, "src/engine/app/device-tiers.generated.ts");
    if (readFileSync(packedPath, "utf8") !== renderPackedContext(context))
        violations.push("src/engine/app/device-tiers.generated.ts: packed context is drifted");
    return violations;
}

if (import.meta.main) {
    const args = Bun.argv.slice(2);
    const rootIndex = args.indexOf("--root");
    const root = resolve(rootIndex === -1 ? process.cwd() : (args[rootIndex + 1] ?? process.cwd()));
    const violations = args.includes("--declarations-only")
        ? readDeviceTierViolations(root)
        : await checkDeviceTiers(root);
    for (const violation of violations) console.error(violation);
    if (violations.length > 0) process.exit(1);
    const context = await deviceTierContext();
    console.log(
        `device declarations pass; standard=${context.standard.tier}, extras=${context.extras.tier}`,
    );
}
