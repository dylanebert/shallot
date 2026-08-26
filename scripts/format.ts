import { resolve } from "node:path";
import { Glob } from "bun";
import { setupGlobals } from "bun-webgpu";
import type { Node } from "../packages/shallot/src";

// the engine references WebGPU globals (e.g. GPUShaderStage) at module scope, so define
// them before importing it — mirrors tests/setup.ts. ES imports are hoisted, hence dynamic.
await setupGlobals();

const {
    State,
    parse,
    stringify,
    DEFAULT_PLUGINS,
    LinesPlugin,
    TextPlugin,
    TweenPlugin,
    AudioPlugin,
} = await import("../packages/shallot/src");
const { register } = await import("../packages/shallot/src/engine/ecs/core");
const { normalizeAttr } = await import("../packages/shallot/src/engine/scene/core");

// the engine defaults plus the opt-in viz extras that add scene-authorable components,
// so normalizeAttr knows every component schema a scene can reference
const PLUGINS = [...DEFAULT_PLUGINS, LinesPlugin, TextPlugin, TweenPlugin, AudioPlugin];

const state = new State();
for (const plugin of PLUGINS) {
    if (plugin.components) {
        for (const [name, component] of Object.entries(plugin.components)) {
            register(name, component, plugin.traits?.[name]);
        }
    }
    if (plugin.systems) {
        for (const system of plugin.systems) state.addSystem(system, plugin.name);
    }
}

function normalizeNodes(nodes: Node[]) {
    for (const node of nodes) {
        for (const attr of node.attrs) {
            if (!attr.value) continue;
            const normalized = normalizeAttr(attr.name, attr.value);
            if (normalized !== null) {
                attr.value = normalized;
            }
        }
        normalizeNodes(node.children);
    }
}

// --check: report-only mode — print the would-change set and exit nonzero without writing.
// Default (no flag) writes, so `bun run format` is unchanged.
const checkOnly = process.argv.includes("--check");

const glob = new Glob("**/*.scene");
const ignore = ["node_modules", "dist", "_legacy"];

let formatted = 0;
let wouldChange = 0;
let unchanged = 0;
let errors = 0;

// anchor on the script location, not process.cwd(), so --check is not vacuously green
// when run from a subdirectory (siblings use import.meta.dir for the same reason)
const root = resolve(import.meta.dir, "..");

for await (const path of glob.scan({ cwd: root })) {
    // segment match so an ignore entry "dist" does not also match "distortion/"
    const segments = path.split("/");
    if (ignore.some((dir) => segments.includes(dir))) continue;

    try {
        const fullPath = resolve(root, path);
        const content = await Bun.file(fullPath).text();
        const nodes = parse(content);
        normalizeNodes(nodes);
        const output = stringify(nodes) + "\n";

        if (content !== output) {
            if (checkOnly) {
                console.log(`would format: ${path}`);
                wouldChange++;
            } else {
                await Bun.write(fullPath, output);
                console.log(`formatted: ${path}`);
                formatted++;
            }
        } else {
            unchanged++;
        }
    } catch (e) {
        console.error(`error: ${path}: ${(e as Error).message}`);
        errors++;
    }
}

if (checkOnly) {
    console.log(`\n${wouldChange} would change, ${unchanged} unchanged, ${errors} errors`);
    process.exit(errors > 0 || wouldChange > 0 ? 1 : 0);
} else {
    console.log(`\n${formatted} formatted, ${unchanged} unchanged, ${errors} errors`);
    process.exit(errors > 0 ? 1 : 0);
}
