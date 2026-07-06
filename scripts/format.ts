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

const glob = new Glob("**/*.scene");
// examples/templates/* are generated from the create-shallot template; leave them byte-identical
const ignore = ["node_modules", "dist", "_legacy", "examples/templates"];

let formatted = 0;
let unchanged = 0;
let errors = 0;

for await (const path of glob.scan({ cwd: process.cwd() })) {
    if (ignore.some((dir) => path.includes(dir))) continue;

    try {
        const content = await Bun.file(path).text();
        const nodes = parse(content);
        normalizeNodes(nodes);
        const output = stringify(nodes) + "\n";

        if (content !== output) {
            await Bun.write(path, output);
            console.log(`formatted: ${path}`);
            formatted++;
        } else {
            unchanged++;
        }
    } catch (e) {
        console.error(`error: ${path}: ${(e as Error).message}`);
        errors++;
    }
}

console.log(`\n${formatted} formatted, ${unchanged} unchanged, ${errors} errors`);
process.exit(errors > 0 ? 1 : 0);
