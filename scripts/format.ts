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
    AnimationPlugin,
    AudioPlugin,
} = await import("../packages/shallot/src");
const { register } = await import("../packages/shallot/src/engine/ecs/core");
const { normalizeAttr } = await import("../packages/shallot/src/engine/scene/core");

// the engine defaults plus the opt-in viz extras that add scene-authorable components,
// so normalizeAttr knows every component schema a scene can reference
const PLUGINS = [...DEFAULT_PLUGINS, LinesPlugin, TextPlugin, AnimationPlugin, AudioPlugin];

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

/** the `key` set of a `k: v; k2: v2` attribute value. */
function attrKeys(value: string): string[] {
    return value
        .split(";")
        .map((pair) => pair.split(":")[0]?.trim() ?? "")
        .filter((k) => k !== "");
}

function normalizeNodes(nodes: Node[]) {
    for (const node of nodes) {
        for (const attr of node.attrs) {
            if (!attr.value) continue;
            const normalized = normalizeAttr(attr.name, attr.value);
            if (normalized !== null) {
                // a normalization is a parse→format round trip, and a field whose parser needs
                // runtime state the formatter doesn't have formats back to its default and drops off
                // the line: every shipped `animator` lost its `clip:` this way, silently, and parked on a
                // green gate. Refuse rather than write — the fix is the component's parse/format pair.
                const before = attrKeys(attr.value);
                const after = new Set(attrKeys(normalized));
                const dropped = before.filter((k) => !after.has(k));
                if (dropped.length > 0) {
                    throw new Error(
                        `formatting "${attr.name}" would drop ${dropped.map((k) => `"${k}"`).join(", ")} (from \`${attr.value}\`) — the component's parse/format pair doesn't round-trip without runtime state; fix it rather than format`,
                    );
                }
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
