import { Glob } from "bun";
import { revealAfterFirstFrame } from "../examples/loading-screen/src/reveal";
import type { Plugin } from "../src/engine";
import { registerGlobalTransform } from "../src/engine/ecs/global-transform";
import { readFields } from "../src/engine/ecs/reflection";
import { World } from "../src/engine/ecs/world";
import { DEFAULT_PLUGINS } from "../src/standard";

export async function componentKeys() {
    const subjects: Record<string, readonly Plugin[]> = {
        default: DEFAULT_PLUGINS,
        "examples/loading-screen/src/reveal.ts:revealAfterFirstFrame": [
            revealAfterFirstFrame(
                {
                    frame: {} as HTMLDivElement,
                    canvas: {} as HTMLCanvasElement,
                    reveal() {},
                    fail() {},
                },
                {},
            ),
        ],
    };
    for (const pattern of ["src/extras/*/index.ts", "examples/*/src/*.ts"]) {
        for (const path of [...new Glob(pattern).scanSync(".")].sort()) {
            if (/\.(test|node|e2e|oracle)\.ts$/.test(path)) continue;
            if (
                path.startsWith("examples/") &&
                !/satisfies Plugin|: Plugin|components:/.test(await Bun.file(path).text())
            )
                continue;
            const exports = await import(`${process.cwd()}/${path}`);
            for (const [name, value] of Object.entries(exports)) {
                if (
                    value &&
                    typeof value === "object" &&
                    "name" in value &&
                    ("components" in value || "systems" in value || "dependencies" in value)
                ) {
                    subjects[`${path}:${name}`] = [value as Plugin];
                }
            }
        }
    }
    const result: Record<string, Record<string, string[]>> = {};
    const keys = new WeakMap<object, string>();
    for (const [name, plugins] of Object.entries(subjects)) {
        const world = new World();
        registerGlobalTransform(world);
        const eid = world.create();
        const seen = new Set<Plugin>();
        const visit = (plugin: Plugin) => {
            if (seen.has(plugin)) return;
            seen.add(plugin);
            for (const dependency of plugin.dependencies ?? []) visit(dependency);
            for (const entry of plugin.components ?? []) world.registry.register(entry);
        };
        for (const plugin of plugins) visit(plugin);
        for (const { key, component } of world.registry.entries()) {
            const previous = keys.get(component);
            if (previous !== undefined && previous !== key) {
                throw new Error(`one record registered as both ${previous} and ${key}`);
            }
            keys.set(component, key);
        }
        result[name] = Object.fromEntries(
            [...world.registry.entries()]
                .sort((a, b) => a.key.localeCompare(b.key))
                .map(({ key, component }) => [
                    key,
                    Object.keys(readFields(world, component, eid)).sort(),
                ]),
        );
        world.dispose();
    }
    return result;
}

if (import.meta.main) console.log(JSON.stringify(await componentKeys(), null, 2));
