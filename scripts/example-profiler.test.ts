import { expect, spyOn, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { parse } from "@babel/parser";
import { type Plugin, World } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin, ProfilePlugin } from "@dylanebert/shallot/extras";

// e88ad2ff: a hidden ProfilePlugin still grants timestamp queries and resolves them every
// frame, so these six small examples stay unprofiled.
const exclusions = new Set([
    "minimal",
    "hud",
    "sound",
    "loading-screen",
    "save-and-load",
    "embedded-scene",
]);
const examples = readdirSync(new URL("../examples/", import.meta.url), { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && !exclusions.has(entry.name))
    .map((entry) => entry.name)
    .sort();

function composition(plugin: Plugin, result = new Set<Plugin>()): Set<Plugin> {
    if (result.has(plugin)) return result;
    result.add(plugin);
    for (const dependency of plugin.dependencies ?? []) composition(dependency, result);
    return result;
}

for (const name of examples) {
    test(`${name}'s entry composes the profiler closed until F3, then open`, async () => {
        const root = new URL(`../examples/${name}/`, import.meta.url);
        const html = readFileSync(new URL("index.html", root), "utf8");
        const script = html.match(/<script\b([^>]*\btype="module"[^>]*)>([\s\S]*?)<\/script>/);
        expect(script).not.toBeNull();
        const src = script![1]!.match(/\bsrc="([^"]+)"/)?.[1];
        const entry = src ? new URL(src.replace(/^\//, ""), root) : new URL("index.html", root);
        const source = src ? readFileSync(entry, "utf8") : script![2]!;
        const imports = new Map<string, { path: string; name: string }>();
        for (const node of parse(source, { sourceType: "module", plugins: ["typescript"] }).program
            .body) {
            if (node.type !== "ImportDeclaration") continue;
            for (const specifier of node.specifiers)
                if (
                    specifier.type === "ImportSpecifier" &&
                    specifier.imported.type === "Identifier"
                )
                    imports.set(specifier.local.name, {
                        path: node.source.value,
                        name: specifier.imported.name,
                    });
        }
        const lists = [...source.matchAll(/\bplugins:\s*\[([^\]]*)\]/g)];
        expect(lists.length).toBeGreaterThan(0);
        for (const [, list] of lists) {
            const plugins = new Set<Plugin>();
            for (const item of list!.split(",")) {
                const binding = imports.get(item.trim());
                if (!binding) continue;
                const module = await import(
                    binding.path.startsWith(".") ? new URL(binding.path, entry).href : binding.path
                );
                composition(module[binding.name], plugins);
            }
            expect(plugins.has(DiagnosticsOverlayPlugin)).toBe(true);
            expect(plugins.has(ProfilePlugin)).toBe(true);
            if (name === "particles") {
                expect([...plugins].some((plugin) => plugin.name === "Fog")).toBe(false);
            }
        }

        const world = new World();
        const window = new EventTarget();
        const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
        Object.defineProperty(globalThis, "window", { configurable: true, value: window });
        const resource = spyOn(world, "resource");
        try {
            DiagnosticsOverlayPlugin.initialize!(world);
            const ui = resource.mock.results[0].value as { visible: boolean; overlay: unknown };
            expect(ui.visible).toBe(false);
            expect(ui.overlay).toBeNull();
            const key = new Event("keydown", { cancelable: true });
            Object.assign(key, { key: "F3", code: "F3" });
            const release = Object.assign(new Event("keyup", { cancelable: true }), {
                key: "F3",
                code: "F3",
            });
            window.dispatchEvent(key);
            window.dispatchEvent(release);
            expect(key.defaultPrevented).toBe(true);
            expect(ui.visible).toBe(true);
            window.dispatchEvent(key);
            window.dispatchEvent(release);
            expect(ui.visible).toBe(false);
        } finally {
            resource.mockRestore();
            world.dispose();
            if (previous) Object.defineProperty(globalThis, "window", previous);
            else Reflect.deleteProperty(globalThis, "window");
        }
    });
}
