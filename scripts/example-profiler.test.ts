import { expect, spyOn, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { type Plugin, World } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin, ProfilePlugin } from "@dylanebert/shallot/extras";

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
        const html = readFileSync(
            new URL(`../examples/${name}/index.html`, import.meta.url),
            "utf8",
        );
        const motes = name === "particles" || name === "volumetric-fog";
        const entry =
            name === "physics"
                ? readFileSync(new URL("../examples/physics/src/main.ts", import.meta.url), "utf8")
                : html;
        expect(entry).toMatch(motes ? /plugins:\s*\[MotesPlugin\]/ : /plugins:\s*\[AppPlugin\]/);
        const module = await import(`../examples/${name}/src/${motes ? "motes" : "app"}.ts`);
        const plugins = composition(motes ? module.MotesPlugin : module.AppPlugin);
        expect(plugins.has(DiagnosticsOverlayPlugin)).toBe(true);
        expect(plugins.has(ProfilePlugin)).toBe(true);
        if (name === "particles") {
            expect([...plugins].some((plugin) => plugin.name === "Fog")).toBe(false);
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
            Object.assign(key, { key: "F3" });
            window.dispatchEvent(key);
            expect(key.defaultPrevented).toBe(true);
            expect(ui.visible).toBe(true);
            window.dispatchEvent(key);
            expect(ui.visible).toBe(false);
        } finally {
            resource.mockRestore();
            world.dispose();
            if (previous) Object.defineProperty(globalThis, "window", previous);
            else Reflect.deleteProperty(globalThis, "window");
        }
    });
}
