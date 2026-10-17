import { expect, spyOn, test } from "bun:test";
import { readFileSync } from "node:fs";
import { type Plugin, World } from "@dylanebert/shallot";
import { DiagnosticsOverlayPlugin, ProfilePlugin } from "@dylanebert/shallot/extras";
import { AppPlugin as city } from "../examples/city/src/app";
import { AppPlugin as cycle } from "../examples/day-night-cycle/src/app";
import { AppPlugin as firstPerson } from "../examples/first-person/src/app";
import { AppPlugin as interact } from "../examples/interact/src/app";
import { AppPlugin as interior } from "../examples/lit-interior/src/app";
import { AppPlugin as materials } from "../examples/materials/src/app";
import { MotesPlugin } from "../examples/particles/src/motes";
import { AppPlugin as voxel } from "../examples/voxel/src/app";
import { AppPlugin as worldUi } from "../examples/world-ui/src/app";

const examples = [
    ["city", city],
    ["voxel", voxel],
    ["lit-interior", interior],
    ["day-night-cycle", cycle],
    ["materials", materials],
    ["first-person", firstPerson],
    ["interact", interact],
    ["world-ui", worldUi],
    ["particles", MotesPlugin],
] as const;

function composition(plugin: Plugin, result = new Set<Plugin>()): Set<Plugin> {
    if (result.has(plugin)) return result;
    result.add(plugin);
    for (const dependency of plugin.dependencies ?? []) composition(dependency, result);
    return result;
}

for (const [name, plugin] of examples) {
    test(`${name}'s entry composes the profiler closed until F3, then open`, () => {
        const html = readFileSync(
            new URL(`../examples/${name}/index.html`, import.meta.url),
            "utf8",
        );
        expect(html).toMatch(
            name === "particles" ? /plugins:\s*\[MotesPlugin\]/ : /plugins:\s*\[AppPlugin\]/,
        );
        const plugins = composition(plugin);
        expect(plugins.has(DiagnosticsOverlayPlugin)).toBe(true);
        expect(plugins.has(ProfilePlugin)).toBe(true);

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
