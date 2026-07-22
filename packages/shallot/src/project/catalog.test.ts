import { describe, expect, test } from "bun:test";
import * as shallot from "@dylanebert/shallot";
import { DEFAULT_PLUGINS } from "@dylanebert/shallot";
import * as avbd from "@dylanebert/shallot/avbd";
import { DEFAULT_PLUGIN_NAMES, KNOWN_ENGINE_PLUGINS, SUBPATH_PLUGIN_MODULES } from "./engine";

// The manifest references an engine plugin by its `.name` (e.g. "Orbit"); the generated `virtual:project`
// resolves it to a lean named import via the `${name}Plugin` convention — the main barrel
// (`import { OrbitPlugin } from "@dylanebert/shallot"`) for most, or a backend plugin's own subpath
// (`SUBPATH_PLUGIN_MODULES`, e.g. `AvbdPlugin` from `@dylanebert/shallot/avbd`) when it isn't
// barrel-listed. This gate keeps that convention exact across every engine plugin, on every module the
// generator can resolve from — add one that breaks it (a `Foo` plugin exported as `FooPlug`, or a
// subpath plugin missing from `SUBPATH_PLUGIN_MODULES`) and manifest resolution would silently miss it.

function pluginExports(ns: object): [string, { name: string }][] {
    const out: [string, { name: string }][] = [];
    for (const [key, value] of Object.entries(ns)) {
        if (!key.endsWith("Plugin")) continue;
        if (typeof value !== "object" || value === null) continue;
        if (typeof (value as { name?: unknown }).name !== "string") continue;
        out.push([key, value as { name: string }]);
    }
    return out;
}

describe("engine plugin naming convention", () => {
    test("every *Plugin export's .name is its identifier minus the Plugin suffix", () => {
        for (const [key, plugin] of [...pluginExports(shallot), ...pluginExports(avbd)]) {
            expect(`${plugin.name}Plugin`).toBe(key);
        }
    });

    test("the barrel exposes the expected plugin surface (sanity floor)", () => {
        // a floor, not an exact count — guards against the namespace import silently resolving to nothing
        expect(pluginExports(shallot).length).toBeGreaterThanOrEqual(15);
    });

    test("DEFAULT_PLUGIN_NAMES (the generator's dep-free list) matches the engine's defaults in order", () => {
        // the generator runs in Node without the plugin objects; this keeps its name list from drifting
        expect(DEFAULT_PLUGINS.map((p) => p.name)).toEqual([...DEFAULT_PLUGIN_NAMES]);
    });

    test("KNOWN_ENGINE_PLUGINS (the toolchain's dep-free union) covers every engine plugin the barrel exports", () => {
        // the manifest-boundary warn validates a `name: true` against this set; a plugin added to the barrel
        // without listing it here would false-warn as unknown, so the gate keeps the union exact
        const real = new Set(
            [...pluginExports(shallot), ...pluginExports(avbd)].map(([, p]) => p.name),
        );
        expect(KNOWN_ENGINE_PLUGINS).toEqual(real);
    });

    test("SUBPATH_PLUGIN_MODULES (the generator's dep-free list) names a real plugin on its declared subpath", () => {
        const avbdNames = new Set(pluginExports(avbd).map(([, p]) => p.name));
        for (const [name, source] of Object.entries(SUBPATH_PLUGIN_MODULES)) {
            expect(source).toBe("@dylanebert/shallot/avbd");
            expect(avbdNames.has(name)).toBe(true);
        }
    });
});
