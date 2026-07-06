import { describe, expect, test } from "bun:test";
import * as shallot from "@dylanebert/shallot";
import { DEFAULT_PLUGINS } from "@dylanebert/shallot";
import { DEFAULT_PLUGIN_NAMES } from "./engine";

// The manifest references an engine plugin by its `.name` (e.g. "Orbit"); the generated `virtual:project`
// resolves it to a lean barrel import via the `${name}Plugin` convention (`import { OrbitPlugin } from
// "@dylanebert/shallot"`). This gate keeps that convention exact across every engine plugin — add one
// that breaks it (a `Foo` plugin exported as `FooPlug`) and manifest resolution would silently miss it.

function pluginExports(): [string, { name: string }][] {
    const out: [string, { name: string }][] = [];
    for (const [key, value] of Object.entries(shallot)) {
        if (!key.endsWith("Plugin")) continue;
        if (typeof value !== "object" || value === null) continue;
        if (typeof (value as { name?: unknown }).name !== "string") continue;
        out.push([key, value as { name: string }]);
    }
    return out;
}

describe("engine plugin naming convention", () => {
    test("every *Plugin export's .name is its identifier minus the Plugin suffix", () => {
        for (const [key, plugin] of pluginExports()) {
            expect(`${plugin.name}Plugin`).toBe(key);
        }
    });

    test("the barrel exposes the expected plugin surface (sanity floor)", () => {
        // a floor, not an exact count — guards against the namespace import silently resolving to nothing
        expect(pluginExports().length).toBeGreaterThanOrEqual(15);
    });

    test("DEFAULT_PLUGIN_NAMES (the generator's dep-free list) matches the engine's defaults in order", () => {
        // the generator runs in Node without the plugin objects; this keeps its name list from drifting
        expect(DEFAULT_PLUGINS.map((p) => p.name)).toEqual([...DEFAULT_PLUGIN_NAMES]);
    });
});
