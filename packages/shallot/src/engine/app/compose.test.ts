import { describe, expect, test } from "bun:test";
import { entries, register } from "../ecs/core";
import type { Plugin } from ".";
import { resolvePlugins } from "./compose";

describe("resolvePlugins", () => {
    test("orders dependencies deterministically while preserving unrelated input order", () => {
        const base: Plugin = { name: "base" };
        const first: Plugin = { name: "first" };
        const dependent: Plugin = { name: "dependent", dependencies: [base] };

        expect(resolvePlugins([dependent, first, base]).plugins).toEqual([first, base, dependent]);
        expect(resolvePlugins([dependent, first, base]).plugins).toEqual([first, base, dependent]);
    });

    test("returns every missing required edge in plugin and dependency order", () => {
        const left: Plugin = { name: "left" };
        const right: Plugin = { name: "right" };
        const a: Plugin = { name: "a", dependencies: [left, right] };
        const b: Plugin = { name: "b", dependencies: [right] };

        const result = resolvePlugins([a, b]);
        expect(result.plugins).toEqual([a, b]);
        expect(
            result.missing.map(({ plugin, dependency }) => [plugin.name, dependency.name]),
        ).toEqual([
            ["a", "left"],
            ["a", "right"],
            ["b", "right"],
        ]);
    });

    test("does not run plugin hooks or mutate the component registry", () => {
        const anchor = { value: [] as number[] };
        register("compose-anchor", anchor);
        const before = [...entries()];
        let effects = 0;
        const dependency: Plugin = { name: "dependency" };
        const plugin: Plugin = {
            name: "plugin",
            dependencies: [dependency],
            components: { added: { value: [] as number[] } },
            initialize: () => {
                effects++;
            },
            warm: () => {
                effects++;
            },
        };

        resolvePlugins([plugin]);
        expect(effects).toBe(0);
        expect([...entries()]).toEqual(before);
    });

    test("rejects cycles without running plugin hooks", () => {
        let effects = 0;
        const a: Plugin = {
            name: "a",
            initialize: () => {
                effects++;
            },
        };
        const b: Plugin = { name: "b", dependencies: [a] };
        Object.assign(a, { dependencies: [b] });

        expect(() => resolvePlugins([a, b])).toThrow("Circular plugin dependency: a, b");
        expect(effects).toBe(0);
    });
});
