import { expect, test } from "bun:test";
import { DEFAULT_PLUGINS, type Plugin, resolvePlugins } from "@dylanebert/shallot";
import { APP_PLUGINS } from "./plugins";

test("the page's first-person plugin list resolves with the engine defaults", () => {
    const plugins = new Set<Plugin>([...DEFAULT_PLUGINS, ...APP_PLUGINS]);
    const pending = [...plugins];
    for (let index = 0; index < pending.length; index++) {
        for (const dependency of pending[index]!.dependencies ?? []) {
            if (plugins.has(dependency)) continue;
            plugins.add(dependency);
            pending.push(dependency);
        }
    }
    const composition = resolvePlugins([...plugins]);
    expect(
        composition.missing.map(({ plugin, dependency }) => `${plugin.name}→${dependency.name}`),
    ).toEqual([]);
});
