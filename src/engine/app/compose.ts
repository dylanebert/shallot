import { declaration } from "../ecs";
import type { Plugin } from ".";

/** a required plugin edge whose dependency is absent from the composition. */
export interface MissingPluginDependency {
    readonly plugin: Plugin;
    readonly dependency: Plugin;
}

/** the device-free result of resolving a plugin composition. */
export interface PluginComposition {
    readonly plugins: readonly Plugin[];
    readonly missing: readonly MissingPluginDependency[];
}

/**
 * resolve required plugin edges without acquiring a device or mutating engine registries.
 * Ordering and missing-edge reporting are stable relative to the supplied plugin order.
 */
export function resolvePlugins(plugins: readonly Plugin[]): PluginComposition {
    const nodes = [...new Set(plugins)];
    const names = new Set<string>();
    for (const plugin of nodes) {
        if (names.has(plugin.name)) {
            throw new Error(
                `plugin "${plugin.name}" is listed more than once; give one of them its own name`,
            );
        }
        names.add(plugin.name);
    }
    const present = new Set(nodes);
    const missing: MissingPluginDependency[] = [];
    const adjacent = new Map(nodes.map((plugin) => [plugin, [] as Plugin[]]));
    const degree = new Map(nodes.map((plugin) => [plugin, 0]));

    const holders = new Map<string, { fields: object; plugin: string }>();
    for (const plugin of nodes) {
        for (const fields of plugin.components ?? []) {
            const { key } = declaration(fields, plugin.name);
            const holder = holders.get(key);
            if (!holder) holders.set(key, { fields, plugin: plugin.name });
            else if (holder.fields !== fields) {
                throw new Error(
                    `component "${key}" is declared by plugin "${holder.plugin}" and by another record in plugin "${plugin.name}"; give one of them its own key`,
                );
            }
        }
        for (const dependency of plugin.dependencies ?? []) {
            if (!present.has(dependency)) {
                missing.push({ plugin, dependency });
                continue;
            }
            adjacent.get(dependency)!.push(plugin);
            degree.set(plugin, degree.get(plugin)! + 1);
        }
    }

    const queue = nodes.filter((plugin) => degree.get(plugin) === 0);
    const ordered: Plugin[] = [];
    while (queue.length > 0) {
        const plugin = queue.shift()!;
        ordered.push(plugin);
        for (const dependent of adjacent.get(plugin)!) {
            const next = degree.get(dependent)! - 1;
            degree.set(dependent, next);
            if (next === 0) queue.push(dependent);
        }
    }

    if (ordered.length !== nodes.length) {
        const cyclic = nodes
            .filter((plugin) => degree.get(plugin)! > 0)
            .map((plugin) => plugin.name);
        throw new Error(`Circular plugin dependency: ${cyclic.join(", ")}`);
    }

    return { plugins: ordered, missing };
}
