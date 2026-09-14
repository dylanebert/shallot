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

/** the device tier selected by a plugin composition. */
export interface DeviceTier {
    readonly tier: "cpu" | "gpu";
    readonly required: readonly string[];
    readonly optional: readonly string[];
}

/**
 * classify a plugin list without acquiring a device or touching engine state. Required device plugins
 * make the composition GPU-bound; optional plugins retain their CPU truth and mirror it when available.
 */
export function deviceTier(plugins: readonly Plugin[]): DeviceTier {
    const required: string[] = [];
    const optional: string[] = [];
    for (const plugin of plugins) {
        if (plugin.device === "required") required.push(plugin.name);
        else if (plugin.device === "optional") optional.push(plugin.name);
    }
    return { tier: required.length > 0 ? "gpu" : "cpu", required, optional };
}

/**
 * resolve required plugin edges without acquiring a device or mutating engine registries.
 * Ordering and missing-edge reporting are stable relative to the supplied plugin order.
 */
export function resolvePlugins(plugins: readonly Plugin[]): PluginComposition {
    const nodes = [...new Set(plugins)];
    const present = new Set(nodes);
    const missing: MissingPluginDependency[] = [];
    const adjacent = new Map(nodes.map((plugin) => [plugin, [] as Plugin[]]));
    const degree = new Map(nodes.map((plugin) => [plugin, 0]));

    for (const plugin of nodes) {
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
