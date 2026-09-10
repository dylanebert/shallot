import type { State } from "../ecs";
import { Registry } from "../utils";
import type { Node } from "./xml";

/** one scene pre-load resolver: scans parsed nodes for references it owns (a glTF mesh name) and awaits
 *  their registration, so `load` resolves every name. Registered by a plugin's `initialize`, removed by its
 *  `dispose`; a disabled plugin leaves no stale resolver. */
export interface Preloader {
    name: string;
    resolve(nodes: Node[], state: State): Promise<void> | void;
}

/**
 * the scene pre-load resolver registry. A plugin whose assets are referenced by name in scenes (the glTF
 * importer) registers a {@link Preloader} in `initialize` and deletes it in `dispose`; {@link preload} runs
 * every registered resolver over the parsed nodes before `load`, so a declarative reference (`part="mesh:
 * model.glb#0"`) triggers its own import. The engine's scene loop awaits it between
 * `parse` and `load`; a custom loader does the same.
 */
export const Preloads = new Registry<Preloader>();

/**
 * run every registered {@link Preloader} over parsed scene nodes: the awaited pre-load resolve pass.
 * Call between `parse` and `load` (the engine's `build` already does).
 *
 * @example
 * const nodes = parse(xml);
 * await preload(nodes, state);
 * load(nodes, state);
 */
export async function preload(nodes: Node[], state: State): Promise<void> {
    for (const p of Preloads) await p.resolve(nodes, state);
}
