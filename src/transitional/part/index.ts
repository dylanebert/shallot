// Destination: standard/rendering; owner: rendering-boundary.md.

import { Meshes, RenderingPlugin, Surfaces } from "../../core/rendering";
import type { Plugin } from "../../engine";
import { registration } from "../../engine";
import { initMeshes } from "./mesh";
import { Color, initializePartState, initPart, MeshInstance, PartSystem, warmPart } from "./part";

export { Color, MeshInstance, partTable } from "./part";

/**
 * the dogfooded MeshInstance producer. ECS-shaped per-entity rendering: `MeshInstance` +
 * `Color` components, the built-in cube mesh, and a GPU pack pipeline that
 * groups MeshInstances by surface and emits one indirect draw per used surface.
 * Renderer-independent: it publishes per-instance data (`globalTransforms`,
 * `color`, `eids`) but registers no surface, so it carries no lighting model
 * and renders under any consumer. The surfaces its entities point at
 * (`MeshInstance.surface` defaults to the name `"default"`) ship with the renderer:
 * the renderer registers `default`/`unlit`/`vertex` against the `eids` + `globalTransforms`
 * instance convention and its own `lit`. Depends on {@link RenderingPlugin}.
 */
export const PartPlugin: Plugin = {
    name: "Part",
    systems: [PartSystem],
    components: [
        registration("MeshInstance", MeshInstance, {
            defaults: (world) => {
                const surfaces = world.resource(Surfaces);
                const meshes = world.resource(Meshes);
                const surface = surfaces.id("default");
                const mesh = meshes.id("cube");
                // Empty registries are valid without a surface producer; populated registries need the defaults.
                if (surfaces.size > 0 && surface === undefined)
                    console.warn(
                        '[part] default surface "default" is not registered — a StandardRenderingPlugin or surface owner must register it; MeshInstance entities will bind whatever surface holds registry id 0',
                    );
                if (meshes.size > 0 && mesh === undefined)
                    console.warn(
                        '[part] default mesh "cube" is not registered — PartPlugin.initialize() registers it via initMeshes(); MeshInstance entities will bind whatever mesh holds registry id 0',
                    );
                return { surface: surface ?? 0, mesh: mesh ?? 0 };
            },
        }),
        registration("Color", Color, { defaults: () => ({ rgba: [1, 1, 1, 1] }) }),
    ],

    dependencies: [RenderingPlugin],

    initialize(world) {
        initializePartState(world);
        initPart(world);
        initMeshes(world);
    },

    warm: warmPart,
};

// MeshInstance's extension surface: the pack's GPU-output registry. `MeshInstance` (the component) + `Color` ride the main
// barrel; `MeshInstances` is the internal pack output — the slot-major `drawArgs` (DrawIndexedIndirect) + packed
// survivor eids a custom pipeline or a GPU-readback oracle reads. GPU handles, not author API, so it lives
// at the extension tier like render's `Draws` / `Surfaces` registries.

export { MeshInstances } from "./part";
