// Destination: standard/rendering; owner: rendering-boundary.md.

import { RenderPlugin } from "../../core/rendering";
import type { Plugin } from "../../engine";
import { initMeshes } from "./mesh";
import {
    Color,
    ColorTraits,
    initializePartState,
    initPart,
    MeshInstance,
    PartSystem,
    PartTraits,
    warmPart,
} from "./part";

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
 * instance convention and its own `lit`. Depends on {@link RenderPlugin}.
 */
export const PartPlugin: Plugin = {
    name: "Part",
    systems: [PartSystem],
    components: { MeshInstance, Color },
    traits: {
        MeshInstance: PartTraits,
        Color: ColorTraits,
    },
    dependencies: [RenderPlugin],

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
