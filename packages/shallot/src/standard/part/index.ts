import type { Plugin } from "../../engine";
import { RenderPlugin } from "../render";
import { SlabPlugin } from "../slab";
import { initMeshes } from "./mesh";
import { Color, ColorTraits, initPart, Part, PartSystem, PartTraits, warmPart } from "./part";

export { Color, Part } from "./part";

/**
 * the dogfooded Part producer. ECS-shaped per-entity rendering: `Part` +
 * `Color` components, the built-in cube mesh, and a GPU pack pipeline that
 * groups Parts by surface and emits one indirect draw per used surface.
 * Renderer-independent — it publishes per-instance data (`transforms`,
 * `color`, `eids`) but registers no surface, so it carries no lighting model
 * and renders under any consumer. The surfaces its entities point at
 * (`Part.surface` defaults to the name `"default"`) ship with the renderer:
 * sear registers `default`/`unlit`/`vertex` against the `eids` + `transforms`
 * instance convention and its own `lit`. Depends on {@link RenderPlugin} (the
 * substrate) + `SlabPlugin`
 */
export const PartPlugin: Plugin = {
    name: "Part",
    systems: [PartSystem],
    components: { Part, Color },
    traits: {
        Part: PartTraits,
        Color: ColorTraits,
    },
    dependencies: [RenderPlugin, SlabPlugin],

    initialize() {
        initPart();
        initMeshes();
    },

    warm: warmPart,
};
