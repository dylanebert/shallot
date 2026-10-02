import { MeshPlugin } from "../../core/mesh";
import { RenderingPlugin } from "../../core/rendering";
import { type Plugin, registration } from "../../engine";
import { Color, initializePartState, initPart, PartSystem, warmPart } from "./part";

/** Packs mesh instances into per-view, per-surface indirect draws. */
export const PartPlugin: Plugin = {
    name: "Part",
    systems: [PartSystem],
    components: [registration("Color", Color, { defaults: () => ({ rgba: [1, 1, 1, 1] }) })],
    dependencies: [RenderingPlugin, MeshPlugin],
    initialize(world) {
        initializePartState(world);
        initPart(world);
    },
    warm: warmPart,
};
