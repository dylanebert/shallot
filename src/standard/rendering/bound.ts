import type { TgpuRenderPipeline } from "typegpu";
import type { MeshIndex } from "../../core/mesh";
import type { MaterialGroupEntry } from "./pipelines";

// A shared leaf lets color and atlas passes bind draws without a runtime import cycle.

/** Caches group 2 and index binding per entry/pipeline; the caller still binds groups 0 and 1. */
export function boundPipeline(
    g: MaterialGroupEntry,
    pipe: TgpuRenderPipeline<any>,
    group: GPUBindGroup,
    depthVariant: boolean,
    index: MeshIndex,
): TgpuRenderPipeline<any> {
    let bound = g.bound.get(pipe);
    if (!bound) {
        bound = pipe.with(g.layout, group);
        if (depthVariant) bound = bound.with(g.layout.depthVariant, group);
        bound = bound.withIndexBuffer(index);
        g.bound.set(pipe, bound);
    }
    return bound;
}
