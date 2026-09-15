import type { TgpuRenderPipeline } from "typegpu";
import type { MeshIndex } from "../render";
import type { SurfaceGroupEntry } from "./pipelines";

// The one form every sear pass records a draw through. It lives in its own module because both callers —
// the color and prepass passes in ./forward and the shadow-atlas passes in ./atlas — would otherwise need a
// value import of the other's module, and ./pipelines already imports ./atlas for its layouts. The imports
// here are type-only, so this module pulls in nothing at runtime.

/**
 * the compiled pipeline with this entry's own group 2 bound — at the surface layout and, for a depth-shape
 * pipeline, its depth variant as well — and its index buffer, built on the entry's first draw and held in
 * the entry's `bound` map. The pass binds groups 0 and 1 per draw. The color pass binds the color layout
 * alone; the prepass and the point/cascade atlas passes bind the depth variant too.
 */
export function boundPipeline(
    g: SurfaceGroupEntry,
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
