import type { Resource, World } from "../ecs";

/** CSS display size and device-pixel ratio for one bound canvas. */
export interface Viewport {
    cssWidth: number;
    cssHeight: number;
    dpr: number;
}

/** Per-World viewport rows keyed by the bound canvas's document/index slot; available without DOM setup. */
export const Viewports: Resource<ReadonlyMap<number, Viewport>> = {
    create: () => new Map<number, Viewport>(),
};

/** Produce a viewport row from the host, an application or a test driver. Sizes are CSS pixels. */
export function resizeViewport(
    world: World,
    index: number,
    width: number,
    height: number,
    dpr: number,
): void {
    const rows = world.resource(Viewports) as Map<number, Viewport>;
    rows.set(index, {
        cssWidth: Math.max(0, Number.isFinite(width) ? width : 0),
        cssHeight: Math.max(0, Number.isFinite(height) ? height : 0),
        dpr: Number.isFinite(dpr) && dpr > 0 ? dpr : 1,
    });
}
