// The transform gizmo's pivot point — what the Move / Rotate / Scale handles measure and turn about for a
// multi-entity selection. Pure data here, the way `frame.ts` holds the orientation set: the pivot identity,
// the dropdown's ordered list, and the anchor each mode resolves to. App owns the active-pivot `$state`; the
// viewport-bar dropdown renders from `PIVOTS`, and `Handles.getOrigin` resolves the anchor via
// {@link pivotAnchor}. Two modes today (Median / Active, Unity's Center / Pivot pair); the list is the
// extension point — Individual Origins / 3D Cursor (Blender's set) drop in as one more entry plus the
// anchor that computes it, no UI change.
import { CircleDot, Crosshair } from "lucide-static";
import type { Vec3 } from "./gizmo";

/** what the gizmo pivots about for a multi-entity selection. Median = the selection centroid; Active = the
 * active (last-picked) entity's own origin. For a lone selection the two coincide. */
export const Pivot = { Median: 0, Active: 1 } as const;
export type Pivot = (typeof Pivot)[keyof typeof Pivot];

export interface PivotDef {
    id: Pivot;
    label: string;
    /** lucide-static SVG markup, rendered through `Icon` */
    icon: string;
}

/** the dropdown order, top to bottom. */
export const PIVOTS: PivotDef[] = [
    { id: Pivot.Median, label: "Median", icon: CircleDot },
    { id: Pivot.Active, label: "Active", icon: Crosshair },
];

/** Median is the default — the selection centroid, the least surprising pivot for a fresh selection. */
export const DEFAULT_PIVOT: Pivot = Pivot.Median;

/** the hotkey that cycles the pivot (. — Blender's pivot-menu key). The single source of truth: the
 * keydown handler matches it and the toolbar tooltip renders its {@link hint}. */
export const PIVOT_KEY = ".";

/** the next pivot in the cycle after `p` (wraps), for the hotkey. */
export function nextPivot(p: Pivot): Pivot {
    const i = PIVOTS.findIndex((d) => d.id === p);
    return PIVOTS[(i + 1) % PIVOTS.length].id;
}

/** the gizmo's world-space anchor for a selection's `points` (index-aligned to the selection, `active`
 * the active entity's index): the centroid for {@link Pivot.Median}, the active entity's origin for
 * {@link Pivot.Active}. Null for an empty selection. */
export function pivotAnchor(mode: Pivot, points: Vec3[], active: number): Vec3 | null {
    if (points.length === 0) return null;
    if (mode === Pivot.Active) return points[active] ?? points[points.length - 1];
    let x = 0;
    let y = 0;
    let z = 0;
    for (const p of points) {
        x += p[0];
        y += p[1];
        z += p[2];
    }
    return [x / points.length, y / points.length, z / points.length];
}
