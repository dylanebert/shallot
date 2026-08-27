import { generateNetwork } from "./overlay/network";

// The boot document's road frame — the fixed geometry `corridorCapture.ts`'s pose derivation anchors to.
// Pulled into its own module, with zero `@dylanebert/shallot/extras`/`render/core` imports, so it stays
// importable under plain `bun test`.
//
// Reuses the *same* boot document every run (`generateNetwork()` — no seed, `terrain.ts`'s own boot
// document) — road 0's own single segment, fixed and never seeded, so a
// SPACING/TILE_RES/DIST_RANGE edit between runs changes only the mesh/atlas resolution, never the network
// geometry the frame is derived from.

/** the boot document's one road, flattened to its own single segment — the fixed geometry every caller
 *  frames against. */
function roadSegment(): { ax: number; az: number; bx: number; bz: number; halfWidth: number } {
    const doc = generateNetwork();
    const [[ax, az], [bx, bz]] = doc.polylines[0].points;
    return { ax, az, bx, bz, halfWidth: doc.polylines[0].halfWidth };
}

export interface RoadFrame {
    ax: number;
    az: number;
    ux: number;
    uz: number;
    len: number;
    nx: number;
    nz: number;
    halfWidth: number;
}

/** {@link roadSegment}'s endpoints resolved into the frame `corridorCapture.ts` anchors to: unit tangent
 *  (`ux`/`uz`), unit outward normal (`nx`/`nz`, `network.ts`'s own convention), and the
 *  segment length. */
export function roadFrame(): RoadFrame {
    const { ax, az, bx, bz, halfWidth } = roadSegment();
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz) || 1;
    const ux = dx / len;
    const uz = dz / len;
    const nx = -uz; // unit normal, network.ts's own convention
    const nz = ux;
    return { ax, az, ux, uz, len, nx, nz, halfWidth };
}
