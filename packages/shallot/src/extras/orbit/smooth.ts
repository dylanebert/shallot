import { f32, sparse, u8 } from "../../engine";

// OrbitSmooth holds the displayed yaw/pitch/distance/size, eased toward the authored values each frame
// (smoothness is the damping), plus the flyActive latch (1 while flying; its falling edge reprojects the
// orbit center so exiting fly is pose-continuous). Added on an entity's first frame with Orbit; that
// membership doubles as the "already snapped" flag, so a fresh camera starts framed. Derived state: never
// authored or serialized, re-snapped on every rebuild, so a reload can't desync it from the authored
// fields. Internal — a sibling export for the overlay and tests, never re-exported from the barrel.
export const OrbitSmooth = {
    yaw: sparse(f32),
    pitch: sparse(f32),
    distance: sparse(f32),
    size: sparse(f32),
    flyActive: sparse(u8),
};
