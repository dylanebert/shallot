import type { Vec3 } from "./gizmo";

/** a world-space bounding sphere — `center` and `radius`. The unit frame-to-fit works in. */
export interface Sphere {
    center: Vec3;
    radius: number;
}

/**
 * enclose a set of world spheres in one bounding sphere, via their common AABB. Returns null for an
 * empty set. A degenerate set of points (every radius 0) yields a zero-radius sphere at their box center —
 * the caller decides whether that's framable (frame-to-fit floors the radius).
 */
export function enclose(spheres: readonly Sphere[]): Sphere | null {
    if (spheres.length === 0) return null;
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (const { center, radius } of spheres) {
        minX = Math.min(minX, center[0] - radius);
        minY = Math.min(minY, center[1] - radius);
        minZ = Math.min(minZ, center[2] - radius);
        maxX = Math.max(maxX, center[0] + radius);
        maxY = Math.max(maxY, center[1] + radius);
        maxZ = Math.max(maxZ, center[2] + radius);
    }
    // center on the AABB center, then take the farthest reach (distance to each sphere's center + its
    // radius). Tighter than the AABB diagonal — a single sphere encloses to exactly itself, where the
    // diagonal would inflate it by √3.
    const center: Vec3 = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
    let radius = 0;
    for (const s of spheres) {
        const reach =
            Math.hypot(s.center[0] - center[0], s.center[1] - center[1], s.center[2] - center[2]) +
            s.radius;
        radius = Math.max(radius, reach);
    }
    return { center, radius };
}

/**
 * the perspective orbit distance that frames a bounding sphere of `radius` so it sits just inside the
 * frustum: `distance = radius·padding / sin(fov/2)`, the standard DCC frame-selected formula (the camera
 * pulls back until the sphere is tangent to the frustum's limiting planes). `fovDeg` is the camera's
 * vertical FOV; `aspect` (width / height) derives the horizontal FOV, and the *smaller* of the two is the
 * limiting axis — so a portrait viewport frames on width, not height. `padding` (≥1) leaves margin around
 * the sphere; 1.3 is a comfortable default.
 */
export function frameDistance(
    radius: number,
    fovDeg: number,
    aspect: number,
    padding = 1.3,
): number {
    const fovV = (fovDeg * Math.PI) / 180;
    const fovH = 2 * Math.atan(Math.tan(fovV / 2) * aspect);
    const fov = Math.min(fovV, fovH);
    return (radius * padding) / Math.sin(fov / 2);
}

/**
 * the orthographic half-height (`Camera.size`) that frames a bounding sphere of `radius`. The half-height
 * must cover the sphere vertically and, on a portrait viewport (`aspect` < 1), its horizontal extent
 * (`size·aspect`) must cover it too — so the fit divides by `min(1, aspect)`. `padding` matches
 * {@link frameDistance}.
 */
export function frameSize(radius: number, aspect: number, padding = 1.3): number {
    return (radius * padding) / Math.min(1, aspect);
}
