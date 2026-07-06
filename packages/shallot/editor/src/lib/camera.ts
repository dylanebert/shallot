import { Camera, Depth, Orbit, type State, Tag, Transform } from "@dylanebert/shallot";
import { entries, lanes } from "@dylanebert/shallot/ecs/core";
import { Overlays } from "./viewport";

// the WYSIWYG boundary: mirror the scene camera's *look* (clearColor background + fov framing), not its
// *navigation rig*. Projection mode, ortho size, and the clip planes are the edit camera's own — so an
// orthographic scene camera doesn't force the edit view into a disorienting free-orbit ortho (the edit cam
// stays perspective, like Unity's / Godot's scene view, whose projection is independent of the game
// camera). `near`/`far` are driven each frame by the editor's adaptive frustum, so they aren't mirrored.
const cameraFields = ["fov", "clearColor"] as const;

// Camera/Transform/Orbit are the editor camera's own pose, and Tag/Depth/Overlays are its viewport-only
// markers (prepass lanes + the per-camera overlay set) — none of these are copied from the scene camera,
// or the `else if` remove branch in syncCameraEffects would strip the outline whenever the scene camera
// lacks them.
const syncExclude = new Set<object>();

/** copy one component field's value between two entities — the raw typed-array/array form and the
 * Single/Pair/Quad storage surface. A Pair/Quad has no scalar `{get,set}`, so it copies through its lanes
 * (each a Single) — an effect component's vec4 field (Glaze's slope/offset/power) is the case. */
export function copyField(target: unknown, fromEid: number, toEid: number): void {
    if (target == null) return;
    if (ArrayBuffer.isView(target) || Array.isArray(target)) {
        const arr = target as number[];
        arr[toEid] = arr[fromEid];
        return;
    }
    const n = lanes(target);
    if (n === 1) {
        const f = target as { get(eid: number): number; set(eid: number, value: number): void };
        f.set(toEid, f.get(fromEid));
    } else if (n >= 2) {
        const f = target as { x: unknown; y: unknown; z?: unknown; w?: unknown };
        copyField(f.x, fromEid, toEid);
        copyField(f.y, fromEid, toEid);
        if (n === 4) {
            copyField(f.z, fromEid, toEid);
            copyField(f.w, fromEid, toEid);
        }
    }
}

/**
 * mirror a scene camera's effect components onto the editor's viewport camera so editing the scene
 * camera previews live. Copies every registered component except the editor-only set above, then the
 * plain Camera value fields by hand. Pure over the State — `bun test`-covered against a `new State()`.
 */
export function syncCameraEffects(s: State, fromEid: number, toEid: number): void {
    if (!syncExclude.size) {
        for (const c of [Camera, Transform, Orbit, Overlays, Tag, Depth]) syncExclude.add(c);
    }
    for (const { component } of entries()) {
        if (syncExclude.has(component)) continue;
        const c = component as Record<string, unknown>;
        if (s.has(fromEid, component as never)) {
            if (!s.has(toEid, component as never)) s.add(toEid, component as never);
            for (const field of Object.keys(c)) copyField(c[field], fromEid, toEid);
        } else if (s.has(toEid, component as never)) {
            s.remove(toEid, component as never);
        }
    }
    for (const field of cameraFields) {
        copyField((Camera as Record<string, unknown>)[field], fromEid, toEid);
    }
}
