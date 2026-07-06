import { describe, expect, test } from "bun:test";
import { compose, invert, multiply, perspective } from "../..";
import {
    CULL_FRUSTUM,
    CULL_VOLUME_FLOATS,
    FRUSTUM_FLOATS,
    frustumPlanes,
    frustumVolume,
} from "./frustum";
import { meshBounds } from "./mesh";

// Pure-CPU logic only (no device). The real-GPU cull truth — survivor counts, survivor identity,
// per-view slot offsets, the View prune on a destroyed camera — lives in the gym `render` scenario
// (`bun bench --scenario render`), the single source of truth for anything that binds a device.

// mirrors the WGSL `visible` test: a sphere is inside when its signed distance
// stays ≥ -radius for all six planes
function inside(planes: Float32Array, x: number, y: number, z: number, r: number): boolean {
    for (let i = 0; i < 6; i++) {
        const o = i * 4;
        const dist = planes[o] * x + planes[o + 1] * y + planes[o + 2] * z + planes[o + 3];
        if (dist < -r) return false;
    }
    return true;
}

// viewProj for a camera at `pos` looking down -Z (identity orientation), the
// same path computeViewProj walks: proj · inverse(world)
function lookFromZ(pos: [number, number, number]): Float32Array {
    const proj = perspective(60, 1, 0.1, 1000);
    const world = compose(...pos, 0, 0, 0, 1, 1, 1, 1);
    return multiply(proj, invert(world));
}

describe("frustumPlanes", () => {
    test("a point in front of the camera is inside, off to the side or behind is out", () => {
        const planes = frustumPlanes(lookFromZ([0, 0, 5]), new Float32Array(FRUSTUM_FLOATS));
        // origin sits centered in front of the camera at (0,0,5)
        expect(inside(planes, 0, 0, 0, 0.9)).toBe(true);
        // far to the +X side — outside the right plane
        expect(inside(planes, 500, 0, 0, 0.9)).toBe(false);
        // behind the camera (camera looks toward -Z, this is past it on +Z)
        expect(inside(planes, 0, 0, 50, 0.9)).toBe(false);
    });

    test("the bounding radius widens the test — a sphere straddling a plane stays in", () => {
        const planes = frustumPlanes(lookFromZ([0, 0, 5]), new Float32Array(FRUSTUM_FLOATS));
        // a hair past the near plane (z = 4.9) as a point is out, but a large
        // sphere centered there still intersects the frustum
        expect(inside(planes, 0, 0, 4.95, 0)).toBe(false);
        expect(inside(planes, 0, 0, 4.95, 1)).toBe(true);
    });
});

describe("frustumVolume", () => {
    test("packs the frustum tag in the header vec4, then the camera's planes after it", () => {
        // two slots, so the slot-1 base offset (not just slot 0) is exercised
        const out = new Float32Array(2 * CULL_VOLUME_FLOATS);
        const viewProj = lookFromZ([0, 0, 5]);
        frustumVolume(out, 1, viewProj);
        const base = CULL_VOLUME_FLOATS;
        expect(out[base]).toBe(CULL_FRUSTUM); // tag in the header vec4's .x
        // the planes equal frustumPlanes written standalone, shifted one vec4 past the header
        const planes = frustumPlanes(viewProj, new Float32Array(FRUSTUM_FLOATS));
        for (let i = 0; i < FRUSTUM_FLOATS; i++) expect(out[base + 4 + i]).toBe(planes[i]);
    });
});

describe("meshBounds", () => {
    test("center is the AABB midpoint, radius the farthest vertex", () => {
        // two vertices on the x axis (8 floats each: pos.xyz + u, normal.xyz + v)
        // prettier-ignore
        const verts = new Float32Array([-1, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 1, 0]);
        const [cx, cy, cz, r] = meshBounds(verts);
        expect(cx).toBe(1); // midpoint of -1 and 3, f32-exact
        expect(cy).toBe(0);
        expect(cz).toBe(0);
        expect(r).toBe(2); // each vertex is 2 from the center, f32-exact
    });

    test("a symmetric box's radius is its half-diagonal", () => {
        // prettier-ignore
        const verts = new Float32Array([-1, -1, -1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]);
        const [, , , r] = meshBounds(verts);
        expect(r).toBeCloseTo(Math.sqrt(3), 6); // irrational half-diagonal; f-precision tier
    });
});
