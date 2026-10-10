import { expect, test } from "bun:test";
import { composeMat4, perspective } from "../../engine/utils/math";
import { CULL_VOLUME_FLOATS, frustumPlanes, frustumVolume } from "./frustum";

/** A pure frustum whose left plane passes exactly through `boundary`. */
const FRUSTUM_FIXTURE = {
    fov: 90,
    aspect: 1,
    near: 1,
    boundary: [-5, 0, -5] as const,
    outside: [-5.01, 0, -5] as const,
    radius: 0,
};

function signedDistance(planes: Float32Array, plane: number, point: readonly number[]): number {
    const base = plane * 4;
    return (
        planes[base] * point[0] +
        planes[base + 1] * point[1] +
        planes[base + 2] * point[2] +
        planes[base + 3]
    );
}

test("the frustum keeps an exactly tangent sphere visible, so a strict boundary cull defect reds", () => {
    const projection = perspective(
        FRUSTUM_FIXTURE.fov,
        FRUSTUM_FIXTURE.aspect,
        FRUSTUM_FIXTURE.near,
    );
    const planes = frustumPlanes(projection, new Float32Array(24));
    const boundaryDistances = Array.from({ length: 6 }, (_, plane) =>
        signedDistance(planes, plane, FRUSTUM_FIXTURE.boundary),
    );
    const outsideDistances = Array.from({ length: 6 }, (_, plane) =>
        signedDistance(planes, plane, FRUSTUM_FIXTURE.outside),
    );
    const inside = (distances: readonly number[]) =>
        distances.every((distance) => distance >= -FRUSTUM_FIXTURE.radius);

    // The fixture point is exactly on the left plane. The public plane extraction and the cull
    // boundary therefore agree without inventing a tolerance or a second projection.
    expect(Math.abs(boundaryDistances[0] ?? Number.NaN)).toBeLessThan(1e-5);
    expect(inside(boundaryDistances)).toBe(true);
    expect(inside(outsideDistances)).toBe(false);
    expect(outsideDistances[0]).toBeLessThan(-FRUSTUM_FIXTURE.radius);
});

test("an infinite perspective frustum still culls beyond its finite far bound", () => {
    const cameraWorld = composeMat4(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
    const projection = perspective(90, 1, 1);
    const volume = new Float32Array(CULL_VOLUME_FLOATS);
    frustumVolume(volume, 0, projection, cameraWorld, 10);
    const planes = volume.subarray(4, 28);
    const beyondFar = [0, 0, -10.1] as const;
    const depth = (projection[10]! * beyondFar[2] + projection[14]!) / -beyondFar[2];
    expect(signedDistance(planes, 4, [0, 0, -9.9])).toBeGreaterThan(0);
    expect(depth).toBeGreaterThan(0);
    expect(signedDistance(planes, 4, beyondFar)).toBeLessThan(0);
});
