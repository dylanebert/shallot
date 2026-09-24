import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import { perspective } from "../../engine/utils/math";
import { frustumPlanes } from "./frustum";

/** A pure frustum whose left plane passes exactly through `boundary`. */
const FRUSTUM_FIXTURE = {
    fov: 90,
    aspect: 1,
    near: 1,
    far: 10,
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

check(
    "the frustum keeps a boundary sphere visible",
    {
        claim: "the frustum keeps an exactly tangent sphere visible, so a strict boundary cull defect reds",
        subject: ["src/core/rendering/frustum.ts", "src/engine/utils/math.ts"],
    },
    () => {
        const projection = perspective(
            FRUSTUM_FIXTURE.fov,
            FRUSTUM_FIXTURE.aspect,
            FRUSTUM_FIXTURE.near,
            FRUSTUM_FIXTURE.far,
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
    },
);
