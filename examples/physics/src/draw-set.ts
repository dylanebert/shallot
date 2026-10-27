import type { World } from "@dylanebert/shallot";
import { overlapAABB } from "@dylanebert/shallot/standard/physics";

type AABB = Parameters<typeof overlapAABB>[1];

/** Box3D camera.h DrawBounds and sample.cpp's shadow-caster union, in metres. */
export function drawingBounds(
    eye: readonly number[],
    forward: readonly number[],
    aspect: number,
    fov = 60,
): AABB {
    const lower = eye.map((v) => v - 1000);
    const upper = eye.map((v) => v + 1000);
    const tan = Math.tan((fov * Math.PI) / 360);
    const sun = [0.5, 0.8, 0.4];
    const length = Math.hypot(...sun);
    // Large World's bounds select Box3D's maximum shadow split distance (80 m).
    const split = (i: number) => 0.5 * 10 * (80 / 10) ** (i / 4) + 0.5 * (10 + (70 * i) / 4);
    for (let i = 0; i < 4; i++) {
        const near = split(i),
            far = split(i + 1);
        const middle = (near + far) / 2;
        const radius = Math.hypot((far - near) / 2, far * tan, far * tan * aspect);
        for (let axis = 0; axis < 3; axis++) {
            const center = eye[axis]! + middle * forward[axis]!;
            const swept = center + (50 * sun[axis]!) / length;
            lower[axis] = Math.min(lower[axis]!, center - radius, swept - radius);
            upper[axis] = Math.max(upper[axis]!, center + radius, swept + radius);
        }
    }
    return {
        lowerBound: { x: lower[0]!, y: lower[1]!, z: lower[2]! },
        upperBound: { x: upper[0]!, y: upper[1]!, z: upper[2]! },
    };
}

export function drawSet(world: World, bounds: AABB): Set<number> {
    const shapes = new Set<number>();
    overlapAABB(world, bounds, (shape) => {
        shapes.add(shape);
        return true;
    });
    return shapes;
}
