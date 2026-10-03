import { expect, test } from "bun:test";
import { pointToSegmentDistance, vec3 } from "./math";

test("pointToSegmentDistance clamps a query beyond either end to the endpoint itself", () => {
    const a = vec3.zero();
    const b = { x: 2, y: 0, z: 0 };
    expect(pointToSegmentDistance(a, b, { x: 1, y: 5, z: 0 })).toEqual({ x: 1, y: 0, z: 0 });
    expect(pointToSegmentDistance(a, b, { x: -3, y: 1, z: 0 })).toBe(a);
    expect(pointToSegmentDistance(a, b, { x: 9, y: 1, z: 0 })).toBe(b);
});
