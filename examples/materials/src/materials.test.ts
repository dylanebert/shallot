import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { ripple, waveVector } from "./materials";

const radius = 0.65;
function sample(position: d.v3f, time: number) {
    const normal = d.vec3f(position.x / radius, position.y / radius, position.z / radius);
    const phase = position.x * waveVector.x + position.y * waveVector.y + position.z * waveVector.z;
    const surface = ripple(position, normal, phase, time);
    return { position: Array.from(surface.position), normal: Array.from(surface.normal) };
}

test("the travelling ripple keeps the sphere within 0.03 of its radius", () => {
    for (let time = 0; time < 2; time += 0.1)
        for (let y = -0.6; y <= 0.6; y += 0.1) {
            const position = d.vec3f(Math.sqrt(radius * radius - y * y), y, 0);
            expect(
                Math.abs(Math.hypot(...sample(position, time).position) - radius),
            ).toBeLessThanOrEqual(0.030001);
        }
    const position = d.vec3f(radius, 0, 0);
    expect(sample(position, 0.7).position).not.toEqual(sample(position, 0).position);
});

test("three ripple crests travel across the sphere's diameter", () => {
    const crests = (time: number) => {
        const offsets = Array.from({ length: 201 }, (_, i) => {
            const y = -radius + (2 * radius * i) / 200;
            return (
                Math.hypot(
                    ...sample(d.vec3f(0, y, Math.sqrt(Math.max(0, radius * radius - y * y))), time)
                        .position,
                ) - radius
            );
        });
        return offsets.flatMap((offset, i) =>
            i > 0 &&
            i < 200 &&
            offset > offsets[i - 1]! &&
            offset > offsets[i + 1]! &&
            offset > 0.02
                ? [i]
                : [],
        );
    };
    expect(crests(0)).toHaveLength(3);
    expect(crests(0.4)).toHaveLength(3);
    expect(crests(0.4)).not.toEqual(crests(0));
});

test("the ripple normal follows the displaced surface's tangents", () => {
    const position = d.vec3f(radius, 0, 0);
    const surface = sample(position, 0.4);
    const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i]!, 0);
    for (const axis of [1, 2]) {
        const point = [radius, 0, 0];
        point[axis] = 0.0001;
        const length = Math.hypot(...point);
        const next = sample(
            d.vec3f(
                (point[0]! * radius) / length,
                (point[1]! * radius) / length,
                (point[2]! * radius) / length,
            ),
            0.4,
        );
        const tangent = next.position.map((value, i) => value - surface.position[i]!);
        const tangentLength = Math.hypot(...tangent);
        expect(Math.abs(dot(surface.normal, tangent) / tangentLength)).toBeLessThan(0.001);
    }
    expect(
        Math.hypot(surface.normal[0]! - 1, surface.normal[1]!, surface.normal[2]!),
    ).toBeGreaterThan(0.02);
    expect(Math.hypot(...surface.normal)).toBeCloseTo(1, 6);
});
