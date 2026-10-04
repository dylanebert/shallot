import { expect, test } from "bun:test";
import { clipVector, solvePlanes } from "@dylanebert/shallot/standard/physics";

test("published plane solvers return caller-owned outputs, including aliased vectors", () => {
    const planes = [
        {
            plane: { normal: { x: 0, y: 1, z: 0 }, offset: 0 },
            pushLimit: 3.4028234663852886e38,
            push: 0,
            clipVelocity: true,
        },
    ];
    const target = { x: 2, y: -1, z: 3 };
    const expected = solvePlanes(target, planes, 1);
    const delta = { x: 99, y: 99, z: 99 };
    const out = { delta, iterationCount: -1 };
    expect(solvePlanes(target, planes, 1, out)).toBe(out);
    expect(out.delta).toBe(delta);
    expect(out).toEqual(expected);
    const velocity = { x: 2, y: -1, z: 3 };
    const clipped = { x: 99, y: 99, z: 99 };
    const expectedClip = clipVector(velocity, planes, 1);
    expect(clipVector(velocity, planes, 1, clipped)).toBe(clipped);
    expect(clipped).toEqual(expectedClip);
    expect(clipVector(velocity, planes, 1, velocity)).toBe(velocity);
    expect(velocity).toEqual(expectedClip);
    expect(clipVector(target, [], 0)).not.toBe(target);
    const alias = { delta: target, iterationCount: -1 };
    expect(solvePlanes(target, planes, 1, alias)).toBe(alias);
    expect(alias).toEqual(expected);
});
