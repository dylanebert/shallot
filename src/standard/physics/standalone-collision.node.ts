import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createApp } from "../../engine";
import {
    getSweepTransform,
    StandardPhysicsPlugin,
    shapeCast,
    shapeDistance,
    timeOfImpact,
} from ".";
import { collisionCases, emptyCache, publicCollisionOutput } from "./standalone-collision.fixture";
import { publicEdgeOutput } from "./standalone-collision-edge.fixture";
import failed from "./standalone-collision-failed.json";
import history from "./standalone-collision-history.json";

setDefaultTimeout(TIMEOUT.node);
test("public standalone queries equal native hit, miss, touching and deep-overlap records and simplex histories", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        expect(publicCollisionOutput(app.world)).toBe(
            await Bun.file(new URL("./standalone-collision.gold.txt", import.meta.url)).text(),
        );
    } finally {
        app.dispose();
    }
});

test("warm moving caches, every truncation capacity, complete misses, failed TOI and rotating sweep transforms equal native", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        expect(publicEdgeOutput(app.world)).toBe(
            await Bun.file(new URL("./standalone-collision-edge.gold.txt", import.meta.url)).text(),
        );
        expect(shapeDistance(app.world, history, emptyCache(), 32).simplexCount).toBe(32);
        expect(timeOfImpact(app.world, failed).state).toBe(1);
    } finally {
        app.dispose();
    }
});

test("standalone queries borrow per-world records, update caches and bound simplex history", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const other = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const input = collisionCases.at(-1)!.input;
        const cache = emptyCache();
        const d = shapeDistance(app.world, input, cache, 32);
        expect(d.simplexCount).toBeGreaterThan(0);
        expect(cache.count).toBeGreaterThan(0);
        const distance = d.distance;
        expect(shapeDistance(app.world, input, cache)).toBe(d);
        expect(d.distance).toBe(distance);
        expect(d.simplexCount).toBe(0);
        expect(shapeDistance(other.world, input, emptyCache(), 32)).not.toBe(d);
        const castInput = {
            ...input,
            translationB: { x: -10, y: 0, z: 0 },
            maxFraction: 1,
            canEncroach: false,
        };
        expect(shapeCast(app.world, castInput)).toBe(shapeCast(app.world, castInput));
        const zero = { x: 0, y: 0, z: 0 },
            q = { v: zero, s: 1 };
        const sweep = {
            localCenter: { x: 1, y: 0, z: 0 },
            c1: zero,
            c2: { x: 4, y: 0, z: 0 },
            q1: q,
            q2: q,
        };
        const t = getSweepTransform(app.world, sweep, 0.5);
        expect(t).toEqual({ p: { x: 1, y: 0, z: 0 }, q });
        expect(getSweepTransform(app.world, sweep, 0)).toBe(t);
        const toiInput = { ...input, sweepA: sweep, sweepB: sweep, maxFraction: 1 };
        expect(timeOfImpact(app.world, toiInput)).toBe(timeOfImpact(app.world, toiInput));
        expect(() => shapeDistance(app.world, input, emptyCache(), 33)).toThrow(RangeError);
        expect(() => shapeDistance(app.world, input, { ...emptyCache(), count: 5 })).toThrow(
            RangeError,
        );
        expect(() =>
            shapeDistance(
                app.world,
                { ...input, proxyA: { ...input.proxyA, count: 0 } },
                emptyCache(),
            ),
        ).toThrow(RangeError);
    } finally {
        app.dispose();
        other.dispose();
    }
});
