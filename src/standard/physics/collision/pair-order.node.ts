import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../../engine";
import { BodyType } from "../common/types";
import { init, threads as resolvedThreads, shutdown, workers } from "../kernel/kernel";
import { makeBoxHull } from "../shapes/build-hull";
import { createHullShape, createSphereShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { hashWorldState } from "../world/hash";
import { worldCounters, worldSetCustomFilterCallback } from "../world/world";

setDefaultTimeout(TIMEOUT.node);

// 1,600 moving spheres over a ground give the pair pass far more jobs than one 64-job block, so 8
// threads split them across workers. The custom filter runs at the serial point in job order, so its call sequence,
// the sorted keys and the contacts they create must not depend on which worker ran which block.
async function run(threads: number): Promise<string[]> {
    const owner = new World();
    try {
        await init(owner, { threads });
        expect(resolvedThreads(owner)).toBe(Math.max(threads, 1));
        expect(workers(owner) !== null).toBe(threads > 1);
        const world = createTestWorld({ enableSleep: false }, owner);
        try {
            const ground = createTestBody(world);
            createHullShape(world, ground, testShapeDef(), makeBoxHull(40, 0.5, 40, owner));
            for (let i = 0; i < 1600; ++i) {
                const body = createTestBody(world, {
                    type: BodyType.Dynamic,
                    position: {
                        x: (i % 40) * 1.1 - 22,
                        y: 1 + Math.floor(i / 400) * 1.1,
                        z: (Math.floor(i / 40) % 10) * 1.1 - 5,
                    },
                });
                createSphereShape(world, body, testShapeDef({ enableCustomFiltering: true }), {
                    center: { x: 0, y: 0, z: 0 },
                    radius: 0.6,
                });
            }
            const lines: string[] = [];
            worldSetCustomFilterCallback(world, (a, b) => {
                lines.push(`F ${a.index1} ${b.index1}`);
                return (a.index1 + b.index1) % 7 !== 0;
            });
            for (let step = 0; step < 40; ++step) {
                stepTestWorld(world, 1 / 60);
                lines.push(
                    `S ${step} ${worldCounters(world).contactCount} ${hashWorldState(world).toString(16)}`,
                );
            }
            return lines;
        } finally {
            destroyTestWorld(world);
        }
    } finally {
        await shutdown(owner);
        owner.dispose();
    }
}

test("custom-filter calls, pair keys and contacts follow job order at 1 and 8 threads", async () => {
    const single = await run(0);
    expect(single.filter((l) => l.startsWith("F ")).length).toBeGreaterThan(1000);
    expect(await run(8)).toEqual(single);
    expect(Bun.hash(single.join("\n")).toString(16)).toBe("babf85e2579d614b");
});
