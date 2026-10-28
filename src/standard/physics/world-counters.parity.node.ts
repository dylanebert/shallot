import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { getWorldCounters, PhysicsWorldDefinition } from ".";
import { counterPile, counterRow, counterSweep } from "./fixtures/world-counters";

setDefaultTimeout(TIMEOUT.node);
for (const workers of [0, 1, 2])
    test(`world counters equal native Box3D on ordinary and graph-overflow piles (${workers} workers)`, async () => {
        const rows: string[] = [];
        for (const overflow of [false, true]) {
            const app = await counterPile(overflow, workers);
            try {
                const world = app.world;
                rows.push(counterRow(getWorldCounters(world)!));
                world.resource(PhysicsWorldDefinition).enableAutomaticStep = true;
                let sawOverflow = false;
                for (let i = 0; i < 60; i++) {
                    world.tick();
                    const counters = getWorldCounters(world)!;
                    sawOverflow ||= counters.colorCounts[23]! > 0;
                    rows.push(counterRow(counters));
                }
                expect(sawOverflow).toBe(overflow);
            } finally {
                app.dispose();
            }
        }
        const sweep = await counterSweep(workers);
        expect(sweep.distanceIterations).toBeGreaterThan(0);
        rows.push(...sweep.rows);
        expect(rows.join("\n") + "\n").toBe(
            await Bun.file(new URL("./world-counters.gold.txt", import.meta.url)).text(),
        );
    });
