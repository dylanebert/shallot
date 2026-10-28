import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createApp } from "../../engine";
import {
    getSensorData,
    type PhysicsSensorData,
    StandardPhysicsPlugin,
    setTransform,
} from "./index";
import { sensorVisitors } from "./sensor-data.fixture";
import gold from "./sensor-data.gold.json";

setDefaultTimeout(TIMEOUT.node);
test("sensor visitors equal native contents and order across begin, end and restore, borrowing one World-owned array", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    const world = app.world;
    try {
        const shapes = sensorVisitors(world);
        const sensor = shapes[0]!;
        expect(getSensorData(world, sensor)).toBeNull();
        let borrowed: PhysicsSensorData | null = null;
        for (let tick = 0; tick < gold.length; tick++) {
            if (tick === 1) setTransform(world, shapes[2]!, [10, 0, 0], [0, 0, 0, 1]);
            if (tick === 2) setTransform(world, shapes[2]!, [0, 0, 0], [0, 0, 0, 1]);
            if (tick === 3) setTransform(world, shapes[1]!, [10, 0, 0], [0, 0, 0, 1]);
            world.tick();
            const result = getSensorData(world, sensor)!;
            if (borrowed) expect(result).toBe(borrowed);
            borrowed = result;
            expect(result.shapes.slice(0, result.count).map((eid) => shapes.indexOf(eid))).toEqual(
                gold[tick]!,
            );
            const saved = world.snapshot();
            world.tick();
            world.restore(saved);
            const restored = getSensorData(world, sensor)!;
            expect(
                restored.shapes.slice(0, restored.count).map((eid) => shapes.indexOf(eid)),
            ).toEqual(gold[tick]!);
        }
        expect(getSensorData(world, shapes[1]!)?.count).toBe(0);
        expect(getSensorData(world, 0)).toBeNull();
        world.destroy(sensor);
        expect(getSensorData(world, sensor)).toBeNull();
    } finally {
        app.dispose();
    }
});
