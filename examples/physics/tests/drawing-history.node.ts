import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, Time } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { drawingHistory } from "../src/drawing-history";

setDefaultTimeout(TIMEOUT.node);

test("query ink interpolates fixed records at 120 Hz without mutating the simulation drawing", async () => {
    const records = [[1, 0, 1, 2, 0, 2, 3]];
    const history = drawingHistory(() => records);
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "Drawing",
                systems: [
                    ...history.systems,
                    {
                        group: "fixed",
                        update() {
                            records[0]![1]! += 0.1;
                            records[0]![4]! += 0.1;
                        },
                    },
                ],
            },
        ],
    });
    try {
        const positions: number[] = [];
        for (let i = 0; i < 10; i++) {
            app.world.step(Time.FIXED_DT / 2);
            positions.push(history.records(app.world)[0]![1]!);
        }
        for (let i = 2; i < positions.length; i++)
            expect(positions[i]! - positions[i - 1]!).toBeCloseTo(0.05, 6);
        expect(records[0]![1]).toBeCloseTo(0.5, 6);
        expect(history.records(app.world)[0]![1]).toBeCloseTo(0.4, 6);
    } finally {
        app.dispose();
    }
});
