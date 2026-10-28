import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BodySamplePlugin } from "../src/bodies";
import { changeControl, controlValues, createScene, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);
test("Precession's read-only rows equal the native sample's averaged rates each step", async () => {
    const gold = (
        await Bun.file(new URL("./gyroscopicPrecession.readouts.gold.txt", import.meta.url)).text()
    )
        .trim()
        .split("\n");
    let bodies: number[] = [];
    const app = await createApp({
        defaults: false,
        plugins: [BodySamplePlugin],
        setup(world) {
            bodies = createScene(world, "gyroscopicPrecession");
        },
    });
    try {
        expect(gold).toHaveLength(600);
        const rows = Object.entries(scenes.gyroscopicPrecession.controls);
        expect(rows.map(([, control]) => control.label)).toEqual(["Expected rate", "Actual rate"]);
        const values = controlValues("gyroscopicPrecession");
        for (let step = 0; step < 600; step++) {
            app.world.tick();
            const expected = gold[step]!.split(" ").map((value) =>
                value === "null" ? null : Math.fround(Number(value)),
            );
            for (const [i, [name, row]] of rows.entries()) {
                expect(row.read!(app.world), `step ${step}, ${name}`).toBe(expected[i]!);
                expect(row.decimals).toBe(4);
                expect(() =>
                    changeControl(app.world, "gyroscopicPrecession", bodies, values, name, 10),
                ).toThrow();
            }
        }
    } finally {
        app.dispose();
    }
});
