import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, BodyType, createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { setBodyType } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { BodySamplePlugin } from "../../../examples/physics/src/bodies";
import {
    changeControl,
    controlValues,
    createScene,
    scenes,
} from "../../../examples/physics/src/scenes";
import { sampleGold } from "../../../examples/physics/tests/gold";

setDefaultTimeout(TIMEOUT.node);

test("setBodyType preserves native call order against reverse entity order", async () => {
    const gold = sampleGold(
        await Bun.file(
            new URL("../../../examples/physics/tests/bodyType.controls.gold.txt", import.meta.url),
        ).text(),
    );
    const sequence = (
        await Bun.file(
            new URL("../../../examples/physics/tests/bodyType.controls.txt", import.meta.url),
        ).text()
    )
        .trim()
        .split("\n")
        .map((line) => {
            const [step, name, value] = line.split("\t");
            return { step: Number(step), name: name!, value: Number(value) };
        });
    let bodies: number[] = [];
    const values = controlValues("bodyType");
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, BodySamplePlugin],
        setup(world) {
            bodies = createScene(world, "bodyType");
        },
    });
    try {
        for (let step = 0; step < 350; step++) {
            for (const change of sequence)
                if (change.step === step) {
                    const option = scenes.bodyType.controls.Type.options.find(
                        (option) => option.label === change.name,
                    );
                    changeControl(
                        app.world,
                        "bodyType",
                        bodies,
                        values,
                        option ? "Type" : change.name,
                        option?.value ?? change.value,
                    );
                }
            app.world.tick();
            expect(hashPhysics(app.world)).toBe(gold[step]!.hash);
        }
        const saved = app.world.snapshot();
        expect(bodies[2]!).toBeGreaterThan(bodies[1]!);
        for (const index of [2, 1, 4, 5, 6])
            setBodyType(app.world, bodies[index]!, BodyType.Dynamic);
        app.world.tick();
        expect(hashPhysics(app.world)).toBe(gold[350]!.hash);
        app.world.restore(saved);
        for (const index of [1, 2, 4, 5, 6])
            setBodyType(app.world, bodies[index]!, BodyType.Dynamic);
        app.world.tick();
        expect(hashPhysics(app.world)).not.toBe(gold[350]!.hash);
        app.world.restore(saved);
        for (const index of [2, 1, 4, 5, 6])
            app.world.storage(Body).type.set(bodies[index]!, BodyType.Dynamic);
        app.world.tick();
        expect(hashPhysics(app.world)).not.toBe(gold[350]!.hash);
    } finally {
        app.dispose();
    }
});
