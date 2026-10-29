import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape } from "@dylanebert/shallot";
import { setAngularVelocity } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { DeterminismPlugin, DeterminismState, determinismScenes } from "../src/determinism";
import { sceneSwitcher } from "../src/sample";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(determinismScenes) as (keyof typeof determinismScenes)[]) {
    test(`${scene} equals native steps and sample readouts and rejects a body change`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        const rows = (
            await Bun.file(new URL(`./${scene}.readouts.gold.txt`, import.meta.url)).text()
        )
            .trim()
            .split("\n");
        const start = performance.now();
        for (const mutated of [false, true]) {
            const app = await createApp({ defaults: false, plugins: [DeterminismPlugin] });
            try {
                sceneSwitcher(app.world)(scene);
                let diverged = false;
                for (const [step, expected] of gold.entries()) {
                    if (mutated && step === 2) {
                        const e = app.world.resource(DeterminismState).bodies[0]!;
                        if (scene === "querySpawn")
                            setAngularVelocity(app.world, e, { x: 10, y: 0, z: 0 });
                        else app.world.storage(Shape).density.set(e, 1001);
                    }
                    app.world.tick();
                    const hash = hashPhysics(app.world);
                    if (mutated) {
                        if (hash !== expected.hash) {
                            diverged = true;
                            break;
                        }
                    } else {
                        expect(hash, `step ${step}`).toBe(expected.hash);
                        const s = app.world.resource(DeterminismState);
                        const hex = (v: number) => v.toString(16).toUpperCase().padStart(8, "0");
                        const row = `${s.sleepStep} ${hex(s.hash)}${scene === "querySpawn" ? ` ${s.spawnCount} ${s.hits} ${hex(s.queryHash)}` : ""}`;
                        expect(row, `readouts ${step}`).toBe(rows[step]!);
                    }
                }
                if (mutated) expect(diverged).toBe(true);
                else if (scene === "querySpawn") {
                    for (let step = 600; step < 681; step++) app.world.tick();
                    const s = app.world.resource(DeterminismState),
                        hex = (v: number) => v.toString(16).toUpperCase().padStart(8, "0");
                    expect(
                        `${s.sleepStep} ${hex(s.hash)} ${s.spawnCount} ${s.hits} ${hex(s.queryHash)}\n`,
                    ).toBe(
                        await Bun.file(
                            new URL("./querySpawn.sleep.gold.txt", import.meta.url),
                        ).text(),
                    );
                }
            } finally {
                app.dispose();
            }
        }
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
