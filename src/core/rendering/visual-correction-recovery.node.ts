import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createApp, type WorldSnapshot } from "../../engine";
import { VisualCorrection } from "./visual-correction";

setDefaultTimeout(TIMEOUT.node);

test("visual correction decays monotonically across confirmation restore and replay in preFixed", async () => {
    let saved: WorldSnapshot;
    let confirm = false;
    let eid = 0;
    const samples: number[] = [];
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "CorrectionRecovery",
                components: [VisualCorrection],
                recovery: "stateless",
                systems: [
                    {
                        group: "preFixed",
                        update(w) {
                            if (!confirm) return;
                            const head = w.time.fixedTick;
                            w.restore(saved);
                            while (w.time.fixedTick < head) w.tick();
                            samples.push(w.storage(VisualCorrection).translation.x.get(eid));
                        },
                    },
                    {
                        group: "draw",
                        update(w) {
                            const x = w.storage(VisualCorrection).translation.x;
                            x.set(eid, x.get(eid) * 0.5);
                            samples.push(x.get(eid));
                        },
                    },
                ],
            },
        ],
    });
    const w = app.world;
    try {
        eid = w.create();
        w.add(eid, VisualCorrection, { translation: [1, 0, 0] });
        saved = w.snapshot();
        w.tick();
        w.tick();
        w.step(0);
        w.step(0);
        expect(samples).toEqual([0.5, 0.25]);
        confirm = true;
        w.step(0);
        expect(w.time.fixedTick).toBe(2);
        expect(samples).toEqual([0.5, 0.25, 0.25, 0.125]);
        w.step(0);
        expect(samples).toEqual([0.5, 0.25, 0.25, 0.125, 0.125, 0.0625]);
    } finally {
        w.dispose();
    }
});
