// Run with `bun test ./diagnostics/placement-history/history.oracle.ts` at each revision.
// Times only the synchronous step; GPU completion is awaited outside the sample.
import { test } from "bun:test";
import { createApp, Time, Transform } from "@dylanebert/shallot";
import { globalTransformTable, RenderingPlugin } from "@dylanebert/shallot/rendering";
import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("report CPU step cost for static and fixed-moving placement histories", async () => {
    const count = 10_000;
    for (const moving of [false, true]) {
        const app = await createApp({ defaults: false, plugins: [RenderingPlugin] });
        try {
            const { world } = app;
            const eids = new Uint32Array(count);
            for (let i = 0; i < count; i++) {
                const eid = world.create();
                eids[i] = eid;
                world.add(eid, Transform, { translation: [i, 0, 0] });
            }
            globalTransformTable(world);
            let tick = 0;
            if (moving) {
                world.addSystem({
                    group: "fixed",
                    update() {
                        tick++;
                        const translation = world.storage(Transform).translation;
                        for (let i = 0; i < count; i++) translation.x.set(eids[i]!, i + tick);
                    },
                });
            }
            const samples: number[] = [];
            for (let i = 0; i < 1300; i++) {
                const start = performance.now();
                world.step(Time.FIXED_DT);
                const elapsed = performance.now() - start;
                await world.gpu.device.queue.onSubmittedWorkDone();
                if (i >= 300) samples.push(elapsed);
            }
            samples.sort((a, b) => a - b);
            console.log(JSON.stringify({
                rows: count,
                moving,
                samples: samples.length,
                medianMs: samples[500],
                p95Ms: samples[950],
                meanMs: samples.reduce((sum, value) => sum + value, 0) / samples.length,
            }));
        } finally {
            app.dispose();
        }
    }
}, 120_000);
