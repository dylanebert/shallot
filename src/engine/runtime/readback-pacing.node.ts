import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp, swapPlugins } from "../app";
import { component, u32 } from "../ecs";
import { probeBuffer } from "./probe";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("a deterministic composition hashes identically under two frame pacings with requested readback active", async () => {
    const Counter = { value: u32 };
    async function run(dt: number, frames: number) {
        let eid = 0;
        let source!: GPUBuffer;
        const requests: ReturnType<typeof probeBuffer>[] = [];
        const app = await createApp({
            defaults: false,
            plugins: [
                {
                    name: "DeterministicCounter",
                    gpu: {},
                    components: [component("Counter", Counter)],
                    initialize(world) {
                        eid = world.create();
                        world.add(eid, Counter);
                        source = world.gpu.device.createBuffer({
                            size: 4,
                            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                        });
                        world.gpu.device.queue.writeBuffer(source, 0, new Uint32Array([42]));
                    },
                    systems: [
                        {
                            group: "fixed",
                            update(world) {
                                const counter = world.storage(Counter).value;
                                counter.set(eid, counter.get(eid) + world.time.fixedTick);
                            },
                        },
                        {
                            group: "draw",
                            update(world) {
                                requests.push(
                                    probeBuffer(world, source, { label: "pacing diagnostic" }),
                                );
                            },
                        },
                    ],
                },
            ],
        });
        try {
            for (let i = 0; i < frames; i++) app.world.step(dt);
            const results = await Promise.all(requests);
            expect(results).toHaveLength(frames);
            for (let i = 0; i < results.length; i++) {
                expect(results[i].frame).toBe(i);
                expect(new Uint32Array(results[i].bytes)[0]).toBe(42);
            }
            expect(app.world.time.fixedTick).toBe(60);
            expect(app.world.storage(Counter).value.get(eid)).toBe(1830);
            const values = app.world.storage(Counter).value;
            return Bun.hash(
                JSON.stringify(
                    [...app.world.query([Counter])].map((eid) => [eid, values.get(eid)]),
                ),
            );
        } finally {
            app.dispose();
        }
    }
    expect(await run(1 / 120, 120)).toBe(await run(1 / 30, 30));
});

test("a determinism declaration is metadata, not runtime readback permission state", async () => {
    const system = { group: "fixed" as const, update() {} };
    const before = { name: "ChangedPermission", deterministic: false, systems: [system] };
    const after = { name: "ChangedPermission", deterministic: true, systems: [{ ...system }] };
    const app = await createApp({ defaults: false, plugins: [before] });
    try {
        expect(await swapPlugins(app.world, [before], [after])).toEqual({
            ok: true,
        });
    } finally {
        app.dispose();
    }
});
