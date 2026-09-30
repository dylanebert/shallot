import { expect, setDefaultTimeout, test } from "bun:test";
import { build, swap } from "../app";
import { field, snapshot, u32 } from "../ecs";
import { probeBuffer } from "./probe";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("a deterministic composition hashes identically under two frame pacings with requested readback active", async () => {
    const Counter = { value: field(u32) };
    async function run(dt: number, frames: number) {
        let eid = 0;
        let source!: GPUBuffer;
        const requests: ReturnType<typeof probeBuffer>[] = [];
        const app = await build({
            defaults: false,
            plugins: [
                {
                    name: "DeterministicCounter",
                    components: { Counter },
                    initialize(state) {
                        eid = state.create();
                        state.add(eid, Counter);
                        source = state.gpu.device.createBuffer({
                            size: 4,
                            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                        });
                        state.gpu.device.queue.writeBuffer(source, 0, new Uint32Array([42]));
                    },
                    systems: [
                        {
                            group: "fixed",
                            update(state) {
                                const counter = state.of(Counter).value;
                                counter.set(eid, counter.get(eid) + state.time.fixedTick);
                            },
                        },
                        {
                            group: "draw",
                            update(state) {
                                requests.push(
                                    probeBuffer(state, source, { label: "pacing diagnostic" }),
                                );
                            },
                        },
                    ],
                },
            ],
        });
        try {
            for (let i = 0; i < frames; i++) app.state.step(dt);
            const results = await Promise.all(requests);
            expect(results).toHaveLength(frames);
            for (let i = 0; i < results.length; i++) {
                expect(results[i].frame).toBe(i);
                expect(new Uint32Array(results[i].bytes)[0]).toBe(42);
            }
            expect(app.state.time.fixedTick).toBe(60);
            expect(app.state.of(Counter).value.get(eid)).toBe(1830);
            return Bun.hash(JSON.stringify(snapshot(app.state)));
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
    const app = await build({ defaults: false, plugins: [before] });
    try {
        expect(await swap(app.state, [before], [after])).toEqual({
            ok: true,
        });
    } finally {
        app.dispose();
    }
});
