import { run, bench, summary } from "mitata";
import { build } from "../src/engine";
import { buf, capacity } from "../src/engine/ecs/capacity";
import { createFieldProxy } from "../src/engine/ecs/component";

const SMALL_N = 1_000;
const LARGE_N = 100_000;

export async function runCapacityBenchmarks() {
    console.log("\n=== Capacity Benchmarks ===\n");

    summary(() => {
        bench(`grow to ${SMALL_N} entities (cold start)`, async function* () {
            yield async () => {
                const state = await build({ plugins: [], defaults: false });
                buf(Float32Array, 4, 0);
                buf(Uint32Array, 1, 0);
                buf(Uint8Array, 1, 0);
                for (let i = 0; i < SMALL_N; i++) state.addEntity();
                state.dispose();
            };
        });

        bench(`grow to ${LARGE_N} entities (cold start)`, async function* () {
            yield async () => {
                const state = await build({ plugins: [], defaults: false });
                buf(Float32Array, 4, 0);
                buf(Uint32Array, 1, 0);
                buf(Uint8Array, 1, 0);
                for (let i = 0; i < LARGE_N; i++) state.addEntity();
                state.dispose();
            };
        });
    });

    const writeState = await build({ plugins: [], defaults: false });
    const writeBuf = buf(Float32Array, 4, 0);
    const writeProxy = createFieldProxy(writeBuf, 4, 0);
    const writeEids: number[] = [];
    for (let i = 0; i < LARGE_N; i++) writeEids.push(writeState.addEntity());

    let frame = 0;
    summary(() => {
        bench(`proxy write at ${LARGE_N} entities`, () => {
            frame++;
            for (let i = 0; i < writeEids.length; i++) writeProxy.set(writeEids[i], frame);
        });
    });

    let _sum = 0;
    summary(() => {
        bench(`proxy read at ${LARGE_N} entities`, () => {
            _sum = 0;
            for (let i = 0; i < writeEids.length; i++) _sum += writeProxy.get(writeEids[i]);
        });
    });

    summary(() => {
        bench(`incremental grow (one entity at a time, ${LARGE_N} total)`, async function* () {
            yield async () => {
                const state = await build({ plugins: [], defaults: false });
                buf(Float32Array, 4, 0);
                buf(Uint32Array, 2, 0);
                buf(Uint8Array, 1, 0);
                buf(Uint16Array, 1, 0);
                for (let i = 0; i < LARGE_N; i++) state.addEntity();
                state.dispose();
            };
        });
    });

    await run();

    if (_sum < 0) console.log("prevent dead code elimination");
    if (capacity() < 0) console.log("never");
}

if (import.meta.main) {
    runCapacityBenchmarks();
}
