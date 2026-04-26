import { run, bench, summary } from "mitata";
import { build, events } from "../src";

const SMALL = 1_000;
const LARGE = 10_000;
const HUGE = 100_000;

interface PlaceEvent {
    tile: number;
    archetype: number;
}

export async function runEventsBenchmarks() {
    console.log("\n=== Events Benchmarks ===\n");

    const state = await build({ plugins: [], defaults: false });

    const Place = events<PlaceEvent>("place");
    const Damage = events<{ amount: number }>("damage");
    const Tick = events<number>("tick");

    summary(() => {
        bench(`send ${SMALL} events`, () => {
            for (let i = 0; i < SMALL; i++) {
                Place.send(state, { tile: i, archetype: 1 });
            }
            state.step(1 / 60);
        });
        bench(`send ${LARGE} events`, () => {
            for (let i = 0; i < LARGE; i++) {
                Place.send(state, { tile: i, archetype: 1 });
            }
            state.step(1 / 60);
        });
        bench(`send ${HUGE} events`, () => {
            for (let i = 0; i < HUGE; i++) {
                Place.send(state, { tile: i, archetype: 1 });
            }
            state.step(1 / 60);
        });
    });

    let _sum = 0;
    summary(() => {
        bench(`read+iterate ${LARGE} events`, () => {
            for (let i = 0; i < LARGE; i++) {
                Place.send(state, { tile: i, archetype: 1 });
            }
            _sum = 0;
            for (const ev of Place.read(state)) {
                _sum += ev.tile + ev.archetype;
            }
            state.step(1 / 60);
        });
    });

    summary(() => {
        bench(`drain (step) after ${LARGE} events`, () => {
            for (let i = 0; i < LARGE; i++) Tick.send(state, i);
            state.step(1 / 60);
        });
    });

    const channels: ReturnType<typeof events<number>>[] = [];
    for (let i = 0; i < 100; i++) channels.push(events<number>(`ch${i}`));
    summary(() => {
        bench("100 channels × 100 events each (write + drain)", () => {
            for (const ch of channels) {
                for (let i = 0; i < 100; i++) ch.send(state, i);
            }
            state.step(1 / 60);
        });
    });

    summary(() => {
        bench("send 10 events × 1000 frames (with state.step drain)", () => {
            for (let f = 0; f < 1000; f++) {
                for (let i = 0; i < 10; i++) Damage.send(state, { amount: i });
                state.step(1 / 60);
            }
        });
    });

    await run();
}

if (import.meta.main) {
    runEventsBenchmarks();
}
