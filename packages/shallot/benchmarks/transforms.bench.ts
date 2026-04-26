import { run, bench, summary } from "mitata";
import { build } from "../src/engine";
import { Transform, TransformsPlugin } from "../src/standard/transforms";
import * as wasm from "../src/standard/transforms/wasm";

const ENTITY_COUNT = 50_000;

export async function runTransformsBenchmarks() {
    console.log("\n=== Transform Benchmarks (50k entities) ===\n");

    await wasm.init();

    const state = await build({ plugins: [TransformsPlugin], defaults: false });

    const entities: number[] = [];
    for (let i = 0; i < ENTITY_COUNT; i++) {
        const eid = state.addEntity();
        state.addComponent(eid, Transform);
        entities.push(eid);
    }

    let frame = 0;
    summary(() => {
        bench("local transform update", () => {
            frame++;
            for (const eid of entities) {
                Transform.posX[eid] = Math.sin(frame * 0.01 + eid);
                Transform.posY[eid] = Math.cos(frame * 0.01 + eid);
                Transform.posZ[eid] = frame * 0.001;
            }
        });
    });

    for (const eid of entities) {
        Transform.posX[eid] = eid;
        Transform.posY[eid] = eid * 2;
        Transform.posZ[eid] = eid * 3;
    }
    state.step();

    let _sum = 0;
    summary(() => {
        bench("transform read", () => {
            _sum = 0;
            for (const eid of entities) {
                _sum += Transform.posX[eid];
                _sum += Transform.posY[eid];
                _sum += Transform.posZ[eid];
            }
        });
    });

    summary(() => {
        bench("transform system (full step)", () => {
            frame++;
            for (const eid of entities) {
                Transform.posX[eid] = Math.sin(frame * 0.01 + eid);
                Transform.posY[eid] = Math.cos(frame * 0.01 + eid);
                Transform.rotY[eid] = frame;
            }
            state.step();
        });
    });

    for (let i = 0; i < ENTITY_COUNT; i++) {
        wasm.posX[i] = i;
        wasm.posY[i] = i * 2;
        wasm.posZ[i] = i * 3;
        wasm.quatX[i] = 0;
        wasm.quatY[i] = 0;
        wasm.quatZ[i] = 0;
        wasm.quatW[i] = 1;
        wasm.scaleX[i] = 1;
        wasm.scaleY[i] = 1;
        wasm.scaleZ[i] = 1;
        wasm.indices[i] = i;
        wasm.parents[i] = wasm.NoParent;
    }

    summary(() => {
        bench("wasm compute only", () => {
            wasm.compute(ENTITY_COUNT);
        });
    });

    await run();
}

if (import.meta.main) {
    runTransformsBenchmarks();
}
