import { run, bench, summary } from "mitata";
import { build } from "../src/engine";

const ENTITY_COUNT = 50_000;

const Position = { x: [] as number[], y: [] as number[], z: [] as number[] };
const Velocity = { x: [] as number[], y: [] as number[], z: [] as number[] };
const Health = { value: [] as number[], max: [] as number[] };

export async function runCoreBenchmarks() {
    console.log("\n=== Core Benchmarks (50k entities) ===\n");

    const baseState = await build({ plugins: [], defaults: false });

    summary(() => {
        bench("entity creation + removal", () => {
            const entities: number[] = [];
            for (let i = 0; i < ENTITY_COUNT; i++) {
                entities.push(baseState.addEntity());
            }
            for (const eid of entities) {
                baseState.removeEntity(eid);
            }
        });
    });

    const stateWithEntities = await build({ plugins: [], defaults: false });
    const entitiesForComponent: number[] = [];
    for (let i = 0; i < ENTITY_COUNT; i++) {
        entitiesForComponent.push(stateWithEntities.addEntity());
    }

    let toggle = false;
    summary(() => {
        bench("component add/remove", () => {
            if (toggle) {
                for (const eid of entitiesForComponent) {
                    stateWithEntities.removeComponent(eid, Position);
                }
            } else {
                for (const eid of entitiesForComponent) {
                    stateWithEntities.addComponent(eid, Position);
                }
            }
            toggle = !toggle;
        });
    });

    const stateForQuery1 = await build({ plugins: [], defaults: false });
    for (let i = 0; i < ENTITY_COUNT; i++) {
        const eid = stateForQuery1.addEntity();
        stateForQuery1.addComponent(eid, Position);
        Position.x[eid] = i;
        Position.y[eid] = i * 2;
        Position.z[eid] = i * 3;
    }

    let _sum = 0;
    summary(() => {
        bench("query 1 component", () => {
            _sum = 0;
            for (const eid of stateForQuery1.query([Position])) {
                _sum += Position.x[eid] + Position.y[eid] + Position.z[eid];
            }
        });
    });

    const stateForQuery3 = await build({ plugins: [], defaults: false });
    for (let i = 0; i < ENTITY_COUNT; i++) {
        const eid = stateForQuery3.addEntity();
        stateForQuery3.addComponent(eid, Position);
        stateForQuery3.addComponent(eid, Velocity);
        stateForQuery3.addComponent(eid, Health);
        Position.x[eid] = i;
        Velocity.x[eid] = i * 0.1;
        Health.value[eid] = 100;
    }

    summary(() => {
        bench("query 3 components", () => {
            _sum = 0;
            for (const eid of stateForQuery3.query([Position, Velocity, Health])) {
                _sum += Position.x[eid] + Velocity.x[eid] + Health.value[eid];
            }
        });
    });

    const stateForWrite = await build({ plugins: [], defaults: false });
    const entitiesForWrite: number[] = [];
    for (let i = 0; i < ENTITY_COUNT; i++) {
        const eid = stateForWrite.addEntity();
        stateForWrite.addComponent(eid, Position);
        entitiesForWrite.push(eid);
    }

    let frame = 0;
    summary(() => {
        bench("component write", () => {
            frame++;
            for (const eid of entitiesForWrite) {
                Position.x[eid] = frame;
                Position.y[eid] = frame * 2;
                Position.z[eid] = frame * 3;
            }
        });
    });

    await run();
}

if (import.meta.main) {
    runCoreBenchmarks();
}
