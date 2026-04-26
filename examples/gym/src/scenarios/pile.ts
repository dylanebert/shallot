import { PhysicsPlugin, RenderPlugin } from "@dylanebert/shallot";
import type { Plugin, State } from "@dylanebert/shallot";
import { Benchmark, BenchmarkState } from "../config";
import type { PileShape } from "./arena";
import {
    StepCounterSystem,
    PileRampSystem,
    initArena,
    pileSpread,
    pileDrop,
    spawnPileBody,
    initPileState,
} from "./arena";

export function buildPhysicsScenarioPlugin(
    count: number,
    ramp: boolean,
    shapes: PileShape[] = [0],
    heightOffset = 0,
): Plugin {
    return {
        name: "PhysicsScenario",
        dependencies: [PhysicsPlugin, RenderPlugin],
        systems: ramp ? [StepCounterSystem, PileRampSystem] : [StepCounterSystem],

        initialize(state: State) {
            const spread = pileSpread(count);
            const drop = pileDrop(count) + heightOffset;
            const arena = initArena(state, spread, drop);

            let seed = 54321;
            const spawned: number[] = [];
            for (let i = 0; i < count; i++) {
                const [eid, newSeed] = spawnPileBody(state, i, spread, drop, seed, shapes);
                seed = newSeed;
                spawned.push(eid);
            }

            initPileState({
                spawned,
                seed,
                arena,
                currentSpread: spread,
                shapes,
                count,
                ecs: state,
            });

            const bench = state.addEntity();
            state.addComponent(bench, Benchmark);
            Benchmark.count[bench] = count;

            const benchState = state.getResource(BenchmarkState);
            if (benchState) benchState.externalSpawner = true;
        },
    };
}
