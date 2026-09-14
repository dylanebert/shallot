import { expect } from "bun:test";
import { check } from "@dylanebert/shallot/harness/check";
import { createPool, maxWorkers, type Pool, type WorkerReady } from "@dylanebert/shallot/physics";

function poolSize(pool: Pool): number {
    return pool.size;
}

function workerIndex(worker: WorkerReady): number {
    return worker.index;
}

check(
    "physics: pool helpers are public",
    {
        claim: "the physics pool helpers and worker-ready types compile through the public physics export",
    },
    () => {
        expect(typeof createPool).toBe("function");
        expect(typeof maxWorkers).toBe("function");
        expect(poolSize).toBeDefined();
        expect(workerIndex).toBeDefined();
    },
);
