import { actions, composition, trappedTick } from "./composition.fixture";

export async function replay(workerCount: number) {
    if (!crossOriginIsolated) throw new Error("Replay requires cross-origin isolation");
    const app = await composition(workerCount);
    try {
        const actualArtifact = app.artifact;
        if (actualArtifact !== "shared")
            throw new Error(`Expected shared artifact, got ${actualArtifact}`);
        const actualWorkers = app.workers;
        if (actualWorkers !== workerCount)
            throw new Error(`Requested ${workerCount} workers, got ${actualWorkers}`);
        for (const _ of actions) trappedTick(app.world);
        return {
            hash: app.hash().toString(),
            workers: actualWorkers,
            artifact: actualArtifact,
            isolated: crossOriginIsolated,
            ticks: app.world.time.fixedTick,
        };
    } finally {
        app.dispose();
    }
}
