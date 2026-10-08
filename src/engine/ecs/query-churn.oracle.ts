import { component, f32 } from "./component";
import { World } from "./world";

const C = component("query-churn", { value: f32 });
const entities = 10_000;
const warmups = 5;
const samples = 21;

function run() {
    const world = new World();
    const query = world.query([C]);
    const ids: number[] = [];
    let spawned = 0;
    let despawned = 0;
    let visited = 0;
    const start = performance.now();
    for (let i = 0; i < entities; i++) {
        const eid = world.create();
        world.add(eid, C);
        ids.push(eid);
        spawned++;
    }
    for (const _ of query) visited++;
    for (const eid of ids) {
        world.destroy(eid);
        despawned++;
    }
    for (const _ of query) visited++;
    for (let i = 0; i < entities; i++) {
        world.add(world.create(), C);
        spawned++;
    }
    for (const _ of query) visited++;
    for (const eid of query) {
        visited++;
        world.destroy(eid);
        despawned++;
    }
    for (const _ of query) visited++;
    const ms = performance.now() - start;
    world.dispose();
    if (spawned !== entities * 2 || despawned !== entities * 2 || visited !== entities * 3)
        throw new Error(`query churn counts diverged: ${spawned}/${despawned}/${visited}`);
    return ms;
}

for (let i = 0; i < warmups; i++) run();
const times = Array.from({ length: samples }, run).sort((a, b) => a - b);
console.log(
    JSON.stringify({
        runtime: Bun.version,
        entities,
        warmups,
        samples,
        spawned: entities * 2,
        despawned: entities * 2,
        visited: entities * 3,
        medianMs: times[Math.floor(samples / 2)],
        minMs: times[0],
        maxMs: times.at(-1),
    }),
);
