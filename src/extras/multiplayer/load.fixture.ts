import { component, f32, type Plugin } from "../../engine";
import { type Game, Replicate } from ".";

export const LoadRow = component("LoadRow", { x: f32 });
export const loadPlugin: Plugin = {
    name: "LoadGame",
    components: [LoadRow],
    systems: [
        {
            group: "fixed",
            update(world) {
                for (const eid of world.query([LoadRow]))
                    world.storage(LoadRow).x.set(eid, world.time.fixedTick);
            },
        },
    ],
};
export function loadGame(peers: number): Game {
    return {
        plugins: [loadPlugin],
        schema: [LoadRow],
        maxClients: peers,
        onCreate(world) {
            for (let i = 0; i < 16; i++) {
                const eid = world.create();
                world.add(eid, LoadRow);
                world.add(eid, Replicate);
            }
        },
        rules: (_world, recipient) => [new Map([[recipient, [LoadRow]]])],
        validateIntent: () => true,
        applyIntent() {},
    };
}

// Oracle-only inspection keeps implementation counts out of the game's public API.
export function collectionSize(object: object, key: string): number {
    const value = (object as Record<string, unknown>)[key];
    if (value instanceof Map || value instanceof Set) return value.size;
    if (Array.isArray(value)) return value.length;
    throw new Error(`Missing measured collection ${key}`);
}
export function counters(object: object) {
    const value = object as { _bytesSent?: number; _bytesReceived?: number };
    if (value._bytesSent === undefined || value._bytesReceived === undefined)
        throw new Error("Missing byte counters");
    return { sent: value._bytesSent, received: value._bytesReceived };
}
export function percentiles(samples: number[]) {
    samples.sort((a, b) => a - b);
    return {
        p50: samples[Math.floor(samples.length * 0.5)] ?? 0,
        p99: samples[Math.min(samples.length - 1, Math.floor(samples.length * 0.99))] ?? 0,
    };
}
