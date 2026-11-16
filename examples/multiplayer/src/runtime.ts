import type { World } from "@dylanebert/shallot";
import { client, local } from "@dylanebert/shallot/multiplayer";
import { GAME } from "./network";

export function connectionFailure(url: string) {
    const host = new URL(url).hostname;
    const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(host);
    const remedy = loopback
        ? "From the example directory, start the server with bun server.ts --host 127.0.0.1 --port 3001 (see RUNNING.md), then reload with its address in ?server=."
        : "Check that its server is running and reachable; see the Remote server section in RUNNING.md.";
    return new Error(`No server answered at ${url}. The connection failed or closed. ${remedy}`);
}

export async function topology(url?: string, onError?: (error: Error) => void) {
    if (url)
        return {
            plugin: client(GAME, { url, onError: () => onError?.(connectionFailure(url)) }),
            pump(_delta: number) {},
            dispose() {},
            faults: undefined,
        };
    const loop = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
    ] as const;
    return local(GAME, {
        seed: 0x51a1107,
        faults: { minDelay: 50, maxDelay: 50, loss: 0.02, duplication: 0 },
        thinClients: [
            {
                seed: 812,
                input(_world, sender, tick) {
                    const move = loop[Math.floor(tick / 45) % loop.length]!;
                    sender.record(tick + 12, "input", [...move, 0, 0, 0, 0]);
                },
            },
        ],
    });
}
export type Topology = Awaited<ReturnType<typeof topology>>;
export const Network = { create: (): { topology?: Topology } => ({}) };
export function network(world: World) {
    const topology = world.resource(Network).topology;
    if (!topology) throw new Error("Network is not initialized");
    return topology;
}
