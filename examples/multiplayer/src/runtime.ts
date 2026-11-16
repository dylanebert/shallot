import type { World } from "@dylanebert/shallot";
import {
    type Connection,
    IntentSender,
    ReplicationReceiver,
    seededLink,
    WebSocketConnection,
} from "@dylanebert/shallot/multiplayer";
import { GameServer, openGame, SCHEMA } from "./network";

export async function topology(url?: string, onError?: (error: Error) => void) {
    if (url) {
        const host = new URL(url).hostname;
        const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(host);
        const remedy = loopback
            ? "From the example directory, start the server with bun server.ts --host 127.0.0.1 --port 3001 (see RUNNING.md), then reload with its address in ?server=."
            : "Check that its server is running and reachable; see the Remote server section in RUNNING.md.";
        const failure = new Error(
            `No server answered at ${url}. The connection failed or closed. ${remedy}`,
        );
        const socket = new WebSocket(url);
        socket.addEventListener("error", () => onError?.(failure));
        socket.addEventListener("close", () => onError?.(failure));
        await new Promise<void>((resolve, reject) => {
            socket.addEventListener("open", () => resolve(), { once: true });
            socket.addEventListener("error", () => reject(failure), { once: true });
            socket.addEventListener("close", () => reject(failure), { once: true });
        });
        return {
            connection: new WebSocketConnection(socket),
            pump(_delta: number) {},
            dispose() {
                socket.close();
            },
            faults: undefined,
        };
    }
    const serverApp = await openGame();
    const thinApp = await openGame();
    const server = new GameServer(serverApp.world);
    // One-way delay in milliseconds; the two endpoints share these live settings.
    const faults = { minDelay: 50, maxDelay: 50, loss: 0.02, duplication: 0 };
    const link = seededLink(0x51a1107, faults);
    const thinLink = seededLink(812, faults);
    const input = new IntentSender(thinLink.b);
    const receiver = new ReplicationReceiver(thinApp.world, SCHEMA, thinLink.b);
    server.connect(link.a);
    server.connect(thinLink.a);
    let clock = 0,
        accumulator = 0;
    const loop = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
    ] as const;
    return {
        connection: link.b as Connection,
        faults,
        pump(delta: number) {
            clock += delta * 1000;
            link.advance(clock);
            thinLink.advance(clock);
            input.flush(receiver.flush());
            accumulator += Math.min(delta, 4 / 60);
            while (accumulator >= 1 / 60) {
                const tick = server.world.time.fixedTick;
                const move = loop[Math.floor(tick / 45) % loop.length]!;
                // This thin client only receives state and sends a recorded input loop; it draws and predicts nothing.
                input.record(tick + 12, "input", [...move, 0, 0, 0, 0]);
                input.flush();
                server.tick();
                accumulator -= 1 / 60;
            }
        },
        dispose() {
            serverApp.dispose();
            thinApp.dispose();
        },
    };
}
export type Topology = Awaited<ReturnType<typeof topology>>;
export const Network = { create: (): { topology?: Topology } => ({}) };
export function network(world: World) {
    const topology = world.resource(Network).topology;
    if (!topology) throw new Error("Network is not initialized");
    return topology;
}
