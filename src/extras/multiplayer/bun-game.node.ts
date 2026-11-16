import { expect, spyOn, test } from "bun:test";
import { type Game, ServerSession } from ".";
import { type ServerWebSocketConnection, serveGame } from "./bun";

interface Socket {
    data: ServerWebSocketConnection | undefined;
    readyState: number;
    sendBinary(): number;
    close(code?: number): void;
}
const game: Game = {
    plugins: [],
    schema: [],
    maxClients: 1,
    rules: () => [],
    validateIntent: () => true,
    applyIntent() {},
};

test("a delayed close after the dead-socket sweep cannot disconnect a recycled entity's new peer", async () => {
    let hooks!: { open(socket: Socket): void; close(socket: Socket): void };
    const endpoint = { port: 0, stop() {} };
    const serve = spyOn(Bun, "serve").mockImplementation((options) => {
        hooks = (options as unknown as { websocket: typeof hooks }).websocket;
        return endpoint as ReturnType<typeof Bun.serve>;
    });
    const listeners = process.listenerCount("SIGTERM");
    const host = await serveGame(game, { port: 0 });
    try {
        const runtime = host.world.resource(ServerSession).server!;
        const socket = (): Socket => ({
            data: undefined,
            readyState: 1,
            sendBinary: () => 1,
            close() {},
        });
        const old = socket();
        hooks.open(old);
        const eid = [...runtime.peers.keys()][0]!;
        old.readyState = 3;
        host.world.tick();
        expect(runtime.peers.size).toBe(0);
        const replacement = socket();
        hooks.open(replacement);
        expect([...runtime.peers.keys()]).toEqual([eid]);
        hooks.close(old);
        expect(runtime.peers.get(eid)!.connection).toBe(replacement.data!);
        const refused = socket();
        const close = spyOn(refused, "close");
        hooks.open(refused);
        expect(close).toHaveBeenCalledWith(1013, "Peer limit reached");
        expect(runtime.peers.size).toBe(1);
        host.stop();
        host.stop();
        expect(runtime.peers.size).toBe(0);
        expect(process.listenerCount("SIGTERM")).toBe(listeners);
    } finally {
        host.stop();
        serve.mockRestore();
    }
});
