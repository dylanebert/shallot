import { parseArgs } from "node:util";
import {
    type ServerWebSocketConnection,
    serveWebSocket,
} from "@dylanebert/shallot/multiplayer/bun";
import { GameServer, openGame } from "./src/network";

const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
        port: { type: "string", default: "3001" },
        host: { type: "string", default: "0.0.0.0" },
    },
});
const port = Number(values.port);
if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw new Error("--port must be an integer in [1, 65535]");
const app = await openGame();
const game = new GameServer(app.world);
const slots = new Map<ServerWebSocketConnection, number>();
const server = serveWebSocket({
    hostname: values.host,
    port,
    connect(connection) {
        if (slots.size >= 2) {
            connection.socket.close(1013, "Two players already connected");
            return;
        }
        slots.set(connection, game.connect(connection));
    },
    disconnect(connection) {
        const slot = slots.get(connection);
        if (slot !== undefined) {
            game.disconnect(slot);
            slots.delete(connection);
        }
    },
});
let previous = performance.now(),
    accumulator = 0;
const timer = setInterval(() => {
    for (const [connection, slot] of slots) {
        if (connection.socket.readyState !== 1) {
            game.disconnect(slot);
            slots.delete(connection);
        }
    }
    const now = performance.now();
    accumulator += Math.min((now - previous) / 1000, 4 / 60);
    previous = now;
    while (accumulator >= 1 / 60) {
        game.tick();
        accumulator -= 1 / 60;
    }
}, 1000 / 60);
process.on("SIGTERM", () => {
    clearInterval(timer);
    server.stop(true);
    app.dispose();
});
console.info(`Multiplayer server: ws://localhost:${server.port} (two players)`);
