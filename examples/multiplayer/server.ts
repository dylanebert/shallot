import { parseArgs } from "node:util";
import { serveGame } from "@dylanebert/shallot/multiplayer/bun";
import { GAME } from "./src/network";

const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
        port: { type: "string", default: process.env.PORT ?? "3001" },
        host: { type: "string", default: "0.0.0.0" },
    },
});
const port = Number(values.port);
// Port 0 takes a free port; the printed address names it.
if (!Number.isInteger(port) || port < 0 || port > 65535)
    throw new Error("--port must be an integer in [0, 65535]");
const server = await serveGame(GAME, { hostname: values.host, port });
console.info(`Multiplayer server: ws://localhost:${server.port} (two players)`);
