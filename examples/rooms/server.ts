import { parseArgs } from "node:util";
import { serveRooms } from "@dylanebert/shallot/multiplayer/bun";
import { GAME } from "./src/network";

const { values } = parseArgs({
    options: {
        host: { type: "string", default: "127.0.0.1" },
        port: { type: "string", default: "3001" },
        origin: { type: "string", default: "*" },
    },
});
const port = Number(values.port);
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error("Invalid port");
const host = serveRooms(GAME, { hostname: values.host, port, roomsOrigin: values.origin });
console.log(`Rooms server listening on port ${host.port}`);
