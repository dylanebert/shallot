import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { serveRooms } from "@dylanebert/shallot/multiplayer/bun";
import { build } from "vite";
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

const site = mkdtempSync(join(tmpdir(), "shallot-rooms-"));
process.once("exit", () => rmSync(site, { recursive: true, force: true }));
await build({
    root: import.meta.dirname,
    define: { ROOMS_SAME_ORIGIN: "true" },
    build: { outDir: site },
    logLevel: "error",
});
const files = new Map(
    Array.from(new Bun.Glob("**/*").scanSync({ cwd: site }), (path) => [
        `/${path}`,
        Bun.file(join(site, path)),
    ]),
);
const host = serveRooms(GAME, {
    hostname: values.host,
    port,
    roomsOrigin: values.origin,
    fetch(request) {
        if (request.method !== "GET" && request.method !== "HEAD") return;
        const path = new URL(request.url).pathname;
        const file = files.get(path === "/" ? "/index.html" : path);
        return file
            ? new Response(request.method === "HEAD" ? null : file, {
                  headers: {
                      "Content-Type": file.type,
                      "Cross-Origin-Opener-Policy": "same-origin",
                      "Cross-Origin-Embedder-Policy": "require-corp",
                  },
              })
            : new Response(null, { status: 404 });
    },
});
console.log(`Rooms server listening on port ${host.port}`);
const address = values.host.includes(":") ? `[${values.host}]` : values.host;
console.log(`Open http://${address}:${host.port}/ in each tab.`);
