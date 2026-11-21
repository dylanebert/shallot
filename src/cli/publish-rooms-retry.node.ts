import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ROOMS_LIST_HEADER } from "../extras/multiplayer";
import { serveWebSocket } from "../extras/multiplayer/bun";
import { verifyPublishedRooms } from "./publish-verify";

setDefaultTimeout(TIMEOUT.node);

function roomsProbeServer(fetch: (request: Request) => Response | undefined, port = 0) {
    const host = serveWebSocket({
        hostname: "127.0.0.1",
        port,
        fetch(request) {
            return (
                fetch(request) ??
                (request.method === "POST"
                    ? Response.json({ code: "TEST", creator: "test" }, { status: 201 })
                    : Response.json([], { headers: { [ROOMS_LIST_HEADER]: "1" } }))
            );
        },
        connect: (connection) => connection.socket.sendText("room:TEST"),
        disconnect() {},
    });
    return { host, endpoint: `ws://127.0.0.1:${host.port}` };
}

test("retries an unmarked startup 503 before classifying the host", async () => {
    let gets = 0;
    const { host, endpoint } = roomsProbeServer((request) =>
        request.method === "GET" && ++gets === 1 ? new Response(null, { status: 503 }) : undefined,
    );
    try {
        expect(await verifyPublishedRooms(endpoint)).toBe(true);
        expect(gets).toBe(2);
    } finally {
        host.stop(true);
    }
});

test("retries a connection refusal while the server is starting", async () => {
    const { host, endpoint } = roomsProbeServer(() => undefined);
    const port = host.port!;
    await host.stop(true);
    const startup = Bun.sleep(80).then(() => roomsProbeServer(() => undefined, port));
    try {
        expect(await verifyPublishedRooms(endpoint)).toBe(true);
    } finally {
        (await startup).host.stop(true);
    }
});

test.each([1, 60])(
    "respects Retry-After: %i without exceeding the verification budget",
    async (seconds) => {
        const posts: number[] = [];
        const { host, endpoint } = roomsProbeServer((request) => {
            if (request.method !== "POST") return;
            posts.push(performance.now());
            if (posts.length === 1)
                return new Response(null, {
                    status: 503,
                    headers: { "Retry-After": String(seconds) },
                });
        });
        try {
            if (seconds === 1) {
                expect(await verifyPublishedRooms(endpoint, 1600)).toBe(true);
                expect(posts).toHaveLength(2);
                expect(posts[1]! - posts[0]!).toBeGreaterThanOrEqual(990);
            } else {
                await expect(verifyPublishedRooms(endpoint, 100)).rejects.toThrow("Retry-After");
                expect(posts).toHaveLength(1);
            }
        } finally {
            host.stop(true);
        }
    },
);
