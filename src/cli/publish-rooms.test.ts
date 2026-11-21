import { expect, test } from "bun:test";
import { ROOMS_LIST_HEADER } from "../extras/multiplayer";
import { serveGame, serveRooms, serveWebSocket } from "../extras/multiplayer/bun";
import type { Game } from "../extras/multiplayer/game";
import { verifyPublishedRooms } from "./publish-verify";

const local = { hostname: "127.0.0.1", port: 0 };
const game: Game = {
    plugins: [],
    schema: [],
    maxClients: 1,
    rules: () => [],
    validateIntent: () => false,
    applyIntent() {},
};
const endpoint = ({ port }: { port: number | undefined }) => `ws://${local.hostname}:${port}`;

test.each(["admit", "refuse", "throw"] as const)(
    "rooms verification distinguishes admission, explicit refusal and hook failure: %s",
    async (outcome) => {
        const credentials: (string | null)[] = [];
        let joins = 0;
        const host = serveRooms(
            {
                ...game,
                onAuth:
                    outcome === "admit"
                        ? undefined
                        : async (credential) => {
                              credentials.push(credential);
                              if (outcome === "throw") throw new Error("Internal failure");
                              return { reason: "Unknown room or invalid intent" };
                          },
                onJoin: () => {
                    joins++;
                },
            },
            local,
        );
        try {
            const verification = verifyPublishedRooms(endpoint(host));
            if (outcome === "throw") await expect(verification).rejects.toThrow("before admission");
            else expect(await verification).toBe(true);
            expect(credentials).toEqual(outcome === "admit" ? [] : [null]);
            expect(joins).toBe(outcome === "admit" ? 1 : 0);
        } finally {
            host.stop();
        }
    },
);

test.each([
    [[], "before admission (code 1008)"],
    [{}, "GET /rooms did not return a room listing"],
] as const)(
    "rejects a pre-admission close or a malformed room listing: %j",
    async (listing, error) => {
        const host = serveWebSocket({
            ...local,
            fetch: (request) =>
                request.method === "POST"
                    ? Response.json({ code: "TEST", creator: "test" }, { status: 201 })
                    : Response.json(listing, { headers: { [ROOMS_LIST_HEADER]: "1" } }),
            connect: (connection) => connection.close(1008, "Unknown room or invalid intent"),
            disconnect() {},
        });
        try {
            await expect(verifyPublishedRooms(endpoint(host))).rejects.toThrow(error);
        } finally {
            host.stop(true);
        }
    },
);

test("a single-room host keeps page verification, but an unreachable host cannot verify", async () => {
    const host = await serveGame(game, { ...local, fetch: () => Response.json([]) });
    const url = endpoint(host);
    try {
        expect(await verifyPublishedRooms(url)).toBe(false);
    } finally {
        host.stop();
    }
    await expect(verifyPublishedRooms(url)).rejects.toThrow("Server verification failed");
});
