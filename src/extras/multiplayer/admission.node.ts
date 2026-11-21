import { expect, test } from "bun:test";
import { type App, createApp, type World } from "../../engine";
import { type ClientId, client, type Game, type PlayerId, ServerSession } from ".";
import { serveRooms } from "./bun";
import { MAX_MESSAGE_BYTES } from "./link";

async function until(ready: () => boolean, reason: string) {
    const deadline = performance.now() + 600;
    while (!ready()) {
        if (performance.now() >= deadline) throw new Error(reason);
        await Bun.sleep(1);
    }
}

test("auth refuses before publication, reserves only its seat, and preserves identity on resume", async () => {
    const joined: { world: World; eid: number; clientId: ClientId; playerId?: PlayerId }[] = [];
    const credentials: (string | null)[] = [];
    const slow = Promise.withResolvers<{ playerId: PlayerId }>();
    const game: Game = {
        plugins: [],
        schema: [],
        maxClients: 2,
        reconnectMs: 100,
        rules: () => [],
        validateIntent: () => false,
        applyIntent() {},
        async onAuth(credential) {
            credentials.push(credential);
            if (credential === "slow + 🐟") return slow.promise;
            if (credential === "primary" || credential === "elsewhere")
                return { playerId: credential };
            return { reason: "Pass expired" };
        },
        onJoin(world, eid, _transport, clientId, playerId) {
            joined.push({ world, eid, clientId, playerId });
        },
    };
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0, reconnectNow: () => 0 });
    const base = `127.0.0.1:${host.port}`;
    const sockets: WebSocket[] = [];
    function open(query: Record<string, string>) {
        const socket = new WebSocket(`ws://${base}/?${new URLSearchParams(query)}`);
        sockets.push(socket);
        const text: string[] = [];
        socket.addEventListener("message", ({ data }) => {
            if (typeof data === "string") text.push(data);
        });
        const closed = new Promise<CloseEvent>((resolve) =>
            socket.addEventListener("close", resolve, { once: true }),
        );
        return { socket, text, closed };
    }
    let app: App | undefined;
    try {
        const errors: Error[] = [];
        app = await createApp({
            defaults: false,
            plugins: [
                client(game, {
                    url: `ws://${base}/?intent=joinOrCreate`,
                    credential: "private-expired-pass",
                    onError: (error) => errors.push(error),
                }),
            ],
        });
        await until(() => errors.length > 0, "refusal did not reach onError");
        expect(errors[0]!.message).toContain("Pass expired");
        expect(errors[0]!.message).not.toContain("private-expired-pass");
        expect(joined).toHaveLength(0);
        expect(await (await fetch(`http://${base}/rooms`)).json()).toEqual([]);

        const oversized = open({
            intent: "joinOrCreate",
            credential: "é".repeat(MAX_MESSAGE_BYTES / 2 + 1),
        });
        expect((await oversized.closed).reason).toBe("Invalid credential");
        expect(credentials).toEqual(["private-expired-pass"]);

        const primary = open({ intent: "joinOrCreate", credential: "primary", announceRoom: "1" });
        await until(
            () => primary.text.some((text) => text.startsWith("room:")),
            "primary did not join",
        );
        const code = primary.text.find((text) => text.startsWith("room:"))!.slice(5);
        const waiting = open({ intent: "join", code, credential: "slow + 🐟", announceRoom: "1" });
        await until(() => credentials.includes("slow + 🐟"), "slow auth did not start");
        const refused = open({ intent: "join", code, credential: "extra" });
        expect((await refused.closed).reason).toMatch(/full/i);
        expect(credentials).not.toContain("extra");
        expect(waiting.text).toEqual([]);
        expect(await (await fetch(`http://${base}/rooms`)).json()).toEqual([
            { code, clients: 1, maxClients: 2 },
        ]);

        const firstWorld = joined[0]!.world;
        const tick = firstWorld.time.fixedTick;
        open({ intent: "joinOrCreate", credential: "elsewhere" });
        await until(() => joined.length === 2, "another room could not join");
        const otherWorld = joined[1]!.world;
        expect(otherWorld).not.toBe(firstWorld);
        await until(
            () => firstWorld.time.fixedTick > tick && otherWorld.time.fixedTick > 0,
            "a room stopped ticking during auth",
        );
        expect(joined).toHaveLength(2);

        slow.resolve({ playerId: "durable/angler" });
        await until(
            () => waiting.text.some((text) => text.startsWith("resume:")),
            "slow seat was not admitted",
        );
        const [, seat, resume] = waiting.text
            .find((text) => text.startsWith("resume:"))!
            .split(":");
        const admitted = joined[2]!;
        expect(admitted.world).toBe(firstWorld);
        expect(admitted.playerId).toBe("durable/angler");
        const authority = firstWorld.resource(ServerSession).server!;
        waiting.socket.close();
        await until(() => !!authority.peers.get(admitted.eid)?.held, "seat was not held");
        const resumed = open({ intent: "join", code, seat: seat!, resume: resume! });
        await until(
            () => resumed.text.some((text) => text.startsWith("resume:")),
            "seat did not resume",
        );
        expect(authority.peers.get(admitted.eid)).toMatchObject({
            held: false,
            clientId: admitted.clientId,
            playerId: admitted.playerId,
        });
        expect(joined).toHaveLength(3);
        expect(credentials).toEqual(["private-expired-pass", "primary", "slow + 🐟", "elsewhere"]);
    } finally {
        app?.dispose();
        for (const socket of sockets) socket.close();
        host.stop();
    }
});
