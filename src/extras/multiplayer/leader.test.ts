import { expect, test } from "bun:test";
import { serveRooms } from "./bun";
import { type Game, ServerSession } from "./game";
import { LEAVE_CLOSE_CODE } from "./websocket";

test("loopback players agree on the first seat, a held leader, and successors by join order", async () => {
    let now = 0;
    const joined: number[] = [];
    const game: Game = {
        plugins: [],
        schema: [],
        maxClients: 4,
        reconnectMs: 100,
        rules: () => [],
        onJoin(_world, eid) {
            joined.push(eid);
        },
        onSend(world, _eid, connection) {
            const leader = world.resource(ServerSession).server!.leader!;
            connection.send(new Uint8Array(new Uint32Array([leader]).buffer));
        },
        validateIntent: () => false,
        applyIntent() {},
    };
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0, reconnectNow: () => now });
    const sockets: WebSocket[] = [];
    let room: string | undefined;
    async function join(resume = "") {
        const intent = room ? `join&code=${room}` : "joinOrCreate";
        const socket = new WebSocket(
            `ws://127.0.0.1:${host.port}/?announceRoom=1&intent=${intent}${resume}`,
        );
        sockets.push(socket);
        socket.binaryType = "arraybuffer";
        let leader: number | undefined;
        const token = await new Promise<string>((resolve, reject) => {
            socket.addEventListener("close", () => reject(new Error("Seat refused")), {
                once: true,
            });
            socket.addEventListener("message", ({ data }) => {
                if (typeof data === "string") {
                    if (data.startsWith("room:")) room = data.slice(5);
                    else resolve(data);
                } else if (data.byteLength === 4) leader = new DataView(data).getUint32(0, true);
            });
        });
        return {
            get leader() {
                return leader;
            },
            resume: `&seat=${token.split(":")[1]}&resume=${token.split(":")[2]}`,
            close(code?: number) {
                return new Promise<void>((resolve) => {
                    socket.addEventListener("close", () => resolve(), { once: true });
                    socket.close(code);
                });
            },
        };
    }
    async function agree(eid: number, peers: Awaited<ReturnType<typeof join>>[]) {
        for (let i = 0; i < 40 && peers.some((peer) => peer.leader !== eid); i++)
            await Bun.sleep(1);
        expect(peers.map((peer) => peer.leader)).toEqual(peers.map(() => eid));
    }
    try {
        const first = await join();
        const second = await join();
        const third = await join();
        await agree(joined[0]!, [first, second, third]);
        await first.close();
        const fourth = await join();
        await agree(joined[0]!, [second, third, fourth]);
        const resumed = await join(first.resume);
        await agree(joined[0]!, [resumed, second, third, fourth]);
        await resumed.close();
        await expect(join()).rejects.toThrow("Seat refused");
        now = 100;
        const replacement = await join();
        await agree(joined[1]!, [second, third, fourth, replacement]);
        await second.close(LEAVE_CLOSE_CODE);
        await agree(joined[2]!, [third, fourth, replacement]);
    } finally {
        host.stop();
        for (const socket of sockets) socket.close();
    }
});
