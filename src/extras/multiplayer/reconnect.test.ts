import { expect, test } from "bun:test";
import { component, createApp, u32, type World } from "../../engine";
import { ClientPrediction, IntentSender, PredictionPlugin, Replicate } from ".";
import { serveRooms, serveWebSocket } from "./bun";
import { ClientOf, client, type Game } from "./game";
import { createProxy } from "./proxy";
import { replaySession, type SessionRecording } from "./recording";

const Player = component("ReconnectPlayer", { value: u32, input: u32 });

const apply: Game["applyIntent"] = (world, eid, tick, records, input) => {
    const row = world.storage(Player);
    if (input) row.input.set(eid, input.values[0]!);
    for (const record of records) {
        if (record.tick === tick) row.value.set(eid, row.value.get(eid) + record.values[0]!);
    }
};

function fixture() {
    let now = 0;
    let world!: World;
    let eid!: number;
    const joined: number[] = [];
    const left: number[] = [];
    const game: Game = {
        plugins: [{ name: "ReconnectPlayer", components: [Player], recovery: "stateless" }],
        schema: [Player],
        maxClients: 2,
        reconnectMs: 100,
        rules: (_world, eid) => [new Map([[eid, [Player]]])],
        onCreate(w) {
            world = w;
        },
        onJoin(w, id) {
            eid = id;
            joined.push(id);
            w.add(id, Player, { value: 42 });
            w.add(id, Replicate, { rule: 0 });
        },
        onLeave(w, id) {
            left.push(w.get(id, Player)!.value);
        },
        validateIntent: () => false,
        applyIntent() {},
    };
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0, reconnectNow: () => now });
    return {
        game,
        host,
        joined,
        left,
        get world() {
            return world;
        },
        get eid() {
            return eid;
        },
        advance(ms: number) {
            now += ms;
        },
        url: `ws://127.0.0.1:${host.port}/?intent=joinOrCreate`,
    };
}

async function until(assertion: () => boolean, message: string) {
    for (let i = 0; i < 100; i++) {
        if (assertion()) return;
        await Bun.sleep(1);
    }
    throw new Error(message);
}

function peer(url: string) {
    const socket = new WebSocket(url);
    const token = new Promise<string>((resolve) => {
        socket.addEventListener("message", ({ data }) => {
            if (typeof data === "string") resolve(data);
        });
    });
    const closed = new Promise<number>((resolve) => {
        socket.addEventListener("close", ({ code }) => resolve(code), { once: true });
    });
    return { socket, token, closed, outcome: Promise.race([token, closed]) };
}

test("disposing a connected client leaves immediately, without advancing the reconnect clock", async () => {
    const f = fixture();
    try {
        const app = await createApp({ defaults: false, plugins: [client(f.game, { url: f.url })] });
        try {
            await until(() => f.joined.length === 1, "client was not admitted");
        } finally {
            app.dispose();
        }
        await until(() => f.left.length === 1, "explicit leave held the seat");
    } finally {
        f.host.stop();
    }
});

test("held seats count toward capacity, reject another seat's token, and expire exactly once", async () => {
    const f = fixture();
    try {
        const first = peer(f.url);
        const [, seat, token] = (await first.token).split(":");
        const second = peer(f.url);
        const [, otherSeat, otherToken] = (await second.token).split(":");
        expect(await peer(`${f.url}&resume=${token}&seat=${seat}`).outcome).toBe(1008);
        first.socket.close();
        second.socket.close();
        await Promise.all([first.closed, second.closed]);
        const listing = `http://127.0.0.1:${f.host.port}/rooms`;
        const rooms = async () =>
            (await (await fetch(listing)).json()) as { code: string; clients: number }[];
        const [room] = await rooms();
        expect(room?.clients).toBe(2);
        expect(await peer(`${f.url}&resume=${otherToken}&seat=${seat}`).outcome).toBe(1008);
        expect(await peer(`${f.url}&resume=${token}&seat=${otherSeat}`).outcome).toBe(1008);
        expect(
            await peer(`${f.url.replace("joinOrCreate", "join")}&code=${room?.code}`).closed,
        ).toBe(1013);
        f.advance(99);
        expect(f.left).toEqual([]);
        f.advance(1);
        await until(() => f.left.length === 2, "the clock did not release held seats");
        expect(f.left).toEqual([42, 42]);
        expect(await peer(`${f.url}&resume=${token}&seat=${seat}`).outcome).toBe(1008);
        expect(await rooms()).toEqual([]);
    } finally {
        f.host.stop();
    }
});

test("a held seat spends its token on one concurrent resume and issues a new one", async () => {
    const f = fixture();
    try {
        const original = peer(f.url);
        const [, seat, token] = (await original.token).split(":");
        original.socket.close();
        await original.closed;
        const resume = `${f.url}&seat=${seat}&resume=`;
        const contenders = [peer(resume + token), peer(resume + token)];
        const outcomes = await Promise.all(contenders.map((p) => p.outcome));
        expect(outcomes.filter((outcome) => outcome === 1008)).toHaveLength(1);
        const winner = outcomes.findIndex((outcome) => typeof outcome === "string");
        expect(winner).toBeGreaterThanOrEqual(0);
        const [, sameSeat, nextToken] = (outcomes[winner] as string).split(":");
        expect(sameSeat).toBe(seat);
        contenders[winner]!.socket.close();
        await contenders[winner]!.closed;
        expect(await peer(resume + token).outcome).toBe(1008);
        expect(await peer(resume + nextToken).outcome).toStartWith(`resume:${seat}:`);
        expect(f.joined).toHaveLength(1);
    } finally {
        f.host.stop();
    }
});

test("a resume credential cannot override the requested room code", async () => {
    const f = fixture();
    try {
        const join = async () => {
            const response = await fetch(`http://127.0.0.1:${f.host.port}/rooms`, {
                method: "POST",
            });
            const { code, creator } = (await response.json()) as { code: string; creator: string };
            return `${f.url.replace("joinOrCreate", "join")}&code=${code.toLowerCase()}&creator=${creator}`;
        };
        const firstRoom = await join();
        const secondRoom = await join();
        const first = peer(firstRoom);
        const [, seat, token] = (await first.token).split(":");
        await peer(secondRoom).token;
        first.socket.close();
        await first.closed;
        const credential = `&seat=${seat}&resume=${token}`;
        expect(await peer(secondRoom + credential).outcome).toBe(1008);
        expect(await peer(firstRoom + credential).outcome).toStartWith(`resume:${seat}:`);
        expect(f.joined).toHaveLength(2);
    } finally {
        f.host.stop();
    }
});

test("a client without reconnection rejects resume text as non-binary", async () => {
    const { game, host: rooms } = fixture();
    rooms.stop();
    delete game.reconnectMs;
    let closed: number | undefined;
    const host = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch(request, server) {
            if (server.upgrade(request)) return;
            return new Response(null, { status: 426 });
        },
        websocket: {
            message() {},
            open(socket) {
                socket.sendText(`resume:${crypto.randomUUID()}:${crypto.randomUUID()}`);
            },
            close(_socket, code) {
                closed = code;
            },
        },
    });
    try {
        const app = await createApp({
            defaults: false,
            plugins: [
                client(game, {
                    url: `ws://127.0.0.1:${host.port}`,
                }),
            ],
        });
        try {
            await until(() => closed !== undefined, "client accepted a text frame");
            expect(closed).toBe(1003);
        } finally {
            app.dispose();
        }
    } finally {
        host.stop(true);
    }
});

test("client backs off after a TCP cut and its running prediction confirms the same seat at a fresh image", async () => {
    const f = fixture();
    const proxy = await createProxy(f.host.port!, 1);
    let prediction!: ClientPrediction;
    let sender!: IntentSender;
    let setups = 0;
    let retry: (() => void) | undefined;
    const delays: number[] = [];
    const errors: Error[] = [];
    let recording!: SessionRecording;
    f.game.plugins = [...f.game.plugins, PredictionPlugin];
    f.game.applyIntent = apply;
    f.game.validateIntent = () => true;
    f.game.onClient = (world, connection) => {
        setups++;
        sender = new IntentSender(connection);
        prediction = new ClientPrediction(
            world,
            f.game.schema,
            connection,
            (_world, entities) => entities.get(f.eid),
            (world, tick, records, input) => {
                const avatar = prediction.receiver.entities.get(f.eid);
                if (avatar !== undefined) apply(world, avatar, tick, records, input);
            },
        );
    };
    const app = await createApp({
        defaults: false,
        plugins: [
            client(f.game, {
                url: f.url.replace(String(f.host.port), String(proxy.port)),
                onError: (error) => errors.push(error),
                recording: {
                    gameVersion: "test",
                    write(value) {
                        recording = value;
                    },
                },
                retryAfter(delay, run) {
                    delays.push(delay);
                    retry = run;
                    return () => {
                        retry = undefined;
                    };
                },
            }),
        ],
    });
    try {
        await until(() => {
            app.world.step(1 / 60);
            return prediction?.receiver.entities.has(f.eid);
        }, "initial image was not confirmed");
        const receiver = prediction.receiver;
        const eid = f.eid;
        const local = receiver.entities.get(eid)!;
        const submit = (operation: number, input: number, expected: number) => {
            const tick = app.world.time.fixedTick + 1;
            prediction.accept([
                sender.record(tick, "operation", [operation]),
                sender.record(tick, "input", [input]),
            ]);
            app.world.tick();
            expect(app.world.get(local, Player)).toEqual({ value: expected, input });
            sender.flush();
        };
        submit(7, 3, 49);
        await until(() => {
            f.world.tick();
            return f.world.get(eid, Player)?.value === 49;
        }, "first operation was not applied");
        expect(f.world.get(eid, Player)?.input).toBe(3);
        const confirmed = receiver.confirmed;
        expect(app.world.time.fixedTick).toBeGreaterThan(confirmed);
        proxy.controls.resetPeer();
        await until(() => !!retry, "close did not schedule retry");
        expect(delays).toEqual([50]);
        expect(f.left).toEqual([]);
        expect(f.world.has(eid, ClientOf)).toBe(true);
        for (let i = 0; i < 12; i++) f.world.tick();
        f.advance(50);
        retry!();
        await until(() => {
            app.world.step(1 / 60);
            return receiver.confirmed > confirmed + 1;
        }, "replacement image was not confirmed by the running prediction");
        expect(prediction.receiver).toBe(receiver);
        expect(receiver.entities.get(eid)).toBe(local);
        expect(prediction.predicted(local)).toBe(true);
        expect(app.world.get(local, Player)).toEqual({ value: 49, input: 3 });
        // Never fed the first acknowledgement to the sender: its first operation repeats
        // on the replacement socket along with the new operation and sampled input.
        expect(sender.records.size).toBe(1);
        submit(11, 5, 60);
        await until(() => {
            f.world.tick();
            const row = f.world.get(eid, Player);
            return row?.input === 5 && row.value >= 60;
        }, "replacement input and operation were not applied");
        expect(f.world.get(eid, Player)?.value).toBe(60);
        const applied = f.world.time.fixedTick;
        await until(() => {
            app.world.step(1 / 60);
            return receiver.confirmed >= applied;
        }, "replacement operation was not confirmed");
        expect(app.world.get(local, Player)).toEqual({ value: 60, input: 5 });
        expect(setups).toBe(1);
        expect(f.joined).toEqual([eid]);
        expect(f.left).toEqual([]);
        expect(errors).toEqual([]);
    } finally {
        app.dispose();
        proxy.stop();
        f.host.stop();
    }
    await replaySession(f.game, "test", recording);
});

test("failed resume attempts double the delay and a refusal stops retries", async () => {
    const { game, host: rooms } = fixture();
    rooms.stop();
    let attempts = 0;
    let cut = () => {};
    const value = crypto.randomUUID();
    const host = serveWebSocket({
        hostname: "127.0.0.1",
        port: 0,
        disconnect() {},
        connect(connection, url) {
            attempts++;
            if (attempts === 1) {
                connection.socket.sendText(`resume:${value}:${value}`);
                cut = () => connection.close(1011);
                return;
            }
            expect(url?.searchParams.get("resume")).toBe(value);
            connection.close(attempts === 2 ? 1011 : 1008);
        },
    });
    const retries: { delay: number; run: () => void }[] = [];
    const errors: Error[] = [];
    let transport!: import("./link").Connection;
    game.onClient = (_world, connection) => {
        transport = connection;
    };
    const app = await createApp({
        defaults: false,
        plugins: [
            client(game, {
                url: `ws://127.0.0.1:${host.port}`,
                onError: (error) => errors.push(error),
                retryAfter(delay, run) {
                    retries.push({ delay, run });
                    return () => {};
                },
            }),
        ],
    });
    try {
        app.world.step(1 / 60);
        cut();
        await until(() => retries.length === 1, "first retry was not scheduled");
        expect(attempts).toBe(1);
        expect(retries[0]!.delay).toBe(50);
        retries[0]!.run();
        await until(() => retries.length === 2, "second retry was not scheduled");
        expect(retries[1]!.delay).toBe(100);
        retries[1]!.run();
        await until(() => errors.length === 1, "refusal was not reported");
        expect(retries).toHaveLength(2);
        expect(transport.closed).toBe(true);
    } finally {
        app.dispose();
        host.stop(true);
    }
});
