import { expect, test } from "bun:test";
import { component, createApp, u32, type World } from "../../engine";
import { ClientPrediction, PredictionPlugin, Replicate } from ".";
import { serveRooms, serveWebSocket } from "./bun";
import { ClientOf, client, type Game } from "./game";
import { createProxy } from "./proxy";
import { replaySession, type SessionRecording } from "./recording";

const Player = component("ReconnectPlayer", { value: u32 });

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

test("client backs off after a TCP cut and its running prediction confirms the same seat at a fresh image", async () => {
    const f = fixture();
    const proxy = await createProxy(f.host.port!, 1);
    let prediction!: ClientPrediction;
    let setups = 0;
    let retry: (() => void) | undefined;
    const delays: number[] = [];
    const errors: Error[] = [];
    let recording!: SessionRecording;
    f.game.plugins = [...f.game.plugins, PredictionPlugin];
    f.game.onClient = (world, connection) => {
        setups++;
        prediction = new ClientPrediction(
            world,
            f.game.schema,
            connection,
            (_world, entities) => entities.get(f.eid),
            () => {},
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
        const confirmed = receiver.confirmed;
        app.world.tick();
        expect(app.world.time.fixedTick).toBeGreaterThan(confirmed);
        proxy.controls.resetPeer();
        await until(() => !!retry, "close did not schedule retry");
        expect(delays).toEqual([50]);
        expect(f.left).toEqual([]);
        expect(f.world.has(eid, ClientOf)).toBe(true);
        f.world.storage(Player).value.set(eid, 99);
        for (let i = 0; i < 12; i++) f.world.tick();
        f.advance(50);
        retry!();
        await until(() => {
            app.world.step(1 / 60);
            return receiver.confirmed > confirmed + 1 && app.world.get(local, Player)?.value === 99;
        }, "replacement image was not confirmed by the running prediction");
        expect(prediction.receiver).toBe(receiver);
        expect(receiver.entities.get(eid)).toBe(local);
        expect(prediction.predicted(local)).toBe(true);
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
