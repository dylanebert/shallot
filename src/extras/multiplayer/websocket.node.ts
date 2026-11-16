import { expect, setDefaultTimeout, test } from "bun:test";
import { createInterface } from "node:readline";
import { Readable } from "node:stream";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { bunChild } from "../../../scripts/bun-child";
import { component, u32, World } from "../../engine";
import { IntentSender, ReplicationReceiver, WebSocketConnection } from ".";

setDefaultTimeout(TIMEOUT.node);
const Value = component("SocketValue", { value: u32 });
type Status = {
    tick: number;
    closed: number;
    peers: { id: number; value: number; inputs: number }[];
};

test("binary sockets carry images, burst frames and isolated inputs; reconnect rejoins at a fresh image", async () => {
    const child = Bun.spawn({
        ...bunChild([`${import.meta.dir}/websocket-server.fixture.ts`]),
        stdin: "pipe",
        stdout: "pipe",
        stderr: "inherit",
    });
    const lines = createInterface({ input: Readable.from(child.stdout) })[Symbol.asyncIterator]();
    const read = async () => JSON.parse((await lines.next()).value!);
    const { port } = await read();
    const rpc = async (step = false): Promise<Status> => {
        child.stdin.write(`${JSON.stringify({ step })}\n`);
        child.stdin.flush();
        return read();
    };
    const worlds: World[] = [];
    const sockets: WebSocket[] = [];
    async function connect() {
        const socket = new WebSocket(`ws://127.0.0.1:${port}`);
        sockets.push(socket);
        const connection = new WebSocketConnection(socket);
        await new Promise<void>((resolve, reject) => {
            socket.addEventListener("open", () => resolve(), { once: true });
            socket.addEventListener("error", reject, { once: true });
        });
        const world = new World();
        worlds.push(world);
        const confirmed: { tick: number; value: number }[] = [];
        const receiver = new ReplicationReceiver(world, [Value], connection, (tick, rows) => {
            expect(rows.size).toBe(1);
            confirmed.push({ tick, value: [...rows.values()][0]!.get(0)![0]! });
        });
        return { socket, connection, receiver, confirmed, intent: new IntentSender(connection) };
    }
    // Poll observable progress rather than assigning socket delivery a wall-clock deadline.
    async function until(predicate: () => boolean | Promise<boolean>) {
        while (!(await predicate())) await new Promise((resolve) => setTimeout(resolve, 1));
    }
    try {
        const a = await connect();
        const b = await connect();
        await until(() => {
            a.receiver.flush();
            b.receiver.flush();
            return a.receiver.confirmed === 10 && b.receiver.confirmed === 10;
        });
        expect(a.confirmed).toEqual([{ tick: 10, value: 100 }]);
        expect(b.confirmed).toEqual([{ tick: 10, value: 200 }]);
        a.intent.record(11, "input", [1234]);
        a.intent.flush();
        await until(async () => (await rpc()).peers[0]!.inputs === 1);
        const history: Status[] = [];
        for (let i = 0; i < 4; i++) history.push(await rpc(true));
        // Neither client drains during the server burst. All queued frames survive one drain.
        await until(() => {
            a.receiver.flush();
            b.receiver.flush();
            return a.receiver.confirmed === 14 && b.receiver.confirmed === 14;
        });
        expect(a.confirmed.slice(1)).toEqual(
            history.map((s) => ({ tick: s.tick, value: s.peers[0]!.value })),
        );
        expect(b.confirmed.slice(1)).toEqual(
            history.map((s) => ({ tick: s.tick, value: s.peers[1]!.value })),
        );
        expect(a.confirmed.at(-1)!.value).toBe(1234);
        expect(b.confirmed.at(-1)!.value).toBe(200);
        a.socket.close();
        await until(async () => (await rpc()).closed === 1);
        await rpc(true);
        const rejoined = await connect();
        await until(() => {
            rejoined.receiver.flush();
            return rejoined.receiver.confirmed === 15;
        });
        expect(rejoined.confirmed).toEqual([{ tick: 15, value: 300 }]);
        rejoined.intent.record(16, "input", [4321]);
        rejoined.intent.flush();
        await until(async () => (await rpc()).peers.find((p) => p.id === 3)!.inputs === 1);
        for (let i = 0; i < 4; i++) {
            const state = await rpc(true);
            await until(() => {
                rejoined.receiver.flush();
                b.receiver.flush();
                return (
                    rejoined.receiver.confirmed === state.tick &&
                    b.receiver.confirmed === state.tick
                );
            });
            expect(rejoined.confirmed.at(-1)).toEqual({
                tick: state.tick,
                value: state.peers.find((p) => p.id === 3)!.value,
            });
            expect(b.confirmed.at(-1)!.value).toBe(200);
        }
        expect(rejoined.confirmed.at(-1)!.value).toBe(4321);
        expect(rejoined.connection.receive()).toEqual([]);
    } finally {
        for (const socket of sockets) socket.close();
        child.stdin.end();
        child.kill();
        await child.exited;
        for (const world of worlds) world.dispose();
    }
});
