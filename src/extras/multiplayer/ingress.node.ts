import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { World } from "../../engine";
import { IntentMessageKind, ReplicationReceiver, WebSocketConnection } from ".";
import { serveRooms } from "./bun";
import type { Game } from "./game";
import { encodeInputs, encodeIntent } from "./intent";

const game: Game = {
    plugins: [],
    schema: [],
    maxClients: 4,
    rules: () => [],
    validateIntent: () => true,
    applyIntent() {},
};

function words(...values: number[]) {
    const bytes = new Uint8Array(values.length * 4);
    const view = new DataView(bytes.buffer);
    for (const [i, value] of values.entries()) view.setUint32(i * 4, value, true);
    return bytes;
}

async function open(url: string) {
    const socket = new WebSocket(url);
    const closed = new Promise<number>((resolve) =>
        socket.addEventListener("close", (event) => resolve(event.code), { once: true }),
    );
    await new Promise((resolve) => socket.addEventListener("open", resolve, { once: true }));
    return { socket, closed };
}

const operation = { id: 1, tick: 1, kind: "operation" as const, values: [] };
const inputs = encodeInputs([{ tick: 1, values: [1] }]);
const malformed = [
    words(IntentMessageKind.Inputs).subarray(0, 3),
    inputs.subarray(0, inputs.length - 1),
    words(IntentMessageKind.Inputs, 5, 0),
    words(IntentMessageKind.Records, 1, 0, 1, 1, 2, 0, 0, 0),
    encodeIntent(IntentMessageKind.Accepted, [operation]),
    words(7, 1, 0),
    words(99, 0),
];
// Four ticks of 126 lanes: a valid 4072-byte packet, under the frame cap.
const largest = encodeInputs(
    [0, 1, 2, 3].map((tick) => ({ tick, values: Array<number>(126).fill(tick) })),
);

test("invalid, oversized and excess client frames close only their own connection while peers in two rooms keep confirming", async () => {
    const host = serveRooms(game, { hostname: "127.0.0.1", port: 0 });
    const base = `ws://127.0.0.1:${host.port}/?intent=`;
    const reserve = async () => {
        const response = await fetch(`http://127.0.0.1:${host.port}/rooms`, { method: "POST" });
        const { code, creator } = (await response.json()) as { code: string; creator: string };
        return `${base}join&code=${code}&creator=${creator}`;
    };
    const peers: { socket: WebSocket; receiver: ReplicationReceiver }[] = [];
    let running = true;
    const pump = (async () => {
        while (running) {
            for (const peer of peers) peer.receiver.flush();
            await Bun.sleep(4);
        }
    })();
    try {
        for (const url of [`${base}joinOrCreate`, await reserve()]) {
            const { socket } = await open(url);
            const receiver = new ReplicationReceiver(
                new World(),
                game.schema,
                new WebSocketConnection(socket),
            );
            peers.push({ socket, receiver });
        }
        const confirming = async () => {
            const from = peers.map((peer) => peer.receiver.confirmed);
            while (peers.some((peer, i) => peer.receiver.confirmed < from[i]! + 30))
                await Bun.sleep(4);
            expect(peers.every((peer) => peer.socket.readyState === WebSocket.OPEN)).toBe(true);
        };
        await confirming();
        const attack = async (url: string, code: number, send: (socket: WebSocket) => void) => {
            const attacker = await open(url);
            send(attacker.socket);
            const held = Bun.sleep(2000).then(() => "still open");
            expect(await Promise.race([attacker.closed, held])).toBe(code);
        };
        const room = `${base}joinOrCreate`;
        await attack(room, 1009, (socket) => socket.send(new Uint8Array(4097)));
        await attack(room, 1003, (socket) => socket.send("text"));
        for (const bytes of malformed)
            await attack(room, 1002, (socket) => socket.send(bytes.slice()));
        await attack(await reserve(), 1002, (socket) => socket.send(malformed[3]!.slice()));
        await attack(room, 1008, (socket) => {
            for (let id = 0; id < 4000; id++) socket.send(words(6, id));
        });
        await attack(room, 1008, (socket) => {
            for (let i = 0; i < 200; i++) socket.send(largest.slice());
        });
        await confirming();
    } finally {
        running = false;
        await pump;
        for (const peer of peers) peer.socket.close();
        host.stop();
    }
});
