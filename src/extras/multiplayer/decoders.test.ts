import { expect, test } from "bun:test";
import { component, createApp, u32, World } from "../../engine";
import {
    type AcceptedIntent,
    IntentMessageKind,
    Replicate,
    ReplicationMessageKind,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";
import { type Game, isClientMessage, ServerSession, server } from "./game";
import { decodeInputs, decodeIntent, encodeInputs, encodeIntent } from "./intent";

const Value = component("DecoderTestValue", { value: u32 });
const schema = [Value];
const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };
const operation: AcceptedIntent = {
    id: 1,
    tick: 3,
    kind: "operation",
    values: [1, -2],
    lateness: 1,
    slack: 0,
};

function words(...values: number[]) {
    const bytes = new Uint8Array(values.length * 4);
    const view = new DataView(bytes.buffer);
    for (const [i, value] of values.entries()) view.setUint32(i * 4, value, true);
    return bytes;
}

function replication() {
    const world = new World();
    const eid = world.create();
    world.add(eid, Value, { value: 7 });
    world.add(eid, Replicate);
    const link = seededLink(1, faults);
    const sender = new ReplicationSender(world, schema, [new Map([[1, [Value]]])], 1, link.a);
    sender.record(1, [operation]);
    sender.flush();
    world.get(eid, Value)!.value = 8;
    sender.record(2);
    sender.flush();
    return link.b.receive();
}

const pong = new Uint8Array(16);
new DataView(pong.buffer).setUint32(0, 7, true);
const clientSeeds = [
    encodeInputs([]),
    encodeInputs([{ tick: 4, values: [] }]),
    encodeInputs([0, 1, 2, 3].map((tick) => ({ tick, values: [tick, 0.5, -1] }))),
    encodeIntent(IntentMessageKind.Records, [operation, { ...operation, id: 2, values: [] }]),
    words(ReplicationMessageKind.Acknowledgement, 2, 1),
    words(6, 9),
    pong,
];
const seeds = [
    ...clientSeeds,
    encodeIntent(IntentMessageKind.Accepted, [operation], 1),
    ...replication(),
];
const boundaries = [0, 1, 2, 4, 5, 6, 7, 12, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff];

/** A bounded seeded set: truncations, extensions, byte flips and boundary words over each valid
 * seed, then arbitrary short frames. */
function invalidInputs(seed: number) {
    const random = () => {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        return seed / 0x100000000;
    };
    const below = (n: number) => Math.floor(random() * n);
    const inputs: Uint8Array[] = [];
    for (const valid of seeds) {
        for (let length = 0; length < valid.length; length++) inputs.push(valid.slice(0, length));
        for (let i = 0; i < 150; i++) {
            const bytes = new Uint8Array(valid.length + (below(4) === 0 ? below(16) : 0));
            bytes.set(valid);
            const view = new DataView(bytes.buffer);
            for (let edits = 1 + below(3); edits > 0; edits--) {
                if (below(2) && bytes.length >= 4)
                    view.setUint32(
                        below(bytes.length >> 2) * 4,
                        boundaries[below(boundaries.length)]!,
                        true,
                    );
                else if (bytes.length) bytes[below(bytes.length)] ^= 1 << below(8);
            }
            inputs.push(bytes);
        }
    }
    for (let i = 0; i < 400; i++) {
        const bytes = new Uint8Array(below(64));
        for (let b = 0; b < bytes.length; b++) bytes[b] = below(256);
        if (bytes.length >= 4 && below(2))
            new DataView(bytes.buffer).setUint32(0, 1 + below(7), true);
        inputs.push(bytes);
    }
    return inputs;
}
const inputs = invalidInputs(0x5eed);

function settle<T>(decode: () => T): { value: T } | Error {
    try {
        return { value: decode() };
    } catch (error) {
        if (!(error instanceof Error)) throw error;
        return error;
    }
}

test("tick input and operation decoders reject invalid bytes or return a result that re-encodes to their length", () => {
    for (const bytes of inputs) {
        const ticks = settle(() => decodeInputs(bytes));
        if (!(ticks instanceof Error)) {
            expect(ticks.value.length).toBeLessThanOrEqual(4);
            expect(new Set(ticks.value.map((input) => input.tick)).size).toBe(ticks.value.length);
            expect(ticks.value.every((input) => input.tick < 0xffffffff)).toBe(true);
            expect(encodeInputs(ticks.value).length).toBe(bytes.length);
        }
        const records = settle(() => decodeIntent(bytes));
        if (!(records instanceof Error)) {
            expect(records.value.every((record) => record.kind === "operation")).toBe(true);
            expect(encodeIntent(0, records.value).length).toBe(bytes.length);
        }
    }
});

test("the replication decoder rejects invalid bytes or confirms one tick with entities bounded by its length", () => {
    let confirmed = 0;
    for (const bytes of [...replication(), ...inputs]) {
        const link = seededLink(1, faults);
        const receiver = new ReplicationReceiver(new World(), schema, link.b);
        link.a.send(bytes);
        if (settle(() => receiver.flush()) instanceof Error || receiver.confirmed < 0) continue;
        expect(receiver.confirmed).toBe(new DataView(bytes.buffer).getUint32(8, true));
        expect(receiver.entities.size).toBeLessThanOrEqual(bytes.length / 8);
        confirmed++;
    }
    expect(confirmed).toBeGreaterThan(1);
});

test("every client message the ingress validator accepts reaches the server without throwing or closing its peer", async () => {
    const game: Game = {
        plugins: [],
        schema,
        maxClients: 1,
        rules: () => [],
        validateIntent: () => true,
        applyIntent() {},
    };
    const app = await createApp({ defaults: false, plugins: [server(game)] });
    try {
        const authority = app.world.resource(ServerSession).server!;
        const link = seededLink(1, faults);
        const eid = authority.connect(link.a)!;
        for (const bytes of clientSeeds) expect(isClientMessage(bytes)).toBe(true);
        let accepted = 0;
        for (const bytes of [...clientSeeds, ...inputs]) {
            if (!isClientMessage(bytes)) continue;
            accepted++;
            link.b.send(bytes);
            app.world.tick();
            link.b.receive();
            expect(authority.peers.has(eid)).toBe(true);
        }
        expect(accepted).toBeGreaterThan(clientSeeds.length);
        expect(accepted).toBeLessThan(inputs.length / 4);
    } finally {
        app.dispose();
    }
});
