import { expect, test } from "bun:test";
import { component, createApp, f32, text, u32, World } from "../../engine";
import {
    type AcceptedIntent,
    decodeServerMessage,
    encodeAcknowledgement,
    encodeFrame,
    IntentMessageKind,
    Replicate,
    ReplicationMessageKind,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";
import { diagnosticKind, writeDiagnostic } from "./diagnostics";
import { type Game, isClientMessage, ServerSession, server } from "./game";
import { decodeAccepted, decodeInputs, decodeIntent, encodeInputs, encodeIntent } from "./intent";

const Value = component("DecoderTestValue", { value: u32, label: text(16) });
const layout = [[u32, Value.label]];
const schema = [Value];
const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };
const operation: AcceptedIntent = {
    id: 1,
    tick: 3,
    kind: "operation",
    values: [1, -2, NaN, -0],
    lateness: 1,
    slack: 0,
    text: "🐟\0é",
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
    world.add(eid, Value, { value: 7, label: "🐟" });
    world.add(eid, Replicate);
    const link = seededLink(1, faults);
    const sender = new ReplicationSender(world, schema, [new Map([[1, [Value]]])], 1, link.a);
    sender.record(1, [operation]);
    sender.flush();
    world.storage(Value).value.set(eid, 8);
    sender.record(2);
    sender.flush();
    const frames = link.b.receive();
    world.dispose();
    return frames;
}

function diagnostic(id: number, held?: number): Uint8Array {
    const bytes = new Uint8Array(held === undefined ? 8 : 16);
    writeDiagnostic(new DataView(bytes.buffer), id, held);
    return bytes;
}
const clientSeeds = [
    encodeInputs([]),
    encodeInputs([{ tick: 4, values: [NaN, -0, Infinity] }]),
    encodeInputs([0, 1, 2, 3].map((tick) => ({ tick, values: [tick, 0.5, -1] }))),
    encodeIntent(IntentMessageKind.Records, [operation, { ...operation, id: 2, values: [] }]),
    encodeAcknowledgement(2, 1),
    diagnostic(9),
    diagnostic(7, 2.25),
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

test("all emitted message kinds and the accepted seeded corpus re-encode byte-exactly", () => {
    const clientKinds = new Set<number>();
    const serverKinds = new Set<number>();
    const serverSeeds = [
        ...replication(),
        encodeIntent(IntentMessageKind.Accepted, [operation], 1),
        diagnostic(2),
        diagnostic(5, 0.25),
    ];
    for (const bytes of clientSeeds) expect(isClientMessage(bytes)).toBe(true);
    for (const bytes of serverSeeds) expect(() => decodeServerMessage(bytes, layout)).not.toThrow();
    for (const bytes of [...clientSeeds, ...serverSeeds, ...inputs]) {
        if (bytes.length < 4) continue;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const kind = view.getUint32(0, true);
        if (isClientMessage(bytes)) {
            clientKinds.add(kind);
            if (diagnosticKind(bytes)) {
                expect(
                    diagnostic(
                        view.getUint32(4, true),
                        kind === 7 ? view.getFloat64(8, true) : undefined,
                    ),
                ).toEqual(bytes);
            } else if (kind === ReplicationMessageKind.Acknowledgement) {
                expect(
                    encodeAcknowledgement(view.getUint32(4, true), view.getUint32(8, true)),
                ).toEqual(bytes);
            } else if (kind === IntentMessageKind.Inputs) {
                expect(encodeInputs(decodeInputs(bytes))).toEqual(bytes);
            } else {
                expect(encodeIntent(kind, decodeIntent(bytes))).toEqual(bytes);
            }
        }
        const decoded = settle(() => decodeServerMessage(bytes, layout));
        if (!(decoded instanceof Error)) {
            serverKinds.add(kind);
            if (decoded.value) {
                expect(encodeFrame(decoded.value, layout)).toEqual(bytes);
            } else if (kind === IntentMessageKind.Accepted) {
                const { records, watermark } = decodeAccepted(bytes);
                expect(encodeIntent(kind, records, watermark)).toEqual(bytes);
            }
        }
    }
    expect([...clientKinds].sort()).toEqual([3, 4, 6, 7]);
    expect([...serverKinds].sort()).toEqual([1, 2, 5, 6, 7]);
});

for (const mode of ["all", "latest"] as const) {
    for (const defect of [
        "future baseline",
        "same-tick baseline",
        "duplicate lanes",
        "unordered lanes",
    ] as const) {
        test(`${mode}: refuses ${defect} before queueing or applying`, () => {
            const world = new World();
            const { a, b } = seededLink(1, faults);
            const errors: Error[] = [];
            const receiver = new ReplicationReceiver(world, schema, {
                ...b,
                onError: (error) => errors.push(error),
            });
            if (defect === "future baseline" || defect === "same-tick baseline") {
                a.send(
                    words(
                        ReplicationMessageKind.Frame,
                        5,
                        1,
                        defect === "future baseline" ? 2 : 1,
                        0,
                        0x80000000,
                    ),
                );
                receiver.flush(mode);
                const sender = new ReplicationSender(world, schema, [], 1, a);
                sender.record(2);
                sender.flush();
            } else if (defect === "duplicate lanes") {
                a.send(
                    words(
                        ReplicationMessageKind.Image,
                        11,
                        0,
                        0,
                        1,
                        0x80000000,
                        1,
                        1,
                        0,
                        2,
                        0,
                        0,
                        12,
                        34,
                    ),
                );
            } else {
                const bytes = new Uint8Array(54);
                bytes.set(
                    words(ReplicationMessageKind.Image, 11, 0, 0, 1, 0x80000000, 1, 1, 0, 2, 1, 0),
                );
                new DataView(bytes.buffer).setUint32(50, 34, true);
                a.send(bytes);
            }
            for (let i = 0; i < 2; i++) expect(receiver.flush(mode)).toEqual([]);
            expect(receiver.confirmed).toBe(-1);
            expect(receiver.entities.size).toBe(0);
            expect(b.closed).toBe(true);
            expect(errors).toHaveLength(1);
            world.dispose();
        });
    }
}

for (const mode of ["all", "latest"] as const) {
    test(`${mode}: an orphan delta cannot hide the first image`, () => {
        const world = new World();
        const { a, b } = seededLink(1, faults);
        const applied: number[] = [];
        const receiver = new ReplicationReceiver(world, schema, b, (tick) => applied.push(tick));
        a.send(words(ReplicationMessageKind.Frame, 5, 1, 0, 0, 0x80000000));
        receiver.flush(mode);
        const sender = new ReplicationSender(world, schema, [], 1, a);
        sender.record(2);
        sender.flush();
        receiver.flush(mode);
        expect(applied).toEqual([2]);
        expect(receiver.confirmed).toBe(2);
        expect(b.closed).toBe(false);
        world.dispose();
    });

    test(`${mode}: waits for a future baseline but skips a discarded baseline`, () => {
        const world = new World();
        const { a, b } = seededLink(1, faults);
        const applied: number[] = [];
        const receiver = new ReplicationReceiver(world, schema, b, (tick) => applied.push(tick));
        const image = (tick: number) =>
            words(ReplicationMessageKind.Image, 5, tick, tick, 0, 0x80000000);
        const frame = (tick: number, base: number) =>
            words(ReplicationMessageKind.Frame, 5, tick, base, 0, 0x80000000);
        a.send(image(2));
        receiver.flush(mode);
        a.send(frame(4, 3));
        receiver.flush(mode);
        expect(applied).toEqual([2]);
        a.send(frame(3, 2));
        receiver.flush(mode);
        expect(applied).toEqual([2, 3, 4]);
        a.send(frame(5, 0));
        a.send(frame(6, 5));
        receiver.flush(mode);
        expect(receiver.confirmed).toBe(4);
        a.send(frame(7, 4));
        receiver.flush(mode);
        a.send(image(9));
        receiver.flush(mode);
        expect(applied).toEqual([2, 3, 4, 7, 9]);
        expect(receiver.confirmed).toBe(9);
        expect(b.closed).toBe(false);
        world.dispose();
    });
}

test("a delayed first image keeps the earliest delta window and lets overflow retry", () => {
    const world = new World();
    const { a, b } = seededLink(1, faults);
    const receiver = new ReplicationReceiver(world, schema, b);
    const frame = (tick: number) => words(ReplicationMessageKind.Frame, 5, tick, 0, 0, 0x80000000);
    for (let tick = 40; tick >= 1; tick--) a.send(frame(tick));
    receiver.flush("latest");
    expect(receiver.confirmed).toBe(-1);
    a.send(words(ReplicationMessageKind.Image, 5, 0, 0, 0, 0x80000000));
    receiver.flush("latest");
    expect(receiver.confirmed).toBe(32);
    for (let tick = 33; tick <= 40; tick++) a.send(frame(tick));
    receiver.flush("latest");
    expect(receiver.confirmed).toBe(40);
    world.dispose();
});

test("structural frame validation does not rerun authored codecs", () => {
    let origin = 0;
    const Local = component("LocalCodec", {
        value: {
            name: "local",
            ctor: Uint8Array,
            lanes: 1 as const,
            encode: (value: number) => value + origin,
            decode: (raw: number) => Math.max(0, raw - origin),
        },
    });
    const server = new World();
    const client = new World();
    const eid = server.create();
    server.add(eid, Local, { value: 7 });
    server.add(eid, Replicate);
    const { a, b } = seededLink(1, faults);
    const sender = new ReplicationSender(server, [Local], [new Map([[1, [Local]]])], 1, a);
    sender.record(0);
    sender.flush();
    origin = 10;
    const receiver = new ReplicationReceiver(client, [Local], b);
    receiver.flush();
    expect(receiver.confirmed).toBe(0);
    expect(b.closed).toBe(false);
    expect(client.get([...receiver.entities.values()][0]!, Local)).toEqual({ value: 0 });
    server.dispose();
    client.dispose();
});

test("structural comparison preserves a float lane's NaN payload bits", () => {
    const bytes = words(
        ReplicationMessageKind.Image,
        10,
        0,
        0,
        1,
        0x80000000,
        1,
        1,
        0,
        1,
        0,
        0x7fc12345,
    );
    const frame = decodeServerMessage(bytes, [[f32]])!;
    expect(encodeFrame(frame, [[f32]])).toEqual(bytes);
});

test("client ingress refuses a nonzero Records watermark before routing", () => {
    const bytes = encodeIntent(IntentMessageKind.Records, [operation]);
    new DataView(bytes.buffer).setUint32(8, 1, true);
    expect(decodeIntent(bytes)).toHaveLength(1);
    expect(isClientMessage(bytes)).toBe(false);
});

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
