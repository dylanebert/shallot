import { expect, test } from "bun:test";
import { createApp, World } from "../../engine";
import {
    ClientPrediction,
    IntentReceiver,
    IntentSender,
    PredictionPlugin,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";
import { decodeIntent, encodeIntent, IntentMessageKind } from "./intent";

const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };

test("3000 confirmed ticks bound Accepted, frame intents and every intent store", () => {
    const server = new World(),
        client = new World();
    try {
        const link = seededLink(1, faults);
        const sender = new IntentSender(link.b),
            receiver = new IntentReceiver(link.a);
        const frames = new ReplicationSender(server, [], [], 1, link.a);
        const prediction = new ClientPrediction(
            client,
            [],
            link.b,
            () => undefined,
            () => {},
        );
        let acceptedBytes = 0,
            frameRecords = 0,
            serverRecords = 0,
            clientRecords = 0,
            predictedRecords = 0;
        for (let tick = 1; tick <= 3000; tick++) {
            prediction.accept([sender.record(tick, "operation", [tick])]);
            sender.flush();
            receiver.receive(link.a.receive(), tick);
            const accepted = link.b.receive();
            for (const bytes of accepted) {
                acceptedBytes = Math.max(acceptedBytes, bytes.length);
                prediction.accept(decodeIntent(bytes));
            }
            sender.flush(accepted);
            frames.record(tick, [...receiver.records.values()]);
            frameRecords = Math.max(frameRecords, receiver.records.size);
            frames.flush();
            prediction.receiver.flush();
            frames.flush();
            // The composition retires only records covered by this recipient's acknowledgement.
            receiver.prune(frames.acknowledged);
            serverRecords = Math.max(serverRecords, receiver.records.size);
            clientRecords = Math.max(clientRecords, sender.records.size);
            predictedRecords = Math.max(predictedRecords, prediction.records.size);
        }
        expect({
            acceptedBytes,
            frameRecords,
            serverRecords,
            clientRecords,
            predictedRecords,
        }).toEqual({
            acceptedBytes: 44,
            frameRecords: 1,
            serverRecords: 0,
            clientRecords: 0,
            predictedRecords: 0,
        });
    } finally {
        server.dispose();
        client.dispose();
    }
});

test("lost Accepted and reordered retries apply operations once and cumulative acceptance stops retries without crossing a gap", () => {
    const link = seededLink(2, faults);
    const sender = new IntentSender(link.b),
        receiver = new IntentReceiver(link.a);
    const first = sender.record(1, "operation", [1]);
    const missing = sender.record(1, "operation", [2]);
    const third = sender.record(1, "operation", [3]);
    const applied: number[] = [];
    for (let tick = 1; tick <= 101; tick++) {
        sender.flush();
        const messages = link.a.receive();
        messages.push(encodeIntent(IntentMessageKind.Records, [third, first, missing]));
        receiver.receive(
            messages.reverse().map((bytes) =>
                encodeIntent(
                    IntentMessageKind.Records,
                    decodeIntent(bytes).filter((record) => tick >= 68 || record.id !== missing.id),
                ),
            ),
            tick,
        );
        for (const record of receiver.records.values())
            if (record.tick === tick) applied.push(record.id);
        receiver.prune(tick);
        const accepted = link.b.receive();
        // Lose every acknowledgement until the gap has closed and all records were pruned.
        if (tick === 101) sender.flush(accepted);
        if (tick === 2) expect([...receiver.records.keys()]).toEqual([third.id]);
    }
    expect(applied.sort()).toEqual([1, 2, 3]);
    expect(sender.records.size).toBe(0);
    expect(link.a.receive()).toEqual([]);
    expect(receiver.records.size).toBe(0);
});

test("a refused future operation cannot pin any store or stall later operations in a composed session", async () => {
    const app = await createApp({ defaults: false, plugins: [PredictionPlugin] });
    const server = new World();
    try {
        const link = seededLink(8, faults);
        const sender = new IntentSender(link.b),
            receiver = new IntentReceiver(link.a);
        const frames = new ReplicationSender(server, [], [], 1, link.a);
        const prediction = new ClientPrediction(
            app.world,
            [],
            link.b,
            () => undefined,
            () => {},
        );
        prediction.onMessages = (messages) => {
            sender.flush(messages);
        };
        prediction.accept([sender.record(4294967294, "operation", [-1])]);
        const applied: number[] = [],
            samples: number[][] = [];
        for (let tick = 1; tick <= 3000; tick++) {
            prediction.accept([sender.record(tick, "operation", [tick])]);
            sender.flush();
            receiver.prune(tick - 1);
            receiver.receive(frames.flush(), tick);
            for (const record of receiver.records.values())
                if (record.tick === tick) applied.push(record.values[0]!);
            frames.record(tick, [...receiver.records.values()]);
            frames.flush();
            app.world.step(0);
            frames.flush();
            if (tick === 1000 || tick === 3000)
                samples.push([
                    receiver.records.size,
                    sender.records.size,
                    prediction.records.size,
                    applied.length,
                ]);
        }
        expect(samples).toEqual([
            [1, 0, 0, 1000],
            [1, 0, 0, 3000],
        ]);
        expect(applied).toEqual(Array.from({ length: 3000 }, (_, index) => index + 1));
    } finally {
        app.dispose();
        server.dispose();
    }
});

test("individual acceptance releases an ID above a still-unaccepted gap", () => {
    const link = seededLink(4, faults);
    const sender = new IntentSender(link.b),
        receiver = new IntentReceiver(link.a);
    const missing = sender.record(1, "operation", [1]);
    sender.record(1, "operation", [2]);
    sender.flush();
    receiver.receive(
        link.a.receive().map((bytes) =>
            encodeIntent(
                IntentMessageKind.Records,
                decodeIntent(bytes).filter((record) => record.id !== missing.id),
            ),
        ),
        1,
    );
    sender.flush(link.b.receive());
    expect([...sender.records.values()]).toEqual([missing]);
});

test("a refused future ID bounds receiver records and frame traffic at 1000 and 3000 ticks", () => {
    const link = seededLink(5, faults);
    const receiver = new IntentReceiver(link.a);
    const samples: number[][] = [];
    const server = new World(),
        client = new World();
    const frameLink = seededLink(5, faults);
    const frames = new ReplicationSender(server, [], [], 1, frameLink.a);
    let frameIntents = 0;
    const replica = new ReplicationReceiver(
        client,
        [],
        frameLink.b,
        () => {},
        (_tick, _rows, intents, apply) => {
            frameIntents = intents.length;
            apply();
        },
    );
    const gap = { id: 1, tick: 4294967294, kind: "operation" as const, values: [0] };
    try {
        for (let tick = 1; tick <= 3000; tick++) {
            receiver.prune(tick - 1);
            receiver.receive(
                [
                    encodeIntent(IntentMessageKind.Records, [
                        gap,
                        { id: tick + 1, tick, kind: "operation", values: [tick] },
                    ]),
                ],
                tick,
            );
            const accepted = link.b.receive()[0]!;
            frames.record(tick, [...receiver.records.values()]);
            frames.flush();
            replica.flush();
            frames.flush();
            if (tick === 1000 || tick === 3000)
                samples.push([
                    receiver.records.size,
                    decodeIntent(accepted).length,
                    frameIntents,
                    accepted.length,
                ]);
        }
        expect(samples).toEqual([
            [1, 1, 1, 44],
            [1, 1, 1, 44],
        ]);
        expect(receiver.records.has(1)).toBe(false);
    } finally {
        server.dispose();
        client.dispose();
    }
});

test("an ID-window stall releases later IDs when the missing ID is supplied", () => {
    const link = seededLink(6, faults);
    const receiver = new IntentReceiver(link.a, 2, 6);
    const records = Array.from({ length: 7 }, (_, index) => ({
        id: index + 1,
        tick: index === 0 ? 10 : 1,
        kind: "operation" as const,
        values: [],
    }));
    const receive = (tick: number) => {
        receiver.receive(
            [
                encodeIntent(
                    IntentMessageKind.Records,
                    records
                        .slice()
                        .reverse()
                        .filter((record) => tick >= 8 || record.id !== 1),
                ),
            ],
            tick,
        );
        return decodeIntent(link.b.receive()[0]!);
    };
    expect(receive(1).map((record) => record.id)).toEqual([6, 5, 4, 3, 2]);
    receiver.prune(7);
    expect(receive(8).find((record) => record.id === 1)?.tick).toBe(10);
    receiver.prune(8);
    expect(receive(9).map((record) => record.id)).toEqual([1, 7]);
    receiver.prune(10);
    expect(receiver.records.size).toBe(0);
    expect(receive(11)).toEqual([]);
});

test("refusal resolves a future ID permanently, including when refused above a missing ID", () => {
    const link = seededLink(9, faults);
    const receiver = new IntentReceiver(link.a, 2, 6);
    const future = { id: 2, tick: 10, kind: "operation" as const, values: [] };
    receiver.receive([encodeIntent(IntentMessageKind.Records, [future])], 1);
    expect(decodeIntent(link.b.receive()[0]!)).toEqual([]);
    receiver.receive(
        [encodeIntent(IntentMessageKind.Records, [{ ...future, id: 1, tick: 8 }, future])],
        8,
    );
    const accepted = link.b.receive()[0]!;
    expect(decodeIntent(accepted).map((record) => record.id)).toEqual([1]);
    expect(new DataView(accepted.buffer, accepted.byteOffset).getUint32(8, true)).toBe(2);
    receiver.prune(8);
    receiver.receive([encodeIntent(IntentMessageKind.Records, [future])], 9);
    expect(decodeIntent(link.b.receive()[0]!)).toEqual([]);
});

test("a never-supplied ID bounds receiver records over 3000 ticks", () => {
    const link = seededLink(10, faults);
    const receiver = new IntentReceiver(link.a, 32, 6);
    for (let tick = 1; tick <= 3000; tick++) {
        receiver.prune(tick - 1);
        receiver.receive(
            [
                encodeIntent(IntentMessageKind.Records, [
                    { id: tick + 1, tick, kind: "operation", values: [] },
                ]),
            ],
            tick,
        );
        expect(decodeIntent(link.b.receive()[0]!).length).toBeLessThanOrEqual(5);
    }
    expect(receiver.records.size).toBe(5);
});

test("the default ID window accepts two reordered operations for every tick of the prediction horizon", () => {
    const link = seededLink(7, faults);
    const receiver = new IntentReceiver(link.a);
    const records = Array.from({ length: 66 }, (_, index) => ({
        id: index + 1,
        tick: 1 + Math.floor(index / 2),
        kind: "operation" as const,
        values: [],
    }));
    receiver.receive([encodeIntent(IntentMessageKind.Records, records.reverse())], 1);
    expect(decodeIntent(link.b.receive()[0]!).length).toBe(66);
    receiver.prune(33);
    expect(receiver.records.size).toBe(0);
});

test("prediction drops folded accepted records but keeps unaccepted local operations", () => {
    const world = new World();
    try {
        const link = seededLink(3, faults);
        const prediction = new ClientPrediction(
            world,
            [],
            link.b,
            () => undefined,
            () => {},
        );
        prediction.accept([
            { id: 1, tick: 0, kind: "operation", values: [], lateness: 0 },
            { id: 2, tick: 0, kind: "operation", values: [] },
            { id: 3, tick: 1, kind: "operation", values: [], lateness: 0 },
        ]);
        expect([...prediction.records.keys()]).toEqual([2, 3]);
        prediction.accept([], 3);
        expect([...prediction.records.keys()]).toEqual([3]);
        prediction.accept([{ id: 2, tick: 0, kind: "operation", values: [] }]);
        expect([...prediction.records.keys()]).toEqual([3]);
    } finally {
        world.dispose();
    }
});
