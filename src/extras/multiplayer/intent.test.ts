import { expect, test } from "bun:test";
import { component, u32, World } from "../../engine";
import {
    type AcceptedIntent,
    ClientPrediction,
    contactOutputKey,
    FixedOutputs,
    IntentReceiver,
    type IntentRecord,
    IntentSender,
    PredictionTarget,
    Replicate,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";

const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };
test("accepted intent keeps an on-time tick, assigns the next tick when late, and reports lateness", () => {
    const link = seededLink(1, faults);
    const sender = new IntentSender(link.b),
        receiver = new IntentReceiver(link.a);
    sender.record(10, "input", [1, -1]);
    sender.record(2, "operation", [7]);
    sender.flush();
    receiver.receive(link.a.receive(), 5);
    sender.flush(link.b.receive());
    expect([...receiver.records.values()].map((r) => [r.tick, r.lateness])).toEqual([
        [10, 0],
        [5, 3],
    ]);
    expect([...sender.records.values()]).toEqual([...receiver.records.values()]);
    expect([...receiver.records.values()].map((r) => r.slack)).toEqual([5, 0]);
    sender.flush();
    receiver.receive(link.a.receive(), 8);
    expect(receiver.records.size).toBe(2);
});
test("a late join image carries accepted operations through its cut-over tick", () => {
    const server = new World(),
        client = new World();
    try {
        const link = seededLink(1, faults);
        const operation: AcceptedIntent = {
            id: 1,
            tick: 7,
            lateness: 2,
            kind: "operation",
            values: [9],
        };
        const future: AcceptedIntent = { ...operation, id: 2, tick: 18 };
        const sender = new ReplicationSender(server, [], [], 1, link.a);
        const delivered: AcceptedIntent[] = [];
        const receiver = new ReplicationReceiver(
            client,
            [],
            link.b,
            () => {},
            (_tick, _rows, intents, apply) => {
                delivered.push(...intents);
                apply();
            },
        );
        sender.record(12, [operation, future]);
        sender.flush();
        receiver.flush();
        sender.flush();
        expect(receiver.confirmed).toBe(12);
        expect(delivered).toEqual([operation]);
        sender.record(13, [operation]);
        sender.flush();
        receiver.flush();
        expect(delivered).toEqual([operation, operation]);
    } finally {
        server.dispose();
        client.dispose();
    }
});

test("the local avatar's default prediction follows its wire identity, not the server's eid", () => {
    const Value = component("PredictionIdentityValue", { value: u32 });
    const server = new World(),
        client = new World();
    try {
        const avatar = server.create();
        server.add(avatar, Value);
        server.add(avatar, Replicate);
        for (let i = 0; i < 7; i++) client.create();
        const link = seededLink(1, faults);
        const sender = new ReplicationSender(server, [Value], [new Map([[1, [Value]]])], 1, link.a);
        const prediction = new ClientPrediction(
            client,
            [Value],
            link.b,
            (_world, entities) => entities.get(1),
            () => {},
        );
        sender.record(7);
        sender.flush();
        prediction.receiver.flush();
        const eid = prediction.receiver.entities.get(1)!;
        expect(eid).not.toBe(avatar);
        expect(client.time.fixedTick).toBe(7);
        expect(prediction.predicted(eid)).toBe(true);
        client.storage(PredictionTarget).enabled.set(eid, 0);
        expect(prediction.predicted(eid)).toBe(false);
        const record: IntentRecord = { id: 1, tick: 2, kind: "operation", values: [1] };
        prediction.accept([{ ...record, tick: 5, lateness: 3 }]);
        prediction.accept([record]);
        expect(prediction.records.get(1)!.tick).toBe(5);
    } finally {
        server.dispose();
        client.dispose();
    }
});

test("fixed outputs distinguish both children of a contact across replay", () => {
    const world = new World();
    try {
        const a = world.ref(world.create()),
            b = world.ref(world.create());
        const outputs = new FixedOutputs(),
            delivered: number[] = [];
        for (let replay = 0; replay < 2; replay++)
            for (const childIndex of [0, 1]) {
                const key = contactOutputKey("begin", { shapeA: a, shapeB: b, childIndex });
                outputs.deliver(2, key, () => delivered.push(childIndex));
            }
        expect(delivered).toEqual([0, 1]);
    } finally {
        world.dispose();
    }
});
