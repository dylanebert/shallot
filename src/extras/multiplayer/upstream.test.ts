import { expect, test } from "bun:test";
import { component, Time, u32, World } from "../../engine";
import {
    IntentMessageKind,
    IntentSender,
    MAX_MESSAGE_BYTES,
    Replicate,
    ReplicationMessageKind,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";
import { decodeInputs, decodeIntent, encodeIntent } from "./intent";

const Value = component("UpstreamTestValue", { value: u32 });
const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };

test("a client stepped and flushed eight times per tick sends at most one acknowledgement, input and operation packet per tick", () => {
    const server = new World();
    const eid = server.create();
    server.add(eid, Value, { value: 0 });
    server.add(eid, Replicate);
    const link = seededLink(1, faults);
    const frames = new ReplicationSender(server, [Value], [new Map([[1, [Value]]])], 1, link.a);
    const client = new World();
    const receiver = new ReplicationReceiver(client, [Value], link.b);
    const intent = new IntentSender(link.b);
    const sent = { acks: 0, inputs: 0, records: 0 };
    const send = link.b.send.bind(link.b);
    link.b.send = (bytes) => {
        const kind = new DataView(bytes.buffer, bytes.byteOffset).getUint32(0, true);
        if (kind === ReplicationMessageKind.Acknowledgement) sent.acks++;
        if (kind === IntentMessageKind.Inputs) sent.inputs++;
        if (kind === IntentMessageKind.Records) sent.records++;
        send(bytes);
    };
    let records = 0;
    for (let tick = 0; tick < 120; tick++) {
        server.get(eid, Value)!.value = tick;
        frames.record(tick);
        frames.flush();
        Object.assign(sent, { acks: 0, inputs: 0, records: 0 });
        for (let frame = 0; frame < 8; frame++) {
            // One unresolved operation stays pending; the server never drains intents.
            if (tick === 10 && frame === 3) intent.record(tick, "operation", [1]);
            if (frame === 0) {
                intent.record(tick, "input", [tick]);
                intent.flush();
            }
            intent.flush(receiver.flush("latest"));
            client.step(Time.FIXED_DT / 8);
        }
        expect(sent.acks).toBeLessThanOrEqual(1);
        expect(sent.inputs).toBe(1);
        expect(sent.records).toBeLessThanOrEqual(1);
        records += sent.records;
    }
    expect(client.time.fixedTick).toBeGreaterThanOrEqual(119);
    expect(receiver.confirmed).toBe(119);
    expect(frames.acknowledged).toBeGreaterThanOrEqual(118);
    expect(records).toBe(120 - 10);
});

test("a pending batch over the message cap goes oldest first in packets within it, and an oversized record throws", () => {
    const link = seededLink(1, faults);
    const intent = new IntentSender(link.b);
    for (let tick = 0; tick < 4; tick++)
        intent.record(tick, "input", Array<number>(300).fill(tick));
    for (let i = 0; i < 100; i++) intent.record(3, "operation", Array<number>(10).fill(i));
    expect(() => intent.record(3, "operation", Array<number>(508).fill(0))).toThrow();
    expect(() => intent.record(3, "input", Array<number>(511).fill(0))).toThrow();
    const ids: number[] = [];
    for (let tick = 4; ids.length < 100; tick++) {
        intent.flush();
        const packets = link.a.receive();
        for (const bytes of packets) expect(bytes.length).toBeLessThanOrEqual(MAX_MESSAGE_BYTES);
        const inputs = packets.find((bytes) => bytes[0] === IntentMessageKind.Inputs);
        const batch = packets.find((bytes) => bytes[0] === IntentMessageKind.Records);
        expect(decodeInputs(inputs!).at(-1)!.tick).toBe(tick - 1);
        const resolved = decodeIntent(batch!).map((record) => record.id);
        expect(resolved[0]).toBe(ids.length + 1);
        ids.push(...resolved);
        intent.flush([encodeIntent(IntentMessageKind.Accepted, [], ids.at(-1))]);
        link.a.receive();
        intent.record(tick, "input", Array<number>(300).fill(tick));
    }
    expect(ids).toEqual(Array.from({ length: 100 }, (_, i) => i + 1));
});
