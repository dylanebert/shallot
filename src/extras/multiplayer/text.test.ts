import { expect, test } from "bun:test";
import { component, text, u8, World } from "../../engine";
import {
    IntentMessageKind,
    IntentSender,
    MAX_MESSAGE_BYTES,
    Replicate,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";
import { isClientMessage } from "./game";
import { decodeIntent, encodeIntent, IntentReceiver } from "./intent";

const Label = component("BoundedTextLabel", { text: text(8), count: u8 });
const link = () => seededLink(1, { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 });

test("bounded text columns restore and replicate whole UTF-8 values alongside numeric fields", () => {
    const server = new World();
    const client = new World();
    const defaults = component(
        "BoundedTextDefaults",
        { value: text(2) },
        { defaults: () => ({ value: "é" }) },
    );
    server.registry.register(defaults);
    const extra = server.create();
    server.add(extra, defaults);
    expect(server.get(extra, defaults)?.value).toBe("é");
    const eid = server.create();
    const exact = "\uFEFF\0🐟";
    server.add(eid, Label, { text: exact, count: 9 });
    server.add(eid, Replicate);
    const field = server.storage(Label).text;
    const saved = server.snapshot();
    field.set(eid, "a");
    expect(field.get(eid)).toBe("a");
    expect(() => field.set(eid, `${exact}!`)).toThrow();
    expect(() => field.set(eid, "\ud800")).toThrow();
    expect(field.get(eid)).toBe("a");
    server.remove(eid, Label);
    expect(field.get(eid)).toBe("");
    server.restore(saved);
    expect(server.get(eid, Label)).toEqual({ text: exact, count: 9 });

    const { a, b } = link();
    const sender = new ReplicationSender(server, [Label], [new Map([[1, [Label]]])], 1, a);
    const receiver = new ReplicationReceiver(client, [Label], b);
    sender.record(0);
    sender.flush();
    const [image] = b.receive();
    expect(image).toBeDefined();
    const valueOffset = 4 + new DataView(image!.buffer).getUint32(4, true) * 4;
    for (const [offset, bad] of [
        [valueOffset, 9],
        [valueOffset + 2, 0xff],
    ] as const) {
        const malformed = image!.slice();
        malformed[offset] = bad;
        const isolated = link();
        isolated.a.send(malformed);
        expect(() => new ReplicationReceiver(client, [Label], isolated.b).flush()).toThrow();
    }
    a.send(image!);
    receiver.flush();
    const remote = [...receiver.entities.values()][0]!;
    expect(client.get(remote, Label)).toEqual({ text: exact, count: 9 });
    sender.flush();
    field.set(eid, "");
    sender.record(1);
    sender.flush();
    receiver.flush();
    expect(client.get(remote, Label)).toEqual({ text: "", count: 9 });
    client.dispose();
    server.dispose();
});

test("operation text is byte-bounded, validates at ingress and counts toward upstream pacing", () => {
    const { a, b } = link();
    const sender = new IntentSender(a);
    const payload = "🐟".repeat(256);
    const operation = sender.record(1, "operation", Array(379).fill(0), payload);
    sender.record(1, "operation", [], "");
    sender.flush();
    const packets = b.receive();
    expect(packets).toHaveLength(1);
    const bytes = packets[0]!;
    expect(bytes.length).toBeLessThanOrEqual(MAX_MESSAGE_BYTES);
    expect(isClientMessage(bytes)).toBe(true);
    expect(decodeIntent(bytes)).toEqual([{ ...operation, lateness: 0 }]);
    const receiver = new IntentReceiver(b);
    receiver.receive(
        packets,
        1,
        (record) => record.kind === "operation" && record.text === payload,
    );
    const accepted = a.receive();
    expect(decodeIntent(accepted[0]!)).toEqual([{ ...operation, lateness: 0, slack: 0 }]);
    sender.flush(accepted);
    expect(sender.records.has(operation.id)).toBe(false);
    expect(() => sender.record(1, "operation", Array(380).fill(0), payload)).toThrow();
    for (const text of [`${payload}!`, "\ud800"]) {
        expect(() => encodeIntent(IntentMessageKind.Records, [{ ...operation, text }])).toThrow();
    }
    const prefix = 36 + operation.values.length * 8;
    const invalid = bytes.slice();
    invalid[prefix + 2] = 0xff;
    const oversized = bytes.slice();
    oversized[prefix] = 1;
    for (const bad of [invalid, oversized, bytes.subarray(0, -1)]) {
        expect(() => decodeIntent(bad)).toThrow();
        expect(isClientMessage(bad)).toBe(false);
    }
});
