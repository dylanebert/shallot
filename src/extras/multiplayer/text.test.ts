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
    const row = () => [
        ...server.fieldStorage(Label, "text").column.slice(eid * 10, (eid + 1) * 10),
    ];
    const original = [8, 0, 0xef, 0xbb, 0xbf, 0, 0xf0, 0x9f, 0x90, 0x9f];
    expect(row()).toEqual(original);
    field.set(eid, "a");
    expect(field.get(eid)).toBe("a");
    expect(row()).toEqual([1, 0, 97, 0, 0, 0, 0, 0, 0, 0]);
    expect(() => field.set(eid, `${exact}!`)).toThrow();
    expect(() => field.set(eid, "\ud800")).toThrow();
    expect(field.get(eid)).toBe("a");
    server.remove(eid, Label);
    expect(field.get(eid)).toBe("");
    server.restore(saved);
    expect(server.get(eid, Label)).toEqual({ text: exact, count: 9 });
    expect(row()).toEqual(original);

    const { a, b } = link();
    const sender = new ReplicationSender(server, [Label], [new Map([[1, [Label]]])], 1, a);
    const receiver = new ReplicationReceiver(client, [Label], b);
    sender.record(0);
    sender.flush();
    const [image] = b.receive();
    expect(image).toBeDefined();
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

const WiderLabel = component("WiderTextLabel", { text: text(9), count: u8 });

for (const defect of ["invalid UTF-8", "over-capacity text", "truncated frame"] as const) {
    test(`${defect} closes the receiver once without confirming unapplied state`, () => {
        const server = new World();
        const client = new World();
        const eid = server.create();
        server.add(eid, WiderLabel, { text: "initial", count: 9 });
        server.add(eid, Replicate);
        const { a, b } = link();
        const sender = new ReplicationSender(
            server,
            [WiderLabel],
            [new Map([[1, [WiderLabel]]])],
            1,
            a,
        );
        const errors: Error[] = [];
        let closes = 0;
        const applied: number[] = [];
        const connection = {
            ...b,
            close() {
                closes++;
                b.close();
            },
            onError(error: Error) {
                errors.push(error);
            },
        };
        const receiver = new ReplicationReceiver(client, [Label], connection, (tick) => {
            expect(receiver.confirmed).toBe(tick);
            applied.push(tick);
        });
        sender.record(0);
        sender.flush();
        receiver.flush("latest");
        expect(receiver.confirmed).toBe(0);

        const field = server.storage(WiderLabel).text;
        field.set(eid, "first");
        sender.record(1);
        field.set(eid, "good");
        sender.record(2);
        field.set(eid, defect === "over-capacity text" ? "123456789" : "a");
        sender.record(3);
        sender.flush();
        for (const frame of b.receive()) {
            const view = new DataView(frame.buffer);
            if (view.getUint32(8, true) !== 3) {
                a.send(frame);
                continue;
            }
            const offset = 4 + view.getUint32(4, true) * 4;
            if (defect === "invalid UTF-8") frame[offset + 2] = 0xff;
            a.send(defect === "truncated frame" ? frame.subarray(0, 12) : frame);
        }
        expect(() => receiver.flush("latest")).not.toThrow();
        expect(applied).toEqual([0, 1, 2]);
        expect(receiver.confirmed).toBe(2);
        expect(client.get([...receiver.entities.values()][0]!, Label)).toEqual({
            text: "good",
            count: 9,
        });
        expect(b.closed).toBe(true);
        expect(() => receiver.flush("latest")).not.toThrow();
        expect(closes).toBe(1);
        expect(errors).toHaveLength(1);
        client.dispose();
        server.dispose();
    });
}

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
    // Supply every declared byte and stay below the packet cap, so only capacity rejects it.
    const textOnly = encodeIntent(IntentMessageKind.Records, [{ ...operation, values: [] }]);
    const oversized = new Uint8Array(textOnly.length + 1);
    oversized.set(textOnly);
    oversized[36] = 1;
    for (const bad of [invalid, oversized, bytes.subarray(0, -1)]) {
        expect(() => decodeIntent(bad)).toThrow();
        expect(isClientMessage(bad)).toBe(false);
    }
});
