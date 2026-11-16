import { expect, test } from "bun:test";
import { decodeInputs, encodeInputs, IntentReceiver, IntentSender } from "./intent";
import { type Connection, seededLink } from "./link";

const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };

function capture(connection: Connection, packets: Uint8Array[]): Connection {
    return {
        close: () => connection.close(),
        receive: () => connection.receive(),
        send(bytes) {
            packets.push(bytes);
            connection.send(bytes);
        },
    };
}

test("input upstream bytes and four-tick packets are independent of round trip", () => {
    const totals: number[] = [];
    for (const roundTrip of [0, 100, 300]) {
        const link = seededLink(42, {
            ...faults,
            minDelay: roundTrip / 2,
            maxDelay: roundTrip / 2,
        });
        const packets: Uint8Array[] = [];
        const sender = new IntentSender(capture(link.b, packets));
        const receiver = new IntentReceiver(link.a);
        for (let tick = 1; tick <= 600; tick++) {
            link.advance((tick * 1000) / 60);
            sender.record(tick, "input", [tick]);
            sender.flush(link.b.receive());
            sender.flush();
            receiver.receive(link.a.receive(), tick);
            expect(sender.inputs.size).toBeLessThanOrEqual(4);
            expect(sender.records.size).toBe(0);
        }
        expect(packets.length).toBe(600);
        for (const packet of packets) expect(decodeInputs(packet).length).toBeLessThanOrEqual(4);
        totals.push(packets.reduce((sum, packet) => sum + packet.length, 0) / 10);
    }
    expect(totals).toEqual([4310.4, 4310.4, 4310.4]);
});

test("3000 lossy ticks apply the latest eligible received input within bounded stores", () => {
    const link = seededLink(916, { ...faults, minDelay: 0, maxDelay: 80, loss: 0.2 });
    const sender = new IntentSender(link.b);
    const receiver = new IntentReceiver(link.a, 12);
    const received = new Map<number, number>();
    for (let tick = 1; tick <= 3000; tick++) {
        link.advance((tick * 1000) / 60);
        sender.record(tick + 8, "input", [tick + 8]);
        sender.flush();
        const packets = link.a.receive();
        for (const packet of packets)
            for (const input of decodeInputs(packet)) received.set(input.tick, input.values[0]!);
        receiver.receive(packets, tick);
        let latest = -1;
        for (const key of received.keys()) if (key <= tick) latest = Math.max(latest, key);
        expect(receiver.inputAt(tick)?.values[0]).toBe(
            latest < 0 ? undefined : received.get(latest),
        );
        expect(sender.inputs.size).toBeLessThanOrEqual(4);
        expect(receiver.inputs.size).toBeLessThanOrEqual(13);
        expect(receiver.records.size).toBe(0);
    }
    expect(link.counts.dropped).toBeGreaterThan(0);
});

test("newest late input applies now; older late inputs never replace it", () => {
    const link = seededLink(1, faults);
    const receiver = new IntentReceiver(link.a);
    receiver.receive(
        [
            encodeInputs([
                { tick: 3, values: [3] },
                { tick: 1, values: [1] },
                { tick: 2, values: [2] },
            ]),
        ],
        10,
    );
    expect(receiver.inputAt(10)?.values).toEqual([3]);
    receiver.receive([encodeInputs([{ tick: 2, values: [200] }])], 11);
    expect(receiver.inputAt(11)?.values).toEqual([3]);
    receiver.receive([encodeInputs([{ tick: 3, values: [300] }])], 12);
    expect(receiver.inputAt(12)?.values).toEqual([3]);
    expect([...receiver.inputs.keys()]).toEqual([3]);
});

test("missing ticks hold input, but never apply a future tick early", () => {
    const link = seededLink(1, faults);
    const receiver = new IntentReceiver(link.a);
    receiver.receive(
        [
            encodeInputs([
                { tick: 2, values: [2] },
                { tick: 5, values: [5] },
            ]),
        ],
        1,
    );
    expect(receiver.inputAt(1)).toBeUndefined();
    expect(receiver.inputAt(2)?.values).toEqual([2]);
    expect(receiver.inputAt(4)?.values).toEqual([2]);
    expect(receiver.inputAt(5)?.values).toEqual([5]);
    expect(receiver.inputAt(6)?.values).toEqual([5]);
});

test("ticks beyond maxAhead and invalid inputs are refused without operation echoes", () => {
    const link = seededLink(1, faults);
    const receiver = new IntentReceiver(link.a, 2);
    receiver.receive(
        [
            encodeInputs([
                { tick: 12, values: [12] },
                { tick: 13, values: [13] },
                { tick: 11, values: [NaN] },
            ]),
        ],
        10,
        (input) => input.values.every(Number.isFinite),
    );
    expect([...receiver.inputs.keys()]).toEqual([12]);
    expect(receiver.records.size).toBe(0);
    expect(link.b.receive()).toEqual([]);
});

test("a cumulative press lost on its own tick arrives on a later tick", () => {
    const link = seededLink(7, faults);
    const sender = new IntentSender(link.b);
    const receiver = new IntentReceiver(link.a);
    sender.record(1, "input", [0]);
    sender.flush();
    receiver.receive(link.a.receive(), 1);
    sender.record(2, "input", [1]);
    sender.flush();
    link.a.receive();
    expect(receiver.inputAt(2)?.values).toEqual([0]);
    sender.record(3, "input", [1]);
    sender.flush();
    receiver.receive(link.a.receive(), 3);
    expect(receiver.inputAt(3)?.values).toEqual([1]);
});

test("input packets refuse oversized windows and malformed tick records", () => {
    expect(() =>
        encodeInputs(Array.from({ length: 5 }, (_, tick) => ({ tick, values: [] }))),
    ).toThrow();
    const packet = encodeInputs([{ tick: 1, values: [1] }]);
    expect(() => decodeInputs(packet.subarray(0, packet.length - 1))).toThrow();
    expect(() =>
        decodeInputs(
            encodeInputs([
                { tick: 1, values: [] },
                { tick: 1, values: [] },
            ]),
        ),
    ).toThrow();
});
