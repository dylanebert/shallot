import { expect, test } from "bun:test";
import { World } from "../../engine";
import {
    ClientPrediction,
    IntentReceiver,
    IntentSender,
    ReplicationReceiver,
    ReplicationSender,
    seededLink,
} from ".";
import { decodeInputs } from "./intent";

const faults = { minDelay: 0, maxDelay: 0, loss: 0, duplication: 0 };

test("same-tick samples replace inputs without sending twice, and decreasing ticks throw", () => {
    const link = seededLink(1, faults);
    const sender = new IntentSender(link.b);
    sender.record(1, "input", [1]);
    sender.record(1, "input", [2]);
    sender.flush();
    expect(decodeInputs(link.a.receive()[0]!)).toEqual([{ tick: 1, values: [2] }]);
    sender.record(1, "input", [3]);
    sender.flush();
    expect(link.a.receive()).toEqual([]);
    sender.record(2, "input", [4]);
    sender.flush();
    expect(decodeInputs(link.a.receive()[0]!)[0]).toEqual({ tick: 1, values: [3] });
    expect(() => sender.record(1, "input", [5])).toThrow();
});

test("an input and operation at the same tick stay independent through acceptance and watermarks", () => {
    for (const inputFirst of [false, true]) {
        const world = new World();
        try {
            const link = seededLink(2, faults);
            const sender = new IntentSender(link.b);
            const receiver = new IntentReceiver(link.a);
            const prediction = new ClientPrediction(
                world,
                [],
                link.b,
                () => undefined,
                () => {},
            );
            const input = sender.record(1, "input", [10]);
            const operation = sender.record(1, "operation", [20]);
            expect(operation.id).toBe(1);
            expect("id" in input).toBe(false);
            prediction.accept(inputFirst ? [input, operation] : [operation, input]);
            expect(prediction.inputs.get(1)?.values).toEqual([10]);
            expect(prediction.records.get(1)?.values).toEqual([20]);
            prediction.accept([], 1);
            expect(prediction.inputs.get(1)?.values).toEqual([10]);
            expect(prediction.records.size).toBe(0);
            sender.flush();
            receiver.receive(link.a.receive(), 1);
            expect(receiver.inputAt(1)?.values).toEqual([10]);
            expect(receiver.records.get(1)?.values).toEqual([20]);
            sender.flush(link.b.receive());
            expect(sender.inputs.get(1)?.values).toEqual([10]);
            expect(sender.records.size).toBe(0);
        } finally {
            world.dispose();
        }
    }
});

test("command age smooths once per server tick and travels once through ordered frames", () => {
    const server = new World();
    const client = new World();
    try {
        const link = seededLink(3, faults);
        const intents = new IntentReceiver(link.a);
        const input = new IntentSender(link.b);
        const sender = new ReplicationSender(server, [], [], 1, link.a);
        const receiver = new ReplicationReceiver(client, [], link.b);
        const samples: number[][] = [];
        receiver.onCommandAge = (tick, age) => samples.push([tick, age]);
        input.record(8, "input", [8]);
        input.flush();
        intents.receive(link.a.receive(), 10);
        expect(intents.commandAge(10)).toBe(0.25);
        expect(intents.commandAge(10)).toBe(0.25);
        expect(intents.commandAge(11)).toBe(0.59375);
        sender.record(10, [], 0.25);
        sender.record(11, [], 0.59375);
        sender.flush();
        receiver.flush("latest");
        receiver.flush("latest");
        expect(samples).toEqual([
            [10, 0.25],
            [11, 0.59375],
        ]);
    } finally {
        server.dispose();
        client.dispose();
    }
});
