import { test } from "bun:test";
import { type ComponentRecord, component, text, World } from "../../engine";
import {
    type AcceptedIntent,
    type Connection,
    IntentMessageKind,
    Replicate,
    ReplicationReceiver,
    ReplicationSender,
} from ".";
import { decodeAccepted, encodeIntent } from "./intent";

const Payload = component("ReceiverModelPayload", {
    number: {
        name: "receiver-model-byte",
        ctor: Uint8Array,
        lanes: 1 as const,
        encode(value: number) {
            if (value < 0) throw new Error("Receiver model cannot write a negative value");
            return value;
        },
        decode(raw: number) {
            if (raw === 255) throw new Error("Receiver model refuses byte 255");
            return raw === 254 ? -1 : raw;
        },
    },
    label: text(12),
});

function require(condition: unknown, message: string): asserts condition {
    if (!condition) throw new Error(message);
}

function model(seed: number, mode: "all" | "latest") {
    let state = seed;
    const draw = (bound: number) => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return Math.floor((state / 0x100000000) * bound);
    };
    const authority = new World();
    const replica = new World();
    const eid = authority.create();
    authority.add(eid, Replicate);
    authority.add(eid, Payload);
    const field = authority.storage(Payload);
    const schema = [Payload];
    const rules = [new Map([[1, schema]])];
    const sent = new Map<number, { bytes: Uint8Array; number: number; label: string }>();
    const operation = (tick: number): AcceptedIntent => ({
        id: tick + 1,
        tick,
        kind: "operation",
        values: [tick],
        text: `🐟${tick}`,
        lateness: 0,
    });
    const produce = (start: number, count: number) => {
        const states: ComponentRecord<typeof Payload>[] = [];
        const send: Connection = {
            close() {},
            receive: () => [], // Keep the image as the baseline for this epoch.
            send(bytes) {
                const tick = new DataView(
                    bytes.buffer,
                    bytes.byteOffset,
                    bytes.byteLength,
                ).getUint32(8, true);
                sent.set(tick, { bytes, ...states[tick - start]! });
            },
        };
        const sender = new ReplicationSender(authority, schema, rules, 1, send);
        for (let tick = start; tick < start + count; tick++) {
            field.number.set(eid, draw(254));
            field.label.set(eid, `é${draw(4096)}`);
            states.push(authority.get(eid, Payload)!);
            sender.record(tick, [operation(tick)]);
        }
        sender.flush();
    };

    let generation = 1;
    let closed = 0;
    let errors = 0;
    let arriving: Uint8Array[] = [];
    const connection: Connection = {
        get generation() {
            return generation;
        },
        get closed() {
            return closed !== 0;
        },
        receive() {
            const bytes = arriving;
            arriving = [];
            return bytes;
        },
        send() {},
        close() {
            closed++;
        },
        onError() {
            errors++;
        },
    };
    const observed: { tick: number; applied: number; number: number; label: string }[] = [];
    const reconciled: { tick: number; intents: readonly AcceptedIntent[] }[] = [];
    const receiver = new ReplicationReceiver(
        replica,
        schema,
        connection,
        (tick) => {
            const row = replica.get([...receiver.entities.values()][0]!, Payload)!;
            observed.push({ tick, applied: receiver.confirmed, ...row });
        },
        (tick, _rows, intents, apply, preceding) => {
            for (const state of preceding) {
                reconciled.push(state);
                state.apply();
            }
            reconciled.push({ tick, intents });
            apply();
        },
    );
    let applied = -1;
    const drain = (ticks: number[], expected: number[], extra: Uint8Array[] = []) => {
        const before = observed.length;
        arriving = [...ticks.map((tick) => sent.get(tick)!.bytes), ...extra];
        const messages = receiver.flush(mode);
        const actual = observed.slice(before);
        require(actual.length ===
            expected.length, `delivered [${ticks}], expected [${expected}], confirmed [${actual.map((row) => row.tick)}]`);
        for (const [i, tick] of expected.entries()) {
            const row = actual[i]!;
            const source = sent.get(tick)!;
            const confirmation = reconciled[before + i]!;
            require(row.tick === tick &&
                row.applied ===
                    tick, `confirmation ${row.applied} did not match applied tick ${tick}`);
            require(row.number === source.number &&
                row.label === source.label, `state at ${tick} was not sent`);
            require(confirmation.tick === tick &&
                confirmation.intents.length === 1 &&
                confirmation.intents[0]!.text === operation(tick).text, `wrong intents at ${tick}`);
        }
        applied = expected.at(-1) ?? applied;
        require(receiver.confirmed === applied, "confirmed passed applied state");
        return messages;
    };

    try {
        // Seed 0 reproduces the missing first baseline; seed 1 delivers its baseline late.
        // Seed 2 reaches a codec rejection without pre-image reordering on the old receiver.
        if (seed % 4 === 0) {
            produce(0, 2);
            drain([1], []);
        }
        for (const start of [8, 16]) {
            if (start === 16) {
                generation++;
                applied = -1;
                replica.jumpToTick(start + draw(3));
            }
            const count = 4 + draw(3);
            produce(start, count);
            const accepted = encodeIntent(
                IntentMessageKind.Accepted,
                [operation(start)],
                start + 1,
            );
            const seen = new Set<number>();
            let next = start;
            const deliver = (ticks: number[], extras?: Uint8Array[]) => {
                for (const tick of ticks) seen.add(tick);
                const expected: number[] = [];
                while (seen.has(next)) expected.push(next++);
                return drain(ticks, expected, extras);
            };
            if (seed % 4 === 1 || (seed > 3 && draw(2))) {
                deliver([start + 1, start + 1]);
                replica.jumpToTick(start + draw(4));
            }
            const messages = deliver([start], [accepted]);
            require(messages.length === 1, "standalone Accepted was lost");
            const { records, watermark } = decodeAccepted(messages[0]!);
            require(watermark === start + 1 &&
                records.length === 1 &&
                records[0]!.text === operation(start).text, "standalone Accepted was altered");
            for (let tick = start + count - 1; tick > start; tick--)
                if (draw(3)) deliver(draw(2) ? [tick] : [tick, tick]);
            // Repair the seeded loss and reorder once; no extra resend after delivery.
            for (let tick = start + 1; tick < start + count; tick++)
                if (!seen.has(tick)) deliver([tick]);
            require(next === start + count &&
                receiver.confirmed === next - 1, "confirmation did not recover after delivery");
        }

        produce(31, 1);
        produce(32, 1);
        let bad: Uint8Array = sent.get(32)!.bytes.slice();
        switch (seed % 4) {
            case 0:
                bad = bad.subarray(0, 12);
                break;
            case 1:
                bad = encodeIntent(IntentMessageKind.Accepted, [operation(32)]);
                bad[bad.length - 1] = 0xff;
                break;
            default: {
                // A valid structure whose first lane is rejected on decode (2) or write (3).
                const view = new DataView(bad.buffer);
                bad[4 + view.getUint32(4, true) * 4] = seed % 4 === 2 ? 255 : 254;
            }
        }
        require(drain([31], [31], [bad]).length === 0, "rejected packet escaped receiver");
        require(closed === 1 &&
            errors === 1, `refusal must close and report once, got ${closed}/${errors}`);
        drain([], []);
        require(closed === 1 && errors === 1, "terminal flush repeated the failure");
    } catch (cause) {
        throw new Error(
            `receiver model seed=${seed} mode=${mode}: ${String(cause)}; replay with SHALLOT_MULTIPLAYER_SEED=${seed}`,
            { cause },
        );
    } finally {
        replica.dispose();
        authority.dispose();
    }
}

for (const mode of ["all", "latest"] as const) {
    test(`${mode}: seeded receiver agrees with sent state and makes progress through faults`, () => {
        const replay = process.env.SHALLOT_MULTIPLAYER_SEED;
        if (replay === undefined) {
            for (let seed = 0; seed < 2048; seed++) model(seed, mode);
        } else {
            const seed = Number(replay);
            require(/^\d+$/.test(replay) &&
                Number.isInteger(seed) &&
                seed <= 0xffffffff, "SHALLOT_MULTIPLAYER_SEED must be a decimal u32");
            model(seed, mode);
        }
    });
}
