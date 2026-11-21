import { test } from "bun:test";
import { component, text, World } from "../../engine";
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
    label: text(12),
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
});

const Companion = component(
    "ReceiverModelCompanion",
    { label: text(4) },
    {
        defaults: { label: "éé" },
    },
);
const Trigger = component("ReceiverModelTrigger", {}, { requires: [Companion] });

function readRow(world: World, eid: number) {
    return {
        ...world.get(eid, Payload)!,
        trigger: world.has(eid, Trigger),
        companion: world.get(eid, Companion)?.label,
    };
}
type Row = ReturnType<typeof readRow>;

function sameRow(a: Row, b: Row): boolean {
    return (
        a.number === b.number &&
        a.label === b.label &&
        a.trigger === b.trigger &&
        a.companion === b.companion
    );
}

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
    for (const world of [authority, replica]) {
        world.registry.register(Companion);
        world.registry.register(Trigger);
    }
    const eid = authority.create();
    authority.add(eid, Replicate);
    authority.add(eid, Payload);
    const field = authority.storage(Payload);
    const schema = [Payload, Trigger, Companion];
    const rules = [new Map([[1, schema]])];
    const sent = new Map<number, Row & { bytes: Uint8Array }>();
    const operation = (tick: number): AcceptedIntent => ({
        id: tick + 1,
        tick,
        kind: "operation",
        values: [tick, -0],
        text: `🐟${tick}`,
        lateness: tick % 3,
        ...(tick % 2 ? { slack: tick % 5 } : {}),
    });
    let acknowledgements: Uint8Array[] = [];
    const produce = (start: number, count: number) => {
        acknowledgements = [];
        const transmitted: number[] = [];
        const states: Row[] = [];
        const send: Connection = {
            close() {},
            receive() {
                const messages = acknowledgements;
                acknowledgements = [];
                return messages;
            },
            send(bytes) {
                const tick = new DataView(
                    bytes.buffer,
                    bytes.byteOffset,
                    bytes.byteLength,
                ).getUint32(8, true);
                sent.set(tick, { bytes, ...states[tick - start]! });
                transmitted.push(tick);
            },
        };
        const sender = new ReplicationSender(authority, schema, rules, 1, send);
        const record = (tick: number) => {
            field.number.set(eid, draw(254));
            field.label.set(eid, `é${draw(4096)}`);
            if (seed % 4 === 0 && tick !== start && tick % 4 !== 0) {
                authority.add(eid, Companion, { label: `c${tick % 100}` });
                if (!authority.has(eid, Trigger)) authority.add(eid, Trigger);
            } else {
                authority.remove(eid, Trigger);
                authority.remove(eid, Companion);
            }
            states.push(readRow(authority, eid));
            sender.record(tick, [operation(tick)]);
        };
        let clock = 0;
        const flush = (advance: number) => {
            transmitted.length = 0;
            sender.flush((clock += advance));
            return transmitted.slice();
        };
        for (let tick = start; tick < start + count; tick++) record(tick);
        flush(0);
        return { sender, record, flush };
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
        send(bytes) {
            acknowledgements.push(bytes);
        },
        close() {
            closed++;
        },
        onError() {
            errors++;
        },
    };
    let attached: number | undefined;
    replica.observeMembership(Companion, (eid, present) => {
        if (!present) return;
        // A consumer drains the attachment's field mark. A second write would publish it again.
        const dirty = replica.fieldStorage(Companion, "label").dirty;
        dirty[eid >>> 5] &= ~(1 << (eid & 31));
        attached = eid;
    });
    const observed: (Row & { tick: number; applied: number; duplicateWrite: boolean })[] = [];
    const reconciled: { tick: number; intents: readonly AcceptedIntent[] }[] = [];
    const batches: number[] = [];
    const checkIntent = (records: readonly AcceptedIntent[], tick: number) => {
        const actual = records[0];
        const expected = operation(tick);
        require(records.length === 1 &&
            actual &&
            actual.id === expected.id &&
            actual.tick === expected.tick &&
            actual.kind === expected.kind &&
            actual.lateness === expected.lateness &&
            actual.slack === expected.slack &&
            actual.text === expected.text &&
            actual.values.length === expected.values.length &&
            actual.values.every((value, i) =>
                Object.is(value, expected.values[i]),
            ), `intent at ${tick} differed from the sender`);
    };
    const receiver = new ReplicationReceiver(
        replica,
        schema,
        connection,
        (tick) => {
            const row = readRow(replica, [...receiver.entities.values()][0]!);
            const duplicateWrite =
                attached !== undefined &&
                !!(
                    replica.fieldStorage(Companion, "label").dirty[attached >>> 5]! &
                    (1 << (attached & 31))
                );
            attached = undefined;
            observed.push({ tick, applied: receiver.confirmed, duplicateWrite, ...row });
        },
        (tick, _rows, intents, apply, preceding) => {
            batches.push(tick);
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
        const batchStart = batches.length;
        arriving = [...ticks.map((tick) => sent.get(tick)!.bytes), ...extra];
        const messages = receiver.flush(mode);
        const actualBatches = batches.slice(batchStart);
        const expectedBatches = mode === "all" ? expected : expected.slice(-1);
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
            require(sameRow(row, source), `state at ${tick} was not sent`);
            require(!row.duplicateWrite, `recursively attached companion was written twice at ${tick}`);
            require(confirmation.tick === tick, `wrong reconciliation tick at ${tick}`);
            checkIntent(confirmation.intents, tick);
        }
        applied = expected.at(-1) ?? applied;
        require(receiver.confirmed === applied, "confirmed passed applied state");
        if (applied >= 0) {
            const live = readRow(replica, [...receiver.entities.values()][0]!);
            const source = sent.get(applied)!;
            require(sameRow(live, source), `live replica differs from confirmed tick ${applied}`);
        }
        require(actualBatches.join() ===
            expectedBatches.join(), `reconciled [${actualBatches}], expected [${expectedBatches}]`);
        return messages;
    };

    try {
        // Seed 0 reproduces the missing first baseline; seed 1 delivers its baseline late.
        // Seed 2 reaches a codec rejection without pre-image reordering on the old receiver.
        if (seed % 4 === 0) {
            produce(0, 2);
            drain([1], []);
        }
        const overflow = seed % 32 === 5;
        for (const start of [8, overflow ? 64 : 16]) {
            if (start !== 8) {
                generation++;
                applied = -1;
                // Reconnect at the same local tick that sent the last generation's ack.
                replica.jumpToTick(replica.time.fixedTick + Number(seed === 3));
            }
            const count = overflow && start === 8 ? 40 + draw(4) : 4 + draw(3);
            const peer = produce(start, count);
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
            if (overflow && start === 8) {
                const deltas = Array.from({ length: count - 1 }, (_, i) => start + count - 1 - i);
                drain(deltas, []);
                for (let tick = start + 1; tick <= start + 32; tick++) seen.add(tick);
            } else if (seed % 4 === 1 || (seed > 3 && draw(2))) {
                deliver([start + 1, start + 1]);
                replica.jumpToTick(replica.time.fixedTick);
            }
            const messages = deliver([start], [accepted]);
            require(messages.length === 1, "standalone Accepted was lost");
            const { records, watermark } = decodeAccepted(messages[0]!);
            require(watermark === start + 1, "standalone Accepted watermark was altered");
            checkIntent(records, start);
            require(acknowledgements.length ===
                1, `first image at ${start} was not acknowledged once after jumpToTick`);
            drain([], []);
            require(acknowledgements.length === 1, "an empty drain repeated the acknowledgement");
            peer.flush(1);
            require(peer.sender.acknowledged ===
                next - 1, "ack did not advance the sender's baseline");
            if (!(overflow && start === 8)) {
                for (let tick = start + count - 2; tick > start; tick--) {
                    if (!draw(3)) continue;
                    replica.jumpToTick(replica.time.fixedTick + 1);
                    deliver(draw(2) ? [tick] : [tick, tick]);
                    peer.flush(1);
                }
            }
            const retry = peer.flush(6000);
            require(retry.length > 0 &&
                retry.every(
                    (tick) => !seen.has(tick),
                ), "sender did not retry only its missing deltas");
            replica.jumpToTick(replica.time.fixedTick + 1);
            deliver(retry);
            require(next === start + count &&
                receiver.confirmed === next - 1, "confirmation did not recover by retransmission");
            peer.flush(1);
            require(peer.sender.acknowledged === next - 1, "sender did not acknowledge recovery");
            peer.record(next);
            require(peer.flush(1).join() ===
                String(next), "sender kept retrying an acknowledged frame");
            const newest = new DataView(sent.get(next)!.bytes.buffer);
            require(newest.getUint32(12, true) ===
                next - 1, "sender did not encode against the acknowledged baseline");
            replica.jumpToTick(replica.time.fixedTick + 1);
            deliver([next]);
            peer.flush(1);
        }

        const terminal = overflow ? 128 : 31;
        produce(terminal, 1);
        produce(terminal + 1, 1);
        let bad: Uint8Array = sent.get(terminal + 1)!.bytes.slice();
        switch (seed % 4) {
            case 0:
                bad = bad.subarray(0, 12);
                break;
            case 1:
                bad = encodeIntent(IntentMessageKind.Accepted, [operation(terminal + 1)]);
                bad[bad.length - 1] = 0xff;
                break;
            default: {
                // The text lane precedes the numeric lane rejected on decode (2) or write (3).
                const view = new DataView(bad.buffer);
                const first = 4 + view.getUint32(4, true) * 4;
                bad[first + 2 + view.getUint16(first, true)] = seed % 4 === 2 ? 255 : 254;
            }
        }
        require(drain([terminal], [terminal], [bad]).length ===
            0, "rejected packet escaped receiver");
        require(closed === 1 &&
            errors === 1, `refusal must close and report once, got ${closed}/${errors}`);
        produce(terminal + 2, 2);
        const accepted = encodeIntent(IntentMessageKind.Accepted, [operation(terminal + 2)]);
        require(drain([terminal + 2, terminal + 3], [], [accepted]).length ===
            0, "terminal receiver routed a valid packet");
        require(drain([], [], [bad]).length === 0, "terminal receiver routed a malformed packet");
        require(closed === 1 && errors === 1, "terminal flush repeated the failure");
        require(acknowledgements.length === 0, "terminal receiver acknowledged a packet");
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
