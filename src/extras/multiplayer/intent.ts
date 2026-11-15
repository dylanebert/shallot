import type { Connection } from "./link";

export const IntentMessageKind = { Records: 4, Accepted: 5 } as const;

/** IDs are consecutive within one player's connection. Values belong to the game. */
export interface IntentRecord {
    id: number;
    tick: number;
    kind: "input" | "operation";
    values: number[];
}
export interface AcceptedIntent extends IntentRecord {
    /** Assigned tick minus requested tick, in fixed ticks. */
    lateness: number;
}

export function encodeIntent(kind: number, records: readonly IntentRecord[]): Uint8Array {
    const bytes = new Uint8Array(
        8 + records.reduce((size, r) => size + 20 + r.values.length * 8, 0),
    );
    const view = new DataView(bytes.buffer);
    view.setUint32(0, kind, true);
    view.setUint32(4, records.length, true);
    let offset = 8;
    for (const r of records) {
        const words = [
            r.id,
            r.tick,
            Number(r.kind === "operation"),
            (r as AcceptedIntent).lateness ?? 0,
            r.values.length,
        ];
        for (const word of words) {
            view.setUint32(offset, word, true);
            offset += 4;
        }
        for (const value of r.values) {
            view.setFloat64(offset, value, true);
            offset += 8;
        }
    }
    return bytes;
}
export function decodeIntent(bytes: Uint8Array): AcceptedIntent[] {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const kind = bytes.length >= 8 ? view.getUint32(0, true) : 0;
    if (kind !== IntentMessageKind.Records && kind !== IntentMessageKind.Accepted)
        throw new Error("Invalid intent packet");
    const records: AcceptedIntent[] = [];
    let offset = 8;
    for (let i = 0; i < view.getUint32(4, true); i++) {
        const id = view.getUint32(offset, true),
            tick = view.getUint32(offset + 4, true);
        const tag = view.getUint32(offset + 8, true);
        if (tag > 1) throw new Error("Invalid intent kind");
        const kind = tag ? "operation" : "input";
        const lateness = view.getUint32(offset + 12, true),
            count = view.getUint32(offset + 16, true);
        offset += 20;
        const values: number[] = [];
        for (let lane = 0; lane < count; lane++) {
            values.push(view.getFloat64(offset, true));
            offset += 8;
        }
        records.push({ id, tick, kind, lateness, values });
    }
    if (offset !== bytes.length) throw new Error("Invalid intent size");
    return records;
}

/** One player's accepted log. Retrying a record never changes its assigned tick. */
export class IntentReceiver {
    readonly records = new Map<number, AcceptedIntent>();
    readonly connection: Connection;
    constructor(connection: Connection) {
        this.connection = connection;
    }
    /** Route messages returned by ReplicationSender.flush here before the next fixed tick. */
    receive(messages: readonly Uint8Array[], nextTick: number): Uint8Array[] {
        if (!Number.isInteger(nextTick) || nextTick < 0 || nextTick >= 0xffffffff)
            throw new Error("Invalid intent tick");
        const unhandled: Uint8Array[] = [];
        for (const bytes of messages) {
            if (
                bytes.length < 8 ||
                new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
                    0,
                    true,
                ) !== IntentMessageKind.Records
            ) {
                unhandled.push(bytes);
                continue;
            }
            for (const record of decodeIntent(bytes)) {
                if (this.records.has(record.id)) continue;
                const tick = Math.max(nextTick, record.tick);
                this.records.set(record.id, { ...record, tick, lateness: tick - record.tick });
            }
        }
        this.connection.send(encodeIntent(IntentMessageKind.Accepted, [...this.records.values()]));
        return unhandled;
    }
}

/** Unaccepted records repeat on every flush, including discrete operations. */
export class IntentSender {
    readonly records = new Map<number, IntentRecord | AcceptedIntent>();
    private _pending = new Set<number>();
    private _next = 1;
    readonly connection: Connection;
    constructor(connection: Connection) {
        this.connection = connection;
    }
    record(tick: number, kind: IntentRecord["kind"], values: number[]): IntentRecord {
        if (!Number.isInteger(tick) || tick < 0 || tick >= 0xffffffff || this._next >= 0xffffffff)
            throw new Error("Invalid intent tick");
        const record = { id: this._next++, tick, kind, values: values.slice() };
        this.records.set(record.id, record);
        this._pending.add(record.id);
        return record;
    }
    /** Route messages returned by ReplicationReceiver.flush here; accepted ticks replace predictions. */
    flush(messages: readonly Uint8Array[] = []): Uint8Array[] {
        const unhandled: Uint8Array[] = [];
        for (const bytes of messages) {
            if (
                bytes.length < 8 ||
                new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
                    0,
                    true,
                ) !== IntentMessageKind.Accepted
            ) {
                unhandled.push(bytes);
                continue;
            }
            for (const record of decodeIntent(bytes)) {
                this.records.set(record.id, record);
                this._pending.delete(record.id);
            }
        }
        if (this._pending.size)
            this.connection.send(
                encodeIntent(
                    IntentMessageKind.Records,
                    [...this._pending].map((id) => this.records.get(id)!),
                ),
            );
        return unhandled;
    }
}
