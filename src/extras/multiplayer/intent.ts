import type { Connection } from "./link";

/** Intent packets carry kind, record count and cumulative resolved-ID watermark as u32s.
 * Records packets carry a zero watermark. */
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
    /** Ticks remaining before the requested tick when first received, zero if late.
     * Omitted for locally supplied confirmations that carry no arrival measurement. */
    slack?: number;
}

export function encodeIntent(
    kind: number,
    records: readonly IntentRecord[],
    watermark = 0,
): Uint8Array {
    const bytes = new Uint8Array(
        12 + records.reduce((size, r) => size + 24 + r.values.length * 8, 0),
    );
    const view = new DataView(bytes.buffer);
    view.setUint32(0, kind, true);
    view.setUint32(4, records.length, true);
    view.setUint32(8, watermark, true);
    let offset = 12;
    for (const r of records) {
        const words = [
            r.id,
            r.tick,
            Number(r.kind === "operation"),
            (r as AcceptedIntent).lateness ?? 0,
            r.values.length,
            (r as AcceptedIntent).slack === undefined ? 0 : (r as AcceptedIntent).slack! + 1,
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
/** Cumulative ID resolved by acceptance or refusal. Call after decoding the packet. */
export function intentWatermark(bytes: Uint8Array): number {
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(8, true);
}

export function decodeIntent(bytes: Uint8Array): AcceptedIntent[] {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const kind = bytes.length >= 12 ? view.getUint32(0, true) : 0;
    if (kind !== IntentMessageKind.Records && kind !== IntentMessageKind.Accepted)
        throw new Error("Invalid intent packet");
    const records: AcceptedIntent[] = [];
    let offset = 12;
    for (let i = 0; i < view.getUint32(4, true); i++) {
        const id = view.getUint32(offset, true),
            tick = view.getUint32(offset + 4, true);
        const tag = view.getUint32(offset + 8, true);
        if (tag > 1) throw new Error("Invalid intent kind");
        const kind = tag ? "operation" : "input";
        const lateness = view.getUint32(offset + 12, true),
            count = view.getUint32(offset + 16, true);
        const arrival = view.getUint32(offset + 20, true);
        offset += 24;
        const values: number[] = [];
        for (let lane = 0; lane < count; lane++) {
            values.push(view.getFloat64(offset, true));
            offset += 8;
        }
        records.push({
            id,
            tick,
            kind,
            lateness,
            values,
            ...(arrival ? { slack: arrival - 1 } : {}),
        });
    }
    if (offset !== bytes.length) throw new Error("Invalid intent size");
    return records;
}

/** One player's accepted log. Retrying a record never changes its assigned tick.
 * Within the ID window, ticks beyond nextTick + maxAhead (32 by default) are
 * refused: resolved without application, and covered by the cumulative watermark.
 * IDs above that watermark + maxIdAhead are ignored, not acknowledged, and may
 * be retried once the ID window advances. Missing IDs are never skipped. */
export class IntentReceiver {
    readonly records = new Map<number, AcceptedIntent>();
    readonly connection: Connection;
    readonly maxAhead: number;
    /** Maximum ID distance above cumulative resolution. Defaults to 2 * (maxAhead + 1):
     * one input and one operation per tick across the inclusive prediction horizon,
     * including fully reordered delivery. The example sends one input per tick with
     * cumulative operations in its values, so fits within this budget.
     * Games with larger operation bursts can raise it. A missing ID can stall later
     * records but cannot retain more than this window above the gap. */
    readonly maxIdAhead: number;
    private _watermark = 0;
    private _resolved = new Set<number>();
    /** Retires records folded into the recipient's acknowledged state.
     * IDs beyond a gap stay until every preceding ID has been resolved. */
    prune(acknowledged: number): void {
        for (const [id, record] of this.records)
            if (id <= this._watermark && record.tick <= acknowledged) this.records.delete(id);
    }
    constructor(connection: Connection, maxAhead = 32, maxIdAhead = 2 * (maxAhead + 1)) {
        this.maxAhead = maxAhead;
        if (!Number.isSafeInteger(maxAhead) || maxAhead < 0)
            throw new Error("Invalid prediction bound");
        if (!Number.isSafeInteger(maxIdAhead) || maxIdAhead < 1)
            throw new Error("Invalid intent ID bound");
        this.maxIdAhead = maxIdAhead;
        this.connection = connection;
    }
    /** Route messages returned by ReplicationSender.flush here before the next fixed tick.
     * Validation refuses a record without application and resolves its ID, just like a tick refusal. */
    receive(
        messages: readonly Uint8Array[],
        nextTick: number,
        validate: (record: IntentRecord) => boolean = () => true,
    ): Uint8Array[] {
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
                if (
                    record.id <= this._watermark ||
                    record.id - this._watermark > this.maxIdAhead ||
                    this._resolved.has(record.id)
                )
                    continue;
                if (record.tick <= nextTick + this.maxAhead && validate(record)) {
                    const tick = Math.max(nextTick, record.tick);
                    this.records.set(record.id, {
                        ...record,
                        tick,
                        lateness: tick - record.tick,
                        slack: Math.max(0, record.tick - nextTick),
                    });
                }
                this._resolved.add(record.id);
                while (this._resolved.delete(this._watermark + 1)) this._watermark++;
            }
        }
        this.connection.send(
            encodeIntent(IntentMessageKind.Accepted, [...this.records.values()], this._watermark),
        );
        return unhandled;
    }
}

/** Unresolved records repeat on every flush, including discrete operations.
 * Acceptance removes an individual ID; the cumulative watermark also removes
 * refused records without applying them. */
export class IntentSender {
    readonly records = new Map<number, IntentRecord>();
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
        return record;
    }
    /** Route messages returned by ReplicationReceiver.flush here; resolved records stop repeating. */
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
            for (const record of decodeIntent(bytes)) this.records.delete(record.id);
            const watermark = intentWatermark(bytes);
            for (const id of this.records.keys()) if (id <= watermark) this.records.delete(id);
        }
        if (this.records.size)
            this.connection.send(
                encodeIntent(IntentMessageKind.Records, [...this.records.values()]),
            );
        return unhandled;
    }
}
