import { Time } from "../../engine";
import { transportClock } from "./diagnostics";
import { type Connection, MAX_MESSAGE_BYTES } from "./link";

type TimedConnection = Connection & { [transportClock]?: () => number };

/** Records and Accepted carry kind, count and cumulative resolved-ID watermark as u32s.
 * Records carry operations and a zero watermark. Inputs carry kind and tick count only. */
export const IntentMessageKind = { Records: 4, Accepted: 5, Inputs: 6 } as const;

/** One sampled fixed tick; values belong to the game. No operation identity. */
export interface TickInput {
    tick: number;
    values: number[];
}
export interface InputRecord extends TickInput {
    kind: "input";
}

/** At most four input ticks, without IDs or acknowledgements. */
export function encodeInputs(inputs: readonly TickInput[]): Uint8Array {
    if (inputs.length > 4) throw new Error("Invalid input window");
    const bytes = new Uint8Array(8 + inputs.reduce((size, r) => size + 8 + r.values.length * 8, 0));
    const view = new DataView(bytes.buffer);
    view.setUint32(0, IntentMessageKind.Inputs, true);
    view.setUint32(4, inputs.length, true);
    let offset = 8;
    for (const input of inputs) {
        view.setUint32(offset, input.tick, true);
        view.setUint32(offset + 4, input.values.length, true);
        offset += 8;
        for (const value of input.values) {
            view.setFloat64(offset, value, true);
            offset += 8;
        }
    }
    return bytes;
}

export function decodeInputs(bytes: Uint8Array): TickInput[] {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (
        bytes.length < 8 ||
        view.getUint32(0, true) !== IntentMessageKind.Inputs ||
        view.getUint32(4, true) > 4
    )
        throw new Error("Invalid input packet");
    const inputs: TickInput[] = [];
    let offset = 8;
    for (let i = 0; i < view.getUint32(4, true); i++) {
        const tick = view.getUint32(offset, true);
        const count = view.getUint32(offset + 4, true);
        offset += 8;
        const values: number[] = [];
        for (let lane = 0; lane < count; lane++, offset += 8)
            values.push(view.getFloat64(offset, true));
        if (tick >= 0xffffffff || inputs.some((input) => input.tick === tick))
            throw new Error("Invalid input tick");
        inputs.push({ tick, values });
    }
    if (offset !== bytes.length) throw new Error("Invalid input size");
    return inputs;
}

/** IDs are consecutive within one player's operation log. Values belong to the game. */
export interface OperationRecord {
    id: number;
    tick: number;
    kind: "operation";
    values: number[];
}
export type IntentRecord = InputRecord | OperationRecord;
export interface AcceptedIntent extends OperationRecord {
    /** Assigned tick minus requested tick, in fixed ticks. */
    lateness: number;
    /** Ticks remaining before the requested tick when first received, zero if late.
     * Omitted for locally supplied confirmations that carry no arrival measurement. */
    slack?: number;
}

export function encodeIntent(
    kind: number,
    records: readonly OperationRecord[],
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
        if (tag !== 1) throw new Error("Inputs do not belong in the operation log");
        const kind = "operation";
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

/** One player's operation log and tick-addressed input buffer.
 * Read inputAt before applying each server tick. Missing ticks hold the newest earlier input;
 * late arrivals replace that held input only when newer, never rewriting a simulated tick.
 * Input ticks beyond nextTick + maxAhead are refused without an acknowledgement.
 * Retrying an operation never changes its assigned tick.
 * Within the ID window, ticks beyond nextTick + maxAhead (32 by default) are
 * refused: resolved without application, and covered by the cumulative watermark.
 * IDs above that watermark + maxIdAhead are ignored, not acknowledged, and may
 * be retried once the ID window advances. Missing IDs are never skipped. */
export class IntentReceiver {
    readonly records = new Map<number, AcceptedIntent>();
    /** Future inputs by requested tick, plus at most one held input. */
    readonly inputs = new Map<number, TickInput>();
    private _inputTick = -1;
    private _newestInput = -1;
    private _ageTick = -1;
    private _commandAge: number | undefined;
    /** Smoothed server tick minus newest received input tick, in fixed ticks (positive is late).
     * Samples once per server tick with weight 1/8, including ticks with no new packet. */
    commandAge(tick: number): number | undefined {
        if (tick !== this._ageTick && this._newestInput >= 0) {
            this._commandAge = ((this._commandAge ?? 0) * 7 + tick - this._newestInput) / 8;
            this._ageTick = tick;
        }
        return this._commandAge;
    }
    /** Returns this tick's input or the latest earlier one received, retiring older entries.
     * Ticks must not decrease. Returned values remain receiver-owned; callers must not mutate them. */
    inputAt(tick: number): TickInput | undefined {
        if (!Number.isInteger(tick) || tick < this._inputTick || tick < 0 || tick >= 0xffffffff)
            throw new Error("Invalid input tick");
        this._inputTick = tick;
        let latest: TickInput | undefined;
        for (const input of this.inputs.values())
            if (input.tick <= tick && (!latest || input.tick > latest.tick)) latest = input;
        for (const key of this.inputs.keys())
            if (key <= tick && key !== latest?.tick) this.inputs.delete(key);
        return latest;
    }
    readonly connection: Connection;
    readonly maxAhead: number;
    /** Maximum operation ID distance above cumulative resolution; defaults to 2 * (maxAhead + 1).
     * Games with larger operation bursts can raise it. A missing ID can stall later
     * operations but cannot retain more than this window above the gap. */
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
     * Validation refuses inputs without storage and resolves refused operation IDs.
     * Input validation receives its requested tick and values, with no operation ID. */
    receive(
        messages: readonly Uint8Array[],
        nextTick: number,
        validate: (record: IntentRecord) => boolean = () => true,
    ): Uint8Array[] {
        if (!Number.isInteger(nextTick) || nextTick < 0 || nextTick >= 0xffffffff)
            throw new Error("Invalid intent tick");
        const unhandled: Uint8Array[] = [];
        let operations = false;
        this.inputAt(nextTick);
        for (const bytes of messages) {
            if (
                bytes.length >= 8 &&
                new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
                    0,
                    true,
                ) === IntentMessageKind.Inputs
            ) {
                for (const input of decodeInputs(bytes)) {
                    if (
                        input.tick > nextTick + this.maxAhead ||
                        !validate({ ...input, kind: "input" })
                    )
                        continue;
                    this._newestInput = Math.max(this._newestInput, input.tick);
                    if (input.tick < nextTick && this.inputs.has(input.tick)) continue;
                    this.inputs.set(input.tick, input);
                    this.inputAt(nextTick);
                }
                continue;
            }
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
            operations = true;
            for (const record of decodeIntent(bytes)) {
                if (record.kind !== "operation") continue;
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
        if (operations || this.records.size || this._watermark)
            this.connection.send(
                encodeIntent(
                    IntentMessageKind.Accepted,
                    [...this.records.values()],
                    this._watermark,
                ),
            );
        return unhandled;
    }
}

/** Unresolved operations, new ones included, go in one packet of at most MAX_MESSAGE_BYTES,
 * oldest first, the rest in later packets. A packet goes on the first flush in each fixed step
 * of the transport clock (performance.now unless the connection supplies one), so a new
 * operation waits at most one step and a flush, recording or flushing more often sends no
 * more, and the same steps resend unresolved operations without the caller's tick.
 * Acceptance removes their IDs; the cumulative watermark also removes refused operations.
 * Inputs instead repeat the newest of the last four ticks that fit in one packet on the first
 * flush after a new sample. */
export class IntentSender {
    readonly records = new Map<number, OperationRecord>();
    /** Last four sampled ticks, independent of round trip and acceptance. */
    readonly inputs = new Map<number, TickInput>();
    private _inputDirty = false;
    private _latestInput = -1;
    private _recordsDue = -Infinity;
    private readonly _clock: () => number;
    private _next = 1;
    readonly connection: Connection;
    constructor(connection: Connection) {
        this.connection = connection;
        this._clock = (connection as TimedConnection)[transportClock] ?? (() => performance.now());
    }
    /** Copies values. A repeated input tick replaces its sample; decreasing ticks throw. A record
     * that cannot fit one MAX_MESSAGE_BYTES packet alone, an input of more than 510 values or an
     * operation of more than 507, is a programming error and throws.
     * Flush after sampling. Each tick sends at most one input packet; replacements before
     * that flush are included. Inputs never enter the operation ID log. */
    record(tick: number, kind: "input", values: number[]): InputRecord;
    record(tick: number, kind: "operation", values: number[]): OperationRecord;
    record(tick: number, kind: IntentRecord["kind"], values: number[]): IntentRecord {
        if (!Number.isInteger(tick) || tick < 0 || tick >= 0xffffffff)
            throw new Error("Invalid intent tick");
        if ((kind === "input" ? 16 : 36) + values.length * 8 > MAX_MESSAGE_BYTES)
            throw new Error(`Intent values exceed a ${MAX_MESSAGE_BYTES}-byte message`);
        if (kind === "input") {
            if (tick < this._latestInput) throw new Error("Input ticks must not decrease");
            if (tick > this._latestInput) this._inputDirty = true;
            this._latestInput = tick;
            const input: InputRecord = { tick, values: values.slice(), kind };
            this.inputs.set(tick, input);
            for (const key of this.inputs.keys()) if (key <= tick - 4) this.inputs.delete(key);
            return input;
        }
        if (this._next >= 0xffffffff) throw new Error("Operation IDs exhausted");
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
        if (this._inputDirty) {
            const inputs = [...this.inputs.values()];
            let size = 8 + inputs.reduce((sum, input) => sum + 8 + input.values.length * 8, 0);
            while (size > MAX_MESSAGE_BYTES) size -= 8 + inputs.shift()!.values.length * 8;
            this.connection.send(encodeInputs(inputs));
            this._inputDirty = false;
        }
        const time = this._clock();
        const step = Time.FIXED_DT * 1000;
        if (this.records.size && time >= this._recordsDue) {
            const records: OperationRecord[] = [];
            let size = 12;
            for (const record of this.records.values()) {
                size += 24 + record.values.length * 8;
                if (size > MAX_MESSAGE_BYTES) break;
                records.push(record);
            }
            this.connection.send(encodeIntent(IntentMessageKind.Records, records));
            // Slots keep a one-step grid, so flushes a hair under a step apart still send
            // every step; after an idle step the grid restarts from now.
            this._recordsDue =
                time - this._recordsDue < step ? this._recordsDue + step : time + step;
        }
        return unhandled;
    }
}
