import {
    type Component,
    component,
    type EntityRef,
    entity,
    type FieldType,
    type Plugin,
    type TypedArray,
    u32,
    type World,
} from "../../engine";
import { fields } from "../../engine/ecs";
import { encodeText, readText, writeText } from "../../engine/ecs/utf8";
import { reportStartup } from "../../engine/runtime";

export {
    ClientOf,
    client,
    type Game,
    local,
    RoomBot,
    RoomLeader,
    ServerSession,
    server,
    type ThinClient,
} from "./game";
export { ClientInterpolation, InterpolationPlugin, InterpolationTarget } from "./interpolation";
export { type Connection, type LinkFaults, MAX_MESSAGE_BYTES, seededLink } from "./link";
export {
    ClientPrediction,
    contactOutputKey,
    FixedOutputs,
    PredictionMotion,
    PredictionPlugin,
    PredictionTarget,
} from "./prediction";
export {
    type RecordingCall,
    type RecordingOptions,
    replaySession,
    type SessionRecording,
    sessionRecording,
} from "./recording";
export { RoomBots } from "./room-host";
export { WebSocketConnection } from "./websocket";

import { requireCanonical } from "./canonical";
import { diagnosticKind, networkDiagnostics } from "./diagnostics";
import {
    type AcceptedIntent,
    decodeAccepted,
    encodeIntent,
    IntentMessageKind,
    validateAccepted,
} from "./intent";
import type { Connection } from "./link";
import { failServerConnection, serverConnectionFailed } from "./link";
import { PredictionRecording } from "./recording-hooks";

export {
    type AcceptedIntent,
    type InputRecord,
    IntentMessageKind,
    IntentReceiver,
    type IntentRecord,
    IntentSender,
    type OperationRecord,
    type TickInput,
} from "./intent";

/** Selects a disclosure rule. Only entities holding this component can be sent. */
export const Replicate = component("Replicate", { rule: u32 });
/** Registers replication disclosure. Sessions record end-local diagnostics in DiagnosticsStore.
 * Client paths use network/ and are removed when the transport closes; a replacement upstream
 * starts fresh histories and counters. Server paths use network/connections/<server-eid>/ and
 * disappear on leave. bytes_sent/bytes_received (B) and messages_sent/messages_received (counts) are
 * cumulative payload counters, including ping/pong, retries and received duplicates, excluding
 * transport framing. Derive rates from their timed histories, not from smoothed counter values.
 * round_trip (ms) is ping send-to-pong receipt minus the remote receipt-to-reply hold time;
 * transport receipt timestamps exclude local drain scheduling. Pings run every 100 ms when
 * polled, including idle and thin clients. No RTT sample exists until a matching pong arrives.
 * time_since_receive (ms) ages from transport receipt, or session observation start before any
 * receipt. confirmed_tick_lag (ticks) is latest sent minus acknowledged on the server, newest
 * received minus contiguous confirmed on the client; it cannot observe an unseen server tick.
 * network/peer_count counts admitted server connections. Prediction adds client-only
 * network/replayed_ticks (cumulative ticks actually replayed), network/correction_translation
 * (world units) and network/correction_rotation (rad): the largest incremental visual correction
 * across predicted entities in each reconciliation, translation length and shortest rotation angle.
 * Histories and smoothing use Diagnostic defaults. No overlay or ProfilePlugin is required.
 */
export const MultiplayerPlugin: Plugin = { name: "Multiplayer", components: [Replicate] };

/** Rules are indexed by Replicate.rule; an absent recipient receives nothing. */
export type ReplicationRules = readonly ReadonlyMap<number, readonly Component[]>[];
/** Both ends must use the same ordered component schemas and engine version. */
export type ReplicationSchema = readonly Component[];
type ReplicatedValue = number | string;
export type ReplicatedRows = Map<number, Map<number, ReplicatedValue[]>>;
interface Version {
    tick: number;
    rows: ReplicatedRows;
}
interface Packet extends Version {
    commandAge?: number;
    base: number;
    image: boolean;
    intents: readonly AcceptedIntent[];
}

/** Little-endian u32 replication kinds. flush returns other kinds unchanged for routing.
 * An Acknowledgement carries the confirmed tick, then a u32 whose bit i reports that the
 * receiver holds tick confirmed + 1 + i. */
export const ReplicationMessageKind = { Image: 1, Frame: 2, Acknowledgement: 3 } as const;

/** @internal The two u32 payload words have no alternate encoding. */
export function encodeAcknowledgement(tick: number, held: number): Uint8Array {
    const bytes = new Uint8Array(12);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, ReplicationMessageKind.Acknowledgement, true);
    view.setUint32(4, tick, true);
    view.setUint32(8, held, true);
    return bytes;
}

function wireLanes(field: FieldType): number {
    return field.byteCapacity === undefined ? (field.components ?? field.lanes) : 1;
}

function layout(schema: ReplicationSchema) {
    return schema.map((c) =>
        fields(c).flatMap(({ field }) =>
            Array<FieldType>(wireLanes(field)).fill(field === entity ? u32 : field),
        ),
    );
}

function writeField(view: DataView, offset: number, field: FieldType, value: number) {
    const raw = field.encode ? field.encode(value) : value;
    switch (field.ctor) {
        case Float32Array:
            view.setFloat32(offset, raw, true);
            break;
        case Float64Array:
            view.setFloat64(offset, raw, true);
            break;
        case Int32Array:
            view.setInt32(offset, raw, true);
            break;
        case Uint32Array:
            view.setUint32(offset, raw, true);
            break;
        case Uint16Array:
            view.setUint16(offset, raw, true);
            break;
        case Uint8Array:
            view.setUint8(offset, raw);
            break;
    }
}

function readField(view: DataView, offset: number, field: FieldType): number {
    let raw: number;
    switch (field.ctor) {
        case Float32Array:
            raw = view.getFloat32(offset, true);
            break;
        case Float64Array:
            raw = view.getFloat64(offset, true);
            break;
        case Int32Array:
            raw = view.getInt32(offset, true);
            break;
        case Uint32Array:
            raw = view.getUint32(offset, true);
            break;
        case Uint16Array:
            raw = view.getUint16(offset, true);
            break;
        case Uint8Array:
            raw = view.getUint8(offset);
            break;
        default:
            throw new Error("Unsupported replicated field type");
    }
    return field.decode ? field.decode(raw) : raw;
}

/** Captures only declared fields; references outside the recipient's disclosure become null. */
function capture(
    world: World,
    schema: ReplicationSchema,
    rules: ReplicationRules,
    recipient: number,
    identities: Map<EntityRef, number>,
    next: { value: number },
): ReplicatedRows {
    const selected = new Map<number, readonly Component[]>();
    const ids = new Map<number, number>();
    const policy = world.storage(Replicate);
    for (const eid of world.query([Replicate])) {
        const components = rules[policy.rule.get(eid)]?.get(recipient);
        if (!components) continue;
        const ref = world.ref(eid);
        let id = identities.get(ref);
        if (id === undefined) {
            id = next.value++;
            identities.set(ref, id);
        }
        ids.set(eid, id);
        selected.set(eid, components);
    }
    const rows: ReplicatedRows = new Map();
    for (const [eid, components] of selected) {
        const row = new Map<number, ReplicatedValue[]>();
        for (const c of components) {
            const index = schema.indexOf(c);
            if (index < 0) throw new Error("Disclosure component missing from replication schema");
            if (!world.has(eid, c)) continue;
            const record = world.get(eid, c)! as Record<string, ReplicatedValue | number[]>;
            const values: ReplicatedValue[] = [];
            for (const { name, field } of fields(c)) {
                const value = record[name]!;
                if (field === entity) values.push(ids.get(value as number) ?? 0);
                else if (!Array.isArray(value)) values.push(value);
                else values.push(...value);
            }
            row.set(index, values);
        }
        rows.set(ids.get(eid)!, row);
    }
    return rows;
}

// A membership record with this count removes its entity or component from the baseline.
const Removed = 0xffffffff;

type Changes<T> = Map<number, Map<number, T[] | null> | null>;
type WireFrame<T> = Omit<Packet, "rows"> & { changes: Changes<T> };
type DecodedFrame = WireFrame<Uint8Array>;

function diff(rows: ReplicatedRows, base: ReplicatedRows): Changes<ReplicatedValue> {
    const changes: Changes<ReplicatedValue> = new Map();
    for (const [id, row] of rows) {
        const before = base.get(id);
        const updates = new Map<number, ReplicatedValue[] | null>();
        for (const [index, values] of row) {
            const previous = before?.get(index);
            if (!previous) {
                updates.set(index, values);
                continue;
            }
            const lanes: ReplicatedValue[] = [];
            for (let i = 0; i < values.length; i++)
                if (!Object.is(values[i], previous[i])) lanes[i] = values[i]!;
            if (lanes.length) updates.set(index, lanes);
        }
        for (const index of before?.keys() ?? []) if (!row.has(index)) updates.set(index, null);
        if (!before || updates.size) changes.set(id, updates);
    }
    for (const id of base.keys()) if (!rows.has(id)) changes.set(id, null);
    return changes;
}

/** @internal Encode the sender's sparse delta in schema-lane order. */
export function encodeFrame(
    packet: WireFrame<ReplicatedValue | Uint8Array>,
    types: FieldType[][],
): Uint8Array {
    const words: number[] = [
        packet.image ? ReplicationMessageKind.Image : ReplicationMessageKind.Frame,
        packet.tick,
        packet.image ? packet.tick : packet.base,
        packet.changes.size,
        packet.commandAge === undefined ? 0x80000000 : Math.round(packet.commandAge * 256),
    ];
    const values: { lane: number | Uint8Array; field: FieldType }[] = [];
    let valueBytes = 0;
    for (const [id, row] of packet.changes) {
        if (!row && packet.image) throw new Error("An image cannot remove an entity");
        words.push(id, row?.size ?? Removed);
        if (!row) continue;
        for (const [c, lanes] of row) {
            const fields = types[c]!;
            words.push(c);
            if (lanes === null) {
                if (packet.image) throw new Error("An image cannot remove a component");
                words.push(Removed);
                continue;
            }
            const countOffset = words.length;
            words.push(0);
            let count = 0;
            for (let i = 0; i < lanes.length; i++) {
                const lane = lanes[i];
                if (lane === undefined) continue;
                count++;
                words.push(i);
                const field = fields[i]!;
                if (typeof lane === "string") {
                    const text = encodeText(lane, field.byteCapacity!);
                    const bytes = new Uint8Array(2 + text.length);
                    writeText(new DataView(bytes.buffer), 0, text);
                    values.push({ lane: bytes, field });
                    valueBytes += bytes.length;
                } else {
                    values.push({ lane, field });
                    valueBytes +=
                        typeof lane === "number" ? field.ctor.BYTES_PER_ELEMENT : lane.length;
                }
            }
            if (packet.image ? count !== fields.length : !count && fields.length)
                throw new Error("Incomplete replication fields");
            words[countOffset] = count;
        }
    }
    const intents = packet.intents.length
        ? encodeIntent(IntentMessageKind.Accepted, packet.intents)
        : new Uint8Array();
    const bytes = new Uint8Array(4 + words.length * 4 + valueBytes + intents.length);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, words[0]!, true);
    view.setUint32(4, words.length, true);
    for (let i = 1; i < words.length; i++) view.setUint32(4 + i * 4, words[i]!, true);
    let offset = 4 + words.length * 4;
    for (const { lane, field } of values) {
        if (typeof lane === "number") {
            writeField(view, offset, field, lane);
            offset += field.ctor.BYTES_PER_ELEMENT;
        } else {
            bytes.set(lane, offset);
            offset += lane.length;
        }
    }
    bytes.set(intents, offset);
    return bytes;
}

/** Decode independently of the baseline: an early or out-of-order frame must already be valid. */
function decodeFrame(bytes: Uint8Array, types: FieldType[][]): DecodedFrame {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const kind = view.getUint32(0, true);
    const count = view.getUint32(4, true);
    let offset = 8;
    let valueOffset = 4 + count * 4;
    const word = () => {
        const v = view.getUint32(offset, true);
        offset += 4;
        return v;
    };
    const tick = word(),
        base = word(),
        entities = word();
    const age = word();
    const commandAge = age === 0x80000000 ? undefined : (age | 0) / 256;
    if (kind !== ReplicationMessageKind.Image && kind !== ReplicationMessageKind.Frame)
        throw new Error("Invalid replication packet");
    if (kind === ReplicationMessageKind.Frame && base >= tick)
        throw new Error("Replication baseline must precede its frame");
    const updates: Changes<Uint8Array> = new Map();
    for (let e = 0; e < entities; e++) {
        const id = word(),
            components = word();
        if (components === Removed) {
            updates.set(id, null);
            continue;
        }
        const row = new Map<number, Uint8Array[] | null>();
        for (let c = 0; c < components; c++) {
            const index = word(),
                changes = word();
            const componentTypes = types[index];
            if (componentTypes === undefined) throw new Error("Unknown replicated component");
            if (changes === Removed) {
                row.set(index, null);
                continue;
            }
            const size = componentTypes.length;
            const values: Uint8Array[] = [];
            for (let f = 0; f < changes; f++) {
                const lane = word();
                if (lane >= size) throw new Error("Invalid replicated field");
                const field = componentTypes[lane]!;
                const start = valueOffset;
                if (field.byteCapacity === undefined) {
                    valueOffset += field.ctor.BYTES_PER_ELEMENT;
                    if (valueOffset > bytes.length) throw new Error("Truncated replication field");
                } else {
                    readText(view, valueOffset, field.byteCapacity);
                    valueOffset += 2 + view.getUint16(valueOffset, true);
                }
                values[lane] = bytes.subarray(start, valueOffset);
            }
            row.set(index, values);
        }
        updates.set(id, row);
    }
    if (offset !== 4 + count * 4 || valueOffset > bytes.length)
        throw new Error("Invalid replication size");
    const intents =
        valueOffset < bytes.length ? decodeAccepted(bytes.subarray(valueOffset)).records : [];
    return {
        tick,
        base,
        changes: updates,
        intents,
        commandAge,
        image: kind === ReplicationMessageKind.Image,
    };
}

function applyFrame(
    frame: DecodedFrame,
    baseline: ReplicatedRows | undefined,
    types: FieldType[][],
): ReplicatedRows {
    const rows: ReplicatedRows = new Map(baseline);
    for (const [id, updates] of frame.changes) {
        if (updates === null) {
            rows.delete(id);
            continue;
        }
        const previous = baseline?.get(id);
        const row = new Map(previous);
        for (const [index, changes] of updates) {
            if (changes === null) {
                row.delete(index);
                continue;
            }
            const before = previous?.get(index);
            row.set(
                index,
                types[index]!.map((field, lane) => {
                    const bytes = changes[lane];
                    if (!bytes)
                        return before?.[lane] ?? (field.byteCapacity === undefined ? 0 : "");
                    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
                    return field.byteCapacity === undefined
                        ? readField(view, 0, field)
                        : readText(view, 0, field.byteCapacity);
                }),
            );
        }
        rows.set(id, row);
    }
    return rows;
}

/** @internal Fully validate a server message before queueing or routing it. */
export function decodeServerMessage(
    bytes: Uint8Array,
    types: FieldType[][],
): DecodedFrame | undefined {
    if (diagnosticKind(bytes)) return;
    const kind = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
    if (kind === ReplicationMessageKind.Image || kind === ReplicationMessageKind.Frame) {
        const frame = decodeFrame(bytes, types);
        requireCanonical(bytes, encodeFrame(frame, types));
        return frame;
    }
    validateAccepted(bytes);
}

/** Retransmission timeout bounds in ticks of `Time.FIXED_DT` (1/60 s). The initial timeout is
 * RFC 6298's 1 s, used until a first sample. The floor is KCP's 100 ms minimum: RFC 6298's 1 s
 * floor allows for delayed acknowledgements, which ReplicationReceiver never sends, and 100 ms
 * keeps a short, steady round trip from retrying after one late client step. The ceiling is
 * RFC 6298's 60 s. */
const RTO = { initial: 60, floor: 6, ceiling: 3600 } as const;
/** Ticks after the confirmed tick an acknowledgement reports as held, one bit each. */
const AckWindow = 32;
/** Later frames reported held before an unheld frame is resent without waiting for its timeout:
 * KCP's fastresend of 2. RFC 5681's threshold of three measured slower here. */
const FastResend = 2;

interface PendingFrame {
    /** Tick of the state bytes was encoded against. */
    base: number;
    bytes: Uint8Array;
    /** Tick of the latest send; -1 until first sent. */
    sent: number;
    /** Tick of the first send. */
    first: number;
    /** Ticks after sent before the next timeout retry; doubles per timeout. */
    timeout: number;
    retried: boolean;
    /** Reported held by the receiver; never retried again. */
    held: boolean;
}

/** One recipient's acknowledged history. record starts at any nonnegative tick and refuses gaps
 * thereafter; recorded ticks are the retransmission clock. flush sends each new frame once and
 * retries a frame the receiver has not reported holding once later frames are reported held
 * (fast retransmit) or after the retransmission timeout, which doubles after each timeout. The
 * timeout follows RFC 6298 from round trips of newly reported frames that were never retried
 * (Karn's rule), within the RTO bounds, so a reliable link resends almost nothing and a lossy
 * one still recovers. A pending frame retains only its bytes; an acknowledgement decodes them
 * against their base, as the receiver did, so the baseline is the receiver's state. */
export class ReplicationSender {
    private _identities = new Map<EntityRef, number>();
    private _next = { value: 1 };
    private _acknowledged: Version | undefined;
    /** Newest frame tick acknowledged by this recipient, or -1 before its first acknowledgement. */
    get acknowledged(): number {
        return this._acknowledged?.tick ?? -1;
    }
    private _pending = new Map<number, PendingFrame>();
    /** Exactly the states some pending frame names as its base, plus the acknowledged state the
     * next record encodes against; before the first acknowledgement, only the image. */
    private _bases = new Map<number, ReplicatedRows>();
    private readonly _layout: FieldType[][];
    private _last = -1;
    private _smoothed = -1;
    private _variance = 0;
    private _rto: number = RTO.initial;
    readonly world: World;
    readonly schema: ReplicationSchema;
    readonly rules: ReplicationRules;
    readonly recipient: number;
    readonly connection: Connection;
    constructor(
        world: World,
        schema: ReplicationSchema,
        rules: ReplicationRules,
        recipient: number,
        connection: Connection,
    ) {
        this.world = world;
        this.schema = schema;
        this.rules = rules;
        this.recipient = recipient;
        this.connection = connection;
        this._layout = layout(schema);
    }
    /** Include this recipient's accepted operations through tick. Inputs never enter frames.
     * commandAge is the server's smoothed tick minus newest received input tick, in fixed ticks;
     * encoded as signed 1/256 ticks in each frame header, including the initial image. */
    record(tick: number, intents: readonly AcceptedIntent[] = [], commandAge?: number) {
        if (
            !Number.isInteger(tick) ||
            tick < 0 ||
            tick >= 0xffffffff ||
            (this._last >= 0 && tick !== this._last + 1)
        )
            throw new Error("Replication ticks must be consecutive nonnegative integers");
        const rows = capture(
            this.world,
            this.schema,
            this.rules,
            this.recipient,
            this._identities,
            this._next,
        );
        const image = this._last < 0;
        // Until the image is acknowledged subsequent frames use its rows, not an empty baseline.
        if (image) this._bases.set(tick, rows);
        const base = image ? tick : (this._acknowledged?.tick ?? this._bases.keys().next().value!);
        const packet = {
            tick,
            rows,
            image,
            intents: intents.filter((r) => r.kind === "operation" && r.tick <= tick),
            commandAge,
            base,
        };
        const changes = diff(rows, image ? new Map() : this._bases.get(base)!);
        const bytes = encodeFrame({ ...packet, changes }, this._layout);
        this._pending.set(tick, {
            base,
            bytes,
            sent: -1,
            first: -1,
            timeout: 0,
            retried: false,
            held: false,
        });
        this._last = tick;
    }
    /** Sends new frames, retries timed-out ones and returns non-acknowledgement messages for
     * application routing. now is the retransmission clock in ticks, by default the latest
     * recorded tick; a caller that keeps flushing after it stops recording passes its tick. */
    flush(now = this._last): Uint8Array[] {
        const unhandled: Uint8Array[] = [];
        for (const bytes of this.connection.receive()) {
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            if (
                bytes.length !== 12 ||
                view.getUint32(0, true) !== ReplicationMessageKind.Acknowledgement
            ) {
                unhandled.push(bytes);
                continue;
            }
            const tick = view.getUint32(4, true);
            const held = view.getUint32(8, true);
            let sample: PendingFrame | undefined;
            for (const [frameTick, frame] of this._pending) {
                const offset = frameTick - tick - 1;
                if (
                    frame.held ||
                    (offset >= 0 && (offset >= AckWindow || !((held >>> offset) & 1)))
                )
                    continue;
                frame.held = true;
                if (!frame.retried) sample ??= frame;
            }
            if (sample) this._sample(now - sample.sent);
            const frame = this._pending.get(tick);
            if (!frame) continue;
            const decoded = decodeFrame(frame.bytes, this._layout);
            const rows = applyFrame(
                decoded,
                decoded.image ? undefined : this._bases.get(decoded.base),
                this._layout,
            );
            this._acknowledged = { tick, rows };
            this._bases.set(tick, rows);
            for (const old of this._pending.keys()) if (old <= tick) this._pending.delete(old);
            const named = new Set([tick]);
            for (const pending of this._pending.values()) named.add(pending.base);
            for (const old of this._bases.keys()) if (!named.has(old)) this._bases.delete(old);
        }
        // First sends of held frames, newest first. A frame is fast-resent once FastResend frames
        // first sent after its latest send are held, so it never fires on an in-order link that loses nothing.
        const later: number[] = [];
        for (const frame of [...this._pending.values()].reverse()) {
            if (frame.held) {
                later.push(frame.first);
                continue;
            }
            if (frame.sent < 0) {
                frame.timeout = this._rto;
                frame.first = now;
            } else if (now - frame.sent >= frame.timeout) {
                frame.timeout = Math.min(frame.timeout * 2, RTO.ceiling);
                frame.retried = true;
            } else if (later.length >= FastResend && later[FastResend - 1]! > frame.sent) {
                frame.retried = true;
            } else continue;
            frame.sent = now;
            this.connection.send(frame.bytes);
        }
        networkDiagnostics(this.world, this.connection)?.record(
            "confirmed_tick_lag",
            Math.max(0, this._last - this.acknowledged),
        );
        return unhandled;
    }
    /** RFC 6298 section 2 with alpha 1/8, beta 1/4, K = 4 and a clock granularity of one tick. */
    private _sample(rtt: number) {
        if (this._smoothed < 0) {
            this._smoothed = rtt;
            this._variance = rtt / 2;
        } else {
            this._variance = 0.75 * this._variance + 0.25 * Math.abs(this._smoothed - rtt);
            this._smoothed = 0.875 * this._smoothed + 0.125 * rtt;
        }
        const rto = this._smoothed + Math.max(1, 4 * this._variance);
        this._rto = Math.min(Math.max(rto, RTO.floor), RTO.ceiling);
    }
}

/** One ordered state preceding a drain's latest state. apply writes its rows and notifies confirm;
 * call once in order before applying the latest state. Borrowed for the reconcile callback. */
export interface ReplicationConfirmation {
    tick: number;
    intents: readonly AcceptedIntent[];
    apply: () => void;
}

type PreparedRows = Map<ReplicatedValue[], (TypedArray | undefined)[]>;
type PreparedPacket = Packet & { prepared?: PreparedRows };

/** Ordered confirmations, including the initial image. Application callbacks may retain rows but not mutate them. */
export class ReplicationReceiver {
    confirmed = -1;
    /** Once per ordered server tick. Command age is in fixed ticks; positive means late. */
    onCommandAge: (tick: number, age: number) => void = () => {};
    readonly entities = new Map<number, number>();
    private _versions = new Map<number, ReplicatedRows>();
    private _pending = new Map<number, DecodedFrame>();
    private readonly _discarded = new Set<number>();
    private _newest = -1;
    private _ackTick: number;
    private readonly _start: number;
    private _generation?: number;
    private readonly _layout: FieldType[][];
    private readonly _fields: ReturnType<typeof fields>[];
    private readonly _needsPreparation: boolean;
    readonly world: World;
    readonly schema: ReplicationSchema;
    readonly connection: Connection;
    readonly confirm: (tick: number, rows: ReplicatedRows) => void;
    /** Synchronous restore/apply/replay boundary. Restore at tick, deliver preceding states in order,
     * then call apply exactly once before replay. Do not retain apply or preceding. */
    readonly reconcile: (
        tick: number,
        rows: ReplicatedRows,
        intents: readonly AcceptedIntent[],
        apply: () => void,
        preceding: readonly ReplicationConfirmation[],
    ) => void;
    constructor(
        world: World,
        schema: ReplicationSchema,
        connection: Connection,
        confirm: (tick: number, rows: ReplicatedRows) => void = () => {},
        reconcile: (
            tick: number,
            rows: ReplicatedRows,
            intents: readonly AcceptedIntent[],
            apply: () => void,
            preceding: readonly ReplicationConfirmation[],
        ) => void = (_tick, _rows, _intents, apply, preceding) => {
            for (const state of preceding) state.apply();
            apply();
        },
    ) {
        this.world = world;
        this.schema = schema;
        this._layout = layout(schema);
        this._fields = schema.map(fields);
        this._needsPreparation = this._fields.some((component) =>
            component.some(({ field }) => field.encode || field.byteCapacity !== undefined),
        );
        this.connection = connection;
        this.confirm = confirm;
        this.reconcile = reconcile;
        this._start = this._ackTick = world.time.fixedTick;
        world.resource(PredictionRecording).receiver?.(this);
    }
    /** Applies confirmations and returns other message kinds for application routing.
     * "latest" reconciles once, passing preceding ordered states to that callback. It must
     * deliver their intents and apply each before the latest state. Every applied state
     * still notifies confirm, so only recovery/replay work is coalesced, not observations.
     * Validates server messages before queueing or routing them. A refusal closes the connection
     * after applying the contiguous state already validated in this drain.
     * Acknowledges once per fixed tick of its World, or, while that World has not ticked, once
     * per drain that received a replication packet; never per call. */
    flush(confirmations: "all" | "latest" = "all"): Uint8Array[] {
        if (serverConnectionFailed(this.connection)) return [];
        const generation = this.connection.generation;
        if (this._generation !== generation) {
            this._generation = generation;
            this._ackTick = -1;
            this.confirmed = -1;
            this._newest = -1;
            this._pending.clear();
            this._discarded.clear();
            this._versions.clear();
        }
        const previous = this.confirmed;
        const latestOnly = confirmations === "latest";
        const unhandled: Uint8Array[] = [];
        let latest: PreparedPacket | undefined;
        let decoded = this.confirmed;
        let failure: Error | undefined;
        const preceding: ReplicationConfirmation[] = [];
        let arrived = false;
        for (const bytes of this.connection.receive()) {
            let frame: DecodedFrame | undefined;
            try {
                frame = decodeServerMessage(bytes, this._layout);
            } catch (error) {
                failure = new Error("Could not decode server message", { cause: error });
                break;
            }
            if (!frame) {
                unhandled.push(bytes);
                continue;
            }
            arrived = true;
            this._newest = Math.max(this._newest, frame.tick);
            if (frame.tick <= this.confirmed) continue;
            if (frame.image) {
                this._discarded.delete(frame.tick);
                this._pending.set(frame.tick, frame);
            } else if (!this._discarded.has(frame.tick) && !this._pending.get(frame.tick)?.image) {
                this._pending.set(frame.tick, frame);
            }
        }
        for (let frame = this._nextFrame(decoded); frame; frame = this._nextFrame(decoded)) {
            const { tick } = frame;
            const baseline = frame.image ? undefined : this._versions.get(frame.base);
            let packet: PreparedPacket;
            try {
                const rows = applyFrame(frame, baseline, this._layout);
                const prepared = this._needsPreparation ? this._prepare(rows) : undefined;
                packet = { ...frame, rows, prepared };
            } catch (cause) {
                failure ??= new Error(`Could not prepare replication tick ${tick}`, { cause });
                break;
            }
            if (packet.commandAge !== undefined) this.onCommandAge(tick, packet.commandAge);
            if (!latestOnly) {
                this.reconcile(tick, packet.rows, packet.intents, () => this._apply(packet), []);
            } else if (latest) {
                const previous = latest;
                preceding.push({
                    tick: previous.tick,
                    intents: previous.intents,
                    apply: () => {
                        this._apply(previous);
                        this.confirm(previous.tick, previous.rows);
                    },
                });
            }
            latest = packet;
            decoded = tick;
            this._versions.set(tick, packet.rows);
            for (const old of this._versions.keys())
                if (old < packet.base) this._versions.delete(old);
            this._pending.delete(tick);
            if (frame.image) {
                for (const old of this._pending.keys()) if (old < tick) this._pending.delete(old);
            }
            for (const old of this._discarded) if (old <= tick) this._discarded.delete(old);
            if (!latestOnly) this.confirm(tick, packet.rows);
        }
        if (latestOnly && latest) {
            const packet = latest;
            this.reconcile(
                packet.tick,
                packet.rows,
                packet.intents,
                () => this._apply(packet),
                preceding,
            );
            this.confirm(packet.tick, packet.rows);
        }
        if (failure) {
            this._pending.clear();
            this._discarded.clear();
            this._versions.clear();
            failServerConnection(this.connection, failure);
            return [];
        }
        if (this.confirmed > previous && typeof window !== "undefined")
            reportStartup({
                kind: "server-state",
                tick: this.confirmed,
                endpoint: this.connection.endpoint,
            });
        // A ticking World acknowledges once per fixed tick, even with nothing new, which
        // repairs a lost acknowledgement while the server is idle; one that never ticks
        // acknowledges each drain that received replication.
        const tick = this.world.time.fixedTick;
        if (this.confirmed >= 0 && (tick !== this._ackTick || (arrived && tick === this._start))) {
            this._ackTick = tick;
            let held = 0;
            for (const tick of this._pending.keys()) {
                const offset = tick - this.confirmed - 1;
                if (offset < AckWindow) held |= 1 << offset;
            }
            this.connection.send(encodeAcknowledgement(this.confirmed, held));
        }
        networkDiagnostics(this.world, this.connection)?.record(
            "confirmed_tick_lag",
            Math.max(0, this._newest - this.confirmed),
        );
        return unhandled;
    }
    private _nextFrame(decoded: number): DecodedFrame | undefined {
        let next = decoded + 1;
        if (decoded >= 0) {
            // Do not wait for an immutable delta that names a baseline already passed or discarded.
            for (; ; next++) {
                if (this._discarded.has(next)) continue;
                const frame = this._pending.get(next);
                if (!frame) break;
                if (frame.image || this._versions.has(frame.base)) return frame;
                this._pending.delete(next);
                this._discarded.add(next);
            }
        }
        let image: DecodedFrame | undefined;
        for (const [tick, frame] of this._pending) {
            if (frame.image) {
                if (!image || tick < image.tick) image = frame;
            } else if (
                decoded >= 0 &&
                ((frame.base < next && !this._versions.has(frame.base)) ||
                    this._discarded.has(frame.base))
            ) {
                this._pending.delete(tick);
                this._discarded.add(tick);
            }
        }
        // A full image can establish a baseline across a gap. While it is in flight, keep
        // the earliest deltas within the acknowledgement window; the sender can retry the rest.
        if (!image && decoded < 0 && this._pending.size > AckWindow) {
            const ticks = [...this._pending.keys()].sort((a, b) => a - b);
            for (let i = AckWindow; i < ticks.length; i++) this._pending.delete(ticks[i]!);
        }
        return image;
    }
    private _apply(packet: PreparedPacket): void {
        this._write(packet.rows, packet.prepared);
        this.confirmed = packet.tick;
    }
    /** Fallible field conversions finish before creating, destroying or writing any entity. */
    private _prepare(rows: ReplicatedRows): PreparedRows {
        const prepared: PreparedRows = new Map();
        for (const row of rows.values()) {
            for (const [index, values] of row) {
                let encoded: (TypedArray | undefined)[] | undefined;
                let lane = 0;
                for (const { field } of this._fields[index]!) {
                    if (field.byteCapacity !== undefined) {
                        const text = encodeText(values[lane] as string, field.byteCapacity);
                        const storage = new Uint8Array(field.lanes);
                        writeText(new DataView(storage.buffer), 0, text);
                        (encoded ??= [])[lane] = storage;
                    } else if (field.encode) {
                        const storage = new field.ctor(field.lanes);
                        for (let i = 0; i < field.lanes; i++)
                            storage[i] = field.encode((values[lane + i] as number) ?? 0);
                        (encoded ??= [])[lane] = storage;
                    }
                    lane += wireLanes(field);
                }
                if (encoded) prepared.set(values, encoded);
            }
        }
        return prepared;
    }
    private _write(rows: ReplicatedRows, prepared?: PreparedRows) {
        for (const [id, eid] of this.entities)
            if (!rows.has(id)) {
                this.world.destroy(eid);
                this.entities.delete(id);
            }
        for (const id of rows.keys())
            if (!this.entities.has(id)) this.entities.set(id, this.world.create());
        for (const [id, row] of rows) {
            const eid = this.entities.get(id)!;
            let written: Set<number> | undefined;
            let writePrepared: ((component: Component) => boolean) | undefined;
            this.schema.forEach((c, index) => {
                const values = row.get(index);
                if (!values) {
                    this.world.remove(eid, c);
                    return;
                }
                if (written?.has(index)) return;
                if (this.world.has(eid, c)) {
                    this._writeComponent(eid, index, values, prepared?.get(values));
                } else {
                    writePrepared ??= (component) => {
                        const index = this.schema.indexOf(component);
                        const values = row.get(index);
                        if (!values) return false;
                        this._writeComponent(eid, index, values, prepared?.get(values));
                        (written ??= new Set()).add(index);
                        return true;
                    };
                    this.world.addPrepared(eid, c, writePrepared);
                }
            });
        }
    }
    private _writeComponent(
        eid: number,
        index: number,
        values: ReplicatedValue[],
        encoded: (TypedArray | undefined)[] | undefined,
    ): void {
        const component = this.schema[index]!;
        const storage = this.world.storage(component) as Record<
            string,
            { set(eid: number, ...values: ReplicatedValue[]): void }
        >;
        let lane = 0;
        for (const { name, field } of this._fields[index]!) {
            const raw = encoded?.[lane];
            const width = wireLanes(field);
            if (raw) {
                const column = this.world.fieldStorage(component, name);
                column.markChanged(eid);
                column.column.set(raw, eid * field.lanes);
            } else if (field === entity) {
                storage[name]!.set(eid, this.entities.get(values[lane] as number) ?? 0);
            } else {
                storage[name]!.set(eid, ...values.slice(lane, lane + width));
            }
            lane += width;
        }
    }
}
