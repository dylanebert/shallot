import {
    type Component,
    component,
    type EntityRef,
    entity,
    type FieldType,
    type Plugin,
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

import { networkDiagnostics } from "./diagnostics";
import { type AcceptedIntent, decodeIntent, encodeIntent, IntentMessageKind } from "./intent";
import type { Connection } from "./link";
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

function encode(packet: Packet, base: ReplicatedRows, types: FieldType[][]): Uint8Array {
    const words: number[] = [
        packet.image ? ReplicationMessageKind.Image : ReplicationMessageKind.Frame,
        packet.tick,
        packet.base,
        0,
        packet.commandAge === undefined ? 0x80000000 : Math.round(packet.commandAge * 256),
    ];
    const values: { value: number | Uint8Array; field: FieldType }[] = [];
    let valueBytes = 0;
    let entities = 0;
    for (const [id, row] of packet.rows) {
        const previousRow = base.get(id);
        const components: number[] = [];
        let changes = 0;
        for (const [c, fields] of row) {
            const previous = previousRow?.get(c);
            const changed = fields.flatMap((v, i) =>
                !previous || !Object.is(v, previous[i]) ? [i] : [],
            );
            if (previous && changed.length === 0) continue;
            components.push(c, changed.length);
            changes++;
            for (const i of changed) {
                components.push(i);
                const field = types[c]![i]!;
                const value =
                    field.byteCapacity === undefined
                        ? (fields[i] as number)
                        : encodeText(fields[i] as string, field.byteCapacity);
                values.push({ value, field });
                valueBytes +=
                    typeof value === "number" ? field.ctor.BYTES_PER_ELEMENT : 2 + value.length;
            }
        }
        for (const c of previousRow?.keys() ?? []) {
            if (row.has(c)) continue;
            components.push(c, Removed);
            changes++;
        }
        if (previousRow && changes === 0) continue;
        words.push(id, changes, ...components);
        entities++;
    }
    for (const id of base.keys()) {
        if (packet.rows.has(id)) continue;
        words.push(id, Removed);
        entities++;
    }
    words[3] = entities;
    const intents = packet.intents.length
        ? encodeIntent(IntentMessageKind.Accepted, packet.intents)
        : new Uint8Array();
    const bytes = new Uint8Array(4 + words.length * 4 + valueBytes + intents.length);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, words[0]!, true);
    view.setUint32(4, words.length, true);
    for (let i = 1; i < words.length; i++) view.setUint32(4 + i * 4, words[i]!, true);
    let offset = 4 + words.length * 4;
    for (const { value, field } of values) {
        if (typeof value === "number") {
            writeField(view, offset, field, value);
            offset += field.ctor.BYTES_PER_ELEMENT;
        } else {
            writeText(view, offset, value);
            offset += 2 + value.length;
        }
    }
    bytes.set(intents, offset);
    return bytes;
}

function decode(
    bytes: Uint8Array,
    versions: Map<number, ReplicatedRows>,
    types: FieldType[][],
): Packet {
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
    const baseline = kind === ReplicationMessageKind.Image ? new Map() : versions.get(base);
    if (!baseline) throw new Error("Missing replication baseline");
    const rows: ReplicatedRows = new Map(baseline);
    for (let e = 0; e < entities; e++) {
        const id = word(),
            components = word();
        if (components === Removed) {
            rows.delete(id);
            continue;
        }
        const row = new Map<number, ReplicatedValue[]>(baseline.get(id));
        for (let c = 0; c < components; c++) {
            const index = word(),
                changes = word();
            const componentTypes = types[index];
            if (componentTypes === undefined) throw new Error("Unknown replicated component");
            if (changes === Removed) {
                row.delete(index);
                continue;
            }
            const size = componentTypes.length;
            const values =
                baseline.get(id)?.get(index)?.slice() ??
                componentTypes.map((field) => (field.byteCapacity === undefined ? 0 : ""));
            for (let f = 0; f < changes; f++) {
                const lane = word();
                if (lane >= size) throw new Error("Invalid replicated field");
                const field = componentTypes[lane]!;
                if (field.byteCapacity === undefined) {
                    values[lane] = readField(view, valueOffset, field);
                    valueOffset += field.ctor.BYTES_PER_ELEMENT;
                } else {
                    values[lane] = readText(view, valueOffset, field.byteCapacity);
                    valueOffset += 2 + view.getUint16(valueOffset, true);
                }
            }
            row.set(index, values);
        }
        rows.set(id, row);
    }
    if (offset !== 4 + count * 4 || valueOffset > bytes.length)
        throw new Error("Invalid replication size");
    const intents = valueOffset < bytes.length ? decodeIntent(bytes.subarray(valueOffset)) : [];
    return { tick, base, rows, intents, commandAge, image: kind === ReplicationMessageKind.Image };
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
        const bytes = encode(packet, image ? new Map() : this._bases.get(base)!, this._layout);
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
            const rows = decode(frame.bytes, this._bases, this._layout).rows;
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

/** Ordered confirmations, including the initial image. Application callbacks may retain rows but not mutate them. */
export class ReplicationReceiver {
    confirmed = -1;
    /** Once per ordered server tick. Command age is in fixed ticks; positive means late. */
    onCommandAge: (tick: number, age: number) => void = () => {};
    readonly entities = new Map<number, number>();
    private _versions = new Map<number, ReplicatedRows>();
    private _pending = new Map<number, Uint8Array>();
    private _newest = -1;
    private _ackTick: number;
    private readonly _start: number;
    private _generation?: number;
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
     * Acknowledges once per fixed tick of its World, or, while that World has not ticked, once
     * per drain that received a replication packet; never per call. */
    flush(confirmations: "all" | "latest" = "all"): Uint8Array[] {
        const generation = this.connection.generation;
        if (this._generation !== generation) {
            this._generation = generation;
            this.confirmed = -1;
            this._newest = -1;
            this._pending.clear();
            this._versions.clear();
        }
        const previous = this.confirmed;
        const latestOnly = confirmations === "latest";
        const unhandled: Uint8Array[] = [];
        let latest: ReturnType<typeof decode> | undefined;
        let latestTick = -1;
        const preceding: ReplicationConfirmation[] = [];
        let arrived = false;
        for (const bytes of this.connection.receive()) {
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            const kind = bytes.length >= 4 ? view.getUint32(0, true) : 0;
            if (kind !== ReplicationMessageKind.Image && kind !== ReplicationMessageKind.Frame) {
                unhandled.push(bytes);
                continue;
            }
            if (bytes.length < 24) throw new Error("Truncated replication packet");
            arrived = true;
            const tick = view.getUint32(8, true);
            this._newest = Math.max(this._newest, tick);
            if (tick <= this.confirmed) continue;
            this._pending.set(tick, bytes);
        }
        while (this._pending.size) {
            const tick =
                this.confirmed < 0 ? Math.min(...this._pending.keys()) : this.confirmed + 1;
            const bytes = this._pending.get(tick);
            if (!bytes) break;
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            if (this.confirmed < 0 && view.getUint32(0, true) !== ReplicationMessageKind.Image)
                break;
            const packet = decode(bytes, this._versions, layout(this.schema));
            if (packet.commandAge !== undefined) this.onCommandAge(tick, packet.commandAge);
            if (!latestOnly)
                this.reconcile(
                    tick,
                    packet.rows,
                    packet.intents,
                    () => this._write(packet.rows),
                    [],
                );
            else if (latest) {
                const previous = latest;
                const previousTick = latestTick;
                preceding.push({
                    tick: previousTick,
                    intents: previous.intents,
                    apply: () => {
                        this._write(previous.rows);
                        this.confirm(previousTick, previous.rows);
                    },
                });
            }
            latest = packet;
            latestTick = tick;
            this._versions.set(tick, packet.rows);
            for (const old of this._versions.keys())
                if (old < packet.base) this._versions.delete(old);
            this.confirmed = tick;
            this._pending.delete(tick);
            if (!latestOnly) this.confirm(tick, packet.rows);
        }
        if (latestOnly && latest) {
            const packet = latest;
            this.reconcile(
                latestTick,
                packet.rows,
                packet.intents,
                () => this._write(packet.rows),
                preceding,
            );
            this.confirm(latestTick, packet.rows);
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
            const ack = new Uint8Array(12);
            const view = new DataView(ack.buffer);
            view.setUint32(0, ReplicationMessageKind.Acknowledgement, true);
            view.setUint32(4, this.confirmed, true);
            view.setUint32(8, held >>> 0, true);
            this.connection.send(ack);
        }
        networkDiagnostics(this.world, this.connection)?.record(
            "confirmed_tick_lag",
            Math.max(0, this._newest - this.confirmed),
        );
        return unhandled;
    }
    private _write(rows: ReplicatedRows) {
        for (const [id, eid] of this.entities)
            if (!rows.has(id)) {
                this.world.destroy(eid);
                this.entities.delete(id);
            }
        for (const id of rows.keys())
            if (!this.entities.has(id)) this.entities.set(id, this.world.create());
        for (const [id, row] of rows) {
            const eid = this.entities.get(id)!;
            this.schema.forEach((c, index) => {
                const values = row.get(index);
                if (!values) {
                    this.world.remove(eid, c);
                    return;
                }
                if (!this.world.has(eid, c)) this.world.add(eid, c);
                const storage = this.world.storage(c) as Record<
                    string,
                    { set(eid: number, ...values: ReplicatedValue[]): void }
                >;
                let lane = 0;
                for (const { name, field } of fields(c)) {
                    const value = values.slice(lane, lane + wireLanes(field));
                    lane += value.length;
                    if (field === entity) value[0] = this.entities.get(value[0] as number) ?? 0;
                    storage[name]!.set(eid, ...value);
                }
            });
        }
    }
}
