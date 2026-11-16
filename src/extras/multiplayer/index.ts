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

export { ClientInterpolation, InterpolationPlugin, InterpolationTarget } from "./interpolation";
export { type Connection, type LinkFaults, seededLink } from "./link";
export {
    ClientPrediction,
    contactOutputKey,
    FixedOutputs,
    PredictionMotion,
    PredictionPlugin,
    PredictionTarget,
} from "./prediction";

import { type AcceptedIntent, decodeIntent, encodeIntent, IntentMessageKind } from "./intent";
import type { Connection } from "./link";

export {
    type AcceptedIntent,
    IntentMessageKind,
    IntentReceiver,
    type IntentRecord,
    IntentSender,
} from "./intent";

/** Selects a disclosure rule. Only entities holding this component can be sent. */
export const Replicate = component("Replicate", { rule: u32 });
export const MultiplayerPlugin: Plugin = { name: "Multiplayer", components: [Replicate] };

/** Rules are indexed by Replicate.rule; an absent recipient receives nothing. */
export type ReplicationRules = readonly ReadonlyMap<number, readonly Component[]>[];
/** Both ends must use the same ordered component schemas and engine version. */
export type ReplicationSchema = readonly Component[];
export type ReplicatedRows = Map<number, Map<number, number[]>>;
interface Version {
    tick: number;
    rows: ReplicatedRows;
}
interface Packet extends Version {
    base: number;
    image: boolean;
    intents: readonly AcceptedIntent[];
}

/** Little-endian u32 replication kinds. flush returns other kinds unchanged for routing. */
export const ReplicationMessageKind = { Image: 1, Frame: 2, Acknowledgement: 3 } as const;

function layout(schema: ReplicationSchema) {
    return schema.map((c) =>
        fields(c).flatMap(({ field }) =>
            Array<FieldType>(field.components ?? field.lanes).fill(field === entity ? u32 : field),
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
        const row = new Map<number, number[]>();
        for (const c of components) {
            const index = schema.indexOf(c);
            if (index < 0) throw new Error("Disclosure component missing from replication schema");
            if (!world.has(eid, c)) continue;
            const record = world.get(eid, c)! as Record<string, number | number[]>;
            const values: number[] = [];
            for (const { name, field } of fields(c)) {
                const value = record[name] as number | number[];
                if (field === entity) values.push(ids.get(value as number) ?? 0);
                else if (typeof value === "number") values.push(value);
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
    ];
    const values: { value: number; field: FieldType }[] = [];
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
                values.push({ value: fields[i]!, field: types[c]![i]! });
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
    const bytes = new Uint8Array(
        4 +
            words.length * 4 +
            values.reduce((size, v) => size + v.field.ctor.BYTES_PER_ELEMENT, 0) +
            intents.length,
    );
    const view = new DataView(bytes.buffer);
    view.setUint32(0, words[0]!, true);
    view.setUint32(4, words.length, true);
    for (let i = 1; i < words.length; i++) view.setUint32(4 + i * 4, words[i]!, true);
    let offset = 4 + words.length * 4;
    for (const { value, field } of values) {
        writeField(view, offset, field, value);
        offset += field.ctor.BYTES_PER_ELEMENT;
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
        const row = new Map<number, number[]>(baseline.get(id));
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
            const values = baseline.get(id)?.get(index)?.slice() ?? Array<number>(size).fill(0);
            for (let f = 0; f < changes; f++) {
                const lane = word();
                if (lane >= size) throw new Error("Invalid replicated field");
                const field = componentTypes[lane]!;
                values[lane] = readField(view, valueOffset, field);
                valueOffset += field.ctor.BYTES_PER_ELEMENT;
            }
            row.set(index, values);
        }
        rows.set(id, row);
    }
    if (offset !== 4 + count * 4 || valueOffset > bytes.length)
        throw new Error("Invalid replication size");
    const intents = valueOffset < bytes.length ? decodeIntent(bytes.subarray(valueOffset)) : [];
    return { tick, base, rows, intents, image: kind === ReplicationMessageKind.Image };
}

/** One recipient's acknowledged history. record starts at any nonnegative tick and refuses gaps thereafter; flush retries pending ticks. */
export class ReplicationSender {
    private _identities = new Map<EntityRef, number>();
    private _next = { value: 1 };
    private _acknowledged: Version | undefined;
    private _pending = new Map<number, { version: Version; bytes: Uint8Array }>();
    private _last = -1;
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
    }
    /** Include this recipient's accepted log; the image and frames carry records through their tick. */
    record(tick: number, intents: readonly AcceptedIntent[] = []) {
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
        const baseline = this._acknowledged;
        // Until the image is acknowledged subsequent frames use its rows, not an empty baseline.
        const base = baseline ?? this._pending.values().next().value?.version;
        const packet = {
            tick,
            rows,
            image,
            intents: intents.filter((r) => r.tick <= tick),
            base: image ? tick : base!.tick,
        };
        const bytes = encode(packet, image ? new Map() : base!.rows, layout(this.schema));
        this._pending.set(tick, { version: { tick, rows }, bytes });
        this._last = tick;
    }
    /** Retries pending frames and returns non-acknowledgement messages for application routing. */
    flush(): Uint8Array[] {
        const unhandled: Uint8Array[] = [];
        for (const bytes of this.connection.receive()) {
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            if (
                bytes.length !== 8 ||
                view.getUint32(0, true) !== ReplicationMessageKind.Acknowledgement
            ) {
                unhandled.push(bytes);
                continue;
            }
            const tick = view.getUint32(4, true);
            const version = this._pending.get(tick)?.version;
            if (!version) continue;
            this._acknowledged = version;
            for (const old of this._pending.keys()) if (old <= tick) this._pending.delete(old);
        }
        for (const { bytes } of this._pending.values()) this.connection.send(bytes);
        return unhandled;
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
    readonly entities = new Map<number, number>();
    private _versions = new Map<number, ReplicatedRows>();
    private _pending = new Map<number, Uint8Array>();
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
    }
    /** Applies confirmations and returns other message kinds for application routing.
     * "latest" reconciles once, passing preceding ordered states to that callback. It must
     * deliver their intents and apply each before the latest state. Every applied state
     * still notifies confirm, so only recovery/replay work is coalesced, not observations. */
    flush(confirmations: "all" | "latest" = "all"): Uint8Array[] {
        const latestOnly = confirmations === "latest";
        const unhandled: Uint8Array[] = [];
        let latest: ReturnType<typeof decode> | undefined;
        let latestTick = -1;
        const preceding: ReplicationConfirmation[] = [];
        for (const bytes of this.connection.receive()) {
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            const kind = bytes.length >= 4 ? view.getUint32(0, true) : 0;
            if (kind !== ReplicationMessageKind.Image && kind !== ReplicationMessageKind.Frame) {
                unhandled.push(bytes);
                continue;
            }
            if (bytes.length < 20) throw new Error("Truncated replication packet");
            const tick = view.getUint32(8, true);
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
        if (this.confirmed >= 0) {
            const ack = new Uint8Array(8);
            const view = new DataView(ack.buffer);
            view.setUint32(0, ReplicationMessageKind.Acknowledgement, true);
            view.setUint32(4, this.confirmed, true);
            this.connection.send(ack);
        }
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
                    { set(eid: number, ...values: number[]): void }
                >;
                let lane = 0;
                for (const { name, field } of fields(c)) {
                    const value = values.slice(lane, lane + (field.components ?? field.lanes));
                    lane += value.length;
                    if (field === entity) value[0] = this.entities.get(value[0]!) ?? 0;
                    storage[name]!.set(eid, ...value);
                }
            });
        }
    }
}
