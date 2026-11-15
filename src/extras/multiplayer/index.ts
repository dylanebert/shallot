import {
    type Component,
    component,
    type EntityRef,
    entity,
    type Plugin,
    u32,
    type World,
} from "../../engine";
import { fields } from "../../engine/ecs";

export { type Connection, type LinkFaults, seededLink } from "./link";

import type { Connection } from "./link";

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
}

function layout(schema: ReplicationSchema) {
    return schema.map((c) =>
        fields(c).flatMap(({ field }) =>
            Array.from({ length: field.components ?? field.lanes }, () => field === entity),
        ),
    );
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

// u32 framing and f64 field values preserve every CPU field, including decoded f16 values.
function encode(packet: Packet, base: ReplicatedRows): Uint8Array {
    const words: number[] = [packet.image ? 1 : 2, packet.tick, packet.base, packet.rows.size];
    const values: number[] = [];
    for (const [id, row] of packet.rows) {
        words.push(id, row.size);
        for (const [c, fields] of row) {
            const previous = base.get(id)?.get(c);
            const changed = fields.flatMap((v, i) =>
                !previous || !Object.is(v, previous[i]) ? [i] : [],
            );
            words.push(c, changed.length);
            for (const i of changed) {
                words.push(i);
                values.push(fields[i]!);
            }
        }
    }
    const bytes = new Uint8Array(4 + words.length * 4 + values.length * 8);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, words.length, true);
    words.forEach((v, i) => {
        view.setUint32(4 + i * 4, v, true);
    });
    values.forEach((v, i) => {
        view.setFloat64(4 + words.length * 4 + i * 8, v, true);
    });
    return bytes;
}

function decode(bytes: Uint8Array, versions: Map<number, ReplicatedRows>, sizes: number[]): Packet {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const count = view.getUint32(0, true);
    let offset = 4;
    let valueOffset = 4 + count * 4;
    const word = () => {
        const v = view.getUint32(offset, true);
        offset += 4;
        return v;
    };
    const kind = word(),
        tick = word(),
        base = word(),
        entities = word();
    if (kind !== 1 && kind !== 2) throw new Error("Invalid replication packet");
    const baseline = kind === 1 ? new Map() : versions.get(base);
    if (!baseline) throw new Error("Missing replication baseline");
    const rows: ReplicatedRows = new Map();
    for (let e = 0; e < entities; e++) {
        const id = word(),
            components = word();
        const row = new Map<number, number[]>();
        for (let c = 0; c < components; c++) {
            const index = word(),
                changes = word();
            const size = sizes[index];
            if (size === undefined) throw new Error("Unknown replicated component");
            const values = baseline.get(id)?.get(index)?.slice() ?? Array<number>(size).fill(0);
            for (let f = 0; f < changes; f++) {
                const lane = word();
                if (lane >= size) throw new Error("Invalid replicated field");
                values[lane] = view.getFloat64(valueOffset, true);
                valueOffset += 8;
            }
            row.set(index, values);
        }
        rows.set(id, row);
    }
    if (offset !== 4 + count * 4 || valueOffset !== bytes.length)
        throw new Error("Invalid replication size");
    return { tick, base, rows, image: kind === 1 };
}

/** One recipient's acknowledged history. Call record once per authoritative tick, flush to retry. */
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
    record(tick: number) {
        if (!Number.isInteger(tick) || tick <= this._last || tick >= 0xffffffff)
            throw new Error("Replication ticks must increase");
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
        const packet = { tick, rows, image, base: image ? tick : base!.tick };
        const bytes = encode(packet, image ? new Map() : base!.rows);
        this._pending.set(tick, { version: { tick, rows }, bytes });
        this._last = tick;
    }
    flush() {
        for (const bytes of this.connection.receive()) {
            if (bytes.length !== 4) continue;
            const tick = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
            const version = this._pending.get(tick)?.version;
            if (!version) continue;
            this._acknowledged = version;
            for (const old of this._pending.keys()) if (old <= tick) this._pending.delete(old);
        }
        for (const { bytes } of this._pending.values()) this.connection.send(bytes);
    }
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
    constructor(
        world: World,
        schema: ReplicationSchema,
        connection: Connection,
        confirm: (tick: number, rows: ReplicatedRows) => void = () => {},
    ) {
        this.world = world;
        this.schema = schema;
        this.connection = connection;
        this.confirm = confirm;
    }
    flush() {
        for (const bytes of this.connection.receive()) {
            if (bytes.length < 20) continue;
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
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
            if (this.confirmed < 0 && view.getUint32(4, true) !== 1) break;
            const packet = decode(
                bytes,
                this._versions,
                layout(this.schema).map((f) => f.length),
            );
            this._write(packet.rows);
            this._versions.set(tick, packet.rows);
            for (const old of this._versions.keys())
                if (old < packet.base) this._versions.delete(old);
            this.confirmed = tick;
            this._pending.delete(tick);
            this.confirm(tick, packet.rows);
        }
        if (this.confirmed >= 0) {
            const ack = new Uint8Array(4);
            new DataView(ack.buffer).setUint32(0, this.confirmed, true);
            this.connection.send(ack);
        }
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
