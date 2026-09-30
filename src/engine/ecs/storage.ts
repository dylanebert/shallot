import type { FieldSchema, Pair, Quad, Single, Type, TypedArray } from "./component";
import { sameComponentSchema } from "./component";

export type FieldStorage<T extends Type> = T["lanes"] extends 1
    ? Single
    : T["lanes"] extends 2
      ? Pair
      : Quad;

type Column = {
    schema: FieldSchema;
    array: TypedArray;
    dirty: Uint32Array;
};

/** world-owned field column. The exposed accessors close over this record, not a component singleton. */
export class WorldField<T extends Type = Type> {
    readonly type: T;
    readonly #column: Column;
    readonly #writeRows = new WeakMap<TypedArray, Map<number, TypedArray[]>>();

    constructor(schema: FieldSchema<T>, initialCapacity: number) {
        this.type = schema.type;
        this.#column = {
            schema,
            array: new schema.type.ctor(initialCapacity * schema.type.lanes),
            dirty: new Uint32Array((initialCapacity + 31) >>> 5),
        };
    }

    ensure(capacity: number): void {
        const column = this.#column;
        const oldCapacity = column.array.length / this.type.lanes;
        if (capacity <= oldCapacity) return;
        let next = Math.max(16, oldCapacity);
        while (next < capacity) next *= 2;
        const array = new this.type.ctor(next * this.type.lanes);
        array.set(column.array);
        column.array = array;
        const dirty = new Uint32Array((next + 31) >>> 5);
        dirty.set(column.dirty);
        column.dirty = dirty;
    }

    /** Publish an eid after writing its current raw column; retained arrays do not follow growth. */
    markChanged(eid: number): void {
        this.ensure(eid + 1);
        this.#column.dirty[eid >>> 5] |= 1 << (eid & 31);
    }

    set(eid: number, x: number, y = 0, z = 0, w = 0): void {
        this.ensure(eid + 1);
        const { array, dirty } = this.#column;
        const base = eid * this.type.lanes;
        const encode = this.type.encode ?? identity;
        array[base] = encode(x);
        if (this.type.lanes >= 2) array[base + 1] = encode(y);
        if (this.type.lanes === 4) {
            array[base + 2] = encode(z);
            array[base + 3] = encode(w);
        }
        dirty[eid >>> 5] |= 1 << (eid & 31);
    }

    /** Copy encoded typed rows and publish the same change marks as scalar setters. */
    write(eids: Uint32Array, source: TypedArray, lane = -1): void {
        const lanes = lane < 0 ? this.type.lanes : 1;
        if (!(eids instanceof Uint32Array))
            throw new Error("WorldField.write: eids must be Uint32Array");
        if (source.constructor !== this.#column.array.constructor) {
            throw new Error(
                `WorldField.write: expected ${this.#column.array.constructor.name}, received ${source.constructor.name}`,
            );
        }
        if (source.length !== eids.length * lanes) {
            throw new Error(
                `WorldField.write: source length ${source.length} does not match ${eids.length} rows with ${lanes} lanes`,
            );
        }
        let capacity = 0;
        for (let i = 0; i < eids.length; i++) capacity = Math.max(capacity, eids[i] + 1);
        this.ensure(capacity);
        let byLanes = this.#writeRows.get(source);
        if (!byLanes) {
            byLanes = new Map();
            this.#writeRows.set(source, byLanes);
        }
        let rows = byLanes.get(lanes);
        if (!rows) {
            rows = [];
            for (let i = 0; i < eids.length; i++)
                rows.push(source.subarray(i * lanes, (i + 1) * lanes) as TypedArray);
            byLanes.set(lanes, rows);
        }
        const { array, dirty } = this.#column;
        for (let i = 0; i < eids.length; i++) {
            const eid = eids[i];
            array.set(rows[i], eid * this.type.lanes + Math.max(0, lane));
            dirty[eid >>> 5] |= 1 << (eid & 31);
        }
    }

    get(eid: number, lane = 0): number {
        const value = this.#column.array[eid * this.type.lanes + lane] ?? 0;
        return this.type.decode ? this.type.decode(value) : value;
    }

    read(eid: number, out: Float32Array): Float32Array {
        for (let lane = 0; lane < this.type.lanes; lane++) out[lane] = this.get(eid, lane);
        return out;
    }

    clear(eid: number): void {
        const { array, dirty } = this.#column;
        const base = eid * this.type.lanes;
        if (base + this.type.lanes > array.length) return;
        for (let lane = 0; lane < this.type.lanes; lane++) array[base + lane] = 0;
        dirty[eid >>> 5] |= 1 << (eid & 31);
    }

    get column(): TypedArray {
        return this.#column.array;
    }

    get dirty(): Uint32Array {
        return this.#column.dirty;
    }

    bind(): FieldStorage<T> {
        const field = this;
        const lane = (offset: number): Single => ({
            write(eids, source) {
                field.write(eids, source, offset);
            },
            set(eid, value) {
                field.setLane(eid, offset, value);
            },
            get(eid) {
                return field.get(eid, offset);
            },
            type: this.type,
            get column() {
                return field.column;
            },
            markChanged(eid) {
                field.markChanged(eid);
            },
        });
        const base = {
            type: this.type,
            markChanged: (eid: number) => this.markChanged(eid),
            write: (eids: Uint32Array, source: TypedArray) => this.write(eids, source),
        };
        if (this.type.lanes === 1) {
            return {
                ...base,
                get column() {
                    return field.column;
                },

                set: (eid: number, value: number) => this.set(eid, value),
                get: (eid: number) => this.get(eid),
            } as unknown as FieldStorage<T>;
        }
        if (this.type.lanes === 2) {
            return {
                ...base,
                get column() {
                    return field.column;
                },

                set: (eid: number, x: number, y: number) => this.set(eid, x, y),
                read: (eid: number, out: Float32Array) => this.read(eid, out),
                x: lane(0),
                y: lane(1),
            } as unknown as FieldStorage<T>;
        }
        return {
            ...base,
            get column() {
                return field.column;
            },
            set: (eid: number, x: number, y: number, z: number, w: number) =>
                this.set(eid, x, y, z, w),
            read: (eid: number, out: Float32Array) => this.read(eid, out),
            x: lane(0),
            y: lane(1),
            z: lane(2),
            w: lane(3),
        } as unknown as FieldStorage<T>;
    }

    private setLane(eid: number, lane: number, value: number): void {
        this.ensure(eid + 1);
        const encode = this.type.encode ?? identity;
        this.#column.array[eid * this.type.lanes + lane] = encode(value);
        this.#column.dirty[eid >>> 5] |= 1 << (eid & 31);
    }
}

function identity(value: number): number {
    return value;
}

export type ComponentStorage<T> = {
    [K in keyof T]: T[K] extends FieldSchema<infer F> ? FieldStorage<F> : T[K];
};

export function sameSchema(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
    return sameComponentSchema(a, b);
}
