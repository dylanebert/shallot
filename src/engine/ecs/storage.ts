import {
    entity,
    type FieldType,
    type ScalarField,
    type TypedArray,
    type Vector2Field,
    type Vector4Field,
} from "./component";
import type { EntityRef } from "./entity";
import type { World } from "./world";

export type FieldStorage<T extends FieldType> = T["lanes"] extends 1
    ? ScalarField
    : T["lanes"] extends 2
      ? Vector2Field
      : Vector4Field;

type Column = {
    array: TypedArray;
    dirty: Uint32Array;
};

/** world-owned field column. The exposed accessors close over this record, not a component singleton. */
export class WorldField<T extends FieldType = FieldType> {
    readonly type: T;
    readonly #column: Column;
    readonly #writeRows = new WeakMap<TypedArray, Map<number, TypedArray[]>>();

    private readonly _world?: World;

    constructor(schema: T, initialCapacity: number, world?: World) {
        if (schema === entity && !world) throw new Error("Entity fields require a World");
        this._world = world;
        this.type = schema;
        this.#column = {
            array: new schema.ctor(initialCapacity * schema.lanes),
            dirty: new Uint32Array((initialCapacity + 31) >>> 5),
        };
    }

    /** @internal Independent column image. */
    snapshot(capacity = this.#column.array.length / this.type.lanes): TypedArray {
        return this.#column.array.slice(0, capacity * this.type.lanes);
    }

    /** @internal Restore without shrinking retained accessors; publish every restored lane. */
    restore(state: TypedArray): void {
        this.ensure(state.length / this.type.lanes);
        this.#column.array.fill(0);
        this.#column.array.set(state);
        this.#column.dirty.fill(0xffffffff);
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
        const lanes = this.type.lanes;
        const column = this.#column;
        const base = eid * lanes;
        if (base >= column.array.length) this.ensure(eid + 1);
        const array = column.array;
        if (this.type === entity) x = this._world!.ref(x);
        const encode = this.type.encode;
        if (encode) {
            array[base] = encode(x);
            if (lanes >= 2) array[base + 1] = encode(y);
            if (lanes === 4) {
                array[base + 2] = encode(z);
                array[base + 3] = encode(w);
            }
        } else {
            array[base] = x;
            if (lanes >= 2) array[base + 1] = y;
            if (lanes === 4) {
                array[base + 2] = z;
                array[base + 3] = w;
            }
        }
        column.dirty[eid >>> 5] |= 1 << (eid & 31);
    }

    /** Copy encoded typed rows and publish the same change marks as scalar setters. */
    writeEncoded(eids: Uint32Array, source: TypedArray, lane = -1): void {
        const lanes = lane < 0 ? this.type.lanes : 1;
        if (!(eids instanceof Uint32Array))
            throw new Error("WorldField.writeEncoded: eids must be Uint32Array");
        if (source.constructor !== this.#column.array.constructor) {
            throw new Error(
                `WorldField.writeEncoded: expected ${this.#column.array.constructor.name}, received ${source.constructor.name}`,
            );
        }
        if (source.length !== eids.length * lanes) {
            throw new Error(
                `WorldField.writeEncoded: source length ${source.length} does not match ${eids.length} rows with ${lanes} lanes`,
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
        const type = this.type;
        if (type === entity) return this._world!.resolve(value as EntityRef);
        return type.decode ? type.decode(value) : value;
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
        if (!this.type.encode && !this.type.decode && this.type !== entity)
            return this.bindIdentity();
        const lane = (offset: number): ScalarField => ({
            writeEncoded(eids, source) {
                field.writeEncoded(eids, source, offset);
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
            writeEncoded: (eids: Uint32Array, source: TypedArray) =>
                this.writeEncoded(eids, source),
        };
        if (this.type.lanes === 1) {
            return {
                ...base,
                get column() {
                    return field.column;
                },

                set: (eid: number, value: number) => this.set(eid, value),
                get:
                    this.type === entity
                        ? (eid: number) =>
                              this._world!.resolve((this.#column.array[eid] ?? 0) as EntityRef)
                        : (eid: number) => this.get(eid),
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

    // Plain fields are the hot path; avoid routing their writes through codec-aware setters.
    private bindIdentity(): FieldStorage<T> {
        const field = this;
        const column = this.#column;
        const lanes = this.type.lanes;
        const type = this.type;
        const lane = (offset: number): ScalarField => ({
            writeEncoded(eids, source) {
                field.writeEncoded(eids, source, offset);
            },
            set(eid, value) {
                const i = eid * lanes + offset;
                if (i >= column.array.length) field.ensure(eid + 1);
                column.array[i] = value;
                column.dirty[eid >>> 5] |= 1 << (eid & 31);
            },
            get(eid) {
                return column.array[eid * lanes + offset] ?? 0;
            },
            type,
            get column() {
                return column.array;
            },
            markChanged(eid) {
                field.markChanged(eid);
            },
        });
        const base = {
            markChanged: (eid: number) => field.markChanged(eid),
            writeEncoded: (eids: Uint32Array, source: TypedArray) =>
                field.writeEncoded(eids, source),
        };
        if (lanes === 1) {
            return Object.defineProperties(
                {
                    type,
                    markChanged: base.markChanged,
                    writeEncoded: base.writeEncoded,
                    set: (eid: number, value: number) => {
                        if (eid >= column.array.length) field.ensure(eid + 1);
                        column.array[eid] = value;
                        column.dirty[eid >>> 5] |= 1 << (eid & 31);
                    },
                    get: (eid: number) => column.array[eid] ?? 0,
                },
                { column: { get: () => column.array, enumerable: true } },
            ) as unknown as FieldStorage<T>;
        }
        if (lanes === 2) {
            return Object.defineProperties(
                {
                    type,
                    markChanged: base.markChanged,
                    writeEncoded: base.writeEncoded,
                    set: (eid: number, x: number, y = 0) => {
                        const o = eid * 2;
                        if (o >= column.array.length) field.ensure(eid + 1);
                        const a = column.array;
                        a[o] = x;
                        a[o + 1] = y;
                        column.dirty[eid >>> 5] |= 1 << (eid & 31);
                    },
                    read: (eid: number, out: Float32Array) => field.read(eid, out),
                    x: lane(0),
                    y: lane(1),
                },
                { column: { get: () => column.array, enumerable: true } },
            ) as unknown as FieldStorage<T>;
        }
        return Object.defineProperties(
            {
                type,
                markChanged: base.markChanged,
                writeEncoded: base.writeEncoded,
                set: (eid: number, x: number, y = 0, z = 0, w = 0) => {
                    const o = eid * 4;
                    if (o >= column.array.length) field.ensure(eid + 1);
                    const a = column.array;
                    a[o] = x;
                    a[o + 1] = y;
                    a[o + 2] = z;
                    a[o + 3] = w;
                    column.dirty[eid >>> 5] |= 1 << (eid & 31);
                },
                read: (eid: number, out: Float32Array) => field.read(eid, out),
                x: lane(0),
                y: lane(1),
                z: lane(2),
                w: lane(3),
            },
            { column: { get: () => column.array, enumerable: true } },
        ) as unknown as FieldStorage<T>;
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
    [K in keyof T]: T[K] extends FieldType ? FieldStorage<T[K]> : T[K];
};
