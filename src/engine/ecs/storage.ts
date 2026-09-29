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
    gpu: GPUBuffer | null;
};

/** world-owned field column. The exposed accessors close over this record, not a component singleton. */
export class WorldField<T extends Type = Type> {
    readonly type: T;
    readonly storage: "sparse" | "slab";
    readonly name?: string;
    readonly #column: Column;
    readonly #observers = new Set<(eid: number) => void>();

    constructor(schema: FieldSchema<T>, initialCapacity: number) {
        this.type = schema.type;
        this.storage = schema.storage;
        this.name = schema.name;
        this.#column = {
            schema,
            array: new schema.type.ctor(initialCapacity * schema.type.lanes),
            dirty: new Uint32Array((initialCapacity + 31) >>> 5),
            gpu: null,
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

    observe(observer: (eid: number) => void): () => void {
        this.#observers.add(observer);
        return () => this.#observers.delete(observer);
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
        for (const observer of this.#observers) observer(eid);
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
        for (const observer of this.#observers) observer(eid);
    }

    get column(): TypedArray {
        return this.#column.array;
    }

    get dirty(): Uint32Array {
        return this.#column.dirty;
    }

    get gpu(): GPUBuffer | null {
        return this.#column.gpu;
    }

    set gpu(buffer: GPUBuffer | null) {
        this.#column.gpu = buffer;
    }

    bind(): FieldStorage<T> {
        const field = this;
        const lane = (offset: number): Single => ({
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
            get dirty() {
                return field.dirty;
            },
            get gpu() {
                return field.gpu;
            },
        });
        const base = { type: this.type };
        if (this.type.lanes === 1) {
            return {
                ...base,
                get column() {
                    return field.column;
                },
                get dirty() {
                    return field.dirty;
                },
                get gpu() {
                    return field.gpu;
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
                get dirty() {
                    return field.dirty;
                },
                get gpu() {
                    return field.gpu;
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
            get dirty() {
                return field.dirty;
            },
            get gpu() {
                return field.gpu;
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
        for (const observer of this.#observers) observer(eid);
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
