import type { TypedArray } from "./component";
import type { WorldField } from "./storage";

/** @internal Field storage owns its capture boundary and never shrinks live columns. */
export class FieldColumns {
    highWater = 0;
    #capacity = Infinity;
    readonly #fields: WorldField[] = [];

    clear(): void {
        this.#fields.length = 0;
        this.#capacity = Infinity;
    }

    register(field: WorldField): void {
        this.#fields.push(field);
        this.#capacity = Math.min(this.#capacity, field.column.length / field.type.lanes);
    }

    ensure(capacity: number): void {
        if (capacity <= this.#capacity) return;
        let next = Infinity;
        for (const field of this.#fields) {
            field.ensure(capacity);
            next = Math.min(next, field.column.length / field.type.lanes);
        }
        this.#capacity = next;
    }

    snapshot(): { highWater: number; fields: TypedArray[]; marks: Uint32Array[] } {
        const words = (this.highWater + 31) >>> 5;
        return {
            highWater: this.highWater,
            fields: this.#fields.map((field) => field.snapshot(this.highWater)),
            marks: this.#fields.map((field) => field.dirty.slice(0, words)),
        };
    }

    restore(state: ReturnType<FieldColumns["snapshot"]>): void {
        this.highWater = state.highWater;
        for (let i = 0; i < this.#fields.length; i++)
            this.#fields[i].restore(state.fields[i], state.marks[i]);
    }
}
