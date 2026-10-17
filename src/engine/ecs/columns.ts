import type { TypedArray } from "./component";
import type { WorldField } from "./storage";

/** @internal Field storage owns its capture boundary and never shrinks live columns. */
export class FieldColumns {
    highWater = 0;
    readonly #fields: WorldField[] = [];

    clear(): void {
        this.#fields.length = 0;
    }

    register(field: WorldField): void {
        this.#fields.push(field);
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
        const extent = this.highWater;
        this.highWater = state.highWater;
        for (let i = 0; i < this.#fields.length; i++)
            this.#fields[i].restore(state.fields[i], state.marks[i], extent);
    }
}
