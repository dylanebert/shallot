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

    snapshot(): { highWater: number; fields: TypedArray[] } {
        return {
            highWater: this.highWater,
            fields: this.#fields.map((field) => field.snapshot(this.highWater)),
        };
    }

    restore(state: ReturnType<FieldColumns["snapshot"]>): void {
        this.highWater = state.highWater;
        for (let i = 0; i < this.#fields.length; i++) this.#fields[i].restore(state.fields[i]);
    }
}
