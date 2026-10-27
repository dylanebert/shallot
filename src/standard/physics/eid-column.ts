/** One number per eid, zero where none is set, growing to the highest eid written. */
export class EidColumn {
    private _values = new Float64Array(64);

    get(eid: number): number {
        return this._values[eid] ?? 0;
    }

    set(eid: number, value: number): void {
        if (eid >= this._values.length) {
            let capacity = this._values.length;
            while (capacity <= eid) capacity *= 2;
            const grown = new Float64Array(capacity);
            grown.set(this._values);
            this._values = grown;
        }
        this._values[eid] = value;
    }

    delete(eid: number): void {
        if (eid < this._values.length) this._values[eid] = 0;
    }

    clear(): void {
        this._values.fill(0);
    }
}
