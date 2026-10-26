/** Eid-indexed reverse lists in join order. Retargeting unlinks and appends to the new list. */
export class BodyDependencies {
    private _heads = new Uint32Array(16);
    private _tails = new Uint32Array(16);
    private _bodies = new Uint32Array(16);
    private _next = new Uint32Array(16);
    private _previous = new Uint32Array(16);

    set(eid: number, body: number): void {
        if ((this._bodies[eid] ?? 0) === body) return;
        const required = Math.max(eid, body);
        if (required >= this._bodies.length) {
            let capacity = this._bodies.length;
            while (capacity <= required) capacity *= 2;
            const grow = (source: Uint32Array) => {
                const target = new Uint32Array(capacity);
                target.set(source);
                return target;
            };
            this._heads = grow(this._heads);
            this._tails = grow(this._tails);
            this._bodies = grow(this._bodies);
            this._next = grow(this._next);
            this._previous = grow(this._previous);
        }
        const old = this._bodies[eid];
        if (old !== 0) {
            const before = this._previous[eid],
                after = this._next[eid];
            if (before) this._next[before] = after;
            else this._heads[old] = after;
            if (after) this._previous[after] = before;
            else this._tails[old] = before;
        }
        this._bodies[eid] = body;
        this._previous[eid] = body ? this._tails[body] : 0;
        this._next[eid] = 0;
        if (body) {
            const tail = this._tails[body];
            if (tail) this._next[tail] = eid;
            else this._heads[body] = eid;
            this._tails[body] = eid;
        }
    }

    first(body: number): number {
        return this._heads[body] ?? 0;
    }
    following(eid: number): number {
        return this._next[eid];
    }

    clear(): void {
        this._heads.fill(0);
        this._tails.fill(0);
        this._bodies.fill(0);
        this._next.fill(0);
        this._previous.fill(0);
    }

    snapshot() {
        return {
            heads: this._heads.slice(),
            tails: this._tails.slice(),
            bodies: this._bodies.slice(),
            next: this._next.slice(),
            previous: this._previous.slice(),
        };
    }

    restore(image: ReturnType<BodyDependencies["snapshot"]>): void {
        this._heads = image.heads.slice();
        this._tails = image.tails.slice();
        this._bodies = image.bodies.slice();
        this._next = image.next.slice();
        this._previous = image.previous.slice();
    }
}
