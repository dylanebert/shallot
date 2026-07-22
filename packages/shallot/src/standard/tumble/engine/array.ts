// A growable typed-array vector — the port's analogue of Box3D's b3Array(T) macro for numeric
// element types, and the pool every SoA column (bodies, contacts, joints) is built from.
//
// Typed arrays are fixed-length, so growth reallocates and copies, mirroring the C realloc. The
// growth constants (push doubles from 8, emplace from 16) match container.h so capacity tracks
// the reference; nothing here is arithmetic, so no fround discipline applies. removeSwap returns
// the moved element's old index, or NULL_INDEX (-1) when the tail was removed — the b3Array
// contract the solver relies on.

export const NULL_INDEX = -1;

interface NumericArray {
    readonly length: number;
    [index: number]: number;
    set(array: ArrayLike<number>, offset?: number): void;
    subarray(begin?: number, end?: number): NumericArray;
}

export class GrowVec<T extends NumericArray> {
    data: T;
    count = 0;
    capacity: number;
    private readonly _make: (n: number) => T;

    constructor(make: (n: number) => T, capacity = 0) {
        this._make = make;
        this.capacity = capacity;
        this.data = make(capacity);
    }

    private grow(newCapacity: number): void {
        const next = this._make(newCapacity);
        next.set(this.data.subarray(0, this.count));
        this.data = next;
        this.capacity = newCapacity;
    }

    reserve(n: number): void {
        if (this.capacity < n) {
            this.grow(n);
        }
    }

    resize(n: number): void {
        this.reserve(n);
        this.count = n;
    }

    push(value: number): void {
        if (this.count >= this.capacity) {
            this.grow(this.capacity === 0 ? 8 : 2 * this.capacity);
        }
        this.data[this.count++] = value;
    }

    pop(): number {
        return this.data[--this.count];
    }

    get(index: number): number {
        return this.data[index];
    }

    set(index: number, value: number): void {
        this.data[index] = value;
    }

    /** Add an uninitialized element and return its index. */
    emplace(): number {
        if (this.count >= this.capacity) {
            this.grow(this.capacity === 0 ? 16 : 2 * this.capacity);
        }
        return this.count++;
    }

    /**
     * Remove by swapping the last element into `index`. Returns the last element's old index
     * (now out of bounds), or NULL_INDEX if `index` was the last element.
     */
    removeSwap(index: number): number {
        this.count -= 1;
        if (index !== this.count) {
            this.data[index] = this.data[this.count];
            return this.count;
        }
        return NULL_INDEX;
    }

    clear(): void {
        this.count = 0;
    }
}

/**
 * Swap-remove from a plain array, mirroring `b3Array_RemoveSwap` for the lifecycle columns (body
 * sims, contact indices, island links) that the scalar port stores as object/number arrays rather
 * than typed-array pools. Moves the last element into `index` and pops. Returns the moved element's
 * old index (the new length), or NULL_INDEX if `index` was the last element — the same contract
 * callers use to fix the moved element's back-index.
 */
export function swapRemove<T>(arr: T[], index: number): number {
    const last = arr.length - 1;
    if (index !== last) {
        arr[index] = arr[last];
        arr.pop();
        return last;
    }
    arr.pop();
    return NULL_INDEX;
}

/**
 * In-place quicksort operating purely on indices through `less`/`swap` callbacks — a faithful port
 * of Box3D's QSORT macro (qsort.h, Alexey Tourbin): median-of-3 pivot with an insertion-sort cutoff
 * at 16. The exact comparison + swap sequence is load-bearing: it fixes the order of equal keys,
 * which the mesh-contact tentative-triangle pass relies on for bit-exact manifold selection.
 */
export function qsort(
    n: number,
    less: (i: number, j: number) => boolean,
    swap: (i: number, j: number) => void,
): void {
    if (n <= 1) {
        return;
    }
    const Thresh = 16;

    const sort3 = (a1: number, a2: number, a3: number): void => {
        if (less(a2, a1)) {
            if (less(a3, a2)) {
                swap(a1, a3);
            } else {
                swap(a1, a2);
                if (less(a3, a2)) swap(a2, a3);
            }
        } else if (less(a3, a2)) {
            swap(a2, a3);
            if (less(a2, a1)) swap(a1, a2);
        }
    };

    let l = 0;
    let r = n - 1;
    const stackL: number[] = [];
    const stackR: number[] = [];

    while (true) {
        if (r - l + 1 >= Thresh) {
            const m = l + ((r - l) >> 1);
            sort3(l + 1, m, r);
            swap(l, m);
            let i = l + 1;
            let j = r;
            while (true) {
                do {
                    i++;
                } while (less(i, l));
                do {
                    j--;
                } while (less(l, j));
                if (i >= j) break;
                swap(i, j);
            }
            i = j + 1;
            swap(l, j);
            j--;

            // Two subfiles [l, j] and [i, r]; recurse on the smaller, stack the larger.
            let l1: number;
            let r1: number;
            let l2: number;
            let r2: number;
            if (j - l >= r - i) {
                l1 = l;
                r1 = j;
                l2 = i;
                r2 = r;
            } else {
                l1 = i;
                r1 = r;
                l2 = l;
                r2 = j;
            }
            if (l2 === r2) {
                l = l1;
                r = r1;
            } else {
                stackL.push(l1);
                stackR.push(r1);
                l = l2;
                r = r2;
            }
        } else {
            for (let i = l + 1; i <= r; i++) {
                for (let j = i; j > l && less(j, j - 1); j--) {
                    swap(j, j - 1);
                }
            }
            if (stackL.length === 0) break;
            l = stackL.pop() as number;
            r = stackR.pop() as number;
        }
    }
}

/** An int32-backed growable vector — the common index/id column. */
export const intVec = (capacity = 0): GrowVec<Int32Array> =>
    new GrowVec((n) => new Int32Array(n), capacity);

/** An f32-backed growable vector — a SoA scalar column. */
export const floatVec = (capacity = 0): GrowVec<Float32Array> =>
    new GrowVec((n) => new Float32Array(n), capacity);
