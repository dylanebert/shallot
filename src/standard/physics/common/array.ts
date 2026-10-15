// Numeric b3Array(T) from Box3D container.h. Growth replaces data; callers must not retain its view.

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
 * Swap-remove from a plain array (b3Array_RemoveSwap). Returns the moved element's old index
 * (the new length), or NULL_INDEX if `index` was the last element, so callers can fix back-indices.
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

// qsort's subfile stacks, kept at their high-water capacity so a sort allocates nothing.
const stackL: number[] = [];
const stackR: number[] = [];

function sort3(
    less: (i: number, j: number) => boolean,
    swap: (i: number, j: number) => void,
    a1: number,
    a2: number,
    a3: number,
): void {
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
}

/**
 * In-place index sort (Box3D QSORT, qsort.h, Alexey Tourbin). Comparison and swap sequence
 * determines equal-key order; keep it aligned with the source.
 * Not reentrant: `less` and `swap` must not sort.
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

    let top = 0;
    let l = 0;
    let r = n - 1;

    while (true) {
        if (r - l + 1 >= Thresh) {
            const m = l + ((r - l) >> 1);
            sort3(less, swap, l + 1, m, r);
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
                stackL[top] = l1;
                stackR[top] = r1;
                top++;
                l = l2;
                r = r2;
            }
        } else {
            for (let i = l + 1; i <= r; i++) {
                for (let j = i; j > l && less(j, j - 1); j--) {
                    swap(j, j - 1);
                }
            }
            if (top === 0) break;
            top--;
            l = stackL[top];
            r = stackR[top];
        }
    }
}

export const intVec = (capacity = 0): GrowVec<Int32Array> =>
    new GrowVec((n) => new Int32Array(n), capacity);
