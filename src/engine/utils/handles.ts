const INDEX_COUNT = 2 ** 20;
const MAX_GENERATION = 2 ** 11 - 1;
const MAX_HANDLE = 2 ** 31 - 1;

/** Masks a {@link HandleAllocator} handle to its dense index, as GPU code reads it; a stale
 * handle still needs its generation compared. @internal */
export const HANDLE_INDEX_MASK = INDEX_COUNT - 1;

/** Numeric handles with a 20-bit dense index and an 11-bit generation, fitting nonnegative i32.
 * Released indices recycle with a new generation; exhausted generations retire, never wrap.
 * At most 2^20 indices can be issued. Snapshots restore the captured allocation timeline. */
export class HandleAllocator {
    private _generations: number[] = [];
    private _live: boolean[] = [];
    private _free: number[] = [];

    allocate(): number {
        let index = this._free.pop();
        if (index === undefined) {
            index = this._generations.length;
            if (index === INDEX_COUNT) throw new Error("handle index space exhausted");
            this._generations.push(0);
        }
        this._live[index] = true;
        return this._generations[index] * INDEX_COUNT + index;
    }

    /** The dense index only when the handle is live and its generation matches. */
    index(handle: number): number | undefined {
        if (!Number.isInteger(handle) || handle < 0 || handle > MAX_HANDLE) return undefined;
        const index = handle % INDEX_COUNT;
        return this._live[index] && this._generations[index] === Math.floor(handle / INDEX_COUNT)
            ? index
            : undefined;
    }

    release(handle: number): boolean {
        const index = this.index(handle);
        if (index === undefined) return false;
        this._live[index] = false;
        if (this._generations[index] < MAX_GENERATION) {
            this._generations[index]++;
            this._free.push(index);
        }
        return true;
    }

    /** Invalidate all live handles without resetting their generations. */
    clear(): void {
        for (let index = 0; index < this._generations.length; index++) {
            if (this._live[index]) this.release(this._generations[index] * INDEX_COUNT + index);
        }
    }

    snapshot(): { generations: number[]; live: boolean[]; free: number[] } {
        return {
            generations: this._generations.slice(),
            live: this._live.slice(),
            free: this._free.slice(),
        };
    }

    restore(state: ReturnType<HandleAllocator["snapshot"]>): void {
        this._generations = state.generations.slice();
        this._live = state.live.slice();
        this._free = state.free.slice();
    }
}
