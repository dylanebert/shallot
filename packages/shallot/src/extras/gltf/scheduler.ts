// A priority-scheduled pool of N work slots — the pure, transport-free core of the decode pool (pool.ts
// wires its `run` to post a DecodeRequest to a worker). Ordering + idle-slot selection + backpressure live
// here so they're unit-testable with no Worker: `submit` queues a task at a priority, `drain` dispatches the
// highest-priority waiter to a free slot, capping in-flight at N (excess waits its turn — the backpressure).
// Reference: three.js `WorkerPool`'s per-slot dispatch + queue, made generic and priority-ordered.

interface Waiter<T, R> {
    task: T;
    priority: number;
    seq: number;
    resolve: (r: R) => void;
    reject: (e: unknown) => void;
}

export class Scheduler<T, R> {
    private readonly _run: (slot: number, task: T) => Promise<R>;
    // free slot indices (0..slots-1); a slot is one concurrent `run`. The transport maps a slot to a worker.
    private readonly _free: number[];
    private readonly _queue: Waiter<T, R>[] = [];
    private _seq = 0;

    constructor(opts: { slots: number; run: (slot: number, task: T) => Promise<R> }) {
        this._run = opts.run;
        this._free = Array.from({ length: opts.slots }, (_, i) => i);
    }

    /**
     * queue a task at a priority (higher dispatches first; ties keep FIFO) and resolve with its `run` result.
     * In-flight never exceeds the slot count: excess waits until a slot frees (the backpressure).
     */
    submit(task: T, priority = 0): Promise<R> {
        return new Promise<R>((resolve, reject) => {
            this._queue.push({ task, priority, seq: this._seq++, resolve, reject });
            this.drain();
        });
    }

    /**
     * drop every queued (not-yet-dispatched) task, rejecting each with `reason`: the cancel-on-dispose seam
     * (pool.ts's `abortDecodes`). In-flight runs are untouched: they finish and free their slot normally, so a
     * decode already on a worker still lands in the asset cache (a late result is safe, the cache is
     * idempotent). Submitting again after this works as usual.
     */
    abort(reason: unknown): void {
        const dropped = this._queue.splice(0, this._queue.length);
        for (const w of dropped) w.reject(reason);
    }

    // dispatch waiters to free slots, highest-priority first, until one runs out. Each settle returns its slot
    // and re-drains, so a queued task starts the moment a slot frees — a reject frees the slot too (no deadlock).
    private drain(): void {
        while (this._free.length > 0 && this._queue.length > 0) {
            const w = this.take();
            const slot = this._free.pop() as number;
            this._run(slot, w.task)
                .then(w.resolve, w.reject)
                .finally(() => {
                    this._free.push(slot);
                    this.drain();
                });
        }
    }

    // remove + return the highest-priority waiter (FIFO within a priority). A linear scan — the queue is at
    // most one scene load's pending-decode count, not a hot path.
    private take(): Waiter<T, R> {
        let best = 0;
        for (let i = 1; i < this._queue.length; i++) {
            const w = this._queue[i];
            const b = this._queue[best];
            if (w.priority > b.priority || (w.priority === b.priority && w.seq < b.seq)) best = i;
        }
        return this._queue.splice(best, 1)[0];
    }
}
