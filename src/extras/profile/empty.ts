import { type FrameTimings, TIMING_HISTORY, TIMING_SMOOTHING } from "./frames";

/** Empty frames behind a pending readback. Only the retained history needs individual timestamps;
 * older zeros compose as an EMA decay product, even with irregular frame intervals.
 */
export class EmptyFrames {
    private readonly _times = new Float64Array(TIMING_HISTORY);
    private readonly _clock = new Float64Array(2);
    private _head = 0;
    private _count = 0;
    private _discarded = 0;
    private _decay = 1;
    private _previous = 0;

    reset(time: number): void {
        this._head = this._count = this._discarded = 0;
        this._decay = 1;
        this._previous = time;
    }

    push(clock: Readonly<Float64Array>): void {
        if (this._count === TIMING_HISTORY) {
            const time = this._times[this._head];
            this._decay *= 1 - Math.min(1, Math.max(0, (time - this._previous) / TIMING_SMOOTHING));
            this._previous = time;
            this._discarded++;
            this._head = (this._head + 1) % TIMING_HISTORY;
            this._count--;
        }
        this._times[(this._head + this._count) % TIMING_HISTORY] = clock[0];
        this._count++;
    }

    drain(frames: FrameTimings): void {
        if (this._discarded > 0) {
            this._clock[0] = this._previous;
            this._clock[1] = this._decay;
            frames.foldZeros(this._clock, this._discarded);
        }
        for (let i = 0; i < this._count; i++) {
            this._clock[0] = this._times[(this._head + i) % TIMING_HISTORY];
            frames.finish(this._clock);
        }
    }
}
