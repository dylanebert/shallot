import { sampleDiagnosticTime } from "./time";

export { sampleDiagnosticTime } from "./time";

/** Slash-separated diagnostic identity; components must be nonempty. */
export class DiagnosticPath {
    readonly path: string;
    constructor(path: string) {
        this.path = path;
        if (!path || path.startsWith("/") || path.endsWith("/") || path.includes("//"))
            throw new Error(`Invalid diagnostic path: ${path}`);
    }
    toString(): string {
        return this.path;
    }
}

/** Measurement time is in monotonic seconds, value in the diagnostic's units. */
export interface DiagnosticMeasurement {
    time: number;
    value: number;
}

/** Bounded history and time-weighted exponential smoothing of one measurement.
 * Defaults to 120 samples and a smoothing factor of 2/21 seconds. Each sample moves the smoothed
 * value toward it by clamp(elapsedSeconds / smoothingFactor, 0, 1); zero disables smoothing.
 * A full history overwrites its oldest record. Missing samples leave history and smoothing unchanged.
 */
export class Diagnostic {
    readonly path: DiagnosticPath;
    readonly suffix: string;
    isEnabled = true;
    private readonly _history: DiagnosticMeasurement[];
    private _cursor = 0;
    private _length = 0;
    private _sum = 0;
    private _ema = 0;
    private readonly _smoothingFactor: number;
    private readonly _sample = new Float64Array(2);
    constructor(
        path: DiagnosticPath | string,
        options: { suffix?: string; maxHistoryLength?: number; smoothingFactor?: number } = {},
    ) {
        this.path = typeof path === "string" ? new DiagnosticPath(path) : path;
        this.suffix = options.suffix ?? "";
        const capacity = options.maxHistoryLength ?? 120;
        if (!Number.isInteger(capacity) || capacity < 1)
            throw new Error("Diagnostic history length must be positive");
        this._smoothingFactor = options.smoothingFactor ?? 2 / 21;
        if (this._smoothingFactor < 0 || !Number.isFinite(this._smoothingFactor))
            throw new Error("Diagnostic smoothing factor must be finite and nonnegative");
        this._history = Array.from({ length: capacity }, () => ({ time: 0, value: 0 }));
    }
    addMeasurement(time: number, value: number): void {
        this._sample[0] = time;
        this._sample[1] = value;
        this.addMeasurementBuffer(this._sample);
    }
    /** @internal Copies [monotonic seconds, value]; input may be reused after return. */
    addMeasurementBuffer(sample: Readonly<Float64Array>): void {
        if (!this.isEnabled) return;
        const time = sample[0];
        const value = sample[1];
        const previous = this.measurement();
        if (!Number.isNaN(value)) {
            const alpha = previous
                ? this._smoothingFactor === 0
                    ? 1
                    : Math.min(1, Math.max(0, (time - previous.time) / this._smoothingFactor))
                : 1;
            this._ema += alpha * (value - this._ema);
        }
        const slot = this._history[this._cursor];
        if (this._length === this._history.length && Number.isFinite(slot.value))
            this._sum -= slot.value;
        if (Number.isFinite(value)) this._sum += value;
        slot.time = time;
        slot.value = value;
        this._cursor = (this._cursor + 1) % this._history.length;
        this._length = Math.min(this._length + 1, this._history.length);
    }
    /** Borrowed until the ring overwrites it. */
    measurement(): Readonly<DiagnosticMeasurement> | undefined {
        return this._length
            ? this._history[(this._cursor + this._history.length - 1) % this._history.length]
            : undefined;
    }
    value(): number | undefined {
        return this.measurement()?.value;
    }
    average(): number | undefined {
        return this._length ? this._sum / this._length : undefined;
    }
    smoothed(): number | undefined {
        return this._length ? this._ema : undefined;
    }
    get historyLength(): number {
        return this._length;
    }
    /** Oldest first; records are borrowed until overwritten. */
    *measurements(): IterableIterator<Readonly<DiagnosticMeasurement>> {
        for (let i = 0; i < this._length; i++)
            yield this._history[
                (this._cursor - this._length + i + this._history.length) % this._history.length
            ];
    }
    clearHistory(): void {
        this._cursor = this._length = this._sum = this._ema = 0;
    }
}

/** World-local diagnostics, independent of any recorder or overlay. */
export class DiagnosticsStore {
    static create(): DiagnosticsStore {
        return new DiagnosticsStore();
    }
    private readonly _diagnostics = new Map<string, Diagnostic>();
    private readonly _sample = new Float64Array(2);
    add(diagnostic: Diagnostic): void {
        this._diagnostics.set(diagnostic.path.path, diagnostic);
    }
    get(path: DiagnosticPath | string): Diagnostic | undefined {
        return this._diagnostics.get(String(path));
    }
    remove(path: DiagnosticPath | string): void {
        this._diagnostics.delete(String(path));
    }
    iter(): IterableIterator<Diagnostic> {
        return this._diagnostics.values();
    }
    /** Register on first observation. Time is monotonic seconds. */
    record(path: DiagnosticPath | string, value: number, time?: number, suffix = ""): void {
        if (time === undefined) sampleDiagnosticTime(this._sample);
        else this._sample[0] = time;
        this._sample[1] = value;
        const diagnostic = this.get(path) ?? this.register(path, suffix);
        diagnostic.addMeasurementBuffer(this._sample);
    }
    private register(path: DiagnosticPath | string, suffix: string): Diagnostic {
        const diagnostic = new Diagnostic(path, { suffix });
        this.add(diagnostic);
        return diagnostic;
    }
}
