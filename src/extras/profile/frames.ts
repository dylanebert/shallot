import { Diagnostic, type DiagnosticsStore } from "../../engine";

export const TIMING_FLOOR_MS = 0.005;
export const TIMING_HISTORY = 120;
export const TIMING_SMOOTHING = 2 / 21;

/** Sum repeated spans, then sample every known label once per presented frame, including zeros.
 * Unlike Bevy render diagnostic `internal.rs::sync_diagnostics` (only finished spans), absent labels
 * must decay because the HUD adds rows across scenes. Frame time and span smoothing share the
 * unscaled presented-frame clock, not the readback arrival clock.
 */
export class FrameTimings {
    private readonly _rows = new Map<
        string,
        { diagnostic: Diagnostic; ms: number; idle: number }
    >();
    private readonly _measurement = new Float64Array(2);

    private readonly _store: DiagnosticsStore;
    private readonly _retire?: (path: string) => void;

    constructor(store: DiagnosticsStore, retire?: (path: string) => void) {
        this._store = store;
        this._retire = retire;
    }

    add(path: string, ms: number): void {
        let row = this._rows.get(path);
        if (!row) {
            const diagnostic =
                this._store.get(path) ??
                new Diagnostic(path, {
                    suffix: "ms",
                    maxHistoryLength: TIMING_HISTORY,
                    smoothingFactor: TIMING_SMOOTHING,
                });
            this._store.add(diagnostic);
            row = { diagnostic, ms: 0, idle: 0 };
            this._rows.set(path, row);
        }
        row.ms += ms;
        row.idle = -1;
    }

    foldZeros(clock: Readonly<Float64Array>, count: number): void {
        for (const row of this._rows.values()) {
            row.diagnostic.foldZeroMeasurements(clock);
            row.idle += count;
        }
    }

    finish(clock: Readonly<Float64Array>): void {
        const measurement = this._measurement;
        measurement[0] = clock[0];
        for (const row of this._rows.values()) {
            measurement[1] = row.ms;
            row.diagnostic.addMeasurementBuffer(measurement);
            row.ms = 0;
            row.idle = row.idle < 0 ? 0 : row.idle + 1;
            if (row.idle >= TIMING_HISTORY && row.diagnostic.smoothed()! < TIMING_FLOOR_MS) {
                const path = row.diagnostic.path.path;
                this._rows.delete(path);
                this._store.remove(path);
                this._retire?.(path);
            }
        }
    }
}
