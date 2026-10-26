import { Diagnostic, type DiagnosticsStore } from "../../engine";

/** Sum repeated spans, then sample every known label once per presented frame, including zeros.
 * Unlike Bevy render diagnostic `internal.rs::sync_diagnostics` (only finished spans), absent labels
 * must decay because the HUD adds rows across scenes. Frame time and span smoothing share the
 * unscaled presented-frame clock, not the readback arrival clock.
 */
export class FrameTimings {
    private readonly _rows = new Map<string, { diagnostic: Diagnostic; ms: number }>();
    private readonly _measurement = new Float64Array(2);

    private readonly _store: DiagnosticsStore;

    constructor(store: DiagnosticsStore) {
        this._store = store;
    }

    add(path: string, ms: number): void {
        let row = this._rows.get(path);
        if (!row) {
            const diagnostic = this._store.get(path) ?? new Diagnostic(path, { suffix: "ms" });
            this._store.add(diagnostic);
            row = { diagnostic, ms: 0 };
            this._rows.set(path, row);
        }
        row.ms += ms;
    }

    finish(clock: Readonly<Float64Array>): void {
        const measurement = this._measurement;
        measurement[0] = clock[0];
        for (const row of this._rows.values()) {
            measurement[1] = row.ms;
            row.diagnostic.addMeasurementBuffer(measurement);
            row.ms = 0;
        }
    }
}
