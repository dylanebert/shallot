import type { Diagnostic, DiagnosticsStore } from "../../engine";

/** @internal Observe sample progress without assuming the producer's clock epoch. */
export class NetworkRows {
    private readonly _seen = new Map<Diagnostic, { time: number | undefined; changed: number }>();

    collect(store: DiagnosticsStore, now: number): [string, string][] {
        const rows: [string, string][] = [];
        const live = new Set<Diagnostic>();
        for (const diagnostic of store.iter()) {
            const path = diagnostic.path.path;
            if (!path.startsWith("network/")) continue;
            live.add(diagnostic);
            const sample = diagnostic.measurement();
            let seen = this._seen.get(diagnostic);
            if (!seen || seen.time !== sample?.time) {
                seen = { time: sample?.time, changed: now };
                this._seen.set(diagnostic, seen);
            }
            const stale = path.endsWith("/time_since_receive") && (sample?.value ?? 0) > 1000;
            const value =
                !sample || now - seen.changed > 1000
                    ? "no data"
                    : `${Number(sample.value.toFixed(2))}${diagnostic.suffix ? " " + diagnostic.suffix : ""}${stale ? " · stale" : ""}`;
            rows.push([path.slice(8), value]);
        }
        for (const diagnostic of this._seen.keys())
            if (!live.has(diagnostic)) this._seen.delete(diagnostic);
        return rows;
    }
}
