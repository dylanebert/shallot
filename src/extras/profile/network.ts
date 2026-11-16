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
            const name = path.slice(path.lastIndexOf("/") + 1);
            const rateUnit =
                diagnostic.suffix === "B"
                    ? "B/s"
                    : name.startsWith("messages_")
                      ? "messages/s"
                      : name.endsWith("replayed_ticks")
                        ? "ticks/s"
                        : undefined;
            let reading = diagnostic.smoothed();
            if (rateUnit) {
                const first = diagnostic.measurements().next().value;
                reading =
                    first && sample && sample.time > first.time && sample.value >= first.value
                        ? (sample.value - first.value) / (sample.time - first.time)
                        : undefined;
            }
            const unit = rateUnit ?? diagnostic.suffix;
            const value =
                !sample || reading === undefined || now - seen.changed > 1000
                    ? "no data"
                    : `${Number(reading.toFixed(2))}${unit ? " " + unit : ""}${stale ? " · stale" : ""}`;
            rows.push([path.slice(8).replaceAll("_", " "), value]);
        }
        for (const diagnostic of this._seen.keys())
            if (!live.has(diagnostic)) this._seen.delete(diagnostic);
        return rows;
    }
}
