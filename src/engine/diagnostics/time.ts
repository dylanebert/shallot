/** Write monotonic seconds to the first cell of a reusable [time, value] measurement. */
export function sampleDiagnosticTime(measurement: Float64Array): void {
    measurement[0] = performance.now() / 1000;
}
