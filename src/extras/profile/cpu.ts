/** Sum outermost recorded timings without allocating on the frame path. */
export function cpuTotal(timings: ReadonlyMap<string, number>): number {
    let total = 0;
    for (const [name, ms] of timings) {
        let part = false;
        for (const parent of timings.keys()) {
            if (
                name.length > parent.length &&
                name.startsWith(parent) &&
                name.charCodeAt(parent.length) === 47
            ) {
                part = true;
                break;
            }
        }
        if (!part) total += ms;
    }
    return total;
}
