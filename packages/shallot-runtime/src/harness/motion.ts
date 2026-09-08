/** Mean absolute difference between two equally-sized numeric frame samples.
 * Returns the reading and throws only on unusable input, so a retrying caller — a Playwright
 * `expect.poll`, which propagates a callback throw instead of sampling again — can wait on the
 * value crossing its threshold. {@link assertMotion} is the one-shot form. */
export function frameDifference(before: ArrayLike<number>, after: ArrayLike<number>): number {
    if (before.length !== after.length || before.length === 0) {
        throw new Error(
            `motion samples must have the same non-zero length (${before.length} != ${after.length})`,
        );
    }
    let sum = 0;
    for (let i = 0; i < before.length; i++) sum += Math.abs(before[i] - after[i]);
    return sum / before.length;
}

/** Assert that two equally-sized numeric frame samples differ by more than `threshold`.
 * Device-free: callers own capture and timing; this helper owns the one motion predicate. */
export function assertMotion(
    before: ArrayLike<number>,
    after: ArrayLike<number>,
    threshold: number,
): number {
    const difference = frameDifference(before, after);
    if (difference <= threshold) {
        throw new Error(
            `samples are parked (mean absolute difference ${difference.toFixed(2)}, need > ${threshold})`,
        );
    }
    return difference;
}
