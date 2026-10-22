/** User Timing mark written when the build hands the World to the game: the loading screen is gone. */
export const HANDOFF_MARK = "shallot:handoff";

/** User Timing span of `runApp`'s mount of `config.ui`. */
export const UI_MOUNT_MEASURE = "shallot:ui-mount";

/** User Timing span prefix for a system's first run: `${prefix}${system}:setup` and `${prefix}${system}:first-update`. */
export const SYSTEM_MEASURE_PREFIX = "shallot:system:";

/** Write a User Timing mark. User Timing is optional and never changes a caller's result. */
export function userTimingMark(name: string): void {
    try {
        if (typeof performance?.mark === "function") performance.mark(name);
    } catch {}
}

/** Run `operation` between a User Timing mark and a measure of the same name. */
export function userTimingSpan<T>(name: string, operation: () => T): T {
    userTimingMark(name);
    try {
        return operation();
    } finally {
        try {
            if (typeof performance?.measure === "function") performance.measure(name, name);
        } catch {}
    }
}
