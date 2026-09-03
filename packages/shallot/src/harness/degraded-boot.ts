const DEGRADED_BOOT_HINT =
    /\b(?:wgsl|shader compilation|pipeline.*invalid|destroyed|validation error|device.*lost|uncaptured|GPUValidationError|GPUInternalError|exceeds the max|crashed|Missing plugin dependency|is not registered|names no clip)\b/i;

/** true when a browser console message identifies a failed or degraded engine boot. */
export function isDegradedBootMessage(text: string): boolean {
    return DEGRADED_BOOT_HINT.test(text);
}

/** @internal the console signature matcher used by `shallot verify`. */
export const degradedBootHint = DEGRADED_BOOT_HINT;
