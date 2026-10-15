/** Clone definition data, rounding numeric fields at the f32 boundary; userData remains opaque. */
export function froundConfig<T>(cfg: T): T {
    if (typeof cfg === "number") return Math.fround(cfg) as T;
    if (cfg === null || typeof cfg !== "object") return cfg;
    if (Array.isArray(cfg)) return cfg.map((v) => froundConfig(v)) as T;
    const source = cfg as Record<string, unknown>;
    // Keep plain records' property layout for the allocation-free marshal path.
    let plain =
        Object.getPrototypeOf(source) === Object.prototype &&
        Object.getOwnPropertySymbols(source).length === 0;
    const descriptors = Object.getOwnPropertyDescriptors(source);
    for (const key in descriptors) {
        const descriptor = descriptors[key];
        if (key === "__proto__" || !descriptor.enumerable || !("value" in descriptor))
            plain = false;
    }
    const out: Record<string, unknown> = plain ? { ...source } : {};
    for (const key in source) {
        out[key] = key === "userData" ? source[key] : froundConfig(source[key]);
    }
    return out as T;
}
