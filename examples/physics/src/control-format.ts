/** Read-only fields keep their authored decimal or significant-digit format without a slider's width limit. */
export function formatReadout(value: number | string, decimals?: number, precision = 3): string {
    if (typeof value === "string") return value;
    if (decimals !== undefined) return value.toFixed(decimals);
    const rounded = Number(value.toPrecision(precision));
    if (rounded === 0) return Object.is(rounded, -0) ? "-0" : "0";
    const exponent = Math.floor(Math.log10(Math.abs(rounded)));
    return exponent < -4 || exponent >= precision
        ? rounded.toExponential().replace("e+", "e")
        : String(rounded);
}

export function formatValue(value: number, decimals?: number): string {
    const text = decimals === undefined ? String(value) : value.toFixed(decimals);
    return text.length > 6 ? value.toExponential().replace("e+", "e") : text;
}
