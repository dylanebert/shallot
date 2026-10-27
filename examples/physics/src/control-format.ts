export function formatValue(value: number, decimals?: number): string {
    const text = decimals === undefined ? String(value) : value.toFixed(decimals);
    return text.length > 6 ? value.toExponential().replace("e+", "e") : text;
}
