export function formatValue(value: number): string {
    const text = String(value);
    return text.length > 6 ? value.toExponential().replace("e+", "e") : text;
}
