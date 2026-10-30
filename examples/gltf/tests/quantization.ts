export function float32Ulp(value: number): number {
    const magnitude = Math.abs(Math.fround(value));
    return magnitude < 2 ** -126 ? 2 ** -149 : 2 ** (Math.floor(Math.log2(magnitude)) - 23);
}

export function decodeQuantized(code: number, offset: number, extent: number) {
    const normalized = Math.fround(code / 65535);
    const product = Math.fround(normalized * extent);
    const value = Math.fround(offset + product);
    // Nearest unorm16 contributes half an axis step. The shader rounds division,
    // multiplication and addition to float32. Division error is magnified by the
    // extent; each operation contributes at most half its intermediate's ulp.
    const bound =
        extent / 131070 +
        (Math.abs(extent) * float32Ulp(normalized)) / 2 +
        float32Ulp(product) / 2 +
        float32Ulp(value) / 2;
    return { value, bound };
}
