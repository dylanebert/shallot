/** @internal Wire encoders specify the one accepted representation of decoded data. */
export function requireCanonical(bytes: Uint8Array, encoded: Uint8Array): void {
    if (bytes.length === encoded.length) {
        let i = 0;
        while (i < bytes.length && bytes[i] === encoded[i]) i++;
        if (i === bytes.length) return;
    }
    throw new Error("Non-canonical multiplayer message");
}

/** @internal Float64 NaN has one wire representation, regardless of JS payload bits. */
export function writeFloat64(view: DataView, offset: number, value: number): void {
    if (Number.isNaN(value)) {
        view.setUint32(offset, 0, true);
        view.setUint32(offset + 4, 0x7ff80000, true);
    } else view.setFloat64(offset, value, true);
}
