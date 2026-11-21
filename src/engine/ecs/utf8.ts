const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

/** @internal Encode without replacing unpaired UTF-16 surrogates. */
export function encodeText(text: string, byteCapacity: number): Uint8Array {
    if (typeof text !== "string") throw new Error("Expected text");
    if (text.length > byteCapacity) throw new Error("Text exceeds byte capacity");
    if (/[\uD800-\uDFFF]/u.test(text)) throw new Error("Invalid UTF-8 text");
    const bytes = encoder.encode(text);
    if (bytes.length > byteCapacity) throw new Error("Text exceeds byte capacity");
    return bytes;
}

/** @internal Decode exactly the length declared by the little-endian u16 prefix. */
export function readText(view: DataView, offset: number, byteCapacity: number): string {
    const size = view.getUint16(offset, true);
    if (size > byteCapacity || offset + 2 + size > view.byteLength)
        throw new Error("Invalid text size");
    return decoder.decode(new Uint8Array(view.buffer, view.byteOffset + offset + 2, size));
}

/** @internal Write a validated UTF-8 payload and its little-endian u16 length. */
export function writeText(view: DataView, offset: number, bytes: Uint8Array): void {
    view.setUint16(offset, bytes.length, true);
    new Uint8Array(view.buffer, view.byteOffset + offset + 2, bytes.length).set(bytes);
}
