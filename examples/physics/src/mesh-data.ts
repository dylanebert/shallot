/** Decode base64 little-endian f32 values. */
export function decodeFloat32(text: string): Float32Array {
    const bytes = Uint8Array.from(atob(text), (char) => char.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    return Float32Array.from({ length: bytes.length / 4 }, (_, i) => view.getFloat32(i * 4, true));
}

type EncodedMesh = { vertices: string; indices: string; materialIndices: string };

/** Decode little-endian f32 vertices, u32 indices and u8 material indices. */
export function decodeMesh(data: EncodedMesh) {
    const bytes = (text: string) => Uint8Array.from(atob(text), (char) => char.charCodeAt(0));
    const indices = bytes(data.indices);
    const words = new DataView(indices.buffer);
    return {
        vertices: decodeFloat32(data.vertices),
        indices: Uint32Array.from({ length: indices.length / 4 }, (_, i) =>
            words.getUint32(i * 4, true),
        ),
        materialIndices: bytes(data.materialIndices),
    };
}
