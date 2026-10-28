type EncodedMesh = { vertices: string; indices: string; materialIndices: string };

/** Decode little-endian f32 vertices, u32 indices and u8 material indices. */
export function decodeMesh(data: EncodedMesh) {
    const bytes = (text: string) => Uint8Array.from(atob(text), (char) => char.charCodeAt(0));
    const vertices = bytes(data.vertices);
    const indices = bytes(data.indices);
    const floats = new DataView(vertices.buffer);
    const words = new DataView(indices.buffer);
    return {
        vertices: Float32Array.from({ length: vertices.length / 4 }, (_, i) =>
            floats.getFloat32(i * 4, true),
        ),
        indices: Uint32Array.from({ length: indices.length / 4 }, (_, i) =>
            words.getUint32(i * 4, true),
        ),
        materialIndices: bytes(data.materialIndices),
    };
}
