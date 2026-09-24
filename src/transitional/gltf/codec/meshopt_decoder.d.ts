// Types the vendored meshoptimizer glTF decoder (meshopt_decoder.js). Transcribed from upstream's shipped
// declaration rather than copied, so it stays formatted like the draco / basis declarations beside it; the
// colocated file resolves the `.js` import for any consumer, including a workspace that checks shallot's
// source through a symlink and whose tsconfig lacks `allowJs` (orrstead). Only `ready` and `decodeGltfBuffer`
// are used (meshopt.ts) — the rest of the surface is declared for fidelity to the artifact.
export const MeshoptDecoder: {
    supported: boolean;
    ready: Promise<void>;
    decodeVertexBuffer: (
        target: Uint8Array,
        count: number,
        size: number,
        source: Uint8Array,
        filter?: string,
    ) => void;
    decodeIndexBuffer: (
        target: Uint8Array,
        count: number,
        size: number,
        source: Uint8Array,
    ) => void;
    decodeIndexSequence: (
        target: Uint8Array,
        count: number,
        size: number,
        source: Uint8Array,
    ) => void;
    decodeGltfBuffer: (
        target: Uint8Array,
        count: number,
        size: number,
        source: Uint8Array,
        mode: string,
        filter?: string,
    ) => void;
    useWorkers: (count: number) => void;
    decodeGltfBufferAsync: (
        count: number,
        size: number,
        source: Uint8Array,
        mode: string,
        filter?: string,
    ) => Promise<Uint8Array>;
};
