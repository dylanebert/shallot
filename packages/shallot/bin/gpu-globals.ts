// Install the fixed WebGPU enum bit-flag constants as globals. The engine barrel references them at module
// top-level (sear's `GPUShaderStage`, the slab/render buffer-usage masks), so importing it under the plain
// `bun` CLI — the build-time feature scan in `features.ts` — needs them defined (a browser has them for
// free). The values are spec-fixed, so this is a self-contained shim: the shipped CLI never reaches for the
// dev-only `bun-webgpu` test package (which carries native wgpu bindings no browser user wants installed).
const FLAGS: Record<string, Record<string, number>> = {
    GPUBufferUsage: {
        MAP_READ: 0x0001,
        MAP_WRITE: 0x0002,
        COPY_SRC: 0x0004,
        COPY_DST: 0x0008,
        INDEX: 0x0010,
        VERTEX: 0x0020,
        UNIFORM: 0x0040,
        STORAGE: 0x0080,
        INDIRECT: 0x0100,
        QUERY_RESOLVE: 0x0200,
    },
    GPUTextureUsage: {
        COPY_SRC: 0x01,
        COPY_DST: 0x02,
        TEXTURE_BINDING: 0x04,
        STORAGE_BINDING: 0x08,
        RENDER_ATTACHMENT: 0x10,
    },
    GPUShaderStage: { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 },
    GPUMapMode: { READ: 0x1, WRITE: 0x2 },
    GPUColorWrite: { RED: 0x1, GREEN: 0x2, BLUE: 0x4, ALPHA: 0x8, ALL: 0xf },
};

/** install the WebGPU enum constants as globals, idempotently (a no-op where they already exist). */
export function installGpuGlobals(): void {
    for (const [name, flags] of Object.entries(FLAGS)) {
        if (!(name in globalThis)) (globalThis as Record<string, unknown>)[name] = flags;
    }
}
