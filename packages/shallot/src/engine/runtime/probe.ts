const COPY_ALIGNMENT = 4;
const ROW_ALIGNMENT = 256;

/** encode the work whose resource boundary a one-shot probe captures. */
export type ProbeEncode = (encoder: GPUCommandEncoder) => void;

/** an owned buffer snapshot. Its bytes are never reused or mutated by the probe. */
export interface BufferProbe {
    readonly bytes: ArrayBuffer;
    readonly offset: number;
    readonly size: number;
}

/** one tightly-packed texture snapshot. Rows carry no WebGPU copy padding. */
export interface TextureProbe {
    readonly bytes: ArrayBuffer;
    readonly format: GPUTextureFormat;
    readonly aspect: GPUTextureAspect;
    readonly width: number;
    readonly height: number;
    readonly depthOrArrayLayers: number;
    readonly bytesPerRow: number;
}

/** buffer range and trigger captured by {@link probeBuffer}. */
export interface BufferProbeOptions {
    /** byte offset in `source`; an unaligned range is copied through an aligned enclosing range. */
    offset?: number;
    /** bytes to return. Defaults to the rest of `source`. */
    size?: number;
    /** records the work under test before the pass-adjacent copy. */
    encode?: ProbeEncode;
    /** diagnostic label for the command encoder and staging buffer. */
    label?: string;
}

/** texture region and trigger captured by {@link probeTexture}. */
export interface TextureProbeOptions {
    /** texel origin within the selected mip. Defaults to `[0, 0, 0]`. */
    origin?: GPUOrigin3D;
    /** texel extent to copy. Defaults to the rest of the selected mip. */
    size?: GPUExtent3D;
    /** source mip level. Defaults to 0. */
    mipLevel?: number;
    /** depth/stencil aspect to copy; combined formats require an explicit aspect. */
    aspect?: GPUTextureAspect;
    /** records the work under test before the pass-adjacent copy. */
    encode?: ProbeEncode;
    /** diagnostic label for the command encoder and staging buffer. */
    label?: string;
}

const alignDown = (n: number, alignment: number) => Math.floor(n / alignment) * alignment;
const alignUp = (n: number, alignment: number) => Math.ceil(n / alignment) * alignment;

function coordinate(value: GPUOrigin3D | undefined): Required<GPUOrigin3DDict> {
    if (!value) return { x: 0, y: 0, z: 0 };
    if (Symbol.iterator in Object(value)) {
        const [x = 0, y = 0, z = 0] = value as Iterable<number>;
        return { x, y, z };
    }
    const dict = value as GPUOrigin3DDict;
    return { x: dict.x ?? 0, y: dict.y ?? 0, z: dict.z ?? 0 };
}

function extent(value: GPUExtent3D): Required<GPUExtent3DDict> {
    if (Symbol.iterator in Object(value)) {
        const [width, height = 1, depthOrArrayLayers = 1] = value as Iterable<number>;
        return { width, height, depthOrArrayLayers };
    }
    const dict = value as GPUExtent3DDict;
    return {
        width: dict.width,
        height: dict.height ?? 1,
        depthOrArrayLayers: dict.depthOrArrayLayers ?? 1,
    };
}

function colorTexelBytes(format: GPUTextureFormat): number | undefined {
    switch (format) {
        case "r8unorm":
        case "r8snorm":
        case "r8uint":
        case "r8sint":
            return 1;
        case "r16unorm":
        case "r16snorm":
        case "r16uint":
        case "r16sint":
        case "r16float":
        case "rg8unorm":
        case "rg8snorm":
        case "rg8uint":
        case "rg8sint":
            return 2;
        case "r32uint":
        case "r32sint":
        case "r32float":
        case "rg16unorm":
        case "rg16snorm":
        case "rg16uint":
        case "rg16sint":
        case "rg16float":
        case "rgba8unorm":
        case "rgba8unorm-srgb":
        case "rgba8snorm":
        case "rgba8uint":
        case "rgba8sint":
        case "bgra8unorm":
        case "bgra8unorm-srgb":
        case "rgb10a2uint":
        case "rgb10a2unorm":
        case "rg11b10ufloat":
        case "rgb9e5ufloat":
            return 4;
        case "rg32uint":
        case "rg32sint":
        case "rg32float":
        case "rgba16unorm":
        case "rgba16snorm":
        case "rgba16uint":
        case "rgba16sint":
        case "rgba16float":
            return 8;
        case "rgba32uint":
        case "rgba32sint":
        case "rgba32float":
            return 16;
        default:
            return undefined;
    }
}

function copyAspect(
    format: GPUTextureFormat,
    requested: GPUTextureAspect | undefined,
): { aspect: GPUTextureAspect; bytesPerTexel: number } {
    const colorBytes = colorTexelBytes(format);
    if (colorBytes !== undefined) {
        if (requested && requested !== "all") {
            throw new Error(`probeTexture: color format ${format} requires aspect "all"`);
        }
        return { aspect: "all", bytesPerTexel: colorBytes };
    }
    if (format === "depth16unorm" || format === "depth32float") {
        if (requested && requested !== "depth-only") {
            throw new Error(`probeTexture: ${format} requires aspect "depth-only"`);
        }
        return { aspect: "depth-only", bytesPerTexel: format === "depth16unorm" ? 2 : 4 };
    }
    if (format === "stencil8") {
        if (requested && requested !== "stencil-only") {
            throw new Error('probeTexture: stencil8 requires aspect "stencil-only"');
        }
        return { aspect: "stencil-only", bytesPerTexel: 1 };
    }
    if (format === "depth24plus") {
        throw new Error("probeTexture: depth24plus has no portable buffer-copy representation");
    }
    if (format === "depth24plus-stencil8" || format === "depth32float-stencil8") {
        if (!requested || requested === "all") {
            throw new Error(
                `probeTexture: ${format} requires an explicit depth-only or stencil-only aspect`,
            );
        }
        if (requested === "stencil-only") return { aspect: requested, bytesPerTexel: 1 };
        if (format === "depth32float-stencil8") {
            return { aspect: requested, bytesPerTexel: 4 };
        }
        throw new Error(
            "probeTexture: depth24plus-stencil8 depth has no portable buffer-copy representation",
        );
    }
    throw new Error(
        `probeTexture: ${format} is compressed; this probe accepts uncompressed copyable textures only`,
    );
}

async function mappedCopy(
    staging: GPUBuffer,
    device: GPUDevice,
    size: number,
): Promise<ArrayBuffer> {
    await device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ, 0, size);
    return staging.getMappedRange(0, size).slice(0);
}

/**
 * capture one raw buffer range after an optional encoded trigger. The call owns one encoder, submission,
 * fence, and staging buffer; the returned bytes belong only to this result and never enter frame wiring.
 *
 * @example const result = await probeBuffer(device, counters, { encode: runPass });
 */
export async function probeBuffer(
    device: GPUDevice,
    source: GPUBuffer,
    options: BufferProbeOptions = {},
): Promise<BufferProbe> {
    if ((source.usage & GPUBufferUsage.COPY_SRC) === 0) {
        throw new Error("probeBuffer: source is missing GPUBufferUsage.COPY_SRC");
    }
    if (source.mapState !== "unmapped") {
        throw new Error(`probeBuffer: source must be unmapped, got ${source.mapState}`);
    }
    const offset = options.offset ?? 0;
    const size = options.size ?? source.size - offset;
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(size) || offset < 0 || size <= 0) {
        throw new RangeError("probeBuffer: offset and size must describe a non-empty byte range");
    }
    if (offset + size > source.size) throw new RangeError("probeBuffer: range exceeds source size");

    const start = alignDown(offset, COPY_ALIGNMENT);
    const end = alignUp(offset + size, COPY_ALIGNMENT);
    if (end > source.size) {
        throw new RangeError(
            "probeBuffer: trailing bytes have no aligned copy range in the source",
        );
    }
    const copySize = end - start;
    if (copySize > device.limits.maxBufferSize) {
        throw new RangeError("probeBuffer: staging copy exceeds device.limits.maxBufferSize");
    }
    const staging = device.createBuffer({
        label: `${options.label ?? source.label ?? "buffer"}-probe-staging`,
        size: copySize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        const encoder = device.createCommandEncoder({ label: options.label ?? "buffer-probe" });
        options.encode?.(encoder);
        encoder.copyBufferToBuffer(source, start, staging, 0, copySize);
        device.queue.submit([encoder.finish()]);
        const copied = await mappedCopy(staging, device, copySize);
        return Object.freeze({
            bytes: copied.slice(offset - start, offset - start + size),
            offset,
            size,
        });
    } finally {
        if (staging.mapState === "mapped") staging.unmap();
        staging.destroy();
    }
}

/**
 * capture one single-sample, uncompressed color or copyable depth/stencil texture region after an
 * optional encoded trigger. WebGPU row padding is stripped from the owned result.
 *
 * @example const depth = await probeTexture(device, target, { aspect: "depth-only", encode: draw });
 */
export async function probeTexture(
    device: GPUDevice,
    source: GPUTexture,
    options: TextureProbeOptions = {},
): Promise<TextureProbe> {
    if ((source.usage & GPUTextureUsage.COPY_SRC) === 0) {
        throw new Error("probeTexture: source is missing GPUTextureUsage.COPY_SRC");
    }
    if (source.sampleCount !== 1)
        throw new Error("probeTexture: multisampled textures cannot be copied");
    const mipLevel = options.mipLevel ?? 0;
    if (!Number.isInteger(mipLevel) || mipLevel < 0 || mipLevel >= source.mipLevelCount) {
        throw new RangeError("probeTexture: mipLevel is outside the source texture");
    }
    const origin = coordinate(options.origin);
    if (
        !Number.isSafeInteger(origin.x) ||
        !Number.isSafeInteger(origin.y) ||
        !Number.isSafeInteger(origin.z) ||
        origin.x < 0 ||
        origin.y < 0 ||
        origin.z < 0
    ) {
        throw new RangeError("probeTexture: origin must contain non-negative integer coordinates");
    }
    const mipWidth = Math.max(1, source.width >> mipLevel);
    const mipHeight = source.dimension === "1d" ? 1 : Math.max(1, source.height >> mipLevel);
    const mipDepth =
        source.dimension === "3d"
            ? Math.max(1, source.depthOrArrayLayers >> mipLevel)
            : source.depthOrArrayLayers;
    const size = options.size
        ? extent(options.size)
        : {
              width: mipWidth - origin.x,
              height: mipHeight - origin.y,
              depthOrArrayLayers: mipDepth - origin.z,
          };
    if (
        !Number.isSafeInteger(size.width) ||
        !Number.isSafeInteger(size.height) ||
        !Number.isSafeInteger(size.depthOrArrayLayers) ||
        size.width <= 0 ||
        size.height <= 0 ||
        size.depthOrArrayLayers <= 0 ||
        origin.x + size.width > mipWidth ||
        origin.y + size.height > mipHeight ||
        origin.z + size.depthOrArrayLayers > mipDepth
    ) {
        throw new RangeError("probeTexture: region is outside the source texture");
    }

    const { aspect, bytesPerTexel } = copyAspect(source.format, options.aspect);
    const bytesPerRow = size.width * bytesPerTexel;
    const paddedBytesPerRow = alignUp(bytesPerRow, ROW_ALIGNMENT);
    const stagingSize = paddedBytesPerRow * size.height * size.depthOrArrayLayers;
    if (!Number.isSafeInteger(stagingSize) || stagingSize > device.limits.maxBufferSize) {
        throw new RangeError("probeTexture: staging copy exceeds device.limits.maxBufferSize");
    }
    const staging = device.createBuffer({
        label: `${options.label ?? source.label ?? "texture"}-probe-staging`,
        size: stagingSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        const encoder = device.createCommandEncoder({ label: options.label ?? "texture-probe" });
        options.encode?.(encoder);
        encoder.copyTextureToBuffer(
            { texture: source, mipLevel, origin, aspect },
            { buffer: staging, bytesPerRow: paddedBytesPerRow, rowsPerImage: size.height },
            size,
        );
        device.queue.submit([encoder.finish()]);
        const padded = new Uint8Array(await mappedCopy(staging, device, stagingSize));
        const bytes = new Uint8Array(bytesPerRow * size.height * size.depthOrArrayLayers);
        const rows = size.height * size.depthOrArrayLayers;
        for (let row = 0; row < rows; row++) {
            bytes.set(
                padded.subarray(row * paddedBytesPerRow, row * paddedBytesPerRow + bytesPerRow),
                row * bytesPerRow,
            );
        }
        return Object.freeze({
            bytes: bytes.buffer,
            format: source.format,
            aspect,
            width: size.width,
            height: size.height,
            depthOrArrayLayers: size.depthOrArrayLayers,
            bytesPerRow,
        });
    } finally {
        if (staging.mapState === "mapped") staging.unmap();
        staging.destroy();
    }
}
