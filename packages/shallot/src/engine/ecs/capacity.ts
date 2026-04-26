type TypedArrayCtor =
    | typeof Float32Array
    | typeof Uint32Array
    | typeof Uint16Array
    | typeof Uint8Array
    | typeof Int32Array;

type TypedArray = Float32Array | Uint32Array | Uint16Array | Uint8Array | Int32Array;

/** typed-array kind tag */
export type ArrayKind = "f32" | "u32" | "u16" | "u8" | "i32";

const kinds = new Map<TypedArrayCtor, ArrayKind>([
    [Float32Array, "f32"],
    [Uint32Array, "u32"],
    [Uint16Array, "u16"],
    [Uint8Array, "u8"],
    [Int32Array, "i32"],
]);

/** entities per chunk; chunks are append-only stable allocations */
export const CHUNK_SIZE = 4096;
/** chunk index = eid >>> CHUNK_SHIFT */
export const CHUNK_SHIFT = 12;
/** local entity index = eid & CHUNK_MASK */
export const CHUNK_MASK = CHUNK_SIZE - 1;

/**
 * stable, append-only chunked typed-array storage.
 * `chunks[chunk = eid >> CHUNK_SHIFT][(eid & CHUNK_MASK) * stride + field]` resolves a slot.
 * existing chunks are never replaced — pointers into them are valid for the buf's lifetime.
 */
export interface Buf<T extends TypedArray = TypedArray> {
    chunks: T[];
    stride: number;
    id: number;
    kind: ArrayKind;
}

interface BufEntry {
    ref: Buf;
    ctor: TypedArrayCtor;
    stride: number;
    fill: number;
}

const managed: BufEntry[] = [];
let _chunkCount = 1;
let _nextBufId = 0;

/** hard ceiling on entities; grow throws past this */
export const MAX_CAPACITY = 1 << 20;
const MAX_CHUNKS = MAX_CAPACITY / CHUNK_SIZE;

/** current entity buffer capacity */
export function capacity(): number {
    return _chunkCount * CHUNK_SIZE;
}

function allocChunk<C extends TypedArrayCtor>(
    ctor: C,
    stride: number,
    fill: number,
): InstanceType<C> {
    const arr = new ctor(CHUNK_SIZE * stride) as InstanceType<C>;
    if (fill !== 0) (arr as TypedArray).fill(fill as never);
    return arr;
}

/** allocate a managed chunked typed array that grows with capacity */
export function buf<C extends TypedArrayCtor>(
    ctor: C,
    stride: number,
    fill: number,
): Buf<InstanceType<C>> {
    const kind = kinds.get(ctor);
    if (!kind) throw new Error("unknown typed-array kind");
    const chunks: InstanceType<C>[] = [];
    for (let i = 0; i < _chunkCount; i++) chunks.push(allocChunk(ctor, stride, fill));
    const ref: Buf<InstanceType<C>> = { chunks, stride, id: _nextBufId++, kind };
    managed.push({ ref: ref as Buf, ctor, stride, fill });
    return ref;
}

export function grow(n: number): void {
    if (n <= capacity()) return;
    let next = _chunkCount;
    while (next * CHUNK_SIZE < n) next *= 2;
    if (next > MAX_CHUNKS) {
        throw new Error(`Entity capacity exceeded (max ${MAX_CAPACITY})`);
    }
    while (_chunkCount < next) {
        for (const entry of managed) {
            entry.ref.chunks.push(allocChunk(entry.ctor, entry.stride, entry.fill));
        }
        _chunkCount++;
    }
}

export function reset(): void {
    managed.length = 0;
    _chunkCount = 1;
    _nextBufId = 0;
}

/** zero (or fill) every chunk. Useful for tests that share module-scoped bufs. */
export function clearBuf<T extends TypedArray>(ref: Buf<T>, value = 0): void {
    for (const chunk of ref.chunks) (chunk as TypedArray).fill(value as never);
}

/**
 * upload `count` entities of a chunked buf to `target`, chunk by chunk.
 * Destination must be sized for full chunk granularity.
 */
export function write(
    queue: GPUQueue,
    target: GPUBuffer,
    targetByteOffset: number,
    ref: Buf,
    count: number,
): void {
    if (count <= 0) return;
    const stride = ref.stride;
    const chunks = ref.chunks;
    const elemSize = (chunks[0] as TypedArray).BYTES_PER_ELEMENT;
    const bytesPerEntity = stride * elemSize;
    let written = 0;
    for (let i = 0; i < chunks.length && written < count; i++) {
        const remaining = count - written;
        const chunkEntities = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
        let elementCount = chunkEntities * stride;
        const tailBytes = elementCount * elemSize;
        if (tailBytes & 3) elementCount += (4 - (tailBytes & 3)) / elemSize;
        queue.writeBuffer(
            target,
            targetByteOffset + written * bytesPerEntity,
            chunks[i] as TypedArray as Uint8Array<ArrayBuffer>,
            0,
            elementCount,
        );
        written += chunkEntities;
    }
}
