import { test, expect, describe, beforeEach, afterEach } from "bun:test";
import { State, capacity, buf, type Buf } from "../src";
import { createFieldProxy } from "../src/engine/ecs/component";
import {
    grow,
    write,
    MAX_CAPACITY,
    CHUNK_SIZE,
    CHUNK_SHIFT,
    CHUNK_MASK,
} from "../src/engine/ecs/capacity";

function growPast(state: State) {
    const initial = capacity();
    while (capacity() === initial) state.addEntity();
}

function bufRead(ref: Buf, eid: number): number {
    return ref.chunks[eid >>> CHUNK_SHIFT][(eid & CHUNK_MASK) * ref.stride];
}

function bufWrite(ref: Buf, eid: number, value: number): void {
    ref.chunks[eid >>> CHUNK_SHIFT][(eid & CHUNK_MASK) * ref.stride] = value;
}

describe("capacity", () => {
    let state: State;

    beforeEach(() => {
        state = new State();
    });

    afterEach(() => {
        state.dispose();
    });

    describe("buf", () => {
        test("starts with one chunk sized to chunkSize * stride", () => {
            const ref = buf(Float32Array, 3, 0);
            expect(ref.chunks.length).toBe(1);
            expect(ref.chunks[0].length).toBe(CHUNK_SIZE * 3);
        });

        test("fills with the given value", () => {
            const ref = buf(Uint32Array, 1, 0xffffffff);
            expect(ref.chunks[0][0]).toBe(0xffffffff);
            expect(ref.chunks[0][CHUNK_SIZE - 1]).toBe(0xffffffff);
        });

        test("fill=0 leaves zeros", () => {
            const ref = buf(Float32Array, 1, 0);
            const chunk = ref.chunks[0];
            for (let i = 0; i < chunk.length; i++) {
                expect(chunk[i]).toBe(0);
            }
        });

        test("stride is recorded", () => {
            const f4 = buf(Float32Array, 4, 0);
            const u1 = buf(Uint32Array, 1, 0);
            expect(f4.stride).toBe(4);
            expect(u1.stride).toBe(1);
        });
    });

    describe("growth is append-only", () => {
        test("existing chunks identity is preserved across grow", () => {
            const ref = buf(Float32Array, 1, 0);
            const chunk0 = ref.chunks[0];
            chunk0[0] = 42;
            chunk0[CHUNK_SIZE - 1] = 99;

            growPast(state);
            expect(ref.chunks[0]).toBe(chunk0);
            expect(ref.chunks[0][0]).toBe(42);
            expect(ref.chunks[0][CHUNK_SIZE - 1]).toBe(99);
        });

        test("grow appends new chunks rather than replacing", () => {
            const ref = buf(Float32Array, 1, 0);
            const before = ref.chunks.length;
            grow(capacity() + 1);
            expect(ref.chunks.length).toBe(before + 1);
            expect(ref.chunks[0]).toBe(ref.chunks[0]);
        });

        test("each grow doubles chunk count", () => {
            const ref = buf(Float32Array, 1, 0);
            grow(capacity() + 1);
            const c1 = ref.chunks.length;
            grow(capacity() + 1);
            expect(ref.chunks.length).toBe(c1 * 2);
        });

        test("fill value extends into newly appended chunks", () => {
            const ref = buf(Uint32Array, 1, 7);
            ref.chunks[0][0] = 100;
            const oldCap = capacity();

            growPast(state);
            expect(ref.chunks[0][0]).toBe(100);
            const tailChunk = ref.chunks[ref.chunks.length - 1];
            expect(tailChunk[0]).toBe(7);
            expect(tailChunk[tailChunk.length - 1]).toBe(7);
            expect(capacity()).toBeGreaterThan(oldCap);
        });

        test("stride > 1 chunks size correctly", () => {
            const ref = buf(Float32Array, 4, 0);
            ref.chunks[0][0] = 1;
            ref.chunks[0][1] = 2;
            ref.chunks[0][2] = 3;
            ref.chunks[0][3] = 4;

            growPast(state);
            for (const chunk of ref.chunks) {
                expect(chunk.length).toBe(CHUNK_SIZE * 4);
            }
            expect(ref.chunks[0][0]).toBe(1);
            expect(ref.chunks[0][1]).toBe(2);
            expect(ref.chunks[0][2]).toBe(3);
            expect(ref.chunks[0][3]).toBe(4);
        });
    });

    describe("multiple buffers grow together", () => {
        test("all managed buffers receive matching chunk count after growth", () => {
            const f32 = buf(Float32Array, 1, 0);
            const u32 = buf(Uint32Array, 2, 0);
            const u8 = buf(Uint8Array, 1, 0);

            bufWrite(f32, 500, 314);
            bufWrite(u32, 600, 42);
            bufWrite(u8, 700, 255);

            growPast(state);

            expect(f32.chunks.length).toBe(u32.chunks.length);
            expect(u32.chunks.length).toBe(u8.chunks.length);

            expect(bufRead(f32, 500)).toBe(314);
            expect(bufRead(u32, 600)).toBe(42);
            expect(bufRead(u8, 700)).toBe(255);
        });
    });

    describe("ref stability", () => {
        test("ref identity is stable across growth", () => {
            const ref = buf(Float32Array, 1, 0);
            const refBefore = ref;

            growPast(state);
            expect(ref).toBe(refBefore);
        });

        test("first chunk identity stays stable as new chunks append", () => {
            const ref = buf(Float32Array, 1, 0);
            const original = ref.chunks[0];

            for (let i = 0; i < 5; i++) growPast(state);
            expect(ref.chunks[0]).toBe(original);
        });

        test("pointer stability stress: many bufs, repeated growth", () => {
            const refs = [
                buf(Float32Array, 4, 0),
                buf(Uint32Array, 1, 0),
                buf(Uint8Array, 1, 0),
                buf(Float32Array, 16, 0),
                buf(Uint16Array, 2, 0),
            ];
            const initialChunks = refs.map((r) => r.chunks[0]);

            for (let i = 0; i < CHUNK_SIZE * 4; i++) state.addEntity();

            for (let i = 0; i < refs.length; i++) {
                expect(refs[i].chunks[0]).toBe(initialChunks[i]);
                expect(refs[i].chunks.length).toBeGreaterThan(1);
            }
        });
    });

    describe("proxy survives growth and crosses chunk boundaries", () => {
        test("createFieldProxy reads/writes through ref across growth", () => {
            const ref = buf(Float32Array, 4, 0);
            const proxy = createFieldProxy(ref, 4, 0);

            proxy.set(5, 42);
            expect(proxy.get(5)).toBe(42);
            expect(proxy[5]).toBe(42);

            growPast(state);

            expect(proxy.get(5)).toBe(42);
            expect(proxy[5]).toBe(42);

            const newSlot = capacity() - 1;
            proxy.set(newSlot, 99);
            expect(proxy.get(newSlot)).toBe(99);
        });

        test("proxy addresses entities in different chunks", () => {
            const ref = buf(Float32Array, 4, 0);
            const proxy = createFieldProxy(ref, 4, 0);

            grow(CHUNK_SIZE * 2);
            proxy.set(CHUNK_SIZE - 1, 11);
            proxy.set(CHUNK_SIZE, 22);
            proxy.set(CHUNK_SIZE * 2 - 1, 33);

            expect(proxy.get(CHUNK_SIZE - 1)).toBe(11);
            expect(proxy.get(CHUNK_SIZE)).toBe(22);
            expect(proxy.get(CHUNK_SIZE * 2 - 1)).toBe(33);

            expect(ref.chunks[0][(CHUNK_SIZE - 1) * 4]).toBe(11);
            expect(ref.chunks[1][0]).toBe(22);
            expect(ref.chunks[1][(CHUNK_SIZE - 1) * 4]).toBe(33);
        });

        test("proxy per-field offsets work across chunks", () => {
            const ref = buf(Float32Array, 4, 0);
            const x = createFieldProxy(ref, 4, 0);
            const y = createFieldProxy(ref, 4, 1);

            grow(CHUNK_SIZE * 2);
            x.set(CHUNK_SIZE + 7, 5);
            y.set(CHUNK_SIZE + 7, 11);

            expect(x.get(CHUNK_SIZE + 7)).toBe(5);
            expect(y.get(CHUNK_SIZE + 7)).toBe(11);
            expect(ref.chunks[1][7 * 4]).toBe(5);
            expect(ref.chunks[1][7 * 4 + 1]).toBe(11);
        });
    });

    describe("dispose resets capacity", () => {
        test("new state after dispose starts fresh", () => {
            buf(Float32Array, 1, 0);
            growPast(state);
            const grownCap = capacity();

            state.dispose();

            state = new State();
            const ref2 = buf(Float32Array, 1, 0);
            expect(capacity()).toBeLessThan(grownCap);
            expect(ref2.chunks.length).toBe(1);
        });
    });

    describe("growth invariants", () => {
        test("capacity is always a multiple of CHUNK_SIZE", () => {
            expect(capacity() % CHUNK_SIZE).toBe(0);
            growPast(state);
            expect(capacity() % CHUNK_SIZE).toBe(0);
            growPast(state);
            expect(capacity() % CHUNK_SIZE).toBe(0);
        });

        test("CHUNK_SIZE is a power of two", () => {
            expect(CHUNK_SIZE & (CHUNK_SIZE - 1)).toBe(0);
        });

        test("grow is idempotent when n <= capacity", () => {
            const before = capacity();
            grow(before);
            expect(capacity()).toBe(before);
            grow(before - 1);
            expect(capacity()).toBe(before);
            grow(1);
            expect(capacity()).toBe(before);
        });

        test("multiple grows preserve all data without copying chunks", () => {
            const ref = buf(Float32Array, 1, 0);
            const chunk0 = ref.chunks[0];
            chunk0[0] = 1;
            chunk0[100] = 2;
            const cap0 = capacity();

            grow(cap0 + 1);
            const cap1 = capacity();
            expect(cap1).toBeGreaterThan(cap0);
            const chunk1 = ref.chunks[ref.chunks.length - 1];
            chunk1[50] = 3;

            grow(cap1 + 1);
            const cap2 = capacity();
            expect(cap2).toBeGreaterThan(cap1);
            const chunk2 = ref.chunks[ref.chunks.length - 1];
            chunk2[50] = 4;

            grow(cap2 + 1);
            expect(capacity()).toBeGreaterThan(cap2);

            expect(ref.chunks[0]).toBe(chunk0);
            expect(ref.chunks[0][0]).toBe(1);
            expect(ref.chunks[0][100]).toBe(2);
        });

        test("grow jumps multiple doublings for large n", () => {
            const before = capacity();
            grow(before * 8 + 1);
            expect(capacity()).toBeGreaterThanOrEqual(before * 8 + 1);
        });
    });

    describe("MAX_CAPACITY", () => {
        test("exported and is a positive multiple of CHUNK_SIZE", () => {
            expect(MAX_CAPACITY).toBeGreaterThan(0);
            expect(MAX_CAPACITY % CHUNK_SIZE).toBe(0);
        });

        test("grow throws when capacity would exceed MAX_CAPACITY", () => {
            expect(() => grow(MAX_CAPACITY + 1)).toThrow(/capacity exceeded/i);
        });

        test("grow to MAX_CAPACITY succeeds and preserves first chunk", () => {
            const ref = buf(Float32Array, 1, 0);
            ref.chunks[0][0] = 42;
            grow(MAX_CAPACITY);
            expect(capacity()).toBe(MAX_CAPACITY);
            expect(ref.chunks[0][0]).toBe(42);
            expect(ref.chunks.length).toBe(MAX_CAPACITY / CHUNK_SIZE);
        });

        test("grow just past MAX_CAPACITY throws", () => {
            grow(MAX_CAPACITY);
            expect(capacity()).toBe(MAX_CAPACITY);
            expect(() => grow(MAX_CAPACITY + 1)).toThrow(/capacity exceeded/i);
            expect(capacity()).toBe(MAX_CAPACITY);
        });
    });

    describe("write", () => {
        interface FakeWrite {
            offset: number;
            src: ArrayBufferView;
            srcStart: number;
            srcLen: number;
        }

        function fakeQueue(): { writes: FakeWrite[]; queue: GPUQueue } {
            const writes: FakeWrite[] = [];
            const queue = {
                writeBuffer(
                    _target: GPUBuffer,
                    offset: number,
                    src: ArrayBufferView,
                    srcStart: number,
                    srcLen: number,
                ) {
                    writes.push({ offset, src, srcStart, srcLen });
                },
            } as unknown as GPUQueue;
            return { writes, queue };
        }

        const target = {} as GPUBuffer;

        test("count=0 issues no writes", () => {
            const ref = buf(Float32Array, 4, 0);
            const { writes, queue } = fakeQueue();
            write(queue, target, 0, ref, 0);
            expect(writes.length).toBe(0);
        });

        test("count <= chunk size issues one write at the given byte offset", () => {
            const ref = buf(Float32Array, 4, 0);
            ref.chunks[0][0] = 1;
            const { writes, queue } = fakeQueue();
            write(queue, target, 64, ref, 10);
            expect(writes.length).toBe(1);
            expect(writes[0].offset).toBe(64);
            expect(writes[0].src).toBe(ref.chunks[0]);
            expect(writes[0].srcLen).toBe(10 * 4);
        });

        test("count spanning chunks issues one write per chunk", () => {
            const ref = buf(Float32Array, 4, 0);
            grow(CHUNK_SIZE * 2);
            const { writes, queue } = fakeQueue();
            write(queue, target, 0, ref, CHUNK_SIZE + 5);
            expect(writes.length).toBe(2);
            expect(writes[0].srcLen).toBe(CHUNK_SIZE * 4);
            expect(writes[0].offset).toBe(0);
            expect(writes[1].src).toBe(ref.chunks[1]);
            expect(writes[1].offset).toBe(CHUNK_SIZE * 4 * 4);
            expect(writes[1].srcLen).toBe(5 * 4);
        });

        test("Uint8 partial-chunk writes pad to 4-byte alignment", () => {
            const ref = buf(Uint8Array, 1, 0);
            const { writes, queue } = fakeQueue();
            write(queue, target, 0, ref, 5);
            expect(writes.length).toBe(1);
            expect(writes[0].srcLen).toBe(8);
        });

        test("count exceeding capacity stops at chunk boundary", () => {
            const ref = buf(Float32Array, 1, 0);
            const initialChunks = ref.chunks.length;
            const { writes, queue } = fakeQueue();
            write(queue, target, 0, ref, capacity() * 2);
            expect(writes.length).toBe(initialChunks);
        });
    });
});
