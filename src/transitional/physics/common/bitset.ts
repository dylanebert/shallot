// A bit set over a growable Uint32Array, ported from Box3D's bitset.c.
//
// Deviation from the C, deliberate: Box3D packs bits in uint64_t blocks; this uses uint32_t
// blocks, because JS bitwise ops are defined on 32-bit integers and BigInt would make the
// per-step island bookkeeping needlessly slow. This is a representation-internal choice with
// zero observable effect — set membership, popcount, union, and ascending-bit-index
// enumeration are identical at either width, and no bit set feeds the world-state hash.

import { popCount32 } from "./bits";

export type BitSet = { bits: Uint32Array; blockCapacity: number; blockCount: number };

const BITS_PER_BLOCK = 32;

function blocksFor(bitCount: number): number {
    return (bitCount + (BITS_PER_BLOCK - 1)) >>> 5;
}

export function createBitSet(bitCapacity: number): BitSet {
    const blockCapacity = blocksFor(bitCapacity);
    return { bits: new Uint32Array(blockCapacity), blockCapacity, blockCount: 0 };
}

export function setBitCountAndClear(bitSet: BitSet, bitCount: number): void {
    const blockCount = blocksFor(bitCount);
    if (bitSet.blockCapacity < blockCount) {
        const newBitCapacity = bitCount + (bitCount >>> 1);
        bitSet.blockCapacity = blocksFor(newBitCapacity);
        bitSet.bits = new Uint32Array(bitSet.blockCapacity);
    }
    bitSet.blockCount = blockCount;
    bitSet.bits.fill(0, 0, blockCount);
}

export function growBitSet(bitSet: BitSet, blockCount: number): void {
    if (blockCount > bitSet.blockCapacity) {
        const oldCapacity = bitSet.blockCapacity;
        bitSet.blockCapacity = blockCount + (blockCount >>> 1);
        const newBits = new Uint32Array(bitSet.blockCapacity);
        newBits.set(bitSet.bits.subarray(0, oldCapacity));
        bitSet.bits = newBits;
    }
    bitSet.blockCount = blockCount;
}

export function setBit(bitSet: BitSet, bitIndex: number): void {
    const blockIndex = bitIndex >>> 5;
    bitSet.bits[blockIndex] |= 1 << (bitIndex & 31);
}

export function setBitGrow(bitSet: BitSet, bitIndex: number): void {
    const blockIndex = bitIndex >>> 5;
    if (blockIndex >= bitSet.blockCount) {
        growBitSet(bitSet, blockIndex + 1);
    }
    bitSet.bits[blockIndex] |= 1 << (bitIndex & 31);
}

export function clearBit(bitSet: BitSet, bitIndex: number): void {
    const blockIndex = bitIndex >>> 5;
    if (blockIndex >= bitSet.blockCount) {
        return;
    }
    bitSet.bits[blockIndex] &= ~(1 << (bitIndex & 31));
}

export function getBit(bitSet: BitSet, bitIndex: number): boolean {
    const blockIndex = bitIndex >>> 5;
    if (blockIndex >= bitSet.blockCount) {
        return false;
    }
    return (bitSet.bits[blockIndex] & (1 << (bitIndex & 31))) !== 0;
}

/** Union setB into setA in place. Both must have the same block count. */
export function inPlaceUnion(setA: BitSet, setB: BitSet): void {
    const blockCount = setA.blockCount;
    for (let i = 0; i < blockCount; ++i) {
        setA.bits[i] |= setB.bits[i];
    }
}

export function countSetBits(bitSet: BitSet): number {
    let count = 0;
    const blockCount = bitSet.blockCount;
    for (let i = 0; i < blockCount; ++i) {
        count += popCount32(bitSet.bits[i]);
    }
    return count;
}
